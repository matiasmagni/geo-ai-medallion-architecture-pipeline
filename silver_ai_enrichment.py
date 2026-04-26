#!/usr/bin/env python3
"""
Silver Layer AI Enrichment with MLflow Tracing.
This script processes Bronze layer data using Llama 3 via Ollama for hazard extraction,
with full MLflow tracing capabilities.
"""

import json
import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import os

# Initialize MLflow
mlflow.set_experiment("GeoAI_Medallion_Architecture")

# Load the prompt from MLflow Prompt Management
def load_hazard_extraction_prompt():
    """Load the Llama 3 hazard extraction prompt from MLflow"""
    try:
        # Get the latest version of the prompt
        prompt = mlflow.prompts.load_prompt(
            name="llama3_hazard_extraction",
            version="1.0"  # In production, you might want to use "latest" or specific version
        )
        return prompt.template
    except Exception as e:
        print(f"Warning: Could not load prompt from MLflow: {e}")
        # Fallback prompt if MLflow is not available
        return """
        You are an expert in extracting structured hazard information from aviation incident reports.
        Given the following incident narrative, extract all mentioned hazards and return them as a JSON array.
        Each hazard object should have the following fields:
        - hazard_type: Category of hazard (e.g., weather, mechanical, human_factor, procedural)
        - description: Concise description of the hazard
        - severity: Estimated severity level (low, medium, high) based on context
        - location: Location where hazard occurred (if mentioned)
        - timestamp: Time related to hazard (if mentioned)
        
        If no hazards are found, return an empty array.
        
        Incident Narrative:
        {incident_text}
        
        Extracted Hazards (JSON array):
        """

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GeoAI_Silver_Enrichment") \
    .getOrCreate()

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLAMA3_MODEL = "llama3:8b"  # Adjust based on your model tag

def call_ollama_llama3(prompt, model=LLAMA3_MODEL):
    """
    Call local Ollama Llama 3 model with error handling
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent extraction
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return {"response": "", "error": str(e)}

@mlflow.trace
def llama3_hazard_extraction(incident_text):
    """
    Extract hazards from incident text using Llama 3 via Ollama.
    This function is traced by MLflow to capture inputs, outputs, and attributes.
    """
    start_time = time.time()
    
    # Load prompt from MLflow
    prompt_template = load_hazard_extraction_prompt()
    prompt = prompt_template.format(incident_text=incident_text)
    
    # Call Ollama
    result = call_ollama_llama3(prompt)
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Parse the response
    extracted_text = result.get("response", "[]")
    
    # Try to parse as JSON, fallback to empty array if invalid
    try:
        # Clean up the response - LLMs sometimes add extra text
        extracted_text = extracted_text.strip()
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        extracted_text = extracted_text.strip()
        
        hazards = json.loads(extracted_text)
        if not isinstance(hazards, list):
            hazards = []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from Llama 3 response: {extracted_text}")
        hazards = []
    
    # Extract token usage if available from Ollama response
    prompt_tokens = result.get("prompt_eval_count", 0)
    completion_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens
    
    # Return structured result with metadata for tracing
    return {
        "hazards": hazards,
        "metadata": {
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": LLAMA3_MODEL
        }
    }

# Pandas UDF for PySpark integration
@pandas_udf("string", PandasUDFType.SCALAR_ITER)
def extract_hazards_udf(incident_text_iter):
    """
    PySpark UDF that applies Llama 3 hazard extraction to a batch of incident texts.
    Each batch is processed within an MLflow Trace Session.
    """
    # Start an MLflow Trace Session for this batch
    with mlflow.start_trace(name="silver_layer_batch_processing") as trace:
        results = []
        
        for incident_text in incident_text_iter:
            if incident_text is None or incident_text == "":
                # Return empty JSON array for null/empty inputs
                results.append("[]")
                continue
                
            # Extract hazards using our traced function
            result = llama3_hazard_extraction(incident_text)
            
            # Add trace information to the result if needed
            # The trace is automatically captured by the @mlflow.trace decorator
            results.append(json.dumps(result["hazards"]))
        
        return iter(results)

def process_bronze_to_silver(bronze_table_path, silver_table_path):
    """
    Main function to process Bronze layer to Silver layer with MLflow tracing.
    """
    # Read Bronze layer data
    bronze_df = spark.read.format("delta").load(bronze_table_path)
    
    # Apply hazard extraction UDF
    silver_df = bronze_df.withColumn(
        "extracted_hazards_json",
        extract_hazards_udf(bronze_df["incident_narrative"])
    ).withColumn(
        "extraction_timestamp",
        spark.sql("current_timestamp()")
    )
    
    # Write to Silver layer
    silver_df.write.format("delta") \
        .mode("overwrite") \
        .save(silver_table_path)
    
    print(f"Successfully processed {bronze_df.count()} records from Bronze to Silver")
    print("MLflow Traces have been captured - view in MLflow UI under 'Traces' tab")
    
    return silver_df

if __name__ == "__main__":
    # Example usage - adjust paths as needed
    BRONZE_PATH = "/mnt/geoai/bronze/aviation_incidents"
    SILVER_PATH = "/mnt/geoai/silver/aviation_incidents_enriched"
    
    # Process the data
    process_bronze_to_silver(BRONZE_PATH, SILVER_PATH)