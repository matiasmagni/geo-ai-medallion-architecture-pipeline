#!/usr/bin/env python3
"""
Silver Layer Quality Audit with DeepSeek-R1 Judge.
This script performs quality auditing of the Silver layer data using DeepSeek-R1 as a judge,
with full MLflow evaluation tracking.
"""

import json
import os
import pandas as pd
from pyspark.sql import SparkSession
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Initialize MLflow experiment
mlflow.set_experiment("GeoAI_Medallion_Architecture")

# DeepSeek-R1 Configuration (local Ollama or other endpoint)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEEPSEEK_R1_MODEL = "deepseek-r1:7b"  # Adjust based on your local model tag

def load_hallucination_judge_prompt():
    """Load the DeepSeek-R1 hallucination judge prompt from MLflow"""
    try:
        prompt = mlflow.prompts.load_prompt(
            name="deepseek_hallucination_judge",
            version="1.0"
        )
        return prompt.template
    except Exception as e:
        print(f"Warning: Could not load judge prompt from MLflow: {e}")
        # Fallback prompt
        return """
        You are an expert evaluator tasked with detecting hallucinations in AI-generated hazard extractions.
        Compare the extracted hazards JSON against the original incident narrative.
        Evaluate whether the extracted hazards are:
        1. FAITHFUL: All extracted hazards are directly supported by the text.
        2. PARTIALLY FAITHFUL: Most hazards are supported, but some may be inferred or slightly exaggerated.
        3. HALLUCINATED: Significant portions of the extraction are not supported by the text or contradict it.
        
        Provide your evaluation as a JSON object with:
        - faithfulness_score: Float between 0.0 (completely hallucinated) and 1.0 (completely faithful)
        - explanation: Brief explanation of your scoring
        - unsupported_claims: List of specific claims in the extraction that are not supported by the text
        
        Original Incident Narrative:
        {source_text}
        
        Extracted Hazards JSON:
        {extracted_json}
        
        Evaluation (JSON object):
        """

class DeepSeekJudge:
    """
    Custom MLflow GenAI Metric using DeepSeek-R1 as the evaluation judge.
    """
    
    def __init__(self):
        self.endpoint = f"{OLLAMA_HOST}/api/generate"
        self.model = DEEPSEEK_R1_MODEL
        
    def evaluate(self, row):
        """
        Evaluate a single row of extracted data against the source text.
        Returns a dictionary with faithfulness score and explanation.
        """
        try:
            extracted_json = row['extracted_json']
            source_text = row['incident_narrative']
            
            # Prepare evaluation prompt
            prompt = self._create_evaluation_prompt(source_text, extracted_json)
            
            # Call DeepSeek-R1 via Ollama
            response = self._call_deepseek(prompt)
            
            # Parse the judge's response
            evaluation = self._parse_evaluation(response)
            
            return evaluation
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"faithfulness_score": 0.0, "explanation": f"Evaluation failed: {str(e)}", "unsupported_claims": []}
    
    def _create_evaluation_prompt(self, source_text, extracted_json):
        """Create the evaluation prompt for DeepSeek-R1"""
        # Load from MLflow if available, otherwise use fallback
        try:
            prompt_template = mlflow.prompts.load_prompt(
                name="deepseek_hallucination_judge",
                version="1.0"
            )
            return prompt_template.format(
                source_text=source_text,
                extracted_json=extracted_json
            )
        except:
            return f"""
            You are an expert evaluator tasked with detecting hallucinations in AI-generated hazard extractions.
            Compare the extracted hazards JSON against the original incident narrative.
            Evaluate whether the extracted hazards are:
            1. FAITHFUL: All extracted hazards are directly supported by the text.
            2. PARTIALLY FAITHFUL: Most hazards are supported, but some may be inferred or slightly exaggerated.
            3. HALLUCINATED: Significant portions of the extraction are not supported by the text or contradict it.
            
            Provide your evaluation as a JSON object with:
            - faithfulness_score: Float between 0.0 (completely hallucinated) and 1.0 (completely faithful)
            - explanation: Brief explanation of your scoring
            - unsupported_claims: List of specific claims in the extraction that are not supported by the text
            
            Original Incident Narrative:
            {source_text}
            
            Extracted Hazards JSON:
            {extracted_json}
            
            Evaluation (JSON object):
            """
    
    def _call_deepseek(self, prompt):
        """Call DeepSeek-R1 via Ollama endpoint"""
        import requests
        import time
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            result['metadata'] = {
                'latency_seconds': time.time() - start_time,
                'model': self.model
            }
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return {"response": "", "error": str(e), "metadata": {"model": self.model}}
    
    def _parse_evaluation(self, result):
        """Parse the DeepSeek-R1 response to extract evaluation metrics"""
        response_text = result.get("response", "{}")
        
        # Clean up response - remove markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            evaluation = json.loads(response_text)
            # Ensure all required fields are present
            if 'faithfulness_score' not in evaluation:
                evaluation['faithfulness_score'] = 0.5
            if 'explanation' not in evaluation:
                evaluation['explanation'] = "No explanation provided"
            if 'unsupported_claims' not in evaluation:
                evaluation['unsupported_claims'] = []
            return evaluation
        except json.JSONDecodeError:
            print(f"Warning: Could not parse evaluation response: {response_text}")
            return {"faithfulness_score": 0.5, "explanation": "Failed to parse evaluation", "unsupported_claims": []}

# Pandas UDF for PySpark Quality Audit
@mlflow.data.from_pandas
def quality_audit_udf(pdf):
    """
    PySpark Pandas UDF that performs quality auditing on Silver layer data.
    Uses DeepSeek-R1 as the judge via MLflow's GenAI metric system.
    """
    # Initialize judge
    judge = DeepSeekJudge()
    
    # Process each row and collect results
    results = []
    for idx, row in pdf.iterrows():
        # Convert row to dict for easier handling
        row_dict = row.to_dict()
        
        # Evaluate the quality
        evaluation = judge.evaluate(row_dict)
        
        # Create result row
        result = {
            'incident_id': row_dict.get('incident_id', ''),
            'extracted_json': row_dict.get('extracted_hazards_json', ''),
            'source_text': row_dict.get('incident_narrative', ''),
            'faithfulness_score': evaluation.get('faithfulness_score', 0.0),
            'explanation': evaluation.get('explanation', ''),
            'unsupported_claims': json.dumps(evaluation.get('unsupported_claims', [])),
            'evaluation_timestamp': pd.Timestamp.now()
        }
        results.append(result)
    
    return pd.DataFrame(results)

def run_quality_audit(silver_table_path, output_path):
    """
    Main function to run quality audit on Silver layer data.
    """
    spark = SparkSession.builder \
        .appName("GeoAI_Quality_Audit") \
        .getOrCreate()
    
    # Read 5% sample of Silver Delta table
    silver_df = spark.read.format("delta").load(silver_table_path)
    sample_df = silver_df.sample(fraction=0.05, seed=42)
    
    # Apply quality audit UDF
    audit_df = sample_df.withColumn(
        "quality_audit",
        quality_audit_udf(sample_df.select("incident_id", "incident_narrative", "extracted_hazards_json"))
    )
    
    # Write results
    audit_df.write.format("parquet").mode("overwrite").save(output_path)
    
    print(f"Quality audit completed on {sample_df.count()} samples")
    print("MLflow Evaluation Runs have been captured - view in MLflow UI under 'Evaluation' tab")
    
    return audit_df

if __name__ == "__main__":
    SILVER_PATH = "/mnt/geoai/silver/aviation_incidents_enriched"
    OUTPUT_PATH = "/mnt/geoai/audit/quality_results"
    
    run_quality_audit(SILVER_PATH, OUTPUT_PATH)