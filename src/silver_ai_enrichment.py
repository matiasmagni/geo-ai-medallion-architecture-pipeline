# src/silver_ai_enrichment.py
import mlflow
import ollama # Assuming you use the ollama Python library
import json
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure MLflow tracking is configured (e.g., by setting MLFLOW_TRACKING_URI environment variable)
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if using a remote tracking server

# Placeholder for the LLM model name and prompt configuration
# In a real scenario, these would be loaded from MLflow Prompts or configuration
LLAMA3_MODEL_NAME = "llama3"
HAZARD_EXTRACTION_PROMPT_NAME = "hazard_extraction" # Loaded from setup_mlflow_prompts.py

# --- Helper Function to Load MLflow Prompt ---
def get_mlflow_prompt(prompt_name: str, version: str = "1.0.0") -> Dict[str, Any]:
    """Loads a prompt template and its associated model configuration from MLflow."""
    try:
        prompt_manager = mlflow.llms.prompts.PromptManager()
        prompt_template_obj = prompt_manager.get_prompt_template(name=prompt_name, version=version)
        return {
            "prompt_template": prompt_template_obj.template,
            "model_config": prompt_template_obj.model_config
        }
    except Exception as e:
        logger.error(f"Error loading MLflow prompt '{prompt_name}@{version}': {e}")
        raise

# --- Mock Ollama API Response for Tracing Attributes ---
# In a real scenario, this would come from the actual ollama library response.
def mock_ollama_response(content: str) -> Dict[str, Any]:
    """Simulates an Ollama API response with token usage and latency."""
    # Simulate token counts and response time
    response_time_ms = 500 # Simulate latency
    # Basic token count simulation (very rough)
    num_tokens = len(content.split())
    return {
        "response": content,
        "model": LLAMA3_MODEL_NAME,
        "created_at": "2023-04-26T10:00:00Z",
        "done_reason": "stop",
        "context": [12345, 54321], # Example context values
        "total_duration": response_time_ms, # ms
        "load_duration": response_time_ms // 2, # ms
        "prompt_eval_count": num_tokens,
        "eval_count": num_tokens,
        "eval_duration": response_time_ms,
    }

# --- MLflow Traced Function for LLM Extraction ---
# This function is decorated with @mlflow.trace to automatically log
# its execution as an MLflow Trace.
@mlflow.trace(
    name="llama3_hazard_extraction",
    attributes={ # These attributes will be logged for each trace
        "model_provider": "ollama",
        "model_name": LLAMA3_MODEL_NAME,
        "prompt_name": HAZARD_EXTRACTION_PROMPT_NAME,
    },
    # Attributes can also be functions that compute values dynamically
    # For example: "token_usage": lambda response: response.get("eval_count")
    # We'll extract these dynamically within the function for clarity
)
def extract_hazards_with_ollama(
    bronze_text: str,
    prompt_template: str,
    model_name: str = LLAMA3_MODEL_NAME,
    ollama_client: ollama = ollama # Pass client for potential mocking/injection
) -> Dict[str, Any]:
    """
    Extracts hazards from bronze text using a specific LLM prompt and Ollama.
    This function is traced by MLflow.

    Args:
        bronze_text: The input text from the Bronze layer.
        prompt_template: The MLflow prompt template string.
        model_name: The name of the Ollama model to use.
        ollama_client: The Ollama client instance.

    Returns:
        A dictionary representing the extracted hazards in JSON format.
    """
    logger.info(f"Processing text for hazard extraction...")
    # Inject input text into the prompt template
    formatted_prompt = prompt_template.replace("{{input_text}}", bronze_text)

    # Simulate start time for latency calculation
    start_time = time.time()

    # --- Call LLM ---
    try:
        # Note: In a real application, you'd typically use an MLflow LLM wrapper
        # or directly call the Ollama API. For demonstration, we'll simulate.
        # Using the ollama library directly.
        # response = ollama_client.chat(
        #     model=model_name,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": formatted_prompt,
        #         },
        #     ],
        #     # Add stream=False if needed, though trace might handle non-streaming
        # )
        # simulated_response_content = response['message']['content']

        # --- Simulate Ollama response for demonstration ---
        simulated_response_content = json.dumps({
            "hazards": [
                {"type": "Data Quality", "description": "Inconsistent date formats found in historical records.", "severity": "Medium"},
                {"type": "Infrastructure", "description": "High latency observed in data retrieval from external API.", "severity": "High"}
            ],
            "confidence_score": 0.95
        })
        simulated_response = mock_ollama_response(simulated_response_content)
        # --- End Simulation ---

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Parse the LLM response
        try:
            extracted_data = json.loads(simulated_response["response"])
        except json.JSONDecodeError:
            logger.error("Failed to decode LLM response JSON.")
            extracted_data = {"hazards": [], "confidence_score": 0.0, "error": "JSON Decode Error"}

        # --- Extract Attributes for MLflow Trace ---
        # These will be logged as part of the trace's attributes.
        # The decorator "@mlflow.trace" can often capture these automatically
        # if the underlying LLM integration is aware of them.
        # Manually adding them here for clarity and to show how to combine.
        token_usage = {
            "prompt_tokens": simulated_response.get("prompt_eval_count", 0),
            "completion_tokens": simulated_response.get("eval_count", 0),
            "total_tokens": simulated_response.get("prompt_eval_count", 0) + simulated_response.get("eval_count", 0)
        }
        # MLflow trace decorator can capture latency. We can also add it manually.
        # Ensure the response has a standard format if possible.
        # For direct Ollama API, `response.get("total_duration")` might be available.
        # We'll use our simulated latency here.
        custom_attributes = {
            "latency_ms": simulated_response.get("total_duration", latency_ms), # Use simulated or actual if available
            "token_usage": token_usage,
            "llm_response_raw": simulated_response.get("response") # Optionally log raw response
        }
        mlflow.log_dict(custom_attributes, "llm_attributes.json") # Log custom attributes if not auto-captured

        return extracted_data

    except Exception as e:
        logger.error(f"Error during LLM hazard extraction: {e}")
        # Log error details and return an error structure
        # If an error occurs, the trace will be marked as failed.
        return {
            "hazards": [],
            "confidence_score": 0.0,
            "error": str(e),
            "error_type": type(e).__name__
        }

# --- PySpark UDF Wrapper ---
# This UDF will be applied to each row in the DataFrame.
# The @mlflow.trace decorator works best on Python functions, not directly on Spark UDFs.
# We call the traced function *from* the UDF.

# Load prompt template once
try:
    prompt_config = get_mlflow_prompt(HAZARD_EXTRACTION_PROMPT_NAME)
    HAZARD_EXTRACTION_PROMPT_TEMPLATE = prompt_config["prompt_template"]
    LLM_MODEL_CONFIG = prompt_config["model_config"] # Not directly used here but good to have
    logger.info(f"Loaded prompt: '{HAZARD_EXTRACTION_PROMPT_NAME}'")
except Exception as e:
    logger.error(f"Failed to load prompt template: {e}")
    HAZARD_EXTRACTION_PROMPT_TEMPLATE = None # Set to None to handle gracefully

def call_traced_hazard_extraction_udf(text: str) -> str:
    """
    PySpark UDF wrapper to call the MLflow traced LLM function.
    """
    if HAZARD_EXTRACTION_PROMPT_TEMPLATE is None:
        logger.error("Prompt template not loaded, cannot perform extraction.")
        return json.dumps({"error": "Prompt template not loaded", "hazards": [], "confidence_score": 0.0})

    # MLflow Trace Session: Group multiple UDF calls into a single trace session
    # This is crucial for batch processing observability.
    # Ensure this code runs within a Spark job where MLflow tracing is enabled.
    # MLflow typically starts a trace session automatically if running within a Spark job
    # and the @mlflow.trace decorator is used on a function called from that job.
    # For explicit session management:
    # with mlflow.start_trace_session(session_name="batch_hazard_extraction_run"):
    #    result = extract_hazards_with_ollama(text, HAZARD_EXTRACTION_PROMPT_TEMPLATE)
    # The decorator handles session start/end implicitly when called from a traced context.

    try:
        result = extract_hazards_with_ollama(text, HAZARD_EXTRACTION_PROMPT_TEMPLATE)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"UDF error during hazard extraction for text: {text[:100]}... - {e}")
        return json.dumps({
            "error": f"UDF execution failed: {e}",
            "hazards": [],
            "confidence_score": 0.0
        })

# Register the UDF
hazard_extraction_udf = udf(call_traced_hazard_extraction_udf, StringType())

# --- Main Spark Processing Logic ---
def process_bronze_data_for_silver(spark: SparkSession, input_path: str, output_path: str):
    """
    Reads data from Bronze layer, enriches it with hazard extraction,
    and writes to Silver layer.
    """
    logger.info(f"Starting MLflow Tracing setup for Silver AI Enrichment...")

    # Start an MLflow run for this batch processing job
    with mlflow.start_run(run_name="Silver Layer - AI Hazard Enrichment Batch"):
        logger.info("MLflow run started for Silver AI Enrichment.")

        # Load Bronze data
        logger.info(f"Loading data from Bronze layer: {input_path}")
        df_bronze = spark.read.format("delta").load(input_path)

        # Apply the UDF to extract hazards
        logger.info("Applying hazard extraction UDF to DataFrame...")
        # IMPORTANT: Replace 'bronze_text_column' with the actual name of your text column in the Bronze Delta Table.
        df_silver = df_bronze.withColumn(
            "extracted_hazards_json",
            hazard_extraction_udf(col("bronze_text_column"))
        )

        # Add a column for the original source text for later evaluation.
        # This assumes 'bronze_text_column' is the source of truth.
        df_silver = df_silver.withColumnRenamed("bronze_text_column", "source_text_for_evaluation")

        # Save to Silver layer
        logger.info(f"Writing enriched data to Silver layer: {output_path}")
        df_silver.write.format("delta").mode("overwrite").save(output_path)

        logger.info("Silver layer data processed and saved.")

        # --- Explicitly Log Trace Session if not automatically managed ---
        # In many Spark execution contexts with MLflow enabled, a trace session
        # might be started automatically around the Spark job. If not, you can
        # explicitly start and end one, but using the @mlflow.trace decorator
        # on the UDF's underlying Python function is the standard approach.
        # The decorator will create a trace, and if multiple are called within
        # a single job run, they might be grouped under the parent MLflow Run.
        # For explicit session management:
        # with mlflow.start_trace_session(session_name="batch_processing_session"):
        #    # ... trigger UDFs here ...
        # The decorator approach is generally preferred for simplicity.

        logger.info("MLflow run for Silver AI Enrichment completed.")


if __name__ == "__main__":
    spark = SparkSession.builder 
        .appName("SilverAIEnhancement") 
        .config("spark.sql.extensions", "io.delta.tables.DeltaSparkSessionExtension") 
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") 
        .getOrCreate()

    # --- User-configurable paths ---
    # IMPORTANT: Replace these with your actual Delta Table paths.
    BRONZE_LAYER_PATH = "/path/to/your/bronze/delta/table" # e.g., "/mnt/data/bronze/logs"
    SILVER_LAYER_PATH = "/path/to/your/silver/delta/table" # e.g., "/mnt/data/silver/enriched_logs"

    # --- Data Loading and Processing ---
    try:
        process_bronze_data_for_silver(spark, BRONZE_LAYER_PATH, SILVER_LAYER_PATH)
        logger.info("Script executed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during script execution: {e}", exc_info=True)
    finally:
        spark.stop()
