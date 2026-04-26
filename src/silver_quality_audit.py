# src/silver_quality_audit.py
import mlflow
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure MLflow tracking is configured
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if using a remote tracking server

# Models to be used:
# - Llama 3 for extraction (this script will call it, or a similar function)
# - DeepSeek for judging: "deepseek-coder-r1"
LLAMA3_MODEL_NAME = "llama3"
DEEPSEEK_JUDGE_MODEL_NAME = "deepseek-coder-r1" # Ensure this model is available
# IMPORTANT: Replace with the actual module path and function name if they differ.
HAZARD_EXTRACTION_FUNC_SOURCE = "src.silver_ai_enrichment"
HAZARD_EXTRACTION_FUNC_NAME = "extract_hazards_with_ollama" # This is the function that is traced

# --- Paths ---
# IMPORTANT: Replace with your actual Silver Delta Table path.
SILVER_DELTA_PATH = "/path/to/your/silver/delta/table" # e.g., "/mnt/data/silver/enriched_logs"

# --- Helper Functions ---

def get_mlflow_prompt_template(prompt_name: str, version: str = "1.0.0") -> str:
    """Loads only the prompt template string from MLflow Prompt Management."""
    try:
        prompt_manager = mlflow.llms.prompts.PromptManager()
        prompt_template_obj = prompt_manager.get_prompt_template(name=prompt_name, version=version)
        return prompt_template_obj.template
    except Exception as e:
        logger.error(f"Error loading MLflow prompt template '{prompt_name}@{version}': {e}")
        raise

# --- Mock Ollama API Response for Judge ---
# In a real scenario, this would come from the actual ollama library response.
def mock_ollama_judge_response(content: str) -> Dict[str, Any]:
    """Simulates an Ollama API response for the judge LLM."""
    # Simulate a judge response
    return {
        "response": json.dumps({
            "score": 0.9, # Example score
            "reasoning": "The extracted JSON accurately identifies 'Data Quality' and 'Infrastructure' hazards, matching the source text. Confidence is high. No significant hallucinations detected."
        }),
        "model": DEEPSEEK_JUDGE_MODEL_NAME,
        "created_at": "2023-04-26T10:05:00Z",
        "done_reason": "stop",
        "context": [67890, 09876],
        "total_duration": 600, # ms
        "load_duration": 300, # ms
        "prompt_eval_count": 100,
        "eval_count": 50,
        "eval_duration": 600,
    }

# --- Custom MLflow GenAI Metric: DeepSeek Judge ---
# This function will be called by mlflow.evaluate for each sample.
# It acts as the 'Judge' to score the output of the 'model' (Llama 3 extractor).
def deepseek_json_quality_judge(
    model, # The LLM model being evaluated (e.g., Llama 3 extractor function)
    context: pd.DataFrame, # DataFrame containing context columns (source_text)
    predictions: pd.DataFrame, # DataFrame containing model predictions (extracted_json)
    prompt_template: str, # The judge prompt template
    judge_model_name: str = DEEPSEEK_JUDGE_MODEL_NAME,
    ollama_client: ollama = ollama # Pass client for potential mocking/injection
) -> Dict[str, Any]:
    """
    Custom MLflow GenAI metric function that uses DeepSeek-R1 as a judge.
    Evaluates the quality of extracted JSON against the source text.

    Args:
        model: The MLflow LLM model object being evaluated (e.g., Llama 3 extractor).
        context: DataFrame containing original source text.
        predictions: DataFrame containing the LLM's extracted JSON output.
        prompt_template: The prompt template for the judge LLM.
        judge_model_name: The name of the LLM model to use as the judge.
        ollama_client: The Ollama client instance.

    Returns:
        A dictionary mapping metric names to their values (e.g., {"judge_score": 0.85}).
    """
    # Extract required columns for context and predictions
    # These column names must match what's used in the MLflow Dataset and the 'model' callable output.
    source_texts = context["source_text_for_evaluation"].tolist()
    extracted_jsons_str = predictions["extracted_hazards_json"].tolist()

    # Ensure prompt template is available
    if not prompt_template:
        raise ValueError("Prompt template for the judge is missing.")

    results = []
    for i, (source_text, extracted_json_str) in enumerate(zip(source_texts, extracted_jsons_str)):
        score = 0.0
        reasoning = "N/A"
        try:
            # Parse the extracted JSON string to ensure it's valid before sending to judge
            try:
                extracted_data = json.loads(extracted_json_str)
                extracted_json_formatted = json.dumps(extracted_data, indent=2)
            except json.JSONDecodeError:
                logger.warning(f"Skipping judge evaluation for row {i}: Invalid JSON in predictions: {extracted_json_str[:200]}...")
                score = 0.0
                reasoning = "Invalid JSON format in prediction."
            except Exception as e:
                logger.error(f"Unexpected error parsing prediction for row {i}: {e}")
                score = 0.0
                reasoning = f"Unexpected error parsing prediction: {e}"
            else:
                # Format prompt for the judge LLM
                formatted_prompt = prompt_template.replace("{{source_text}}", source_text)
                formatted_prompt = formatted_prompt.replace("{{extracted_json}}", extracted_json_formatted)

                # --- Call the Judge LLM (DeepSeek) ---
                start_time_judge = time.time()
                # response = ollama_client.chat(
                #     model=judge_model_name,
                #     messages=[{"role": "user", "content": formatted_prompt}],
                #     stream=False
                # )
                # judge_response_content = response['message']['content']

                # --- Simulate Judge response for demonstration ---
                simulated_judge_response_content = mock_ollama_judge_response("")["response"]
                # --- End Simulation ---

                try:
                    judge_output = json.loads(simulated_judge_response_content)
                    score = judge_output.get("score", 0.0)
                    reasoning = judge_output.get("reasoning", "No reasoning provided by judge.")
                except json.JSONDecodeError:
                    logger.error(f"Judge LLM returned invalid JSON: {simulated_judge_response_content}")
                    score = 0.0
                    reasoning = "Judge LLM returned invalid JSON."
                except Exception as e:
                    logger.error(f"Error processing judge output: {e}")
                    score = 0.0
                    reasoning = f"Error processing judge output: {e}"
        except Exception as e:
            logger.error(f"Error processing row {i} for judge: {e}")
            score = 0.0
            reasoning = f"Error during processing: {e}"
        finally:
            results.append({"judge_score": score, "judge_reasoning": reasoning})

    # MLflow expects a DataFrame-like structure for metrics if returning multiple values per sample.
    # For simplicity, we'll return the average score. In a real scenario, you might return per-row metrics.
    if not results:
        return {"judge_score": 0.0}

    avg_score = sum([r["judge_score"] for r in results]) / len(results)
    return {"judge_score": avg_score}


# --- MLflow Evaluation Function ---
def evaluate_silver_data_quality(spark: SparkSession):
    """
    Evaluates the quality of extracted JSON from Silver layer data using MLflow.
    """
    logger.info("Starting MLflow Evaluation for Silver Layer Data Quality...")

    # 1. Load Sample Data
    logger.info(f"Loading 5% sample from Silver Delta Table: {SILVER_DELTA_PATH}")
    try:
        df_silver = spark.read.format("delta").load(SILVER_DELTA_PATH)
    except Exception as e:
        logger.error(f"Failed to load Silver Delta Table from {SILVER_DELTA_PATH}: {e}")
        return

    # Sample 5% of the data and convert to Pandas DataFrame
    # Ensure your Silver table has 'source_text_for_evaluation' and 'extracted_hazards_json' columns.
    silver_sample_df = df_silver.sample(withReplacement=False, fraction=0.05, seed=42).toPandas()

    if silver_sample_df.empty:
        logger.warning("No data found in the Silver sample. Skipping evaluation.")
        return

    # Create MLflow Pandas Dataset
    mlflow_dataset = mlflow.data.from_pandas(
        silver_sample_df,
        source=f"delta://{SILVER_DELTA_PATH}", # Indicate source
        # context_columns are passed to metric functions. 'predictions_column' is inferred from model output if not specified.
        context_columns=["source_text_for_evaluation"] # Column containing original source text
    )
    logger.info(f"Created MLflow Dataset from {len(silver_sample_df)} records.")

    # 2. Load Judge Prompt Template
    try:
        JUDGE_PROMPT_TEMPLATE = get_mlflow_prompt_template(
            prompt_name="hallucination_judge",
            version="1.0.0" # Load the specific version
        )
        logger.info("Loaded judge prompt template from MLflow.")
    except Exception as e:
        logger.error(f"Failed to load judge prompt: {e}")
        return

    # 3. Get the model/function to be evaluated (Llama 3 extractor)
    # This needs to be a callable that MLflow evaluate can use.
    # It should accept a Pandas DataFrame and return a Pandas Series of predictions.
    # We'll dynamically load the extraction function and its prompt.

    # First, get the Llama 3 extractor prompt
    try:
        LLAMA3_EXTRACTION_PROMPT_TEMPLATE = get_mlflow_prompt_template(
            prompt_name="hazard_extraction",
            version="1.0.0"
        )
        logger.info("Loaded extractor prompt template from MLflow.")
    except Exception as e:
        logger.error(f"Failed to load extractor prompt: {e}")
        return

    # Dynamically import the extractor function to avoid circular dependencies if this script imports silver_ai_enrichment directly.
    # In a real project, you might manage imports more explicitly.
    try:
        # This is a placeholder. In a real setup, ensure the function is importable.
        # If `extract_hazards_with_ollama` is in `src.silver_ai_enrichment`, you'd do:
        # from src.silver_ai_enrichment import extract_hazards_with_ollama
        # For this example, we'll create a local wrapper that calls it.
        # This function will be what `mlflow.evaluate` calls.
        def llama3_extractor_for_evaluation_wrapper(
            df: pd.DataFrame, # MLflow passes a DataFrame
            prompt_template: str = LLAMA3_EXTRACTION_PROMPT_TEMPLATE,
            model_name: str = LLAMA3_MODEL_NAME
        ) -> pd.Series:
            """
            Wrapper function for MLflow evaluate. Takes a DataFrame and returns a Series of predictions.
            Calls the MLflow-traced extraction function.
            """
            # Assumes 'source_text_for_evaluation' is in the DataFrame passed by mlflow.evaluate
            # This column name MUST match the one in the MLflow Dataset's context_columns or the data itself.
            texts_to_process = df["source_text_for_evaluation"]
            
            predictions_list = []
            for text in texts_to_process:
                try:
                    # Call the MLflow traced function
                    # Note: This assumes `mlflow.llms.extract_hazards_with_ollama` is available.
                    # If not, you'd need to import `extract_hazards_with_ollama` from `src.silver_ai_enrichment`
                    # and call it directly.
                    # For this context, we're simulating calling the traced function.
                    # To ensure traces are logged, the `mlflow.evaluate` needs to invoke the *actual* traced function.
                    # The `model` argument is supposed to be a callable that accepts data and produces predictions.
                    # If `extract_hazards_with_ollama` is correctly defined and importable, MLflow will trace it.
                    # Let's assume `extract_hazards_with_ollama` is accessible in the evaluation environment.
                    # This requires `src/silver_ai_enrichment.py` to be importable or its functions available.

                    # Re-simulating call to the traced function for clarity in the wrapper
                    # In a real project, you'd import `extract_hazards_with_ollama` from `src.silver_ai_enrichment`
                    # and call it here.
                    # For this generated code, we'll call it as if it's in the same scope or imported.
                    # If `src/silver_ai_enrichment.py` is in the PYTHONPATH, this import works:
                    # from src.silver_ai_enrichment import extract_hazards_with_ollama
                    
                    # Placeholder for actual LLM call (simulated)
                    # In a real run, this would call the actual LLM inference function
                    # which might be traced.
                    simulated_extraction_output = {
                        "hazards": [{"type": "Placeholder", "description": "Simulated extraction for evaluation", "severity": "Low"}],
                        "confidence_score": 0.8
                    }
                    # Here, we're effectively bypassing the actual traced function call for the purpose of code generation,
                    # but in a real MLflow.evaluate() run, it *would* call the traced function if correctly set up.
                    # To ensure traces are logged, the `mlflow.evaluate` needs to invoke the *actual* traced function.
                    # The `model` argument is supposed to be a callable that accepts data and produces predictions.
                    # If `extract_hazards_with_ollama` is correctly defined and importable, MLflow will trace it.
                    # Let's assume `extract_hazards_with_ollama` is accessible in the evaluation environment.
                    # This requires `src/silver_ai_enrichment.py` to be importable or its functions available.

                    # Re-simulating call to the traced function for clarity in the wrapper
                    # In a real project, you'd import `extract_hazards_with_ollama` from `src.silver_ai_enrichment`
                    # and call it here.
                    # For this generated code, we'll call it as if it's in the same scope or imported.
                    # If `src/silver_ai_enrichment.py` is in the PYTHONPATH, this import works:
                    # from src.silver_ai_enrichment import extract_hazards_with_ollama
                    
                    # Placeholder for actual LLM call (simulated)
                    # In a real run, this would call the actual LLM inference function
                    # which might be traced.
                    
                    # Using a placeholder call that simulates the traced function's output
                    # In a real execution, this part would call the actual traced function.
                    # For demonstration, let's simulate the traced function's output.
                    # This simulates the JSON output *after* the traced function runs.
                    # The trace itself will be logged by the decorator on the actual function.
                    extracted_data = {
                        "hazards": [{"type": "Evaluation Simulation", "description": f"Simulated hazard for text: {text[:50]}...", "severity": "Low"}],
                        "confidence_score": 0.85
                    }

                    predictions_list.append(json.dumps(extracted_data))
                except Exception as e:
                    logger.error(f"Error during LLM extraction in evaluation wrapper for text: {text[:50]}... - {e}")
                    predictions_list.append(json.dumps({
                        "error": f"Evaluation wrapper failed: {e}",
                        "hazards": [],
                        "confidence_score": 0.0
                    }))
            return pd.Series(predictions_list)
            
        # The 'model' argument in mlflow.evaluate can be a callable.
        # If it's a callable that takes a DataFrame and returns a Series of predictions,
        # it works directly.
        model_callable_for_eval = lambda df: llama3_extractor_for_evaluation_wrapper(df)

    except ImportError:
        logger.error("Could not import 'extract_hazards_with_ollama' from src.silver_ai_enrichment. Ensure the file is accessible and the function is defined.")
        return
    except Exception as e:
        logger.error(f"Failed to set up model callable for evaluation: {e}")
        return


    # 4. Create Custom MLflow GenAI Metric
    quality_judge_metric = mlflow.metrics.genai.make_genai_metric(
        name="json_quality_judge",
        query=JUDGE_PROMPT_TEMPLATE,
        context_cols=["source_text_for_evaluation"], # Column from the MLflow Dataset to use as context for the judge
        human_readable_name="JSON Extraction Quality Judge",
        metric_fn=deepseek_json_quality_judge # Our custom scoring function
    )

    # 5. Execute MLflow Evaluation
    logger.info("Running mlflow.evaluate()...")
    try:
        # If you have ground truth JSONs, you'd load them and pass to 'targets'.
        # Here, we're focusing on judging the output quality against the source text.
        # The 'model' callable generates the predictions.
        eval_results = mlflow.evaluate(
            data=mlflow_dataset,
            model=model_callable_for_eval, # The LLM function to evaluate
            targets=None, # No specific ground truth for JSON structure in this example, judge scores output quality.
                          # If you had a column of 'correct_json', you'd pass it here.
            extra_metrics=[quality_judge_metric], # Our custom judge metric
            model_batch_size=10, # Process in batches for efficiency
            # Pass any necessary arguments to the model callable if needed via `model_kwargs`
            # model_kwargs={"prompt_template": LLAMA3_EXTRACTION_PROMPT_TEMPLATE} # Example
        )
        logger.info("MLflow Evaluation run completed.")
        logger.info(f"Evaluation results summary: {eval_results.metrics}") # Display summary metrics

    except Exception as e:
        logger.error(f"An error occurred during MLflow evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    spark = SparkSession.builder 
        .appName("SilverQualityAudit") 
        .config("spark.sql.extensions", "io.delta.tables.DeltaSparkSessionExtension") 
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") 
        .getOrCreate()

    # --- Start a parent MLflow run for the entire audit process ---
    # This run will contain the evaluation run as a child run.
    with mlflow.start_run(run_name="Silver Layer - Quality Audit Run"):
        logger.info("MLflow run started for Silver Quality Audit.")
        evaluate_silver_data_quality(spark)
        logger.info("Script execution finished.")
    finally:
        spark.stop()
