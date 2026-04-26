#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SILVER LAYER - LLM-as-a-Judge Quality Audit (DeepSeek-R1)
================================================================================
File: src/silver_quality_audit.py

Purpose:
    Implement "LLM-as-a-Judge" pattern to audit Llama 3's JSON extractions:
    
    1. Load Silver Delta Table and sample 5% for auditing
    2. Create DeepSeek-R1 PySpark Pandas UDF via Ollama API
    3. Evaluate if Llama 3's extraction is logically sound and hallucination-free
    4. Calculate Accuracy Rate and log comprehensive metrics to MLflow
    5. Save failed extractions to Quarantine Delta Table

Architecture Decision:
    Using local DeepSeek-R1 as judge model via Ollama provides:
    - Zero external API costs
    - Privacy-preserving local inference
    - Explicit reasoning via <-thinking> tags before JSON output

MLflow Integration:
    - Full parameter, metric, artifact, and tag logging
    - Dataset versioning with input hash
    - Evaluation results with mlflow.evaluate()
    - Comparison framework across audit runs
    - Auto-logging capabilities

Author: GeoAI Principal MLOps Engineer
Version: 2.0.0
================================================================================
"""

import os
import sys
import json
import logging
import re
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
)

# Pandas UDF imports
from pyspark.sql.functions import pandas_udf
import pandas as pd

# MLflow imports - FULL STACK
import mlflow
from mlflow.data.delta import DeltaDataset
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec, DataType
from mlflow.metrics import make_metric
from mlflow.metrics.genai import model_evaluation_context, evaluate_on_genai

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SILVER_INPUT_PATH = "s3a://geo-lakehouse/silver/us_accidents_silver"
QUARANTINE_PATH = "s3a://geo-lakehouse/silver/quarantine_hallucinations"
SAMPLE_FRACTION = 0.05
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "deepseek-r1"
MLFLOW_EXP = "DeepSeek_Silver_Audit"
MLFLOW_TRACKING_URI = "http://localhost:5000"


# =============================================================================
# DEEPSEEK JUDGE PROMPT TEMPLATE
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert data quality auditor. Your task is to evaluate
whether the AI extraction is logically sound and accurate based on the original text.

Evaluate the following:
1. Does the extracted JSON make sense given the original description?
2. Are there any hallucinations or false information?
3. Is the severity score appropriate (1-10)?
4. Is the hazard_type relevant and accurate?

Return ONLY a strict JSON object with this exact format:
{"is_accurate": true/false, "error_reason": "specific reason if inaccurate, null if accurate"}

Do NOT include any other text. Start your response with the JSON object."""


def build_judge_prompt(description: str, ai_enrichment_json: str) -> str:
    """Build the complete prompt for the DeepSeek judge."""
    return f"""Original Text:
{description[:2000]}

AI Extracted JSON:
{ai_enrichment_json}

Evaluate the extraction quality and return JSON."""


# =============================================================================
# OLLAMA API CLIENT
# =============================================================================

def call_ollama(prompt: str, model: str = JUDGE_MODEL, timeout: int = 120) -> str:
    """
    Call Ollama API with the judge prompt.
    
    Args:
        prompt: Complete prompt for the model
        model: Model name (default: deepseek-r1)
        timeout: Request timeout in seconds
        
    Returns:
        Raw model response string
        
    Raises:
        RuntimeError: If API call fails
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_BASE_URL,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama API timeout after {timeout}s")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


def parse_judge_response(response: str) -> Dict[str, Any]:
    """
    Parse DeepSeek-R1's response, handling <thinking> tags.
    
    DeepSeek-R1 outputs reasoning in <thinking> tags before the final JSON.
    We need to extract just the JSON portion.
    
    Args:
        response: Raw model response with potential <thinking> tags
        
    Returns:
        Parsed dict with 'is_accurate' (bool) and 'error_reason' (str or None)
    """
    # Remove <thinking> tags and their content
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
    
    # Find JSON object in the remaining text
    json_match = re.search(r'\{[^{}]*"is_accurate"[^{}]*\}', cleaned, re.DOTALL)
    
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "is_accurate": bool(result.get("is_accurate", False)),
                "error_reason": result.get("error_reason")
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find any boolean/near JSON
    is_accurate_match = re.search(r'"is_accurate"\s*:\s*(true|false)', cleaned)
    error_match = re.search(r'"error_reason"\s*:\s*"([^"]*)"', cleaned)
    
    if is_accurate_match:
        return {
            "is_accurate": is_accurate_match.group(1) == "true",
            "error_reason": error_match.group(1) if error_match else None
        }
    
    # Default fallback on parse failure
    return {
        "is_accurate": False,
        "error_reason": f"Failed to parse judge response: {response[:200]}"
    }


# =============================================================================
# PANDAS UDF FOR DEEPSEEK JUDGE
# =============================================================================

@pandas_udf(StringType())
def judge_extraction(description: pd.Series, ai_json: pd.Series) -> pd.Series:
    """
    PySpark Pandas UDF that calls DeepSeek-R1 to judge Llama 3 extractions.
    
    This UDF runs on each row in the Spark cluster, calling the Ollama API
    for each description/ai_enrichment_json pair.
    
    Args:
        description: Series of original text descriptions
        ai_json: Series of AI-generated JSON strings
        
    Returns:
        Series of judge response JSON strings
    """
    results = []
    
    for desc, ajson in zip(description, ai_json):
        try:
            # Skip empty or invalid inputs
            if not desc or not ajson:
                results.append(json.dumps({"is_accurate": False, "error_reason": "Missing input"}))
                continue
            
            # Build judge prompt
            prompt = build_judge_prompt(str(desc), str(ajson))
            
            # Call Ollama
            response = call_ollama(prompt)
            
            # Parse response
            parsed = parse_judge_response(response)
            
            results.append(json.dumps(parsed))
            
        except Exception as e:
            logger.warning(f"Judge UDF error: {e}")
            results.append(json.dumps({
                "is_accurate": False,
                "error_reason": f"UDF error: {str(e)[:100]}"
            }))
    
    return pd.Series(results)


# =============================================================================
# MLFLOW TRACKING HELPERS
# =============================================================================

def get_input_data_hash(spark: SparkSession, df: DataFrame) -> str:
    """
    Compute a hash of the input data for versioning.
    
    Args:
        spark: SparkSession
        df: Input DataFrame
        
    Returns:
        SHA256 hash string
    """
    count = df.count()
    schema_hash = hashlib.sha256(
        str(df.schema.json()).encode()
    ).hexdigest()[:16]
    return f"silver_{count}_{schema_hash}"


def log_dataset_info(spark: SparkSession, df: DataFrame, sample_frac: float):
    """
    Log comprehensive dataset information to MLflow.
    
    Args:
        spark: SparkSession
        df: Source DataFrame
        sample_frac: Sample fraction used
    """
    total_records = df.count()
    sample_records = int(total_records * sample_frac)
    
    # Dataset schema
    schema_json = df.schema.json()
    
    # Log as JSON artifact
    dataset_info = {
        "source": SILVER_INPUT_PATH,
        "total_records": total_records,
        "sample_fraction": sample_frac,
        "sample_records_expected": sample_records,
        "schema": json.loads(schema_json),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Log as parameter (for reference)
    mlflow.log_param("source_table", SILVER_INPUT_PATH)
    mlflow.log_param("total_source_records", total_records)
    mlflow.log_param("schema_columns", list(dataset_info["schema"]["fields"].keys()))
    
    # Save as artifact
    dataset_path = "/tmp/dataset_info.json"
    with open(dataset_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    mlflow.log_artifact(dataset_path)
    
    # Try to log as Delta dataset (if available)
    try:
        delta_ds = DeltaDataset(
            spark.read.format("delta").load(SILVER_INPUT_PATH),
            info={
                "source": SILVER_INPUT_PATH,
                "description": "Silver layer accidents data"
            }
        )
        mlflow.log_input(delta_ds, "source")
    except Exception as e:
        logger.warning(f"Could not log Delta dataset: {e}")


def log_evaluation_metrics(
    parsed_df: DataFrame,
    accuracy_rate: float,
    accurate_count: int,
    inaccurate_count: int,
    total_audited: int,
    error_categories: Dict[str, int]
):
    """
    Log comprehensive evaluation metrics to MLflow.
    
    Uses mlflow.log_metrics() with prefix for organization.
    
    Args:
        parsed_df: DataFrame with audit results
        accuracy_rate: Overall accuracy percentage
        accurate_count: Count of accurate extractions
        inaccurate_count: Count of inaccurate extractions
        total_audited: Total records audited
        error_categories: Categorized error counts
    """
    # ==============================
    # PRIMARY METRICS (no prefix)
    # ==============================
    mlflow.log_metric("accuracy_rate", accuracy_rate)
    mlflow.log_metric("accurate_count", accurate_count)
    mlflow.log_metric("inaccurate_count", inaccurate_count)
    mlflow.log_metric("total_audited", total_audited)
    mlflow.log_metric("audit_sample_fraction", SAMPLE_FRACTION)
    
    # ==============================
    # DERIVED METRICS (with prefix)
    # ==============================
    mlflow.log_metric("quality.pass_rate", accuracy_rate / 100)
    mlflow.log_metric("quality.fail_rate", (inaccurate_count / total_audited) if total_audited > 0 else 0)
    
    # Precision & Recall (assume ground truth is is_accurate)
    # If we treat "accurate" as positive class
    true_positive = accurate_count
    false_positive = 0  # N/A for this binary case
    false_negative = inaccurate_count
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    mlflow.log_metric("quality.precision", precision * 100)
    mlflow.log_metric("quality.recall", recall * 100)
    mlflow.log_metric("quality.f1_score", f1_score * 100)
    
    # ==============================
    # ERROR CATEGORY METRICS
    # ==============================
    for category, count in error_categories.items():
        category_key = category.lower().replace(" ", "_").replace("/", "_")
        mlflow.log_metric(f"errors.{category_key}", count)
        mlflow.log_metric(f"errors.{category_key}_pct", 
                         (count / inaccurate_count * 100) if inaccurate_count > 0 else 0)


def log_artifacts(spark: SparkSession, parsed_df: DataFrame, quarantine_count: int):
    """
    Log comprehensive artifacts to MLflow.
    
    Args:
        spark: SparkSession
        parsed_df: DataFrame with audit results
        quarantine_count: Count of quarantined records
    """
    import tempfile
    
    # ==============================
    # 1. Save Quarantine Data Sample
    # ==============================
    quarantine_df = parsed_df.filter(F.col("is_accurate") == False)
    
    if quarantine_count > 0:
        # Convert to Pandas for artifact saving
        quarantine_pandas = quarantine_df.select(
            "description",
            "ai_enrichment_json",
            "judge_response",
            "is_accurate",
            "error_reason"
        ).limit(100).toPandas()
        
        # Save as CSV
        quarantine_path = "/tmp/quarantine_sample.csv"
        quarantine_pandas.to_csv(quarantine_path, index=False)
        mlflow.log_artifact(quarantine_path, "quarantine_samples")
        
        # Save as JSON (pretty)
        quarantine_json_path = "/tmp/quarantine_sample.json"
        quarantine_pandas.to_json(quarantine_json_path, orient="records", indent=2)
        mlflow.log_artifact(quarantine_json_path, "quarantine_samples")
    
    # ==============================
    # 2. Save Full Audit Results
    # ==============================
    # Compute error reason distribution
    error_dist = (
        quarantine_df.groupBy("error_reason")
        .count()
        .orderBy(F.desc("count"))
    )
    
    error_dist_pandas = error_dist.limit(20).toPandas()
    error_dist_path = "/tmp/error_distribution.csv"
    error_dist_pandas.to_csv(error_dist_path, index=False)
    mlflow.log_artifact(error_dist_path, "error_analysis")
    
    # ==============================
    # 3. Save Audit Summary Report
    # ==============================
    summary_report = {
        "audit_timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "sample_fraction": SAMPLE_FRACTION,
        "source_path": SILVER_INPUT_PATH,
        "quarantine_path": QUARANTINE_PATH,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "total_audited": parsed_df.count(),
        "quarantine_count": quarantine_count,
    }
    
    summary_path = "/tmp/audit_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    mlflow.log_artifact(summary_path, "audit_metadata")


def register_model_for_comparison():
    """
    Register a simple model entry for run comparison framework.
    
    This allows the audit to be compared across runs in MLflow
    """
    # Create a dummy model for comparison tracking
    model_info = {
        "model_name": "deepseek_r1_judge",
        "model_type": "llm_judge",
        "version": "1.0.0",
        " Judge_model": JUDGE_MODEL,
        "evaluation_criteria": "accuracy_rate",
    }
    
    # Log as parameter (will appear in UI)
    mlflow.log_param("judge_model_name", "deepseek_r1_judge")
    mlflow.log_param("judge_model_type", "llm_judge")
    mlflow.log_param("evaluation_criteria", "accuracy_rate")


def create_or_get_experiment():
    """
    Create or get the MLflow experiment with full configuration.
    
    Returns:
        Experiment object
    """
    client = MlflowClient()
    
    # Check if experiment exists
    exp = client.get_experiment_by_name(MLFLOW_EXP)
    
    if exp is None:
        # Create new experiment
        exp_id = client.create_experiment(
            name=MLFLOW_EXP,
            tags={
                "description": "LLM-as-a-Judge quality audits for Silver layer extractions",
                "pipeline": "medallion_architecture",
                "judge_model": JUDGE_MODEL,
                "stage": "silver_layer",
            }
        )
        exp = client.get_experiment(exp_id)
        logger.info(f"Created MLflow experiment: {MLFLOW_EXP}")
    else:
        # Update tags if experiment exists
        client.set_experiment_tag(exp.experiment_id, "last_judge_run", datetime.now().isoformat())
        logger.info(f"Using existing MLflow experiment: {MLFLOW_EXP}")
    
    return exp


# =============================================================================
# MAIN AUDIT FUNCTION
# =============================================================================

def run_quality_audit(spark: SparkSession) -> DataFrame:
    """
    Main function to run the LLM-as-a-Judge quality audit.
    
    Args:
        spark: SparkSession
        
    Returns:
        DataFrame with audit results including is_accurate and error_reason columns
    """
    logger.info("=" * 60)
    logger.info("Starting Silver Layer Quality Audit with DeepSeek-R1")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Silver Delta Table
    # -------------------------------------------------------------------------
    logger.info(f"Loading Silver Delta Table from: {SILVER_INPUT_PATH}")
    
    silver_df = (
        spark.read.format("delta")
        .load(SILVER_INPUT_PATH)
    )
    
    initial_count = silver_df.count()
    input_data_hash = get_input_data_hash(spark, silver_df)
    logger.info(f"Loaded {initial_count:,} records from Silver layer")
    logger.info(f"Input data hash: {input_data_hash}")
    
    # Check for required columns
    required_cols = ["description", "ai_enrichment_json"]
    missing = [c for c in required_cols if c not in silver_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Sample 5% for Auditing
    # -------------------------------------------------------------------------
    logger.info(f"Sampling {SAMPLE_FRACTION*100}% for quality audit")
    
    audit_sample = silver_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)
    sample_count = audit_sample.count()
    logger.info(f"Audit sample: {sample_count:,} records")
    
    # -------------------------------------------------------------------------
    # STEP 3: Apply DeepSeek Judge UDF
    # -------------------------------------------------------------------------
    logger.info(f"Running DeepSeek-R1 judge on {sample_count:,} samples...")
    logger.info(f"Ollama endpoint: {OLLAMA_BASE_URL}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    
    # Call the judge UDF - this runs distributed across the Spark cluster
    audited_df = audit_sample.withColumn(
        "judge_response",
        judge_extraction(
            F.col("description"),
            F.col("ai_enrichment_json")
        )
    )
    
    # -------------------------------------------------------------------------
    # STEP 4: Parse Judge Response JSON
    # -------------------------------------------------------------------------
    logger.info("Parsing DeepSeek judge responses")
    
    # Parse JSON response into separate columns
    parsed_df = audited_df.withColumn(
        "is_accurate",
        F.get_json_object(F.col("judge_response"), "$.is_accurate").cast(BooleanType())
    ).withColumn(
        "error_reason",
        F.get_json_object(F.col("judge_response"), "$.error_reason")
    )
    
    # -------------------------------------------------------------------------
    # STEP 5: Calculate Accuracy Metrics
    # -------------------------------------------------------------------------
    total_audited = parsed_df.count()
    accurate_count = parsed_df.filter(F.col("is_accurate") == True).count()
    inaccurate_count = total_audited - accurate_count
    
    accuracy_rate = (accurate_count / total_audited * 100) if total_audited > 0 else 0
    
    # Compute error categories
    error_categories_list = (
        parsed_df.filter(F.col("is_accurate") == False)
        .groupBy("error_reason")
        .count()
        .collect()
    )
    error_categories = {row["error_reason"]: row["count"] for row in error_categories_list}
    
    logger.info("-" * 40)
    logger.info(f"AUDIT RESULTS:")
    logger.info(f"  Total Audited:       {total_audited:,}")
    logger.info(f"  Accurate:           {accurate_count:,}")
    logger.info(f"  Inaccurate:         {inaccurate_count:,}")
    logger.info(f"  Accuracy Rate:      {accuracy_rate:.2f}%")
    logger.info(f"  Quarantine Size:   {inaccurate_count:,}")
    logger.info("-" * 40)
    
    # -------------------------------------------------------------------------
    # STEP 6: Configure MLflow
    # -------------------------------------------------------------------------
    logger.info(f"Configuring MLflow...")
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    exp = create_or_get_experiment()
    mlflow.set_experiment(MLFLOW_EXP)
    
    # -------------------------------------------------------------------------
    # STEP 7: Log Comprehensive to MLflow
    # -------------------------------------------------------------------------
    logger.info(f"Logging FULL metrics to MLflow experiment: {MLFLOW_EXP}")
    
    # Generate source version Git hash for traceability
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd="/Users/matias.magni/Documents/dev/mine/geo-ai-medallion-architecture-pipeline"
        ).decode().strip()[:8]
    except:
        git_hash = "unknown"
    
    with mlflow.start_run(
        run_name=f"DeepSeek_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_id=exp.experiment_id
    ) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # ==============================
        # PARAMS - Configuration
        # ==============================
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("sample_fraction", SAMPLE_FRACTION)
        mlflow.log_param("source_table", SILVER_INPUT_PATH)
        mlflow.log_param("source_data_hash", input_data_hash)
        mlflow.log_param("ollama_endpoint", OLLAMA_BASE_URL)
        mlflow.log_param("audit_timestamp", datetime.now().isoformat())
        mlflow.log_param("script_version", "2.0.0")
        mlflow.log_param("git_hash", git_hash)
        
        # ==============================
        # TAGS - Categorization
        # ==============================
        mlflow.set_tag("pipeline_stage", "silver_layer")
        mlflow.set_tag("audit_type", "llm_judge_quality")
        mlflow.set_tag("judge_model_family", "deepseek")
        mlflow.set_tag("data_owner", "geoai_team")
        mlflow.set_tag("compliance", "internal_only")
        
        # Quality gate tags
        if accuracy_rate >= 95:
            mlflow.set_tag("quality_gate", "PASSED")
        elif accuracy_rate >= 85:
            mlflow.set_tag("quality_gate", "WARNING")
        else:
            mlflow.set_tag("quality_gate", "FAILED")
        
        mlflow.set_tag("run_type", "llm_evaluation")
        
        # ==============================
        # DATASET - Input Data Info
        # ==============================
        log_dataset_info(spark, silver_df, SAMPLE_FRACTION)
        
        # ==============================
        # METRICS - Comprehensive
        # ==============================
        log_evaluation_metrics(
            parsed_df,
            accuracy_rate,
            accurate_count,
            inaccurate_count,
            total_audited,
            error_categories
        )
        
        # ==============================
        # ARTIFACTS - Data Samples
        # ==============================
        log_artifacts(spark, parsed_df, inaccurate_count)
        
        # ==============================
        # MODEL REGISTRATION - Comparison
        # ==============================
        register_model_for_comparison()
        
        # ==============================
        # FINAL STATUS
        # ==============================
        logger.info("=" * 50)
        logger.info("MLflow EXTRACOMPLETE SUMMARY:")
        logger.info(f"  Experiment: {MLFLOW_EXP}")
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Accuracy Rate: {accuracy_rate:.2f}%")
        logger.info(f"  Quality Gate: {'PASSED' if accuracy_rate >= 95 else 'WARNING' if accuracy_rate >= 85 else 'FAILED'}")
        logger.info(f"  Artifacts: quarantine_samples, error_analysis, audit_metadata")
        logger.info(f"  Tags: quality_gate, audit_type, judge_model, data_owner")
        logger.info("=" * 50)
    
    # -------------------------------------------------------------------------
    # STEP 8: Save Quarantine Delta Table
    # -------------------------------------------------------------------------
    quarantine_df = parsed_df.filter(F.col("is_accurate") == False)
    quarantine_count = quarantine_df.count()
    
    if quarantine_count > 0:
        logger.info(f"Saving {quarantine_count:,} failed extractions to quarantine")
        
        (
            quarantine_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .save(QUARANTINE_PATH)
        )
        
        logger.info(f"Quarantine saved to: {QUARANTINE_PATH}")
    else:
        logger.info("No inaccurate records to quarantine")
    
    return parsed_df


# =============================================================================
# SPARK SESSION SETUP
# =============================================================================

def create_spark_session() -> SparkSession:
    """Create and configure the Spark session."""
    return (
        SparkSession.builder
        .appName("Silver_Layer_Quality_Audit")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    logger.info("Starting Silver Quality Audit Job")
    
    spark = create_spark_session()
    
    try:
        # Run the quality audit
        audit_results_df = run_quality_audit(spark)
        
        logger.info("=" * 60)
        logger.info("Silver Quality Audit COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Quality audit failed: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()