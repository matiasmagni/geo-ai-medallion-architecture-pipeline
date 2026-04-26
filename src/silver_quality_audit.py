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
    4. Calculate Accuracy Rate and log to MLflow
    5. Save failed extractions to Quarantine Delta Table

Architecture Decision:
    Using local DeepSeek-R1 as judge model via Ollama provides:
    - Zero external API costs
    - Privacy-preserving local inference
    - Explicit reasoning via <-thinking> tags before JSON output

Author: GeoAI Principal MLOps Engineer
Version: 1.0.0
================================================================================
"""

import os
import sys
import json
import logging
import re
import requests
from datetime import datetime
from typing import Dict, Any, Optional
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

# MLflow imports
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    logger.info(f"Loaded {initial_count:,} records from Silver layer")
    
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
    
    logger.info("-" * 40)
    logger.info(f"AUDIT RESULTS:")
    logger.info(f"  Total Audited:     {total_audited:,}")
    logger.info(f"  Accurate:         {accurate_count:,}")
    logger.info(f"  Inaccurate:       {inaccurate_count:,}")
    logger.info(f"  Accuracy Rate:     {accuracy_rate:.2f}%")
    logger.info("-" * 40)
    
    # -------------------------------------------------------------------------
    # STEP 6: Log to MLflow
    # -------------------------------------------------------------------------
    logger.info(f"Logging metrics to MLflow experiment: {MLFLOW_EXP}")
    
    mlflow.set_experiment(MLFLOW_EXP)
    
    with mlflow.start_run(run_name="DeepSeek_Silver_Audit"):
        # Log parameters
        mlflow.log_param("sample_fraction", SAMPLE_FRACTION)
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("total_records", initial_count)
        mlflow.log_param("audited_records", total_audited)
        
        # Log metrics
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        mlflow.log_metric("accurate_count", accurate_count)
        mlflow.log_metric("inaccurate_count", inaccurate_count)
        mlflow.log_metric("audit_sample_size", sample_count)
        
        logger.info("MLflow metrics logged successfully")
    
    # -------------------------------------------------------------------------
    # STEP 7: Save Quarantine Delta Table
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