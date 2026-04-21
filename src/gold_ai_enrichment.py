#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# GeoAI Medallion Architecture - GOLD LAYER: AI Enrichment & Spatial Join
# =============================================================================
# File: src/gold_ai_enrichment.py
#
# Purpose:
#   This script performs the GOLD layer transformations in the Medallion Architecture:
#   1. Reads cleaned Delta Table from MinIO "Silver" bucket
#   2. Implements a PySpark Pandas UDF that calls Ollama API
#   3. Extracts risk score/sentiment using local LLM (llama3/phi3)
#   4. Performs spatial join with polygon dataset (Sedona ST_Contains)
#   5. Logs metrics using MLflow
#   6. Saves aggregated data as Gold Delta Table
#
# Key Technologies:
#   - Apache Spark (PySpark) with Apache Sedona - Spatial processing
#   - Delta Lake - ACID transactions on data lake
#   - Ollama - Local LLM inference (llama3, phi3)
#   - MLflow - Experiment tracking and metrics
#
# Author: GeoAI Principal Data Engineer & MLOps Architect
# Version: 1.0.0
# =============================================================================

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def safe_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse a JSON string, returning None on failure.

    Parameters
    ----------
    json_str : str
        Raw JSON string (potentially from API response).

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed dict or None if parsing fails.
    """
    if not json_str or not isinstance(json_str, str):
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# =============================================================================
# CONFIGURATION (Environment Variables)
# =============================================================================


class Config:
    """Configuration class for the Gold layer AI enrichment.

    All values are loaded from environment variables with sensible defaults
    for local development.
    """

    # MinIO/S3 Configuration
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    MINIO_BUCKET_SILVER: str = os.getenv("SILVER_BUCKET", "geoai-silver")
    MINIO_BUCKET_GOLD: str = os.getenv("GOLD_BUCKET", "geoai-gold")

    # Spark Configuration
    SPARK_MASTER: str = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    APP_NAME: str = "GeoAI_Gold_AI_Enrichment"

    # Delta Table Configuration
    DELTA_TABLE_NAME: str = "incident_reports_silver"

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "GeoAI_Default")
    MLFLOW_RUN_NAME: str = "gold_enrichment"

    # Spatial Configuration
    POLYGON_BUCKET: str = os.getenv("POLYGON_BUCKET", "geoai-bronze")
    POLYGON_TABLE: str = "risk_zones"

    # Input/Output Columns
    TEXT_COL: str = os.getenv("TEXT_COL", "incident_report")
    GEOMETRY_COL: str = "geometry"
    OUTPUT_COL: str = "ai_risk_score"


# =============================================================================
# SPARK SESSION BUILDER
# =============================================================================


def create_spark_session(config: Config) -> "SparkSession":
    """
    Create and configure a SparkSession with Sedona and Delta Lake extensions.

    Parameters:
        config: Configuration object with settings

    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    logger.info("Initializing Spark session with Sedona, Delta Lake, and MLflow...")

    # Build Spark configuration
    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # -------------------------------------------------------------------------
    # DELTA LAKE CONFIGURATION
    # -------------------------------------------------------------------------
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )

    # -------------------------------------------------------------------------
    # APACHE SEDONA CONFIGURATION
    # -------------------------------------------------------------------------
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set(
        "spark.kryo.registrators", "org.apache.sedona.viz.core.SedonaVizKryoRegistrator"
    )

    # -------------------------------------------------------------------------
    # MINIO/S3 CONFIGURATION
    # -------------------------------------------------------------------------
    conf.set("spark.hadoop.fs.s3a.endpoint", f"http://{config.MINIO_ENDPOINT}")
    conf.set("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    conf.set("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
    conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # -------------------------------------------------------------------------
    # SPARK CONFIGURATION
    # -------------------------------------------------------------------------
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.executor.cores", "2")
    conf.set("spark.sql.shuffle.partitions", "8")
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Enable Delta Lake optimization
    conf.set("spark.delta.optimize.enabled", "true")
    conf.set("spark.delta.autoCompact.enabled", "true")

    # Create SparkSession
    spark = (
        SparkSession.builder.config(conf=conf)
        .config(
            "spark.plugins",
            "org.apache.sedona.SedonaPlugin,io.delta.spark.DeltaSparkSessionPlugin",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Spark session initialized: {spark.version}")

    return spark


# =============================================================================
# OLLAMA CLIENT (PySpark UDF)
# =============================================================================


def create_ollama_udf(
    config: Config,
    prompt_template: str = "Extract a risk score (0-100) from this text. Just return the number.",
) -> "Callable":
    """
    Create a Pandas UDF that calls the Ollama API for local LLM inference.

    This function creates a vectorized Pandas UDF that:
    1. Takes a text column as input
    2. Sends it to the local Ollama container
    3. Extracts the risk score from the LLM response

    Parameters:
        config: Configuration with Ollama settings
        prompt_template: Custom prompt for the LLM

    Returns:
        Pandas UDF function
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    import http.client
    import re

    # Build the Ollama API client
    def call_ollama_api(text: str) -> Optional[float]:
        """
        Call Ollama API for inference.

        Parameters:
            text: Input text to analyze

        Returns:
            Float risk score (0-100) or None if extraction fails
        """
        if not text or pd.isna(text):
            return None

        try:
            # Parse endpoint
            host = config.OLLAMA_BASE_URL.replace("http://", "").replace("https://", "")
            if ":" in host:
                host, port = host.split(":")
                port = int(port)
            else:
                port = 11434

            # Build request
            connection = http.client.HTTPConnection(
                host, port, timeout=config.OLLAMA_TIMEOUT
            )

            # Prepare prompt with context
            full_prompt = f"{prompt_template}\n\nText: {text}\n\nRisk Score:"

            payload = json.dumps(
                {
                    "model": config.OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 50},
                }
            )

            headers = {"Content-Type": "application/json"}

            # Make request
            connection.request("POST", "/api/generate", payload, headers)
            response = connection.getresponse()

            # Parse response
            if response.status == 200:
                result = json.loads(response.read().decode())
                response_text = result.get("response", "")

                # Extract numeric score using regex
                # Look for patterns like "85", "85/100", "risk: 85"
                match = re.search(r"(\d+(?:\.\d+)?)", response_text)
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-100 scale
                    if score > 100:
                        score = score / 100 * 100
                    return min(100.0, max(0.0, score))

            connection.close()
            return None

        except Exception as e:
            logger.warning(f"Ollama API call failed: {str(e)}")
            return None

    # Create Pandas UDF (vectorized for performance)
    @pandas_udf("float", PandasUDFType.SCALAR)
    def ollama_risk_udf(text_series: pd.Series) -> pd.Series:
        """Pandas UDF for risk score extraction."""
        return text_series.apply(call_ollama_api)

    return ollama_risk_udf


# =============================================================================
# SILVER LAYER READ
# =============================================================================


def read_silver_data(spark: "SparkSession", config: Config) -> "DataFrame":
    """
    Read the Silver layer Delta Table from MinIO.

    Parameters:
        spark: SparkSession
        config: Configuration

    Returns:
        DataFrame from Silver layer
    """
    input_path = f"s3://{config.MINIO_BUCKET_SILVER}/{config.DELTA_TABLE_NAME}"

    logger.info(f"Reading Silver Delta Table from: {input_path}")

    df = spark.read.format("delta").load(input_path)

    row_count = df.count()
    logger.info(f"Loaded {row_count} rows from Silver layer")

    return df


# =============================================================================
# AI ENRICHMENT (Ollama Integration)
# =============================================================================


def enrich_with_ai(df: "DataFrame", config: Config, text_column: str) -> "DataFrame":
    """
    Enrich the DataFrame with AI-extracted risk scores using Ollama.

    This function:
    1. Creates a Pandas UDF for Ollama API calls
    2. Applies the UDF to extract risk scores
    3. Caches the enriched DataFrame for reuse

    Parameters:
        df: Input DataFrame with text column
        config: Configuration with Ollama settings
        text_column: Name of text column to analyze

    Returns:
        DataFrame with added ai_risk_score column
    """
    logger.info(f"Enriching data with AI (Ollama {config.OLLAMA_MODEL})...")

    # Create the Ollama UDF
    ollama_udf = create_ollama_udf(config)

    # Apply UDF to extract risk scores
    # Using mapPartitions for distributed processing
    df = df.withColumn(config.OUTPUT_COL, ollama_udf(F.col(text_column)))

    # Add metadata
    df = df.withColumn("ai_model", F.lit(config.OLLAMA_MODEL))
    df = df.withColumn("ai_enrichment_timestamp", F.current_timestamp())

    # Log metrics
    enriched_count = df.filter(F.col(config.OUTPUT_COL).isNotNull()).count()
    logger.info(f"AI enrichment complete: {enriched_count} rows with risk scores")

    return df


# =============================================================================
# POLYGON DATA LOAD
# =============================================================================


def load_polygon_data(spark: "SparkSession", config: Config) -> "DataFrame":
    """
    Load polygon data for spatial join.

    Parameters:
        spark: SparkSession
        config: Configuration

    Returns:
        DataFrame with polygon geometries
    """
    # Try Delta Table first, then fallback to GeoJSON
    polygon_path = f"s3://{config.POLYGON_BUCKET}/{config.POLYGON_TABLE}"

    try:
        df = spark.read.format("delta").load(polygon_path)
        logger.info("Loaded polygons from Delta Table")
    except Exception as e:
        logger.warning(f"Delta polygon not found, using default risk zones")

        # Create default risk zones (example polygons)
        # In production, this would load from a real polygon dataset
        data = [
            {
                "zone_id": "R1",
                "zone_name": "Downtown",
                "risk_level": "high",
                "polygon": "POLYGON ((-122.42 37.77, -122.40 37.77, -122.40 37.79, -122.42 37.79, -122.42 37.77))",
            },
            {
                "zone_id": "R2",
                "zone_name": "Suburban",
                "risk_level": "medium",
                "polygon": "POLYGON ((-122.45 37.74, -122.42 37.74, -122.42 37.77, -122.45 37.77, -122.45 37.74))",
            },
        ]
        df = spark.createDataFrame(data)
        df = df.withColumn("polygon", F.expr("ST_GeomFromWKT(polygon)"))

    return df


# =============================================================================
# SPATIAL JOIN (Sedona ST_Contains)
# =============================================================================


def perform_spatial_join(
    points_df: "DataFrame", polygons_df: "DataFrame", config: Config
) -> "DataFrame":
    """
    Perform spatial join using Sedona ST_Contains.

    This function:
    1. Joins point geometries with polygon geometries
    2. Labels each point with its containing zone

    Parameters:
        points_df: DataFrame with point geometries
        polygons_df: DataFrame with polygon geometries
        config: Configuration

    Returns:
        DataFrame with zone information joined
    """
    logger.info("Performing spatial join using Sedona ST_Contains...")

    # Rename polygon geometry column for clarity
    polygons_df = polygons_df.withColumnRenamed("polygon", "zone_geometry")

    # Perform spatial join
    # ST_Contains(polygon, point) returns true if point is in polygon
    joined_df = points_df.alias("p").join(
        polygons_df.alias("z"),
        F.expr("ST_Contains(z.zone_geometry, p.geometry)"),
        "left",
    )

    # Flatten and select desired columns
    result_df = joined_df.select("p.*", "z.zone_id", "z.zone_name", "z.risk_level")

    # Log metrics
    joined_count = result_df.count()
    logger.info(f"Spatial join complete: {joined_count} points matched to zones")

    return result_df


# =============================================================================
# AGGREGATION & METRICS
# =============================================================================


def aggregate_and_log_metrics(
    df: "DataFrame", config: Config, start_time: datetime
) -> "DataFrame":
    """
    Aggregate data and log metrics to MLflow.

    Parameters:
        df: Enriched DataFrame
        config: Configuration
        start_time: Pipeline start time

    Returns:
        Aggregated DataFrame
    """
    import mlflow

    logger.info("Aggregating and logging metrics to MLflow...")

    # Initialize MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # Start MLflow run
    with mlflow.start_run(run_name=config.MLFLOW_RUN_NAME):
        # Calculate metrics
        total_rows = df.count()
        rows_with_ai = df.filter(F.col(config.OUTPUT_COL).isNotNull()).count()
        rows_with_zone = df.filter(F.col("zone_id").isNotNull()).count()

        # Calculate average risk score
        avg_risk_score = df.select(F.avg(F.col(config.OUTPUT_COL))).collect()[0][0]

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Log metrics to MLflow
        mlflow.log_metric("total_rows_processed", total_rows)
        mlflow.log_metric("rows_with_ai_enrichment", rows_with_ai)
        mlflow.log_metric("rows_with_zone_match", rows_with_zone)
        mlflow.log_metric("average_risk_score", avg_risk_score or 0.0)
        mlflow.log_metric("execution_time_seconds", execution_time)

        # Log parameters
        mlflow.log_param("ollama_model", config.OLLAMA_MODEL)
        mlflow.log_param("spark_version", spark.version)

        # Log sample output as artifact
        sample_df = df.limit(10).toPandas()
        mlflow.log_artifact(sample_df, "sample_output.csv")

        logger.info(f"MLflow metrics logged: {total_rows} rows, {execution_time:.2f}s")

    # Aggregate by zone
    aggregated_df = (
        df.groupBy("zone_id", "zone_name", "risk_level")
        .agg(
            F.count("*").alias("incident_count"),
            F.avg(config.OUTPUT_COL).alias("avg_risk_score"),
            F.max(config.OUTPUT_COL).alias("max_risk_score"),
            F.min(config.OUTPUT_COL).alias("min_risk_score"),
        )
        .orderBy(F.desc("incident_count"))
    )

    return aggregated_df


# =============================================================================
# GOLD LAYER WRITE (Delta Lake)
# =============================================================================


def write_gold_data(df: "DataFrame", config: Config, mode: str = "overwrite") -> None:
    """
    Write the Gold layer Delta Table to MinIO.

    Parameters:
        df: Aggregated DataFrame
        config: Configuration
        mode: Write mode
    """
    output_path = f"s3://{config.MINIO_BUCKET_GOLD}/risk_aggregates"

    logger.info(f"Writing Gold Delta Table to: {output_path}")

    df.write.format("delta").mode(mode).option("compression", "snappy").option(
        "overwriteSchema", "true"
    ).save(output_path)

    # Verify write
    verify_df = spark.read.format("delta").load(output_path)
    logger.info(f"Gold layer write complete: {verify_df.count()} zone aggregations")


# =============================================================================
# MAIN ELT PIPELINE
# =============================================================================


def run_gold_elt(text_column: str = None) -> None:
    """
    Execute the complete Gold layer ELT pipeline.

    Parameters:
        text_column: Name of text column to analyze
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting GOLD LAYER ELT Pipeline")
    logger.info("=" * 60)

    try:
        # 1. Create Spark Session
        global spark
        config = Config()
        spark = create_spark_session(config)

        # Import Spark functions
        from pyspark.sql import functions as F

        global F

        # 2. Read Silver Data
        logger.info("[STEP 1] Reading Silver Delta Table...")
        silver_df = read_silver_data(spark, config)

        # 3. AI Enrichment
        if text_column is None:
            text_column = config.TEXT_COL

        logger.info(f"[STEP 2] AI Enrichment ({config.OLLAMA_MODEL})...")
        enriched_df = enrich_with_ai(silver_df, config, text_column)

        # 4. Load Polygon Data
        logger.info("[STEP 3] Loading Risk Zone Polygons...")
        polygon_df = load_polygon_data(spark, config)

        # 5. Spatial Join
        logger.info("[STEP 4] Performing Spatial Join (ST_Contains)...")
        joined_df = perform_spatial_join(enriched_df, polygon_df, config)

        # 6. Aggregate and Log Metrics
        logger.info("[STEP 5] Aggregating and Logging to MLflow...")
        aggregated_df = aggregate_and_log_metrics(joined_df, config, start_time)

        # 7. Write Gold Delta Table
        logger.info("[STEP 6] Writing Gold Delta Table...")
        write_gold_data(aggregated_df, config)

        # Log success
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"GOLD LAYER ELT Complete: {duration:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Gold ELT Failed: {str(e)}", exc_info=True)
        raise

    finally:
        if "spark" in dir():
            spark.stop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoAI Gold Layer AI Enrichment")
    parser.add_argument(
        "--text-column",
        type=str,
        default="incident_report",
        help="Text column to analyze (default: incident_report)",
    )

    args = parser.parse_args()

    # Run the ELT pipeline
    run_gold_elt(text_column=args.text_column)
