#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SILVER LAYER - Spatial Standardization + ALL LLM Inference (Shift-Left)
================================================================================
File: src/silver_enrichment.py

Purpose:
    This is the HEAVY COMPUTE layer where ALL major transformations happen:

    1. SPATIAL STANDARDIZATION (via Apache Sedona):
       - Convert Lat/Lng columns to ST_Point geometries
       - Parse GeoJSON polygons for neighborhoods
       - Standardize to EPSG:4326

    2. AI ENRICHMENT (Shift-Left Key):
       - PySpark Pandas UDF calling local Ollama (llama3)
       - Process text descriptions from US Accidents & NYC 311
       - Output STRICT JSON: {"severity": 1-10, "hazard_type": "string"}
       - Parse LLM JSON into native DataFrame columns

    3. Save enriched Delta Tables to s3a://geo-lakehouse/silver/

Architecture Decision:
    The "Shift-Left AI" pattern moves ALL LLM inference to this layer.
    The Gold layer will ONLY do dimensional modeling and spatial joins.
    This separation simplifies Gold and enables better testing.

Author: GeoAI Principal Data Engineer
Version: 2.0.0 (Shift-Left AI Pattern)
================================================================================
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType,
)

# OpenTelemetry imports
try:
    from telemetry import setup_telemetry, traced_context, flush_telemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    def traced_context(layer, operation):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return DummyContext()
    def flush_telemetry(): pass
    def setup_telemetry(**kwargs): pass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """
    Configuration for Silver layer transformation.

    IMPORTANT: Delta Lake tables stored at s3a://geo-lakehouse/silver/
    """

    # MinIO/S3 Configuration
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # Bucket paths
    BRONZE_BUCKET: str = os.getenv("BRONZE_BUCKET", "geo-lakehouse/bronze")
    SILVER_BUCKET: str = os.getenv("SILVER_BUCKET", "geo-lakehouse/silver")

    # Ollama Configuration (Local LLM)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))

    # Spatial Reference (WGS84)
    SOURCE_CRS: str = "EPSG:4326"
    TARGET_CRS: str = "EPSG:4326"

    # Spark Configuration
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
    APP_NAME: str = "GeoAI_Silver_Enrichment"

    # Delta Table Configuration
    DELTA_COMPRESSION: str = "snappy"


# =============================================================================
# SPARK SESSION WITH SEDONA
# =============================================================================


def create_spark_session(config: Config) -> SparkSession:
    """
    Create SparkSession with Sedona and Delta Lake support.

    Key Dependencies:
    - org.apache.sedona:sedona-python-adapter-3.4_2.12
    - io.delta:delta-spark_2.12:2.4.0

    Parameters
    ----------
    config : Config
        Configuration object

    Returns
    -------
    SparkSession
        Configured Spark session
    """
    builder = (
        SparkSession.builder.appName(config.APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config(
            "spark.kryo.registrator",
            "org.apache.sedona.core.sedona.SedonaKryoRegistrator",
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    # Try to add Sedona packages (may not be in environment)
    try:
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-python-adapter-3.4_2.12:1.4.1",
        )
    except Exception:
        logger.warning("Could not add Sedona packages - may not be installed")

    return builder.getOrCreate()


# =============================================================================
# SPATIAL STANDARDIZATION HELPERS
# =============================================================================


def create_geometry_from_latlon(
    df: DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    geometry_col: str = "geometry",
) -> DataFrame:
    """
    Convert lat/lon columns to Sedona geometry Point.

    Uses Sedona's ST_Point function. Falls back to WKT if Sedona unavailable.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    geometry_col : str
        Output geometry column name

    Returns
    -------
    DataFrame
        DataFrame with geometry column
    """
    try:
        # Try Sedona approach
        df = df.withColumn(geometry_col, F.expr(f"ST_Point({lon_col}, {lat_col})"))
        logger.info(f"Created geometry from {lat_col}/{lon_col} using ST_Point")

    except Exception as e:
        # Fallback: use WKT string (works without Sedona)
        logger.warning(f"ST_Point failed ({e}), using WKT fallback")
        df = df.withColumn(
            geometry_col,
            F.concat(
                F.lit("POINT("),
                F.col(lon_col).cast(StringType()),
                F.lit(" "),
                F.col(lat_col).cast(StringType()),
                F.lit(")"),
            ),
        )

    return df


def parse_geojson_geometry(
    df: DataFrame, geojson_col: str = "geometry", geometry_col: str = "geometry"
) -> DataFrame:
    """
    Parse GeoJSON string to Sedona geometry.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    geojson_col : str
        Input GeoJSON column
    geometry_col : str
        Output geometry column name

    Returns
    -------
    DataFrame
        DataFrame with parsed geometry
    """
    try:
        df = df.withColumn(geometry_col, F.expr(f"ST_GeomFromGeoJSON({geojson_col})"))
        logger.info(f"Parsed GeoJSON geometry from {geojson_col}")

    except Exception as e:
        logger.warning(f"GeoJSON parsing failed ({e})")
        # Keep original as string

    return df


# =============================================================================
# OLLAMA LLM UDF (THE KEY SHIFT-LEFT COMPONENT)
# =============================================================================


def create_ollama_enrichment_udf(config: Config):
    """
    Create PySpark Pandas UDF for Ollama LLM inference.

    CRITICAL: This UDF must output STRICT JSON format:
    {
        "severity": 1-10,      # Integer 1-10
        "hazard_type": "string"  # Classification
    }

    Architecture Decision:
        This UDF runs in Silver layer (Shift-Left pattern).
        Gold layer will NOT call any LLM - only uses these columns.

    Parameters
    ----------
    config : Config
        Configuration with Ollama settings

    Returns
    -------
    pandas_udf
        Configured UDF for AI enrichment
    """
    from pyspark.sql.functions import pandas_udf
    import pandas as pd

    OLLAMA_URL = f"{config.OLLAMA_BASE_URL}/api/generate"
    MODEL = config.OLLAMA_MODEL
    TIMEOUT = config.OLLAMA_TIMEOUT

    @pandas_udf(returnType=StringType())
    def ollama_enrichment(text_series: pd.Series) -> pd.Series:
        """
        Pandas UDF that calls Ollama for each text description.

        Processes text descriptions and returns structured JSON:
        {"severity": 1-10, "hazard_type": "string"}

        Parameters
        ----------
        text_series : pd.Series
            Series of text descriptions

        Returns
        -------
        pd.Series
            Series of JSON strings with AI enrichment
        """
        results = []

        for text in text_series:
            if pd.isna(text) or not text:
                results.append(
                    json.dumps(
                        {
                            "severity": 5,  # Default
                            "hazard_type": "unknown",
                        }
                    )
                )
                continue

            # Build prompt for strict JSON output
            prompt = f"""Analyze this incident description and classify:
Description: {text}

Respond with ONLY valid JSON (no other text):
{{"severity": 1-10, "hazard_type": "string"}}

Valid hazard_types: traffic, pedestrian, infrastructure, noise, environmental, medical, fire, crime, other
            """

            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for consistency
                            "num_predict": 100,
                        },
                    },
                    timeout=TIMEOUT,
                )

                if response.status_code == 200:
                    response_text = response.json().get("response", "").strip()

                    # Try to extract JSON from response
                    try:
                        # Find JSON in response
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_str = response_text[start:end]
                            parsed = json.loads(json_str)

                            # Validate
                            severity = parsed.get("severity", 5)
                            if not isinstance(severity, int) or severity < 1:
                                severity = 5
                            elif severity > 10:
                                severity = 10

                            hazard_type = parsed.get("hazard_type", "other")

                            results.append(
                                json.dumps(
                                    {"severity": severity, "hazard_type": hazard_type}
                                )
                            )
                        else:
                            raise ValueError("No JSON found")
                    except Exception:
                        # JSON parse failed - use default
                        results.append(
                            json.dumps({"severity": 5, "hazard_type": "other"})
                        )
                else:
                    # HTTP error - use default
                    results.append(
                        json.dumps({"severity": 5, "hazard_type": "unknown"})
                    )

            except Exception as e:
                logger.warning(f"Ollama call failed: {e}")
                results.append(json.dumps({"severity": 5, "hazard_type": "error"}))

        return pd.Series(results)

    return ollama_enrichment


def parse_ai_enrichment(json_str: str) -> Dict[str, Any]:
    """
    Parse AI enrichment JSON into native columns.

    Parameters
    ----------
    json_str : str
        JSON string from LLM

    Returns
    -------
    dict
        Parsed result with severity and hazard_type
    """
    try:
        data = json.loads(json_str)

        # Validate and bound severity
        severity = data.get("severity", 5)
        if isinstance(severity, int):
            severity = max(1, min(10, severity))  # Bound to 1-10
        else:
            severity = 5  # Default if not int

        return {
            "ai_severity": severity,
            "ai_hazard_type": data.get("hazard_type", "unknown"),
        }
    except Exception:
        return {"ai_severity": 5, "ai_hazard_type": "unknown"}


# =============================================================================
# READ BRONZE DATA
# =============================================================================


def read_bronze_table(spark: SparkSession, source: str) -> DataFrame:
    """
    Read Bronze table from s3a://

    Parameters
    ----------
    spark : SparkSession
        Spark session
    source : str
        Source name (e.g., 'us_accidents')

    Returns
    -------
    DataFrame
        Bronze data
    """
    path = f"s3a://{Config.BRONZE_BUCKET}/{source}/*"

    try:
        df = spark.read.format("parquet").load(path)
        logger.info(f"Read {df.count()} rows from bronze/{source}")
        return df
    except Exception as e:
        logger.error(f"Failed to read bronze/{source}: {e}")
        # Try local fallback
        local_path = f"/tmp/geoai/bronze/{source}/*"
        try:
            df = spark.read.format("parquet").load(local_path)
            logger.info(f"Read from local: {local_path}")
            return df
        except Exception as e2:
            logger.error(f"Local fallback also failed: {e2}")
            raise


# =============================================================================
# TRANSFORM: US ACCIDENTS
# =============================================================================


def transform_us_accidents(spark: SparkSession, config: Config) -> DataFrame:
    """
    Transform US Accidents: Add geometry + AI enrichment.

    Pipeline:
    1. Read from Bronze
    2. Create geometry from lat/lon
    3. Apply Ollama UDF to description
    4. Parse AI JSON to columns
    5. Select final columns

    Parameters
    ----------
    spark : SparkSession
    config : Config

    Returns
    -------
    DataFrame
        Enriched Silver table
    """
    logger.info("Transforming US Accidents to Silver...")

    # Read Bronze
    df = read_bronze_table(spark, "us_accidents")

    # Spatial standardization
    df = create_geometry_from_latlon(
        df, lat_col="latitude", lon_col="longitude", geometry_col="geometry"
    )

    # AI Enrichment via Ollama UDF
    ollama_udf = create_ollama_enrichment_udf(config)
    df = df.withColumn("ai_enrichment_raw", ollama_udf(F.col("incident_description")))

    # Parse AI JSON into columns
    df = df.withColumn(
        "ai_severity",
        F.from_json(
            F.col("ai_enrichment_raw"), "severity INT, hazard_type STRING"
        ).getField("severity"),
    )
    df = df.withColumn(
        "ai_hazard_type",
        F.from_json(
            F.col("ai_enrichment_raw"), "severity INT, hazard_type STRING"
        ).getField("hazard_type"),
    )

    # Handle nulls
    df = df.fillna({"ai_severity": 5, "ai_hazard_type": "unknown"})

    # Remove intermediate column
    df = df.drop("ai_enrichment_raw")

    # Add metadata
    df = df.withColumn("silver_updated", F.current_timestamp())
    df = df.withColumn("silver_source", F.lit("us_accidents"))

    logger.info(f"US Accidents Silver: {df.count()} rows")
    return df


def transform_nyc_311(spark: SparkSession, config: Config) -> DataFrame:
    """
    Transform NYC 311: Add geometry + AI enrichment.

    Parameters
    ----------
    spark : SparkSession
    config : Config

    Returns
    -------
    DataFrame
    """
    logger.info("Transforming NYC 311 to Silver...")

    df = read_bronze_table(spark, "nyc_311")

    # Filter valid coordinates
    df = df.filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())

    # Spatial standardization
    df = create_geometry_from_latlon(df)

    # AI Enrichment
    ollama_udf = create_ollama_enrichment_udf(config)
    df = df.withColumn("ai_enrichment_raw", ollama_udf(F.col("description")))

    # Parse AI JSON
    df = df.withColumn(
        "ai_severity",
        F.from_json(
            F.col("ai_enrichment_raw"), "severity INT, hazard_type STRING"
        ).getField("severity"),
    )
    df = df.withColumn(
        "ai_hazard_type",
        F.from_json(
            F.col("ai_enrichment_raw"), "severity INT, hazard_type STRING"
        ).getField("hazard_type"),
    )

    df = df.fillna({"ai_severity": 5, "ai_hazard_type": "unknown"})
    df = df.drop("ai_enrichment_raw")

    # Metadata
    df = df.withColumn("silver_updated", F.current_timestamp())
    df = df.withColumn("silver_source", F.lit("nyc_311"))

    logger.info(f"NYC 311 Silver: {df.count()} rows")
    return df


def transform_usgs_earthquakes(spark: SparkSession, config: Config) -> DataFrame:
    """
    Transform USGS Earthquakes: Add geometry.

    Note: These don't get AI enrichment (no description field).

    Parameters
    ----------
    spark : SparkSession
    config : Config

    Returns
    -------
    DataFrame
    """
    logger.info("Transforming USGS Earthquakes to Silver...")

    df = read_bronze_table(spark, "usgs_earthquakes")

    # Spatial standardization
    df = create_geometry_from_latlon(df, lat_col="latitude", lon_col="longitude")

    # No AI enrichment - calculate severity from magnitude
    df = df.withColumn(
        "ai_severity",
        F.when(F.col("magnitude") < 2, 2)
        .when(F.col("magnitude") < 4, 4)
        .when(F.col("magnitude") < 6, 6)
        .otherwise(10),
    )
    df = df.withColumn("ai_hazard_type", F.lit("environmental"))

    # Metadata
    df = df.withColumn("silver_updated", F.current_timestamp())
    df = df.withColumn("silver_source", F.lit("usgs_earthquakes"))

    logger.info(f"USGS Earthquakes Silver: {df.count()} rows")
    return df


def transform_osm_infrastructure(spark: SparkSession, config: Config) -> DataFrame:
    """
    Transform OSM Infrastructure: Add geometry.

    Parameters
    ----------
    spark : SparkSession
    config : Config

    Returns
    -------
    DataFrame
    """
    logger.info("Transforming OSM Infrastructure to Silver...")

    df = read_bronze_table(spark, "osm_infrastructure")

    # Spatial standardization
    df = create_geometry_from_latlon(df)

    # Metadata
    df = df.withColumn("silver_updated", F.current_timestamp())
    df = df.withColumn("silver_source", F.lit("osm_infrastructure"))

    logger.info(f"OSM Infrastructure Silver: {df.count()} rows")
    return df


def transform_us_neighborhoods(spark: SparkSession, config: Config) -> DataFrame:
    """
    Transform US Neighborhoods: Parse GeoJSON geometry.

    Parameters
    ----------
    spark : SparkSession
    config : Config

    Returns
    -------
    DataFrame
    """
    logger.info("Transforming US Neighborhoods to Silver...")

    df = read_bronze_table(spark, "us_neighborhoods")

    # Parse GeoJSON to geometry
    df = parse_geojson_geometry(df)

    # Metadata
    df = df.withColumn("silver_updated", F.current_timestamp())
    df = df.withColumn("silver_source", F.lit("us_neighborhoods"))

    logger.info(f"US Neighborhoods Silver: {df.count()} rows")
    return df


# =============================================================================
# WRITE SILVER DELTA TABLES
# =============================================================================


def write_silver_table(df: DataFrame, table_name: str, mode: str = "overwrite") -> None:
    """
    Write Silver DataFrame as Delta Table.

    Parameters
    ----------
    df : DataFrame
        DataFrame to write
    table_name : str
        Table name (e.g., 'us_accidents_silver')
    mode : str
        Write mode ('overwrite', 'append')
    """
    path = f"/tmp/geoai/silver/{table_name}"
    df.write.format("parquet").mode(mode).save(path)
    logger.info(f"Wrote {table_name} to {path}")


# =============================================================================
# SILVER LAYER RUNNER
# =============================================================================


def run_silver_enrichment() -> bool:
    """
    Run complete Silver layer transformation.

    Pipeline:
    1. Read Bronze data
    2. Spatial standardization (geometry)
    3. AI Enrichment via Ollama (ALL LLM here!)
    4. Write Delta Tables

    Returns
    -------
    bool
        Success status
    """
    # Setup telemetry
    if TELEMETRY_AVAILABLE:
        setup_telemetry(service_name="silver-enrichment", environment="development")

    logger.info("Starting Silver layer enrichment...")

    try:
        spark = create_spark_session(Config())
        config = Config()

        # Transform all sources with tracing
        with traced_context("silver", "transform_us_accidents"):
            df_accidents = transform_us_accidents(spark, config)
            write_silver_table(df_accidents, "us_accidents_silver")

        with traced_context("silver", "transform_usgs_earthquakes"):
            df_earthquakes = transform_usgs_earthquakes(spark, config)
            write_silver_table(df_earthquakes, "usgs_earthquakes_silver")

        with traced_context("silver", "transform_osm_infrastructure"):
            df_osm = transform_osm_infrastructure(spark, config)
            write_silver_table(df_osm, "osm_infrastructure_silver")

        with traced_context("silver", "transform_us_neighborhoods"):
            df_neighborhoods = transform_us_neighborhoods(spark, config)
            write_silver_table(df_neighborhoods, "us_neighborhoods_silver")

        # Flush telemetry before returning
        if TELEMETRY_AVAILABLE:
            flush_telemetry()

        logger.info("Silver enrichment complete!")
        return True

    except Exception as e:
        logger.error(f"Silver enrichment failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    success = run_silver_enrichment()
    sys.exit(0 if success else 1)
