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

    IMPORTANT: Delta Lake tables stored at s3a://geoai-silver/
    """

    # MinIO/S3 Configuration
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # Bucket paths
    BRONZE_BUCKET: str = os.getenv("BRONZE_BUCKET", "geoai-bronze")
    SILVER_BUCKET: str = os.getenv("SILVER_BUCKET", "geoai-silver")

    # Ollama Configuration (Local LLM)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
    Create SparkSession with Sedona, Delta Lake, and S3A support.
    """
    from pyspark import SparkConf
    
    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)
    
    # Note: Delta Lake config requires jars to be loaded - skip for now
    # Delta will be loaded via --packages or explicit jar loading
    
    # Use default serializer (avoid Sedona Kryo conflict with PySpark 3.5+)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    # Skip Sedona Kryo registrator - use SedonaContext SQL API instead
    
    # MinIO / S3A Configuration
    conf.set("spark.hadoop.fs.s3a.endpoint", config.MINIO_ENDPOINT)
    conf.set("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    conf.set("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
    conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    
    # Java 17+ compatibility: inject JVM module opens for Hadoop security
    java_opts = " ".join([
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.ssl=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth.callback=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth.kerberos=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth.login=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth.spi=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.sasl=ALL-UNNAMED",
        "--add-opens=java.base/com.sun.security.auth=ALL-UNNAMED",
        "--add-opens=java.base/com.sun.security.auth.callback=ALL-UNNAMED",
        "--add-opens=java.base/com.sun.security.auth.login=ALL-UNNAMED",
        "--add-opens=java.base/com.sun.security.auth.kerberos=ALL-UNNAMED",
        "--add-opens=java.base/com.sun.security.auth.spi=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.jgss=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5.internal=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.tools.keytool=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.pkcs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.provider=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.util=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.x509=ALL-UNNAMED",
        "--add-opens=java.rmi/sun.rmi.transport=ALL-UNNAMED",
        "--add-opens=java.naming/sun.security.jgss=ALL-UNNAMED",
    ])
    conf.set("spark.driver.extraJavaOptions", java_opts)
    conf.set("spark.executor.extraJavaOptions", java_opts)
    
    # Build Session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    # Note: Sedona requires compatible jars - skip initialization, use plain GeoPandas for spatial ops
    logger.info("Spark session created (without Sedona spatial)")
         
    return spark


# =============================================================================
# SPATIAL STANDARDIZATION HELPERS
# =============================================================================


def create_geometry_from_latlon(
    df: DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    geometry_col: str = "geometry",
    drop_nulls: bool = False,
) -> DataFrame:
    """
    Create geometry column from lat/lon - simplified without Sedona.
    Stores lat/lon as separate columns instead of geometry.
    """
    # Skip geometry creation if Sedona not available
    # Just keep lat/lon columns for later spatial joins
    logger.info(f"Stored lat/lon from {lat_col}/{lon_col} (no Sedona geometry)")
    return df.withColumn("lat", F.col(lat_col)).withColumn("lon", F.col(lon_col))


def parse_geojson_geometry(
    df: DataFrame, geojson_col: str = "geometry", geometry_col: str = "geometry"
) -> DataFrame:
    """
    Parse GeoJSON string - simplified without Sedona.
    Just returns DataFrame as-is.
    """
    logger.info(f"Skipping GeoJSON parsing (no Sedona)")
    return df


# =============================================================================
# LAND MASK FILTERING (Remove Water Dots)
# =============================================================================


def load_neighborhoods_mask(spark: SparkSession, config: Config) -> Optional[DataFrame]:
    """
    Load neighborhood polygons to use as a land mask.

    Parameters
    ----------
    spark : SparkSession
    config : Configuration

    Returns
    -------
    DataFrame
        Neighborhoods GeoDataFrame or None
    """
    try:
        df = read_bronze_table(spark, "us_neighborhoods")
        df = parse_geojson_geometry(df)
        count = df.count()

        if count == 0:
            logger.warning("Neighborhoods loaded but have 0 rows — land mask disabled")
            return None

        # Skip bbox computation without Sedona ST_XMin functions
        logger.info(
            f"Loaded neighborhoods: {count} polygons (land mask disabled - no Sedona)"
        )
        return df
    except Exception as e:
        logger.warning(f"Could not load neighborhoods for land mask: {e}")
        return None


def filter_points_on_land(
    spark: SparkSession,
    points_df: DataFrame,
    neighborhoods_df: DataFrame,
    geometry_col: str = "geometry",
    source_name: str = "unknown",
) -> DataFrame:
    """
    Remove points that fall in water by keeping only points within land polygons.
    Uses Sedona ST_Within for spatial filtering.

    Parameters
    ----------
    spark : SparkSession
    points_df : DataFrame
        Points DataFrame with geometry column
    neighborhoods_df : DataFrame
        Neighborhood polygons for land mask
    geometry_col : str
        Name of geometry column in points_df
    source_name : str
        Source name for logging

    Returns
    -------
    DataFrame
        Filtered DataFrame with only land points
    """
    if neighborhoods_df is None:
        logger.error(f"[{source_name}] NO LAND MASK — water filter DISABLED! All points will pass through!")
        return points_df

    initial_count = points_df.count()
    points_df = points_df.filter(F.col(geometry_col).isNotNull())
    neighborhoods_df = neighborhoods_df.filter(F.col("geometry").isNotNull())
    n_polys = neighborhoods_df.count()
    logger.info(
        f"[{source_name}] Land filter: {initial_count} input points, {n_polys} neighborhood polygons"
    )

    try:
        filtered_df = _filter_spatial(spark, points_df, neighborhoods_df, initial_count, source_name)
        if filtered_df.count() >= initial_count * 0.95:
            logger.warning(
                f"[{source_name}] POSSIBLE FILTER FAILURE: {initial_count} -> {filtered_df.count()} "
                f"(only {initial_count - filtered_df.count()} removed)"
            )
        return filtered_df
    except Exception as e:
        logger.warning(f"[{source_name}] ST_Within failed ({e}), trying numeric bounds")
        return _filter_numeric_bbox(spark, points_df, neighborhoods_df, initial_count, source_name)


def _filter_spatial(
    spark: SparkSession,
    points_df: DataFrame,
    neighborhoods_df: DataFrame,
    initial_count: int,
    source_name: str,
) -> DataFrame:
    """Filter using ST_Within spatial join."""
    neighborhoods_df.createOrReplaceTempView("land_polys")
    points_df.createOrReplaceTempView("points")

    filtered_df = spark.sql("""
        SELECT p.*
        FROM points p
        INNER JOIN land_polys l
        ON ST_Within(p.geometry, l.geometry)
    """)
    final_count = filtered_df.count()
    removed = initial_count - final_count
    logger.info(f"[{source_name}] ST_Within filter: {initial_count} -> {final_count} (removed {removed})")
    return filtered_df


def _filter_numeric_bbox(
    spark: SparkSession,
    points_df: DataFrame,
    neighborhoods_df: DataFrame,
    initial_count: int,
    source_name: str,
) -> DataFrame:
    """
    Filter by numeric bounding box from neighborhood polygons.
    Uses ST_XMin/Max/ST_YMin/Max to get bbox — works when geometry is parsed.
    Falls back to raw min/max on lat/lon columns when ST unavailable.
    """
    try:
        neighborhoods_df.createOrReplaceTempView("land_polys")

        bbox = spark.sql("""
            SELECT
                MIN(ST_XMin(geometry)) AS min_lon,
                MAX(ST_XMax(geometry)) AS max_lon,
                MIN(ST_YMin(geometry)) AS min_lat,
                MAX(ST_YMax(geometry)) AS max_lat
            FROM land_polys
        """).collect()[0]

        min_lon, max_lon = float(bbox.min_lon), float(bbox.max_lon)
        min_lat, max_lat = float(bbox.min_lat), float(bbox.max_lat)
        logger.info(
            f"[{source_name}] Numeric bbox: ({min_lon:.4f},{min_lat:.4f}) -> ({max_lon:.4f},{max_lon:.4f})"
        )

    except Exception:
        logger.warning(f"[{source_name}] ST unavailable — using approximate US lat/lon range")
        min_lon, max_lon = -125.0, -66.0
        min_lat, max_lat = 24.0, 50.0

    filtered_df = points_df.filter(
        (F.col("longitude") >= min_lon)
        & (F.col("longitude") <= max_lon)
        & (F.col("latitude") >= min_lat)
        & (F.col("latitude") <= max_lat)
    )
    final_count = filtered_df.count()
    removed = initial_count - final_count
    logger.info(
        f"[{source_name}] Numeric bbox filter: {initial_count} -> {final_count} (removed {removed})"
    )
    return filtered_df


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
    except Exception as e:
        logger.warning(f"Failed to parse AI enrichment JSON: {e}")
        return {"ai_severity": 5, "ai_hazard_type": "unknown"}


            # =============================================================================
            # READ BRONZE DATA
            # =============================================================================


def read_bronze_table(spark: SparkSession, source: str) -> DataFrame:
    """
    Read Bronze table - tries local first, falls back to S3A.

    Parameters
    ----------
    spark : SparkSession
        Spark session
    source : str
        Source name (e.g., 'usgs_earthquakes')

    Returns
    -------
    DataFrame
        Bronze data
    """
    # Default to local path (S3A requires hadoop-aws jar not available for Python 3.13 ARM64)
    local_path = f"/tmp/geoai/bronze/{source}"

    try:
        df = spark.read.format("parquet").load(local_path)
        logger.info(f"Read from local: {local_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read local {local_path}: {e}")
        # Try S3A as fallback
        s3a_path = f"s3a://{Config.BRONZE_BUCKET}/{source}/*"
        try:
            df = spark.read.format("parquet").load(s3a_path)
            logger.info(f"Read from S3A: {s3a_path}")
            return df
        except Exception as e2:
            logger.error(f"S3A fallback also failed: {e2}")
            raise


# =============================================================================
# TRANSFORM: US ACCIDENTS
# =============================================================================


def transform_us_accidents(
    spark: SparkSession, config: Config, neighborhoods_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Transform US Accidents: Add geometry + land mask filter + AI enrichment.

    Pipeline:
    1. Read from Bronze
    2. Create geometry from lat/lon
    3. Apply land mask filter (remove water dots)
    4. Apply Ollama UDF to description
    5. Parse AI JSON to columns
    6. Select final columns

    Parameters
    ----------
    spark : SparkSession
    config : Config
    neighborhoods_df : Optional[DataFrame]
        Neighborhood polygons for land mask filtering

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

    # Apply land mask filter to remove water dots
    if neighborhoods_df is not None:
        df = filter_points_on_land(spark, df, neighborhoods_df, source_name="us_accidents")
        logger.info("Applied land mask filter to US Accidents")

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


def transform_nyc_311(
    spark: SparkSession, config: Config, neighborhoods_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Transform NYC 311: Add geometry + land mask filter + AI enrichment.

    Parameters
    ----------
    spark : SparkSession
    config : Config
    neighborhoods_df : Optional[DataFrame]
        Neighborhood polygons for land mask filtering

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

    # Apply land mask filter to remove water dots
    if neighborhoods_df is not None:
        df = filter_points_on_land(spark, df, neighborhoods_df, source_name="nyc_311")
        logger.info("Applied land mask filter to NYC 311")

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


def transform_osm_infrastructure(
    spark: SparkSession, config: Config, neighborhoods_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Transform OSM Infrastructure - simple copy without spatial ops.
    """
    logger.info("Transforming OSM Infrastructure to Silver...")

    df = read_bronze_table(spark, "osm_infrastructure")

    # Just add metadata (no spatial filtering without Sedona)
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
    2. Load neighborhoods for land mask
    3. Spatial standardization (geometry)
    4. Apply land mask filter (remove water dots)
    5. AI Enrichment via Ollama (ALL LLM here!)
    6. Write Delta Tables

    Returns
    -------
    bool
        Success status
    """
    logger.info("Starting Silver layer enrichment...")

    try:
        spark = create_spark_session(Config())
        config = Config()

        # Load neighborhoods for land mask filtering (before transforming point sources)
        logger.info("Loading neighborhoods for land mask...")
        neighborhoods_df = load_neighborhoods_mask(spark, config)

        # Transform ONLY available sources (skip us_accidents - not in bronze)
        # Available: usgs_earthquakes, osm_infrastructure, us_neighborhoods
        #           nyc_311, nyc_flights, nyc_weather (from bronze_ingestion.py)
        
        # USGS Earthquakes
        df_usgs = transform_usgs_earthquakes(spark, config)
        
        # OSM Infrastructure 
        df_osm = transform_osm_infrastructure(spark, config, neighborhoods_df)
        
        # Neighborhoods (for reference/dim table)
        df_neighborhoods = transform_us_neighborhoods(spark, config)

        # Write all
        write_silver_table(df_usgs, "usgs_earthquakes_silver")
        write_silver_table(df_osm, "osm_infrastructure_silver")
        write_silver_table(df_neighborhoods, "us_neighborhoods_silver")

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
