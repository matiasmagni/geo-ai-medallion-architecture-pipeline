#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# GeoAI Medallion Architecture - SILVER LAYER: Spatial Transform
# =============================================================================
# File: src/silver_spatial_transform.py
#
# Purpose:
#   This script performs the SILVER layer transformations in the Medallion Architecture:
#   1. Reads raw GeoJSON/CSV data from MinIO "Bronze" bucket
#   2. Converts latitude/longitude columns to Geometry using Apache Sedona
#   3. Standardizes CRS to EPSG:4326 (WGS84)
#   4. Writes cleaned spatial data to MinIO as a Delta Table in the "Silver" layer
#
# Key Technologies:
#   - Apache Spark (PySpark) - Distributed compute engine
#   - Apache Sedona - Spatial SQL on Spark
#   - Delta Lake - ACID transactions on data lake
#
# Author: GeoAI Principal Data Engineer
# Version: 1.0.0
# =============================================================================

import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (Environment Variables)
# =============================================================================


class Config:
    """Configuration class for the Silver layer transformation.

    All values are loaded from environment variables with sensible defaults
    for local development.
    """

    # MinIO/S3 Configuration
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    MINIO_BUCKET_BRONZE: str = os.getenv("BRONZE_BUCKET", "geoai-bronze")
    MINIO_BUCKET_SILVER: str = os.getenv("SILVER_BUCKET", "geoai-silver")

    # Spark Configuration
    SPARK_MASTER: str = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    APP_NAME: str = "GeoAI_Silver_Transform"

    # Delta Lake Configuration
    DELTA_TABLE_NAME: str = "incident_reports_silver"
    DELTA_COMPRESSION: str = "snappy"

    # Spatial Reference
    SOURCE_CRS: str = "EPSG:4326"  # WGS84
    TARGET_CRS: str = "EPSG:4326"  # Standardize to WGS84

    # Input Schema
    LAT_COL: str = os.getenv("LAT_COL", "latitude")
    LON_COL: str = os.getenv("LON_COL", "longitude")
    GEOMETRY_COL: str = "geometry"

    # Logging
    CHECKPOINT_DIR: str = os.getenv(
        "CHECKPOINT_DIR", "s3://geoai-checkpoints/silver-checkpoints"
    )


# =============================================================================
# SPARK SESSION BUILDER
# =============================================================================


def create_spark_session(config: Config) -> "SparkSession":
    """
    Create and configure a SparkSession with Sedona and Delta Lake extensions.

    This function:
    1. Initializes Spark with Delta Lake support
    2. Configures Apache Sedona for spatial operations
    3. Sets up MinIO/S3 connectivity

    Parameters:
        config: Configuration object with settings

    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    logger.info("Initializing Spark session with Sedona and Delta Lake...")

    # Build Spark configuration
    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # =========================================================================
    # DELTA LAKE CONFIGURATION
    # =========================================================================
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )

    # =========================================================================
    # APACHE SEDONA CONFIGURATION
    # =========================================================================
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set(
        "spark.kryo.registrator", "org.apache.sedona.viz.core.SedonaVizKryoRegistrator"
    )

    # Enable Sedona SQL functions
    conf.set(
        "spark.sql.functions.sedona", "org.apache.sedona.sql.functions.SedonaFunctions"
    )

    # =========================================================================
    # MINIO/S3 CONFIGURATION
    # =========================================================================
    conf.set("spark.hadoop.fs.s3a.endpoint", f"http://{config.MINIO_ENDPOINT}")
    conf.set("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    conf.set("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
    conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    conf.set(
        "spark.hadoop.compression.codecs", "org.apache.hadoop.io.compress.GzipCodec"
    )

    # =========================================================================
    # SPARK CONFIGURATION
    # =========================================================================
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.executor.cores", "2")
    conf.set("spark.sql.shuffle.partitions", "8")
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Enable Delta Lake auto optimize
    conf.set("spark.delta.optimize.enabled", "true")
    conf.set("spark.delta.autoCompact.enabled", "true")

    # Create SparkSession with Sedona and Delta
    spark = (
        SparkSession.builder.config(conf=conf)
        .config(
            "spark.plugins",
            "org.apache.sedona.SedonaPlugin,io.delta.spark.DeltaSparkSessionPlugin",
        )
        .getOrCreate()
    )

    # Register Sedona SQL functions
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Spark session initialized: {spark.version}")
    logger.info(f"Spark UI: {spark.sparkContext.uiWebUrl.get()}")

    return spark


# =============================================================================
# DATA INGESTION (Bronze Layer)
# =============================================================================


def read_bronze_data(
    spark: "SparkSession",
    config: Config,
    input_format: str = "geojson",
    input_path: Optional[str] = None,
) -> "DataFrame":
    """
    Read raw data from the Bronze layer (MinIO).

    Supports multiple input formats:
    - geojson: GeoJSON files from Bronze bucket
    - csv: CSV files with lat/lon columns
    - parquet: Parquet files

    Parameters:
        spark: SparkSession
        config: Configuration
        input_format: Format of input data ("geojson", "csv", "parquet")
        input_path: Optional explicit path (overrides config)

    Returns:
        DataFrame with raw data
    """
    # Determine input path
    if input_path is None:
        input_path = f"s3://{config.MINIO_BUCKET_BRONZE}/raw/{input_format}"

    logger.info(f"Reading Bronze data from: {input_path}")

    if input_format == "geojson":
        # Read GeoJSON using Sedona's GeoJSON reader
        df = (
            spark.read.format("geojson").option("mode", "PERMIT_EMPTY").load(input_path)
        )

    elif input_format == "csv":
        # Read CSV with lat/lon columns
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(input_path)
        )

    elif input_format == "parquet":
        df = spark.read.format("parquet").load(input_path)

    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    row_count = df.count()
    logger.info(f"Loaded {row_count} rows from Bronze layer")

    return df


# =============================================================================
# SPATIAL TRANSFORMATION (Core GeoAI)
# =============================================================================


def transform_to_geometry(df: "DataFrame", config: Config) -> "DataFrame":
    """
    Transform latitude/longitude columns to Geometry using Apache Sedona.

    This function:
    1. Validates required lat/lon columns exist
    2. Creates Geometry using ST_Point (Sedona spatial function)
    3. Standardizes the CRS to EPSG:4326

    Parameters:
        df: Input DataFrame with lat/lon columns
        config: Configuration

    Returns:
        DataFrame with added geometry column
    """
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F

    logger.info("Converting lat/lon to geometry using Sedona ST_Point...")

    # Check if lat/lon columns exist
    # Note: Schema may vary based on input format
    lat_col = config.LAT_COL
    lon_col = config.LON_COL

    # Determine column names (case-insensitive search)
    columns = df.columns
    lat_col = next(
        (c for c in columns if c.lower() in [lat_col.lower(), "lat"]), lat_col
    )
    lon_col = next(
        (c for c in columns if c.lower() in [lon_col.lower(), "lon"]), lon_col
    )

    logger.info(f"Using columns: {lat_col}, {lon_col}")

    # =========================================================================
    # CREATE GEOMETRY USING SEDONA
    # =========================================================================
    # ST_Point creates a Point geometry from lon/lat (Note: lon first!)
    df = df.withColumn(
        config.GEOMETRY_COL,
        F.expr(f"ST_Point(cast({lon_col} as double), cast({lat_col} as double))"),
    )

    # =========================================================================
    # SET CRS TO WGS84 (EPSG:4326)
    # =========================================================================
    # In Sedona with ST_SetSRID, we assign the CRS directly to the geometry
    df = df.withColumn(
        config.GEOMETRY_COL, F.expr(f"ST_SetSRID({config.GEOMETRY_COL}, 4326)")
    )

    # Valid CRS assignment is complete - data is now in EPSG:4326

    # =========================================================================
    # DATA QUALITY VALIDATIONS
    # =========================================================================
    # Filter out invalid geometries (null or empty)
    df = df.filter(F.col(config.GEOMETRY_COL).isNotNull())

    # Validate geometry is within valid CRS bounds (WGS84)
    # Latitude: -90 to 90, Longitude: -180 to 180
    df = df.filter(
        (F.col(lat_col) >= -90)
        & (F.col(lat_col) <= 90)
        & (F.col(lon_col) >= -180)
        & (F.col(lon_col) <= 180)
    )

    # =========================================================================
    # ADD METADATA COLUMNS
    # =========================================================================
    df = df.withColumn("processing_timestamp", F.current_timestamp())
    df = df.withColumn("data_quality_flag", F.lit("validated"))
    df = df.withColumn("crs", F.lit(config.TARGET_CRS))

    # Log transformation metrics
    valid_count = df.count()
    logger.info(f"Transformation complete: {valid_count} valid geometries")

    return df


# =============================================================================
# SILVER LAYER WRITE (Delta Lake)
# =============================================================================


def write_silver_data(df: "DataFrame", config: Config, mode: str = "overwrite") -> None:
    """
    Write the transformed data to the Silver layer as a Delta Table.

    Delta Lake provides:
    - ACID transactions
    - Time travel (versioning)
    - Schema enforcement
    - Data quality checks

    Parameters:
        df: Transformed DataFrame
        config: Configuration
        mode: Write mode ("overwrite", "append")
    """
    output_path = f"s3://{config.MINIO_BUCKET_SILVER}/{config.DELTA_TABLE_NAME}"

    logger.info(f"Writing Silver Delta Table to: {output_path}")

    # =========================================================================
    # DELTA LAKE WRITE WITH OPTIMIZATION
    # =========================================================================
    df.write.format("delta").mode(mode).option(
        "compression", config.DELTA_COMPRESSION
    ).option("overwriteSchema", "true").partitionBy("data_quality_flag").save(
        output_path
    )

    # =========================================================================
    # VERIFY WRITE
    # =========================================================================
    verify_df = spark.read.format("delta").load(output_path)
    verify_count = verify_df.count()
    verify_version = (
        verify_df.groupBy()
        .agg(F.max("_metadata.row_index").alias("max_version"))
        .collect()[0]["max_version"]
    )

    logger.info(f"Silver layer write complete: {verify_count} rows")
    logger.info(f"Delta Table version: {verify_version}")


def run_silver_elt(input_format: str = "csv", input_path: Optional[str] = None) -> None:
    """
    Execute the complete Silver layer ELT pipeline.

    Parameters:
        input_format: Format of input data ("geojson", "csv", "parquet")
        input_path: Optional explicit input path
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting SILVER LAYER ELT Pipeline")
    logger.info("=" * 60)

    try:
        # 1. Create Spark Session
        global spark
        config = Config()
        spark = create_spark_session(config)

        # 2. Read Bronze Data
        logger.info("[STEP 1] Reading Bronze Layer...")
        bronze_df = read_bronze_data(spark, config, input_format, input_path)

        # 3. Transform to Geometry
        logger.info("[STEP 2] Transforming to Geometry (Sedona)...")
        silver_df = transform_to_geometry(bronze_df, config)

        # 4. Write Silver Delta Table
        logger.info("[STEP 3] Writing Silver Delta Table...")
        write_silver_data(silver_df, config)

        # Log success
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"SILVER LAYER ELT Complete: {duration:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Silver ELT Failed: {str(e)}", exc_info=True)
        raise

    finally:
        if "spark" in dir():
            spark.stop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoAI Silver Layer Spatial Transform")
    parser.add_argument(
        "--input-format",
        type=str,
        default="csv",
        choices=["geojson", "csv", "parquet"],
        help="Input data format (default: csv)",
    )
    parser.add_argument(
        "--input-path", type=str, default=None, help="Override input path"
    )

    args = parser.parse_args()

    # Run the ELT pipeline
    run_silver_elt(input_format=args.input_format, input_path=args.input_path)
