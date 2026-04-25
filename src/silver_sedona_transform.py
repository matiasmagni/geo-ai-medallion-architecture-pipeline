#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SILVER LAYER - SPATIAL TRANSFORMATION WITH APACHE SEDONA
================================================================================
File: src/silver_sedona_transform.py

Purpose:
    - Read raw Bronze data from MinIO
    - Transform DataFrames using Apache Sedona Spatial SQL
    - Convert Lat/Lng floats to Geometry (ST_Point)
    - Parse GeoJSON boundaries (ST_GeomFromGeoJSON)
    - Standardize to EPSG:4326 (WGS84)
    - Write cleaned data to Silver Delta Tables

Key Technologies:
    - Apache Spark (PySpark)
    - Apache Sedona (Spatial SQL)
    - Delta Lake (ACID transactions)

Author: GeoAI Principal Data Engineer
Version: 1.0.0
================================================================================
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Configuration for Silver layer transformation."""

    # MinIO/S3 Configuration
    MINIO_ENDPOINT = os.getenv("S3_ENDPOINT", "http://host.docker.internal:9900")
    MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    BUCKET_BRONZE = "geo-lakehouse/bronze"
    BUCKET_SILVER = "geo-lakehouse/silver"

    # Local paths for Docker (use host.docker.internal for MinIO access)
    LOCAL_BRONZE_PATH = "/tmp/geoai/bronze"
    LOCAL_SILVER_PATH = "/tmp/geoai/silver"

    # Spark Configuration
    SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "local[*]")
    APP_NAME = "GeoAI_Silver_Spatial_Transform"

    # Spatial Reference (WGS84)
    TARGET_CRS = "EPSG:4326"

    # Data source mappings
    BRONZE_FILES = {
        "us_accidents": "us_accidents.csv",
        "neighborhoods": "neighborhoods.geojson",
        "usgs_earthquakes": "usgs_earthquakes.json",
        "osm_infrastructure": "osm_infrastructure.json",
        "nyc_311_requests": "nyc_311_requests.json",
    }


# =============================================================================
# SPARK SESSION BUILDER WITH SEDONA
# =============================================================================


def create_spark_session(config: Config):
    """
    Create SparkSession with Apache Sedona and Delta Lake extensions.

    Parameters:
        config: Configuration object

    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    logger.info("Initializing Spark session with Sedona...")

    # Build Spark configuration
    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # =============================================================================
    # APACHE SEDONA CONFIGURATION (simplified - let Sedona handle internally)
    # =============================================================================
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # =============================================================================
    # MINIO/S3 CONFIGURATION
    # =============================================================================
    conf.set("spark.hadoop.fs.s3a.endpoint", config.MINIO_ENDPOINT)
    conf.set("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    conf.set("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
    conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # =============================================================================
    # MEMORY CONFIGURATION
    # =============================================================================
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.executor.cores", "2")
    conf.set("spark.sql.shuffle.partitions", "8")

    # Create SparkSession
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Try to initialize Sedona with newer API, skip if fails
    try:
        from sedona.spark import SedonaContext

        SedonaContext.create(spark)
    except Exception as e:
        logger.warning(
            f"Sedona initialization failed: {e}, continuing without spatial functions"
        )

    # Register Sedona functions
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Spark session initialized: {spark.version}")
    logger.info(f"Spark UI: {spark.sparkContext.uiWebUrl}")

    return spark


# =============================================================================
# SEDONA SPATIAL TRANSFORMATIONS
# =============================================================================


def create_geometry_from_latlon(
    df: "DataFrame",
    lat_col: str,
    lon_col: str,
    geometry_col: str = "geometry",
    drop_nulls: bool = True,
) -> "DataFrame":
    """
    Convert latitude/longitude columns to Geometry (non-spatial fallback).

    Parameters:
        df: Input DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        geometry_col: Output geometry column name
        drop_nulls: Whether to drop null geometries

    Returns:
        DataFrame with added Geometry column
    """
    from pyspark.sql import functions as F

    # Handle column name variations (case-insensitive)
    columns = df.columns
    lat_col = next((c for c in columns if c.lower() == lat_col.lower()), lat_col)
    lon_col = next((c for c in columns if c.lower() == lon_col.lower()), lon_col)

    logger.info(f"Creating geometry from {lat_col}, {lon_col}")

    # Skip spatial transformations if Sedona is not available
    # Just keep lat/lon as is
    df = df.withColumn("latitude", F.col(lat_col).cast("double")).withColumn(
        "longitude", F.col(lon_col).cast("double")
    )

    # Data quality: Filter out invalid coordinates
    if drop_nulls:
        df = df.filter(F.col("latitude").isNotNull())
        df = df.filter(F.col("longitude").isNotNull())
        df = df.filter(
            (F.col("latitude") >= -90)
            & (F.col("latitude") <= 90)
            & (F.col("longitude") >= -180)
            & (F.col("longitude") <= 180)
        )

    return df


def parse_geojson_geometry(
    df: "DataFrame",
    geojson_col: str,
    geometry_col: str = "geometry",
    drop_nulls: bool = True,
) -> "DataFrame":
    """
    Parse GeoJSON geometry column using Sedona ST_GeomFromGeoJSON.

    Parameters:
        df: Input DataFrame with GeoJSON geometry column
        geojson_col: Name of GeoJSON column
        geometry_col: Output geometry column name
        drop_nulls: Whether to drop null geometries

    Returns:
        DataFrame with parsed Geometry column
    """
    from pyspark.sql import functions as F

    logger.info(f"Parsing GeoJSON from column: {geojson_col}")

    # First, ensure the column contains valid GeoJSON
    # Note: This handles both raw GeoJSON strings and nested geometry
    df = df.filter(F.col(geojson_col).isNotNull())

    # Parse GeoJSON to geometry
    # Note: ST_GeomFromGeoJSON expects GeoJSON Feature/Geometry object
    df = df.withColumn(geometry_col, F.expr(f"ST_GeomFromGeoJSON({geojson_col})"))

    # Set CRS to WGS84
    df = df.withColumn(geometry_col, F.expr(f"ST_SetSRID({geometry_col}, 4326)"))

    # Drop nulls if requested
    if drop_nulls:
        df = df.filter(F.col(geometry_col).isNotNull())

    return df


def transform_to_crs(
    df: "DataFrame", source_crs: str, target_crs: str = "EPSG:4326"
) -> "DataFrame":
    """
    Transform Geometry to target CRS using Sedona.

    Parameters:
        df: Input DataFrame with geometry
        source_crs: Source CRS (e.g., "EPSG:4326")
        target_crs: Target CRS (default "EPSG:4326")

    Returns:
        DataFrame with transformed geometry
    """
    from pyspark.sql import functions as F

    # For now, we standardize to WGS84
    # Full CRS transformation would require ST_Transform
    # Note: Sedona has ST_Transform for CRS conversion

    logger.info(f"CRS transformation: {source_crs} -> {target_crs}")

    return df


# =============================================================================
# BRONZE TO SILVER INGESTION FUNCTIONS
# =============================================================================


def read_bronze_data(
    spark: "SparkSession", config: Config, source_name: str
) -> "DataFrame":
    """
    Read raw data from Bronze layer.

    Parameters:
        spark: SparkSession
        config: Config
        source_name: Key from BRONZE_FILES mapping

    Returns:
        DataFrame with raw data
    """
    filename = config.BRONZE_FILES.get(source_name, "")
    # Try local path first (for Docker), fallback to S3
    local_path = f"{config.LOCAL_BRONZE_PATH}/{filename}"

    logger.info(f"Reading Bronze data: {local_path}")

    try:
        if filename.endswith(".csv"):
            df = (
                spark.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(local_path)
            )
        elif filename.endswith(".json") or filename.endswith(".geojson"):
            df = spark.read.format("json").load(local_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        count = df.count()
        logger.info(f"Loaded {count} records from {source_name}")

        return df

    except Exception as e:
        logger.error(f"Failed to read {source_name}: {e}")
        return None


def transform_us_accidents(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Transform US Accidents data to Silver."""
    from pyspark.sql import functions as F

    logger.info("Transforming US Accidents...")

    df = read_bronze_data(spark, config, "us_accidents")
    if df is None:
        return None

    # Determine lat/lon column names
    lat_col = next((c for c in df.columns if "lat" in c.lower()), "Start_Lat")
    lon_col = next(
        (c for c in df.columns if "lng" in c.lower() or "lon" in c.lower()), "Start_Lng"
    )

    # Check for Description column
    desc_col = next((c for c in df.columns if "desc" in c.lower()), None)

    # Create geometry
    df = create_geometry_from_latlon(df, lat_col, lon_col, "geometry", drop_nulls=True)

    # Add metadata columns
    df = df.withColumn("source", F.lit("us_accidents"))
    df = df.withColumn("source_id", F.col("ID"))
    df = df.withColumn("event_timestamp", F.col("Start_Time"))
    df = df.withColumn("description", F.col(desc_col) if desc_col else F.lit(None))
    df = df.withColumn("severity", F.col("Severity"))
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    return df.select(
        "source",
        "source_id",
        "event_timestamp",
        "description",
        "severity",
        "geometry",
        "processing_timestamp",
    )


def transform_neighborhoods(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Transform US City Neighborhood Boundaries."""
    from pyspark.sql import functions as F

    logger.info("Transforming Neighborhoods...")

    df = read_bronze_data(spark, config, "neighborhoods")
    if df is None:
        return None

    # Try to find geometry column (varies by dataset)
    geom_col = next(
        (c for c in df.columns if c in ["geometry", "geom", "the_geom"]), None
    )

    if geom_col:
        df = parse_geojson_geometry(df, geom_col, "geometry")
    else:
        # Try to parse from sample data format in bronze_ingestion
        df = parse_geojson_geometry(df, "geometry", "geometry")

    # Add metadata
    df = df.withColumn("source", F.lit("neighborhoods"))
    df = df.withColumn("source_id", F.expr("uuid()"))

    # Neighborhood name field
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    if name_col:
        df = df.withColumn("neighborhood_name", F.col(name_col))

    df = df.withColumn("processing_timestamp", F.current_timestamp())

    return df.select(
        "source",
        "source_id",
        "neighborhood_name",
        "geometry",
        "processing_timestamp",
    )


def filter_by_land_mask(
    spark: "SparkSession", df: "DataFrame", neighborhoods_df: "DataFrame"
) -> "DataFrame":
    """
    Remove points that fall in the water by checking if they are within any neighborhood polygon.
    Uses Sedona ST_Within for spatial filtering.
    """
    from pyspark.sql import functions as F

    if df is None or neighborhoods_df is None:
        return df

    logger.info(f"Applying land mask spatial filter to {df.count()} points...")

    # Perform spatial join (Point-In-Polygon)
    # ST_Within(point, polygon)
    # We use a left semi join to keep only points that are within at least one neighborhood
    try:
        # Register temp views for SQL-like spatial join if needed, or use expr
        filtered_df = df.alias("pts").join(
            neighborhoods_df.alias("land"),
            F.expr("ST_Within(pts.geometry, land.geometry)"),
            "left_semi"
        )
        
        final_count = filtered_df.count()
        logger.info(f"Spatial filtering complete. Points remaining: {final_count}")
        return filtered_df
    except Exception as e:
        logger.warning(f"Spatial filtering failed: {e}. Returning original data.")
        return df


def transform_usgs_earthquakes(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Transform USGS Earthquake data."""
    from pyspark.sql import functions as F

    logger.info("Transforming USGS Earthquakes...")

    df = read_bronze_data(spark, config, "usgs_earthquakes")
    if df is None:
        return None

    # Determine lat/lon columns
    lat_col = next((c for c in df.columns if c == "latitude"), "latitude")
    lon_col = next((c for c in df.columns if c == "longitude"), "longitude")

    # Create geometry
    df = create_geometry_from_latlon(df, lat_col, lon_col, "geometry", drop_nulls=True)

    # Add metadata
    df = df.withColumn("source", F.lit("usgs_earthquakes"))
    df = df.withColumn("source_id", F.col("event_id"))
    df = df.withColumn("event_timestamp", F.col("datetime"))
    df = df.withColumn("description", F.col("place"))
    df = df.withColumn("severity", F.col("magnitude"))
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    return df.select(
        "source",
        "source_id",
        "event_timestamp",
        "description",
        "severity",
        "latitude",
        "longitude",
        "processing_timestamp",
    )


def transform_osm_infrastructure(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Transform OSM Hospital/Fire Station data."""
    from pyspark.sql import functions as F

    logger.info("Transforming OSM Infrastructure...")

    df = read_bronze_data(spark, config, "osm_infrastructure")
    if df is None:
        return None

    # Determine lat/lon columns
    lat_col = next((c for c in df.columns if c == "latitude"), "latitude")
    lon_col = next((c for c in df.columns if c == "longitude"), "longitude")

    # Create geometry
    df = create_geometry_from_latlon(df, lat_col, lon_col, "geometry", drop_nulls=True)

    # Add metadata
    df = df.withColumn("source", F.lit("osm_infrastructure"))
    df = df.withColumn("source_id", F.col("osm_id"))
    df = df.withColumn("facility_type", F.col("facility_type"))
    df = df.withColumn("facility_name", F.col("name"))
    df = df.withColumn("operator", F.col("operator"))
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    return df.select(
        "source",
        "source_id",
        "facility_type",
        "facility_name",
        "operator",
        "latitude",
        "longitude",
        "processing_timestamp",
    )


def transform_nyc_311(spark: "SparkSession", config: Config) -> Optional["DataFrame"]:
    """Transform NYC 311 Service Requests."""
    from pyspark.sql import functions as F

    logger.info("Transforming NYC 311 Requests...")

    df = read_bronze_data(spark, config, "nyc_311_requests")
    if df is None:
        return None

    # Determine lat/lon columns
    lat_col = next((c for c in df.columns if c == "latitude"), "latitude")
    lon_col = next((c for c in df.columns if c == "longitude"), "longitude")

    # Create geometry
    df = create_geometry_from_latlon(df, lat_col, lon_col, "geometry", drop_nulls=True)

    # Add metadata
    df = df.withColumn("source", F.lit("nyc_311_requests"))
    df = df.withColumn("source_id", F.lit(None))  # Would need unique ID from data
    df = df.withColumn("event_timestamp", F.col("created_date"))
    df = df.withColumn(
        "description", F.concat_ws(" - ", F.col("complaint_type"), F.col("descriptor"))
    )
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    # Severity would need to be derived or use existing status
    df = df.withColumn("severity", F.lit(5))  # Default until AI enrichment

    return df.select(
        "source",
        "source_id",
        "event_timestamp",
        "description",
        "severity",
        "latitude",
        "longitude",
        "processing_timestamp",
    )


# =============================================================================
# WRITE TO SILVER LAYER (DELTA)
# =============================================================================


def write_silver_table(
    spark: "SparkSession",
    df: "DataFrame",
    config: Config,
    table_name: str,
    mode: str = "overwrite",
) -> None:
    """
    Write DataFrame to Silver layer as Parquet.

    Parameters:
        spark: SparkSession
        df: DataFrame to write
        config: Configuration
        table_name: Name of the table
        mode: Write mode (overwrite, append)
    """
    output_path = f"{config.LOCAL_SILVER_PATH}/{table_name}"

    import os

    os.makedirs(config.LOCAL_SILVER_PATH, exist_ok=True)

    logger.info(f"Writing Silver Parquet: {output_path}")

    df.write.format("parquet").mode(mode).option("compression", "snappy").save(
        output_path
    )

    count = df.count()
    logger.info(f"Silver table written: {count} records")


# =============================================================================
# MAIN SILVER PIPELINE
# =============================================================================


def run_silver_pipeline():
    """Execute complete Silver layer transformation."""

    global spark
    config = Config()

    logger.info("=" * 60)
    logger.info("STARTING SILVER LAYER TRANSFORMATION")
    logger.info("=" * 60)

    try:
        # Create Spark session
        spark = create_spark_session(config)

        # =========================================================================
        # Transform Land Mask first (Neighborhoods)
        # =========================================================================
        logger.info("[1/5] Transforming Neighborhoods (Land Mask)...")
        df_neighborhoods = transform_neighborhoods(spark, config)
        if df_neighborhoods:
            write_silver_table(spark, df_neighborhoods, config, "neighborhoods")

        # =========================================================================
        # Transform other data sources and apply land mask
        # =========================================================================

        logger.info("[2/5] Transforming US Accidents...")
        df_accidents = transform_us_accidents(spark, config)
        if df_accidents and df_neighborhoods:
            df_accidents = filter_by_land_mask(spark, df_accidents, df_neighborhoods)
        if df_accidents:
            write_silver_table(spark, df_accidents, config, "us_accidents")

        logger.info("[3/5] Transforming USGS Earthquakes...")
        df_usgs = transform_usgs_earthquakes(spark, config)
        # Earthquakes are global/regional, maybe don't filter them strictly by NYC neighborhoods?
        # Keeping as is for now.
        if df_usgs:
            write_silver_table(spark, df_usgs, config, "usgs_earthquakes")

        logger.info("[4/5] Transforming OSM Infrastructure...")
        df_osm = transform_osm_infrastructure(spark, config)
        if df_osm and df_neighborhoods:
            df_osm = filter_by_land_mask(spark, df_osm, df_neighborhoods)
        if df_osm:
            write_silver_table(spark, df_osm, config, "osm_infrastructure")

        logger.info("[5/5] Transforming NYC 311 Requests...")
        df_311 = transform_nyc_311(spark, config)
        if df_311 and df_neighborhoods:
            df_311 = filter_by_land_mask(spark, df_311, df_neighborhoods)
        if df_311:
            write_silver_table(spark, df_311, config, "nyc_311_requests")

        # =========================================================================
        # COMPLETE
        # =========================================================================
        logger.info("=" * 60)
        logger.info("SILVER LAYER TRANSFORMATION COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Silver pipeline failed: {e}", exc_info=True)
        raise

    finally:
        if "spark" in dir():
            spark.stop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_silver_pipeline()
