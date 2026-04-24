#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SILVER LAYER - SPATIAL TRANSFORMATION WITH PYSPARK + SHAPELY
================================================================================
File: src/silver_spatial_transform.py

Purpose:
    - Read raw Bronze data from parquet files
    - Transform DataFrames using Shapely for geometry
    - Convert Lat/Lng floats to WKT Geometry (ST_Point equivalent)
    - Parse GeoJSON boundaries to WKT
    - Standardize to EPSG:4326 (WGS84)
    - Write cleaned data to Silver Parquet Tables

Key Technologies:
    - Apache Spark (PySpark)
    - Shapely (geometry operations)
    - Parquet (columnar storage)

Author: GeoAI Principal Data Engineer
Version: 2.0.0
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

    # Local paths for Docker (where bronze ingestion writes parquet files)
    LOCAL_BRONZE_PATH = "/tmp/geoai/bronze"
    LOCAL_SILVER_PATH = "/tmp/geoai/silver"

    # Spark Configuration
    SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "local[*]")
    APP_NAME = "GeoAI_Silver_Spatial_Transform"

    # Spatial Reference (WGS84)
    TARGET_CRS = "EPSG:4326"

    # Data source mappings
    BRONZE_FILES = {
        "us_accidents": "us_accidents/part-*.parquet",
        "neighborhoods": "us_neighborhoods/part-*.parquet",
        "usgs_earthquakes": "usgs_earthquakes/part-*.parquet",
        "osm_infrastructure": "osm_infrastructure/part-*.parquet",
        "nyc_311_requests": "nyc_311/part-*.parquet",
    }


# =============================================================================
# SPARK SESSION BUILDER WITH SEDONA
# =============================================================================


def create_spark_session(config: Config):
    """
    Create SparkSession with Delta Lake extension.

    Parameters:
        config: Configuration object

    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    logger.info("Initializing Spark session...")

    # Build Spark configuration
    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # =============================================================================
# DELTA LAKE CONFIGURATION
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
    Convert latitude/longitude columns to Geometry using Shapely-based UDF.

    Parameters:
        df: Input DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        geometry_col: Output geometry column name
        drop_nulls: Whether to drop null geometries

    Returns:
        DataFrame with added Geometry column (as WKT string for Delta)
    """
    from pyspark.sql import functions as F
    from shapely import wkt
    from shapely.geometry import Point

    # Handle column name variations (case-insensitive)
    columns = df.columns
    lat_col = next((c for c in columns if c.lower() == lat_col.lower()), lat_col)
    lon_col = next((c for c in columns if c.lower() == lon_col.lower()), lon_col)

    logger.info(f"Creating geometry from {lat_col}, {lon_col}")

    # Create geometry using Shapely UDF - returns WKT string
    @F.udf(returnType="string")
    def make_point_udf(lat, lon):
        if lat is None or lon is None:
            return None
        try:
            point = Point(float(lon), float(lat))  # lon, lat order
            return point.wkt  # Convert to WKT string
        except:
            return None

    # Use the UDF to create WKT geometry
    df = df.withColumn(geometry_col, make_point_udf(F.col(lat_col), F.col(lon_col)))

    # Data quality: Filter out invalid coordinates
    if drop_nulls:
        df = df.filter(F.col(geometry_col).isNotNull())
        # Also filter by coordinate bounds for WGS84
        df = df.filter(
            (F.col(lat_col) >= -90)
            & (F.col(lat_col) <= 90)
            & (F.col(lon_col) >= -180)
            & (F.col(lon_col) <= 180)
        )

    logger.info(f"Created geometry column with {df.count()} valid records")
    return df


def parse_geojson_geometry(
    df: "DataFrame",
    geojson_col: str,
    geometry_col: str = "geometry",
    drop_nulls: bool = True,
) -> "DataFrame":
    """
    Parse GeoJSON geometry column using Shapely-based UDF.

    Parameters:
        df: Input DataFrame with GeoJSON geometry column
        geojson_col: Name of GeoJSON column
        geometry_col: Output geometry column name
        drop_nulls: Whether to drop null geometries

    Returns:
        DataFrame with parsed Geometry column (as WKT string)
    """
    from pyspark.sql import functions as F
    from shapely.geometry import shape
    from shapely import wkt

    logger.info(f"Parsing GeoJSON from column: {geojson_col}")

    # First, ensure the column contains valid GeoJSON
    df = df.filter(F.col(geojson_col).isNotNull())

    # Parse GeoJSON using Shapely UDF
    @F.udf(returnType="string")
    def parse_geojson_udf(geojson_str):
        if not geojson_str:
            return None
        try:
            import json
            if isinstance(geojson_str, str):
                gj = json.loads(geojson_str)
            else:
                gj = geojson_str
            geom = shape(gj)
            return geom.wkt
        except:
            return None

    df = df.withColumn(geometry_col, parse_geojson_udf(F.col(geojson_col)))

    if drop_nulls:
        df = df.filter(F.col(geometry_col).isNotNull())

    logger.info(f"Parsed GeoJSON geometry with {df.count()} valid records")
    return df

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
    Read raw data from Bronze layer (local or MinIO).

    Parameters:
        spark: SparkSession
        config: Config
        source_name: Key from BRONZE_FILES mapping

    Returns:
        DataFrame with raw data
    """
    filename = config.BRONZE_FILES.get(source_name, "")
    # Try local path first, then MinIO s3a://
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
        elif filename.endswith(".parquet"):
            df = spark.read.format("parquet").load(local_path)
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

    # Add metadata columns with correct column names from bronze data
    df = df.withColumn("source", F.lit("us_accidents"))
    df = df.withColumn("source_id", F.col("incident_id"))
    df = df.withColumn("event_timestamp", F.col("start_time"))
    df = df.withColumn("description", F.col("incident_description"))
    df = df.withColumn("severity", F.col("severity"))
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
    # Common names: geometry, geom, the_geom, boundary
    geom_col = next(
        (c for c in df.columns if c in ["geometry", "geom", "the_geom"]), None
    )

    if geom_col:
        # Use existing geometry column
        df = parse_geojson_geometry(df, geom_col, "geometry")
    else:
        # Create empty WKT column (GeoJSON parsing requires different approach)
        df = df.withColumn("geometry", F.lit("").cast("string"))  # Placeholder

    # Add metadata
    df = df.withColumn("source", F.lit("neighborhoods"))
    # Use a proper string for source_id instead of None literal
    id_col = next((c for c in df.columns if "id" in c.lower()), "neighborhood_id")
    df = df.withColumn("source_id", F.col(id_col).cast("string"))

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
    df = df.withColumn("source_id", F.col("event_id").cast("string"))
    df = df.withColumn("event_timestamp", F.col("time"))
    df = df.withColumn("description", F.col("place"))
    df = df.withColumn("severity", F.col("magnitude"))
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
    df = df.withColumn("source_id", F.col("facility_id").cast("string"))
    df = df.withColumn("facility_type", F.col("facility_type"))
    df = df.withColumn("facility_name", F.col("name"))
    df = df.withColumn("operator", F.lit(None).cast("string"))  # Not in source data
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    return df.select(
        "source",
        "source_id",
        "facility_type",
        "facility_name",
        "operator",
        "geometry",
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
    df = df.withColumn("source_id", F.col("request_id").cast("string"))
    df = df.withColumn("event_timestamp", F.col("created_date"))
    df = df.withColumn("description", F.col("description"))  # Already exists
    df = df.withColumn("processing_timestamp", F.current_timestamp())

    # Severity would need to be derived or use existing status
    df = df.withColumn("severity", F.lit(5))  # Default until AI enrichment

    return df.select(
        "source",
        "source_id",
        "event_timestamp",
        "description",
        "severity",
        "geometry",
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
    Write DataFrame to Silver layer as Delta Table.

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

    logger.info(f"Writing Silver Delta Table: {output_path}")

    try:
        df.write.format("delta").mode(mode).option("compression", "snappy").save(
            output_path
        )
    except Exception as e:
        logger.warning(f"Delta write failed: {e}, falling back to Parquet")
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
        # Transform each data source
        # =========================================================================

        logger.info("[1/5] Transforming US Accidents...")
        df_accidents = transform_us_accidents(spark, config)
        if df_accidents:
            write_silver_table(spark, df_accidents, config, "us_accidents")

        logger.info("[2/5] Transforming Neighborhoods...")
        df_neighborhoods = transform_neighborhoods(spark, config)
        if df_neighborhoods:
            write_silver_table(spark, df_neighborhoods, config, "neighborhoods")

        logger.info("[3/5] Transforming USGS Earthquakes...")
        df_usgs = transform_usgs_earthquakes(spark, config)
        if df_usgs:
            write_silver_table(spark, df_usgs, config, "usgs_earthquakes")

        logger.info("[4/5] Transforming OSM Infrastructure...")
        df_osm = transform_osm_infrastructure(spark, config)
        if df_osm:
            write_silver_table(spark, df_osm, config, "osm_infrastructure")

        logger.info("[5/5] Transforming NYC 311 Requests...")
        df_311 = transform_nyc_311(spark, config)
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
