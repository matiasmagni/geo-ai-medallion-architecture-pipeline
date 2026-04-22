#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GOLD LAYER - Kimball Star Schema Dimensional Modeling
================================================================================
File: src/gold_dimensional_modeling.py

Purpose:
    Build Kimball Star Schema from Silver Delta Tables.

    IMPORTANT: NO LLM inference allowed in this layer!
    All AI processing was done in Silver (Shift-Left pattern).

    This layer does:
    1. Read enriched Delta Tables from Silver
    2. Build Dimension Tables (DIM_*)
    3. Build Fact Table (FACT_*)
    4. Perform spatial joins (ST_Within, ST_Distance)
    5. Write Gold Delta Tables

Architecture:
    - DIM_NEIGHBORHOODS: Geographic dimensions
    - DIM_INFRASTRUCTURE: Hospitals, fire stations
    - FACT_HAZARD_EVENTS: All hazard events with foreign keys

Spatial Joins:
    - Events → Neighborhoods (ST_Within) → neighborhood_id
    - Events → Nearest Infrastructure (ST_Distance) → nearest_hospital_distance

Author: GeoAI Principal Data Engineer & Data Architect
Version: 2.0.0 (Star Schema - No LLM)
================================================================================
"""

import os
import sys
import logging
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
    Configuration for Gold layer dimensional modeling.

    IMPORTANT: No LLM configuration here!
    All AI is in Silver layer.
    """

    # MinIO/S3 Configuration
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # Bucket paths
    SILVER_BUCKET: str = os.getenv("SILVER_BUCKET", "geo-lakehouse/silver")
    GOLD_BUCKET: str = os.getenv("GOLD_BUCKET", "geo-lakehouse/gold")

    # Spark Configuration
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
    APP_NAME: str = "GeoAI_Gold_Dimensional"

    # Delta Table Configuration
    DELTA_COMPRESSION: str = "snappy"


# =============================================================================
# SPARK SESSION
# =============================================================================


def create_spark_session(config: Config) -> SparkSession:
    """
    Create SparkSession with Sedona and Delta Lake.

    Parameters
    ----------
    config : Config

    Returns
    -------
    SparkSession
    """
    builder = (
        SparkSession.builder.appName(config.APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.adaptive.enabled", "true")
    )

    try:
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-python-adapter-3.4_2.12:1.4.1,io.delta:delta-spark_2.12:2.4.0",
        )
    except Exception:
        pass

    return builder.getOrCreate()


# =============================================================================
# READ SILVER DELTA TABLES
# =============================================================================


def read_silver_table(spark: SparkSession, table: str) -> DataFrame:
    """
    Read Silver Delta Table.

    Parameters
    ----------
    spark : SparkSession
    table : str
        Table name (e.g., 'us_accidents_silver')

    Returns
    -------
    DataFrame
    """
    path = f"s3a://{Config.SILVER_BUCKET}/{table}"

    try:
        df = spark.read.format("delta").load(path)
        logger.info(f"Read Delta: {table}")
        return df
    except Exception as e:
        logger.warning(f"Delta read failed ({e}), trying Parquet")
        try:
            df = spark.read.format("parquet").load(f"/tmp/geoai/silver/{table}")
            logger.info(f"Read Parquet: {table}")
            return df
        except Exception as e2:
            logger.error(f"Failed to read {table}: {e2}")
            raise


# =============================================================================
# DIMENSION: DIM_NEIGHBORHOODS
# =============================================================================


def create_dim_neighborhoods(spark: SparkSession) -> DataFrame:
    """
    Create DIM_NEIGHBORHOODS from Silver neighborhoods.

    Kimball Star Schema:
    - Surrogate key: neighborhood_sk (int, auto-increment)
    - Natural keys: neighborhood_id, county_fips, state_fips
    - Attributes: name, county_fips, state_fips

    Parameters
    ----------
    spark : SparkSession

    Returns
    -------
    DataFrame
        Dimension table
    """
    logger.info("Creating DIM_NEIGHBORHOODS...")

    df = read_silver_table(spark, "us_neighborhoods_silver")

    # Select and rename for star schema
    dim = df.select(
        F.col("neighborhood_id").alias("neighborhood_id"),
        F.col("name").alias("neighborhood_name"),
        F.col("county_fips").alias("county_fips"),
        F.col("state_fips").alias("state_fips"),
        F.col("geometry").alias("neighborhood_geometry"),
        F.col("silver_updated").alias("effective_date"),
    )

    # Add surrogate key
    window = Window.orderBy(F.col("neighborhood_id"))
    dim = dim.withColumn("neighborhood_sk", F.row_number().over(window))

    # Reorder columns
    dim = dim.select(
        "neighborhood_sk",
        "neighborhood_id",
        "neighborhood_name",
        "county_fips",
        "state_fips",
        "neighborhood_geometry",
        "effective_date",
    )

    logger.info(f"DIM_NEIGHBORHOODS: {dim.count()} rows")
    return dim


# =============================================================================
# DIMENSION: DIM_INFRASTRUCTURE
# =============================================================================


def create_dim_infrastructure(spark: SparkSession) -> DataFrame:
    """
    Create DIM_INFRASTRUCTURE from Silver OSM data.

    Kimball Star Schema:
    - Surrogate key: infrastructure_sk
    - Natural key: facility_id
    - Attributes: name, facility_type, location

    Parameters
    ----------
    spark : SparkSession

    Returns
    -------
    DataFrame
        Dimension table
    """
    logger.info("Creating DIM_INFRASTRUCTURE...")

    df = read_silver_table(spark, "osm_infrastructure_silver")

    # Select for dimension
    dim = df.select(
        F.col("facility_id").alias("facility_id"),
        F.col("name").alias("facility_name"),
        F.col("facility_type").alias("facility_type"),
        F.col("latitude").alias("facility_lat"),
        F.col("longitude").alias("facility_lon"),
        F.col("address").alias("facility_address"),
        F.col("geometry").alias("facility_geometry"),
        F.col("silver_updated").alias("effective_date"),
    )

    # Add surrogate key
    window = Window.orderBy(F.col("facility_id"))
    dim = dim.withColumn("infrastructure_sk", F.row_number().over(window))

    # Reorder
    dim = dim.select(
        "infrastructure_sk",
        "facility_id",
        "facility_name",
        "facility_type",
        "facility_lat",
        "facility_lon",
        "facility_address",
        "facility_geometry",
        "effective_date",
    )

    logger.info(f"DIM_INFRASTRUCTURE: {dim.count()} rows")
    return dim


# =============================================================================
# FACT: FACT_HAZARD_EVENTS
# =============================================================================


def create_fact_hazard_events(spark: SparkSession) -> DataFrame:
    """
    Create FACT_HAZARD_EVENTS by unioning all Silver event sources.

    Kimball Star Schema:
    - Surrogate key: event_sk
    - Foreign keys: neighborhood_sk (from spatial join)
    - Metrics: ai_severity, magnitude, etc.
    - Dimensions: ai_hazard_type (degenerate)

    IMPORTANT: This uses the AI columns from Silver!
    NO LLM calls here - just reads pre-computed columns.

    Parameters
    ----------
    spark : SparkSession

    Returns
    -------
    DataFrame
        Fact table
    """
    logger.info("Creating FACT_HAZARD_EVENTS...")

    # Read all event sources
    accidents = read_silver_table(spark, "us_accidents_silver")
    earthquakes = read_silver_table(spark, "usgs_earthquakes_silver")

    # Standardize columns for union
    # US Accidents
    acc = accidents.select(
        F.col("incident_id").alias("event_id"),
        F.lit("us_accidents").alias("source_system"),
        F.col("incident_description").alias("event_description"),
        F.col("latitude").alias("event_lat"),
        F.col("longitude").alias("event_lon"),
        F.col("geometry").alias("event_geometry"),
        F.col("ai_severity").alias("severity"),
        F.col("ai_hazard_type").alias("hazard_type"),
        F.col("start_time").alias("event_time"),
        F.col("address").alias("event_address"),
        F.col("city").alias("event_city"),
        F.col("state").alias("event_state"),
    ).withColumn("source_table", F.lit("us_accidents"))

    # USGS Earthquakes
    eq = earthquakes.select(
        F.col("event_id").alias("event_id"),
        F.lit("usgs_earthquakes").alias("source_system"),
        F.col("description").alias("event_description"),
        F.col("latitude").alias("event_lat"),
        F.col("longitude").alias("event_lon"),
        F.col("geometry").alias("event_geometry"),
        F.col("ai_severity").alias("severity"),
        F.col("ai_hazard_type").alias("hazard_type"),
        F.col("time").alias("event_time"),
        F.col("place").alias("event_address"),  # Use place as address
        F.lit(None).alias("event_city"),
        F.lit(None).alias("event_state"),
    ).withColumn("source_table", F.lit("usgs_earthquakes"))

    # Union all sources
    fact = acc.unionByName(eq, allowMissing=True)

    # Add surrogate key
    window = Window.orderBy(F.col("event_id"))
    fact = fact.withColumn("event_sk", F.row_number().over(window))

    # Rename for star schema clarity
    fact = fact.select(
        "event_sk",
        "event_id",
        "source_system",
        "source_table",
        "event_description",
        "event_lat",
        "event_lon",
        "event_geometry",
        "event_time",
        "severity",  # From Silver AI enrichment
        "hazard_type",  # From Silver AI enrichment
        "event_address",
        "event_city",
        "event_state",
        # Foreign keys (to be populated by spatial join)
        F.lit(None).alias("neighborhood_sk"),
        F.lit(None).alias("nearest_hospital_sk"),
        F.lit(None).alias("nearest_hospital_distance"),
    )

    logger.info(f"FACT_HAZARD_EVENTS: {fact.count()} rows")
    return fact


# =============================================================================
# SPATIAL JOINS
# =============================================================================


def spatial_join_events_to_neighborhoods(
    fact: DataFrame, dim_neighborhoods: DataFrame
) -> DataFrame:
    """
    Spatial join FACT events to DIM neighborhoods.

    Uses Sedona's ST_Within to find which neighborhood
    each event falls within.

    Parameters
    ----------
    fact : DataFrame
        FACT_HAZARD_EVENTS
    dim_neighborhoods : DataFrame
        DIM_NEIGHBORHOODS

    Returns
    -------
    DataFrame
        Fact with neighborhood_sk populated
    """
    logger.info("Spatial joining events to neighborhoods...")

    try:
        # Use Sedona ST_Within
        joined = fact.join(
            dim_neighborhoods,
            F.expr(
                "ST_Within(fact.event_geometry, dim_neighborhoods.neighborhood_geometry)"
            ),
            "left",
        )

        # Select final columns (with neighborhood_sk)
        joined = joined.select(
            fact["*"],
            F.col("dim_neighborhoods.neighborhood_sk").alias("neighborhood_sk"),
        )

    except Exception as e:
        logger.warning(f"ST_Within spatial join failed ({e})")
        # Fallback: leave neighborhood_sk as null
        joined = fact.withColumn("neighborhood_sk", F.lit(None))

    logger.info(
        f"Events with neighborhood_sk: {joined.filter('neighborhood_sk IS NOT NULL').count()}"
    )
    return joined


def spatial_join_events_to_nearest_infrastructure(
    fact: DataFrame, dim_infrastructure: DataFrame
) -> DataFrame:
    """
    Find nearest infrastructure (hospital/fire station) for each event.

    Uses Sedona's ST_Distance to calculate distance to
    nearest hospital for each event.

    Parameters
    ----------
    fact : DataFrame
        FACT_HAZARD_EVENTS
    dim_infrastructure : DataFrame
        DIM_INFRASTRUCTURE

    Returns
    -------
    DataFrame
        Fact with nearest hospital info
    """
    logger.info("Calculating nearest infrastructure for events...")

    # Filter to hospitals only
    hospitals = dim_infrastructure.filter(F.col("facility_type") == "hospital")

    # Cross join and calculate distance (expensive but accurate)
    # In production, use broadcast join with pre-calculated distances
    try:
        # For each event, find min distance to hospital
        events_with_dist = fact.crossJoin(hospitals).withColumn(
            "distance",
            F.expr("ST_Distance(fact.event_geometry, hospitals.facility_geometry)"),
        )

        # Window: for each event, find min distance
        window = Window.partitionBy("event_sk").orderBy("distance")
        events_with_min = events_with_dist.withColumn(
            "rank", F.row_number().over(window)
        ).filter("rank = 1")

        # Select nearest
        fact = events_with_min.select(
            fact["*"],
            F.col("hospitals.infrastructure_sk").alias("nearest_hospital_sk"),
            F.col("distance").alias("nearest_hospital_distance"),
        )

    except Exception as e:
        logger.warning(f"ST_Distance spatial join failed ({e})")
        # Fallback
        fact = fact.withColumn("nearest_hospital_sk", F.lit(None)).withColumn(
            "nearest_hospital_distance", F.lit(None)
        )

    # Count events with nearest hospital
    with_hospital = fact.filter("nearest_hospital_sk IS NOT NULL").count()
    logger.info(f"Events with nearest hospital: {with_hospital}")

    return fact


# =============================================================================
# AGGREGATE METRICS
# =============================================================================


def aggregate_hazard_metrics(fact: DataFrame) -> DataFrame:
    """
    Aggregate metrics for reporting.

    Generates summary statistics per neighborhood
    for dashboards.

    Parameters
    ----------
    fact : DataFrame
        FACT_HAZARD_EVENTS with spatial joins

    Returns
    -------
    DataFrame
        Aggregated metrics
    """
    logger.info("Aggregating hazard metrics...")

    # By neighborhood and hazard type
    metrics = fact.groupBy("neighborhood_sk", "hazard_type").agg(
        F.count("*").alias("event_count"),
        F.avg("severity").alias("avg_severity"),
        F.max("severity").alias("max_severity"),
        F.min("severity").alias("min_severity"),
    )

    return metrics


# =============================================================================
# WRITE GOLD DELTA TABLES
# =============================================================================


def write_gold_table(df: DataFrame, table_name: str, mode: str = "overwrite") -> None:
    """
    Write Gold DataFrame as Delta Table.

    Parameters
    ----------
    df : DataFrame
    table_name : str
    mode : str
    """
    path = f"s3a://{Config.GOLD_BUCKET}/{table_name}"

    try:
        df.write.format("delta").mode(mode).option(
            "compression", Config.DELTA_COMPRESSION
        ).save(path)
        logger.info(f"Wrote {table_name} to {path}")
    except Exception as e:
        logger.warning(f"Delta write failed ({e}), trying Parquet")
        df.write.format("parquet").mode(mode).save(f"/tmp/geoai/gold/{table_name}")


# =============================================================================
# GOLD LAYER RUNNER
# =============================================================================


def run_gold_dimensional() -> bool:
    """
    Run complete Gold layer dimensional modeling.

    Pipeline:
    1. Read Silver Delta Tables
    2. CreateDIM_NEIGHBORHOODS
    3. Create DIM_INFRASTRUCTURE
    4. Create FACT_HAZARD_EVENTS
    5. Spatial join events to neighborhoods (ST_Within)
    6. Spatial join events to nearest hospital (ST_Distance)
    7. Aggregate metrics
    8. Write Gold Delta Tables

    IMPORTANT: NO LLM calls in this file!
    All AI was already done in Silver.

    Returns
    -------
    bool
    """
    # Setup telemetry
    if TELEMETRY_AVAILABLE:
        setup_telemetry(service_name="gold-dimensional", environment="development")

    logger.info("Starting Gold dimensional modeling...")

    try:
        spark = create_spark_session(Config())

        # Create dimensions with tracing
        with traced_context("gold", "create_dim_neighborhoods"):
            dim_neighborhoods = create_dim_neighborhoods(spark)
            write_gold_table(dim_neighborhoods, "dim_neighborhoods")

        with traced_context("gold", "create_dim_infrastructure"):
            dim_infrastructure = create_dim_infrastructure(spark)
            write_gold_table(dim_infrastructure, "dim_infrastructure")

        # Create fact with tracing
        with traced_context("gold", "create_fact_hazard_events"):
            fact = create_fact_hazard_events(spark)

        # Spatial joins
        with traced_context("gold", "spatial_join_neighborhoods"):
            fact = spatial_join_events_to_neighborhoods(fact, dim_neighborhoods)

        with traced_context("gold", "spatial_join_infrastructure"):
            fact = spatial_join_events_to_nearest_infrastructure(fact, dim_infrastructure)

        # Aggregate and write
        with traced_context("gold", "aggregate_metrics"):
            metrics = aggregate_hazard_metrics(fact)
            write_gold_table(metrics, "agg_hazard_metrics")

        write_gold_table(fact, "fact_hazard_events")

        # Flush telemetry
        if TELEMETRY_AVAILABLE:
            flush_telemetry()

        logger.info("Gold dimensional modeling complete!")
        return True

    except Exception as e:
        logger.error(f"Gold modeling failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    success = run_gold_dimensional()
    sys.exit(0 if success else 1)
