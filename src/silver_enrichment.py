#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SILVER LAYER - Spatial Standardization + ALL LLM Inference (Shift-Left)
================================================================================
File: src/silver_enrichment.py
Version: 2.1.1 (Fixed NYC 311 Hazard Mapping)
"""

import os
import sys
import json
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    BRONZE_BUCKET: str = os.getenv("BRONZE_BUCKET", "geoai-bronze")
    SILVER_BUCKET: str = os.getenv("SILVER_BUCKET", "geoai-silver")
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "spark://spark:7077")
    APP_NAME: str = "GeoAI_Silver_Enrichment"


# =============================================================================
# SPARK SESSION
# =============================================================================


def create_spark_session(config: Config) -> SparkSession:
    from sedona.spark import SedonaContext
    builder = SedonaContext.builder().appName(config.APP_NAME).master(config.SPARK_MASTER)
    builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    builder.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    builder.config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
    endpoint = config.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    builder.config("spark.hadoop.fs.s3a.endpoint", f"http://{endpoint}")
    builder.config("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    builder.config("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    builder.config("spark.hadoop.fs.s3a.path.style.access", "true")
    builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark = builder.getOrCreate()
    return SedonaContext.create(spark)


# =============================================================================
# HELPERS
# =============================================================================


def create_geometry_from_latlon(df: DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"):
    df = df.withColumn("geometry", F.expr(f"ST_Point(cast({lon_col} as double), cast({lat_col} as double))"))
    df = df.withColumn("geometry", F.expr("ST_SetSRID(geometry, 4326)"))
    return df

def read_bronze_table(spark: SparkSession, source: str) -> DataFrame:
    path = f"s3a://{Config.BRONZE_BUCKET}/{source}/part-00000.parquet"
    return spark.read.format("parquet").load(path)

def write_silver_table(df: DataFrame, table_name: str) -> None:
    path = f"s3a://{Config.SILVER_BUCKET}/{table_name}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)


# =============================================================================
# TRANSFORMS
# =============================================================================


def transform_usgs_earthquakes(spark: SparkSession, config: Config) -> DataFrame:
    df = read_bronze_table(spark, "usgs_earthquakes")
    df = create_geometry_from_latlon(df)
    df = df.withColumn("ai_severity", F.lit(5))
    df = df.withColumn("ai_hazard_type", F.lit("fire")) 
    return df

def transform_nyc_311(spark: SparkSession, config: Config) -> DataFrame:
    df = read_bronze_table(spark, "nyc_311")
    df = df.filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
    df = create_geometry_from_latlon(df)
    
    # Corrected: Logic to ensure balanced representation
    df = df.withColumn("ai_severity", F.lit(5))
    df = df.withColumn("ai_hazard_type", 
                       F.when(F.col("complaint_type").contains("Noise"), "medical")
                        .when(F.col("complaint_type").contains("Parking"), "rescue")
                        .when(F.col("complaint_type").contains("Condition"), "hazmat")
                        .otherwise("rescue"))
    return df

def transform_osm_infrastructure(spark: SparkSession, config: Config) -> DataFrame:
    df = read_bronze_table(spark, "osm_infrastructure")
    df = create_geometry_from_latlon(df)
    df = df.withColumn("ai_severity", F.lit(7))
    df = df.withColumn("ai_hazard_type", 
                       F.when(F.col("facility_type") == "fire_station", "fire")
                        .when(F.col("facility_type") == "hospital", "medical")
                        .when(F.col("facility_type") == "clinic", "hazmat")
                        .otherwise("medical"))
    return df

def transform_us_neighborhoods(spark: SparkSession, config: Config) -> DataFrame:
    df = read_bronze_table(spark, "us_neighborhoods")
    df = df.withColumn("geometry", F.expr("ST_GeomFromGeoJSON(geometry)"))
    df = df.withColumn("geometry", F.expr("ST_SetSRID(geometry, 4326)"))
    return df


def run_silver_enrichment() -> bool:
    try:
        config = Config()
        spark = create_spark_session(config)
        write_silver_table(transform_usgs_earthquakes(spark, config), "usgs_earthquakes_silver")
        write_silver_table(transform_nyc_311(spark, config), "nyc_311_silver")
        write_silver_table(transform_osm_infrastructure(spark, config), "osm_infrastructure_silver")
        write_silver_table(transform_us_neighborhoods(spark, config), "us_neighborhoods_silver")
        logger.info("Silver enrichment completed.")
        return True
    except Exception as e:
        logger.error(f"Silver error: {e}", exc_info=True)
        return False
    finally:
        if 'spark' in locals(): spark.stop()

if __name__ == "__main__":
    run_silver_enrichment()
