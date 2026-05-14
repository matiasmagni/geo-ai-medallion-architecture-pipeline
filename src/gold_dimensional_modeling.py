#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GOLD LAYER - Kimball Star Schema Dimensional Modeling
================================================================================
File: src/gold_dimensional_modeling.py
Version: 2.0.6 (Fixed nyc_311 column mapping)
"""

import os
import sys
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Config:
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    SILVER_BUCKET: str = os.getenv("SILVER_BUCKET", "geo-lakehouse/silver")
    GOLD_BUCKET: str = os.getenv("GOLD_BUCKET", "geo-lakehouse/gold")
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "spark://spark:7077")
    APP_NAME: str = "GeoAI_Gold_Dimensional"
    DELTA_COMPRESSION: str = "snappy"

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

def read_silver_table(spark: SparkSession, table: str) -> DataFrame:
    path = f"s3a://{Config.SILVER_BUCKET}/{table}"
    logger.info(f"Reading {table}")
    return spark.read.format("delta").load(path)


def create_dim_neighborhoods(spark: SparkSession) -> DataFrame:
    df = read_silver_table(spark, "us_neighborhoods_silver")
    return df.select(
        F.col("geoid").alias("neighborhood_id"),
        F.col("name").alias("neighborhood_name"),
        F.col("geometry").alias("neighborhood_geometry"),
        F.col("state_fips").alias("state_fips"),
        F.col("county_fips").alias("county_fips")
    )


def create_dim_infrastructure(spark: SparkSession) -> DataFrame:
    df = read_silver_table(spark, "osm_infrastructure_silver")
    return df.select(
        F.col("osm_id").cast("string").alias("infrastructure_id"),
        F.col("name").alias("infrastructure_name"),
        F.col("geometry").alias("infrastructure_geometry"),
        F.col("facility_type").alias("infrastructure_type")
    )


def create_fact_hazard_events(spark: SparkSession) -> DataFrame:
    df_usgs = read_silver_table(spark, "usgs_earthquakes_silver").select(
        F.col("earthquake_id").alias("event_id"),
        F.col("place").alias("event_description"),
        F.col("geometry").alias("event_geometry"),
        F.col("ai_severity").alias("severity"),
        F.col("ai_hazard_type").alias("hazard_type"),
        F.col("time").alias("event_time")
    )
    df_311 = read_silver_table(spark, "nyc_311_silver").select(
        F.col("request_id").alias("event_id"),
        F.col("complaint_type").alias("event_description"),
        F.col("geometry").alias("event_geometry"),
        F.col("ai_severity").alias("severity"),
        F.col("ai_hazard_type").alias("hazard_type"),
        F.col("created_date").alias("event_time")
    )
    df_osm = read_silver_table(spark, "osm_infrastructure_silver").select(
        F.col("osm_id").cast("string").alias("event_id"),
        F.col("name").alias("event_description"),
        F.col("geometry").alias("event_geometry"),
        F.col("ai_severity").alias("severity"),
        F.col("ai_hazard_type").alias("hazard_type"),
        F.current_timestamp().alias("event_time")
    )
    return df_usgs.unionByName(df_311).unionByName(df_osm)


def aggregate_hazard_metrics(spark: SparkSession) -> DataFrame:
    fact = create_fact_hazard_events(spark)
    return fact.groupBy("hazard_type").agg(
        F.count("*").alias("event_count"),
        F.avg("severity").alias("avg_severity"),
        F.max("severity").alias("max_severity")
    )


def spatial_join_events_to_neighborhoods(spark: SparkSession, events_df: DataFrame, neighborhoods_df: DataFrame) -> DataFrame:
    from sedona.sql.functions import ST_Contains
    return events_df.alias("e").crossJoin(
        neighborhoods_df.alias("n")
    ).filter(
        ST_Contains(F.col("n.neighborhood_geometry"), F.col("e.event_geometry"))
    ).select(
        F.col("e.*"),
        F.col("n.neighborhood_id"),
        F.col("n.neighborhood_name")
    )


def spatial_join_events_to_nearest_infrastructure(spark: SparkSession, events_df: DataFrame, infra_df: DataFrame) -> DataFrame:
    from sedona.sql.functions import ST_Distance, ST_Buffer
    buffered = infra_df.withColumn("buffer", ST_Buffer(F.col("infrastructure_geometry"), 0.01))
    return events_df.alias("e").crossJoin(
        buffered.alias("i")
    ).filter(
        ST_Contains(F.col("i.buffer"), F.col("e.event_geometry"))
    ).select(
        F.col("e.*"),
        F.col("i.infrastructure_id"),
        F.col("i.infrastructure_name"),
        F.col("i.infrastructure_type")
    )

def write_gold_table(df: DataFrame, table_name: str) -> None:
    path = f"s3a://{Config.GOLD_BUCKET}/{table_name}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)


def run_gold_dimensional() -> bool:
    logger.info("Starting Gold dimensional modeling...")
    try:
        config = Config()
        spark = create_spark_session(config)

        # 1. USGS
        df_usgs = read_silver_table(spark, "usgs_earthquakes_silver").select(
            F.col("earthquake_id").alias("event_id"),
            F.col("place").alias("event_description"),
            F.col("geometry").alias("event_geometry"),
            F.col("ai_severity").alias("severity"),
            F.col("ai_hazard_type").alias("hazard_type"),
            F.col("time").alias("event_time")
        )

        # 2. NYC 311 (Fixed request_id mapping)
        df_311 = read_silver_table(spark, "nyc_311_silver").select(
            F.col("request_id").alias("event_id"),
            F.col("complaint_type").alias("event_description"),
            F.col("geometry").alias("event_geometry"),
            F.col("ai_severity").alias("severity"),
            F.col("ai_hazard_type").alias("hazard_type"),
            F.col("created_date").alias("event_time")
        )

        # 3. OSM
        df_osm = read_silver_table(spark, "osm_infrastructure_silver").select(
            F.col("osm_id").cast("string").alias("event_id"),
            F.col("name").alias("event_description"),
            F.col("geometry").alias("event_geometry"),
            F.col("ai_severity").alias("severity"),
            F.col("ai_hazard_type").alias("hazard_type"),
            F.current_timestamp().alias("event_time")
        )

        # UNION ALL
        fact = df_usgs.unionByName(df_311).unionByName(df_osm)
        
        # Save as Parquet for heatmap exporter
        gold_path = f"s3a://{config.GOLD_BUCKET}/fact_hazard_events_parquet"
        fact.write.format("parquet").mode("overwrite").save(gold_path)
        
        logger.info(f"Gold complete! Total facts: {fact.count()}")
        return True
    except Exception as e:
        logger.error(f"Gold failed: {e}", exc_info=True)
        return False
    finally:
        if 'spark' in locals(): spark.stop()

if __name__ == "__main__":
    run_gold_dimensional()
