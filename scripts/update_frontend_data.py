#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FRONTEND DATA EXPORTER - Gold Layer to Leaflet GeoJSON
================================================================================
File: scripts/update_frontend_data.py
Version: 1.0.2 (Added Hazmat categorization)
"""

import os
import sys
import json
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Config:
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    GOLD_BUCKET: str = os.getenv("GOLD_BUCKET", "geoai-gold")
    OUTPUT_PATH: str = os.getenv("GEOJSON_OUTPUT", "/home/jovyan/geo-ai-heatmap/public/data/heatmap.geojson")
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "spark://spark:7077")

def create_spark_session(config: Config) -> SparkSession:
    from sedona.spark import SedonaContext
    builder = SedonaContext.builder().appName("GeoAI_Frontend_Exporter").master(config.SPARK_MASTER)
    builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    builder.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    endpoint = config.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    builder.config("spark.hadoop.fs.s3a.endpoint", f"http://{endpoint}")
    builder.config("spark.hadoop.fs.s3a.access.key", config.MINIO_ACCESS_KEY)
    builder.config("spark.hadoop.fs.s3a.secret.key", config.MINIO_SECRET_KEY)
    builder.config("spark.hadoop.fs.s3a.path.style.access", "true")
    builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark = builder.getOrCreate()
    return SedonaContext.create(spark)

def run_export():
    config = Config()
    spark = create_spark_session(config)
    try:
        fact_path = f"s3a://{config.GOLD_BUCKET}/fact_hazard_events_parquet"
        df = spark.read.format("parquet").load(fact_path)
        
        df = df.withColumn("longitude", F.expr("ST_X(event_geometry)")) \
               .withColumn("latitude", F.expr("ST_Y(event_geometry)"))
        
        data_rows = df.select("event_id", "event_description", "hazard_type", "severity", "latitude", "longitude").collect()
        
        features = []
        for row in data_rows:
            cat = str(row.hazard_type).lower()
            # Explicit Hazmat classification
            if any(k in cat for k in ['hazmat', 'chemical', 'spill', 'asbestos']):
                category = 'hazmat'
            elif cat in ['fire', 'fire_station']:
                category = 'fire'
            elif cat in ['medical', 'hospital', 'clinic']:
                category = 'medical'
            elif cat in ['rescue', 'safety', 'emergency']:
                category = 'rescue'
            else:
                category = 'rescue' if row.severity > 7 else 'medical'
                
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row.longitude, row.latitude]},
                "properties": {
                    "id": row.event_id,
                    "category": category,
                    "label": category.capitalize(),
                    "weight": row.severity / 10.0,
                    "description": row.event_description
                }
            })
        
        output_file = Path(config.OUTPUT_PATH)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        logger.info(f"Successfully exported {len(features)} features to {config.OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
    finally:
        spark.stop()

if __name__ == "__main__":
    run_export()
