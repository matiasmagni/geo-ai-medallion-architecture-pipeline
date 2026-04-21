#!/usr/bin/env python3
"""
Simplified Gold Pipeline - No Sedona Plugin
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import re
import json

# Config
LOCAL_SILVER_PATH = "/workspace/data/silver"
LOCAL_GOLD_PATH = "/workspace/data/gold"
OLLAMA_BASE_URL = "http://172.19.0.2:11434"
OLLAMA_MODEL = "llama3"


def create_spark():
    conf = SparkConf()
    conf.setAppName("GeoAI_Gold")
    conf.setMaster("local[*]")
    conf.set("spark.driver.memory", "2g")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def create_ollama_udf():
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    import http.client

    def analyze_hazard(text):
        if not text or pd.isna(text):
            return json.dumps({"severity": 5, "hazard_type": "unknown"})
        try:
            url = OLLAMA_BASE_URL.replace("http://", "").split(":")
            host = url[0]
            port = int(url[1]) if len(url) > 1 else 11434

            prompt = f"""Analyze this hazard description and return ONLY a JSON:
{{"severity": 1-10, "hazard_type": "traffic|weather|fire|medical|infrastructure|crime|other"}}
Description: {str(text)[:300]}
JSON:"""

            payload = json.dumps(
                {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 30},
                }
            )

            conn = http.client.HTTPConnection(host, port, timeout=30)
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/api/generate", payload, headers)
            response = conn.getresponse()

            if response.status == 200:
                result = json.loads(response.read().decode())
                response_text = result.get("response", "")
                match = re.search(r"\{.*\}", response_text) if "re" in dir() else None
                if match:
                    try:
                        return json.loads(match.group())
                    except:
                        pass
            conn.close()
        except Exception as e:
            logger.warning(f"Ollama failed: {e}")
        return json.dumps({"severity": 5, "hazard_type": "unknown"})

    @pandas_udf("string", PandasUDFType.SCALAR)
    def ollama_hazard_udf(s: pd.Series) -> pd.Series:
        return s.apply(analyze_hazard)

    return ollama_hazard_udf


def main():
    logger.info("=" * 60)
    logger.info("STARTING GOLD PIPELINE (Simplified)")
    logger.info("=" * 60)

    spark = create_spark()

    try:
        # Read Silver
        logger.info("Reading Silver layer...")
        silver_df = spark.read.parquet(f"{LOCAL_SILVER_PATH}/usgs_earthquakes")
        logger.info(f"Loaded {silver_df.count()} records from Silver")

        # Create FACT table
        logger.info("Creating FACT_HAZARD_EVENTS...")
        fact_df = silver_df.withColumn("source_system", F.lit("usgs_earthquakes"))
        fact_df = fact_df.withColumn("event_id", F.col("source_id"))
        fact_df = fact_df.withColumn("event_time", F.col("event_timestamp"))
        fact_df = fact_df.withColumn("event_uuid", F.expr("uuid()"))
        fact_df = fact_df.withColumn("ai_enrichment_timestamp", F.current_timestamp())
        fact_df = fact_df.withColumn("ai_enrichment_model", F.lit(OLLAMA_MODEL))

        # Get sample for AI enrichment (to avoid calling Ollama 3320 times)
        logger.info("Applying AI enrichment to sample...")
        sample_df = fact_df.filter(F.col("description").isNotNull()).limit(5)

        # Apply UDF
        try:
            ollama_udf = create_ollama_udf()
            sample_enriched = sample_df.withColumn(
                "ai_enrichment", ollama_udf(F.col("description"))
            )

            # Parse JSON
            json_schema = "severity INT, hazard_type STRING"
            sample_enriched = sample_enriched.withColumn(
                "parsed", F.from_json(F.col("ai_enrichment"), json_schema)
            )
            sample_enriched = sample_enriched.select(
                "*",
                F.col("parsed.severity").alias("ai_severity"),
                F.col("parsed.hazard_type").alias("ai_hazard_type"),
            )

            logger.info("AI enrichment results:")
            sample_enriched.select("description", "ai_severity", "ai_hazard_type").show(
                5, truncate=50
            )
        except Exception as e:
            logger.warning(f"AI enrichment failed: {e}")

        # Write Gold
        logger.info("Writing Gold layer...")
        fact_df.write.mode("overwrite").parquet(f"{LOCAL_GOLD_PATH}/fact_hazard_events")

        logger.info(f"Gold FACT table written: {fact_df.count()} records")

        # Verify
        gold_df = spark.read.parquet(f"{LOCAL_GOLD_PATH}/fact_hazard_events")
        logger.info(f"Verification: {gold_df.count()} records in Gold")
        logger.info("Columns: " + str(gold_df.columns))

        logger.info("=" * 60)
        logger.info("GOLD PIPELINE COMPLETE!")
        logger.info("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
