#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GOLD LAYER - STAR SCHEMA + AI ENRICHMENT + SPATIAL RELATIONSHIPS
================================================================================
File: src/gold_schema_and_ai_enrichment.py

Purpose:
    - Star Schema Design with Fact and Dimension tables
    - AI Enrichment using Ollama (llama3) for severity/hazard_type extraction
    - Spatial joins using Sedona (ST_Within, ST_Distance)
    - Gold Delta Tables output

Key Components:
    FACT_HAZARD_EVENTS: Unified event table from all hazard sources
    DIM_NEIGHBORHOODS: Geographic area boundaries
    DIM_INFRASTRUCTURE: Hospitals and fire stations

Author: GeoAI Principal Data Engineer
Version: 1.0.0
================================================================================
"""

import os
import sys
import json
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
    """Configuration for Gold layer."""

    # MinIO/S3
    MINIO_ENDPOINT = os.getenv("S3_ENDPOINT", "http://host.docker.internal:9900")
    MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    BUCKET_SILVER = "geo-lakehouse/silver"
    BUCKET_GOLD = "geo-lakehouse/gold"

    # Local paths for development
    LOCAL_DATA_PATH = "/tmp/geoai"
    LOCAL_SILVER_PATH = os.getenv("LOCAL_SILVER_PATH", "/tmp/geoai/silver")
    LOCAL_GOLD_PATH = os.getenv("LOCAL_GOLD_PATH", "/tmp/geoai/gold")

    # Spark
    SPARK_MASTER = os.getenv("SPARK_MASTER_URL", "local[*]")
    APP_NAME = "GeoAI_Gold_Schema_AI_Enrichment"

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_TIMEOUT = 60


# =============================================================================
# SPARK SESSION
# =============================================================================


def create_spark_session(config: Config):
    """Create Spark session without Sedona (using local PySpark with Shapely)."""
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    logger.info("Creating Spark session for Gold layer...")

    conf = SparkConf()
    conf.setAppName(config.APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # Memory
    conf.set("spark.driver.memory", "2g")
    conf.set("spark.executor.memory", "2g")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


# =============================================================================
# OLLAMA AI ENRICHMENT UDF
# =============================================================================


def create_ollama_enrichment_udf(config: Config):
    """
    Create PySpark Pandas UDF that calls Ollama API for hazard analysis.

    Returns a UDF that extracts: {"severity": 1-10, "hazard_type": "string"}
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    import http.client
    import re
    import json

    def analyze_hazard(text: str) -> Optional[str]:
        """Call Ollama to analyze hazard description."""
        if not text or pd.isna(text):
            return json.dumps({"severity": 5, "hazard_type": "unknown"})

        try:
            # Parse Ollama endpoint
            url = config.OLLAMA_BASE_URL.replace("http://", "").split(":")
            host = url[0]
            port = int(url[1]) if len(url) > 1 else 11434

            # Build prompt
            prompt = f"""Analyze this hazard/incident description and extract:
Return ONLY a JSON object with:
{{"severity": 1-10, "hazard_type": "string"}}

Where:
- severity: 1=minor to 10=critical
- hazard_type: one of [traffic, weather, fire, medical, infrastructure, crime, other]

Description: {text[:500]}

Response:"""

            payload = json.dumps(
                {
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 50},
                }
            )

            conn = http.client.HTTPConnection(host, port, timeout=config.OLLAMA_TIMEOUT)
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/api/generate", payload, headers)
            response = conn.getresponse()

            if response.status == 200:
                result = json.loads(response.read().decode())
                response_text = result.get("response", "")

                # Parse JSON from response
                try:
                    # Try to find JSON in response
                    match = re.search(r"\{.*\}", response_text)
                    if match:
                        parsed = json.loads(match.group())
                        return json.dumps(parsed)
                except:
                    pass

            conn.close()
            return json.dumps({"severity": 5, "hazard_type": "unknown"})

        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return json.dumps({"severity": 5, "hazard_type": "unknown"})

    @pandas_udf("string", PandasUDFType.SCALAR)
    def ollama_hazard_udf(text_series: pd.Series) -> pd.Series:
        """Pandas UDF for hazard analysis."""
        return text_series.apply(analyze_hazard)

    return ollama_hazard_udf


def parse_ai_enrichment(json_str: str) -> Dict[str, Any]:
    """Parse AI enrichment JSON string into columns."""
    try:
        return (
            json.loads(json_str)
            if json_str
            else {"severity": 5, "hazard_type": "unknown"}
        )
    except:
        return {"severity": 5, "hazard_type": "unknown"}


# =============================================================================
# READ SILVER TABLES
# =============================================================================


def read_silver_table(
    spark: "SparkSession", config: Config, table_name: str
) -> "DataFrame":
    """Read Delta from Silver layer (local path)."""
    path = f"{config.LOCAL_SILVER_PATH}/{table_name}"
    logger.info(f"Reading Silver table: {path}")
    try:
        return spark.read.format("delta").load(path)
    except:
        return spark.read.parquet(path)


def read_all_sources(spark: "SparkSession", config: Config) -> Dict[str, "DataFrame"]:
    """Read all Silver tables into dictionary."""
    sources = {}

    # Only try tables that exist
    tables = ["us_accidents", "usgs_earthquakes", "osm_infrastructure", "nyc_311_requests", "neighborhoods"]

    for table in tables:
        try:
            df = read_silver_table(spark, config, table)
            sources[table] = df
            logger.info(f"Loaded {table}: {df.count()} records")
        except Exception as e:
            logger.warning(f"Failed to load {table}: {e}")

    return sources


# =============================================================================
# FACT TABLE CREATION
# =============================================================================


def create_fact_hazard_events(
    spark: "SparkSession", config: Config, sources: Dict[str, "DataFrame"]
) -> "DataFrame":
    """
    Create unified FACT_HAZARD_EVENTS table.

    Parameters:
        spark: SparkSession
        config: Configuration
        sources: Dictionary of source DataFrames

    Returns:
        Unified fact table
    """
    from pyspark.sql import functions as F

    logger.info("Creating FACT_HAZARD_EVENTS...")

    all_events = []

    # Process each source
    for source_name, df in sources.items():
        if df is None or source_name in ["neighborhoods", "osm_infrastructure"]:
            continue

        # Ensure required columns exist
        df = df.withColumn("source_system", F.lit(source_name))

        # Handle description (required for AI enrichment)
        if "description" not in df.columns:
            df = df.withColumn("description", F.lit(""))

        # Handle severity (if not present, default to 5)
        if "severity" not in df.columns:
            df = df.withColumn("severity", F.lit(5))

        # Ensure geometry column exists, if not create from lat/lon
        if "geometry" not in df.columns and "latitude" in df.columns and "longitude" in df.columns:
            df = df.withColumn("geometry", F.expr("ST_Point(cast(longitude as double), cast(latitude as double))"))
            df = df.withColumn("geometry", F.expr("ST_SetSRID(geometry, 4326)"))

        # Select columns for unified schema
        cols = [
            F.col("source_system"),
            F.col("source_id").alias("event_id"),
            F.col("event_timestamp").alias("event_time"),
            F.col("description"),
            F.col("severity"),
            F.col("geometry"),
            F.col("processing_timestamp"),
        ]
        
        # Only select columns that exist
        final_cols = [c for c in cols if c._jc.toString().split(" AS ")[0].split(".")[-1].strip("`") in df.columns or "ST_Point" in c._jc.toString()]
        df = df.select(*cols)

        all_events.append(df)

    # UNION all sources
    if all_events:
        # Get first dataframe as base
        fact_df = all_events[0]

        # UNION with others
        for df in all_events[1:]:
            fact_df = fact_df.unionByName(df, allowMissingColumns=True)

        # Add unique event ID
        fact_df = fact_df.withColumn("event_uuid", F.expr("uuid()"))

        # Add AI enrichment timestamp
        fact_df = fact_df.withColumn("ai_enrichment_timestamp", F.current_timestamp())
        fact_df = fact_df.withColumn("ai_enrichment_model", F.lit(config.OLLAMA_MODEL))

        logger.info(f"FACT_HAZARD_EVENTS created: {fact_df.count()} records")
        return fact_df

    return None


def ai_enrich_fact_table(
    spark: "SparkSession", config: Config, fact_df: "DataFrame"
) -> "DataFrame":
    """
    Apply AI enrichment to FACT_HAZARD_EVENTS.

    Uses Ollama to extract severity and hazard_type from descriptions.
    """
    from pyspark.sql import functions as F

    if fact_df is None:
        return None

    logger.info("Applying AI enrichment to FACT_HAZARD_EVENTS...")

    # Create Ollama UDF
    ollama_udf = create_ollama_enrichment_udf(config)

    # Apply to descriptions
    # Note: Filter to non-null descriptions
    df_with_ai = fact_df.filter(F.col("description").isNotNull() & (F.col("description") != ""))
    df_without_ai = fact_df.filter(F.col("description").isNull() | (F.col("description") == ""))

    if df_with_ai.count() > 0:
        # Apply AI enrichment
        df_enriched = df_with_ai.withColumn(
            "ai_enrichment_json", ollama_udf(F.col("description"))
        )

        # Parse JSON into columns
        json_schema = "severity INT, hazard_type STRING"
        df_enriched = df_enriched.withColumn(
            "parsed_enrichment", F.from_json(F.col("ai_enrichment_json"), json_schema)
        )

        df_enriched = df_enriched.select(
            "*",
            F.col("parsed_enrichment.severity").alias("ai_severity"),
            F.col("parsed_enrichment.hazard_type").alias("ai_hazard_type"),
        )
        
        # Handle rows without descriptions (use default)
        df_final = df_enriched.unionByName(df_without_ai, allowMissingColumns=True)
    else:
        df_final = df_without_ai
        df_final = df_final.withColumn("ai_severity", F.lit(None).cast("int"))
        df_final = df_final.withColumn("ai_hazard_type", F.lit(None).cast("string"))

    # Recalculate severity: AI takes priority
    df_final = df_final.withColumn(
        "final_severity", F.coalesce(F.col("ai_severity"), F.col("severity"))
    )

    # Drop temporary columns
    df_final = df_final.drop("ai_enrichment_json", "parsed_enrichment")

    count = df_final.count()
    logger.info(f"AI enrichment complete: {count} events enriched")

    return df_final


# =============================================================================
# DIMENSION TABLES
# =============================================================================


def create_dim_neighborhoods(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Create DIM_NEIGHBORHOODS dimension table."""
    from pyspark.sql import functions as F

    logger.info("Creating DIM_NEIGHBORHOODS...")

    try:
        df = read_silver_table(spark, config, "neighborhoods")

        # Add surrogate key
        df = df.withColumn("neighborhood_id", F.expr("uuid()"))
        
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if name_col:
            df = df.withColumn("neighborhood_name", F.col(name_col))
        else:
            df = df.withColumn("neighborhood_name", F.lit("Unknown"))

        # Select key columns
        df = df.select(
            "neighborhood_id",
            "neighborhood_name",
            "geometry",
            "source",
            "processing_timestamp",
        )

        logger.info(f"DIM_NEIGHBORHOODS created: {df.count()} neighborhoods")
        return df

    except Exception as e:
        logger.error(f"Failed to create DIM_NEIGHBORHOODS: {e}")
        return None


def create_dim_infrastructure(
    spark: "SparkSession", config: Config
) -> Optional["DataFrame"]:
    """Create DIM_INFRASTRUCTURE dimension table."""
    from pyspark.sql import functions as F

    logger.info("Creating DIM_INFRASTRUCTURE...")

    try:
        df = read_silver_table(spark, config, "osm_infrastructure")

        # Add surrogate key
        df = df.withColumn("infrastructure_id", F.expr("uuid()"))

        # Standardize facility type
        df = df.withColumn(
            "facility_category",
            F.when(F.col("facility_type") == "hospital", "hospital")
            .when(F.col("facility_type") == "fire_station", "fire_station")
            .otherwise("other"),
        )

        # Select key columns
        df = df.select(
            "infrastructure_id",
            "facility_category",
            "facility_name",
            "geometry",
            "source",
            "processing_timestamp",
        )

        logger.info(f"DIM_INFRASTRUCTURE created: {df.count()} facilities")
        return df

    except Exception as e:
        logger.error(f"Failed to create DIM_INFRASTRUCTURE: {e}")
        return None


# =============================================================================
# SPATIAL RELATIONSHIPS (JOINS)
# =============================================================================


def spatial_join_events_to_neighborhoods(
    fact_df: "DataFrame", dim_neighborhoods: "DataFrame"
) -> "DataFrame":
    """
    Perform join to find which neighborhood each event is in.
    Uses simple join (Sedona ST_Within not available).
    """
    from pyspark.sql import functions as F

    logger.info("Join to neighborhoods (simple join, Sedona not available)...")

    # Without Sedona, just return the fact table with a placeholder neighborhood
    # In production, you'd use Shapely for point-in-polygon
    result = fact_df.withColumn("neighborhood_id", F.lit(None).cast("string"))
    result = result.withColumn("neighborhood_name", F.lit(None).cast("string"))

    return result


def spatial_join_events_to_nearest_infrastructure(
    fact_df: "DataFrame", dim_infrastructure: "DataFrame"
) -> "DataFrame":
    """
    Calculate distance to nearest infrastructure using simple lat/lon.
    """
    from pyspark.sql import functions as F

    logger.info("Calculating nearest infrastructure join (simple join)...")

    # Without Sedona, add placeholder distance columns
    result = fact_df.withColumn("nearest_hospital_dist_m", F.lit(None).cast("double"))
    result = result.withColumn("nearest_firestation_dist_m", F.lit(None).cast("double"))

    return result


# =============================================================================
# WRITE GOLD TABLES
# =============================================================================


def write_gold_table(
    df: "DataFrame", config: Config, table_name: str, mode: str = "overwrite"
) -> None:
    """Write DataFrame to Gold layer as Delta Table."""
    import os
    os.makedirs(config.LOCAL_GOLD_PATH, exist_ok=True)
    path = f"{config.LOCAL_GOLD_PATH}/{table_name}"
    logger.info(f"Writing Gold Delta table: {path}")

    try:
        df.write.format("delta").mode(mode).option("compression", "snappy").save(path)
    except Exception as e:
        logger.warning(f"Delta write failed: {e}, falling back to Parquet")
        df.write.format("parquet").mode(mode).option("compression", "snappy").save(path)

    logger.info(f"Gold table {table_name} written: {df.count()} records")


# =============================================================================
# MAIN GOLD PIPELINE
# =============================================================================


def run_gold_pipeline():
    """Execute complete Gold layer pipeline."""
    global spark
    config = Config()

    logger.info("=" * 60)
    logger.info("STARTING GOLD LAYER PIPELINE")
    logger.info("=" * 60)

    try:
        # Create Spark session
        spark = create_spark_session(config)

        # =========================================================================
        # STEP 1: Read Silver sources
        # =========================================================================
        logger.info("[STEP 1] Reading Silver sources...")
        sources = read_all_sources(spark, config)

        # =========================================================================
        # STEP 2: Create FACT_HAZARD_EVENTS
        # =========================================================================
        logger.info("[STEP 2] Creating FACT_HAZARD_EVENTS...")
        fact_df = create_fact_hazard_events(spark, config, sources)

        # =========================================================================
        # STEP 3: AI Enrichment (Ollama)
        # =========================================================================
        logger.info("[STEP 3] Applying AI enrichment (Ollama)...")
        fact_df = ai_enrich_fact_table(spark, config, fact_df)

        # =========================================================================
        # STEP 4: Spatial Join to Neighborhoods
        # =========================================================================
        logger.info("[STEP 4] Spatial join to DIM_NEIGHBORHOODS...")
        dim_neighborhoods = create_dim_neighborhoods(spark, config)
        if dim_neighborhoods and fact_df:
            fact_df = spatial_join_events_to_neighborhoods(fact_df, dim_neighborhoods)

        # =========================================================================
        # STEP 5: Spatial Distance to Infrastructure
        # =========================================================================
        logger.info("[STEP 5] Calculating distance to infrastructure...")
        dim_infrastructure = create_dim_infrastructure(spark, config)
        if dim_infrastructure and fact_df:
            fact_df = spatial_join_events_to_nearest_infrastructure(
                fact_df, dim_infrastructure
            )

        # =========================================================================
        # STEP 6: Write Gold Tables
        # =========================================================================
        logger.info("[STEP 6] Writing Gold tables...")

        # Write FACT_HAZARD_EVENTS
        if fact_df:
            write_gold_table(fact_df, config, "fact_hazard_events")

        # Write DIM_NEIGHBORHOODS
        if dim_neighborhoods:
            write_gold_table(dim_neighborhoods, config, "dim_neighborhoods")

        # Write DIM_INFRASTRUCTURE
        if dim_infrastructure:
            write_gold_table(dim_infrastructure, config, "dim_infrastructure")

        # =========================================================================
        # COMPLETE
        # =========================================================================
        logger.info("=" * 60)
        logger.info("GOLD LAYER PIPELINE COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Gold pipeline failed: {e}", exc_info=True)
        raise

    finally:
        if "spark" in dir():
            spark.stop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_gold_pipeline()
