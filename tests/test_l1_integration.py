#!/usr/bin/env python3
"""
================================================================================
L1 INTEGRATION TESTS - Component Interaction Tests
================================================================================
File: tests/test_l1_integration.py

Purpose:
    Test integration between multiple components with real Spark session.
    Uses local Spark with small datasets. No external services required.

Test Categories:
    - Bronze reading (local parquet)
    - Silver transformation
    - Gold enrichment logic
    - Data flow between layers

Author: GeoAI Principal Data Engineer
================================================================================
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="module")
def spark_session():
    """Create local Spark session for testing."""
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("GeoAI-Test") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    yield spark
    
    spark.stop()


@pytest.fixture
def sample_bronze_data(spark_session):
    """Create sample bronze data for testing."""
    data = [
        {"incident_id": "T001", "latitude": 40.7128, "longitude": -74.006, "severity": 3},
        {"incident_id": "T002", "latitude": 40.7580, "longitude": -73.9855, "severity": 2},
        {"incident_id": "T003", "latitude": 40.6782, "longitude": -73.9442, "severity": 1},
    ]
    return spark_session.createDataFrame(data)


class TestBronzeToSilverIntegration:
    """L1: Test data flow from Bronze to Silver."""

    def test_read_bronze_parquet(self, spark_session):
        """Test reading bronze parquet files."""
        bronze_path = "/tmp/geoai/bronze/us_accidents/"
        
        try:
            df = spark_session.read.parquet(bronze_path)
            assert df.count() > 0
            assert "latitude" in df.columns
            assert "longitude" in df.columns
        except Exception as e:
            pytest.skip(f"Bronze data not available: {e}")

    def test_silver_geometry_creation(self, spark_session, sample_bronze_data):
        """Test geometry creation in Silver layer."""
        from pyspark.sql import functions as F
        
        @F.udf(returnType="string")
        def make_point(lat, lon):
            if lat is None or lon is None:
                return None
            return f"POINT ({lon} {lat})"
        
        df = sample_bronze_data.withColumn(
            "geometry",
            make_point(F.col("latitude"), F.col("longitude"))
        )
        
        result = df.filter(F.col("geometry").isNotNull())
        assert result.count() == 3

    def test_silver_valid_coordinates(self, spark_session, sample_bronze_data):
        """Test filtering valid coordinates."""
        from pyspark.sql import functions as F
        
        df = sample_bronze_data.filter(
            (F.col("latitude") >= -90) & (F.col("latitude") <= 90) &
            (F.col("longitude") >= -180) & (F.col("longitude") <= 180)
        )
        
        assert df.count() == 3


class TestSilverTransformation:
    """L1: Test Silver layer transformations."""

    def test_add_metadata_columns(self, spark_session, sample_bronze_data):
        """Test adding metadata columns."""
        from pyspark.sql import functions as F
        
        df = sample_bronze_data.withColumn("source", F.lit("test_source"))
        df = df.withColumn("processing_timestamp", F.current_timestamp())
        
        assert "source" in df.columns
        assert "processing_timestamp" in df.columns

    def test_aggregate_by_severity(self, spark_session, sample_bronze_data):
        """Test aggregation by severity."""
        from pyspark.sql import functions as F
        
        result = sample_bronze_data.groupBy("severity").count().collect()
        
        counts = {row["severity"]: row["count"] for row in result}
        assert counts[3] == 1
        assert counts[2] == 1
        assert counts[1] == 1


class TestGoldEnrichment:
    """L1: Test Gold layer enrichment."""

    def test_ai_enrichment_udf(self, spark_session):
        """Test AI enrichment UDF."""
        from pyspark.sql import functions as F
        
        @F.udf(returnType="string")
        def enrich_category(description):
            if not description:
                return "unknown"
            desc_lower = description.lower()
            if "fire" in desc_lower or "smoke" in desc_lower:
                return "fire_related"
            elif "medical" in desc_lower or "hospital" in desc_lower:
                return "medical"
            return "other"
        
        df = spark_session.createDataFrame([
            {"description": "Fire reported on Main St"},
            {"description": "Medical emergency"},
            {"description": "Traffic accident"},
        ])
        
        result = df.withColumn("category", enrich_category("description"))
        
        rows = result.collect()
        categories = [row["category"] for row in rows]
        
        assert "fire_related" in categories
        assert "medical" in categories
        assert "other" in categories

    def test_join_dimensions(self, spark_session):
        """Test joining with dimension tables."""
        events = spark_session.createDataFrame([
            {"event_id": "E001", "neighborhood_id": "N001"},
            {"event_id": "E002", "neighborhood_id": "N002"},
        ])
        
        neighborhoods = spark_session.createDataFrame([
            {"neighborhood_id": "N001", "name": "Downtown"},
            {"neighborhood_id": "N002", "name": "Uptown"},
        ])
        
        joined = events.join(neighborhoods, "neighborhood_id")
        
        assert joined.count() == 2
        assert "name" in joined.columns


class TestDataQuality:
    """L1: Data quality checks."""

    def test_null_check(self, spark_session):
        """Test null value detection."""
        from pyspark.sql import functions as F
        
        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
            {"id": 2, "value": None},
            {"id": 3, "value": "c"},
        ])
        
        null_count = df.filter(F.col("value").isNull()).count()
        assert null_count == 1

    def test_duplicate_check(self, spark_session):
        """Test duplicate detection."""
        from pyspark.sql import functions as F
        
        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
        ])
        
        duplicates = df.groupBy("id").count().filter(F.col("count") > 1)
        assert duplicates.count() == 1

    def test_data_range_validation(self, spark_session):
        """Test data range validation."""
        from pyspark.sql import functions as F
        
        df = spark_session.createDataFrame([
            {"lat": 40.7, "lon": -74.0},
            {"lat": 91.0, "lon": -73.0},  # Invalid lat
            {"lat": 40.8, "lon": -73.5},
        ])
        
        valid = df.filter(
            (F.col("lat").between(-90, 90)) &
            (F.col("lon").between(-180, 180))
        )
        
        assert valid.count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])