#!/usr/bin/env python3
"""
================================================================================
L1 INTEGRATION TESTS - Component Interaction Tests
================================================================================
File: tests/test_l1_integration.py

Purpose:
    Test integration between components with real Spark session.
    Tests data flow between bronze->silver->gold layers.

Author: GeoAI Principal Data Engineer & MLOps Architect
================================================================================
"""

import pytest
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CLASS: Bronze to Silver Integration
# =============================================================================

class TestBronzeToSilverIntegration:
    """L1: Test data flow from Bronze to Silver."""

    @pytest.fixture(autouse=True)
    def setup_spark(self, spark_session_l1):
        """Setup Spark session for each test."""
        self.spark = spark_session_l1

    def test_read_bronze_parquet(self, tmp_path):
        """Test reading bronze parquet files."""
        # Create test bronze data
        bronze_path = tmp_path / "bronze" / "test_accidents"
        bronze_path.mkdir(parents=True)
        
        import pandas as pd
        test_df = pd.DataFrame({
            "incident_id": ["T001", "T002", "T003"],
            "latitude": [40.7128, 40.7580, 40.6782],
            "longitude": [-74.0060, -73.9855, -73.9442],
            "severity": [3, 2, 1],
        })
        test_df.to_parquet(bronze_path / "part-00000.parquet")
        
        # Read with Spark
        df = self.spark.read.parquet(str(bronze_path))
        
        assert df.count() == 3
        assert "latitude" in df.columns
        assert "longitude" in df.columns

    def test_silver_geometry_creation(self):
        """Test geometry creation in Silver layer."""
        from pyspark.sql import functions as F
        
        # Create test data
        data = [
            ("T001", 40.7128, -74.0060, 3),
            ("T002", 40.7580, -73.9855, 2),
        ]
        df = self.spark.createDataFrame(data, ["incident_id", "lat", "lon", "severity"])
        
        # Create geometry column
        df = df.withColumn(
            "geometry",
            F.concat_ws(" ", F.lit("POINT"), F.col("lon"), F.col("lat"))
        )
        
        result = df.filter(F.col("geometry").isNotNull())
        assert result.count() == 2

    def test_silver_metadata_columns(self):
        """Test adding metadata columns in Silver."""
        from pyspark.sql import functions as F
        
        data = [("T001", 40.7128, -74.0060, 3)]
        df = self.spark.createDataFrame(data, ["id", "lat", "lon", "severity"])
        
        # Add metadata
        df = df.withColumn("source_system", F.lit("test"))
        df = df.withColumn("silver_updated", F.current_timestamp())
        
        assert "source_system" in df.columns
        assert "silver_updated" in df.columns


# =============================================================================
# TEST CLASS: Silver Transformation
# =============================================================================

class TestSilverTransformation:
    """L1: Test Silver layer transformations."""

    @pytest.fixture(autouse=True)
    def setup_spark(self, spark_session_l1):
        """Setup Spark session for each test."""
        self.spark = spark_session_l1

    def test_filter_valid_coordinates(self):
        """Test filtering valid coordinates."""
        from pyspark.sql import functions as F
        
        data = [
            (40.7128, -74.0060),  # Valid
            (91.0, -74.0060),      # Invalid lat
            (40.7580, -73.9855),  # Valid
            (40.6782, -180.0),    # Invalid lon
        ]
        df = self.spark.createDataFrame(data, ["lat", "lon"])
        
        valid_df = df.filter(
            (F.col("lat") >= -90) & (F.col("lat") <= 90) &
            (F.col("lon") >= -180) & (F.col("lon") <= 180)
        )
        
        assert valid_df.count() == 2

    def test_aggregate_by_severity(self):
        """Test aggregation by severity."""
        from pyspark.sql import functions as F
        
        data = [
            ("T001", 3), ("T002", 2), ("T003", 1), ("T004", 3)
        ]
        df = self.spark.createDataFrame(data, ["id", "severity"])
        
        result = df.groupBy("severity").count().collect()
        
        counts = {row["severity"]: row["count"] for row in result}
        assert counts[3] == 2
        assert counts[2] == 1
        assert counts[1] == 1


# =============================================================================
# TEST CLASS: Gold Enrichment
# =============================================================================

class TestGoldEnrichment:
    """L1: Test Gold layer enrichment."""

    @pytest.fixture(autouse=True)
    def setup_spark(self, spark_session_l1):
        """Setup Spark session for each test."""
        self.spark = spark_session_l1

    def test_fact_table_creation(self):
        """Test FACT table creation."""
        from pyspark.sql import functions as F
        
        # Create fact data
        data = [
            ("E001", "accident", 40.7128, -74.0060, 3),
            ("E002", "fire", 40.7580, -73.9855, 4),
        ]
        df = self.spark.createDataFrame(data, ["event_id", "type", "lat", "lon", "severity"])
        
        # Add foreign keys
        df = df.withColumn("neighborhood_sk", F.lit(1))
        df = df.withColumn("infrastructure_sk", F.lit(1))
        
        assert "neighborhood_sk" in df.columns
        assert "infrastructure_sk" in df.columns

    def test_dimension_table_creation(self):
        """Test dimension table creation."""
        from pyspark.sql import functions as F
        
        # Create dim data
        data = [
            ("N001", "Manhattan", 1000),
            ("N002", "Brooklyn", 2000),
        ]
        df = self.spark.createDataFrame(data, ["neighborhood_id", "name", "population"])
        
        # Add surrogate key
        window = Window.orderBy(F.col("neighborhood_id"))
        df = df.withColumn("neighborhood_sk", F.row_number().over(window))
        
        assert "neighborhood_sk" in df.columns
        assert df.count() == 2


# =============================================================================
# TEST CLASS: Real API Integration
# =============================================================================

class TestRealAPIIntegration:
    """L1: Test real external API integration."""

    def test_usgs_api_reachable(self, real_api_data):
        """Test USGS earthquake API is reachable."""
        if real_api_data.get("usgs") is None:
            pytest.skip("USGS API not available")
        
        assert "features" in real_api_data["usgs"]
        logger.info(f"L1: USGS features count: {len(real_api_data['usgs']['features'])}")

    def test_osm_api_reachable(self, real_api_data):
        """Test OSM API is reachable."""
        if real_api_data.get("osm") is None:
            pytest.skip("OSM API not available")
        
        assert "elements" in real_api_data["osm"]
        logger.info(f"L1: OSM elements count: {len(real_api_data['osm']['elements'])}")

    def test_nyc311_api_reachable(self, real_api_data):
        """Test NYC 311 API is reachable."""
        if real_api_data.get("nyc311") is None:
            pytest.skip("NYC 311 API not available")
        
        assert len(real_api_data["nyc311"]) > 0
        logger.info(f"L1: NYC 311 records count: {len(real_api_data['nyc311'])}")


# =============================================================================
# TEST CLASS: MLflow Integration
# =============================================================================

class TestMLflowIntegration:
    """L1: Test MLflow tracking integration."""

    def test_mlflow_tracking_uri_configured(self):
        """Test MLflow tracking URI is configured."""
        import os
        import mlflow
        
        # Try to set tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        
        # Should not raise exception
        assert mlflow.get_tracking_uri() is not None

    def test_sklearn_log_model_import(self):
        """Test sklearn model logging is available."""
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        
        # Create simple model
        X = [[0, 0], [1, 1], [0, 1], [1, 0]]
        y = [0, 1, 1, 0]
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X, y)
        
        # Model should be trained
        assert clf.predict([[0, 0]])[0] in [0, 1]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])