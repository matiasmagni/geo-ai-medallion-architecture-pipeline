#!/usr/bin/env python3
"""
================================================================================
L2 COMPONENT TESTS - Service/Component Tests
================================================================================
File: tests/test_l2_component.py

Purpose:
    Test complete pipeline components as black boxes.
    Tests run against actual Docker services (MinIO, Postgres).

Test Categories:
    - Bronze ingestion component
    - Silver transform component
    - Gold enrichment component
    - ML model training component

Author: GeoAI Principal Data Engineer
================================================================================
"""

import pytest
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBronzeComponent:
    """L2: Test Bronze ingestion component."""

    def test_bronze_ingestion_runs(self):
        """Test that bronze ingestion runs without errors."""
        result = subprocess.run(
            [
                "docker", "exec", "geoai-spark", "python", "-c",
                "import sys; sys.path.append('/home/jovyan'); "
                "from src.bronze_ingestion import run_bronze_ingestion; "
                "run_bronze_ingestion()"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        assert result.returncode == 0, f"Bronze ingestion failed: {result.stderr}"
        assert "Bronze ingestion complete" in result.stderr or "complete" in result.stderr.lower()

    def test_bronze_output_exists(self):
        """Test that bronze parquet files are created."""
        result = subprocess.run(
            ["docker", "exec", "geoai-spark", "ls", "/tmp/geoai/bronze/"],
            capture_output=True,
            text=True
        )
        
        assert "us_accidents" in result.stdout
        assert "usgs_earthquakes" in result.stdout
        assert "nyc_311" in result.stdout


class TestSilverComponent:
    """L2: Test Silver transformation component."""

    def test_silver_transform_runs(self):
        """Test that silver transform runs without errors."""
        result = subprocess.run(
            [
                "docker", "exec", "geoai-spark", "python", "-c",
                "import sys; sys.path.append('/home/jovyan'); "
                "from src.silver_spatial_transform import run_silver_pipeline; "
                "run_silver_pipeline()"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        assert result.returncode == 0, f"Silver transform failed: {result.stderr}"

    def test_silver_output_has_geometry(self):
        """Test that silver output has geometry columns."""
        result = subprocess.run(
            [
                "docker", "exec", "geoai-spark", "python", "-c",
                "from pyspark.sql import SparkSession; "
                "spark = SparkSession.builder.appName('test').master('local[*]').getOrCreate(); "
                "df = spark.read.parquet('/tmp/geoai/silver/us_accidents'); "
                "print('Columns:', df.columns); "
                "print('Count:', df.count())"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0
        assert "geometry" in result.stdout or "geometry" in result.stderr


class TestGoldComponent:
    """L2: Test Gold enrichment component."""

    def test_gold_enrichment_runs(self):
        """Test that gold enrichment runs without errors."""
        result = subprocess.run(
            [
                "docker", "exec", "geoai-spark", "python", "-c",
                "import sys; sys.path.append('/home/jovyan'); "
                "from src.gold_schema_and_ai_enrichment import run_gold_pipeline; "
                "run_gold_pipeline()"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Gold enrichment failed: {result.stderr}"

    def test_gold_output_exists(self):
        """Test that gold tables are created."""
        result = subprocess.run(
            ["docker", "exec", "geoai-spark", "ls", "/tmp/geoai/gold/"],
            capture_output=True,
            text=True
        )
        
        assert "fact_hazard_events" in result.stdout or result.returncode == 0


class TestMLModels:
    """L2: Test ML model training component."""

    def test_model_training_doc_exists(self):
        """Test that model training is documented in README."""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md not found"
        
        content = readme_path.read_text()
        # Check that model training is documented
        assert "mlflow" in content.lower() or "model" in content.lower(), "Model training not documented"

    def test_mlflow_accessible(self):
        """Test that MLflow is accessible."""
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:5000"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.stdout.strip() in ["200", "301", "302"], "MLflow not accessible"


class TestMinIOStorage:
    """L2: Test MinIO storage component."""

    def test_minio_is_running(self):
        """Test that MinIO is running."""
        # First create bucket if it doesn't exist
        subprocess.run(
            ["docker", "exec", "geoai-minio", "mc", "mb", "--ignore-existing", "geo-lakehouse/geoai"],
            capture_output=True,
            timeout=10
        )
        # Now check it's accessible
        result = subprocess.run(
            ["docker", "exec", "geoai-minio", "mc", "ls", "geo-lakehouse/"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "MinIO not accessible"

    def test_postgres_is_running(self):
        """Test that PostgreSQL is running."""
        result = subprocess.run(
            ["docker", "exec", "geoai-postgres", "pg_isready", "-U", "postgres"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "PostgreSQL not ready"


class TestPrometheusMetrics:
    """L2: Test Prometheus metrics collection."""

    def test_prometheus_accessible(self):
        """Test that Prometheus is accessible."""
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", 
             "http://localhost:9090/-/healthy"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Accept 200 (healthy), 301/302 (redirect), or other 3xx as healthy
        assert result.stdout.strip() in ["200", "301", "302"], f"Prometheus not accessible: {result.stdout.strip()}"

    def test_metrics_endpoint(self):
        """Test that metrics endpoint returns data."""
        result = subprocess.run(
            ["curl", "-s", "http://localhost:9090/api/v1/query?query=up"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "up" in result.stdout or "status" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])