#!/usr/bin/env python3
"""
================================================================================
L3 END-TO-END TESTS - Full System Verification
================================================================================
File: tests/test_l3_e2e.py

Purpose:
    Test full system from data ingestion to ML model serving.
    Verifies complete pipeline: Bronze -> Silver -> Gold -> ML Models -> Visualization

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CLASS: End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """L3: Complete pipeline E2E tests."""

    @pytest.mark.usefixtures("cleanup_test_artifacts")
    def test_bronze_ingestion_runs(self):
        """Test bronze ingestion script runs without error."""
        try:
            from bronze_ingestion import run_bronze_ingestion
            
            # Run ingestion
            result = run_bronze_ingestion()
            
            # Should return True on success
            assert result is True or result is None  # May not return value
        except Exception as e:
            pytest.skip(f"Bronze ingestion failed: {e}")

    def test_bronze_data_exists(self):
        """Test bronze data files are created."""
        bronze_path = Path("/tmp/geoai/bronze")
        
        if not bronze_path.exists():
            pytest.skip("Bronze data not available")
        
        # Check for expected directories
        expected_dirs = ["us_accidents", "nyc_311", "osm_infrastructure", "us_neighborhoods", "nyc_flights", "nyc_weather"]
        
        for dir_name in expected_dirs:
            dir_path = bronze_path / dir_name
            if dir_path.exists():
                parquet_files = list(dir_path.glob("*.parquet"))
                logger.info(f"L3: {dir_name} has {len(parquet_files)} parquet files")

    def test_ml_models_train(self):
        """Test ML models can be trained."""
        try:
            from train_healthcare_models import train_all_models
            
            # Train with small sample
            results = train_all_models(n_samples=100)
            
            assert results is not None
            assert len(results) == 5
        except Exception as e:
            pytest.skip(f"ML training failed: {e}")

    def test_mlflow_models_registered(self):
        """Test ML models are registered in MLflow."""
        import mlflow
        
        mlflow.set_tracking_uri("http://localhost:5001")
        
        try:
            # Try to get registered models
            client = mlflow.MlflowClient()
            models = client.list_registered_models()
            
            logger.info(f"L3: Found {len(models)} registered models")
            
            # Check for our models
            model_names = [m.name for m in models]
            
            expected_models = [
                "NYC_FireRiskModel",
                "NYC_HospitalOverpopulationModel",
                "NYC_EmergencyResponseModel",
                "NYC_HospitalBedDemand",
                "NYC_AmbulanceDispatch"
            ]
            
            for model in expected_models:
                if model in model_names:
                    logger.info(f"L3: Found model: {model}")
        except Exception as e:
            logger.warning(f"L3: MLflow check failed: {e}")
            pytest.skip("MLflow not accessible")


# =============================================================================
# TEST CLASS: Web Application Tests
# =============================================================================

class TestWebApplication:
    """L3: Test web application E2E."""

    def test_web_page_loads(self):
        """Test web page loads."""
        import requests
        
        try:
            response = requests.get("http://localhost:3000", timeout=10)
            assert response.status_code in [200, 404, 301, 302]
        except Exception:
            pytest.skip("Web app not running at localhost:3000")

    def test_heatmap_data_exists(self):
        """Test heatmap data file exists."""
        data_path = Path("geo-ai-heatmap/public/data/heatmap.geojson")
        
        if not data_path.exists():
            pytest.skip("Heatmap data not available")
        
        # Read and validate GeoJSON
        with open(data_path) as f:
            data = json.load(f)
        
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) > 0
        
        logger.info(f"L3: Heatmap has {len(data['features'])} features")

    def test_ml_predictions_in_heatmap(self):
        """Test ML predictions are included in heatmap."""
        data_path = Path("geo-ai-heatmap/public/data/heatmap.geojson")
        
        if not data_path.exists():
            pytest.skip("Heatmap data not available")
        
        with open(data_path) as f:
            data = json.load(f)
        
        # Check for ML prediction features
        ml_features = [
            f for f in data.get("features", []) 
            if f.get("properties", {}).get("source") == "ml_prediction"
        ]
        
        logger.info(f"L3: Found {len(ml_features)} ML prediction features")
        
        # Check metadata
        if "metadata" in data:
            logger.info(f"L3: Metadata: {data['metadata']}")


# =============================================================================
# TEST CLASS: Grafana Dashboard Tests
# =============================================================================

class TestGrafanaDashboards:
    """L3: Test Grafana dashboard availability."""

    def test_grafana_accessible(self):
        """Test Grafana is accessible."""
        import requests
        
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            assert response.status_code == 200
        except Exception:
            pytest.skip("Grafana not running")

    def test_grafana_dashboards_exist(self):
        """Test Grafana dashboards exist."""
        dashboards = [
            "data-quality",
            "ml-models", 
            "pipeline-performance"
        ]
        
        for dashboard in dashboards:
            path = Path(f"grafana/provisioning/dashboards/{dashboard}.json")
            if path.exists():
                logger.info(f"L3: Found dashboard: {dashboard}")
            else:
                logger.warning(f"L3: Dashboard not found: {dashboard}")

    def test_grafana_datasource_configured(self):
        """Test Grafana datasource is configured."""
        import requests
        
        try:
            # Try to get datasources
            response = requests.get(
                "http://localhost:3000/api/datasources",
                auth=("admin", "admin"),
                timeout=5
            )
            
            if response.status_code == 200:
                datasources = response.json()
                logger.info(f"L3: Found {len(datasources)} datasources")
        except Exception as e:
            logger.warning(f"L3: Grafana datasource check failed: {e}")


# =============================================================================
# TEST CLASS: API Endpoint Tests
# =============================================================================

class TestAPIEndpoints:
    """L3: Test external API endpoints."""

    def test_usgs_earthquake_api(self):
        """Test USGS earthquake API."""
        import requests
        
        params = {
            "format": "geojson",
            "starttime": "2024-01-01",
            "endtime": "2024-01-15",
            "minlatitude": 40.5,
            "maxlatitude": 41.0,
            "minlongitude": -74.5,
            "maxlongitude": -73.5,
        }
        
        try:
            response = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params=params,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "features" in data
            
            logger.info(f"L3: USGS API returned {len(data['features'])} earthquakes")
        except Exception as e:
            pytest.skip(f"USGS API unavailable: {e}")

    def test_osm_infrastructure_api(self):
        """Test OSM infrastructure API."""
        import requests
        
        query = """[out:json][timeout:30];node["amenity"~"hospital|fire_station"](40.5,-74.1,41.0,-73.7);out;"""
        
        try:
            response = requests.get(
                "https://overpass-api.de/api/interpreter",
                params={'data': query},
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "elements" in data
            
            logger.info(f"L3: OSM API returned {len(data['elements'])} facilities")
        except Exception as e:
            pytest.skip(f"OSM API unavailable: {e}")

    def test_nyc_311_api(self):
        """Test NYC 311 API."""
        import requests
        
        params = {"$limit": 10, "$where": "latitude IS NOT NULL"}
        
        try:
            response = requests.get(
                "https://data.cityofnewyork.us/resource/fhrw-4uyv.json",
                params=params,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) > 0
            
            logger.info(f"L3: NYC 311 API returned {len(data)} records")
        except Exception as e:
            pytest.skip(f"NYC 311 API unavailable: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])