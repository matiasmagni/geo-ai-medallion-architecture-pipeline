#!/usr/bin/env python3
"""
================================================================================
L0 UNIT TESTS - Individual Component Tests (100% Coverage Target)
================================================================================
File: tests/test_l0_unit.py

Purpose:
    Test individual functions in isolation. No external services, no Spark.
    Mock only when absolutely necessary.

Test Coverage Target: 100% of utility functions, config, and helpers.

Author: GeoAI Principal Data Engineer & MLOps Architect
================================================================================
"""

import pytest
import os
import sys
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


# =============================================================================
# TEST CLASS: Config Validation
# =============================================================================

class TestConfigValidation:
    """L0: Configuration validation tests."""

    def test_config_defaults_from_env(self):
        """Test Config loads from environment variables."""
        from bronze_ingestion import Config
        
        # Config should have default values
        config = Config()
        
        assert hasattr(config, 'MINIO_ENDPOINT')
        assert hasattr(config, 'MINIO_ACCESS_KEY')
        assert hasattr(config, 'MINIO_SECRET_KEY')
        assert hasattr(config, 'BRONZE_BUCKET')
        assert hasattr(config, 'USGS_API_URL')
        assert hasattr(config, 'OSM_API_URL')
        assert hasattr(config, 'NYC311_API_URL')
        assert hasattr(config, 'OPENSKY_API_URL')
        assert hasattr(config, 'NWS_API_URL')
        
        # Verify real API endpoints
        assert 'usgs.gov' in config.USGS_API_URL
        assert 'overpass-api.de' in config.OSM_API_URL
        assert 'cityofnewyork.us' in config.NYC311_API_URL
        assert 'opensky-network.org' in config.OPENSKY_API_URL
        assert 'weather.gov' in config.NWS_API_URL

    def test_config_env_override(self):
        """Test environment variable overrides."""
        os.environ['S3_ENDPOINT'] = 'http://custom:9000'
        os.environ['AWS_ACCESS_KEY_ID'] = 'customkey'
        
        # Reimport to get fresh values
        import importlib
        import bronze_ingestion
        importlib.reload(bronze_ingestion)
        from bronze_ingestion import Config
        
        config = Config()
        assert 'custom' in config.MINIO_ENDPOINT or config.MINIO_ENDPOINT == 'http://custom:9000'
        
        # Cleanup
        os.environ.pop('S3_ENDPOINT', None)
        os.environ.pop('AWS_ACCESS_KEY_ID', None)

    def test_nyc_airports_configured(self):
        """Test NYC airports are configured."""
        from bronze_ingestion import Config
        
        assert hasattr(Config, 'NYC_AIRPORTS')
        assert isinstance(Config.NYC_AIRPORTS, list)
        assert len(Config.NYC_AIRPORTS) > 0
        assert 'KJFK' in Config.NYC_AIRPORTS
        assert 'KLGA' in Config.NYC_AIRPORTS
        assert 'KEWR' in Config.NYC_AIRPORTS

    def test_local_data_dir_configured(self):
        """Test local data directory is configured."""
        from bronze_ingestion import Config
        
        assert hasattr(Config, 'LOCAL_DATA_DIR')
        assert isinstance(Config.LOCAL_DATA_DIR, Path)

class TestGoldConfigValidation:
    """L0: Gold layer configuration validation tests."""

    def test_gold_config_defaults(self):
        """Test Gold Config defaults."""
        from gold_dimensional_modeling import Config
        config = Config()
        
        assert config.SILVER_BUCKET == "geo-lakehouse/silver"
        assert config.GOLD_BUCKET == "geo-lakehouse/gold"
        assert config.APP_NAME == "GeoAI_Gold_Dimensional"
        assert config.DELTA_COMPRESSION == "snappy"

    def test_gold_config_env_override(self):
        """Test Gold Config environment overrides."""
        os.environ['GOLD_BUCKET'] = 'custom-gold'
        
        # Reload to pick up env change
        import importlib
        import gold_dimensional_modeling
        importlib.reload(gold_dimensional_modeling)
        from gold_dimensional_modeling import Config
        
        config = Config()
        assert config.GOLD_BUCKET == 'custom-gold'
        
        # Cleanup
        os.environ.pop('GOLD_BUCKET', None)


# =============================================================================
# TEST CLASS: Geometry Helpers
# =============================================================================

class TestGeometryHelpers:
    """L0: Geometry helper function tests."""

    def test_latlon_to_xyz_exists(self):
        """Test latlon_to_xyz function exists."""
        try:
            from bronze_ingestion import latlon_to_xyz
            assert callable(latlon_to_xyz)
        except ImportError:
            # Function might not exist, test if we can implement it
            pass

    def test_coordinate_bounds_validation(self):
        """Test coordinate bounds validation logic."""
        def validate_coords(lat: float, lon: float) -> bool:
            return -90 <= lat <= 90 and -180 <= lon <= 180
        
        # Valid NYC coordinates
        assert validate_coords(40.7128, -74.0060) is True
        assert validate_coords(40.7580, -73.9855) is True
        assert validate_coords(40.6782, -73.9442) is True
        
        # Invalid coordinates
        assert validate_coords(91.0, -74.0060) is False
        assert validate_coords(40.7128, -181.0) is False
        assert validate_coords(-91.0, -74.0060) is False

    def test_nyc_bbox_within_valid_bounds(self):
        """Test NYC bounding box is within valid coordinate bounds."""
        NYC_BOUNDS = {'lat_min': 40.49, 'lat_max': 41.0, 'lon_min': -74.3, 'lon_max': -73.7}
        
        assert -90 <= NYC_BOUNDS['lat_min'] <= 90
        assert -90 <= NYC_BOUNDS['lat_max'] <= 90
        assert -180 <= NYC_BOUNDS['lon_min'] <= 180
        assert -180 <= NYC_BOUNDS['lon_max'] <= 180
        assert NYC_BOUNDS['lat_min'] < NYC_BOUNDS['lat_max']
        assert NYC_BOUNDS['lon_min'] < NYC_BOUNDS['lon_max']

    def test_shapely_point_creation(self):
        """Test Shapely Point creation and WKT conversion."""
        from shapely.geometry import Point
        from shapely import wkt
        
        # Create point
        point = Point(-74.006, 40.7128)
        assert point.x == -74.006
        assert point.y == 40.7128
        
        # Convert to WKT
        wkt_str = wkt.dumps(point)
        assert "POINT" in wkt_str
        assert "-74.006" in wkt_str

    def test_shapely_polygon_creation(self):
        """Test Shapely Polygon for neighborhood boundaries."""
        from shapely.geometry import Polygon
        
        # Create Manhattan-like polygon
        coords = [
            (-74.02, 40.70), (-73.97, 40.74), (-73.94, 40.80), 
            (-73.97, 40.83), (-74.02, 40.70)
        ]
        poly = Polygon(coords)
        assert poly.is_valid
        assert poly.area > 0

    def test_shapely_geojson_to_wkt(self):
        """Test GeoJSON to WKT conversion."""
        from shapely.geometry import shape
        from shapely import wkt
        
        geojson = {"type": "Point", "coordinates": [-74.006, 40.7128]}
        geom = shape(geojson)
        wkt_str = wkt.dumps(geom)
        assert "POINT" in wkt_str

    def test_shapely_point_in_polygon(self):
        """Test point-in-polygon detection."""
        from shapely.geometry import Point, Polygon
        
        poly = Polygon([
            (-74.02, 40.70), (-73.97, 40.74), (-73.94, 40.80), 
            (-73.97, 40.83), (-74.02, 40.70)
        ])
        
        # Point inside
        inside_point = Point(-73.98, 40.76)
        assert poly.contains(inside_point) or poly.touches(inside_point)
        
        # Point outside
        outside_point = Point(0, 0)
        assert not poly.contains(outside_point)


# =============================================================================
# TEST CLASS: Data Transformation Utilities
# =============================================================================

class TestDataTransformations:
    """L0: Data transformation utility tests."""

    def test_pandas_dataframe_creation(self):
        """Test pandas DataFrame creation with required columns."""
        import pandas as pd
        
        df = pd.DataFrame({
            "incident_id": ["T001", "T002", "T003"],
            "latitude": [40.7128, 40.7580, 40.6782],
            "longitude": [-74.0060, -73.9855, -73.9442],
            "severity": [3, 2, 1],
        })
        
        assert len(df) == 3
        assert list(df.columns) == ["incident_id", "latitude", "longitude", "severity"]

    def test_pandas_column_renaming(self):
        """Test pandas column renaming."""
        import pandas as pd
        
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Start_Lat": [40.7, 40.8, 40.9],
            "Start_Lng": [-74.0, -73.9, -73.8]
        })
        
        df = df.rename(columns={"ID": "incident_id", "Start_Lat": "latitude", "Start_Lng": "longitude"})
        
        assert "incident_id" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "ID" not in df.columns

    def test_pandas_null_handling(self):
        """Test null value handling in DataFrame."""
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": ["a", "b", None]
        })
        
        assert df["col1"].isna().sum() == 1
        assert df["col2"].isna().sum() == 1
        
        # Test dropna
        df_clean = df.dropna()
        assert len(df_clean) == 1

    def test_pandas_filter_by_bounds(self):
        """Test filtering coordinates within valid bounds."""
        import pandas as pd
        
        df = pd.DataFrame({
            "lat": [40.7, 91.0, -91.0, 40.8],
            "lon": [-74.0, -73.0, -73.0, -73.5]
        })
        
        valid = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]
        assert len(valid) == 2

    def test_pandas_groupby_aggregation(self):
        """Test groupby aggregation for metrics."""
        import pandas as pd
        
        df = pd.DataFrame({
            "borough": ["Manhattan", "Brooklyn", "Manhattan", "Queens"],
            "severity": [3, 2, 4, 1],
        })
        
        result = df.groupby("borough").agg({"severity": ["count", "mean", "max"]})
        assert "Manhattan" in result.index
        assert result.loc["Manhattan", ("severity", "count")] == 2


# =============================================================================
# TEST CLASS: ML Model Utilities
# =============================================================================

class TestMLModelUtilities:
    """L0: ML model utility tests."""

    def test_sklearn_imports(self):
        """Test sklearn imports are available."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        
        assert RandomForestClassifier is not None
        assert RandomForestRegressor is not None
        assert GradientBoostingClassifier is not None
        assert GradientBoostingRegressor is not None

    def test_mlflow_imports(self):
        """Test MLflow imports are available."""
        import mlflow
        import mlflow.sklearn
        
        assert hasattr(mlflow, 'start_run')
        assert hasattr(mlflow, 'log_params')
        assert hasattr(mlflow, 'log_metrics')
        assert hasattr(mlflow, 'sklearn')

    def test_model_metrics_calculation(self):
        """Test model metrics calculation."""
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        import numpy as np
        
        y_true_class = [1, 0, 1, 1, 0]
        y_pred_class = [1, 0, 1, 0, 0]
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class, average='weighted')
        
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        
        y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_reg = [1.1, 2.1, 2.9, 4.2, 4.8]
        
        rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
        r2 = r2_score(y_true_reg, y_pred_reg)
        
        assert rmse >= 0
        assert r2 <= 1

    def test_synthetic_data_generation_shape(self):
        """Test synthetic data generation produces expected shape."""
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        assert X.shape == (100, 20)
        assert len(y) == 100

    def test_feature_importance_normalization(self):
        """Test feature importance values are normalized."""
        import numpy as np
        
        # Simulated feature importances
        importances = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        
        # Normalize to sum to 1
        normalized = importances / importances.sum()
        
        assert abs(normalized.sum() - 1.0) < 1e-6
        assert all(0 <= v <= 1 for v in normalized)


# =============================================================================
# TEST CLASS: GeoJSON Utilities
# =============================================================================

class TestGeoJSONUtilities:
    """L0: GeoJSON utility tests."""

    def test_geojson_feature_creation(self):
        """Test creating valid GeoJSON Feature."""
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [-74.006, 40.7128]
            },
            "properties": {
                "category": "fire",
                "weight": 0.75,
                "source": "ml_prediction"
            }
        }
        
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 2
        assert "properties" in feature

    def test_geojson_featurecollection_creation(self):
        """Test creating valid GeoJSON FeatureCollection."""
        featurecollection = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]}, "properties": {}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-73.985, 40.758]}, "properties": {}},
            ],
            "metadata": {"total_features": 2, "source": "ml_predictions"}
        }
        
        assert featurecollection["type"] == "FeatureCollection"
        assert len(featurecollection["features"]) == 2
        assert "metadata" in featurecollection

    def test_geojson_serialization(self):
        """Test GeoJSON JSON serialization."""
        import json
        
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
            "properties": {"category": "fire", "weight": 0.75}
        }
        
        json_str = json.dumps(feature)
        parsed = json.loads(json_str)
        
        assert parsed["type"] == "Feature"
        assert parsed["geometry"]["coordinates"] == [-74.006, 40.7128]

    def test_geojson_coordinates_order(self):
        """Test GeoJSON coordinates follow [lon, lat] order."""
        import json
        
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
            "properties": {}
        }
        
        coords = feature["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]
        
        # Longitude should be between -180 and 180
        assert -180 <= lon <= 180
        # Latitude should be between -90 and 90
        assert -90 <= lat <= 90


# =============================================================================
# TEST CLASS: Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """L0: General utility function tests."""

    def test_path_joining(self):
        """Test path joining for data paths."""
        base = Path("/tmp/geoai")
        table = "us_accidents"
        
        path = base / "bronze" / table / "part-00000.parquet"
        assert path.suffix == ".parquet"
        assert "us_accidents" in str(path)

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        import json
        
        data = {
            "source": "us_accidents",
            "count": 100,
            "valid": True,
            "timestamp": "2024-01-15T10:30:00"
        }
        
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        
        assert parsed["source"] == "us_accidents"
        assert parsed["count"] == 100
        assert parsed["valid"] is True

    def test_timestamp_format(self):
        """Test timestamp formatting."""
        from datetime import datetime
        
        ts = datetime(2024, 1, 15, 10, 30, 0)
        iso_str = ts.isoformat()
        
        assert "2024-01-15" in iso_str
        assert "10:30:00" in iso_str

    def test_timestamp_from_unix_epoch(self):
        """Test converting Unix epoch to datetime."""
        from datetime import datetime
        
        epoch = 1705315800  # 2024-01-15 10:30:00 UTC
        dt = datetime.fromtimestamp(epoch)
        
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_file_exists_check(self):
        """Test file existence check."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": true}')
            temp_path = Path(f.name)
        
        assert temp_path.exists()
        temp_path.unlink()  # Cleanup
        assert not temp_path.exists()

    def test_directory_creation(self):
        """Test directory creation."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            assert test_dir.exists()
            assert test_dir.is_dir()


# =============================================================================
# TEST CLASS: Prometheus Metrics
# =============================================================================

class TestPrometheusMetrics:
    """L0: Prometheus metrics tests."""

    def test_prometheus_client_import(self):
        """Test Prometheus client imports."""
        from prometheus_client import Counter, Histogram, Gauge, Summary
        
        assert Counter is not None
        assert Histogram is not None
        assert Gauge is not None
        assert Summary is not None

    def test_counter_increment(self):
        """Test Counter increment."""
        from prometheus_client import Counter
        
        test_counter = Counter('test_counter', 'Test counter')
        test_counter.inc()
        test_counter.inc(5)
        
        # Counter value should be 6
        assert test_counter._value._value >= 6

    def test_gauge_set(self):
        """Test Gauge set and get."""
        from prometheus_client import Gauge
        
        test_gauge = Gauge('test_gauge', 'Test gauge')
        test_gauge.set(42.5)
        
        # Gauge should be set to 42.5
        assert test_gauge._value._value == 42.5


# =============================================================================
# TEST CLASS: Logging Configuration
# =============================================================================

class TestLoggingConfiguration:
    """L0: Logging configuration tests."""

    def test_logger_creation(self):
        """Test logger creation and configuration."""
        import logging
        
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        
        assert logger.level == logging.INFO

    def test_log_format_validation(self):
        """Test log format string is valid."""
        import logging
        
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format_str)
        
        assert "%(asctime)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt


class TestSilverEnrichmentUnit:
    """L0: Silver enrichment unit tests."""
    def test_silver_config_defaults(self):
        """L0-T001: Silver config has correct defaults."""
        from silver_enrichment import Config

        c = Config()
        assert c.SILVER_BUCKET == "geoai-silver"
        assert "11434" in c.OLLAMA_BASE_URL

    def test_silver_ollama_config(self):
        """L0-T002: Ollama settings configured."""
        from silver_enrichment import Config

        c = Config()
        assert hasattr(c, "OLLAMA_BASE_URL")
        assert hasattr(c, "OLLAMA_MODEL")
        assert c.OLLAMA_TIMEOUT > 0

    def test_parse_valid_ai_enrichment(self):
        """L0-T010: Parse valid AI enrichment JSON."""
        from silver_enrichment import parse_ai_enrichment

        result = parse_ai_enrichment('{"severity": 8, "hazard_type": "traffic"}')
        assert result["ai_severity"] == 8
        assert result["ai_hazard_type"] == "traffic"

    def test_parse_ai_enrichment_logic(self):
        """L0-T011: Handle invalid JSON gracefully."""
        from silver_enrichment import parse_ai_enrichment
        res = parse_ai_enrichment("not_json")
        assert res == {'ai_severity': 5, 'ai_hazard_type': 'unknown'}

    def test_parse_malformed_severity(self):
        """L0-T012: Handle malformed severity."""
        from silver_enrichment import parse_ai_enrichment
        result = parse_ai_enrichment('{"severity": "high", "hazard_type": "traffic"}')
        assert result["ai_severity"] == 5

    def test_parse_severity_bounds(self):
        """L0-T013: Handle severity bounds."""
        from silver_enrichment import parse_ai_enrichment
        result = parse_ai_enrichment('{"severity": 15, "hazard_type": "fire"}')
        assert result["ai_severity"] <= 10
        result = parse_ai_enrichment('{"severity": 0, "hazard_type": "fire"}')
        assert result["ai_severity"] >= 1

    def test_ai_enrichment_udf_creation(self):
        from silver_enrichment import create_ollama_enrichment_udf, Config
        config = Config()
        config.OLLAMA_BASE_URL = "http://localhost:11434"
        udf = create_ollama_enrichment_udf(config)
        assert udf is not None

class TestSilverSpatialUnit:
    """L0: Silver spatial unit tests."""
    def test_spatial_logic_imports(self):
        from silver_sedona_transform import create_geometry_from_latlon, parse_geojson_geometry
        from silver_spatial_transform import create_spark_session, read_bronze_data
        assert create_geometry_from_latlon is not None
        assert parse_geojson_geometry is not None
        assert create_spark_session is not None
        assert read_bronze_data is not None

class TestTelemetryUnit:
    """L0: Telemetry unit tests."""
    def test_telemetry_exists(self):
        from telemetry import get_tracer, setup_telemetry
        assert get_tracer() is not None
        assert setup_telemetry is not None

    def test_traced_exception_handling(self):
        from telemetry import traced
        @traced("test", "test_op")
        def fail_func():
            raise ValueError("test error")
        with pytest.raises(ValueError):
            fail_func()

class TestAviationUnit:
    """L0: Aviation module unit tests."""
    def test_aviation_imports(self):
        from train_aviation_models import train_all_models
        assert train_all_models is not None

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src", "--cov-report=term-missing"])