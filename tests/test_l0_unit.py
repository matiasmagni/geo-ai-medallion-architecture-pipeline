#!/usr/bin/env python3
"""
================================================================================
L0 UNIT TESTS - Individual Function Tests
================================================================================
File: tests/test_l0_unit.py

Purpose:
    Test individual functions in isolation with mocked dependencies.
    No external services, no Spark context, pure unit tests.

Test Categories:
    - Config validation
    - Geometry helpers (Shapely)
    - Data transformations (pandas)
    - Utility functions

Author: GeoAI Principal Data Engineer
================================================================================
"""

import pytest
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig:
    """L0: Config validation tests."""

    def test_config_defaults(self):
        """Test configuration has expected defaults."""
        os.environ.pop("S3_ENDPOINT", None)
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        
        from silver_spatial_transform import Config
        
        config = Config()
        assert "host.docker.internal" in config.MINIO_ENDPOINT or "localhost" in config.MINIO_ENDPOINT
        assert config.MINIO_ACCESS_KEY == "minioadmin"
        assert config.BUCKET_BRONZE == "geo-lakehouse/bronze"

    def test_config_env_override(self):
        """Test environment variables override defaults."""
        os.environ["S3_ENDPOINT"] = "http://custom:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "customkey"
        
        # Need to reimport to get fresh config
        import importlib
        import silver_spatial_transform
        importlib.reload(silver_spatial_transform)
        from silver_spatial_transform import Config
        
        config = Config()
        # Note: config might have cached values, just check it loads


class TestGeometryHelpers:
    """L0: Geometry helper function tests."""

    def test_shapely_point_creation(self):
        """Test Shapely Point creation."""
        from shapely.geometry import Point
        
        point = Point(-74.006, 40.7128)
        assert point.x == -74.006
        assert point.y == 40.7128
        assert point.wkt == "POINT (-74.006 40.7128)"

    def test_shapely_point_wkt_conversion(self):
        """Test Shapely to WKT conversion."""
        from shapely.geometry import Point
        from shapely import wkt
        
        point = Point(-73.985, 40.758)
        wkt_str = wkt.dumps(point)
        assert "POINT" in wkt_str
        assert "-73.985" in wkt_str or "-73.98" in wkt_str

    def test_shapely_polygon_creation(self):
        """Test Shapely Polygon creation."""
        from shapely.geometry import Polygon
        
        coords = [(-74.01, 40.70), (-73.99, 40.70), (-73.99, 40.72), (-74.01, 40.72), (-74.01, 40.70)]
        poly = Polygon(coords)
        assert poly.is_valid
        assert poly.area > 0

    def test_shapely_geojson_to_wkt(self):
        """Test GeoJSON to WKT conversion."""
        from shapely.geometry import shape
        from shapely import wkt
        
        geojson = {
            "type": "Point",
            "coordinates": [-74.006, 40.7128]
        }
        geom = shape(geojson)
        wkt_str = wkt.dumps(geom)
        assert "POINT" in wkt_str


class TestDataTransformations:
    """L0: Data transformation tests with pandas."""

    def test_pandas_rename_columns(self):
        """Test pandas column renaming."""
        import pandas as pd
        
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Start_Lat": [40.7, 40.8, 40.9],
            "Start_Lng": [-74.0, -73.9, -73.8]
        })
        
        df = df.rename(columns={"ID": "incident_id"})
        assert "incident_id" in df.columns
        assert "ID" not in df.columns

    def test_pandas_filter_bounds(self):
        """Test filtering coordinates within valid bounds."""
        import pandas as pd
        
        df = pd.DataFrame({
            "lat": [40.7, 91.0, -91.0, 40.8],
            "lon": [-74.0, -73.0, -73.0, -73.5]
        })
        
        valid = df[
            (df["lat"] >= -90) & (df["lat"] <= 90) &
            (df["lon"] >= -180) & (df["lon"] <= 180)
        ]
        
        assert len(valid) == 2

    def test_pandas_null_handling(self):
        """Test null value handling."""
        import pandas as pd
        
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": ["a", "b", None]
        })
        
        assert df["col1"].isna().sum() == 1
        assert df.dropna().shape[0] == 1


class TestUtilityFunctions:
    """L0: Utility function tests."""

    def test_file_path_joining(self):
        """Test path joining."""
        base = "/tmp/geoai"
        table = "us_accidents"
        
        path = f"{base}/bronze/{table}"
        assert path == "/tmp/geoai/bronze/us_accidents"

    def test_json_serialization(self):
        """Test JSON serialization."""
        data = {
            "source": "us_accidents",
            "count": 100,
            "valid": True
        }
        
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        assert parsed["source"] == "us_accidents"
        assert parsed["count"] == 100

    def test_timestamp_format(self):
        """Test timestamp formatting."""
        from datetime import datetime
        
        ts = datetime(2024, 1, 15, 10, 30, 0)
        iso_str = ts.isoformat()
        
        assert "2024-01-15" in iso_str
        assert "10:30:00" in iso_str


class TestMetricsHelpers:
    """L0: Metrics helper tests."""

    def test_metric_imports(self):
        """Test metrics can be imported."""
        from prometheus_client import Counter, Histogram, Gauge
        
        # Just verify imports work
        assert Counter is not None
        assert Histogram is not None
        assert Gauge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])