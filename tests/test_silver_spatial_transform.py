#!/usr/bin/env python3
"""
================================================================================
TEST SUITE FOR SILVER SPATIAL TRANSFORM - UNIT TESTS
================================================================================
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig:
    """Test Config class."""

    def test_config_has_attributes(self):
        """Test Config has required attributes."""
        from silver_spatial_transform import Config
        config = Config()
        
        assert hasattr(config, 'APP_NAME') or hasattr(config, 'SPARK_MASTER')

    def test_config_app_name(self):
        """Test APP_NAME is set."""
        from silver_spatial_transform import Config
        config = Config()
        
        assert config.APP_NAME is not None


class TestFunctions:
    """Test module functions."""

    def test_transform_functions_exist(self):
        """Test transform functions can be imported."""
        from silver_spatial_transform import (
            transform_us_accidents,
            transform_neighborhoods,
            transform_usgs_earthquakes,
            transform_osm_infrastructure
        )
        
        assert callable(transform_us_accidents)
        assert callable(transform_neighborhoods)

    def test_spark_session_exists(self):
        """Test create_spark_session exists."""
        from silver_spatial_transform import create_spark_session
        assert callable(create_spark_session)

    def test_read_bronze_exists(self):
        """Test read_bronze_data exists."""
        from silver_spatial_transform import read_bronze_data
        assert callable(read_bronze_data)

    def test_write_silver_exists(self):
        """Test write_silver_table exists."""
        from silver_spatial_transform import write_silver_table
        assert callable(write_silver_table)

    def test_run_pipeline_exists(self):
        """Test run_silver_pipeline exists."""
        from silver_spatial_transform import run_silver_pipeline
        assert callable(run_silver_pipeline)


class TestGeometry:
    """Test geometry functions."""

    def test_geometry_functions_exist(self):
        """Test geometry functions exist."""
        from silver_spatial_transform import (
            create_geometry_from_latlon,
            parse_geojson_geometry,
            transform_to_crs
        )
        
        assert callable(create_geometry_from_latlon)
        assert callable(parse_geojson_geometry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])