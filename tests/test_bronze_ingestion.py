#!/usr/bin/env python3
"""
================================================================================
TEST SUITE FOR BRONZE LAYER - UNIT TESTS
================================================================================
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bronze_ingestion import (
    Config,
    get_minio_client,
    upload_to_bronze,
    inc_bronze_errors,
    fetch_usgs_earthquakes,
)


class TestConfig:
    """Test Config class."""

    def test_config_has_attributes(self):
        """Test Config has required attributes."""
        config = Config()
        
        # Check for at least one known attribute
        assert hasattr(config, 'MINIO_ENDPOINT') or hasattr(config, 'MINIO_ACCESS_KEY')

    def test_config_minio_endpoint(self):
        """Test MINIO_ENDPOINT is set."""
        config = Config()
        
        assert 'localhost' in config.MINIO_ENDPOINT or '9000' in config.MINIO_ENDPOINT

    def test_config_bucket(self):
        """Test BRONZE_BUCKET is set."""
        config = Config()
        
        assert 'geo' in config.BRONZE_BUCKET.lower()

    def test_config_local_data_dir(self):
        """Test LOCAL_DATA_DIR is a Path."""
        config = Config()
        
        assert isinstance(config.LOCAL_DATA_DIR, Path)

    def test_config_api_urls(self):
        """Test API URLs are configured."""
        config = Config()
        
        assert hasattr(config, 'USGS_API_URL')
        assert hasattr(config, 'OSM_API_URL')
        assert hasattr(config, 'NYC311_API_URL')


class TestErrorHandling:
    """Test error handling."""

    def test_inc_bronze_errors_runs(self):
        """Test inc_bronze_errors runs."""
        inc_bronze_errors()


class TestFetch:
    """Test fetch functions."""

    @patch('bronze_ingestion.requests.get')
    def test_fetch_usgs_returns_response(self, mock_get):
        """Test fetch_usgs_earthquakes returns response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"features": []}
        mock_get.return_value = mock_response
        
        result = fetch_usgs_earthquakes()
        
        mock_get.assert_called()


class TestMinIOClient:
    """Test MinIO client function exists."""

    def test_get_minio_client_exists(self):
        """Test get_minio_client is callable."""
        assert callable(get_minio_client)


class TestUpload:
    """Test upload function exists."""

    def test_upload_to_bronze_exists(self):
        """Test upload_to_bronze is callable."""
        assert callable(upload_to_bronze)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])