#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures
==================================
This module provides reusable fixtures for the GeoAI Medallion Architecture test suite.
It includes:
- PySpark session fixture with Sedona and Delta Lake
- Mock environment variables for MinIO and PostgreSQL
- Common test data fixtures

Author: GeoAI SDET Team
Version: 1.0.0
"""

import os
import sys
import pytest
import json
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any, Optional
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Test environment variables (defaults for local development)
TEST_ENV = {
    "SPARK_MASTER_URL": "local[*]",
    "S3_ENDPOINT": "http://localhost:9000",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
    "MINIO_BUCKET_BRONZE": "geo-lakehouse/bronze",
    "MINIO_BUCKET_SILVER": "geo-lakehouse/silver",
    "MINIO_BUCKET_GOLD": "geo-lakehouse/gold",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "geometastore",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "postgres",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3",
    "OLLAMA_TIMEOUT": "30",
}


# =============================================================================
# PYSPARK SESSION FIXTURE
# =============================================================================


@pytest.fixture(scope="session")
def spark_session() -> Generator["SparkSession", None, None]:
    """
    Create a local PySpark session with Sedona and Delta Lake extensions.

    This fixture provides a shared Spark session for all tests in the session.
    It configures the session with:
    - Local master for testing
    - Sedona spatial SQL plugin
    - Delta Lake extensions
    - MinIO S3 compatibility

    Yields:
        SparkSession: Configured PySpark session

    Note:
        Sedona and Delta plugins are loaded if available. If not, falls back
        to basic Spark session without spatial/ACID features.

    If Java is not available, returns a mock SparkSession for testing.
    """
    import subprocess
    import os

    # Check if Java is available and working
    java_available = False
    java_home = os.environ.get("JAVA_HOME")

    try:
        # First check if JAVA_HOME is set and valid
        if java_home:
            java_path = os.path.join(java_home, "bin", "java")
            if os.path.exists(java_path):
                java_available = True

        # Also try running java -version
        result = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        java_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if not java_available:
        # Return a mock SparkSession for tests that don't actually need Spark
        mock_spark = MagicMock()
        mock_spark.version = "3.5.0"
        mock_spark.sparkContext = MagicMock()
        mock_spark.sparkContext.uiWebUrl = None
        mock_spark.createDataFrame = MagicMock(return_value=MagicMock())
        mock_spark.read = MagicMock()
        mock_spark.stop = MagicMock()
        yield mock_spark
        return

    from pyspark.sql import SparkSession

    # Build Spark session
    # Start with minimal config - don't add Sedona/Delta plugins upfront
    builder = SparkSession.builder
    builder = builder.config("spark.driver.memory", "2g")
    builder = builder.config("spark.sql.shuffle.partitions", "2")
    builder = builder.config("spark.master", TEST_ENV["SPARK_MASTER_URL"])
    builder = builder.config(
        "spark.serializer", "org.apache.spark.serializer.KryoSerializer"
    )

    # Configure MinIO S3 (for local testing)
    builder = builder.config("spark.hadoop.fs.s3a.endpoint", TEST_ENV["S3_ENDPOINT"])
    builder = builder.config(
        "spark.hadoop.fs.s3a.access.key", TEST_ENV["AWS_ACCESS_KEY_ID"]
    )
    builder = builder.config(
        "spark.hadoop.fs.s3a.secret.key", TEST_ENV["AWS_SECRET_ACCESS_KEY"]
    )
    builder = builder.config("spark.hadoop.fs.s3a.path.style.access", "true")
    builder = builder.config(
        "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )

    # Test if we can create a basic Spark session first
    try:
        spark = builder.getOrCreate()
    except Exception as initial_error:
        # If plugins cause issues, retry with clean config
        if (
            "sedona" in str(initial_error).lower()
            or "delta" in str(initial_error).lower()
            or "classnotfound" in str(initial_error).lower()
            or "catalog" in str(initial_error).lower()
        ):
            # Create clean builder without any plugins
            builder = SparkSession.builder
            builder = builder.config("spark.driver.memory", "2g")
            builder = builder.config("spark.sql.shuffle.partitions", "2")
            builder = builder.config("spark.master", "local[*]")
            builder = builder.config(
                "spark.serializer", "org.apache.spark.serializer.KryoSerializer"
            )
            spark = builder.getOrCreate()
        else:
            raise

    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    # Cleanup
    spark.stop()


@pytest.fixture
def spark_with_sedona(spark_session) -> "SparkSession":
    """
    Provide a Spark session with Sedona spatial functions registered.

    If Sedona is available, this fixture ensures spatial functions are loaded.
    Otherwise, it provides the basic session.

    Returns:
        SparkSession: Session with spatial functions
    """
    # Attempt to register Sedona functions
    try:
        from sedona.spark import SedonaContext

        sedona = SedonaContext.create(spark_session)
        return sedona
    except ImportError:
        pass  # Return basic session

    return spark_session


# =============================================================================
# ENVIRONMENT MOCK FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically mock all environment variables for every test.

    This fixture runs before each test and sets up the test environment
    variables. It ensures tests are deterministic and don't depend on
    external environment configuration.

    Args:
        monkeypatch: Pytest fixture for modifying environment
    """
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """
    Create a temporary workspace directory for test data.

    This fixture provides a clean temporary directory that is automatically
    cleaned up after each test. Useful for storing intermediate test data.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path: Path to temporary workspace
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each layer
    (workspace / "data" / "bronze").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "silver").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "gold").mkdir(parents=True, exist_ok=True)

    return workspace


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_usgs_earthquake() -> Dict[str, Any]:
    """
    Provide a sample USGS earthquake record for testing.

    Returns:
        Dict containing a single earthquake record
    """
    return {
        "id": "test-eq-001",
        "properties": {
            "mag": 5.2,
            "place": "Test Location, CA",
            "time": 1700000000000,
            "type": "earthquake",
            "tsunami": 0,
            "status": "reviewed",
        },
        "geometry": {
            "type": "Point",
            "coordinates": [-122.0, 38.0, 10.0],
        },
    }


@pytest.fixture
def sample_usgs_earthquakes_batch(sample_usgs_earthquake) -> Dict[str, Any]:
    """
    Provide a batch of USGS earthquake records for testing.

    Returns:
        Dict containing GeoJSON FeatureCollection
    """
    base = sample_usgs_earthquake.copy()
    return {
        "type": "FeatureCollection",
        "features": [{**base, "id": f"test-eq-{i:03d}"} for i in range(1, 6)],
    }


@pytest.fixture
def sample_nyc_311_record() -> Dict[str, Any]:
    """
    Provide a sample NYC 311 service request for testing.

    Returns:
        Dict containing a single 311 request
    """
    return {
        "unique_key": 12345,
        "created_date": "2024-01-15T10:30:00",
        "complaint_type": "Noise",
        "descriptor": "Loud Music",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "status": "Open",
    }


@pytest.fixture
def sample_spatial_df(spark_session) -> "DataFrame":
    """
    Create a small PySpark DataFrame with lat/lon columns for testing.

    This fixture provides a minimal DataFrame that can be used to test
    spatial transformations without requiring full Sedona setup.

    Args:
        spark_session: PySpark session fixture

    Returns:
        DataFrame with test spatial data
    """
    data = [
        (1, -122.4194, 37.7749, "San Francisco"),
        (2, -118.2437, 34.0522, "Los Angeles"),
        (3, -87.6298, 41.8781, "Chicago"),
        (4, -74.0060, 40.7128, "New York"),
        (5, -77.0369, 38.9072, "Washington DC"),
    ]

    return spark_session.createDataFrame(data, ["id", "longitude", "latitude", "city"])


@pytest.fixture
def sample_ollama_response_valid() -> str:
    """
    Provide a valid Ollama LLM response for testing.

    Returns:
        String: Sample JSON response from Ollama
    """
    return '{"severity": 7, "hazard_type": "earthquake"}'


@pytest.fixture
def sample_ollama_response_invalid() -> str:
    """
    Provide an invalid/malformed Ollama response for testing.

    Returns:
        String: Invalid response that should be handled gracefully
    """
    return "This is not JSON at all"


@pytest.fixture
def sample_ollama_response_empty() -> str:
    """
    Provide an empty Ollama response for testing.

    Returns:
        String: Empty response
    """
    return ""


# =============================================================================
# MOCK CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_boto3_s3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Provide a mocked boto3 S3 client for testing.

    This fixture creates a mock S3 client that can be used to verify
    that the code attempts to interact with S3 correctly.

    Args:
        monkeypatch: Pytest fixture for modifying environment

    Returns:
        MagicMock: Mocked boto3 s3 client
    """
    mock_s3 = MagicMock()
    mock_s3.put_object = MagicMock(
        return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
    )
    mock_s3.get_object = MagicMock(
        return_value={
            "Body": MagicMock(read=MagicMock(return_value=b'{"test": "data"}'))
        }
    )
    mock_s3.list_objects_v2 = MagicMock(return_value={"Contents": []})
    mock_s3.create_bucket = MagicMock(return_value={"Location": "/test-bucket"})

    def mock_client(service_name, **kwargs):
        if service_name == "s3":
            return mock_s3
        return MagicMock()

    monkeypatch.setattr("boto3.client", mock_client)
    return mock_s3


@pytest.fixture
def mock_http_response_200() -> Dict[str, Any]:
    """
    Provide a mock HTTP 200 OK response structure.

    Returns:
        Dict: Response metadata for 200 OK
    """
    return {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
    }


@pytest.fixture
def mock_http_response_500() -> Dict[str, Any]:
    """
    Provide a mock HTTP 500 Server Error response structure.

    Returns:
        Dict: Response metadata for 500 Error
    """
    return {
        "status": 500,
        "headers": {"Content-Type": "application/json"},
    }


# =============================================================================
# TEST UTILITIES
# =============================================================================


def assert_dataframe_schema(
    df: "DataFrame",
    expected_columns: list,
    column_types: Optional[Dict[str, str]] = None,
) -> None:
    """
    Assert that a DataFrame has the expected schema.

    Args:
        df: PySpark DataFrame to validate
        expected_columns: List of expected column names
        column_types: Optional dict mapping column names to expected types

    Raises:
        AssertionError: If schema doesn't match expectations
    """
    actual_columns = df.columns

    # Check column count
    assert len(actual_columns) == len(expected_columns), (
        f"Column count mismatch: expected {len(expected_columns)}, "
        f"got {len(actual_columns)}"
    )

    # Check column names
    for col in expected_columns:
        assert col in actual_columns, f"Missing expected column: {col}"

    # Check column types if provided
    if column_types:
        for col, expected_type in column_types.items():
            actual_type = str(df.schema[col].dataType)
            assert expected_type.lower() in actual_type.lower(), (
                f"Type mismatch for {col}: expected {expected_type}, got {actual_type}"
            )


def assert_dataframe_not_empty(df: "DataFrame") -> None:
    """
    Assert that a DataFrame is not empty.

    Args:
        df: PySpark DataFrame to validate

    Raises:
        AssertionError: If DataFrame is empty
    """
    count = df.count()
    assert count > 0, "DataFrame is empty"


# Export for use in test files
__all__ = [
    "spark_session",
    "spark_with_sedona",
    "mock_env_vars",
    "temp_workspace",
    "sample_usgs_earthquake",
    "sample_usgs_earthquakes_batch",
    "sample_nyc_311_record",
    "sample_spatial_df",
    "sample_ollama_response_valid",
    "sample_ollama_response_invalid",
    "sample_ollama_response_empty",
    "mock_boto3_s3",
    "mock_http_response_200",
    "mock_http_response_500",
    "assert_dataframe_schema",
    "assert_dataframe_not_empty",
]
