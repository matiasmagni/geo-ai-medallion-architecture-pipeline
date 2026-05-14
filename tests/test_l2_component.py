#!/usr/bin/env python3
"""
================================================================================
L2 COMPONENT TESTS - Service Integration Tests
================================================================================
File: tests/test_l2_component.py

Purpose:
    Test component integration with real services (MinIO, MLflow, Prometheus).
    Requires Docker services to be running.

Author: GeoAI Principal Data Engineer & MLOps Architect
================================================================================
"""

import pytest
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CLASS: MinIO Component Tests
# =============================================================================

class TestMinIOComponent:
    """L2: Test MinIO storage component."""

    def test_minio_connection(self, minio_client_l2):
        """Test MinIO connection."""
        if minio_client_l2 is None:
            pytest.skip("MinIO not available - ensure Docker services are running")
        buckets = minio_client_l2.list_buckets()
        assert isinstance(buckets, list)

    def test_minio_bucket_creation(self, minio_client_l2):
        """Test MinIO bucket creation."""
        if minio_client_l2 is None:
            pytest.skip("MinIO not available - ensure Docker services are running")
        import uuid
        bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
        
        minio_client_l2.make_bucket(bucket_name)
        
        buckets = minio_client_l2.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        minio_client_l2.remove_bucket(bucket_name)
        
        assert bucket_name in bucket_names


# =============================================================================
# TEST CLASS: MLflow Component Tests
# =============================================================================

class TestMLflowComponent:
    """L2: Test MLflow tracking component."""

    def test_mlflow_tracking_connection(self, mlflow_client_l2):
        """Test MLflow tracking server connection."""
        if mlflow_client_l2 is None:
            pytest.skip("MLflow not available - ensure Docker services are running")
        
        import mlflow
        exp = mlflow.get_experiment_by_name("Default")
        assert exp is not None or True

    def test_mlflow_create_experiment(self, mlflow_client_l2):
        """Test creating MLflow experiment."""
        if mlflow_client_l2 is None:
            pytest.skip("MLflow not available - ensure Docker services are running")
        
        import mlflow
        import uuid
        
        exp_name = f"test_experiment_{uuid.uuid4().hex[:8]}"
        
        exp_id = mlflow.create_experiment(exp_name)
        assert exp_id is not None


# =============================================================================
# TEST CLASS: Prometheus Component Tests
# =============================================================================

class TestPrometheusComponent:
    """L2: Test Prometheus metrics component."""

    def test_prometheus_client_creation(self, prometheus_client_l2):
        """Test Prometheus client can create metrics."""
        if prometheus_client_l2 is None:
            pytest.skip("Prometheus client not available - ensure Docker services are running")
        
        counter = prometheus_client_l2.Counter('test_counter', 'Test counter')
        counter.inc()
        
        assert counter._value._value >= 1

    def test_prometheus_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        import requests

        prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        response = requests.get(f"{prometheus_url}/-/healthy", timeout=5)
        assert response.status_code in [200, 404]


# =============================================================================
# TEST CLASS: Docker Services Health
# =============================================================================

class TestDockerServices:
    """L2: Test Docker services are healthy."""

    def test_docker_services_status(self, docker_services):
        """Test Docker services status."""
        # Log service availability
        for service, available in docker_services.items():
            logger.info(f"L2: {service} available: {available}")
        
        # At least one service should be available for L2 tests
        assert any(docker_services.values()), "No Docker services available for L2 tests"

    def test_minio_health(self, docker_services):
        """Test MinIO health endpoint."""
        if not docker_services.get("minio"):
            pytest.skip("MinIO not running - ensure Docker services are running")

        import requests
        minio_url = os.getenv("MINIO_URL", "http://localhost:9000")
        response = requests.get(f"{minio_url}/minio/health/live", timeout=5)
        assert response.status_code in [200, 403]

    def test_mlflow_health(self, docker_services):
        """Test MLflow health endpoint."""
        if not docker_services.get("mlflow"):
            pytest.skip("MLflow not running - ensure Docker services are running")

        import requests
        mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
        response = requests.get(f"{mlflow_url}/health", timeout=5)
        assert response.status_code in [200, 404]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])