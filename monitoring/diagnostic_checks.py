#!/usr/bin/env python3
"""
Diagnostic checks for GeoAI Medallion Architecture with MLflow integration.
This script performs pre-flight checks before running the main pipeline.
"""

import os
import sys
import subprocess
import requests
import json

def check_python_version():
    """Check Python version meets requirements."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ is required")
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")

def check_mlflow_installed():
    """Check if MLflow is installed and meets version requirements."""
    print("Checking MLflow installation...")
    try:
        import mlflow
        version = mlflow.__version__
        major, minor, _ = map(int, version.split('.')[:3])
        if major < 2 or (major == 2 and minor < 14):
            raise RuntimeError(f"MLflow 2.14+ required, found {version}")
        print(f"✓ MLflow {version} is installed")
    except ImportError:
        raise RuntimeError("MLflow is not installed. Run: pip install mlflow==2.14.*")

def check_ollama_availability():
    """Check if Ollama is available and models are loaded."""
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    print(f"Checking Ollava availability at {ollama_host}...")
    
    try:
        # Check if Ollama is reachable
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✓ Ollama API is accessible")
            
            # Check for required models
            models_response = requests.get(f"{ollama_host}/api/tags", timeout=10)
            if models_response.status_code == 200:
                models = models_response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                required_models = ['llama3', 'deepseek-r1']
                for model in required_models:
                    if any(model in name.lower() for name in model_names):
                        print(f"✓ {model} model is available")
                    else:
                        print(f"⚠️  {model} model not found (optional)")
        else:
            print("⚠️  Ollama API returned unexpected status")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Ollama not accessible: {e}")
        print("   LLM features will be limited")

def check_mlflow_server():
    """Check if MLflow tracking server is accessible."""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    print(f"Checking MLflow server at {tracking_uri}...")
    
    try:
        response = requests.get(f"{tracking_uri}/api/2.0/mlflow/version", timeout=10)
        if response.status_code == 200:
            print(f"✓ MLflow server is accessible")
        else:
            print(f"⚠️  MLflow server returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  MLflow server not accessible: {e}")
        print("   Consider starting MLflow server: mlflow server --host 0.0.0.0 --port 5000")

def check_spark_environment():
    """Check if PySpark environment is properly configured."""
    print("Checking PySpark environment...")
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("DiagnosticCheck") \
            .getOrCreate()
        
        print(f"✓ PySpark session created (version: {spark.version})")
        spark.stop()
    except Exception as e:
        print(f"⚠️  PySpark environment issue: {e}")
        print("   Ensure Spark is properly installed")

def check_disk_space():
    """Check available disk space for MLflow runs."""
    print("Checking disk space...")
    try:
        stat = os.statvfs('/tmp')
        available_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
        
        if available_gb > 10:
            print(f"✓ Sufficient disk space: {available_gb:.1f} GB available")
        else:
            print(f"⚠️  Low disk space: {available_gb:.1f} GB available")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")

def check_dependencies():
    """Check all required Python packages."""
    print("Checking Python dependencies...")
    required_packages = ['mlflow', 'pyspark', 'pandas', 'numpy']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install {' '.join(required_packages)}")
    else:
        print("✓ All required packages are installed")

def run_all_checks():
    """Run all diagnostic checks."""
    print("Running diagnostic checks for GeoAI Medallion Architecture...")
    print(f"Started at: {__import__('datetime').datetime.now().isoformat()}\n")
    
    try:
        check_python_version()
        check_mlflow_installed()
        check_ollama_availability()
        check_mlflow_server()
        check_spark_environment()
        check_disk_space()
        check_dependencies()
        
        print("\n✅ All diagnostic checks completed!")
        return True
        
    except RuntimeError as e:
        print(f"\n❌ Diagnostic check failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)