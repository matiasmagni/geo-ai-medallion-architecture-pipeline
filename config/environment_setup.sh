#!/bin/bash
# Environment setup for GeoAI Medallion Architecture with MLflow

set -e  # Exit on error

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is required but not installed."
    exit 1
fi

# Install required Python packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install mlflow==2.14.*
pip install pyspark pandas numpy scikit-learn requests

# Check if Ollama is available (optional)
if command -v ollama &> /dev/null; then
    echo "Ollama is installed. Checking if models are available..."
    ollama list
else
    echo "⚠️  Ollama not found. LLM functionality will be limited."
    echo "   Install Ollama from https://ollama.com/ to use LLM features."
fi

# Check MLflow installation
echo "Checking MLflow installation..."
python3 -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"

# Check if running in Colab or similar environment
if [ -f "/content/drive/MyDrive" ]; then
    echo "Detected Colab environment. Setting up paths..."
    export MLFLOW_TRACKING_URI="file:///content/drive/MyDrive/mlruns"
    export OLLAMA_HOST="http://localhost:11434"
elif [ -f "/mnt/data" ]; then
    echo "Detected mounted data drive. Setting up paths..."
    export MLFLOW_TRACKING_URI="file:///mnt/data/mlruns"
    export OLLAMA_HOST="http://localhost:11434"
fi

# Create necessary directories
mkdir -p /tmp/mlflow_runs
mkdir -p /tmp/geoai_medallion

echo ""
echo "✅ Environment setup complete!"
echo "MLflow tracking URI: ${MLFLOW_TRACKING_URI:-http://localhost:5000}"
echo "OLLAMA_HOST: ${OLLAMA_HOST:-http://localhost:11434}"
echo ""
echo "To start MLflow tracking server (optional):"
echo "  mlflow server --host 0.0.0.0 --port 5000"
echo ""
echo "To run the pipeline:"
echo "  python pipeline_orchestrator.py"