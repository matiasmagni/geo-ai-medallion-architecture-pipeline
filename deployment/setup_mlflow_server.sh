#!/bin/bash
# Setup MLflow tracking server for production deployment

set -e

echo "Setting up MLflow tracking server..."

# Create necessary directories
mkdir -p /data/mlflow_runs
mkdir -p /data/mlflow_artifacts

# Set environment variables
export MLFLOW_TRACKING_URI="file:///data/mlflow_runs"
export MLFLOW_ARTIFACT_LOCATION="file:///data/mlflow_artifacts"

# Start MLflow server
echo "Starting MLflow tracking server..."
nohup mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////data/mlflow_runs/mlflow.db \
    --default-artifact-root file:///data/mlflow_artifacts \
    --serve-artifacts \
    --workers 4 \
    > /var/log/mlflow_server.log 2>&1 &

# Wait for server to start
echo "Waiting for MLflow server to start..."
sleep 10

# Check if server is running
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ MLflow server is running on http://localhost:5000"
    echo "   UI available at: http://localhost:5000"
    echo "   Tracking URI: ${MLFLOW_TRACKING_URI}"
else
    echo "❌ MLflow server failed to start. Check /var/log/mlflow_server.log"
    exit 1
fi

# Optional: Set up authentication if needed
# echo "Setting up authentication..."
# mlflow auth-model create --name local-file --config-file auth_config.yaml