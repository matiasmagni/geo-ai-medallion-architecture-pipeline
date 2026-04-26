#!/bin/bash
# Spark startup script with all fixes

set -e

echo "=== Starting GeoAI Spark ==="

# Ensure Spark logs directory exists and has correct permissions
SPARK_LOG_DIR="/usr/local/spark/logs"
echo "Ensuring Spark log directory exists: ${SPARK_LOG_DIR}"
# Create directory with broad permissions for write access
mkdir -p "${SPARK_LOG_DIR}"
chmod 777 "${SPARK_LOG_DIR}" 

# Start Spark Master
echo "Starting Spark Master..."
/usr/local/spark/sbin/start-master.sh

# Start Spark Worker
# Ensure it connects to the correct master URL.
# Use the internal service name 'spark' and the master port 7077.
echo "Starting Spark Worker..."
/usr/local/spark/sbin/start-worker.sh spark://spark:7077

# Install Python packages
echo "Installing Python packages..."
pip install --no-cache-dir pyspark==3.5.0 boto3 delta-spark==3.1.0 requests -q

# Download S3 jars
echo "Downloading S3 jars..."
mkdir -p /home/jovyan/jars
cd /home/jovyan/jars

# Only download if not present
if [ ! -f hadoop-aws.jar ]; then
    echo "Downloading hadoop-aws.jar..."
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -O hadoop-aws.jar
fi

if [ ! -f aws-sdk.jar ]; then
    echo "Downloading aws-sdk.jar..."
    wget -q https://repo1.maven.org/maven2/software/amazon/aws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar -O aws-sdk.jar
fi

ls -la /home/jovyan/jars/

echo "Starting Jupyter Lab..."
exec jupyter lab --ServerApp.token=''
