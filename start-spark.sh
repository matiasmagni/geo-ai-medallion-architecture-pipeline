#!/bin/bash
# Spark startup script - v4.0

set -e

echo "=== Starting GeoAI Spark ==="

# Find SPARK_HOME
SPARK_HOME="${SPARK_HOME:-/usr/local/spark}"
echo "SPARK_HOME: ${SPARK_HOME}"

# Check actual user
echo "Running as: $(whoami)"

# Set Java - Use the ARM64 path from docker-compose.yml
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-arm64"
export SPARK_JAVA_OPTS="$SPARK_JAVA_OPTS --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
echo "JAVA_HOME: ${JAVA_HOME}"

# Inject JVM option to disable security manager
# Removed '-Djava.security.manager=allow' as it caused JVM initialization errors.
# Keeping '--add-opens' flags for Java 17+ compatibility.
export SPARK_JAVA_OPTS="$SPARK_JAVA_OPTS --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"

# Create logs dir in user's home (jovyan)
LOG_DIR="${HOME}/spark_logs"
echo "Creating log dir in home: ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
chmod 755 "${LOG_DIR}"

# Start Spark Master
echo "Starting Spark Master..."
${SPARK_HOME}/sbin/start-master.sh

# Start Spark Worker
echo "Starting Spark Worker..."
${SPARK_HOME}/sbin/start-worker.sh spark://spark:7077

# Install Python packages
echo "Installing Python packages..."
pip install --no-cache-dir pyspark==3.5.0 boto3 delta-spark requests -q

# S3 jars
echo "Setting up S3 jars..."
mkdir -p /home/jovyan/jars
cd /home/jovyan/jars
[ ! -f hadoop-aws.jar ] && wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar -O hadoop-aws.jar
[ ! -f aws-sdk.jar ] && wget -q https://repo1.maven.org/maven2/software/amazon/aws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar -O aws-sdk.jar
ls -la /home/jovyan/jars/

echo "=== Spark Ready ==="
exec jupyter lab --ServerApp.token=''