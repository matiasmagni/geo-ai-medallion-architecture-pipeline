# =============================================================================
# GeoAI Workspace - JupyterLab Dockerfile
# =============================================================================
# This Dockerfile builds the development environment for:
# - Apache Spark (PySpark) with Sedona spatial extensions
# - Delta Lake for ACID transactions
# - MLflow for experiment tracking
# - Ollama Python client for LLM inference
# - Boto3 for S3/MinIO access
#
# Base: JupyterLab with PySpark support
# =============================================================================

FROM jupyter/pyspark-notebook:spark-3.5.0

# =============================================================================
# LABEL METADATA
# =============================================================================
LABEL maintainer="GeoAI Team <dev@geospatial-ai.local>"
LABEL description="GeoAI Medallion Architecture - JupyterLab Workspace"
LABEL version="1.0.0"

# =============================================================================
# ENVIRONMENT VARIABLES (Default values for local development)
# =============================================================================
ENV SPARK_VERSION=3.5.0 \
    HADOOP_VERSION=3.4.0 \
    PYSPARK_VERSION=3.5.0 \
    PYTHON_VERSION=3.11

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
USER root

# Install system-level dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Geometry libraries
    libgeos-dev \
    libproj-dev \
    libproj-data \
    # GeoTIFF and raster support
    libtiff-dev \
    libpng-dev \
    # Additional build tools
    cmake \
    build-essential \
    # PostgreSQL client for metastore testing
    libpq-dev \
    # curl for health checks
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# =============================================================================
# WORKING DIRECTORY
# =============================================================================
WORKDIR /home/${NB_USER}

# =============================================================================
# PYTHON DEPENDENCIES
# =============================================================================
# Create conda environment matching Spark version
RUN conda create -y -n geoai && \
    conda run -n geoai pip install --no-cache-dir && \
    conda clean -afy

# Install core PyData stack
RUN conda run -n geoai pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    folium \
    geopandas \
    shapely

# Install PySpark with matching version
RUN conda run -n geoai pip install --no-cache-dir \
    pyspark==${PYSPARK_VERSION}

# Install Apache Sedona (Spatial SQL on Spark)
RUN conda run -n geoai pip install --no-cache-dir \
    apache-sedona[spark]=='1.5.0' \
    sedona-sql-3-4=='1.5.0' \
    sedona-core-3-4=='1.5.0'

# Install Delta Lake for ACID transactions
RUN conda run -n geoai pip install --no-cache-dir \
    delta-spark=='3.1.0'

# Install MLflow for experiment tracking
RUN conda run -n geoai pip install --no-cache-dir \
    mlflow==2.10.0 \
    boto3==1.34.0

# Install Ollama Python client for LLM inference
RUN conda run -n geoai pip install --no-cache-dir \
    ollama==0.1.0 \
    httpx==0.26.0

# Install PostgreSQL adapter for Hive Metastore
RUN conda run -n geoai pip install --no-cache-dir \
    psycopg2-binary==2.9.9

# Install geospatial utilities
RUN conda run -n geoai pip install --no-cache-dir \
    rtree==1.1.0 \
    pygeocode==1.0.2 \
    geocoder==1.38.1

# Install notebook extensions
RUN conda run -n geoai pip install --no-cache-dir \
    ipywidgets==8.1.1 \
    ipyleaflet==0.17.0 \
    ipyvolume==0.6.0

# =============================================================================
# ENVIRONMENT EXTENSIONS
# =============================================================================
# Copy custom Jupyter configuration
COPY jupyter_notebook_config.py /home/${NB_USER}/.jupyter/
RUN chmod 644 /home/${NB_USER}/.jupyter/jupyter_notebook_config.py

# Copy startup scripts
COPY scripts/ /home/${NB_USER}/scripts/
RUN chmod +x /home/${NB_USER}/scripts/*.sh

# =============================================================================
# JUPYTER CONFIGURATION
# =============================================================================
# Set default Python environment
ENV PYTHONPATH=/opt/conda/envs/geoai/bin/python
ENV JAVA_HOME=/usr/local/openjdk

# Expose Jupyter port
EXPOSE 8888

# =============================================================================
# HEALTH CHECK
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# =============================================================================
# DEFAULT COMMAND
# =============================================================================
# Start JupyterLab with extended timeout
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'", "--NotebookApp.quit_button=False"]