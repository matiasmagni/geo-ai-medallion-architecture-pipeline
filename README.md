# GeoAI Medallion Architecture Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=flat&logo=apache-spark&logoColor=white" alt="Spark">
  <img src="https://img.shields.io/badge/Apache%20Sedona-1.5.0-4CAF50?style=flat" alt="Sedona">
  <img src="https://img.shields.io/badge/Delta%20Lake-3.1-1E88E5?style=flat" alt="Delta Lake">
  <img src="https://img.shields.io/badge/MinIO-FF6F00?style=flat&logo=minio&logoColor=white" alt="MinIO">
  <img src="https://img.shields.io/badge/Ollama-latest-FF4081?style=flat" alt="Ollama">
</p>

A production-ready **Local Databricks Clone** for GeoAI portfolio projects using Medallion Architecture with Apache Spark, Apache Sedona, Delta Lake, MinIO, and local LLM integration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Data Flow](#data-flow)
4. [Getting Started](#getting-started)
5. [Pipeline Components](#pipeline-components)
6. [Testing Pyramid](#testing-pyramid)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)
9. [API References](#api-references)

---

## Architecture Overview

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph Sources["📥 Data Sources"]
        direction LR
        S1[("US Accidents<br/>Kaggle CSV")] ~~~ S2[("US Neighborhoods<br/>Kaggle GeoJSON")] ~~~ S3[("USGS Earthquakes<br/>Live API")] ~~~ S4[("OSM Hospitals/Fire<br/>Overpass API")] ~~~ S5[("NYC 311<br/>Socrata API")]
    end
    
    subgraph Bronze["🥉 Bronze Layer: Raw Ingestion"]
        B[("MinIO<br/>s3://geo-lakehouse/bronze")]
    end
    
    subgraph Silver["🥈 Silver Layer: Spatial Transform"]
        ST[("Apache Sedona<br/>ST_Point, ST_GeomFromGeoJSON<br/>EPSG:4326")]
        SDelta[("Delta Lake<br/>ACID Transactions")]
    end
    
    subgraph Gold["🥇 Gold Layer: Star Schema + AI"]
        Ollama[("Ollama<br/>llama3")]
        SJ[("Sedona Spatial Joins<br/>ST_Within, ST_Distance")]
        GF[("Gold Delta Tables<br/>Fact + Dimensions")]
    end
    
    Sources --> Bronze
    Bronze --> ST
    ST --> SDelta
    SDelta --> Ollama
    Ollama --> SJ
    SJ --> GF
```

### Medallion Architecture Layers

```mermaid
flowchart TB
    subgraph BRONZE["🥉 Bronze: Raw Data"]
        direction LR
        B1[US Accidents CSV] ~~~ B2[USGS JSON] ~~~ B3[OSM GeoJSON]
    end
    
    subgraph SILVER["🥈 Silver: Cleaned & Spatial"]
        direction LR
        S1[Accidents + Geometry] ~~~ S2[Earthquakes + Geometry] ~~~ S3[Neighborhoods + Polygons]
    end
    
    subgraph GOLD["🥇 Gold: Enriched & Joined"]
        direction LR
        G1[FACT_HAZARD_EVENTS] ~~~ G2[DIM_NEIGHBORHOODS] ~~~ G3[DIM_INFRASTRUCTURE]
    end
    
    BRONZE -->|"Ingest"| SILVER
    SILVER -->|"Transform"| GOLD
```

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Compute Engine** | Apache Spark (PySpark) | 3.5.0 | Distributed processing |
| **Spatial Engine** | Apache Sedona | 1.5.0 | Geospatial SQL on Spark |
| **Storage** | Delta Lake | 3.1.0 | ACID on data lake |
| **Object Storage** | MinIO | Latest | S3-compatible storage |
| **Database** | PostgreSQL | 15 | Hive Metastore + MLflow |
| **MLOps** | MLflow | 2.10.0 | Experiment tracking |
| **LLM** | Ollama | Latest | Local LLM inference |

---

## Data Flow

### Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Bronze as Bronze Ingestion
    participant MinIO as MinIO Storage
    participant Spark as Spark Cluster
    participant Sedona as Apache Sedona
    participant Ollama as Ollama LLM
    participant Gold as Gold Layer
    
    User->>Bronze: Run ingestion
    Bronze->>MinIO: Upload raw data (CSV/JSON)
    MinIO-->>Bronze: Confirm upload
    
    Bronze->>Spark: Trigger silver transform
    Spark->>Sedona: Apply ST_Point, ST_GeomFromGeoJSON
    Sedona-->>Spark: Cleaned DataFrames
    Spark->>MinIO: Write Silver Delta Tables
    
    Spark->>Gold: Trigger gold enrichment
    Gold->>Ollama: Extract severity/hazard_type
    Ollama-->>Gold: JSON enrichment
    Gold->>Sedona: ST_Within, ST_Distance joins
    Sedona-->>Gold: Enriched events
    Gold->>MinIO: Write Gold Delta Tables
    
    User->>Gold: Query results
```

### Data Source Integration

```mermaid
flowchart TB
    subgraph "📥 Ingestion Sources"
        direction LR
        K1[Kaggle<br/>US Accidents]
        K2[Kaggle<br/>Neighborhoods]
        U[USGS API<br/>Earthquakes]
        O[OSM Overpass<br/>Hospitals]
        N[NYC 311 API<br/>Requests]
    end
    
    subgraph "🪣 Bronze Layer"
        B[MinIO Bucket<br/>geo-lakehouse/bronze]
    end
    
    subgraph "🔧 Silver Processing"
        P[PySpark<br/>Sedona]
    end
    
    K1 --> B
    K2 --> B
    U --> B
    O --> B
    N --> B
    B --> P
```

---

## Getting Started

### Prerequisites

```bash
# Core requirements
Docker >= 20.10
Docker Compose >= 2.0
Python >= 3.9
8GB RAM (16GB recommended)
```

### Quick Start

```bash
# 1. Clone and navigate
cd geo-ai-medallion-architecture-pipeline

# 2. Copy environment template
cp .env.example .env

# 3. Start infrastructure
docker compose up -d

# 4. Verify services
./scripts/run_pipeline.sh status

# 5. Run ingestion (Bronze)
python src/bronze_ingestion.py

# 6. Run Silver transform
spark-submit src/silver_sedona_transform.py

# 7. Run Gold enrichment
spark-submit src/gold_schema_and_ai_enrichment.py
```

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| Spark UI | 9080 | http://localhost:9080 |
| MinIO Console | 9901 | http://localhost:9901 |
| PostgreSQL | 5434 | localhost:5434 |
| Ollama | 11434 | http://localhost:11434 |

---

## Pipeline Components

### 1. Bronze Layer (Ingestion)

**File:** `src/bronze_ingestion.py`

```python
# Key functions
def fetch_usgs_earthquakes(config, days_back=30, min_magnitude=2.0):
    """Fetch from USGS FDSN Web Services API"""

def fetch_osm_infrastructure(config, bbox=(-125, 24, -66, 50)):
    """Fetch hospitals/fire stations via Overpass API"""

def fetch_nyc_311_requests(config, limit=100000):
    """Fetch 311 service requests from Socrata"""
```

**Output:** Raw JSON/CSV in `s3://geo-lakehouse/bronze/`

---

### 2. Silver Layer (Spatial Transform)

**File:** `src/silver_sedona_transform.py`

```python
# Key transformations
def create_geometry_from_latlon(df, lat_col, lon_col):
    """ST_Point - Convert lat/lng to geometry"""
    return df.withColumn(
        "geometry",
        F.expr("ST_Point(cast(lon as double), cast(lat as double))")
    )

def parse_geojson_geometry(df, geojson_col):
    """ST_GeomFromGeoJSON - Parse GeoJSON polygons"""
```

**Output:** Delta Tables in `s3://geo-lakehouse/silver/`

---

### 3. Gold Layer (AI + Spatial Joins)

**File:** `src/gold_schema_and_ai_enrichment.py`

```python
# AI Enrichment UDF
def create_ollama_enrichment_udf(config):
    @pandas_udf("string")
    def ollama_hazard_udf(text_series):
        return text_series.apply(analyze_hazard)
    return ollama_hazard_udf

# Spatial Joins
def spatial_join_events_to_neighborhoods(fact_df, dim_neighborhoods):
    """ST_Within - Point in Polygon"""
    return fact_df.join(
        dim_neighborhoods,
        F.expr("ST_Within(fact.geometry, dim.geometry)")
    )
```

**Output:** Star Schema in `s3://geo-lakehouse/gold/`

---

## Testing Pyramid

This project follows the **test pyramid** methodology with four levels of testing:

```mermaid
block-beta
columns 13 

space:4 L3["🔴 L3: End-to-End<br/>Full Pipeline"]:5 space:4
space:3 L2["🟠 L2: Integration Services<br/>Real Services / Local Docker"]:7 space:3
space:1 L1["🟡 L1: Unit with Mocks<br/>Mocked Dependencies / Partial Mocks"]:11 space:1
L0["🟢 L0: Unit Isolation (Most Tests)<br/>Pure Functions / No I/O / No Dependencies"]:13

style L3 fill:#ff5722,color:#fff,stroke:#333,stroke-width:2px
style L2 fill:#ff9800,color:#fff,stroke:#333,stroke-width:2px
style L1 fill:#ffc107,color:#000,stroke:#333,stroke-width:2px
style L0 fill:#ffeb3b,color:#000,stroke:#333,stroke-width:2px
```

### Test Levels

| Level | File | Purpose | Dependencies |
|-------|------|---------|--------------|
| **L0** | `tests/L0.py` | Unit isolation, config, utils | None (mocked) |
| **L1** | `tests/L1.py` | Component integration | Mocked Spark/MinIO |
| **L2** | `tests/L2.py` | Pipeline stages | Docker services |
| **L3** | `tests/L3.py` | Full E2E | All running services |

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# By level
python -m pytest tests/L0.py -v  # Unit tests
python -m pytest tests/L1.py -v  # Integration
python -m pytest tests/L2.py -v  # Pipeline
python -m pytest tests/L3.py -v  # E2E
```

### Test Coverage Goals

```
L0: 80%+ (most tests - fast, no I/O)
L1: 15% (component integration)
L2: 4% (pipeline validation)
L3: 1% (smoke tests)
```

---

## Deployment

### Docker Compose Services

```yaml
services:
  spark:
    image: jupyter/pyspark-notebook:spark-3.5.0
    ports:
      - "9077:7077"  # Spark driver
      - "9080:8080"   # Spark UI
      
  minio:
    image: minio/minio:latest
    ports:
      - "9900:9000"
      - "9901:9001"
      
  postgres:
    image: postgres:15-alpine
    ports:
      - "5434:5432"
      
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

### Environment Variables

```bash
# Required
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin123
export POSTGRES_DB=geometastore
export POSTGRES_USER=geoai
export POSTGRES_PASSWORD=geoi_secure_pass_2024

# Optional (for Kaggle)
export KAGGLE_USERNAME=your_username
export KAGGLE_API_KEY=your_api_key
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Spark not connecting to MinIO | Check MinIO endpoint in config |
| Ollama API timeout | Increase `OLLAMA_TIMEOUT` |
| Sedona functions not found | Ensure Sedona plugin registered |
| Null geometries after transform | Check lat/lon column names |

### Debug Commands

```bash
# Check Spark logs
docker compose logs spark

# Test MinIO connectivity
docker exec geoai-spark python -c "import boto3; print('OK')"

# Verify Delta tables
spark.read.format("delta").load("s3://bucket/table").printSchema()
```

---

## API References

### Sedona Spatial Functions

```python
# Point from coordinates
ST_Point(longitude, latitude)

# GeoJSON to geometry
ST_GeomFromGeoJSON(geojson_string)

# Point in polygon
ST_Within(point_geom, polygon_geom)

# Distance between geometries (meters)
ST_Distance(point1, point2)

# Set CRS
ST_SetSRID(geometry, 4326)
```

### Delta Lake Operations

```python
# Read
spark.read.format("delta").load("s3://bucket/table")

# Write
df.write.format("delta").mode("overwrite").save("s3://bucket/table")

# Time travel
spark.read.format("delta").option("versionAsOf", 1).load("s3://bucket/table")
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests (follow the pyramid)
4. Ensure L0 tests pass
5. Submit a PR

---

## License

Apache 2.0 - See LICENSE file for details.

---

## Credits

- [Apache Sedona](https://sedona.apache.org/) - Geospatial SQL on Spark
- [Delta Lake](https://delta.io/) - ACID transactions on data lakes
- [Ollama](https://ollama.ai/) - Local LLM inference

---

<p align="center">
  Built with ❤️ for GeoAI Portfolio Projects
</p>
