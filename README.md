# GeoAI Medallion Architecture Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=flat&logo=apache-spark&logoColor=white" alt="Spark">
  <img src="https://img.shields.io/badge/Apache%20Sedona-1.5.0-4CAF50?style=flat" alt="Sedona">
  <img src="https://img.shields.io/badge/Delta%20Lake-3.1-1E88E5?style=flat" alt="Delta Lake">
  <img src="https://img.shields.io/badge/MinIO-FF6F00?style=flat&logo=minio&logoColor=white" alt="MinIO">
  <img src="https://img.shields.io/badge/Ollama-latest-FF4081?style=flat" alt="Ollama">
  <img src="https://img.shields.io/badge/OpenTelemetry-latest-28a745?style=flat&logo=opentelemetry" alt="OpenTelemetry">
  <img src="https://img.shields.io/badge/MLflow-latest-0194E2?style=flat&logo=mlflow" alt="MLflow">
  <img src="https://img.shields.io/badge/Blender-latest-E58E00?style=flat&logo=blender" alt="Blender">
</p>

A production-ready **Local Databricks Clone** for GeoAI portfolio projects using Medallion Architecture with Apache Spark, Apache Sedona, Delta Lake, MinIO, OpenTelemetry, MLflow, Blender 3D visualization, and local LLM integration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Data & Model Flow](#data-and-model-flow)
4. [Web App Architecture](#web-app-architecture)
5. [Getting Started](#getting-started)
6. [Pipeline Components](#pipeline-components)
7. [Testing Pyramid](#testing-pyramid)
8. [Blender 3D Integration](#blender-3d-integration)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)
11. [API References](#api-references)

---

## Architecture Overview

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph Sources["📥 Data Sources"]
        direction LR
        S1[("US Accidents<br/>Kaggle CSV")]
        S2[("US Neighborhoods<br/>Kaggle GeoJSON")]
        S3[("USGS Earthquakes<br/>Live API")]
        S4[("OSM Hospitals/Fire<br/>Overpass API")]
        S5[("NYC 311<br/>Socrata API")]
        S6[("Aviation Data<br/>OpenSky Network")]
        S7[("Weather Data<br/>NOAA NWS")]
        S8[("NYC Land Mask<br/>NYC Open Data NTA")]
    end
    
    subgraph Bronze["<b>🥉 Bronze Layer:</b> Raw Ingestion"]
        B[("MinIO<br/>s3://geo-lakehouse/bronze")]
    end
    
    subgraph Silver["<b>🥈 Silver Layer:</b> Spatial Transform + Land Mask"]
        ST[("Apache Sedona<br/>ST_Point, ST_GeomFromGeoJSON<br/>EPSG:4326")]
        Mask[("Land Mask<br/>ST_Within Filter")]
        SDelta[("Delta Lake<br/>ACID Transactions")]
    end
    
    subgraph Gold["<b>🥇 Gold Layer:</b> Star Schema + AI + MLflow"]
        Ollama[("Ollama<br/>llama3.2:1b")]
        SJ[("Sedona Spatial Joins<br/>ST_Within, ST_Distance")]
        GF[("Gold Delta Tables<br/>Fact + Dimensions")]
        MLflow[("MLflow<br/>Tracking & Registry")]
    end
    
    subgraph Viz["🎨 Visualization"]
        WA["Next.js Web App"]
        B3D["Blender 3D Simulation"]
    end
    
    Sources --> Bronze
    Bronze --> ST
    ST --> Mask
    S8 -.->|Land mask<br/>filtering| Mask
    Mask --> SDelta
    SDelta --> Ollama
    Ollama --> SJ
    SJ --> GF
    GF -.->|Model Tracking| MLflow
    GF --> WA
    GF --> B3D
```

---

## Data and Model Flow

### Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Bronze as Bronze Ingestion
    participant MinIO as MinIO Storage
    participant Spark as Spark Cluster
    participant Sedona as Apache Sedona
    participant OTel as OpenTelemetry
    participant MLflow as MLflow
    participant Ollama as Ollama LLM
    participant Gold as Gold Layer
    
    User->>Bronze: Run ingestion
    Bronze->>MinIO: Upload raw data
    OTel->>Spark: Trace pipeline execution
    
    Bronze->>Spark: Trigger silver transform
    Spark->>Sedona: Apply ST_Point, ST_Within (Land Mask)
    Sedona-->>Spark: Cleaned & Filtered DataFrames
    Spark->>MinIO: Write Silver Delta Tables
    
    Spark->>Gold: Trigger gold enrichment
    Gold->>Ollama: Extract severity/hazard_type/flight_risk
    Ollama-->>Gold: JSON enrichment
    Gold->>Sedona: ST_Within, ST_Distance joins
    Gold->>MLflow: Log Model Metrics/Run
    Gold->>MinIO: Write Gold Delta Tables
    
    User->>Gold: Query results
    User->>MLflow: Inspect Training Runs
```

### MLflow Model Training Flow

```mermaid
flowchart TB
    subgraph Training["🚀 Model Training & Tracking"]
        D[("Gold Delta Tables")] -->|"Load"| T("Training Scripts<br/>train_aviation_models.py")
        T -->|"Train"| M("Model (Scikit-Learn/XGBoost)")
        M -->|"Log params, metrics, model"| ML("MLflow Tracking")
        ML -->|"Register Version"| MR("MLflow Model Registry")
    end
    
    subgraph Inference["🔮 Production Inference"]
        MR -->|"Deploy"| S("Inference Service")
        S -->|"Score"| NewD("New Data")
    end
```

---

## Web App Architecture

```mermaid
flowchart TD
    User["🌍 End User"]
    subgraph Frontend["🎨 Next.js (Client)"]
        UI["React / MapClient.tsx"]
        L["Leaflet.js<br/>Heatmap / Markers"]
    end
    
    subgraph Backend["⚙️ Pipeline Data"]
        API["/public/data/heatmap.geojson"]
    end
    
    User --> UI
    UI --> L
    L -- "Fetch" --> API
```

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Compute Engine** | Apache Spark (PySpark) | 3.5.0 | Distributed processing |
| **Spatial Engine** | Apache Sedona | 1.5.0 | Geospatial SQL on Spark |
| **Storage** | Delta Lake | 3.1.0 | ACID on data lake |
| **Object Storage** | MinIO | Latest | S3-compatible storage |
| **Observability** | OpenTelemetry | Latest | Metrics & Trace collection |
| **Database** | PostgreSQL | 15 | Hive Metastore + MLflow DB |
| **MLOps** | MLflow | 2.10.0 | Experiment tracking |
| **LLM** | Ollama | Latest | Local LLM inference |
| **Viz** | Blender | Latest | 3D Simulation Rendering |

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

# 7. Run Gold enrichment & Model training
spark-submit src/gold_schema_and_ai_enrichment.py
python src/train_aviation_models.py
```

---

## Pipeline Components

### 1. Bronze Layer (Ingestion)

**File:** `src/bronze_ingestion.py`

### 2. Silver Layer (Spatial Transform + Land Masking)

**File:** `src/silver_sedona_transform.py`

### 3. Gold Layer (AI + Spatial Joins + MLflow)

**File:** `src/gold_schema_and_ai_enrichment.py`

### 4. Training (MLflow)

**File:** `src/train_aviation_models.py`

---

## Testing Pyramid

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

---

## Blender 3D Integration

**File:** `scripts/render_flight_simulation.py`

Automates 3D data visualization from Gold layer Parquet files. Uses Python API within Blender to procedurally generate NYC flight simulation environments.

```bash
# Generate 3D simulation
blender -b examples/nyc_ml_heatmap.blend -P scripts/render_flight_simulation.py
```

---

## Heatmap Visualization

**File:** `scripts/quick_heatmap.py`

Reads gold layer parquet files and generates a Leaflet heatmap.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Spark not connecting to MinIO | Check MinIO endpoint in config |
| Ollama API timeout | Increase `OLLAMA_TIMEOUT` |
| Sedona functions not found | Ensure Sedona plugin registered |
| Missing telemetry data | Ensure OpenTelemetry collector is running |
| MLflow not tracking | Check `MLFLOW_TRACKING_URI` environment variable |

---

## Credits

- [Apache Sedona](https://sedona.apache.org/) - Geospatial SQL on Spark
- [Delta Lake](https://delta.io/) - ACID transactions on data lakes
- [Ollama](https://ollama.ai/) - Local LLM inference
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework
- [MLflow](https://mlflow.org/) - MLOps platform
- [Blender](https://www.blender.org/) - 3D Content Creation

---

<p align="center">
  Built with ❤️ for GeoAI Portfolio Projects
</p>
