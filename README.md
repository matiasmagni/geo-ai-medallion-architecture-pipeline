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
3. [Infrastructure](#infrastructure)
4. [Data & Model Flow](#data-and-model-flow)
5. [Web App Architecture](#web-app-architecture)
6. [Getting Started](#getting-started)
7. [Pipeline Components](#pipeline-components)
8. [Testing Pyramid](#testing-pyramid)
9. [Blender 3D Integration](#blender-3d-integration)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)
12. [API References](#api-references)

---

## Architecture Overview

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph Sources["📥 Data Sources"]
        direction LR
        S1[("US Accidents<br/>Kaggle CSV")] --> S2[("USGS Earthquakes<br/>Live API")]
        S3[("OSM Hospital/Fire<br/>Overpass API")] --> S4[("NYC 311<br/>Socrata API")]
        S5[("NYC Flights<br/>OpenSky API")] --> S6[("NYC Weather<br/>NOAA NWS API")]
        S7[("US Neighborhoods<br/>Kaggle GeoJSON")] --> S8[("NYC Land Mask<br/>NYC Open Data NTA")]
    end
    
    subgraph Bronze["<b>🥉 Bronze Layer:</b> Raw Ingestion"]
        B[("MinIO<br/>s3://geo-lakehouse/bronze")]
    end
    
    subgraph Silver["<b>🥈 Silver Layer:</b> Spatial Transform + Land Mask"]
        ST[("Apache Sedona<br/>ST_Point, ST_GeomFromGeoJSON<br/>EPSG:4326")]
        PANDAS[("Pandas<br/>Data Cleaning")]
        Mask[("Land Mask<br/>ST_Within Filter")]
        SDelta[("Delta Lake<br/>ACID Transactions")]
    end
    
    subgraph Gold["<b>🥇 Gold Layer:</b> Star Schema + AI + MLflow"]
        Ollama[("Ollama<br/>llama3.2:1b")]
        AI[("AI Enrichment<br/>severity, hazard_type")]
        SJ[("Sedona Spatial Joins<br/>ST_Within, ST_Distance")]
        GF[("Gold Delta Tables<br/>Fact + Dimensions")]
        ML[("MLflow<br/>Tracking & Registry")]
    end
    
    subgraph Training["🚀 ML Training"]
        T[("train_healthcare_models.py<br/>5 Models")]
        T --> ML
        ML --> MR[("MLflow Registry<br/>Production Ready")]
    end
    
    subgraph Viz["🎨 Visualization"]
        WA["Next.js Web App"]
        B3D["Blender 3D Simulation"]
    end
    
    Sources --> Bronze
    Bronze --> PANDAS
    PANDAS --> ST
    ST --> Mask
    S8 -.->|Land mask<br/>filtering| Mask
    Mask --> SDelta
    SDelta --> Ollama
    Ollama --> AI
    AI --> SJ
    SJ --> GF
    GF -.->|Model Tracking| ML
    GF --> WA
    GF --> B3D
    T --> GF
```

### Medallion Layers

```mermaid
flowchart TB
    subgraph BRONZE["<b>🥉 Bronze:</b> Raw Data"]
        direction LR
        B1[US Accidents CSV] ~~~ B2[USGS JSON] ~~~ B3[OSM GeoJSON] ~~~ B4[NYC 311] ~~~ B5[NYC Flights] ~~~ B6[NYC Weather] ~~~ B7[US Neighborhoods]
    end

    subgraph SILVER["<b>🥈 Silver:</b> Cleaned, Spatial, Enriched"]
        direction LR
        S1[Accidents + Geometry] ~~~ S2[Earthquakes + Geometry] ~~~ S3[Infrastructure + Geo] ~~~ S4[311 + Geo] ~~~ S5[Flights + Geo] ~~~ S6[Weather] ~~~ S7[Neighborhoods + Polygons]
    end

    subgraph GOLD["<b>🥇 Gold:</b> Star Schema + AI Insights"]
        direction LR
        G1[FACT_HAZARD_EVENTS] ~~~ G2[DIM_NEIGHBORHOODS] ~~~ G3[DIM_INFRASTRUCTURE] ~~~ G4[DIM_WEATHER] ~~~ G5[5 ML Models]
    end

    BRONZE -->|"Ingest"| SILVER
    SILVER -->|"Transform"| GOLD
```

---

## Infrastructure

### Docker Containers Architecture

The pipeline runs entirely in Docker containers orchestrated via `docker-compose.yml`:

```mermaid
graph TD
    subgraph Host["Host Machine (Docker Engine)"]
        direction TB
        Spark("Spark Master<br/>:9080")
        MinIO("MinIO<br/>:9900 :9901")
        Postgres("PostgreSQL<br/>:5434")
        Ollama("Ollama<br/>:11434")
        OTel("OTel Collector<br/>:4317 :4318")
        MLflow("MLflow<br/>:5001")
        Web("Next.js<br/>:3000")
        Grafana("Grafana<br/>:3001 :3002")
    end
```

### Service Details

| Service | Port | Image | Purpose |
|---------|------|-------|---------|
| **Spark Master** | 9080 | bitnami/spark:3.5 | Distributed compute engine |
| **MinIO** | 9900/9901 | minio/minio | S3-compatible object storage |
| **PostgreSQL** | 5434 | postgres:15 | Hive metastore + MLflow backend |
| **Ollama** | 11434 | ollama/ollama | Local LLM inference |
| **OTel Collector** | 4317/4318 | otel/opentelemetry-collector | Metrics & traces |
| **MLflow** | 5001 | mlflow/mlflow | Experiment tracking |
| **Next.js** | 3000 | node:20-alpine | Web app frontend |
| **Grafana** | 3001/3002 | grafana/grafana | Dashboards & visualization |

### Quick Start

```bash
# Start all infrastructure
docker compose up -d

# Verify services
docker compose ps

# View logs
docker compose logs -f spark
docker compose logs -f minio

# Check health
curl -s http://localhost:9900/minio/health/live
curl -s http://localhost:5001/health
curl -s http://localhost:3001/api/health
```

### Individual Service Access

```bash
# MinIO Console (admin / minio123)
http://localhost:9900

# MLflow
http://localhost:5001

# Grafana (admin / admin)
http://localhost:3001

# Next.js Web App
http://localhost:3000
```

### Environment Variables

```bash
# .env.example
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=minio123
POSTGRES_PASSWORD=postgres
MLFLOW_TRACKING_URI=http://localhost:5001
SPARK_MASTER=spark://localhost:7077
OLLAMA_BASE_URL=http://localhost:11434
```

### Stopping and Cleanup

```bash
# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# Remove all containers, volumes, and images
docker compose down --rmi all -v
```

### Troubleshooting

```bash
# Check container status
docker compose ps -a

# Restart a specific service
docker compose restart spark

# View service logs
docker compose logs --tail=100 spark

# Shell into a container
docker compose exec spark bash
docker compose exec minio sh
```

### Data Persistence

Data persists in Docker volumes:

- `postgres_data` - PostgreSQL database
- `minio_data` - MinIO storage
- `mlflow_artifacts` - MLflow model artifacts

---

## Data and Model Flow

### Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Bronze as Bronze Ingestion
    participant Sources as External APIs
    participant MinIO as MinIO Storage
    participant Spark as Spark Cluster
    participant Sedona as Apache Sedona
    participant OTel as OpenTelemetry
    participant MLflow as MLflow
    participant Ollama as Ollama LLM
    participant Gold as Gold Layer
    
    User->>Bronze: Run ingestion
    Sources->>Bronze: USGSi Earthquakes, OSM Hospitals, NYC 311, Flights, Weather
    Bronze->>MinIO: Upload raw data (Bronze Parquet)
    
    OTel->>Spark: Trace pipeline execution
    
    Bronze->>Spark: Trigger silver transform (pandas + Sedona)
    Spark->>Sedona: Apply ST_Point, ST_GeomFromGeoJSON, ST_Within (Land Mask)
    Sedona-->>Spark: Cleaned & Filtered DataFrames
    Spark->>MinIO: Write Silver Delta Tables
    
    Spark->>Gold: Trigger gold enrichment
    Gold->>Ollama: Extract severity, hazard_type, flight_risk
    Ollama-->>Gold: JSON enrichment
    Gold->>Sedona: ST_Within, ST_Distance joins
    Gold->>MLflow: Log metrics, register models
    Gold->>MinIO: Write Gold Delta Tables
    
    User->>Gold: Query results
    User->>MLflow: Inspect training runs
```

### MLflow Model Training Flow

```mermaid
flowchart TB
    subgraph Training["🚀 Model Training & Tracking"]
        D[("Gold Delta Tables")] -->|"Load"| T("Training Scripts<br/>train_healthcare_models.py<br/>train_aviation_models.py")
        T -->|"Train"| M1(Model 1: Fire Risk)
        T --> M2(Model 2: Hospital Overpopulation)
        T --> M3(Model 3: Emergency Response)
        T --> M4(Model 4: Hospital Bed Demand)
        T --> M5(Model 5: Ambulance Dispatch)
        M1 --> ML[("MLflow Tracking<br/>localhost:5001")]
        M2 --> ML
        M3 --> ML
        M4 --> ML
        M5 --> ML
        ML --> MR[("MLflow Registry<br/>5 Registered Models")]
    end
    
    subgraph Inference["🔮 Production Inference"]
        MR -->|"Deploy"| S("Inference Service")
        S -->|"Score"| NewD("New Hazard Events")
    end
```

---

## Web App Architecture

### Maps + React + Next.js Stack

```mermaid
flowchart TD
    User["🌍 End User"]
    
    subgraph Client["🎨 Next.js Client (Browser)"]
        N[("Next.js 14<br/>:3000")]
        R["React 18"]
        L["Leaflet.js<br/>Heatmap / Markers"]
        M["Mapbox / CARTO<br/> basemap tiles"]
        T["Three.js<br/>3D WebGL"]
    end
    
    subgraph Data["📊 Gold Layer Data"]
        G["Gold Delta Tables<br/>MinIO"]
        P["Parquet Files"]
    end
    
    subgraph ML["🤖 ML Predictions"]
        Ollama["Ollama<br/>llama3.2:1b"]
        MLflow["MLflow<br/>5 Models"]
    end
    
    User --> N
    N --> R
    R --> L
    L --> M
    L -- "Query risk data" --> G
    G -- "ML inference" --> Ollama
    Ollama -- "model predictions" --> R
    R --> T
    
    style N fill:#339,color:#fff
    style R fill:#61dafb,color:#000
    style L fill:#2ecc71,color:#000
    style M fill:#f39c12,color:#000
```

### Features

- **Interactive Heatmap**: Leaflet.js with CARTO tiles
- **5 ML Model Predictions**: Fire Risk, Hospital Overpopulation, Emergency Response, Bed Demand, Ambulance Dispatch
- **Real-time filtering**: Filter by hazard type
- **3D Visualization**: Three.js for WebGL rendering
- **Next.js 14 App Router**: Modern React full-stack

### Running the Web App

```bash
# Development
cd web && npm run dev

# Production build
cd web && npm run build
npm start

# Access
# http://localhost:3000
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
| **Maps** | Leaflet + Mapbox | Latest | Web mapping |
| **Frontend** | React + Next.js | 14 | Web app |
| **3D Engine** | Three.js | Latest | WebGL rendering |

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

# 5. Run ingestion (Bronze) - all 8 data sources
python src/bronze_ingestion.py

# 6. Run Silver transforms (pandas + Sedona)
spark-submit src/silver_pandas_transform.py
spark-submit src/silver_sedona_transform.py

# 7. Run Gold enrichment
spark-submit src/gold_dimensional_modeling.py

# 8. Train ML models
python src/train_healthcare_models.py
python src/train_aviation_models.py

# 9. Run full pipeline
python src/run_pipeline.py
```

---

## Pipeline Components

### 1. Bronze Layer (Raw Data Ingestion)

**File:** `src/bronze_ingestion.py`

Ingests raw data from 8 external sources into MinIO Bronze layer:

| Source | API | Format |
|--------|-----|-------|
| US Accidents | Kaggle CSV | CSV |
| USGS Earthquakes | USGS FDSN API | JSON (GeoJSON) |
| OSM Infrastructure | Overpass API | XML → JSON |
| NYC 311 | Socrata API | JSON |
| NYC Flights | OpenSky Network API | JSON |
| NYC Weather | NOAA NWS API | JSON |
| US Neighborhoods | Kaggle GeoJSON | GeoJSON |
| NYC Land Mask | NYC Open Data NTA | GeoJSON |

**Execution:**
```bash
python src/bronze_ingestion.py
```

### 2. Silver Layer (Cleaned + Spatial Transform)

Two-step transformation:

#### Step 1: Pandas Cleaning
**File:** `src/silver_pandas_transform.py`

Cleans, deduplicates, handles missing values.

```bash
spark-submit src/silver_pandas_transform.py
```

#### Step 2: Sedona Spatial
**File:** `src/silver_sedona_transform.py`

Applies spatial operations:
- `ST_Point()` - Creates geometries from lat/lon
- `ST_GeomFromGeoJSON()` - Parses GeoJSON
- `ST_Within()` - Land mask filtering (NYCNTA boundary)
- `ST_Distance()` - Proximity calculations
- `ST_Buffer()` - Service area buffers

```bash
spark-submit src/silver_sedona_transform.py
```

### 3. Gold Layer (Star Schema + AI Enrichment)

**File:** `src/gold_dimensional_modeling.py`

Creates dimensional star schema:

| Table | Type | Description |
|-------|------|------------|
| `fact_hazard_events` | Fact | Combined hazard events with AI-enriched labels |
| `dim_neighborhoods` | Dimension | NYC neighborhoods with risk scores |
| `dim_infrastructure` | Dimension | Hospitals, fire stations, EMS |
| `dim_weather` | Dimension | Weather conditions |

AI Enrichment via Ollama:
- `severity`: Critical, High, Medium, Low
- `hazard_type`: Fire, Medical, Weather, Infrastructure
- `flight_risk`: High, Medium, Low

```bash
spark-submit src/gold_dimensional_modeling.py
```

### 4. ML Training

#### Healthcare Models
**File:** `src/train_healthcare_models.py`

Trains 5 supervised models:

| Model | Task | Algorithm | Target |
|-------|------|----------|--------|
| NYC Fire Risk Model | Classification | Fire probability |
| Hospital Overpopulation | Classification | Overcapacity |
| Emergency Response | Regression | Response time |
| Hospital Bed Demand | Regression | Bed shortage |
| Ambulance Dispatch | Classification | Dispatch optimization |

```bash
python src/train_healthcare_models.py
```

#### Aviation Models
**File:** `src/train_aviation_models.py`

Flight delay prediction models.

```bash
python src/train_aviation_models.py
```

### 5. Full Pipeline Orchestration

**File:** `src/run_pipeline.py`

Runs complete Bronze → Silver → Gold pipeline:

```bash
python src/run_pipeline.py
```

---

## Testing Pyramid

```mermaid
block-beta
columns 30

space:9 L3["🔴 L3: End-to-End<br/>Full Pipeline"]:12 space:9
space:6 L2["🟠 L2: Integration Services<br/>Real Services / Local Docker<br/>(MinIO, MLflow, Prometheus)"]:18 space:6
space:3 L1["🟡 L1: Unit with Mocks<br/>Mocked Dependencies<br/>(USGS, OSM, NYC311 APIs)"]:24 space:3
L0["🟢 L0: Unit Isolation<br/>Pure Functions / No I/O"]:30

style L3 fill:#ff5722,color:#fff,stroke:#333,stroke-width:2px
style L2 fill:#ff9800,color:#fff,stroke:#333,stroke-width:2px
style L1 fill:#ffc107,color:#000,stroke:#333,stroke-width:2px
style L0 fill:#ffeb3b,color:#000,stroke:#333,stroke-width:2px
```

### Test Results

```bash
# Run all tests
pytest

# Run L0 unit tests only
pytest tests/test_l0_*.py

# Run L1 integration tests
pytest tests/test_l1_*.py

# Run L2 component tests
pytest tests/test_l2_*.py

# Run L3 E2E tests
pytest tests/test_l3_*.py

# Result: 102 passed, 14 warnings
# No skipped tests - per Constitution rule
```

---

## Blender 3D Integration

**File:** `scripts/render_flight_simulation.py`

Automates 3D data visualization from Gold layer Parquet files. Uses Python API within Blender to procedurally generate NYC flight simulation environments.

```bash
# Generate 3D simulation
blender -b examples/nyc_ml_heatmap.blend -P scripts/render_flight_simulation.py
```

**Features:**
- Procedural building generation from NYC GeoJSON
- Flight path visualization
- Heatmap coloring by ML risk scores
- Real-time camera animation

---

## Heatmap Visualization

**File:** `scripts/quick_heatmap.py`

Reads gold layer parquet files and generates a Leaflet heatmap.

```bash
python scripts/quick_heatmap.py
```

Output: `public/data/heatmap.geojson`

---

## Deployment

### Docker Production Build

```bash
# Build all images
docker compose build

# Scale Spark workers
docker compose up -d --scale spark-worker=3

# Enable MLflow registry
docker compose up -d mlflow-server
```

### K8s (Future)

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Spark not connecting to MinIO | Check MinIO endpoint in config |
| Ollama API timeout | Increase `OLLAMA_TIMEOUT` |
| Sedona functions not found | Ensure Sedona plugin registered |
| Missing telemetry data | Ensure OpenTelemetry collector is running |
| MLflow not tracking | Check `MLFLOW_TRACKING_URI` |
| OSM API 406 error | Add `User-Agent: curl/8.7.1` header |
| Java 25 incompatible | Use `openjdk@17` for Spark |

### Service Health Checks

```bash
# MinIO
curl -s http://localhost:9900/minio/health/live

# MLflow
curl -s http://localhost:5001/health

# Grafana
curl -s http://localhost:3001/api/health

# PostgreSQL
docker compose exec postgres pg_isready -U postgres

# Spark
curl -s http://localhost:9080
```

---

## API References

### External APIs

| API | Endpoint | Documentation |
|-----|---------|-------------|
| USGS | https://earthquake.usgs.gov/fdsnws/event/1/query | USGS Docs |
| OSM Overpass | https://overpass-api.de/api/interpreter | OSM Docs |
| NYC 311 | https://data.cityofnewyork.us/resource/fhrw-4uyv.json | NYC Open Data |
| NYC Flights | https://opensky-network.org/api/states/all | OpenSky API |
| NOAA Weather | https://api.weather.gov | NWS API |

### Internal APIs

| Service | Endpoint |
|--------|----------|
| Spark Master | spark://localhost:7077 |
| Spark UI | http://localhost:9090 |
| MinIO Console | http://localhost:9900 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3001 |
| Next.js | http://localhost:3000 |

---

## Credits

- [Apache Sedona](https://sedona.apache.org/) - Geospatial SQL on Spark
- [Delta Lake](https://delta.io/) - ACID transactions on data lakes
- [Ollama](https://ollama.ai/) - Local LLM inference
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework
- [MLflow](https://mlflow.org/) - MLOps platform
- [Blender](https://www.blender.org/) - 3D Content Creation
- [USGS](https://earthquake.usgs.gov/) - Earthquake data
- [OpenStreetMap](https://www.openstreetmap.org/) - Infrastructure data
- [NYC Open Data](https://data.cityofnewyork.us/) - NYC municipal data
- [OpenSky Network](https://opensky-network.org/) - Aviation data
- [NOAA NWS](https://www.weather.gov/) - Weather data

---

<p align="center">
  Built with ❤️ for GeoAI Portfolio Projects
</p>