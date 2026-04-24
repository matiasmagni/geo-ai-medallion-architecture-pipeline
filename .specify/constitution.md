# GeoAI Medallion Architecture - Constitution

## Core Principles

### I. Architecture Immutable (NON-NEGOTIABLE)
The README.md architecture is the **single source of truth**. No component listed in the tech stack shall be removed, replaced, or deprecated without explicit user consent and documentation update. The architecture includes:
- Apache Spark 3.5.0 for distributed processing
- Delta Lake 3.1.0 for ACID transactions on data lake
- Shapely (NOT Sedona) for spatial processing
- MinIO for S3-compatible storage
- PostgreSQL 15 for Hive Metastore + MLflow
- MLflow 2.x for experiment tracking & model registry
- Ollama for local LLM inference
- Prometheus + Grafana for monitoring

### II. Medallion Architecture Pattern (NON-NEGOTIABLE)
Three-layer architecture MUST be preserved:
- **Bronze**: Raw data ingestion from 5 sources (US Accidents, US Neighborhoods, USGS Earthquakes, OSM Hospitals/Fire, NYC 311)
- **Silver**: Spatial transformation + ALL AI enrichment via Ollama
- **Gold**: Kimball Star Schema (no LLM calls) with spatial joins

### III. Shift-Left AI Pattern (NON-NEGOTIABLE)
ALL LLM inference happens in Silver layer. Gold layer is purely declarative SQL transformations.

### IV. Testing Pyramid (REQUIRED)
Four test levels MUST pass before deployment:
- L0: Unit tests (imports, config, helpers)
- L1: Integration tests (Spark data flow)
- L2: Component tests (Docker services)
- L3: E2E tests (full pipeline execution)

### V. Production-Ready Standards
- All tests pass (L0+L1+L2+L3)
- No mocking unless explicitly approved by user
- Real data pipelines (no fake data generators for production)
- Delta Lake with Parquet fallback for storage
- Full observability (Prometheus + Grafana dashboards)

## Technology Stack Constraints

| Component | Requirement | Rationale |
|-----------|------------|----------|
| Spark | 3.5.0 | Distributed processing required |
| Delta Lake | 3.1.0 | ACID transactions essential |
| Shapely | Latest | Replace for Sedona (JVM issues) |
| MinIO | Latest | S3-compatible storage |
| Ollama | Latest | Local LLM for AI enrichment |
| MLflow | 2.x | Model registry |
| Prometheus | Latest | Metrics collection |
| Grafana | Latest | Dashboards |

## Development Workflow

### Before Any Implementation
1. Reference README.md for architecture
2. Verify component is in tech stack table
3. Check test pyramid levels

### Before Commit
1. Run L0 tests: `pytest tests/test_l0_unit.py`
2. Run L1 tests (in Docker): `docker exec geoai-spark pytest tests/test_l1_integration.py`
3. Run L2 tests: `pytest tests/test_l2_component.py`
4. Run L3 tests: `pytest tests/test_l3_e2e.py`

### Quality Gates
- Zero test failures across all levels
- No removal of core components (Delta Lake, Shapely, etc.)
- Architecture documentation matches implementation

## Governance

### Breaking Changes
Requires explicit user approval and:
1. Documentation update in README.md
2. Migration plan for dependent systems
3. Test coverage verification

### Component Addition
New components require:
1. README.md tech stack table update
2. Implementation in appropriate layer
3. Tests in appropriate test level

**Version**: 1.0.0 | **Ratified**: 2026-04-24 | **Last Amended**: 2026-04-24