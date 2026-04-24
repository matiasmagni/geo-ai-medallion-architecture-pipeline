<!-- SPECKIT START -->

# GeoAI Medallion Architecture Pipeline - Agent Guidelines

## CRITICAL: Read Constitution First
Before ANY implementation work, read `.specify/constitution.md` to understand:
- Core principles (architecture immutable, medallion pattern, shift-left AI)
- Technology stack constraints (Delta Lake 3.1.0, Shapely NOT Sedona)
- Test requirements (L0, L1, L2, L3 must pass)

## Tech Stack (Immutable Reference)
| Component | Technology | Version |
|-----------|------------|---------| 
| **Compute Engine** | Apache Spark (PySpark) | 3.5.0 |
| **Spatial Processing** | Shapely + PySpark | - |
| **Storage** | Delta Lake | 3.1.0 |
| **Object Storage** | MinIO | Latest |
| **Database** | PostgreSQL | 15 |
| **MLOps** | MLflow | 2.x |
| **LLM** | Ollama | Latest |
| **Monitoring** | Prometheus + Grafana | Latest |

## Important Notes

### DO NOT REMOVE COMPONENTS
- Delta Lake 3.1.0 is REQUIRED for ACID transactions
- Shapely is used (NOT Sedona) due to Sedona JVM classpath issues
- MinIO is required for S3-compatible storage

### Architecture Layers
- Bronze: Raw data ingestion (5 sources)
- Silver: Spatial + ALL AI enrichment (Ollama calls HERE only)
- Gold: Star Schema (NO LLM calls) + spatial joins

### Test Levels
- tests/test_l0_unit.py - Unit tests
- tests/test_l1_integration.py - Integration tests (run in Docker)
- tests/test_l2_component.py - Component tests
- tests/test_l3_e2e.py - E2E tests

### Running Tests
```bash
# L0: Unit tests
pytest tests/test_l0_unit.py -v

# L1: Integration tests (in Docker)
docker exec geoai-spark python -m pytest /home/jovyan/tests/test_l1_integration.py -v

# L2: Component tests  
pytest tests/test_l2_component.py -v

# L3: E2E tests
pytest tests/test_l3_e2e.py -v
```
<!-- SPECKIT END -->