# SPEC Kit Constitution
## GeoAI Medallion Architecture Pipeline

**Version:** 1.0.0
**Date:** 2026-05-14
**Status:** ACTIVE - IMMUTABLE CORE

---

## Preamble

This constitution establishes the foundational principles governing all software development for the GeoAI Medallion Architecture Pipeline. All decisions, implementations, and processes MUST align with these principles. Violations constitute technical debt and MUST be addressed immediately.

**Source of Truth:** GitHub SPEC Kit (https://github.com/github/spec-kit)
**Development Methodology:** Spec-Driven Development

---

## Article I: Core Principles

### Section 1.1 - DRY (Don't Repeat Yourself)

1. **Every piece of knowledge MUST have a single, unambiguous, authoritative representation.**
2. Code duplication is a cardinal sin. Extract to shared modules, utilities, or base classes.
3. Configuration MUST be centralized in `.env` - NO hardcoded values anywhere.
4. API endpoints, URLs, credentials MUST use `os.getenv()` with `.env` fallbacks.
5. Common transformations, validators, and helpers go into `src/common/` or `src/utils/`.

### Section 1.2 - KISS (Keep It Simple Stupid)

1. **Complexity is the enemy of reliability.** Prefer obvious over clever.
2. Write code that a junior developer can understand in 30 seconds.
3. If you need a comment to explain it, refactor until it's self-explanatory.
4. "It works" is NOT enough - it must be readable, maintainable, and testable.
5. Solve the problem, then optimize - don't pre-optimize.

### Section 1.3 - CLEAN Architecture

1. **Separation of Concerns:** Each layer has defined responsibilities.
2. **Dependency Rule:** Dependencies point inward only. Inner layers know nothing about outer layers.
3. **Layers:**
   - `src/domain/` - Business entities, value objects, domain logic
   - `src/application/` - Use cases, application services, DTOs
   - `src/infrastructure/` - External services, persistence, adapters
   - `src/presentation/` - API, CLI, UI handlers

### Section 1.4 - GITFLOW

1. **Branch Strategy:**
   - `main` - Production-ready, never force-pushed
   - `develop` - Integration branch, merged into main for release
   - `feature/*` - New features, branched from `develop`
   - `fix/*` - Bug fixes, branched from `develop`
   - `release/*` - Release preparation, branched from `develop`
   - `hotfix/*` - Production fixes, branched from `main`

2. **Commit Messages:** Conventional Commits (feat:, fix:, refactor:, docs:, test:, chore:)
3. **Pull Requests:** Required before merging to `develop` or `main`
4. **Never commit secrets, credentials, or .env files to version control.**

---

## Article II: TTD (Test Through Development)

### Section 2.1 - Test Levels (L0 to L3)

All tests are MANDATORY. No skips allowed. Test-driven development is the ONLY acceptable workflow.

#### L0: Unit Tests (`tests/test_l0_unit.py`)
- **Purpose:** Verify individual functions, classes, and modules in isolation
- **Coverage Target:** 100% of pure business logic
- **Mocking:** Mock ALL external dependencies (API calls, DB, file system)
- **Execution:** < 5 seconds per test class
- **Naming:** `Test<ClassName>::test_<method>_<expected_behavior>`

#### L1: Integration Tests (`tests/test_l1_integration.py`)
- **Purpose:** Verify interactions between components
- **Scope:** Database, file system, service connections
- **Real Services:** Use actual MinIO, Postgres, MLflow when available
- **Cleanup:** Test must clean up after itself (teardown)
- **Execution:** < 30 seconds per test class

#### L2: Component Tests (`tests/test_l2_component.py`)
- **Purpose:** Verify complete services or components
- **Scope:** Docker containers, microservices, API contracts
- **Real Environment:** Must work against live infrastructure
- **Execution:** < 60 seconds per test class

#### L3: E2E Tests (`tests/test_l3_e2e.py`)
- **Purpose:** Verify complete user workflows
- **Scope:** Full pipeline from ingestion to gold layer
- **Environment:** Full production-like setup
- **Execution:** < 5 minutes total
- **Skips:** NEVER - E2E must always run

### Section 2.2 - Test Execution Requirements

```
# Run ALL tests (L0-L3) - NO SKIPS
pytest tests/ -v --tb=short

# Run specific level
pytest tests/test_l0_unit.py -v
pytest tests/test_l1_integration.py -v
pytest tests/test_l2_component.py -v
pytest tests/test_l3_e2e.py -v
```

### Section 2.3 - Test Quality Standards

1. **Assertion First:** Write assertion before implementation (TDD)
2. **Single Responsibility:** Each test verifies ONE behavior
3. **Descriptive Names:** Test name describes what it verifies
4. **Independent:** Tests can run in any order, no dependencies
5. **Fast Feedback:** Unit tests < 5s, Integration < 30s
6. **Deterministic:** Same input = same output, no flakiness

---

## Article III: SSD (System Design Document)

### Section 3.1 - Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GEOAI MEDALLION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │ BRONZE  │───▶│ SILVER  │───▶│  GOLD   │───▶│INFERENCE│          │
│  │  Layer  │    │  Layer  │    │  Layer  │    │  Layer  │          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│       │              │              │              │                │
│       ▼              ▼              ▼              ▼                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  MinIO  │    │ Spark   │    │  MLflow │    │ Ollama  │          │
│  │  (S3)   │    │  + HDFS │    │   (ML)  │    │ (LLM)   │          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│       │              │              │              │                │
│       └──────────────┴──────────────┴──────────────┘                │
│                              │                                      │
│                    ┌─────────▼─────────┐                          │
│                    │   PostgreSQL     │                          │
│                    │ (Metastore + UI)  │                          │
│                    └───────────────────┘                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MONITORING STACK                                │   │
│  │   Prometheus → Grafana → AlertManager                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Section 3.2 - Data Flow

1. **Bronze Layer (Raw Ingestion)**
   - Real data from external APIs (USGS, OSM, NYC311, OpenSky, NWS)
   - Stored as Parquet in MinIO `geoai-bronze` bucket
   - Immutable - never modified after creation

2. **Silver Layer (Enriched & Cleaned)**
   - Deduplicated, validated, enriched with spatial data
   - Stored in MinIO `geoai-silver` bucket
   - Ready for analysis

3. **Gold Layer (Aggregated & Model-Ready)**
   - Dimension tables, fact tables, aggregations
   - Stored in MinIO `geoai-gold` bucket
   - Optimized for ML workloads

4. **Inference Layer (AI-Powered)**
   - LLM enrichment via Ollama
   - ML model training via MLflow
   - Predictions served to consumers

### Section 3.3 - Component Responsibilities

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| `bronze_ingestion.py` | Raw API ingestion | MinIO, External APIs |
| `silver_*.py` | Data cleaning, enrichment | Spark, MinIO, Ollama |
| `gold_*.py` | Aggregation, modeling | Spark, MinIO, MLflow |
| `train_*.py` | Model training | MLflow, Spark |
| `inference.py` | Predictions | Ollama, MinIO |

---

## Article IV: Configuration Management

### Section 4.1 - Environment Variables

**ALL configuration MUST use `.env` as single source of truth.**

```env
# REQUIRED - No defaults allowed for external APIs
USGS_API_URL=https://earthquake.usgs.gov/fdsnws/event/1/query
OSM_API_URL=https://overpass-api.de/api/interpreter
NYC311_API_URL=https://data.cityofnewyork.us/resource/fhrw-4uyv.json
OPENSKY_API_URL=https://opensky-network.org/api/states
NWS_API_URL=https://api.weather.gov

# REQUIRED - Internal services
MINIO_URL=http://localhost:9000
MLFLOW_URL=http://localhost:5000
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3001
OLLAMA_URL=http://localhost:11434
SPARK_UI_URL=http://localhost:9080

# REQUIRED - Credentials (NO DEFAULTS)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=<secure-password>
POSTGRES_PASSWORD=<secure-password>

# REQUIRED - ML/AI
MLFLOW_TRACKING_URI=http://localhost:5000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Section 4.2 - Configuration Access Pattern

```python
# CORRECT - Pull from environment with fallback
class Config:
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    USGS_API_URL: str = os.getenv("USGS_API_URL")  # NO FALLBACK for external APIs

# INCORRECT - Hardcoded URL (FORBIDDEN)
USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
```

---

## Article V: Code Standards

### Section 5.1 - Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Files | snake_case | `bronze_ingestion.py` |
| Classes | PascalCase | `class DataIngestion:` |
| Functions | snake_case | `def upload_to_bronze()` |
| Constants | UPPER_SNAKE | `MAX_RETRIES = 3` |
| Tests | `test_<module>_<function>` | `test_bronze_ingestion_uploads_data` |

### Section 5.2 - Code Structure

```
src/
├── domain/           # Business logic, entities
│   └── models.py
├── application/      # Use cases, services
│   └── ingestion_service.py
├── infrastructure/   # External adapters
│   ├── minio_client.py
│   └── api_clients.py
├── presentation/     # Entry points
│   └── cli.py
├── common/           # SHARED utilities (DRY)
│   ├── config.py
│   ├── validators.py
│   └── utils.py
└── tests/            # MUST mirror src structure
    ├── l0_unit/
    ├── l1_integration/
    ├── l2_component/
    └── l3_e2e/
```

### Section 5.3 - Imports

```python
# Order: stdlib → third-party → local
import os
import logging
from typing import List, Optional

import pandas as pd
import boto3
from botocore.client import Config as BotoConfig

from src.common.config import Config
from src.domain.models import GeoRecord
```

---

## Article VI: Enforcement

### Section 6.1 - Pre-Commit Checklist

Before ANY commit, verify:

- [ ] All tests pass: `pytest tests/ -v --tb=short`
- [ ] No hardcoded URLs (grep for `https?://` in code)
- [ ] Configuration uses `os.getenv()`
- [ ] No secrets in code (check .env is in .gitignore)
- [ ] Code follows DRY, KISS, CLEAN principles
- [ ] Commit message follows Conventional Commits

### Section 6.2 - CI/CD Requirements

1. **Lint:** Code passes all linting checks
2. **Type Check:** No type errors
3. **Tests:** All L0-L3 tests pass (NO SKIPS)
4. **Coverage:** > 80% for domain/application layers

### Section 6.3 - Violations

| Violation | Severity | Action |
|-----------|----------|--------|
| Hardcoded URL | CRITICAL | Fix immediately, commit PR |
| Test skip | CRITICAL | Remove skip or justify |
| Code duplication | HIGH | Refactor to shared module |
| Missing test | HIGH | Add test immediately |
| .env committed | CRITICAL | Remove from git, rotate secrets |

---

## Article VII: Immutable Core

### Section 7.1 - Non-Negotiable Rules

1. **Never skip tests** - Tests exist to catch bugs, skipping defeats purpose
2. **Never hardcode URLs** - Use `.env` always
3. **Never commit secrets** - .env, credentials.json, keys
4. **Never duplicate code** - DRY or justified exception
5. **Never bypass linting** - Fix issues, don't disable checks

### Section 7.2 - Amendment Process

This constitution can only be amended through:
1. Issue proposing change with rationale
2. Code review by 2+ senior engineers
3. Passes all existing tests post-change
4. Merged via PR to `main`

---

## Appendix A: Test Commands

```bash
# Full test suite
pytest tests/ -v --tb=short

# L0 Unit Tests
pytest tests/test_l0_unit.py -v

# L1 Integration Tests
pytest tests/test_l1_integration.py -v

# L2 Component Tests
pytest tests/test_l2_component.py -v

# L3 E2E Tests
pytest tests/test_l3_e2e.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Appendix B: Git Commands

```bash
# Feature branch
git checkout -b feature/new-feature develop

# Fix branch
git checkout -b fix/bug-description develop

# Commit with conventional message
git commit -m "feat: add new ingestion source

- Added USGS earthquake API integration
- Updated Config to use env variables
- Added unit tests for api_client"

# Merge to develop
git checkout develop
git merge --no-ff feature/new-feature
```

---

**This constitution is binding on all code, commits, and contributors.**
**Violations will be corrected immediately upon discovery.**