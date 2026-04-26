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
- **Bronze**: Raw data ingestion from sources.
- **Silver**: Spatial transformation + ALL AI enrichment via Ollama.
- **Gold**: Kimball Star Schema (no LLM calls) with spatial joins.

### III. Shift-Left AI Pattern (NON-NEGOTIABLE)
ALL LLM inference happens in Silver layer. Gold layer is purely declarative SQL transformations.

### IV. Development Philosophy (MANDATORY)
- **TDD (Test-Driven Development)**: Write failing tests before implementation.
- **SOLID**: Follow SOLID principles for all new code and refactors.
- **DRY (Don't Repeat Yourself)**: Eliminate redundancy; consolidate logic.
- **KISS (Keep It Simple, Stupid)**: Favor simple, maintainable solutions.
- **Clean Architecture**: Keep business logic decoupled from frameworks/I/O.

### V. Testing Pyramid (REQUIRED)
Four test levels MUST pass before deployment:
- L0: Unit tests (pure functions, no I/O, no dependencies)
- L1: Integration tests (Spark data flow)
- L2: Component tests (Docker services)
- L3: E2E tests (full pipeline execution)

### VI. Testing Requirements (MANDATORY)
1. **100% Code Coverage**: All source modules MUST have 100% test coverage.
2. **100% Test Pass Rate**: All tests in the suite MUST pass 100% before any deployment.
3. **Zero Test Skips**: Test skipping is prohibited. All features must be tested, mocked only where absolutely necessary for isolation, and E2E tests MUST run against real infrastructure.
4. **No Service Mocking in L2**: L2 component tests MUST connect to real services. If MinIO, MLflow, Prometheus, or any Docker service is unavailable, the test MUST FAIL with a clear error - NOT skip or mock. Fix the infrastructure, don't mock the test.
5. **Hard Failures**: Service connection failures must cause test failures. No try/except wrappers that swallow errors. No MagicMock fallback fixtures.
6. **No Rollbacks**: Forward progress is mandatory. Regression failures must be fixed immediately without reverting committed infrastructure/code.

### VII. GitHub Spec Kit Integration (MANDATORY)
- All new features and architectural changes MUST be defined via the GitHub Spec Kit (.specify/ directory).
- Each spec must include:
    - User requirements
    - Architecture diagrams (Mermaid)
    - Acceptance criteria (TDD/E2E test definitions)

### VIII. GitFlow Branching Strategy (REQUIRED)
- **main**: Production-ready code (Protected).
- **develop**: Integration branch (Protected).
- **feature/*** / **bugfix/*** / **hotfix/***: Feature/fix branches.
- **No direct commits to main or develop.**

**Version**: 1.1.0 | **Ratified**: 2026-04-25 | **Last Amended**: 2026-04-25
