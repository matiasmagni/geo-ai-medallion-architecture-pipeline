# GeoAI Medallion Pipeline Constitution

## Source of Truth

**PRIMARY REFERENCE:** SPEC Kit (https://github.com/github/spec-kit)
**SECONDARY REFERENCE:** SPEC_KIT_CONSTITUTION.md (local)

All development follows Spec-Driven Development methodology from SPEC Kit.
Deviate from SPEC Kit ONLY when local SPEC_KIT_CONSTITUTION.md specifies otherwise.

## Core Mandates

1. **NO NEW FILES:** Do not create any new files in this repository under any circumstances.
2. **TEST PLACEMENT:** All new tests must be added EXCLUSIVELY to the following four files:
   - `tests/test_l0_unit.py`
   - `tests/test_l1_integration.py`
   - `tests/test_l2_component.py`
   - `tests/test_l3_e2e.py`
3. **NO EXCEPTIONS:** Any test logic or component verification must be integrated into these existing files, maintaining the established layering (L0-L3).
4. **ALL debugging screenshots MUST be saved to:** `debugging/screenshots/` directory
   - **NEVER commit screenshots to git** - they are already in `.gitignore`
   - When taking screenshots during debugging, always use the `filename` parameter to save to `debugging/screenshots/`

## SPEC Kit Principles (Enforced)

- **DRY:** No code duplication - use shared modules in `src/common/`
- **KISS:** Keep it simple - complexity is the enemy of reliability
- **CLEAN Architecture:** Domain → Application → Infrastructure → Presentation
- **GITFLOW:** Use feature/fix/release/hotfix branches from `develop`
- **TTD:** Test-Driven Development - L0 to L3 tests, NO SKIPS

## Configuration (Enforced by SPEC Kit)

- **ALL URLs from `.env`** - NEVER hardcode URLs in code
- Use `os.getenv()` with fallbacks for local development
- External APIs: USGS_API_URL, OSM_API_URL, NYC311_API_URL, OPENSKY_API_URL, NWS_API_URL
- Internal services: MINIO_URL, MLFLOW_URL, PROMETHEUS_URL, GRAFANA_URL, OLLAMA_URL
