# GeoAI Medallion Pipeline Constitution

## Core Mandates
1. **NO NEW FILES:** Do not create any new files in this repository under any circumstances.
2. **TEST PLACEMENT:** All new tests must be added EXCLUSIVELY to the following four files:
   - `tests/test_l0_unit.py`
   - `tests/test_l1_integration.py`
   - `tests/test_l2_component.py`
   - `tests/test_l3_e2e.py`
3. **NO EXCEPTIONS:** Any test logic or component verification must be integrated into these existing files, maintaining the established layering (L0-L3).
