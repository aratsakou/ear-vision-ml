# Audit Log

## 2025-12-01 - Release Candidate Hardening Started

### Initial Setup
- Created `docs/review/` directory.
- Initialized `docs/review/00-audit-log.md`.
- Created comprehensive task list in `task.md`.

### Next Steps
- Release!

### Progress
- **Step 6: Documentation Overhaul**
    - Created `docs/review/05-release-readiness.md`.
    - Updated `README.md`, `docs/experiments.md`.
    - Created `docs/models.md`, `docs/export.md`, `docs/testing.md`.
    - Renamed `repo_rules.md` to `CONTRIBUTING.md`.
- **Step 7: Final Pass**
    - Added `coremltools` to `pyproject.toml`.
    - Implemented `test_segmentation_smoke.py` and `test_cropper_smoke.py`.
    - Committed changes.
- **Step 5: Tests**
    - Ran full test suite.
    - Fixed `cli.py` and `model_factory.py` issues.
    - Deleted obsolete `test_ensembles.py`.
- **Step 4: Functional Verification**
    - Verified Training (fixed `ModelBuilder` registration).
    - Verified Dataset Building.
    - Verified Export.
    - Verified Evaluation/Explainability.
- **Step 3: Hydra Config Cleanup**
    - Executed simplification (merged tuning, deleted ensembles/experiments).
    - Updated code (`vertex_vizier.py`, entrypoints).
- **Step 2: Architecture Comprehension**
    - Created `docs/review/02-architecture.md`.
    - Created `docs/review/03-functionality-catalogue.md`.
- **Step 1: Full Repo Scan**
    - Enumerated all files.
    - Created `docs/review/01-repo-map.md` mapping structure and entry points.
