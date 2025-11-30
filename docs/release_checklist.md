# Release Checklist: v1.0.0-rc1

**Date:** 2025-11-30
**Status:** ✅ PASSED

## 1. Inventory & Sanity Audit
- [x] **Repo Tree**: Clean, no orphan files.
- [x] **Entrypoints**:
    - `src.tasks.classification.entrypoint`
    - `src.tasks.segmentation.entrypoint`
    - `src.tasks.cropper.entrypoint`
- [x] **Configs**: Hydra configs in `configs/` cover all tasks, models, and runtimes.
- [x] **Export**: SavedModel, TFLite (Float/Quantized), Core ML.

## 2. Invariants Enforced
- [x] **No hardcoded paths**: Verified via grep search.
- [x] **Swappable Preprocessing**: Verified via `test_preprocess_registry.py`.
- [x] **Manifest Schemas**: Verified via `test_dataset_manifest_schema.py`.
- [x] **Offline Tests**: All tests passed with no network access.

## 3. Test Results
**Command:** `pytest -q`
**Result:** 76 passed, 1 skipped, 0 failed.
**Duration:** ~68s

### Key Validations
- ✅ **Contracts**: ROI, Manifests, Model Factory.
- ✅ **Integration**: Dataset Build → Train Smoke → Export.
- ✅ **Runtimes**: Image (Batch, TTA), Video (Sampling).
- ✅ **Logging**: Multi-layer logging, Experiment reporting.

## 4. Documentation Reconciliation
- [x] **README.md**: Updated with accurate links and "Quick Start".
- [x] **CONTRIBUTING.md**: Created with detailed workflow.
- [x] **docs/README.md**: Created index.
- [x] **Extension Guides**: Added to `preprocessing.md` and `datasets.md`.
- [x] **Vertex Readiness**: Verified `vertex_submit.sh` and offline safety.

## 5. Known Limitations
1.  **Grad-CAM**: Skipped in tests due to Keras 3 Functional API requirement.
2.  **Core ML**: Requires `coremltools` installed (optional dependency).
3.  **Vertex Experiments**: Requires Google Cloud auth (gracefully degrades locally).

## 6. Release Decision
**Outcome:** GO for Release Candidate 1.
**Next Step:** Tag `v1.0.0-rc1` and deploy to staging.
