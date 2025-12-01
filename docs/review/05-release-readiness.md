# Release Readiness Report

## Verification Status

### Verified Components
- **Training**: Verified via `src/tasks/classification/entrypoint.py` (1 epoch smoke test).
- **Dataset Building**: Verified via `scripts/build_otoscopic_dataset.py`.
- **Export**: Verified via `tests/integration/test_export_smoke.py` (TFLite export).
- **Evaluation/Explainability**: Verified via `src/core/explainability/cli.py`.
- **Dependency Injection**: Verified via `tests/unit/test_di.py` and `tests/unit/test_di_container.py`.

### Unverified / Partially Verified Components
- **CoreML Export**: Code implemented and `coremltools` added, but runtime verification skipped due to Keras 3 / coremltools incompatibility (`Sequential` object has no attribute `_get_save_spec`).
- **Cloud Ensembles**: Code was removed as it was unused.
- **Vertex AI Tuning**: Verified via unit tests (`tests/unit/test_tuning.py`) but not end-to-end on actual Google Cloud infrastructure (requires credentials/billing).
- **Segmentation/Cropper Tasks**: Entrypoints updated to use DI, but full end-to-end training not run in this session (relied on classification smoke test as representative).

## Known Limitations
1.  **CoreML Dependencies**: `coremltools` must be installed manually for CoreML export to work.
2.  **Vertex AI**: Requires GCP credentials and project setup.
3.  **Dataset**: The `build_otoscopic_dataset.py` script assumes a specific directory structure for source data.

## Roadmap
- [x] Add `coremltools` to `pyproject.toml` optional dependencies.
- [x] Implement end-to-end integration tests for Segmentation and Cropper tasks.
- [ ] Add CI/CD pipeline configuration (e.g., GitHub Actions).
