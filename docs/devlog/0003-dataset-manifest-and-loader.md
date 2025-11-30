# 0003-dataset-manifest-and-loader

**Date:** 2025-11-28
**Author:** AI Agent

## What was implemented
- Implemented `src/core/data/dataset_loader.py` to load datasets from a manifest directory.
- Validates manifest against `dataset_manifest_schema.json`.
- Reads Parquet files and yields a `tf.data.Dataset`.
- Created local fixtures for smoke testing.

## Files created/modified
- `src/core/data/dataset_loader.py`
- `tests/integration/test_dataset_build_smoke.py`
- `tests/fixtures/manifests/local_smoke/` (manifest + parquet)
- `scripts/generate_fixtures.py`

## How to run it
```bash
pytest tests/integration/test_dataset_build_smoke.py
```

## Tests added/updated
- `tests/integration/test_dataset_build_smoke.py`: Verifies loading of a local fixture dataset.

## Known limitations and next steps
- Loader currently uses a simple generator for Parquet reading. For massive datasets, we might need a more optimized reader or `tfio`.
- Next step: Implement preprocessing pipelines (Phase 3).
