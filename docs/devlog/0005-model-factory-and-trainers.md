# 0005-model-factory-and-trainers

**Date:** 2025-11-29
**Author:** AI Agent

## What was implemented
- Verified `model_factory.py` implements MobileNetV3 and U-Net.
- Updated `src/tasks/classification/trainer.py` to support manifest loading and preprocessing.
- Updated `src/tasks/segmentation/trainer.py` to support manifest loading and preprocessing (image + mask).
- Updated `scripts/generate_fixtures.py` to generate real images and masks for smoke testing.
- Implemented `tests/integration/test_training_smoke.py` covering both classification and segmentation.

## Files created/modified
- `src/tasks/classification/trainer.py`
- `src/tasks/segmentation/trainer.py`
- `src/core/data/dataset_loader.py` (minor fix for Path typing)
- `scripts/generate_fixtures.py`
- `tests/integration/test_training_smoke.py`

## How to run it
```bash
python scripts/generate_fixtures.py
pytest tests/integration/test_training_smoke.py
```

## Tests added/updated
- `test_classification_training_smoke`
- `test_segmentation_training_smoke`

## Known limitations and next steps
- `cropper` trainer is not yet fully integrated with manifest loading (requires bbox targets in parquet), but classification and segmentation are.
- Next step: Export logic (Phase 5) - already partially covered in trainers, but needs dedicated verification.
