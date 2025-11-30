# 0004-preprocess-pipelines

**Date:** 2025-11-28
**Author:** AI Agent

## What was implemented
- Implemented `full_frame_v1` and `cropper_fallback_v1` pipelines.
- Implemented `debug_viz.py` for overlay generation.
- Verified pipeline swapping and ROI contract adherence.

## Files created/modified
- `src/core/preprocess/pipelines/full_frame_v1.py`
- `src/core/preprocess/pipelines/cropper_fallback_v1.py`
- `src/core/preprocess/debug_viz.py`
- `tests/unit/test_preprocess_registry.py`

## How to run it
```bash
pytest tests/unit/test_preprocess_registry.py
```

## Tests added/updated
- `test_full_frame_pipeline_outputs_bbox`
- `test_cropper_fallback_pipeline`

## Known limitations and next steps
- `cropper_fallback_v1` currently mocks the model inference part (always falls back).
- Next step: Implement Model Factory and Trainers (Phase 4).
