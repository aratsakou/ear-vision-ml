# 0002-config-system

**Date:** 2025-11-28
**Author:** AI Agent

## What was implemented
- Implemented task entrypoints for `cropper`, `classification`, and `segmentation`.
- Verified Hydra configuration loading and resolution.

## Files created/modified
- `src/tasks/cropper/entrypoint.py`
- `src/tasks/classification/entrypoint.py`
- `src/tasks/segmentation/entrypoint.py`
- `docs/devlog/0002-config-system.md`
- `docs/adr/0002-config-strategy.md`

## How to run it
Run any task entrypoint:
```bash
python src/tasks/cropper/entrypoint.py task=cropper
python src/tasks/classification/entrypoint.py task=classification
python src/tasks/segmentation/entrypoint.py task=segmentation
```

## Tests added/updated
- Manual verification of entrypoint execution.

## Known limitations and next steps
- Entrypoints currently only print the config.
- Next step: Implement dataset manifest loader (Phase 2).
