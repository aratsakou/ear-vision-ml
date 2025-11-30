# Preprocessing (ROI)

Preprocessing is a first-class component. Pipelines must be swappable via config.

## ROI contract
See `src/core/contracts/roi_contract.py`.

## Pipelines (MVP)
- full_frame_v1
- cropper_model_v1 (requires cropper model path)
- cropper_fallback_v1

## Debug overlays
See `src/core/preprocess/debug_viz.py`.
