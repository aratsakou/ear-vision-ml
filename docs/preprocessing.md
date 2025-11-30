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

## How to extend
1. Create a new pipeline file in `src/core/preprocess/pipelines/` (e.g., `my_pipeline_v1.py`).
2. Implement the `apply` function matching the signature: `(image, metadata, cfg) -> (image, metadata)`.
3. Register it in `src/core/preprocess/registry.py` using `@register_pipeline("my_pipeline_v1")`.
4. Create a config file in `configs/preprocess/my_pipeline_v1.yaml`.
5. Add a unit test in `tests/unit/test_preprocess_registry.py`.
