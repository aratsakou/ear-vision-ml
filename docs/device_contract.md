# Device ensemble contract (Swift)

This repo assumes the current iOS pipeline:
1) Cropper Core ML model outputs ROI bounding box coordinates (normalised xyxy).
2) Swift performs the crop on the original frame/image.
3) Cropped image is passed to one or more downstream Core ML models.
4) If multiple downstream models are used, Swift combines results (soft/weighted voting).

## ROI output contract
- bbox_xyxy_norm = [x1, y1, x2, y2] in [0,1]
- confidence in [0,1]
- Invalid bbox (x1>=x2 or y1>=y2) must trigger fallback strategy.

## Class ordering
Every downstream model artefact must ship a `model_manifest.json` containing:
- `class_labels` in the exact order of softmax output.

Swift must treat this ordering as canonical.
