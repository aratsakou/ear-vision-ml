# iOS deployment

Current device flow:
1) Cropper Core ML model outputs bbox coords.
2) Swift crops and passes the crop to downstream models.
3) Downstream models output class probabilities / segmentation masks.

Contracts:
- ROI bbox schema: `roi_bbox_xyxy_norm` in xyxy normalised form.
- Downstream class ordering must be recorded in `model_manifest.json`.
