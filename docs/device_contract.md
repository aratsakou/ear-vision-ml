# Device Pipeline Contract

## Overview
This document defines the strict interface contract between ML models and iOS Swift code for on-device inference.

## Cropper Model Contract

### Input
- **Tensor name**: `image`
- **Shape**: `[1, H, W, 3]` where H, W are model-specific (typically 224x224)
- **Type**: Float32
- **Range**: [0.0, 1.0] (normalized RGB)
- **Color space**: RGB (not BGR)

### Output
- **Tensor name**: `bbox_conf`
- **Shape**: `[1, 5]`
- **Type**: Float32
- **Format**: `[x1, y1, x2, y2, confidence]`
- **Coordinate system**: Normalized [0.0, 1.0] relative to input image dimensions
- **Confidence range**: [0.0, 1.0]

### Swift Integration
```swift
// 1. Preprocess image to [0,1] RGB
// 2. Run cropper model
// 3. Extract bbox: [x1, y1, x2, y2, conf]
// 4. If conf < threshold: use full frame
// 5. Crop original image using bbox
// 6. Run downstream model on cropped region
```

## Classification Model Contract

### Input
- **Tensor name**: `image`
- **Shape**: `[1, H, W, 3]`
- **Type**: Float32
- **Range**: [0.0, 1.0]

### Output
- **Tensor name**: `probs`
- **Shape**: `[1, num_classes]`
- **Type**: Float32
- **Format**: Softmax probabilities summing to 1.0

### Class Ordering
Class indices must match the order in `model_manifest.json` → `class_labels` array.

## Segmentation Model Contract

### Input
- **Tensor name**: `image`
- **Shape**: `[1, H, W, 3]`
- **Type**: Float32
- **Range**: [0.0, 1.0]

### Output
- **Tensor name**: `probs`
- **Shape**: `[1, H, W, num_classes]`
- **Type**: Float32
- **Format**: Per-pixel softmax probabilities

## Threshold Conventions

- **Cropper confidence threshold**: 0.5 (configurable in manifest)
- **Classification confidence threshold**: 0.7 (application-specific)
- **Segmentation pixel threshold**: 0.5 per class

## Model Manifest Requirements

Every exported model must include `model_manifest.json` with:
```json
{
  "model_id": "unique_id",
  "task_name": "classification|segmentation|cropper",
  "input_shape": [224, 224, 3],
  "outputs": {"probs": [1, num_classes]},
  "class_labels": ["class_0", "class_1", ...],
  "preprocess_pipeline_id": "full_frame_v1",
  "preprocess_pipeline_version": "1.0.0"
}
```

## Version Compatibility

- Models trained with different preprocessing pipelines are **not** interchangeable
- Always check `preprocess_pipeline_id` and `preprocess_pipeline_version` match
- Swift code must apply the same preprocessing as specified in manifest
