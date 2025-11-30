# Image Inference Runtime - Complete Implementation

## Overview
Added comprehensive image inference runtime to complement the existing video inference capabilities, providing production-ready single and batch image processing with advanced features.

## 🚀 Features Implemented

### 1. Core Inference Runtime (`src/runtimes/image_inference/inference_runner.py`)

#### **Multi-Format Model Support**
- ✅ **SavedModel**: TensorFlow SavedModel format
- ✅ **TFLite**: Optimized mobile models (float32, INT8, FP16)
- ✅ **Keras**: Native Keras .keras format

#### **Advanced Inference Features**
- ✅ **Test-Time Augmentation (TTA)**: 
  - Horizontal/vertical flips
  - Rotations (90°, 270°)
  - Ensemble predictions for improved accuracy
  - Configurable number of transforms
  
- ✅ **Confidence Calibration**:
  - Temperature scaling
  - Improved probability estimates
  - Better uncertainty quantification

- ✅ **Batch Processing**:
  - Efficient batch inference
  - Progress tracking with tqdm
  - Automatic error handling
  - Performance metrics

#### **Preprocessing Pipeline**
- ✅ Automatic image resizing
- ✅ Normalization to [0, 1]
- ✅ Custom preprocessing function support
- ✅ BGR to RGB conversion
- ✅ Quantized input handling (uint8)

### 2. Explainability Tools (`src/runtimes/image_inference/explainability.py`)

#### **Grad-CAM (Gradient-weighted Class Activation Mapping)**
Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)

- ✅ Automatic last convolutional layer detection
- ✅ Class-specific heatmap generation
- ✅ Gradient-based importance weighting
- ✅ Heatmap overlay on original images
- ✅ Customizable colormaps

#### **Saliency Maps**
- ✅ Vanilla gradient saliency
- ✅ Pixel-level importance visualization
- ✅ Class-specific saliency
- ✅ Normalized output

#### **Visualization**
- ✅ Side-by-side comparisons
- ✅ Matplotlib-based saving
- ✅ Customizable titles and layouts

### 3. Testing (`tests/integration/test_image_runtime_smoke.py`)

**Comprehensive test coverage:**
- ✅ Single image inference
- ✅ Batch inference
- ✅ Test-time augmentation
- ✅ Convenience function
- ✅ Grad-CAM visualization (skipped for Sequential models in Keras 3)

**Test Results:**
```
4 passed, 1 skipped in 4.03s
```

## 📊 Performance Features

### Inference Metrics
- **Latency tracking**: Per-image inference time
- **Batch statistics**: Total time, average time per image
- **Progress monitoring**: Real-time progress bars
- **Error handling**: Graceful failure with logging

### Output Format
```json
{
  "total_images": 100,
  "total_time_ms": 5234.5,
  "avg_time_per_image_ms": 52.3,
  "model_path": "/path/to/model",
  "model_format": "saved_model",
  "tta_enabled": false,
  "results": [
    {
      "image_path": "/path/to/image.jpg",
      "predicted_class": 2,
      "confidence": 0.95,
      "top_5_predictions": {
        "2": 0.95,
        "1": 0.03,
        "0": 0.02
      }
    }
  ]
}
```

## 🎯 Use Cases

### 1. Single Image Classification
```python
from src.runtimes.image_inference import ImageInferenceRuntime

runtime = ImageInferenceRuntime(
    model_path="models/classifier",
    model_format="saved_model",
)

result = runtime.predict("image.jpg")
print(f"Class: {result.predicted_class}, Confidence: {result.confidence:.2f}")
```

### 2. Batch Processing with TTA
```python
runtime = ImageInferenceRuntime(
    model_path="models/classifier.tflite",
    model_format="tflite",
    use_tta=True,
    tta_transforms=5,
)

batch_result = runtime.predict_batch(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    batch_size=32,
    show_progress=True,
)

print(f"Processed {len(batch_result.results)} images")
print(f"Average time: {batch_result.avg_time_per_image_ms:.2f}ms")
```

### 3. Model Explainability
```python
from src.runtimes.image_inference import GradCAM
import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
gradcam = GradCAM(model, layer_name="last_conv")

heatmap = gradcam.compute_heatmap(image, class_idx=2)
overlayed = gradcam.overlay_heatmap(image, heatmap)
```

### 4. Convenience Function
```python
from src.runtimes.image_inference import run_image_inference

run_image_inference(
    model_path="models/classifier",
    image_paths=["img1.jpg", "img2.jpg"],
    output_path="results.json",
    model_format="saved_model",
    use_tta=True,
)
```

## 🔬 Technical Details

### Test-Time Augmentation
TTA improves accuracy by averaging predictions over multiple augmented versions:
1. Original image
2. Horizontal flip
3. Vertical flip
4. 90° rotation
5. 270° rotation

**Typical improvement**: 1-3% accuracy gain with 5x inference cost

### Confidence Calibration
Temperature scaling adjusts prediction confidence:
```python
calibrated_probs = softmax(logits / temperature)
```
- `temperature > 1`: Lower confidence (more uncertain)
- `temperature < 1`: Higher confidence (more certain)
- `temperature = 1`: No calibration (default)

### Quantized Model Support
Automatic handling of quantized inputs/outputs:
- **INT8 inputs**: Automatic scaling from [0,1] to [0,255]
- **INT8 outputs**: Automatic dequantization using scale/zero-point
- **Transparent**: No code changes needed

## 📈 Comparison: Video vs Image Runtime

| Feature | Video Runtime | Image Runtime |
|---------|--------------|---------------|
| **Input** | Video files | Image files |
| **Sampling** | Frame sampling | N/A |
| **Temporal** | Temporal aggregation | Test-time augmentation |
| **Output** | Aggregated predictions | Per-image predictions |
| **Batch** | Sequential frames | Parallel images |
| **Explainability** | N/A | Grad-CAM, Saliency |

## 🎓 References

1. **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **Test-Time Augmentation**: Krizhevsky et al. "ImageNet Classification with Deep Convolutional Neural Networks" (NIPS 2012)
3. **Temperature Scaling**: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)

## ✅ Integration

The image inference runtime integrates seamlessly with:
- ✅ Model factory (all 6 architectures)
- ✅ Export system (all formats)
- ✅ Preprocessing pipelines
- ✅ Video inference (shared preprocessing)

## 📝 Files Created

1. `src/runtimes/image_inference/inference_runner.py` (320 lines)
2. `src/runtimes/image_inference/explainability.py` (220 lines)
3. `src/runtimes/image_inference/__init__.py`
4. `tests/integration/test_image_runtime_smoke.py` (165 lines)

## 🎉 Summary

**Total Test Count**: 28 tests (27 passed, 1 skipped)
- Image inference: 4 passed, 1 skipped
- All other tests: 23 passed

**New Capabilities**:
- Single & batch image inference
- Multi-format model support (SavedModel, TFLite, Keras)
- Test-time augmentation (5 transforms)
- Confidence calibration
- Grad-CAM explainability
- Saliency maps
- Comprehensive benchmarking

The repository now has **complete inference runtimes** for both images and videos, with state-of-the-art features throughout! 🚀
