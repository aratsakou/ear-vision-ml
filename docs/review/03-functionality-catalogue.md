# Functionality Catalogue

## 1. Dataset Management

### Build Otoscopic Dataset
Converts raw image folders into a structured Parquet dataset.
- **File**: `scripts/build_otoscopic_dataset.py`
- **Command**:
  ```bash
  python scripts/build_otoscopic_dataset.py \
    --source-dir /path/to/raw_images \
    --output-dir data/otoscopic/v1
  ```
- **Inputs**: Directory with subfolders for each class (e.g., `Normal/`, `Acute Otitis Media/`).
- **Outputs**:
    - `data/otoscopic/v1/data/*.parquet`: The data files.
    - `data/otoscopic/v1/manifest.json`: Metadata.
    - `data/otoscopic/v1/stats.json`: Statistics.

### Analyze Images
Computes statistics (mean, std, dimensions) for a folder of images.
- **File**: `scripts/analyze_otoscopic_images.py`
- **Command**:
  ```bash
  python scripts/analyze_otoscopic_images.py --source-dir /path/to/images
  ```

## 2. Training

### Train Classification Model
Trains a model to classify ear conditions.
- **Entrypoint**: `src/tasks/classification/entrypoint.py`
- **Command**:
  ```bash
  python src/tasks/classification/entrypoint.py \
    task=classification \
    model=cls_mobilenetv3 \
    data.dataset.manifest_path=data/otoscopic/v1/manifest.json
  ```
- **Outputs**: `experiments/classification/<date>/saved_model/`

### Train Segmentation Model
Trains a model to segment ear parts (e.g., eardrum).
- **Entrypoint**: `src/tasks/segmentation/entrypoint.py`
- **Command**:
  ```bash
  python src/tasks/segmentation/entrypoint.py \
    task=segmentation \
    model=seg_unet
  ```

## 3. Export

### Export Model
Converts a trained model to TFLite and CoreML.
- **File**: `src/core/export/exporter.py` (usually called automatically after training).
- **Manual Trigger**:
  ```bash
  # Typically run via training config, but logic is in StandardExporter
  ```
- **Outputs**:
    - `model_float32.tflite`
    - `model_int8.tflite`
    - `model.mlpackage` (CoreML)
    - `model_manifest.json`

## 4. Inference

### Video Inference (Offline)
Runs a model on a video file.
- **File**: `src/runtimes/video_inference/offline_runner.py`
- **Command**:
  ```bash
  python src/runtimes/video_inference/offline_runner.py \
    --video-path video.mp4 \
    --model-path model.tflite
  ```

## 5. Utilities

### Generate Fixtures
Creates dummy data for testing.
- **File**: `scripts/generate_fixtures.py`
- **Command**: `python scripts/generate_fixtures.py`
