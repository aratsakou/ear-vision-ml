# Explainability Framework

The Explainability Framework provides auditable insights into datasets, ROI preprocessing, model predictions, and pipeline health. It is designed to be config-driven, reproducible, and integrated into the training and evaluation lifecycles.

## Overview

The framework operates at four levels:
1.  **Dataset Audit**: Checks for class imbalance, leakage, and label consistency.
2.  **ROI Audit**: Validates bounding box quality, confidence, and jitter (for video).
3.  **Model Explainability**: Generates attribution heatmaps (Classification) and uncertainty maps (Segmentation).
4.  **Prediction Reporting**: detailed reports for individual samples or video sequences.

## Configuration

Explainability is controlled via Hydra configs in `configs/explainability/`.

### Global Settings (`configs/explainability/default.yaml`)
```yaml
enabled: true
output_dir: "${run.artifacts_dir}/explainability"
seed: 42
max_samples: 10
```

### Classification (`configs/explainability/classification.yaml`)
Selects the attribution method (Integrated Gradients or Grad-CAM).

### Segmentation (`configs/explainability/segmentation.yaml`)
Controls uncertainty map generation (Entropy).

### ROI (`configs/explainability/roi.yaml`)
Sets thresholds for valid bounding boxes and jitter.

### Dataset Audit (`configs/explainability/dataset_audit.yaml`)
Enables leakage checks and distribution analysis.

## Usage

### Training Integration
The framework runs automatically at the end of training if `explainability.enabled=true`.

### CLI Usage
(Coming in Phase 6)
```bash
python -m src.core.explainability.cli run_id=...
```

## Artifacts

All artifacts are stored in `artifacts/runs/<run_id>/explainability/`.
The entry point is `explainability_manifest.json`, which lists all generated files.

### Key Artifacts
- `dataset_audit.json`: Dataset statistics and leakage report.
- `roi_audit.json`: ROI quality metrics.
- `attribution_summary.json`: Model attribution metrics.
- `seg_explain.json`: Segmentation uncertainty metrics.
- `overlays/`: Visualizations (heatmaps, uncertainty maps).
