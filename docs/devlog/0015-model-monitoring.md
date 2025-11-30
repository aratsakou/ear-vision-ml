# Devlog 0015: Model Monitoring (Drift Detection)

**Date:** 2025-11-30
**Author:** System
**Status:** ✅ Complete

## Summary
Implemented statistical drift detection to monitor data distribution shifts between training (baseline) and production (target) data. This ensures model reliability by alerting when input data deviates significantly.

## Changes Made

### 1. Drift Detector (`src/core/monitoring/drift_detector.py`)
-   **`DriftDetector`**: Implements:
    -   **PSI (Population Stability Index)**: Measures distributional shift. PSI >= 0.2 indicates significant drift.
    -   **KS-Test (Kolmogorov-Smirnov)**: Statistical test for equality of distributions. p-value < 0.05 indicates drift.

### 2. Dataset Builder Integration (`src/core/data/dataset_builder.py`)
-   Updated `DatasetBuilder` to compute and save baseline statistics (mean, std, min, max, histograms) for all numeric columns during dataset creation.
-   These stats are saved in `stats.json` and `manifest.json`.

### 3. Monitoring Task (`src/tasks/monitoring/entrypoint.py`)
-   Created a standalone task to compare two datasets (baseline vs target).
-   Outputs a `drift_report.json` and logs warnings if drift is detected.

## Testing
-   **Unit Tests**: Created `tests/unit/test_drift_detection.py` covering:
    -   PSI calculation (no drift vs significant drift).
    -   KS-statistic calculation.
    -   Full drift detection flow.
-   **Status**: All tests passed.

## Usage
Run the monitoring task:
```bash
python -m src.tasks.monitoring.entrypoint \
  monitoring.baseline_data_path=data/train.parquet \
  monitoring.target_data_path=data/production_batch.parquet \
  monitoring.features=['brightness', 'contrast']
```
