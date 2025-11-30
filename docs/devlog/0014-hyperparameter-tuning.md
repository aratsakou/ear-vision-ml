# Devlog 0014: Hyperparameter Tuning

**Date:** 2025-11-30
**Author:** System
**Status:** ✅ Complete

## Summary
Integrated Vertex AI Vizier support for automated hyperparameter tuning. This allows the training job to report metrics (e.g., validation accuracy) to Vertex AI, which then suggests new hyperparameter values for subsequent trials.

## Changes Made

### 1. Tuning Interface (`src/core/tuning/`)
-   **`HyperparameterTuner`**: Abstract base class defining the interface.
-   **`VertexVizierTuner`**: Implementation using the `hypertune` library (standard for Vertex AI Training). It gracefully degrades to printing to stdout when running locally.

### 2. Trainer Integration (`src/core/training/standard_trainer.py`)
-   Added logic to check `cfg.tuning.enabled`.
-   If enabled, initializes `VertexVizierTuner` and adds a `LambdaCallback` to report metrics (`val_accuracy` or `val_loss`) at the end of each epoch.

### 3. Configuration (`configs/tuning/default.yaml`)
-   Added default search space configuration for `learning_rate` and `dropout`.

### 4. Documentation
-   Created `docs/hyperparameter_tuning.md` with usage instructions.

## Testing
-   **Unit Tests**: Created `tests/unit/test_tuning.py` to verify:
    -   Tuner initialization.
    -   Local fallback behavior (printing to stdout).
-   **Status**: All tests passed.

## Usage
Enable tuning in your config:
```yaml
tuning:
  enabled: true
```
When running on Vertex AI as a Hyperparameter Tuning Job, the metrics will be automatically captured.
