# Hyperparameter Tuning

Integrate with Vertex AI Vizier for automated hyperparameter optimization.

## Implementation
- **Interface**: `src/core/tuning/hyperparam_tuner.py`
- **Vertex Implementation**: `src/core/tuning/vertex_vizier.py` (wraps `hypertune` library)
- **Trainer Integration**: `StandardTrainer` adds a callback to report metrics if `tuning.enabled=true`.

## Configuration
See `configs/tuning/default.yaml`.

## Usage (Vertex AI)
When submitting a job to Vertex AI, you must configure the `hyperparameter_tuning_job` spec.
The training code inside the container will automatically report metrics to Vizier via the `hypertune` library.

```bash
# Example submission (conceptual)
gcloud ai hp-tuning-jobs create \
  --display-name="tuning-job" \
  --max-trial-count=20 \
  --parallel-trial-count=2 \
  --config=configs/tuning/default.yaml \
  ...
```
