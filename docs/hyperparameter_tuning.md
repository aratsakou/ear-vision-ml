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

## Local Tuning (Keras Tuner)
You can run hyperparameter optimization locally using Keras Tuner's Bayesian Optimization (Gaussian Process).

### Usage
Run the `tune_locally.py` script:
```bash
python scripts/tune_locally.py +experiment=otoscopic_baseline
```

### Configuration
The search space is defined in `configs/tuning/default.yaml`. You can create new tuning configs (e.g., `configs/tuning/experiment_1.yaml`) and select them via the command line.

**Example Config (`configs/tuning/default.yaml`)**:
```yaml
study_name: "otoscopic_optimization"
objective: "val_acc"
direction: "maximize"
max_trials: 10
parameters:
  learning_rate:
    target: "training.learning_rate" # Dot-notation path in main config
    type: DOUBLE
    min_value: 1e-5
    max_value: 1e-2
    scale: log
```

**Running with custom config**:
```bash
python scripts/tune_locally.py tuning=experiment_1
```

### Output
- **Logs**: Progress is printed to the console.
- **Artifacts**: Each trial creates a separate artifact directory.
- **Results**: Tuning results are saved in the `tuning_results` directory.

