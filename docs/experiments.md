# Experiments

## Local runs
Use task entrypoints under `src/tasks/*/entrypoint.py`.

All runs write a local record under:
- `artifacts/runs/<run_id>/run.json`

## Vertex runs
Use `scripts/vertex_submit.sh` to generate a `gcloud ai custom-jobs create` command.

## Vertex Experiments tracking
Enable via config:
- `run.log_vertex_experiments: true`

Implementation lives in `src/core/logging/vertex_experiments.py`.

## Configuration Overrides
Hydra allows overriding any config parameter from the command line:

```bash
# Override model and learning rate
python -m src.tasks.classification.entrypoint \
    model=cls_efficientnetb0 \
    training.learning_rate=0.0005

# Override nested parameters
python -m src.tasks.classification.entrypoint \
    data.dataset.mode=synthetic \
    run.name=experiment_v1
```

## Hyperparameter Tuning
Tuning is now integrated into the training config. To enable:

```bash
python -m src.tasks.classification.entrypoint \
    training.tuning.enabled=true \
    training.tuning.max_trials=10
```

## Analyzing Results
Results are saved to `artifacts/runs/<run_id>/`.
- `run.json`: Metadata and final metrics.
- `metrics.json`: Per-epoch metrics.
- `saved_model/`: Exported model artifacts.
- `logs/`: TensorBoard logs.

