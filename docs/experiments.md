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
