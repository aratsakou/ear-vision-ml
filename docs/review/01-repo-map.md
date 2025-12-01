# Repository Map

## Top-Level Structure

- `configs/`: Hydra configuration files.
- `docs/`: Documentation.
- `scripts/`: Utility scripts and entry points.
- `src/`: Source code.
    - `src/core/`: Core infrastructure (DI, training, export, data, logging).
    - `src/tasks/`: Task-specific implementations (classification, segmentation, cropper).
    - `src/runtimes/`: Inference runtimes (image, video).
    - `src/ensembles/`: Ensemble logic.
- `tests/`: Unit and integration tests.

## Entry Points

### Training
- `src/tasks/classification/entrypoint.py`: Classification training.
- `src/tasks/segmentation/entrypoint.py`: Segmentation training.
- `src/tasks/cropper/entrypoint.py`: Cropper training.

### Data Processing
- `scripts/build_otoscopic_dataset.py`: Builds otoscopic dataset (parquet).
- `scripts/analyze_otoscopic_images.py`: Analyzes image statistics.

### Export
- `scripts/export_model.sh`: Shell script wrapper for export.
- `src/core/export/exporter.py`: Main export logic (SavedModel, TFLite).

### Inference/Runtime
- `src/runtimes/video_inference/offline_runner.py`: Offline video inference.
- `src/runtimes/image_inference/inference_runner.py`: Single image inference.

## Configuration (Hydra)

Configs are located in `configs/` and resolved via Hydra.
- `configs/config.yaml`: Main entry point.
- `configs/task/`: Task-specific defaults.
- `configs/model/`: Model architecture definitions.
- `configs/data/`: Data loading configurations.
- `configs/training/`: Training hyperparameters.

## Contracts & Schemas

- `src/core/contracts/dataset_manifest_schema.json`: Schema for dataset manifests.
- `src/core/contracts/model_manifest_schema.json`: Schema for model export manifests.
- `src/core/contracts/ontology.yaml`: Label ontology.

## Artifacts

- `experiments/`: (Default) Training artifacts (checkpoints, logs).
- `data/`: Local datasets (parquet files).

## Logging & Experiment Tracking

- `src/core/logging/`: Abstractions for logging.
    - `vertex_experiments.py`: Vertex AI Experiments integration.
    - `bq_logger.py`: BigQuery logging.
    - `local_logger.py`: Local file logging.

## Testing

- `tests/unit/`: Unit tests for core components.
- `tests/integration/`: Integration tests for workflows (training, export).
