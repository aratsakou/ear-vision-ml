# Repository Map

## 1. Top-level Inventory

### Directory Tree Summary
- `configs/`: Hydra configuration files. Public API for the repo.
- `src/`: Source code.
    - `core/`: Shared components (data, training, models, etc.).
    - `tasks/`: Task-specific implementations (classification, segmentation, cropper).
    - `ensembles/`: Ensemble logic.
    - `runtimes/`: Inference runtimes.
- `scripts/`: Utility scripts and entrypoint wrappers.
- `tests/`: Test suite.
- `docs/`: Documentation.
- `artifacts/`: Local output directory for runs.

### Key Locations
- **Configs**: `configs/` (Root: `configs/config.yaml`)
- **Contracts/Schemas**: `src/core/contracts/`
- **Training Code**: `src/core/training/` and `src/tasks/*/training.py`
- **Inference Code**: `src/runtimes/`
- **Tests**: `tests/`
- **Docs**: `docs/`

## 2. Entrypoints Discovery

### Python Modules
- **Classification Training**: `python -m src.tasks.classification.entrypoint`
    - **Inputs**: Hydra config overrides (e.g., `model=...`, `data=...`).
    - **Outputs**: Trained model, metrics, logs in `run.artifacts_dir`.
- **Otoscopic Analysis**: `scripts/analyze_otoscopic_images.py`
    - **Command**: `python scripts/analyze_otoscopic_images.py`
    - **Purpose**: Analyze dataset statistics/properties.
- **Dataset Builder**: `scripts/build_otoscopic_dataset.py`
    - **Command**: `python scripts/build_otoscopic_dataset.py`
    - **Purpose**: Construct datasets from raw sources.
- **Fixture Generator**: `scripts/generate_fixtures.py`
    - **Command**: `python scripts/generate_fixtures.py`
    - **Purpose**: Create synthetic data for testing.

### Shell Scripts
- **Baseline Runner**: `scripts/run_otoscopic_baselines.sh`
    - **Purpose**: Runs training for MobileNetV3, EfficientNetB0, ResNet50V2.
- **E2E Test**: `scripts/run_e2e_test.sh`
    - **Purpose**: Full end-to-end test run.
- **Quick E2E**: `scripts/run_quick_e2e.sh`
    - **Purpose**: Faster version of E2E test.
- **Vertex Submit**: `scripts/vertex_submit.sh`
    - **Purpose**: Submit jobs to Vertex AI.

## 3. Config Surface Discovery

### Hydra Configuration
- **Root**: `configs/config.yaml`
- **Config Groups**:
    - `task`: `classification`, etc.
    - `model`: `cls_mobilenetv3`, `cls_efficientnetb0`, `cls_resnet50v2`, etc.
    - `data`: `local`, etc.
    - `preprocess`: `full_frame_v1`, etc.
    - `training`: `default`, etc.
    - `export`: `tflite`, etc.
    - `evaluation`: `default`, etc.
    - `ensemble`: `cloud_soft_voting`, etc.
    - `explainability`: `default`, `classification`, `segmentation`, `roi`, `dataset_audit`.

### Overrides
- Standard Hydra syntax: `key=value` or `+group=option`.
- Example: `python -m ... +experiment=otoscopic_baseline model=cls_mobilenetv3`.

## 4. Artefact and Logging Discovery

### Local Run Records
- Location: `artifacts/` (configurable via `run.artifacts_dir`).
- Structure: Typically `artifacts/<experiment_name>/<run_name>/`.

### Cloud Integration
- **Vertex Experiments**: Controlled by `log_vertex_experiments` (default: `false`).
- **BigQuery**: Controlled by `log_bigquery` (default: `false`).
- **SQL Dataset Logging**: Controlled by `log_sql_dataset_version` (default: `false`).

### Artefacts Produced
- SavedModels / TFLite files.
- Metrics (JSON/CSV).
- Explainability outputs (overlays, heatmaps).
