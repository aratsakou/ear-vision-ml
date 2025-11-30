# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ear-vision-ml` is a TensorFlow 2.17 domain monorepo for otoscopy vision models (ROI cropper + downstream classification/segmentation), designed for local macOS experimentation and Vertex AI custom training jobs. The repository is **test-driven** and **documentation-driven**.

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f config/env/conda-tf217.yml
conda activate ear-vision-ml

# Install repository (editable)
pip install -e .
```

### Testing
```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/unit/test_roi_contract.py -v

# Run integration tests
pytest tests/integration/ -v

# Run logging tests
pytest tests/unit/test_logging.py -v

# Run image runtime tests
pytest tests/integration/test_image_runtime_smoke.py -v

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Lint with ruff
ruff check .

# Format with ruff
ruff format .

# Type check core modules with mypy
mypy src/core/
```

### Training
```bash
# Local smoke training (classification)
python -m src.tasks.classification.entrypoint

# Local smoke training (segmentation)
python -m src.tasks.segmentation.entrypoint

# Local smoke training (cropper)
python -m src.tasks.cropper.entrypoint

# Local training with custom config overrides
python -m src.tasks.classification.entrypoint task=classification model=cls_mobilenetv3
```

### Vertex AI Submission
```bash
# Generate Vertex AI submission command (does not auto-submit)
./scripts/vertex_submit.sh
```

## Architecture Principles

### ROI-Centric Design
The repository treats ROI (Region of Interest) as a first-class concept:
- **ROI Contract**: All preprocessing pipelines must adhere to the contract defined in [src/core/contracts/roi_contract.py](src/core/contracts/roi_contract.py)
- **Normalized coordinates**: Bounding boxes are in `[x1, y1, x2, y2]` format with values in `[0,1]` relative to image dimensions
- **Source tracking**: Every ROI includes its source: `cropper`, `fallback`, or `full_frame`
- **Swappable pipelines**: Preprocessing pipelines are configured via Hydra configs without code changes

### Configuration System (Hydra)
All experiments are controlled via Hydra configs in [configs/](configs/):
- **Main config**: [configs/config.yaml](configs/config.yaml) defines defaults for all subsystems
- **Task configs**: [configs/task/](configs/task/) define classification, segmentation, cropper, and video runtime
- **Model configs**: [configs/model/](configs/model/) define model architectures (MobileNetV3, U-Net, etc.)
- **Preprocess configs**: [configs/preprocess/](configs/preprocess/) define ROI preprocessing pipelines
- **Data configs**: [configs/data/](configs/data/) define dataset sources and formats
- **Training configs**: [configs/training/](configs/training/) define training settings, mixed precision, distributed training

To run with custom config: use Hydra override syntax like `task=cropper preprocess=cropper_model_v1`

### Dataset System
Datasets follow a strict contract:
- **Immutable**: Once marked active, datasets are never modified; changes create new versions
- **Parquet format**: Task datasets stored as Parquet shards (train/val/test splits)
- **Manifest-driven**: Each dataset folder contains `manifest.json` (validated against [src/core/contracts/dataset_manifest_schema.json](src/core/contracts/dataset_manifest_schema.json)) and `stats.json`
- **URI + timestamps**: Data rows reference media via URI and optional timestamps (sampling on-the-fly)
- **Registry logging**: Dataset versions logged to SQL (minimal) and BigQuery (rich metadata)

Schema validation happens at load time; see [src/core/data/dataset_loader.py](src/core/data/dataset_loader.py)

### Preprocessing Pipeline Registry
Preprocessing pipelines are versioned and registered:
- **Registry**: [src/core/preprocess/registry.py](src/core/preprocess/registry.py) maintains the pipeline catalog
- **Pipelines**: Located in [src/core/preprocess/pipelines/](src/core/preprocess/pipelines/)
  - `full_frame_v1`: Resize/normalize only (no ROI cropping)
  - `cropper_model_v1`: Uses cropper model to produce ROI bbox
  - `cropper_fallback_v1`: Uses cropper when confident, else deterministic fallback
- **Interface**: All pipelines implement `apply(image, metadata, cfg) -> (image_out, metadata_out)`
- **Debug visualization**: Use [src/core/preprocess/debug_viz.py](src/core/preprocess/debug_viz.py) to generate overlay images

### Model Factory Pattern
Models are created via [src/core/models/factories/model_factory.py](src/core/models/factories/model_factory.py):
- Register new models in the factory
- Define corresponding config in [configs/model/](configs/model/)
- Factory builds models from config; unit tests verify forward passes on dummy tensors

### Training Tasks
Three task types with parallel structure:
- **Classification**: [src/tasks/classification/](src/tasks/classification/) - multi-class otoscopy classification
- **Segmentation**: [src/tasks/segmentation/](src/tasks/segmentation/) - segmentation with U-Net baseline
- **Cropper**: [src/tasks/cropper/](src/tasks/cropper/) - ROI detection model (outputs bbox + confidence)

Each task has: `entrypoint.py` (Hydra-based CLI), `trainer.py` (task-specific training logic), `evaluation.py` (metrics/eval)

### Experiment Tracking
Multi-layered logging system:
- **Console**: User-facing info
- **File**: Detailed debug logs
- **JSON**: Machine-readable logs
- **Reporting**: HTML/Markdown reports for setup and results
- **Vertex Experiments**: Enable via config `run.log_vertex_experiments: true`

### Inference Runtimes
- **Video**: [src/runtimes/video_inference/](src/runtimes/video_inference/) - Frame sampling, temporal aggregation
- **Image**: [src/runtimes/image_inference/](src/runtimes/image_inference/) - TTA, batch processing, explainability

## Documentation-Driven Development

**Critical**: Every meaningful development step must be documented:

1. **Devlog entries**: Create/update markdown files in [docs/devlog/](docs/devlog/)
   - Include: what was implemented, files created/modified, how to run, tests added, known limitations
   - Use template: [docs/devlog/0000-template.md](docs/devlog/0000-template.md)

2. **Architecture Decision Records (ADRs)**: Document major decisions in [docs/adr/](docs/adr/)
   - Required for: contract changes, schema changes, pipeline additions, major architectural shifts
   - Use template: [docs/adr/0000-template.md](docs/adr/0000-template.md)

3. **Topic docs**: Update relevant docs in [docs/](docs/):
   - [docs/datasets.md](docs/datasets.md) - dataset schemas and manifests
   - [docs/preprocessing.md](docs/preprocessing.md) - ROI pipelines
   - [docs/experiments.md](docs/experiments.md) - local and Vertex runs
   - [docs/deployment_ios.md](docs/deployment_ios.md) - device pipeline contracts
   - [docs/ensembles.md](docs/ensembles.md) - ensemble strategies

See [docs/repo_rules.md](docs/repo_rules.md) for contribution guidelines.

## Repository Rules (Non-Negotiables)

From [docs/repo_rules.md](docs/repo_rules.md):
- **No hardcoded dataset paths** in training code; use data config and contracts
- **Preprocessing pipelines are swappable** via config and must adhere to ROI contract
- **Datasets are immutable** once marked active; changes create new versions
- **Core modules** ([src/core/](src/core/) and [src/core/contracts/](src/core/contracts/)) require review and must maintain backward compatibility

## Adding New Components

### Adding a new model
1. Implement model in [src/core/models/](src/core/models/) (use existing backbones from [src/core/models/backbones/](src/core/models/backbones/))
2. Register in [src/core/models/factories/model_factory.py](src/core/models/factories/model_factory.py)
3. Create config file in [configs/model/](configs/model/)
4. Add unit test in [tests/unit/test_model_factory.py](tests/unit/test_model_factory.py)
5. Document in devlog

### Adding a new preprocessing pipeline
1. Implement pipeline in [src/core/preprocess/pipelines/](src/core/preprocess/pipelines/)
2. Register in [src/core/preprocess/registry.py](src/core/preprocess/registry.py)
3. Create config file in [configs/preprocess/](configs/preprocess/)
4. Add unit test in [tests/unit/test_preprocess_registry.py](tests/unit/test_preprocess_registry.py)
5. Document in [docs/preprocessing.md](docs/preprocessing.md) and devlog

### Adding/modifying contracts
1. Update contract in [src/core/contracts/](src/core/contracts/)
2. Add comprehensive unit tests
3. Create ADR entry in [docs/adr/](docs/adr/)
4. Update all affected code
5. Verify backward compatibility or document breaking changes

## Testing Philosophy

- **No network access in tests**: Use local fixtures in [tests/fixtures/](tests/fixtures/)
- **Smoke tests**: Integration tests run 2-step trainings on fixture datasets
- **Unit tests**: Core contracts, schemas, and factories must have unit tests
- **Test first**: Every feature should have tests before implementation
- Tests are the CI gates; see [pyproject.toml](pyproject.toml) for pytest configuration

## Tech Stack

- **Framework**: TensorFlow 2.17.x (Keras model subclassing, tf.data pipelines)
- **Config**: Hydra + OmegaConf
- **Data**: Parquet (via pyarrow), pandas, numpy
- **Cloud**: google-cloud-aiplatform, google-cloud-storage, google-cloud-bigquery
- **Media**: opencv-python-headless for image/video processing
- **Quality**: pytest, ruff (lint/format), mypy (type checking)

## Vertex AI Integration

The repository supports Vertex AI custom training using official TF 2.17 prebuilt containers:
- **Container**: `europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest`
- **Submission**: Use [scripts/vertex_submit.sh](scripts/vertex_submit.sh) to generate `gcloud ai custom-jobs create` command
- **Same entrypoints**: Local and Vertex runs use identical entrypoints with Hydra configs
- **Experiment tracking**: Vertex Experiments integration via [src/core/logging/vertex_experiments.py](src/core/logging/vertex_experiments.py)

## Common Workflows

### Running a local experiment
```bash
# 1. Ensure environment is active
conda activate ear-vision-ml

# 2. Run training with custom config
python -m src.tasks.classification.entrypoint \
  task=classification \
  model=cls_mobilenetv3 \
  preprocess=cropper_fallback_v1

# 3. Find outputs in artifacts/runs/<run_id>/
```

### Switching preprocessing pipelines
```bash
# Full frame (no cropping)
python -m src.tasks.classification.entrypoint preprocess=full_frame_v1

# With cropper model
python -m src.tasks.classification.entrypoint preprocess=cropper_model_v1

# With fallback strategy
python -m src.tasks.classification.entrypoint preprocess=cropper_fallback_v1
```

### Exporting models
```bash
# Run export via script
./scripts/export_model.sh

# Or directly via Python module (implementation in src/core/export/exporter.py)
```

## Key Files to Understand

When working in this repository, familiarize yourself with these core contracts:
- [src/core/contracts/roi_contract.py](src/core/contracts/roi_contract.py) - ROI bounding box contract
- [src/core/contracts/dataset_manifest_schema.json](src/core/contracts/dataset_manifest_schema.json) - Dataset manifest validation schema
- [src/core/contracts/model_manifest_schema.json](src/core/contracts/model_manifest_schema.json) - Model artefact manifest schema
- [docs/repo_rules.md](docs/repo_rules.md) - Repository contribution rules
- [configs/config.yaml](configs/config.yaml) - Hydra config defaults
