# Quickstart Guide

Get up and running with `ear-vision-ml` in under 5 minutes.

## Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or `venv`
- **Git**

## 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ear-vision-ml

# Create conda environment
conda env create -f config/env/conda-tf217.yml
conda activate ear-vision-ml

# Install package in editable mode
pip install -e .
```

## 2. Verify Installation

```bash
# Run the test suite (should complete in ~70s)
pytest -v

# Expected output: 71 tests, 70 passed, 1 skipped
```

## 3. Run Your First Training (Smoke Test)

```bash
# Train a lightweight classification model with synthetic data
python -m src.tasks.classification.entrypoint \
  data.dataset.mode=synthetic \
  model=cls_mobilenetv3 \
  training.epochs=2 \
  run.name=quickstart_test
```

**Expected Output:**
- Training completes in ~30 seconds
- Artifacts saved to `artifacts/quickstart_test/`
- Model checkpoint, metrics, and logs generated

## 4. Explore Configurations

```bash
# View available model configs
ls configs/model/

# View available training configs
ls configs/training/

# Override any config parameter
python -m src.tasks.classification.entrypoint \
  model=cls_efficientnetb0 \
  training.batch_size=16 \
  training.learning_rate=0.001
```

## 5. Next Steps

- **Real Data**: See [datasets.md](datasets.md) for manifest format
- **Preprocessing**: See [preprocessing.md](preprocessing.md) for ROI pipelines
- **Export**: Models auto-export to SavedModel and TFLite
- **Vertex AI**: Use `scripts/vertex_submit.sh` for cloud training

## Common Commands

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Lint code
ruff check .

# Format code
ruff format .

# Type check
mypy src/core/
```

## Troubleshooting

If you encounter issues, see [troubleshooting.md](troubleshooting.md) or check the logs in `outputs/`.
