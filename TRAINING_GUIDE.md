# Training Guide: Configuration, Local & GCP

This guide explains how to configure and train models using the `ear-vision-ml` repository. It is designed for engineers joining the team to get up to speed quickly.

## 1. Configuration System (Hydra)

We use [Hydra](https://hydra.cc/) for configuration management. This allows us to compose complex configurations from smaller, reusable files and override them dynamically from the command line.

### Directory Structure
Configs are located in the `configs/` directory. The structure is hierarchical:

```text
configs/
├── config.yaml              # Main entry point (defaults)
├── model/                   # Model architectures
│   ├── cls_mobilenetv3.yaml
│   └── cls_efficientnet.yaml
├── data/                    # Dataset & loader settings
│   ├── local.yaml
│   └── gcp.yaml
├── training/                # Hyperparameters (epochs, optimizer)
│   └── default.yaml
└── ...
```

### Understanding `config.yaml`
The `configs/config.yaml` file defines the **defaults**. When you run a script, these are the configs that are loaded unless you override them.

```yaml
# configs/config.yaml
defaults:
  - task: classification
  - model: cls_mobilenetv3
  - data: local
  - training: default
  - _self_
```

### How to Use Hydra Configs

#### 1. Command Line Overrides
You can change *any* value in the config from the command line using dot notation. You do **not** need to edit YAML files for quick experiments.

**Syntax:** `key=value` or `group.key=value`

| Goal | Command Line Argument | Explanation |
| :--- | :--- | :--- |
| **Change scalar value** | `training.epochs=10` | Sets `epochs` inside `training` group to 10. |
| **Change nested value** | `training.optimizer.lr=1e-4` | Sets learning rate deep inside the config. |
| **Swap entire config group** | `model=cls_efficientnet` | Replaces the entire `model` config with `configs/model/cls_efficientnet.yaml`. |
| **Disable a group** | `export=null` | Removes the `export` config group entirely. |

#### 2. Creating New Configurations
If you find yourself using the same overrides repeatedly, create a new YAML file.

**Scenario:** You want a "fast debug" training config.
1. Create `configs/training/debug.yaml`:
   ```yaml
   # configs/training/debug.yaml
   epochs: 1
   batch_size: 2
   optimizer:
     learning_rate: 0.01
   ```
2. Use it in your command:
   ```bash
   ./scripts/local_train.sh training=debug
   ```

#### 3. Multi-Run (Sweeps)
Hydra can run multiple jobs with different configs automatically using the `-m` (multirun) flag.

**Example:** Train with three different learning rates sequentially.
```bash
python -m src.tasks.classification.entrypoint -m \
    training.optimizer.lr=1e-3,1e-4,1e-5
```

---

## 2. Local Training Examples

Training locally is useful for debugging and small-scale experiments.

### Prerequisites
- Python environment set up (see `README.md`).
- Data available locally (or use synthetic data).

### Scenario A: Smoke Test (Does it run?)
Use synthetic data to verify your code changes without needing real data.
```bash
./scripts/local_train.sh \
    data=synthetic \
    training.epochs=1 \
    model.num_classes=3
```

### Scenario B: Overfitting a Small Batch
Verify your model can learn by overfitting a tiny amount of real data.
```bash
./scripts/local_train.sh \
    data=local \
    data.dataset.batch_size=4 \
    training.epochs=50 \
    training.optimizer.lr=1e-3
```

### Scenario C: Changing Model Architecture
Switch from MobileNet (default) to EfficientNet.
```bash
./scripts/local_train.sh \
    model=cls_efficientnet \
    training.epochs=5
```

---

## 3. GCP Training (Vertex AI)

For full-scale training, we use Google Cloud Vertex AI.

### Prerequisites
- `gcloud` CLI installed and authenticated.
- Data uploaded to Google Cloud Storage (GCS).

### Submitting a Job
Use the `scripts/vertex_submit.sh` script.

**Syntax:**
```bash
./scripts/vertex_submit.sh <TASK> <CONFIG_NAME> <GCS_STAGING_BUCKET> <REGION>
```

**Example:**
```bash
./scripts/vertex_submit.sh \
    classification \
    config \
    gs://my-staging-bucket \
    europe-west2
```

### Advanced GCP Configuration
The `vertex_submit.sh` script passes arguments to the container. You can modify the script to inject specific Hydra overrides if needed, or create a dedicated config file (e.g., `configs/config_gcp_experiment.yaml`) and pass that as the `<CONFIG_NAME>`.

---

## 4. Advanced Scenarios

### A. Resuming Training
If a job is interrupted, you can resume it by pointing to the checkpoint directory.

```bash
./scripts/local_train.sh \
    training.resume_from_checkpoint=outputs/2023-10-27/10-00-00/checkpoints/latest.ckpt
```

### B. Debugging Data Pipeline
If you suspect issues with data augmentation or loading, enable the overlay saver. This will save images with bounding boxes/masks drawn on them to the output directory.

```bash
./scripts/local_train.sh \
    debug.save_preprocess_overlays=true \
    debug.overlay_samples=10
```

### C. Using a Specific GPU on GCP
To use a more powerful GPU (e.g., A100) on Vertex AI, you need to modify the `scripts/vertex_submit.sh` script or create a custom submission command.

**Example modification to `vertex_submit.sh`:**
```bash
# Change accelerator-type to NVIDIA_TESLA_A100 and machine-type to a2-highgpu-1g
gcloud ai custom-jobs create \
  ... \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1...
```

### D. Fine-Tuning / Transfer Learning
You can freeze the backbone of the model to fine-tune only the head (classification layers). This is useful when training on small datasets using pre-trained weights.

**Configuration:**
```bash
./scripts/local_train.sh \
    model.transfer_learning.freeze_backbone=true
```

**Partial Unfreezing:**
To unfreeze the last N layers of the backbone (e.g., for gradual fine-tuning):
```bash
./scripts/local_train.sh \
    model.transfer_learning.freeze_backbone=true \
    model.transfer_learning.unfreeze_top_n_layers=10
```

---

## 5. Common Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `Config composition error` | The config group (e.g., `model=foo`) does not exist. | Check `configs/model/` for the correct filename (without `.yaml`). |
| `Key 'foo' not in struct` | You are trying to add a new key that isn't in the schema/config. | Use `+foo=bar` to append new keys (if strict mode allows), or check spelling. |
| `Missing mandatory value` | A required config field (e.g., `???`) was not set. | Provide the value via command line `group.key=value`. |
| `OOM (Out of Memory)` | Batch size too large for GPU memory. | Reduce `data.dataset.batch_size` (e.g., `data.dataset.batch_size=16`). |

## 6. Quick Reference Cheat Sheet

```bash
# Run default
./scripts/local_train.sh

# Run with synthetic data
./scripts/local_train.sh data=synthetic

# Change hyperparameters
./scripts/local_train.sh training.optimizer.lr=0.001 training.epochs=20

# Change model backbone
./scripts/local_train.sh model=cls_resnet50

# Run a sweep (try multiple values)
python -m src.tasks.classification.entrypoint -m training.optimizer.lr=0.1,0.01

# Debug data loading
./scripts/local_train.sh debug.save_preprocess_overlays=true
```
