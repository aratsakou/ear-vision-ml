# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### Conda Environment Creation Fails

**Symptom:** `conda env create` fails with dependency conflicts.

**Solution:**
```bash
# Clear conda cache
conda clean --all

# Retry with explicit channel priority
conda env create -f config/env/conda-tf217.yml --channel conda-forge
```

### TensorFlow Import Errors

**Symptom:** `ImportError: cannot import name 'xyz' from 'tensorflow'`

**Solution:**
```bash
# Verify TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"
# Should output: 2.17.x

# Reinstall if version is wrong
pip install --force-reinstall tensorflow==2.17.0
```

## Training Issues

### Out of Memory (OOM) Errors

**Symptom:** Training crashes with `ResourceExhaustedError`.

**Solution:**
```bash
# Reduce batch size
python -m src.tasks.classification.entrypoint \
  training.batch_size=8  # Default is 32

# Enable mixed precision (reduces memory by ~50%)
python -m src.tasks.classification.entrypoint \
  training=mixed_precision
```

### Manifest Not Found

**Symptom:** `FileNotFoundError: Manifest not found at ...`

**Solution:**
1. Verify manifest path in config: `configs/data/local.yaml`
2. Ensure `manifest.json` exists in the specified directory
3. Check manifest schema with:
   ```bash
   python -c "from src.core.data.dataset_loader import load_manifest; load_manifest('path/to/manifest.json')"
   ```

### Config Override Not Working

**Symptom:** Hydra config override has no effect.

**Solution:**
```bash
# Use correct syntax (no spaces around =)
python -m src.tasks.classification.entrypoint model=cls_mobilenetv3  # ✅ Correct
python -m src.tasks.classification.entrypoint model = cls_mobilenetv3  # ❌ Wrong

# For nested configs, use dot notation
python -m src.tasks.classification.entrypoint training.learning_rate=0.001

# For new config groups, use +
python -m src.tasks.classification.entrypoint +experiment=otoscopic_baseline
```

## Test Issues

### Tests Fail with Network Errors

**Symptom:** Tests fail trying to connect to GCS/BigQuery/Vertex.

**Solution:**
```bash
# Ensure cloud logging is disabled in test configs
# Check that configs/config.yaml has:
# log_vertex_experiments: false
# log_bigquery: false

# Tests should never require network access
# If they do, this is a bug - please report it
```

### Pytest Collection Errors

**Symptom:** `pytest` fails to collect tests.

**Solution:**
```bash
# Clear pytest cache
rm -rf .pytest_cache

# Reinstall package in editable mode
pip install -e .

# Run pytest with verbose output
pytest -v --collect-only
```

## Export Issues

### TFLite Conversion Fails

**Symptom:** `RuntimeError: TFLite conversion failed`

**Solution:**
1. Ensure model is trained and saved:
   ```bash
   ls artifacts/<run_name>/saved_model/
   ```
2. Check model compatibility:
   ```bash
   # Some ops are not TFLite compatible
   # Use the export config to specify conversion options
   python -m src.tasks.classification.entrypoint \
     export.tflite.quantize=false
   ```

### Core ML Export Not Available

**Symptom:** `ImportError: coremltools not found`

**Solution:**
```bash
# Install optional dependency
pip install coremltools

# Or skip Core ML export
# (TFLite and SavedModel will still be generated)
```

## Vertex AI Issues

### Submission Script Fails

**Symptom:** `scripts/vertex_submit.sh` fails with authentication error.

**Solution:**
```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify credentials
gcloud auth list
```

### Container Build Fails

**Symptom:** Vertex job fails during container build.

**Solution:**
1. Check Docker configuration in `config/docker/`
2. Verify base image is accessible
3. Test build locally:
   ```bash
   docker build -f config/docker/Dockerfile .
   ```

## Performance Issues

### Training is Very Slow

**Symptom:** Training takes much longer than expected.

**Solutions:**
```bash
# 1. Enable mixed precision (2x speedup)
python -m src.tasks.classification.entrypoint training=mixed_precision

# 2. Reduce preprocessing overhead
# Check if you're using expensive augmentations
# Disable debug overlays:
debug.save_preprocess_overlays=false

# 3. Use larger batch size (if memory allows)
training.batch_size=64

# 4. Enable XLA compilation (experimental)
# Set environment variable:
export TF_XLA_FLAGS=--tf_xla_auto_jit=2
```

## Data Issues

### Images Not Loading

**Symptom:** `tf.errors.NotFoundError: No such file or directory`

**Solution:**
1. Check image URIs in manifest use `file://` prefix
2. Verify paths are absolute or relative to manifest directory
3. Test image loading:
   ```bash
   python -c "import tensorflow as tf; print(tf.io.read_file('path/to/image.jpg'))"
   ```

### Incorrect Label Distribution

**Symptom:** Model predicts only one class.

**Solution:**
1. Check label distribution in manifest:
   ```bash
   python scripts/analyze_otoscopic_images.py
   ```
2. Verify `num_classes` in config matches data
3. Consider using focal loss for class imbalance:
   ```bash
   python -m src.tasks.classification.entrypoint training.loss=focal
   ```

## Getting Help

If your issue is not listed here:

1. **Check logs**: Look in `outputs/` for detailed error messages
2. **Search devlogs**: Check `docs/devlog/` for related implementation notes
3. **Review ADRs**: Check `docs/adr/` for architectural decisions
4. **Run in debug mode**: Add `--cfg job` to see full Hydra config
   ```bash
   python -m src.tasks.classification.entrypoint --cfg job
   ```

## Known Limitations

1. **Grad-CAM**: Requires Keras 3 Functional API (currently skipped in tests)
2. **Core ML**: Optional dependency, not installed by default
3. **Vertex Experiments**: Requires Google Cloud authentication
4. **BigQuery Logging**: Requires project setup and credentials
