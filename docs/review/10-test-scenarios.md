# Test Scenarios Matrix

Comprehensive test coverage for the `ear-vision-ml` repository.

## Test Execution Summary

| Category | Tests | Status | Coverage |
|----------|-------|--------|----------|
| Contracts/Schemas | 7 | ✅ Passing | Dataset manifests, ROI contract |
| Data Loading | 7 | ✅ Passing | Manifest + synthetic loaders |
| Preprocessing | 3 | ✅ Passing | Pipeline registry, swapping |
| Training | 5 | ✅ Passing | Standard trainer, DI, distillation |
| Export | 3 | ✅ Passing | SavedModel, TFLite, manifests |
| Explainability | 12 | ✅ Passing | All modules, integration |
| Runtimes | 4 | ✅ Passing | Image + video inference |
| Integration | 5 | ✅ Passing | End-to-end workflows |

## Detailed Test Scenarios

### 1. Contracts and Schemas

#### Scenario: Dataset Manifest Validation
- **Command**: `pytest tests/unit/test_dataset_manifest_schema.py -v`
- **Modules**: `src.core.contracts.dataset_manifest_schema.json`, `src.core.data.dataset_loader`
- **Artifacts**: None (validation only)
- **Pass Criteria**: 
  - Valid manifests pass validation
  - Missing required fields raise errors
  - Invalid enum values rejected

#### Scenario: ROI Contract Validation
- **Command**: `pytest tests/unit/test_roi_contract.py -v`
- **Modules**: `src.core.contracts.roi_contract`
- **Artifacts**: None
- **Pass Criteria**:
  - Bbox coordinates validated
  - Confidence scores in [0, 1]
  - Fallback logic tested

### 2. Data Loading

#### Scenario: Manifest-Based Dataset Loading
- **Command**: `pytest tests/integration/test_dataset_build_smoke.py::test_load_dataset_smoke -v`
- **Modules**: `src.core.data.dataset_loader.ManifestDataLoader`
- **Config**: `data.dataset.mode=manifest`
- **Artifacts**: TFRecord batches
- **Pass Criteria**:
  - Parquet files loaded correctly
  - Images decoded and resized
  - Labels one-hot encoded
  - Batching works

#### Scenario: Synthetic Data Generation
- **Command**: `pytest tests/unit/test_data_loader_strategy.py -v`
- **Modules**: `src.core.data.dataset_loader.SyntheticDataLoader`
- **Config**: `data.dataset.mode=synthetic`
- **Artifacts**: Synthetic tensors
- **Pass Criteria**:
  - Correct shapes for classification/segmentation/cropper
  - No file I/O required
  - Deterministic with seed

### 3. Preprocessing

#### Scenario: Pipeline Registry and Swapping
- **Command**: `pytest tests/unit/test_preprocess_registry.py -v`
- **Modules**: `src.core.preprocess.registry`
- **Config**: `preprocess=full_frame_v1` or `preprocess=roi_v1`
- **Artifacts**: None
- **Pass Criteria**:
  - Pipelines registered correctly
  - Config-driven selection works
  - ROI contract enforced

### 4. Training

#### Scenario: Minimal Classification Training
- **Command**: 
  ```bash
  python -m src.tasks.classification.entrypoint \
    data.dataset.mode=synthetic \
    model=cls_mobilenetv3 \
    training.epochs=2 \
    run.name=test_cls_minimal
  ```
- **Modules**: `src.core.training.standard_trainer`, `src.tasks.classification.entrypoint`
- **Config**: `task=classification`, `model=cls_mobilenetv3`, `training=default`
- **Artifacts**: 
  - `artifacts/test_cls_minimal/saved_model/`
  - `artifacts/test_cls_minimal/checkpoints/`
  - `artifacts/test_cls_minimal/metrics.json`
- **Pass Criteria**:
  - Training completes without errors
  - Loss decreases
  - Model saved
  - Metrics logged

#### Scenario: Segmentation Training
- **Command**:
  ```bash
  python -m src.tasks.segmentation.entrypoint \
    data.dataset.mode=synthetic \
    model=seg_unet \
    training.epochs=2 \
    run.name=test_seg_minimal
  ```
- **Modules**: `src.tasks.segmentation.entrypoint`
- **Config**: `task=segmentation`, `model=seg_unet`
- **Artifacts**: Saved model, masks
- **Pass Criteria**:
  - Dice coefficient computed
  - Mask predictions generated
  - Model exports successfully

#### Scenario: Distillation Training
- **Command**: `pytest tests/unit/test_distillation.py -v`
- **Modules**: `src.core.training.distillation.Distiller`
- **Config**: `training.distillation.enabled=true`
- **Artifacts**: Student model
- **Pass Criteria**:
  - Distillation loss computed
  - Student learns from teacher
  - Temperature scaling works

### 5. Export

#### Scenario: SavedModel Export
- **Command**: `pytest tests/integration/test_export_smoke.py -v`
- **Modules**: `src.core.export.saved_model_exporter`
- **Config**: `export=saved_model`
- **Artifacts**: `saved_model/` directory
- **Pass Criteria**:
  - SavedModel created
  - Loadable with `tf.keras.models.load_model`
  - Inference works

#### Scenario: TFLite Export
- **Command**: Included in `test_export_smoke.py`
- **Modules**: `src.core.export.tflite_exporter`
- **Config**: `export=tflite`
- **Artifacts**: 
  - `model.tflite` (float)
  - `model_quantized.tflite` (int8)
- **Pass Criteria**:
  - TFLite files generated
  - Quantization works
  - Inference equivalence (within tolerance)

### 6. Explainability

#### Scenario: Classification Explainability
- **Command**: `pytest tests/unit/test_attribution.py -v`
- **Modules**: `src.core.explainability.attribution`
- **Config**: `explainability.enabled=true`, `explainability.classification.enabled=true`
- **Artifacts**:
  - Heatmaps
  - Overlays
  - `explainability_manifest.json`
- **Pass Criteria**:
  - Heatmaps generated
  - Manifest links correct
  - Deterministic sampling

#### Scenario: Dataset Audit
- **Command**: `pytest tests/unit/test_dataset_audit.py -v`
- **Modules**: `src.core.explainability.dataset_audit`
- **Config**: `explainability.dataset_audit.enabled=true`
- **Artifacts**: Audit report
- **Pass Criteria**:
  - Class distribution computed
  - Sample images extracted
  - Report generated

### 7. Runtimes

#### Scenario: Image Inference (Batch)
- **Command**: `pytest tests/integration/test_image_runtime_smoke.py -v`
- **Modules**: `src.runtimes.image_inference`
- **Config**: `runtime=image_batch`
- **Artifacts**: Predictions JSON
- **Pass Criteria**:
  - Batch inference works
  - Predictions match expected format
  - ROI preprocessing applied

#### Scenario: Video Inference
- **Command**: `pytest tests/integration/test_video_runtime_smoke.py -v`
- **Modules**: `src.runtimes.video_inference`
- **Config**: `runtime=video`
- **Artifacts**: Frame-by-frame predictions
- **Pass Criteria**:
  - Temporal sampling works
  - Smoothing applied
  - Deterministic output

### 8. Integration Tests

#### Scenario: End-to-End Training Pipeline
- **Command**: `pytest tests/integration/test_training_smoke.py -v`
- **Modules**: Full pipeline
- **Config**: Default + synthetic data
- **Artifacts**: Complete run directory
- **Pass Criteria**:
  - Data loads
  - Training runs
  - Export succeeds
  - Explainability runs (if enabled)

## Offline Test Guarantee

All tests run without network access. Cloud integrations are mocked or gracefully degrade:

```python
# Example: Vertex Experiments disabled in tests
assert cfg.run.log_vertex_experiments == False

# Example: BigQuery logging disabled
assert cfg.run.log_bigquery == False
```

## Running Tests

```bash
# All tests
pytest -v

# Specific category
pytest tests/unit/test_dataset_manifest_schema.py -v

# Parallel execution
pytest -n auto

# With coverage
pytest --cov=src --cov-report=html
```

## Test Fixtures

Fixtures are generated via:
```bash
python scripts/generate_fixtures.py
```

This creates:
- Synthetic images
- Mock manifests
- Sample configs
- Test models

All fixtures are deterministic (seeded) for reproducibility.
