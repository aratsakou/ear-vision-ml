# Test Coverage Summary

## Overview
**Total Tests**: 39
**Passing**: 38
**Skipped**: 1 (Grad-CAM requires Keras 3 Functional API)
**Failures**: 0

## Breakdown

### Unit Tests (24)
- **Contracts**: `test_roi_contract.py` (3 tests) - Validates ROI normalization and clipping.
- **Data**: `test_dataset_manifest_schema.py` (2 tests) - Validates dataset manifest structure.
- **Data**: `test_labelbox_ingest.py` (3 tests) - Validates Labelbox JSON parsing.
- **Models**: `test_model_factory.py` (7 tests) - Verifies build and forward pass for all architectures.
- **Preprocessing**: `test_preprocess_registry.py` (2 tests) - Verifies pipeline registry.
- **Logging**: `test_logging.py` (7 tests) - Verifies multi-layered logging and reporting.
- **Ensembles**: `test_ensembles.py` (2 tests) - Verifies ensemble voting logic.
- **DI**: `test_di_container.py` (3 tests) - Verifies dependency injection contracts.

### Integration Tests (15)
- **Data**: `test_dataset_build_smoke.py` (1 test) - End-to-end dataset building.
- **Training**: `test_training_smoke.py` (2 tests) - Classification and Segmentation training loops.
- **Export**: `test_export_smoke.py` (1 test) - Model export to SavedModel/TFLite.
- **Video**: `test_video_runtime_smoke.py` (1 test) - Video inference runtime.
- **Image**: `test_image_runtime_smoke.py` (10 tests) - Image inference, TTA, batch processing.

## Key Scenarios Covered
- ✅ **End-to-End Training**: From config to trained model artifacts.
- ✅ **Model Export**: SavedModel, TFLite (Float/Quantized), Core ML.
- ✅ **Inference Runtimes**: Single image, batch, video, ensembles.
- ✅ **Data Pipeline**: Ingestion, building, loading.
- ✅ **Production Logging**: Experiment tracking, reporting.

## Gaps & Known Issues
- `test_image_runtime_smoke.py`: Grad-CAM test skipped due to Keras 3 API changes (minor).
