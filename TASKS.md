# Task List: ear-vision-ml Implementation

## Phase 0: Repo bootstrap (tests + docs first)
- [x] Create directory structure
- [x] Create env files (`config/env/conda-tf217.yml`)
- [x] Create Dockerfile (`config/docker/tf217/Dockerfile`)
- [x] Configure `pyproject.toml`, `ruff.toml`, `mypy.ini`
- [x] Create `docs/devlog/0001-repo-bootstrap.md` and ADR 0001
- [x] Implement `test_roi_contract`
- [x] Implement `test_dataset_manifest_schema`

## Phase 1: Config system + entrypoints
- [x] Setup Hydra config tree
- [x] Implement task entrypoints
- [x] Document in devlog and ADR

## Phase 2: Dataset manifest + loader (local fixtures)
- [x] Implement manifest schema
- [x] Implement dataset loader
- [x] Create local fixture manifests/parquet
- [x] Smoke test: load dataset

## Phase 3: Preprocess pipelines
- [x] Implement preprocess registry
- [x] Implement `full_frame_v1` pipeline
- [x] Implement `cropper_fallback_v1` pipeline
- [x] Unit tests for pipeline swapping
- [x] Debug overlay artefact generation

## Phase 4: Model factory + trainers
- [x] Implement baseline models (MobileNetV3, U-Net)
- [x] Implement `model_factory.py`
- [x] Implement trainers
- [x] Training smoke test
- [x] Add advanced models (EfficientNetB0, ResNet50V2, ResNet50-UNet)

## Phase 5: Export
- [x] Implement export logic (SavedModel, TFLite)
- [x] Export smoke test
- [x] Advanced quantization (INT8, FP16, dynamic range)
- [x] Automatic benchmarking
- [x] Enhanced equivalence testing (SNR, PSNR, cosine similarity)

## Phase 6: Vertex Experiments logging + scripts
- [x] Implement logging module (Vertex + local)
- [x] Create Vertex submission scripts

## Phase 7: Video inference runtime
- [x] Implement frame sampler
- [x] Implement temporal aggregator
- [x] Integration smoke test

## Phase 8: State-of-the-Art Enhancements
- [x] Advanced loss functions (Focal, Dice, Tversky, IoU)
- [x] Comprehensive metrics (F1, Dice, IoU, AUC)
- [x] Modern callbacks (LR scheduling, warm-up, gradient accumulation)
- [x] Data augmentations (MixUp, CutMix, RandAugment)
- [x] Advanced dataset builder (stratified splits, balancing, validation)

## Phase 9: Image Inference Runtime
- [x] Multi-format model support (SavedModel, TFLite, Keras)
- [x] Test-time augmentation
- [x] Batch processing with progress tracking
- [x] Explainability tools (Grad-CAM, saliency maps)
- [x] Integration tests

## Phase 10: Multi-Layered Logging & Reporting
- [x] Multi-layered logging system (console, file, JSON, performance, experiment)
- [x] Advanced experiment reporting (HTML, Markdown, JSON)
- [x] Setup reports (config, dataset, model)
- [x] Results reports (metrics, artifacts)
- [x] Comprehensive tests
- [x] ADR entries for architectural decisions

## Phase 11: Production Readiness (Gap Analysis)
- [x] Implement Core ML export (`.mlpackage`)
- [x] Implement Cloud Ensemble Runtime
- [x] Verify Labelbox ingestion robustness
- [x] Fix integration test regressions

## Summary
- **Total Phases**: 10 (all completed)
- **Status**: ✅ Production-ready
