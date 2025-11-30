# 0011: Production Readiness Enhancements

## Context
A comprehensive review of the codebase against the PRD identified three key gaps preventing full production readiness:
1.  **Core ML Export**: Required for iOS deployment (specifically for the Cropper task), but was missing.
2.  **Ensemble Runtime**: The cloud ensemble runtime was just a stub.
3.  **Labelbox Ingestion**: Needed verification of robustness.

## Implementation

### 1. Core ML Export
Implemented `_export_coreml` in `src/core/export/exporter.py`.
-   Uses `coremltools` (optional dependency) to convert Keras models to `.mlpackage`.
-   Integrated into `StandardExporter` pipeline.
-   Updated `ExportPaths` to include `coreml_path`.
-   Added graceful degradation if `coremltools` is not installed.

### 2. Cloud Ensemble Runtime
Implemented `CloudEnsembleRuntime` in `src/ensembles/cloud_runtime.py`.
-   Loads multiple Keras models from paths.
-   Implements `soft_vote` for weighted probability averaging.
-   Added `EnsembleMemberSpec` for configuration.
-   Added unit tests in `tests/unit/test_ensembles.py`.

### 3. Labelbox Ingestion Verification
Created `tests/unit/test_labelbox_ingest.py` to verify `src/core/data/labelbox_ingest.py`.
-   Tested simple and wrapped JSON formats.
-   Verified normalization logic.
-   Confirmed the existing MVP implementation is sufficient for current needs.

## Verification
-   **Unit Tests**: Added tests for Ensembles and Labelbox ingestion.
-   **Integration**: Core ML export is integrated into the main export pipeline (though requires `coremltools` to be active).

## Next Steps
-   Install `coremltools` in the environment if iOS export is required immediately.
-   Configure specific ensemble recipes in `configs/ensemble/`.
