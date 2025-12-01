# ADR 0005: Explainability Framework

## Status
Accepted

## Context
We need to provide auditable insights into our ML pipelines, covering dataset quality, preprocessing (ROI) reliability, and model decision-making. This is critical for medical applications (otoscopy) to ensure trust and safety.

## Decision
We will implement a modular, config-driven Explainability Framework integrated into the existing repository structure.

### Key Components
1.  **Registry**: Central entrypoint (`src/core/explainability/registry.py`) that dispatches tasks based on config.
2.  **Audits**: Dedicated modules for Dataset and ROI audits.
3.  **Attribution**:
    - **Classification**: Integrated Gradients (IG) as the primary method due to its axiomatic properties (sensitivity, implementation invariance). Grad-CAM supported as a fallback/alternative.
    - **Segmentation**: Entropy-based uncertainty maps.
4.  **Artifacts**: All outputs stored in `artifacts/runs/<run_id>/explainability/` with a strictly defined `explainability_manifest.json`.

### Constraints
- **Offline Tests**: No network calls in tests.
- **Hydra Configs**: All behavior controlled via `configs/explainability/`.
- **Reproducibility**: Deterministic execution seeded by `run.seed`.

## Consequences
- **Positive**: Standardized audit trail for every model trained. Improved debugging and trust.
- **Negative**: Increased training time (mitigated by configurable sample limits). Added dependency on `tensorflow` gradients (already present).

## Alternatives Considered
- **SHAP**: Rejected due to high computational cost for image data.
- **External Platforms (e.g., Fiddler, Arize)**: Rejected to maintain self-contained, offline-capable repository and avoid vendor lock-in for core logic.
