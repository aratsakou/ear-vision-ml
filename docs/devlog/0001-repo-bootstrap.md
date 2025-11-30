# 0001-repo-bootstrap

**Date:** 2025-11-28
**Author:** AI Agent

## What was implemented
- Initialized repository structure.
- Created environment configuration:
    - `config/env/conda-tf217.yml`
    - `config/docker/tf217/Dockerfile`
    - `requirements.in` -> `requirements.txt`
- Configured quality gates:
    - `ruff.toml`
    - `mypy.ini`
    - `pyproject.toml` (verified)
- Created initial documentation structure (`docs/devlog`, `docs/adr`).

## Files created/modified
- `config/env/conda-tf217.yml`
- `config/docker/tf217/Dockerfile`
- `requirements.in`
- `requirements.txt`
- `ruff.toml`
- `mypy.ini`
- `docs/devlog/0001-repo-bootstrap.md`
- `docs/adr/0001-repo-structure.md`

## How to run it
1. Create conda environment:
   ```bash
   conda env create -f config/env/conda-tf217.yml
   conda activate ear-vision-ml
   ```
2. Build Docker image:
   ```bash
   cd config/docker/tf217
   docker build -t ear-vision-ml:latest .
   ```

## Tests added/updated
- `tests/unit/test_roi_contract.py` (pending implementation)
- `tests/unit/test_dataset_manifest_schema.py` (pending implementation)

## Known limitations and next steps
- Tests are currently placeholders/pending.
- Next step: Implement the tests and the code they test (ROI contract, manifest schema).
