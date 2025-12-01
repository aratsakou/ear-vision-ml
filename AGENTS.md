# Repository Guidelines

## Project Structure & Module Organization
- **`src/`**: DI-based core (`core/` contracts & training utilities, `tasks/` entrypoints, `runtimes/` inference helpers).
- **`configs/`**: Hydra bundles grouped by `model/`, `training/`, `data/`, `preprocess/`; never hardcode parameters.
- **`tests/`**: Unit and integration suites mirroring `src/`; fixtures live beside tests or are generated via `scripts/generate_fixtures.py`.
- **`docs/` + briefs**: ADRs, devlogs, readiness plans, and cheat sheets referenced during reviews.
- **`scripts/` & `config/env/`**: Submission helpers plus the pinned Conda/TensorFlow 2.17 spec.

## Build, Test, and Development Commands
- `conda env create -f config/env/conda-tf217.yml && conda activate ear-vision-ml`: Reproduce the TF 2.17 toolchain.
- `pip install -e .`: Install the package with editable sources for quick iteration.
- `pytest -n auto`: Run the full suite locally; expect ~71 tests with 1 skip.
- `ruff check .` / `ruff format .`: Lint and format Python uniformly.
- `mypy src/core`: Enforce typing on the DI, contract, and training code paths.
- `python -m src.tasks.classification.entrypoint model=cls_efficientnetb0`: Example Hydra-driven training run; swap configs as needed.

## Coding Style & Naming Conventions
- Follow **PEP 8** with 4-space indents, Google-style docstrings, and strict type hints (fail the PR otherwise).
- Register every new model, loss, or pipeline via the provided registry modules; names use `snake_case` for files and configs, `PascalCase` for classes, and `lower_snake` Hydra keys.
- Never hardcode hyperparameters—extend the matching file in `configs/` and document changes in a devlog/ADR entry.

## Testing Guidelines
- Mirror module paths (e.g., `tests/core/test_training.py`) and use descriptive `test_<behavior>` names.
- Keep tests offline, deterministic, and fixture-driven; regenerate assets with `scripts/generate_fixtures.py` when inputs change.
- Maintain the published 98%+ pass rate before merging; add integration coverage for new task entrypoints or exporters.

## Commit & Pull Request Guidelines
- Use **Conventional Commits** (`feat:`, `fix:`, `docs:`) per `git log`; keep subjects present-tense imperative.
- Every PR links the tracked issue/PRD, references the devlog/ADR entry, includes metrics/screenshots for training changes, and completes the checklist in `CONTRIBUTING.md` (tests, lint, fmt, mypy).
- Reject PRs lacking updated docs, Hydra configs, or ROI contract confirmations when code paths change.

## Security & Configuration Tips
- Keep secrets and dataset paths external; configs reference logical names resolved by CI/CD.
- When adjusting DI bindings (`src/core/di.py`), ensure providers honor `src/core/contracts/` so Vertex builds stay reproducible.
