# Repository rules

## Non-negotiables
- No training code may hardcode dataset paths. Use the data config (manifest) and contracts.
- Preprocessing pipelines are swappable via config and must adhere to ROI contract.
- Datasets are immutable once marked active. Changes create new versions.
- Core modules (`src/core/*` and contracts) require review and must maintain backward compatibility.

## Contributing
- Add new models via `src/core/models/factories/model_factory.py` + a new `configs/model/*.yaml`.
- Add new preprocessing pipelines under `src/core/preprocess/pipelines/` and register in `registry.py`.
- Add/modify contracts only with accompanying tests and an ADR entry.

## Documentation-driven development
Every implementation step must update:
- a devlog entry under `docs/devlog/`
- and, where applicable, an ADR entry under `docs/adr/`
