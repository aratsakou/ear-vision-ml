# 2. Configuration Strategy

Date: 2025-11-28

## Status
Accepted

## Context
We need a flexible configuration system that supports:
- Local and Vertex experiments.
- Swappable components (models, datasets, preprocessing).
- Reproducibility (logging the exact config used).

## Decision
We will use **Hydra** (`hydra-core`) for configuration management.
- Configs live in `configs/`.
- `config.yaml` is the root config.
- Task-specific configs are in `configs/task/`.
- Entrypoints use `@hydra.main` to load and resolve configs.

## Consequences
- **Positive**: Powerful composition, command-line overrides, and automatic logging.
- **Negative**: Adds a dependency on Hydra.
