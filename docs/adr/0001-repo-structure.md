# 1. Repo Structure and Ownership Boundaries

Date: 2025-11-28

## Status
Accepted

## Context
We need a repository structure that supports multiple engineers, reproducible experiments (local + Vertex), and strict separation of concerns between models, data, and preprocessing.

## Decision
We will adopt the structure defined in the PRD, which features:
- `config/`: Environment and Docker configs.
- `configs/`: Hydra configurations (the "public API" for experiments).
- `src/core/`: Shared logic (contracts, data loaders, preprocessing, model factories).
- `src/tasks/`: Task-specific entrypoints and trainers.
- `src/runtimes/`: Inference runtimes (e.g., video).
- `tests/`: Mirrored structure for unit and integration tests.

## Consequences
- **Positive**: Clear separation of concerns. Hydra configs drive experiments.
- **Negative**: Rigid structure requires discipline to maintain.
