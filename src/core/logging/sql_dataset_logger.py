"""SQL dataset version logger (schema locked).

This is a placeholder interface. Implementations vary by your internal SQL service.
MVP requirement: provide a single function signature and a no-op default.

The AI agent implementing the repo must NOT attempt to change the SQL schema.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SqlDatasetLoggerConfig:
    enabled: bool = False
    # Add connection details via env vars or your internal client library.
    # Intentionally not specified here.


def log_dataset_version(cfg: SqlDatasetLoggerConfig, dataset_version: str) -> None:
    if not cfg.enabled:
        return
    # Implement using your internal SQL service client.
    # Must remain schema-compatible (version logging only).
    raise NotImplementedError("Wire this into your existing SQL dataset version service.")
