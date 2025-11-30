"""Vertex Experiments logging helper.

MVP requirements:
- Must be safe to import and call without crashing when not authenticated.
- Must be a no-op if cfg.run.log_vertex_experiments == false.

This module intentionally avoids hard dependencies on project/region; those are provided via env vars or runtime configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from google.cloud import aiplatform  # type: ignore
except ImportError:
    aiplatform = None  # type: ignore


@dataclass
class VertexExperimentLogger:
    enabled: bool
    experiment_name: str
    run_name: str

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or aiplatform is None:
            return
        try:
            aiplatform.log_params(params)
        except Exception:
            return

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self.enabled or aiplatform is None:
            return
        try:
            # Cast to dict[str, Any] to satisfy mypy invariance
            aiplatform.log_metrics(dict(metrics))
        except Exception:
            return
