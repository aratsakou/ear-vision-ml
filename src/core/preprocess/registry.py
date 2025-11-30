from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class PreprocessPipeline(Protocol):
    pipeline_id: str
    version: str

    def apply(self, image: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]: ...


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    version: str
    output_size: tuple[int, int]
    normalisation: str


def get_pipeline(cfg: Any) -> PreprocessPipeline:
    pipeline_id = str(cfg.preprocess.pipeline_id)
    if pipeline_id == "full_frame_v1":
        from src.core.preprocess.pipelines.full_frame_v1 import FullFrameV1
        return FullFrameV1.from_cfg(cfg)
    if pipeline_id == "cropper_model_v1":
        from src.core.preprocess.pipelines.cropper_model_v1 import CropperModelV1
        return CropperModelV1.from_cfg(cfg)
    if pipeline_id == "cropper_fallback_v1":
        from src.core.preprocess.pipelines.cropper_fallback_v1 import CropperFallbackV1
        return CropperFallbackV1.from_cfg(cfg)
    raise ValueError(f"Unknown preprocess pipeline_id: {pipeline_id}")
