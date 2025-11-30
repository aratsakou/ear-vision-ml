from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.contracts.roi_contract import full_frame_bbox


def _resize(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    t = tf.convert_to_tensor(image, dtype=tf.float32)
    t = tf.image.resize(t, size_hw, method="bilinear")
    return t.numpy()


def _normalise(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "0_1":
        return image / 255.0
    raise ValueError(f"Unsupported normalisation mode: {mode}")


@dataclass(frozen=True)
class FullFrameV1:
    pipeline_id: str
    version: str
    output_size: tuple[int, int]
    normalisation: str

    @classmethod
    def from_cfg(cls, cfg: Any) -> FullFrameV1:
        h, w = cfg.preprocess.output_size
        return cls(
            pipeline_id=str(cfg.preprocess.pipeline_id),
            version=str(cfg.preprocess.version),
            output_size=(int(h), int(w)),
            normalisation=str(cfg.preprocess.normalisation),
        )

    def apply(self, image: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        resized = _resize(image, self.output_size)
        out = _normalise(resized, self.normalisation)
        md = dict(metadata)
        bbox = full_frame_bbox().as_xyxy()
        md["roi_bbox_xyxy_norm"] = [float(x) for x in bbox]
        md["roi_confidence"] = 1.0
        md["roi_source"] = "full_frame"
        return out.astype(np.float32), md
