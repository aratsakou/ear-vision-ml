from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.contracts.roi_contract import RoiBBox, clip_bbox, validate_bbox
from src.core.preprocess.pipelines.full_frame_v1 import _normalise, _resize


@dataclass(frozen=True)
class CropperModelV1:
    pipeline_id: str
    version: str
    output_size: tuple[int, int]
    normalisation: str
    saved_model_path: str | None
    confidence_threshold: float

    _model: tf.keras.Model | None = None  # loaded lazily

    @classmethod
    def from_cfg(cls, cfg: Any) -> CropperModelV1:
        h, w = cfg.preprocess.output_size
        return cls(
            pipeline_id=str(cfg.preprocess.pipeline_id),
            version=str(cfg.preprocess.version),
            output_size=(int(h), int(w)),
            normalisation=str(cfg.preprocess.normalisation),
            saved_model_path=cfg.preprocess.cropper.saved_model_path,
            confidence_threshold=float(cfg.preprocess.cropper.confidence_threshold),
            _model=None,
        )

    def _load_model(self) -> tf.keras.Model:
        if not self.saved_model_path:
            raise ValueError("cropper.saved_model_path is required for cropper_model_v1")
        model = tf.keras.models.load_model(self.saved_model_path)
        return model

    def apply(self, image: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        model = self._model or self._load_model()
        # Cropper model expects same input size; resize to its input.
        resized = _resize(image, self.output_size)
        inp = _normalise(resized, self.normalisation)[None, ...]  # [1,H,W,3]
        pred = model.predict(inp, verbose=0)
        pred = np.asarray(pred).reshape(-1)
        if pred.shape[0] < 5:
            raise ValueError("Cropper model must output at least 5 values: x1,y1,x2,y2,conf")
        bbox = RoiBBox(float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]), float(pred[4]), "cropper")
        bbox = clip_bbox(bbox)
        ok, _ = validate_bbox(bbox)
        if (not ok) or bbox.confidence < self.confidence_threshold:
            raise ValueError("Cropper bbox invalid or below confidence_threshold")

        # Apply the crop to the *original* image for maximum fidelity.
        h0, w0 = image.shape[0], image.shape[1]
        x1 = int(bbox.x1 * w0)
        x2 = int(bbox.x2 * w0)
        y1 = int(bbox.y1 * h0)
        y2 = int(bbox.y2 * h0)
        crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        crop_resized = _resize(crop, self.output_size)
        out = _normalise(crop_resized, self.normalisation).astype(np.float32)

        md = dict(metadata)
        md["roi_bbox_xyxy_norm"] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
        md["roi_confidence"] = bbox.confidence
        md["roi_source"] = bbox.source
        return out, md
