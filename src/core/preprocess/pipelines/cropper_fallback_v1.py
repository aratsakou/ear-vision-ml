from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.contracts.roi_contract import RoiBBox, full_frame_bbox
from src.core.preprocess.pipelines.full_frame_v1 import _normalise, _resize


def _centre_square_bbox(margin: float) -> RoiBBox:
    """Returns a centre square bbox with margin."""
    m = float(margin)
    m = max(0.0, min(0.49, m))
    return RoiBBox(
        bbox_xyxy_norm=(m, m, 1.0 - m, 1.0 - m),
        confidence=1.0,
        source="fallback"
    )

@dataclass(frozen=True)
class CropperFallbackV1:
    pipeline_id: str
    version: str
    output_size: tuple[int, int]
    normalisation: str
    saved_model_path: str | None
    confidence_threshold: float
    fallback_margin: float

    @classmethod
    def from_cfg(cls, cfg: Any) -> CropperFallbackV1:
        h, w = cfg.preprocess.output_size
        return cls(
            pipeline_id=str(cfg.preprocess.pipeline_id),
            version=str(cfg.preprocess.version),
            output_size=(int(h), int(w)),
            normalisation=str(cfg.preprocess.normalisation),
            saved_model_path=cfg.preprocess.get("cropper", {}).get("saved_model_path"),
            confidence_threshold=float(cfg.preprocess.get("cropper", {}).get("confidence_threshold", 0.5)),
            fallback_margin=float(cfg.preprocess.get("fallback", {}).get("safety_margin", 0.1)),
        )

    def apply(self, image: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        # MVP: Always use fallback for now (mocking cropper logic)
        # In a real implementation, we would load the SavedModel here or in __init__
        
        bbox = _centre_square_bbox(self.fallback_margin)
        
        # Crop and resize
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.as_xyxy()
        
        # Convert to pixel coords
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        # Ensure valid crop
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        
        if px2 <= px1 or py2 <= py1:
             # Fallback to full frame if crop is invalid
             crop = image
             bbox = full_frame_bbox()
        else:
             crop = image[py1:py2, px1:px2]

        resized = _resize(crop, self.output_size)
        out = _normalise(resized, self.normalisation)
        
        md = dict(metadata)
        md["roi_bbox_xyxy_norm"] = list(bbox.as_xyxy())
        md["roi_confidence"] = bbox.confidence
        md["roi_source"] = bbox.source
        
        return out.astype(np.float32), md
