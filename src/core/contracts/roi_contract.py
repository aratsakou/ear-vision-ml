from dataclasses import dataclass
from typing import Literal
import math

from src.core.constants import ROI_BBOX_EPSILON


@dataclass(frozen=True)
class RoiBBox:
    """
    Defines the contract for Region of Interest (ROI).
    
    Attributes:
        bbox_xyxy_norm: Bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].
        confidence: Confidence score of the ROI, between 0 and 1.
        source: Source of the ROI ('cropper', 'fallback', 'full_frame').
    """
    bbox_xyxy_norm: tuple[float, float, float, float]
    confidence: float
    source: Literal["cropper", "fallback", "full_frame"]

    def __post_init__(self) -> None:
        """Validates the ROI contract with comprehensive checks."""
        # Validate tuple structure
        if not isinstance(self.bbox_xyxy_norm, tuple) or len(self.bbox_xyxy_norm) != 4:
            raise ValueError(f"bbox must be a tuple of 4 coordinates, got {type(self.bbox_xyxy_norm)} with length {len(self.bbox_xyxy_norm) if isinstance(self.bbox_xyxy_norm, (tuple, list)) else 'N/A'}")
        
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        
        # Check for NaN/Inf coordinates
        for i, coord in enumerate(self.bbox_xyxy_norm):
            if not isinstance(coord, (int, float)):
                raise ValueError(f"Coordinate at index {i} must be numeric, got {type(coord)}")
            if not math.isfinite(coord):
                raise ValueError(f"Coordinates must be finite (no NaN/Inf), got {self.bbox_xyxy_norm}")
        
        # Validate normalized range [0, 1]
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            raise ValueError(f"Coordinates must be normalized in [0,1], got {self.bbox_xyxy_norm}")
            
        # Validate coordinate order
        if x1 > x2 or y1 > y2:
             raise ValueError(f"Invalid coordinate order: x1={x1} > x2={x2} or y1={y1} > y2={y2}")
        
        # Validate non-empty bbox (with small epsilon for floating point comparison)
        width = x2 - x1
        height = y2 - y1
        if width < ROI_BBOX_EPSILON or height < ROI_BBOX_EPSILON:
            raise ValueError(f"Bounding box is empty or degenerate: width={width:.6f}, height={height:.6f}")

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(f"Confidence must be numeric, got {type(self.confidence)}")
        if not math.isfinite(self.confidence):
            raise ValueError(f"Confidence must be finite, got {self.confidence}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Returns the bbox as (x1, y1, x2, y2)."""
        return self.bbox_xyxy_norm

def full_frame_bbox() -> RoiBBox:
    """Returns a RoiBBox representing the full frame."""
    return RoiBBox(
        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
        confidence=1.0,
        source="full_frame"
    )
