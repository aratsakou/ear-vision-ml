from dataclasses import dataclass
from typing import Literal


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
        """Validates the ROI contract."""
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        
        # Validate coordinates are within [0, 1]
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            raise ValueError(f"Coordinates must be normalized [0, 1], got {self.bbox_xyxy_norm}")
            
        # Validate coordinates order
        if x1 > x2 or y1 > y2:
             raise ValueError(f"Invalid coordinates order: x1={x1} > x2={x2} or y1={y1} > y2={y2}")

        # Validate confidence
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be [0, 1], got {self.confidence}")

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
