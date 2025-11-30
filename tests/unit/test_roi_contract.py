import pytest

from src.core.contracts.roi_contract import RoiBBox, full_frame_bbox


def test_roi_contract_valid() -> None:
    """Test valid ROI contract creation."""
    roi = RoiBBox(
        bbox_xyxy_norm=(0.1, 0.1, 0.9, 0.9),
        confidence=0.95,
        source="cropper"
    )
    assert roi.bbox_xyxy_norm == (0.1, 0.1, 0.9, 0.9)
    assert roi.confidence == 0.95
    assert roi.source == "cropper"

def test_roi_contract_invalid_coordinates_range() -> None:
    """Test ROI contract with coordinates out of range."""
    with pytest.raises(ValueError, match="Coordinates must be normalized"):
        RoiBBox(
            bbox_xyxy_norm=(-0.1, 0.1, 0.9, 0.9),
            confidence=0.95,
            source="cropper"
        )
    
    with pytest.raises(ValueError, match="Coordinates must be normalized"):
        RoiBBox(
            bbox_xyxy_norm=(0.1, 0.1, 1.1, 0.9),
            confidence=0.95,
            source="cropper"
        )

def test_roi_contract_invalid_coordinates_order() -> None:
    """Test ROI contract with invalid coordinate order (x1 > x2 or y1 > y2)."""
    with pytest.raises(ValueError, match="Invalid coordinates order"):
        RoiBBox(
            bbox_xyxy_norm=(0.9, 0.1, 0.1, 0.9),
            confidence=0.95,
            source="cropper"
        )

def test_roi_contract_invalid_confidence() -> None:
    """Test ROI contract with invalid confidence."""
    with pytest.raises(ValueError, match="Confidence must be"):
        RoiBBox(
            bbox_xyxy_norm=(0.1, 0.1, 0.9, 0.9),
            confidence=1.5,
            source="cropper"
        )

def test_roi_contract_full_frame() -> None:
    """Test ROI contract for full frame source."""
    roi = full_frame_bbox()
    assert roi.source == "full_frame"
    assert roi.bbox_xyxy_norm == (0.0, 0.0, 1.0, 1.0)
    assert roi.confidence == 1.0
