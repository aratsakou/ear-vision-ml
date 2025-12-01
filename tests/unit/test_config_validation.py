"""
Tests for safe config access utilities and improved ROI validation.
"""
import pytest
import math
from omegaconf import OmegaConf

from src.core.config_utils import (
    safe_get,
    safe_get_bool,
    safe_get_int,
    safe_get_float,
    safe_get_str,
)
from src.core.contracts.roi_contract import RoiBBox


class TestSafeConfigAccess:
    """Tests for safe config access utilities."""
    
    def test_safe_get_nested_value(self):
        """Test retrieving nested config values."""
        cfg = OmegaConf.create({"a": {"b": {"c": 123}}})
        
        assert safe_get(cfg, "a.b.c", 0) == 123
        assert safe_get(cfg, "a.b.missing", 0) == 0
        assert safe_get(cfg, "missing.path", "default") == "default"
    
    def test_safe_get_bool(self):
        """Test boolean config access."""
        cfg = OmegaConf.create({
            "enabled": True,
            "disabled": False,
            "nested": {"feature": True}
        })
        
        assert safe_get_bool(cfg, "enabled") is True
        assert safe_get_bool(cfg, "disabled") is False
        assert safe_get_bool(cfg, "nested.feature") is True
        assert safe_get_bool(cfg, "missing", False) is False
    
    def test_safe_get_int(self):
        """Test integer config access."""
        cfg = OmegaConf.create({"count": 42, "nested": {"value": 100}})
        
        assert safe_get_int(cfg, "count") == 42
        assert safe_get_int(cfg, "nested.value") == 100
        assert safe_get_int(cfg, "missing", 10) == 10
    
    def test_safe_get_float(self):
        """Test float config access."""
        cfg = OmegaConf.create({"rate": 0.5, "nested": {"lr": 0.001}})
        
        assert safe_get_float(cfg, "rate") == 0.5
        assert safe_get_float(cfg, "nested.lr") == 0.001
        assert safe_get_float(cfg, "missing", 1.0) == 1.0
    
    def test_safe_get_str(self):
        """Test string config access."""
        cfg = OmegaConf.create({"name": "test", "nested": {"model": "resnet"}})
        
        assert safe_get_str(cfg, "name") == "test"
        assert safe_get_str(cfg, "nested.model") == "resnet"
        assert safe_get_str(cfg, "missing", "default") == "default"
    
    def test_safe_get_with_none(self):
        """Test handling of None values."""
        cfg = OmegaConf.create({"value": None})
        
        assert safe_get(cfg, "value", "default") is None
        assert safe_get_str(cfg, "value", "default") == "default"


class TestROIValidation:
    """Tests for improved ROI validation."""
    
    def test_valid_roi(self):
        """Test creating valid ROI bbox."""
        roi = RoiBBox(
            bbox_xyxy_norm=(0.1, 0.2, 0.8, 0.9),
            confidence=0.95,
            source="cropper"
        )
        assert roi.as_xyxy() == (0.1, 0.2, 0.8, 0.9)
    
    def test_nan_coordinates(self):
        """Test that NaN coordinates are rejected."""
        with pytest.raises(ValueError, match="finite"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, float('nan'), 0.8, 0.9),
                confidence=0.95,
                source="cropper"
            )
    
    def test_inf_coordinates(self):
        """Test that Inf coordinates are rejected."""
        with pytest.raises(ValueError, match="finite"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.2, float('inf'), 0.9),
                confidence=0.95,
                source="cropper"
            )
    
    def test_empty_bbox(self):
        """Test that empty bounding boxes are rejected."""
        with pytest.raises(ValueError, match="empty or degenerate"):
            RoiBBox(
                bbox_xyxy_norm=(0.5, 0.5, 0.5, 0.5),  # Zero width and height
                confidence=0.95,
                source="cropper"
            )
    
    def test_degenerate_bbox_width(self):
        """Test that bbox with zero width is rejected."""
        with pytest.raises(ValueError, match="empty or degenerate"):
            RoiBBox(
                bbox_xyxy_norm=(0.5, 0.2, 0.5, 0.9),  # Zero width
                confidence=0.95,
                source="cropper"
            )
    
    def test_degenerate_bbox_height(self):
        """Test that bbox with zero height is rejected."""
        with pytest.raises(ValueError, match="empty or degenerate"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.5, 0.8, 0.5),  # Zero height
                confidence=0.95,
                source="cropper"
            )
    
    def test_invalid_tuple_length(self):
        """Test that incorrect tuple length is rejected."""
        with pytest.raises(ValueError, match="tuple of 4 coordinates"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.2, 0.8),  # Only 3 coordinates
                confidence=0.95,
                source="cropper"
            )
    
    def test_out_of_range_coordinates(self):
        """Test that coordinates outside [0,1] are rejected."""
        with pytest.raises(ValueError, match="normalized in"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.2, 1.5, 0.9),
                confidence=0.95,
                source="cropper"
            )
    
    def test_invalid_coordinates_order(self):
        """Test that invalid coordinate order is rejected."""
        with pytest.raises(ValueError, match="Invalid coordinate order"):
            RoiBBox(
                bbox_xyxy_norm=(0.8, 0.2, 0.1, 0.9),  # x1 > x2
                confidence=0.95,
                source="cropper"
            )
    
    def test_nan_confidence(self):
        """Test that NaN confidence is rejected."""
        with pytest.raises(ValueError, match="finite"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.2, 0.8, 0.9),
                confidence=float('nan'),
                source="cropper"
            )
    
    def test_out_of_range_confidence(self):
        """Test that confidence outside [0,1] is rejected."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            RoiBBox(
                bbox_xyxy_norm=(0.1, 0.2, 0.8, 0.9),
                confidence=1.5,
                source="cropper"
            )
    
    def test_very_small_but_valid_bbox(self):
        """Test that very small but valid bbox is accepted."""
        # Slightly larger than epsilon
        roi = RoiBBox(
            bbox_xyxy_norm=(0.5, 0.5, 0.51, 0.51),
            confidence=0.95,
            source="cropper"
        )
        assert roi.as_xyxy() == (0.5, 0.5, 0.51, 0.51)
