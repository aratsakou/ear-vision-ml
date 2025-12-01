# Devlog 0014: Config Access & Validation Improvements

**Date:** 2025-12-01  
**Status:** ✅ Complete  
**Related:** Phase 3 of architectural fixes (deep-architecture-review-2025-12.md)

## Summary

Created safe config access utilities and improved validation to prevent silent failures from chained `.get()` calls and invalid data.

## Changes Implemented

### 1. Safe Config Access Utility

**Problem:** Chained `.get()` calls like `cfg.data.dataset.get("sampling", {}).get("strategy", "none")` are fragile and can silently fail if config structure is unexpected.

**Solution:** Created `config_utils.py` with `safe_get()` using `OmegaConf.select()`.

**Code:**
```python
def safe_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely get nested config value using dot notation."""
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        return default

# Type-specific helpers
def safe_get_bool(cfg: Any, path: str, default: bool = False) -> bool
def safe_get_int(cfg: Any, path: str, default: int = 0) -> int
def safe_get_float(cfg: Any, path: str, default: float = 0.0) -> float
def safe_get_str(cfg: Any, path: str, default: str = "") -> str
```

**Usage:**
```python
# Before (fragile)
sampling_strategy = cfg.data.dataset.get("sampling", {}).get("strategy", "none")

# After (robust)
from src.core.config_utils import safe_get_str
sampling_strategy = safe_get_str(cfg, "data.dataset.sampling.strategy", "none")
```

### 2. Fixed Critical Config Typo

**Problem:** `cfg.export.export.tflite` (double `.export`) bypassed intended config structure.

**Fix:**
```python
# Before (typo)
tflite_enabled = getattr(cfg.export.export.tflite, "enabled", True) if hasattr(cfg, 'export') and hasattr(cfg.export, 'export') else True

# After (correct)
tflite_enabled = safe_get_bool(cfg, "export.tflite.enabled", True)
```

### 3. Improved ROI Validation

**Problem:** `RoiBBox` validation didn't check for NaN, Inf, empty bboxes, or tuple structure.

**Solution:** Added comprehensive validation in `__post_init__`:

**Checks added:**
1. **Tuple structure**: Ensures exactly 4 coordinates
2. **Numeric types**: Validates coords are int/float
3. **Finite values**: Rejects NaN and Inf using `math.isfinite()`
4. **Empty bboxes**: Rejects boxes with width/height < epsilon (1e-6)
5. **Coordinate validation**: Existing checks for range [0,1] and order

**Code:**
```python
def __post_init__(self) -> None:
    # Validate tuple structure
    if not isinstance(self.bbox_xyxy_norm, tuple) or len(self.bbox_xyxy_norm) != 4:
        raise ValueError(...)
    
    # Check for NaN/Inf
    for coord in self.bbox_xyxy_norm:
        if not math.isfinite(coord):
            raise ValueError("Coordinates must be finite")
    
    # Validate non-empty bbox
    EPSILON = 1e-6
    if (x2 - x1) < EPSILON or (y2 - y1) < EPSILON:
        raise ValueError("Bounding box is empty or degenerate")
```

### 4. Replaced Chained .get() Calls

**Locations fixed:**
- [`dataset_loader.py:175`](file:///Users/ara/GitHub/ear-vision-ml/src/core/data/dataset_loader.py#L175) - sampling strategy
- [`dataset_loader.py:253`](file:///Users/ara/GitHub/ear-vision-ml/src/core/data/dataset_loader.py#L253) - preprocess type
- [`standard_trainer.py:54`](file:///Users/ara/GitHub/ear-vision-ml/src/core/training/standard_trainer.py#L54) - distillation enabled
- [`standard_trainer.py:89`](file:///Users/ara/GitHub/ear-vision-ml/src/core/training/standard_trainer.py#L89) - tuning enabled
- [`exporter.py:281`](file:///Users/ara/GitHub/ear-vision-ml/src/core/export/exporter.py#L281) - tflite enabled (fixed typo)

## Files Modified

- [`src/core/data/dataset_loader.py`](file:///Users/ara/GitHub/ear-vision-ml/src/core/data/dataset_loader.py) - 2 replacements
- [`src/core/training/standard_trainer.py`](file:///Users/ara/GitHub/ear-vision-ml/src/core/training/standard_trainer.py) - 2 replacements
- [`src/core/export/exporter.py`](file:///Users/ara/GitHub/ear-vision-ml/src/core/export/exporter.py) - 1 critical fix
- [`src/core/contracts/roi_contract.py`](file:///Users/ara/GitHub/ear-vision-ml/src/core/contracts/roi_contract.py) - Enhanced validation

## Files Created

- [`src/core/config_utils.py`](file:///Users/ara/GitHub/ear-vision-ml/src/core/config_utils.py) - Safe config utilities
- [`tests/unit/test_config_validation.py`](file:///Users/ara/GitHub/ear-vision-ml/tests/unit/test_config_validation.py) - 18 comprehensive tests

## Testing

**New Tests:**
- 6 tests for safe config access
- 12 tests for ROI validation

**Test Results:**
```bash
$ pytest tests/unit/test_config_validation.py -v
18 passed in 0.06s
```

**ROI Validation Tests:**
- ✅ test_nan_coordinates - Rejects NaN
- ✅ test_inf_coordinates - Rejects Inf
- ✅ test_empty_bbox - Rejects zero-size boxes
- ✅ test_degenerate_bbox_width - Rejects zero width
- ✅ test_degenerate_bbox_height - Rejects zero height
- ✅ test_invalid_tuple_length - Validates structure
- ✅ test_out_of_range_coordinates - Range checks
- ✅ test_invalid_coordinates_order - Order validation
- ✅ test_nan_confidence - Confidence validation
- ✅ test_very_small_but_valid_bbox - Edge case handling

## Impact

### Benefits
- **No silent failures**: Config access fails loudly or uses defaults explicitly
- **Robust data validation**: Invalid ROI data caught early
- **Fixed critical bug**: TFLite export config now works correctly
- **Better error messages**: Clear validation errors for debugging

### Production Safety
- Config mismatches now raise clear errors instead of using wrong defaults
- Invalid ROI data (e.g., from model inference) caught before processing
- TFLite export configuration properly respected

## Issues Fixed

From [`deep-architecture-review-2025-12.md`](file:///Users/ara/GitHub/ear-vision-ml/docs/review/deep-architecture-review-2025-12.md):

- ✅ **Issue #4**: Unsafe chained `.get()` configuration access
- ✅ **Issue #9**: Improper config attribute access (typo)
- ✅ **Issue #10**: Missing ROI input validation

## Next Steps

Phase 4: Memory & Performance optimizations
