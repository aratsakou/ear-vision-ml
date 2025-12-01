"""
Safe configuration access utilities to prevent silent failures from chained .get() calls.
"""
from typing import Any, Optional
from omegaconf import OmegaConf


def safe_get(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Safely get nested config value using dot notation.
    
    Args:
        cfg: OmegaConf config object
        path: Dot-separated path (e.g., "data.dataset.sampling.strategy")
        default: Default value if path not found
    
    Returns:
        Config value at path or default if not found
    
    Example:
        >>> cfg = OmegaConf.create({"a": {"b": {"c": 123}}})
        >>> safe_get(cfg, "a.b.c", 0)
        123
        >>> safe_get(cfg, "a.b.missing", 0)
        0
    """
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        return default


def safe_get_bool(cfg: Any, path: str, default: bool = False) -> bool:
    """
    Safely get boolean config value.
    
    Args:
        cfg: OmegaConf config object
        path: Dot-separated path
        default: Default boolean value
    
    Returns:
        Boolean value at path or default
    """
    value = safe_get(cfg, path, default)
    return bool(value)


def safe_get_int(cfg: Any, path: str, default: int = 0) -> int:
    """
    Safely get integer config value.
    
    Args:
        cfg: OmegaConf config object
        path: Dot-separated path
        default: Default integer value
    
    Returns:
        Integer value at path or default
    """
    value = safe_get(cfg, path, default)
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_get_float(cfg: Any, path: str, default: float = 0.0) -> float:
    """
    Safely get float config value.
    
    Args:
        cfg: OmegaConf config object
        path: Dot-separated path
        default: Default float value
    
    Returns:
        Float value at path or default
    """
    value = safe_get(cfg, path, default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_get_str(cfg: Any, path: str, default: str = "") -> str:
    """
    Safely get string config value.
    
    Args:
        cfg: OmegaConf config object
        path: Dot-separated path
        default: Default string value
    
    Returns:
        String value at path or default
    """
    value = safe_get(cfg, path, default)
    return str(value) if value is not None else default
