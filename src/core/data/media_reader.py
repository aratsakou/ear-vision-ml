"""Media reader for local files and (optionally) GCS URIs.

MVP:
- Support local image files (path or file://).
- Provide a clean extension point for gs:// URIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # HWC uint8
    uri: str
    timestamp_ms: int | None = None


def _strip_file_prefix(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def read_image(uri: str) -> np.ndarray:
    path = Path(_strip_file_prefix(uri))
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {uri}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


def read_frame(uri: str, timestamp_ms: int | None = None) -> Frame:
    # Timestamp is ignored for images; for video frames, a future implementation may decode from uri+timestamp.
    return Frame(image=read_image(uri), uri=uri, timestamp_ms=timestamp_ms)
