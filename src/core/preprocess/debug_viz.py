from pathlib import Path

import cv2
import numpy as np


def save_debug_overlay(image: np.ndarray, roi_bbox_xyxy_norm: tuple[float, float, float, float], path: Path) -> None:
    """
    Draws the ROI bounding box on the image and saves it.
    
    Args:
        image: HWC numpy array (uint8 or float). If float [0,1], converted to uint8 [0,255].
        roi_bbox_xyxy_norm: (x1, y1, x2, y2) normalized coordinates.
        path: Path to save the image.
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi_bbox_xyxy_norm
    
    # Convert to pixel coordinates
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    
    # Draw rectangle (Green, thickness 2)
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
