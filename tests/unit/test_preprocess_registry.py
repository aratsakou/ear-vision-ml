import numpy as np
from omegaconf import OmegaConf

from src.core.preprocess.registry import get_pipeline


def test_full_frame_pipeline_outputs_bbox() -> None:
    cfg = OmegaConf.create({
        "preprocess": {
            "pipeline_id": "full_frame_v1",
            "version": "1.0.0",
            "output_size": [64, 64],
            "normalisation": "0_1",
        }
    })
    pipe = get_pipeline(cfg)
    img = (np.random.rand(80, 120, 3) * 255).astype(np.uint8)
    out, md = pipe.apply(img, {})
    assert out.shape == (64, 64, 3)
    assert "roi_bbox_xyxy_norm" in md
    assert md["roi_source"] == "full_frame"

def test_cropper_fallback_pipeline() -> None:
    cfg = OmegaConf.create({
        "preprocess": {
            "pipeline_id": "cropper_fallback_v1",
            "version": "1.0.0",
            "output_size": [64, 64],
            "normalisation": "0_1",
            "fallback": {"safety_margin": 0.1}
        }
    })
    pipe = get_pipeline(cfg)
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    out, md = pipe.apply(img, {})
    
    assert out.shape == (64, 64, 3)
    assert md["roi_source"] == "fallback"
    # Margin 0.1 -> 0.1 to 0.9
    assert md["roi_bbox_xyxy_norm"] == [0.1, 0.1, 0.9, 0.9]
