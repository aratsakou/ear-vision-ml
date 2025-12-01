import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

log = logging.getLogger(__name__)

class VideoExplainer:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.output_dir = artifacts_dir / "video_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_video_report(self, video_id: str, frames: List[Dict[str, Any]]):
        """
        Generates a report for a video sequence.
        
        Args:
            video_id: Unique ID of the video.
            frames: List of frame data (timestamp, prediction, confidence, roi_jitter).
        """
        # Sort frames by timestamp/index
        frames.sort(key=lambda x: x.get("frame_index", 0))
        
        # Extract traces
        confidences = [f.get("confidence", 0.0) for f in frames]
        jitter_values = [f.get("roi_jitter", 0.0) for f in frames]
        
        # Smooth confidence (simple moving average)
        window_size = 5
        smoothed_conf = np.convolve(confidences, np.ones(window_size)/window_size, mode='valid')
        
        report = {
            "video_id": video_id,
            "frame_count": len(frames),
            "mean_confidence": float(np.mean(confidences)),
            "max_jitter": float(np.max(jitter_values)) if jitter_values else 0.0,
            "traces": {
                "confidence": confidences,
                "smoothed_confidence": smoothed_conf.tolist(),
                "jitter": jitter_values
            },
            "frames": frames
        }
        
        out_path = self.output_dir / f"video_{video_id}.json"
        out_path.write_text(json.dumps(report, indent=2))
        
        return str(out_path)
