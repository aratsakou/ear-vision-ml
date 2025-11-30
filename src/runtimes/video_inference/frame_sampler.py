from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np


class FrameSampler:
    def __init__(self, video_path: Path, sample_rate_hz: float = 1.0):
        self.video_path = video_path
        self.sample_rate_hz = sample_rate_hz

    def sample(self) -> Generator[tuple[float, np.ndarray], None, None]:
        """Yields (timestamp_sec, frame_rgb)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0 # Fallback

        frame_interval = int(fps / self.sample_rate_hz)
        if frame_interval < 1:
            frame_interval = 1

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                # CV2 is BGR, convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield timestamp, frame_rgb
            
            count += 1
        
        cap.release()
