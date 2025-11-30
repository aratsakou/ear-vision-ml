from pathlib import Path

import cv2
import numpy as np
import pytest

from src.runtimes.video_inference.offline_runner import run_video_inference


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    path = tmp_path / "test_video.mp4"
    # Create a 1-second video at 10 FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (640, 480))
    
    for _ in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path

def test_video_runtime_smoke(dummy_video: Path, tmp_path: Path) -> None:
    """Smoke test for video inference runtime."""
    output_path = tmp_path / "report.json"
    
    # Mock model function
    def mock_model(x: np.ndarray) -> np.ndarray:
        # Returns random probs for 3 classes
        return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
    
    run_video_inference(
        video_path=dummy_video,
        model_fn=mock_model,
        output_path=output_path,
        sample_rate_hz=5.0
    )
    
    assert output_path.exists()
    import json
    with open(output_path) as f:
        data = json.load(f)
    
    assert "final_label_index" in data
    assert data["final_label_index"] == 1
    assert len(data["frame_predictions"]) > 0
