import pytest
import json
from pathlib import Path
from src.core.explainability.prediction_report import PredictionReporter
from src.core.explainability.video_explain import VideoExplainer

def test_prediction_reporter(tmp_path):
    reporter = PredictionReporter(tmp_path)
    
    reporter.log_prediction(
        sample_id="s1",
        inputs={"uri": "img1.jpg"},
        outputs={"label": "cat", "confidence": 0.9},
        flags=["high_confidence"]
    )
    
    assert (tmp_path / "prediction_reports.jsonl").exists()
    
    with open(tmp_path / "prediction_reports.jsonl") as f:
        line = f.readline()
        data = json.loads(line)
        
    assert data["sample_id"] == "s1"
    assert data["outputs"]["label"] == "cat"

def test_video_explainer(tmp_path):
    explainer = VideoExplainer(tmp_path)
    
    frames = [
        {"frame_index": 0, "confidence": 0.8, "roi_jitter": 0.1},
        {"frame_index": 1, "confidence": 0.9, "roi_jitter": 0.0},
        {"frame_index": 2, "confidence": 0.85, "roi_jitter": 0.2},
        {"frame_index": 3, "confidence": 0.9, "roi_jitter": 0.1},
        {"frame_index": 4, "confidence": 0.95, "roi_jitter": 0.0},
    ]
    
    report_path = explainer.generate_video_report("v1", frames)
    
    assert Path(report_path).exists()
    
    with open(report_path) as f:
        data = json.load(f)
        
    assert data["video_id"] == "v1"
    assert len(data["traces"]["smoothed_confidence"]) == 1 # 5 frames, window 5 -> 1 valid point
