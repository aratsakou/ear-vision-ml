import pytest
import pandas as pd
import json
from pathlib import Path
from omegaconf import OmegaConf
from src.core.explainability.roi_audit import ROIAuditor

@pytest.fixture
def mock_cfg(tmp_path):
    return OmegaConf.create({
        "data": {
            "dataset": {
                "mode": "manifest",
                "manifest_path": str(tmp_path / "manifest.json"),
                "split": "train"
            }
        },
        "explainability": {
            "roi": {
                "validity_check": {
                    "enabled": True,
                    "min_area_ratio": 0.01,
                    "min_confidence": 0.5
                },
                "jitter_check": {
                    "enabled": True,
                    "max_center_variance": 0.1  # Low threshold for testing
                }
            }
        }
    })

@pytest.fixture
def mock_manifest(tmp_path):
    data = [
        # Valid
        {"bbox": [0.1, 0.1, 0.9, 0.9], "roi_confidence": 0.9, "split": "train", "video_id": "v1", "frame_index": 1},
        # Small area
        {"bbox": [0.1, 0.1, 0.15, 0.15], "roi_confidence": 0.9, "split": "train", "video_id": "v1", "frame_index": 2},
        # Low confidence
        {"bbox": [0.1, 0.1, 0.9, 0.9], "roi_confidence": 0.4, "split": "train", "video_id": "v2", "frame_index": 1},
        # Jittery sequence (v3) - Make area clearly valid (>0.01)
        {"bbox": [0.1, 0.1, 0.25, 0.25], "roi_confidence": 0.9, "split": "train", "video_id": "v3", "frame_index": 1},
        {"bbox": [0.8, 0.8, 0.95, 0.95], "roi_confidence": 0.9, "split": "train", "video_id": "v3", "frame_index": 2},
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "manifest.json"
    df.to_json(path, orient="records")
    return path

def test_roi_audit_run(mock_cfg, mock_manifest, tmp_path):
    auditor = ROIAuditor(mock_cfg, tmp_path)
    
    # Run audit
    results = auditor.run_audit({})
    
    # Check outputs
    assert Path(results["roi_audit_json"]).exists()
    assert Path(results["roi_audit_md"]).exists()
    
    # Verify JSON content
    with open(results["roi_audit_json"]) as f:
        data = json.load(f)
        
    assert "train" in data
    res = data["train"]
    
    # 5 samples total
    assert res["total_samples"] == 5
    
    # 1 valid (first one)
    # 2nd is small area -> invalid
    # 3rd is low conf -> invalid
    # 4th is valid ROI but part of jittery sequence
    # 5th is valid ROI but part of jittery sequence
    # So 3 valid ROIs?
    # Let's check logic:
    # 1: area=0.64, conf=0.9 -> Valid
    # 2: area=0.0025 < 0.01 -> Invalid
    # 3: area=0.64, conf=0.4 < 0.5 -> Invalid
    # 4: area=0.01, conf=0.9 -> Valid (barely)
    # 5: area=0.01, conf=0.9 -> Valid
    
    assert res["valid_roi_count"] == 3
    assert res["invalid_roi_count"] == 2
    
    # Jitter check
    # v1: center [0.5, 0.5] vs [0.125, 0.125]. Variance > 0.1?
    # v3: center [0.15, 0.15] vs [0.85, 0.85]. Variance definitely > 0.1
    assert res["jitter_issues_count"] >= 1
    
    # Check issues list
    issues = str(res["issues"])
    assert "Small area" in issues
    assert "Low confidence" in issues
    assert "High jitter" in issues

def test_roi_audit_no_bbox(mock_cfg, tmp_path):
    # Create manifest without bbox
    df = pd.DataFrame([{"split": "train", "label": "cat"}])
    path = tmp_path / "manifest.json"
    df.to_json(path, orient="records")
    
    auditor = ROIAuditor(mock_cfg, tmp_path)
    results = auditor.run_audit({})
    
    with open(results["roi_audit_json"]) as f:
        data = json.load(f)
        
    assert "note" in data["train"]
    assert "No 'bbox' column" in data["train"]["note"]
