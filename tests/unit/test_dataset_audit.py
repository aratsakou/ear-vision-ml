import pytest
import pandas as pd
import json
from pathlib import Path
from omegaconf import OmegaConf
from src.core.explainability.dataset_audit import DatasetAuditor

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
            "dataset_audit": {
                "class_distribution": {"enabled": True},
                "leakage_check": {
                    "enabled": True,
                    "match_on": "image_uri"
                }
            }
        }
    })

@pytest.fixture
def mock_manifest(tmp_path):
    data = [
        {"image_uri": "img1.jpg", "label": "cat", "split": "train"},
        {"image_uri": "img2.jpg", "label": "dog", "split": "train"},
        {"image_uri": "img3.jpg", "label": "cat", "split": "val"},
        {"image_uri": "img4.jpg", "label": "dog", "split": "val"},
        {"image_uri": "img1.jpg", "label": "cat", "split": "val"}, # Leakage
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "manifest.json"
    df.to_json(path, orient="records")
    return path

def test_dataset_audit_run(mock_cfg, mock_manifest, tmp_path):
    auditor = DatasetAuditor(mock_cfg, tmp_path)
    
    # Run audit
    results = auditor.run_audit({})
    
    # Check outputs
    assert Path(results["dataset_audit_json"]).exists()
    assert Path(results["dataset_audit_md"]).exists()
    
    # Verify JSON content
    with open(results["dataset_audit_json"]) as f:
        data = json.load(f)
        
    # Check class distribution
    assert "class_distribution" in data
    assert data["class_distribution"]["train"]["cat"] == 1
    assert data["class_distribution"]["train"]["dog"] == 1
    
    # Check leakage
    assert "leakage" in data
    assert "train_vs_val" in data["leakage"]
    assert data["leakage"]["train_vs_val"]["count"] == 1
    assert "img1.jpg" in data["leakage"]["train_vs_val"]["examples"]

def test_dataset_audit_no_manifest(mock_cfg, tmp_path):
    # Point to non-existent manifest
    mock_cfg.data.dataset.manifest_path = "non_existent.json"
    
    auditor = DatasetAuditor(mock_cfg, tmp_path)
    results = auditor.run_audit({})
    
    with open(results["dataset_audit_json"]) as f:
        data = json.load(f)
        
    assert "note" in data
    assert "skipped" in data["note"]
