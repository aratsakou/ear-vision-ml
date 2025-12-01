import pytest
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf

from src.tasks.segmentation.entrypoint import main as segmentation_main

def test_segmentation_smoke(tmp_path: Path) -> None:
    """
    Smoke test for segmentation task.
    Runs 1 epoch with synthetic data and verifies artifacts.
    """
    # Create config override
    input_shape = (128, 128, 3)
    
    cfg = OmegaConf.create({
        "task": {"name": "segmentation"},
        "model": {
            "name": "seg_unet",
            "input_shape": list(input_shape),
            "num_classes": 2,
            "base_filters": 16
        },
        "data": {
            "dataset": {
                "mode": "synthetic",
                "name": "test_seg_dataset",
                "id": "test_seg_dataset",
                "num_classes": 2,
                "samples": 10,
                "image_size": [128, 128],
                "batch_size": 2
            },
            "loader": {
                "batch_size": 2,
                "shuffle_buffer": 10
            }
        },
        "preprocess": {
            "pipeline_id": "full_frame_v1",
            "version": "1.0"
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "mixed_precision": {"enabled": False},
            "checkpoint": {"enabled": True, "monitor": "val_loss"},
            "tensorboard": {"enabled": False},
            "csv_logger": {"enabled": True},
            "tuning": {"enabled": False}
        },
        "export": {
            "tflite": {"enabled": True, "quantize": False},
            "coreml": {"enabled": False},
            "saved_model": {"enabled": True}
        },
        "evaluation": {
            "enabled": True
        },
        "run": {
            "name": "test_seg_smoke",
            "seed": 42,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "log_vertex_experiments": False,
            "log_bigquery": False,
            "log_sql_dataset_version": False
        },
        "debug": {
            "save_preprocess_overlays": False,
            "overlay_samples": 0
        }
    })
    
    # Run entrypoint
    segmentation_main(cfg)
    
    # Verify artifacts
    artifacts_dir = tmp_path / "artifacts"
    assert artifacts_dir.exists()
    print(f"DEBUG: Artifacts content: {list(artifacts_dir.glob('*'))}")
    assert (artifacts_dir / "run.json").exists()
    assert (artifacts_dir / "metrics.json").exists()
    assert (artifacts_dir / "training_log.csv").exists()
    
    # Verify model export
    assert (artifacts_dir / "saved_model").exists()
    assert (artifacts_dir / "model.tflite").exists()
