from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tasks.classification.trainer import run_classification
from src.tasks.segmentation.trainer import run_segmentation


@pytest.fixture
def fixture_manifest_dir() -> Path:
    return Path("tests/fixtures/manifests/local_smoke")

def test_classification_training_smoke(fixture_manifest_dir: Path, tmp_path: Path) -> None:
    """Smoke test for classification training using local fixture."""
    cfg = OmegaConf.create({
        "task": {"name": "classification"},
        "model": {
            "type": "classification",
            "name": "cls_mobilenetv3",
            "input_shape": [224, 224, 3],
            "num_classes": 3,
            "dropout": 0.2
        },
        "data": {
            "dataset": {
                "mode": "manifest",
                "manifest_path": str(fixture_manifest_dir),
                "image_size": [224, 224],
                "num_classes": 3,
                "batch_size": 2
            }
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping": {"enabled": False}
        },
        "preprocess": {
            "pipeline_id": "full_frame_v1",
            "version": "1.0.0",
            "output_size": [224, 224],
            "normalisation": "0_1"
        },
        "export": {
            "export": {
                "tflite": {"enabled": True, "quantize": False}
            }
        },
        "run": {
            "name": "test_cls_run",
            "artifacts_dir": str(tmp_path / "artifacts_cls"),
            "log_vertex_experiments": False,
            "log_bigquery": False,
            "log_sql_dataset_version": False
        }
    })
    
    run_classification(cfg)
    
    run_dir = list((tmp_path / "artifacts_cls" / "runs").iterdir())[0]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "exports" / "cls_mobilenetv3" / "saved_model").exists()

def test_segmentation_training_smoke(fixture_manifest_dir: Path, tmp_path: Path) -> None:
    """Smoke test for segmentation training using local fixture."""
    cfg = OmegaConf.create({
        "task": {"name": "segmentation"},
        "model": {
            "type": "segmentation",
            "name": "seg_unet",
            "input_shape": [224, 224, 3],
            "num_classes": 3,
            "base_filters": 16
        },
        "data": {
            "dataset": {
                "mode": "manifest",
                "manifest_path": str(fixture_manifest_dir),
                "image_size": [224, 224],
                "num_classes": 3,
                "batch_size": 2
            }
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "early_stopping": {"enabled": False}
        },
        "preprocess": {
            "pipeline_id": "full_frame_v1",
            "version": "1.0.0",
            "output_size": [224, 224],
            "normalisation": "0_1"
        },
        "export": {
            "export": {
                "tflite": {"enabled": True, "quantize": False}
            }
        },
        "run": {
            "name": "test_seg_run",
            "artifacts_dir": str(tmp_path / "artifacts_seg"),
            "log_vertex_experiments": False,
            "log_bigquery": False,
            "log_sql_dataset_version": False
        }
    })
    
    run_segmentation(cfg)
    
    run_dir = list((tmp_path / "artifacts_seg" / "runs").iterdir())[0]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "exports" / "seg_unet" / "saved_model").exists()
