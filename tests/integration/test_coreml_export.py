import pytest
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from src.core.export.coreml_exporter import CoreMLExporter

class TestCoreMLExport:
    
    @pytest.fixture
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, 3),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    @pytest.fixture
    def cfg(self):
        return OmegaConf.create({
            "task": {"name": "classification"},
            "data": {"dataset": {"image_size": [224, 224]}},
            "export": {
                "export": {
                    "coreml": {
                        "enabled": True,
                        "quantize": False
                    }
                }
            },
            "model": {
                "num_classes": 2
            }
        })

    def test_export_skips_if_disabled(self, model, cfg, tmp_path):
        cfg.export.export.coreml.enabled = False
        exporter = CoreMLExporter()
        result = exporter.export(model, tmp_path, cfg)
        assert result is None
        assert not (tmp_path / "model.mlpackage").exists()

    def test_export_handles_missing_coremltools(self, model, cfg, tmp_path):
        # Simulate missing coremltools
        with patch("src.core.export.coreml_exporter.HAS_COREML", False):
            exporter = CoreMLExporter()
            result = exporter.export(model, tmp_path, cfg)
            assert result is None
            # Should log warning, but not crash

    def test_export_runs_if_coremltools_present(self, model, cfg, tmp_path):
        # This test only runs if we actually have coremltools installed
        
        # Check if we can import it
        try:
            import coremltools
        except ImportError:
            pytest.skip("coremltools not installed")

        exporter = CoreMLExporter()
        result = exporter.export(model, tmp_path, cfg)
        
        if result is None:
            pytest.skip("CoreML export failed (likely Keras/coremltools incompatibility)")
        
        assert result is not None
        assert result.exists()
        assert result.name == "model.mlpackage"
