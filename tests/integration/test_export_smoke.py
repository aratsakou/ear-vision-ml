from pathlib import Path

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from src.core.export.exporter import StandardExporter

def test_export_smoke(tmp_path: Path) -> None:
    """Smoke test for model export."""
    # Create a dummy model
    input_shape = (32, 32, 3)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Config
    cfg = OmegaConf.create({
        "task": {"name": "test_task"},
        "model": {
            "name": "test_model",
            "input_shape": list(input_shape)
        },
        "preprocess": {
            "pipeline_id": "test_pipe",
            "version": "1.0"
        },
        "export": {
            "export": {
                "tflite": {
                    "enabled": True,
                    "quantize": True
                }
            }
        }
    })
    
    out_dir = tmp_path / "export_test"
    exporter = StandardExporter()
    paths_dict = exporter.export(model, cfg, out_dir)
    
    # Verify paths
    assert Path(paths_dict["saved_model_dir"]).exists()
    assert Path(paths_dict["tflite_path"]).exists()
    assert Path(paths_dict["tflite_int8_path"]).exists()
    assert Path(paths_dict["manifest_path"]).exists()
    
    # Verify TFLite inference
    interpreter = tf.lite.Interpreter(model_path=str(paths_dict["tflite_path"]))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    assert output_data.shape == (1, 2)
