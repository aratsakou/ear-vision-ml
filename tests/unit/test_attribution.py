import pytest
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from omegaconf import OmegaConf
from src.core.explainability.attribution_classification import ClassificationAttributor

@pytest.fixture
def mock_cfg(tmp_path):
    return OmegaConf.create({
        "explainability": {
            "max_samples": 2,
            "classification": {
                "method": "integrated_gradients",
                "integrated_gradients": {"steps": 10}
            }
        }
    })

@pytest.fixture
def mock_model():
    # Simple model: linear transformation
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

@pytest.fixture
def mock_dataset():
    # Create a dummy dataset
    images = np.random.rand(5, 32, 32, 3).astype(np.float32)
    labels = np.random.randint(0, 3, size=(5,))
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)
    return ds

def test_attribution_run(mock_cfg, mock_model, mock_dataset, tmp_path):
    attributor = ClassificationAttributor(mock_cfg, tmp_path, mock_model)
    
    # Run attribution
    results = attributor.run_attribution({"val": mock_dataset})
    
    # Check outputs
    assert Path(results["attribution_summary_json"]).exists()
    assert Path(results["overlays_dir"]).exists()
    
    # Verify JSON content
    with open(results["attribution_summary_json"]) as f:
        data = json.load(f)
        
    assert len(data) == 2  # max_samples
    assert "heatmap_path" in data[0]
    
    # Check if image files exist
    assert Path(data[0]["heatmap_path"]).exists()

def test_integrated_gradients_shape(mock_cfg, mock_model, tmp_path):
    attributor = ClassificationAttributor(mock_cfg, tmp_path, mock_model)
    
    image = tf.random.uniform((1, 32, 32, 3))
    target_class = 1
    
    heatmap = attributor._integrated_gradients(image, target_class, steps=5)
    
    assert heatmap.shape == (32, 32)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
