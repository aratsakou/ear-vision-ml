import pytest
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from omegaconf import OmegaConf
from src.core.explainability.explain_segmentation import SegmentationExplainer

@pytest.fixture
def mock_cfg(tmp_path):
    return OmegaConf.create({
        "explainability": {
            "max_samples": 2,
            "segmentation": {
                "uncertainty": {"enabled": True}
            }
        }
    })

@pytest.fixture
def mock_model_binary():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(inputs)
    return tf.keras.Model(inputs, x)

@pytest.fixture
def mock_model_multiclass():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(3, (1, 1), activation="softmax")(inputs)
    return tf.keras.Model(inputs, x)

@pytest.fixture
def mock_dataset():
    images = np.random.rand(5, 32, 32, 3).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(images).batch(1)
    return ds

def test_segmentation_explain_binary(mock_cfg, mock_model_binary, mock_dataset, tmp_path):
    explainer = SegmentationExplainer(mock_cfg, tmp_path, mock_model_binary)
    
    results = explainer.run_explainability({"val": mock_dataset})
    
    assert Path(results["seg_explain_json"]).exists()
    assert Path(results["segmentation_maps_dir"]).exists()
    
    with open(results["seg_explain_json"]) as f:
        data = json.load(f)
        
    assert len(data) == 2
    assert "mean_uncertainty" in data[0]

def test_segmentation_explain_multiclass(mock_cfg, mock_model_multiclass, mock_dataset, tmp_path):
    explainer = SegmentationExplainer(mock_cfg, tmp_path, mock_model_multiclass)
    
    results = explainer.run_explainability({"val": mock_dataset})
    
    assert Path(results["seg_explain_json"]).exists()

def test_entropy_computation(mock_cfg, mock_model_binary, tmp_path):
    explainer = SegmentationExplainer(mock_cfg, tmp_path, mock_model_binary)
    
    # Binary entropy
    # p=0.5 -> max entropy
    pred = tf.ones((1, 1, 1)) * 0.5
    entropy = explainer._compute_entropy(pred)
    assert np.isclose(entropy, 0.693, atol=0.01) # ln(2)
    
    # p=0 or p=1 -> min entropy (0)
    pred = tf.zeros((1, 1, 1))
    entropy = explainer._compute_entropy(pred)
    assert np.isclose(entropy, 0.0, atol=0.01)
