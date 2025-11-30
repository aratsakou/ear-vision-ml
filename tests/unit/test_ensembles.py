import numpy as np
import tensorflow as tf
import pytest
from pathlib import Path
from src.ensembles.cloud_runtime import CloudEnsembleRuntime, EnsembleMemberSpec, soft_vote

def test_soft_vote():
    # Model 1: Confident in class 0
    p1 = np.array([[0.9, 0.1], [0.8, 0.2]])
    # Model 2: Confident in class 1
    p2 = np.array([[0.1, 0.9], [0.2, 0.8]])
    
    # Equal weights
    avg = soft_vote([p1, p2], [1.0, 1.0])
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(avg, expected)
    
    # Weighted
    weighted = soft_vote([p1, p2], [3.0, 1.0]) # 0.75 * p1 + 0.25 * p2
    expected_weighted = 0.75 * p1 + 0.25 * p2
    np.testing.assert_allclose(weighted, expected_weighted)

def test_ensemble_runtime(tmp_path):
    # Create dummy models
    model1_path = tmp_path / "model1.keras"
    model2_path = tmp_path / "model2.keras"
    
    # Simple model that outputs constant
    input_layer = tf.keras.layers.Input(shape=(10,))
    # Model 1 outputs [1, 0]
    x1 = tf.keras.layers.Dense(2, kernel_initializer='zeros', bias_initializer=tf.constant_initializer([10.0, -10.0]))(input_layer)
    out1 = tf.keras.layers.Softmax()(x1)
    model1 = tf.keras.Model(input_layer, out1)
    model1.save(model1_path)
    
    # Model 2 outputs [0, 1]
    x2 = tf.keras.layers.Dense(2, kernel_initializer='zeros', bias_initializer=tf.constant_initializer([-10.0, 10.0]))(input_layer)
    out2 = tf.keras.layers.Softmax()(x2)
    model2 = tf.keras.Model(input_layer, out2)
    model2.save(model2_path)
    
    # Create runtime
    members = [
        EnsembleMemberSpec(model_path=str(model1_path), weight=1.0),
        EnsembleMemberSpec(model_path=str(model2_path), weight=1.0)
    ]
    runtime = CloudEnsembleRuntime(members)
    
    # Run prediction
    inputs = np.random.rand(5, 10).astype(np.float32)
    preds = runtime.predict(inputs)
    
    # Expect roughly [0.5, 0.5]
    expected = np.array([[0.5, 0.5]] * 5)
    np.testing.assert_allclose(preds, expected, atol=1e-4)
