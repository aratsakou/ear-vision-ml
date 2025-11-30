import tensorflow as tf
import pytest
from src.core.training.distillation import DistillationLoss, Distiller
from omegaconf import OmegaConf

class MockModel(tf.keras.Model):
    def __init__(self, output_dim=3):
        super().__init__()
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)

def test_distillation_loss_computation():
    # Setup
    student_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    dist_loss = DistillationLoss(student_loss_fn, alpha=0.5, temperature=2.0)
    
    y_true = tf.constant([[0.0, 1.0, 0.0]])
    y_pred = tf.constant([[0.0, 2.0, 0.0]]) # Logits
    teacher_logits = tf.constant([[0.0, 2.0, 0.0]]) # Same as student for simplicity
    
    # Compute
    loss = dist_loss.compute_loss(y_true, y_pred, teacher_logits)
    
    # Check
    assert loss is not None
    assert loss.dtype == tf.float32

def test_distiller_train_step():
    # Setup
    cfg = OmegaConf.create({
        "training": {
            "loss": {"name": "categorical_crossentropy"},
            "distillation": {
                "alpha": 0.5,
                "temperature": 2.0
            }
        }
    })
    
    student = MockModel(3)
    teacher = MockModel(3)
    
    distiller = Distiller(student, teacher, cfg)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    )
    
    x = tf.random.normal((2, 10))
    y = tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Train step
    results = distiller.train_step((x, y))
    
    assert "loss" in results
    assert "student_loss" in results
    assert "distillation_loss" in results
    assert "categorical_accuracy" in results

def test_distiller_call_delegates_to_student():
    cfg = OmegaConf.create({
        "training": {
            "loss": {"name": "categorical_crossentropy"},
            "distillation": {
                "alpha": 0.5,
                "temperature": 2.0
            }
        }
    })
    student = MockModel(3)
    teacher = MockModel(3)
    distiller = Distiller(student, teacher, cfg)
    
    x = tf.random.normal((1, 10))
    
    # Call
    output = distiller(x)
    
    # Should match student output (shape)
    assert output.shape == (1, 3)
