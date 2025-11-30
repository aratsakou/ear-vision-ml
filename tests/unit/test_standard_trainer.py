"""
Unit tests for StandardTrainer.
"""
import pytest
import tensorflow as tf
from omegaconf import OmegaConf
from unittest.mock import Mock, patch

from src.core.training.standard_trainer import StandardTrainer
from src.core.interfaces import Trainer


class TestStandardTrainer:
    """Test the StandardTrainer implementation."""
    
    def test_trainer_implements_interface(self):
        """Test that StandardTrainer implements Trainer interface."""
        trainer = StandardTrainer()
        assert isinstance(trainer, Trainer)
        assert hasattr(trainer, 'train')
        assert callable(trainer.train)
    
    def test_trainer_compiles_model_for_classification(self):
        """Test that trainer compiles model with classification loss/metrics."""
        trainer = StandardTrainer()
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Create synthetic dataset
        x = tf.random.uniform((16, 5))
        y = tf.one_hot(tf.random.uniform((16,), maxval=3, dtype=tf.int32), depth=3)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "training": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1
            },
            "run": {
                "artifacts_dir": "/tmp/test"
            }
        })
        
        with patch('src.core.training.standard_trainer.make_callbacks', return_value=[]):
            result = trainer.train(model, ds, ds, cfg)
        
        # Check that model was compiled
        assert model.optimizer is not None
        assert model.compiled_loss is not None
        
        # Check that training happened
        assert hasattr(result, 'history')
    
    def test_trainer_compiles_model_for_segmentation(self):
        """Test that trainer compiles model with segmentation loss/metrics."""
        trainer = StandardTrainer()
        
        # Create a simple segmentation model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(2, 1, padding='same'),
            tf.keras.layers.Softmax(axis=-1)
        ])
        
        # Create synthetic dataset
        x = tf.random.uniform((8, 32, 32, 3))
        y = tf.one_hot(tf.random.uniform((8, 32, 32), maxval=2, dtype=tf.int32), depth=2)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        
        cfg = OmegaConf.create({
            "task": {"name": "segmentation"},
            "training": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1
            },
            "run": {
                "artifacts_dir": "/tmp/test"
            }
        })
        
        with patch('src.core.training.standard_trainer.make_callbacks', return_value=[]):
            result = trainer.train(model, ds, ds, cfg)
        
        assert model.optimizer is not None
        assert hasattr(result, 'history')
    
    def test_trainer_compiles_model_for_cropper(self):
        """Test that trainer compiles model with cropper loss."""
        trainer = StandardTrainer()
        
        # Create a simple cropper model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='sigmoid')
        ])
        
        # Create synthetic dataset
        x = tf.random.uniform((8, 224, 224, 3))
        y = tf.random.uniform((8, 5))
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        
        cfg = OmegaConf.create({
            "task": {"name": "cropper"},
            "training": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1
            },
            "run": {
                "artifacts_dir": "/tmp/test"
            }
        })
        
        with patch('src.core.training.standard_trainer.make_callbacks', return_value=[]):
            result = trainer.train(model, ds, ds, cfg)
        
        assert model.optimizer is not None
        assert hasattr(result, 'history')
    
    def test_trainer_uses_callbacks(self):
        """Test that trainer uses callbacks from config."""
        trainer = StandardTrainer()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        x = tf.random.uniform((16, 5))
        y = tf.one_hot(tf.random.uniform((16,), maxval=3, dtype=tf.int32), depth=3)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "training": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1
            },
            "run": {
                "artifacts_dir": "/tmp/test"
            }
        })
        
        mock_callback = Mock(spec=tf.keras.callbacks.Callback)
        
        with patch('src.core.training.standard_trainer.make_callbacks', return_value=[mock_callback]):
            result = trainer.train(model, ds, ds, cfg)
        
        # Callback should have been used
        assert mock_callback.on_train_begin.called or mock_callback.set_model.called
    
    def test_trainer_handles_unknown_task_gracefully(self):
        """Test that trainer handles unknown task types with fallback."""
        trainer = StandardTrainer()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(3)
        ])
        
        x = tf.random.uniform((16, 5))
        y = tf.random.uniform((16, 3))
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        
        cfg = OmegaConf.create({
            "task": {"name": "unknown_task_type"},
            "training": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1
            },
            "run": {
                "artifacts_dir": "/tmp/test"
            }
        })
        
        # Should not raise error, should use fallback loss/metrics
        with patch('src.core.training.standard_trainer.make_callbacks', return_value=[]):
            result = trainer.train(model, ds, ds, cfg)
        
        assert model.optimizer is not None
        assert hasattr(result, 'history')
