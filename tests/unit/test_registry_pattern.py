"""
Unit tests for Registry Pattern in Model Factory.
"""
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from src.core.models.factories.model_factory import (
    RegistryModelBuilder,
    register_model,
    build_model,
    _builder
)


class TestRegistryPattern:
    """Test the registry pattern implementation."""
    
    def test_register_model_decorator(self):
        """Test that @register_model decorator registers models."""
        # Create a new builder for testing
        builder = RegistryModelBuilder()
        
        # Register a test model
        @register_model("test_model")
        def build_test_model(cfg):
            return tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(5,))
            ])
        
        # Check that it's registered in the global builder
        assert "test_model" in _builder._registry
    
    def test_registry_builder_raises_on_unknown_model(self):
        """Test that building unknown model raises ValueError."""
        builder = RegistryModelBuilder()
        
        cfg = OmegaConf.create({
            "model": {"name": "nonexistent_model_xyz"}
        })
        
        with pytest.raises(ValueError, match="Unsupported model.*Available"):
            builder.build(cfg)
    
    def test_registry_lists_available_models(self):
        """Test that error message lists available models."""
        builder = RegistryModelBuilder()
        
        cfg = OmegaConf.create({
            "model": {"name": "nonexistent_model"}
        })
        
        try:
            builder.build(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Should list some known models
            assert "cls_mobilenetv3" in error_msg or "Available" in error_msg
    
    def test_all_registered_models_are_buildable(self):
        """Test that all registered models can be built with valid config."""
        # Test configurations for each model type
        test_configs = {
            "cls_mobilenetv3": {
                "model": {
                    "name": "cls_mobilenetv3",
                    "input_shape": [224, 224, 3],
                    "num_classes": 3,
                    "dropout": 0.2
                }
            },
            "cls_efficientnetb0": {
                "model": {
                    "name": "cls_efficientnetb0",
                    "input_shape": [224, 224, 3],
                    "num_classes": 3,
                    "dropout": 0.2
                }
            },
            "cls_resnet50v2": {
                "model": {
                    "name": "cls_resnet50v2",
                    "input_shape": [224, 224, 3],
                    "num_classes": 3,
                    "dropout": 0.2
                }
            },
            "seg_unet": {
                "model": {
                    "name": "seg_unet",
                    "input_shape": [256, 256, 3],
                    "num_classes": 2,
                    "base_filters": 32
                }
            },
            "seg_resnet50_unet": {
                "model": {
                    "name": "seg_resnet50_unet",
                    "input_shape": [256, 256, 3],
                    "num_classes": 2
                }
            },
            "cropper_mobilenetv3": {
                "model": {
                    "name": "cropper_mobilenetv3",
                    "input_shape": [224, 224, 3]
                }
            },
            "cropper_resnet50v2": {
                "model": {
                    "name": "cropper_resnet50v2",
                    "input_shape": [224, 224, 3]
                }
            }
        }
        
        for model_name, config_dict in test_configs.items():
            cfg = OmegaConf.create(config_dict)
            model = build_model(cfg)
            
            assert isinstance(model, tf.keras.Model), f"{model_name} should return a Keras model"
            assert model.name == model_name, f"Model name should match {model_name}"
    
    def test_backward_compatibility_with_functional_api(self):
        """Test that old functional API still works."""
        cfg = OmegaConf.create({
            "model": {
                "name": "cls_mobilenetv3",
                "input_shape": [224, 224, 3],
                "num_classes": 3,
                "dropout": 0.2
            }
        })
        
        # Old functional API should still work
        model = build_model(cfg)
        assert isinstance(model, tf.keras.Model)
        assert model.name == "cls_mobilenetv3"
    
    def test_registry_builder_implements_interface(self):
        """Test that RegistryModelBuilder implements ModelBuilder interface."""
        from src.core.interfaces import ModelBuilder
        
        builder = RegistryModelBuilder()
        assert isinstance(builder, ModelBuilder)
    
    def test_case_insensitive_model_names(self):
        """Test that model names are case-insensitive."""
        cfg_lower = OmegaConf.create({
            "model": {
                "name": "cls_mobilenetv3",
                "input_shape": [224, 224, 3],
                "num_classes": 3,
                "dropout": 0.2
            }
        })
        
        cfg_upper = OmegaConf.create({
            "model": {
                "name": "CLS_MOBILENETV3",
                "input_shape": [224, 224, 3],
                "num_classes": 3,
                "dropout": 0.2
            }
        })
        
        model_lower = build_model(cfg_lower)
        model_upper = build_model(cfg_upper)
        
        # Both should build successfully
        assert model_lower.name == "cls_mobilenetv3"
        assert model_upper.name == "cls_mobilenetv3"
