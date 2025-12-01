import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from src.core.models.factories.model_factory import _builder


def test_build_classification_forward():
    cfg = OmegaConf.create({
        "model": {"type": "classification", "name": "cls_mobilenetv3", "input_shape": [64, 64, 3], "num_classes": 3, "dropout": 0.1}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 3)

def test_build_classification_efficientnet():
    cfg = OmegaConf.create({
        "model": {"type": "classification", "name": "cls_efficientnetb0", "input_shape": [64, 64, 3], "num_classes": 3, "dropout": 0.1}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 3)

def test_build_classification_resnet50v2():
    cfg = OmegaConf.create({
        "model": {"type": "classification", "name": "cls_resnet50v2", "input_shape": [64, 64, 3], "num_classes": 3, "dropout": 0.1}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 3)

def test_build_segmentation_forward():
    cfg = OmegaConf.create({
        "model": {"type": "segmentation", "name": "seg_unet", "input_shape": [64, 64, 3], "num_classes": 2, "base_filters": 8}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape[1:] == (64, 64, 2)

def test_build_segmentation_resnet50():
    cfg = OmegaConf.create({
        "model": {"type": "segmentation", "name": "seg_resnet50_unet", "input_shape": [64, 64, 3], "num_classes": 2}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape[1:] == (64, 64, 2)

def test_build_cropper_forward():
    cfg = OmegaConf.create({
        "model": {"type": "cropper", "name": "cropper_mobilenetv3", "input_shape": [64, 64, 3]}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 5)

def test_build_cropper_resnet50v2():
    cfg = OmegaConf.create({
        "model": {"type": "cropper", "name": "cropper_resnet50v2", "input_shape": [64, 64, 3]}
    })
    m = _builder.build(cfg)
    x = tf.convert_to_tensor(np.random.rand(1, 64, 64, 3).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 5)
