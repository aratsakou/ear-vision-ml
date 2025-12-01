from collections.abc import Callable
from typing import Any

import tensorflow as tf

from src.core.interfaces import ModelBuilder

ModelFactoryFn = Callable[[Any], tf.keras.Model]

class RegistryModelBuilder(ModelBuilder):
    def __init__(self):
        self._registry: dict[str, ModelFactoryFn] = {}

    def register(self, name: str, factory: ModelFactoryFn):
        self._registry[name.lower()] = factory

    def build(self, cfg: Any) -> tf.keras.Model:
        name = str(cfg.model.name).lower()
        if name not in self._registry:
             raise ValueError(f"Unsupported model: {name}. Available: {list(self._registry.keys())}")
        return self._registry[name](cfg)

# Global registry instance
_builder = RegistryModelBuilder()

def register_model(name: str):
    def decorator(fn: ModelFactoryFn):
        _builder.register(name, fn)
        return fn
    return decorator



# --- Implementations ---

@register_model("cls_mobilenetv3")
def build_cls_mobilenetv3(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="cls_mobilenetv3")


@register_model("cls_efficientnetb0")
def build_cls_efficientnetb0(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="cls_efficientnetb0")


@register_model("cls_resnet50v2")
def build_cls_resnet50v2(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="cls_resnet50v2")


@register_model("seg_unet")
def build_seg_unet(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    base_filters = int(cfg.model.base_filters)

    inp = tf.keras.Input(shape=input_shape, name="image")

    def conv_block(x: tf.Tensor, f: int) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return x

    c1 = conv_block(inp, base_filters)
    p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, base_filters * 4)

    u2 = tf.keras.layers.UpSampling2D()(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c4 = conv_block(u2, base_filters * 2)
    u1 = tf.keras.layers.UpSampling2D()(c4)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c5 = conv_block(u1, base_filters)

    logits = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="logits")(c5)
    out = tf.keras.layers.Softmax(axis=-1, name="probs")(logits)
    return tf.keras.Model(inp, out, name="seg_unet")


@register_model("seg_resnet50_unet")
def build_seg_resnet50_unet(cfg: Any) -> tf.keras.Model:
    """U-Net with ResNet50 backbone (simplified)."""
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    
    inp = tf.keras.Input(shape=input_shape, name="image")
    
    # Encoder (ResNet50)
    base = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    
    x = base(inp)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation="bilinear")(x) # Rough upsampling back to input size
    
    logits = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="logits")(x)
    # Ensure output matches input spatial dims if upsampling wasn't perfect (e.g. odd input dims)
    logits = tf.keras.layers.Resizing(input_shape[0], input_shape[1])(logits)
    out = tf.keras.layers.Softmax(axis=-1, name="probs")(logits)
    
    return tf.keras.Model(inp, out, name="seg_resnet50_unet")


@register_model("cropper_mobilenetv3")
def build_cropper_mobilenetv3(cfg: Any) -> tf.keras.Model:
    """Cropper model output: [x1, y1, x2, y2, conf] in 0..1."""
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf")(x)
    return tf.keras.Model(inp, out, name="cropper_mobilenetv3")


@register_model("cropper_resnet50v2")
def build_cropper_resnet50v2(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf")(x)
    return tf.keras.Model(inp, out, name="cropper_resnet50v2")
