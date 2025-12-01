from collections.abc import Callable
from typing import Any
import threading

import tensorflow as tf

from src.core.interfaces import ModelBuilder

ModelFactoryFn = Callable[[Any], tf.keras.Model]

class RegistryModelBuilder(ModelBuilder):
    """Thread-safe model registry builder."""
    
    def __init__(self):
        self._registry: dict[str, ModelFactoryFn] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def register(self, name: str, factory: ModelFactoryFn):
        """Register a model factory (thread-safe)."""
        with self._lock:
            self._registry[name.lower()] = factory

    def build(self, cfg: Any) -> tf.keras.Model:
        """Build a model from config (thread-safe)."""
        name = str(cfg.model.name).lower()
        
        # Get snapshot of registry to avoid holding lock during build
        with self._lock:
            if name not in self._registry:
                available = list(self._registry.keys())
                raise ValueError(f"Unsupported model: {name}. Available: {available}")
            factory = self._registry[name]
        
        # Build model without holding lock (may take time)
        return factory(cfg)

# Global registry instance
_builder = RegistryModelBuilder()

def register_model(name: str):
    """Decorator to register a model factory function."""
    def decorator(fn: ModelFactoryFn):
        _builder.register(name, fn)
        return fn
    return decorator

# Backward compatibility
build_model = _builder.build

# Expose the class for DI
__all__ = ["RegistryModelBuilder", "register_model", "build_model"]



# --- Helper for regularization ---
def _get_regularizer(cfg: Any) -> tf.keras.regularizers.Regularizer | None:
    reg_cfg = getattr(cfg.training, "regularizer", None)
    if not reg_cfg or not getattr(reg_cfg, "enabled", False):
        return None
    
    l1 = float(getattr(reg_cfg, "l1", 0.0))
    l2 = float(getattr(reg_cfg, "l2", 0.0))
    
    if l1 > 0 and l2 > 0:
        return tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    elif l1 > 0:
        return tf.keras.regularizers.L1(l1=l1)
    elif l2 > 0:
        return tf.keras.regularizers.L2(l2=l2)
    return None


# --- Helper for transfer learning ---
def _freeze_layers(model: tf.keras.Model, cfg: Any) -> None:
    """
    Freeze layers for transfer learning based on config.
    
    Config schema:
    model:
      transfer_learning:
        freeze_backbone: bool (default False)
        unfreeze_top_n_layers: int (default 0)
    """
    tl_cfg = getattr(cfg.model, "transfer_learning", None)
    if not tl_cfg or not getattr(tl_cfg, "freeze_backbone", False):
        return

    unfreeze_n = int(getattr(tl_cfg, "unfreeze_top_n_layers", 0))
    
    # We assume the model is constructed as Backbone + Head.
    # If the model is a functional model where the backbone is a layer (e.g. MobileNetV3Small layer),
    # we might need to dig into it.
    # However, in our factory functions below, we often use `base = Application(...)` 
    # and then `x = base(inp)`. In Keras 3 / recent TF, `base` is a Model/Layer.
    
    # Strategy:
    # 1. Identify the backbone. In our builders, we don't return the backbone separately.
    #    But we can iterate over layers.
    # 2. If the model is a functional API model, `model.layers` includes the backbone layer 
    #    if it was added as a single layer (common in Keras applications if used as a layer).
    #    OR it includes all layers if `include_top=False` was used and we built on top.
    
    # Let's look at how we build them.
    # e.g. base = MobileNetV3Small(...); x = base(inp) -> base is a layer in the new model.
    
    # We will try to find the "backbone" layer. It's usually the one with the most parameters 
    # or specifically named if we named it. We didn't name it in the builders explicitly 
    # but Keras assigns names like "mobilenetv3small".
    
    # A safer generic approach for this repo's style (base = App(...); x = base(inp)):
    # The backbone is likely the second layer (after Input).
    
    # Let's try to find a layer that looks like a backbone (Functional model).
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
            
    if backbone:
        # Freeze the backbone
        backbone.trainable = False
        
        # If we need to unfreeze top N layers OF THE BACKBONE
        if unfreeze_n > 0:
            # We need to set trainable=True for the last N layers of the backbone
            # Note: Setting backbone.trainable = False sets it for all its layers recursively usually.
            # To unfreeze specific layers, we might need to set backbone.trainable = True 
            # but set all non-target layers to False.
            
            backbone.trainable = True
            for layer in backbone.layers[:-unfreeze_n]:
                layer.trainable = False
    else:
        # Fallback: if we can't find a backbone sub-model, maybe the model IS the backbone + head flattened.
        # This happens if we did `x = Conv2D(...)(inp)` manually.
        # In that case, we just freeze the first K layers? Hard to guess.
        # For now, we only support the "Backbone as a Layer" pattern used in this factory.
        pass



# --- Implementations ---

@register_model("cls_mobilenetv3")
def build_cls_mobilenetv3(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs", kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inp, out, name="cls_mobilenetv3")
    _freeze_layers(model, cfg)
    return model


@register_model("cls_efficientnetb0")
def build_cls_efficientnetb0(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs", kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inp, out, name="cls_efficientnetb0")
    _freeze_layers(model, cfg)
    return model


@register_model("cls_resnet50v2")
def build_cls_resnet50v2(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    dropout = float(cfg.model.dropout)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")
    base = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs", kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inp, out, name="cls_resnet50v2")
    _freeze_layers(model, cfg)
    return model


@register_model("seg_unet")
def build_seg_unet(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    base_filters = int(cfg.model.base_filters)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")

    def conv_block(x: tf.Tensor, f: int) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
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

    logits = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="logits", kernel_regularizer=regularizer)(c5)
    out = tf.keras.layers.Softmax(axis=-1, name="probs")(logits)
    return tf.keras.Model(inp, out, name="seg_unet")


@register_model("seg_resnet50_unet")
def build_seg_resnet50_unet(cfg: Any) -> tf.keras.Model:
    """U-Net with ResNet50 backbone (simplified)."""
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    num_classes = int(cfg.model.num_classes)
    regularizer = _get_regularizer(cfg)
    
    inp = tf.keras.Input(shape=input_shape, name="image")
    
    # Encoder (ResNet50)
    base = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    
    x = base(inp)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation="bilinear")(x) # Rough upsampling back to input size
    
    logits = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="logits", kernel_regularizer=regularizer)(x)
    # Ensure output matches input spatial dims if upsampling wasn't perfect (e.g. odd input dims)
    logits = tf.keras.layers.Resizing(input_shape[0], input_shape[1])(logits)
    out = tf.keras.layers.Softmax(axis=-1, name="probs")(logits)
    
    out = tf.keras.layers.Softmax(axis=-1, name="probs")(logits)
    
    model = tf.keras.Model(inp, out, name="seg_resnet50_unet")
    _freeze_layers(model, cfg)
    return model


@register_model("cropper_mobilenetv3")
def build_cropper_mobilenetv3(cfg: Any) -> tf.keras.Model:
    """Cropper model output: [x1, y1, x2, y2, conf] in 0..1."""
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf", kernel_regularizer=regularizer)(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf", kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inp, out, name="cropper_mobilenetv3")
    _freeze_layers(model, cfg)
    return model


@register_model("cropper_resnet50v2")
def build_cropper_resnet50v2(cfg: Any) -> tf.keras.Model:
    input_shape = tuple(int(x) for x in cfg.model.input_shape)
    regularizer = _get_regularizer(cfg)

    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf", kernel_regularizer=regularizer)(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="bbox_conf", kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inp, out, name="cropper_resnet50v2")
    _freeze_layers(model, cfg)
    return model
