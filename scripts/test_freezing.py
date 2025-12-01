import tensorflow as tf
from omegaconf import OmegaConf
from src.core.models.factories.model_factory import build_cls_mobilenetv3

def test_freezing():
    print("Testing model freezing...")
    
    # 1. Test without freezing
    cfg_normal = OmegaConf.create({
        "model": {
            "name": "cls_mobilenetv3",
            "input_shape": [224, 224, 3],
            "num_classes": 10,
            "dropout": 0.2
        },
        "training": {}
    })
    
    model_normal = build_cls_mobilenetv3(cfg_normal)
    trainable_count_normal = len(model_normal.trainable_weights)
    print(f"Normal model trainable weights: {trainable_count_normal}")
    
    # 2. Test with backbone freezing
    cfg_frozen = OmegaConf.create({
        "model": {
            "name": "cls_mobilenetv3",
            "input_shape": [224, 224, 3],
            "num_classes": 10,
            "dropout": 0.2,
            "transfer_learning": {
                "freeze_backbone": True
            }
        },
        "training": {}
    })
    
    model_frozen = build_cls_mobilenetv3(cfg_frozen)
    trainable_count_frozen = len(model_frozen.trainable_weights)
    print(f"Frozen model trainable weights: {trainable_count_frozen}")
    
    if trainable_count_frozen >= trainable_count_normal:
        print("FAIL: Frozen model has same or more trainable weights than normal model.")
        exit(1)
        
    # Check if backbone is actually frozen
    # We expect the backbone layer to be non-trainable
    backbone = None
    for layer in model_frozen.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
            
    if not backbone:
        print("FAIL: Could not find backbone in frozen model.")
        exit(1)
        
    if backbone.trainable:
        print("FAIL: Backbone layer is still marked as trainable.")
        exit(1)
        
    print("PASS: Backbone freezing works.")
    
    # 3. Test unfreezing top N layers
    cfg_unfreeze = OmegaConf.create({
        "model": {
            "name": "cls_mobilenetv3",
            "input_shape": [224, 224, 3],
            "num_classes": 10,
            "dropout": 0.2,
            "transfer_learning": {
                "freeze_backbone": True,
                "unfreeze_top_n_layers": 5
            }
        },
        "training": {}
    })
    
    model_unfreeze = build_cls_mobilenetv3(cfg_unfreeze)
    
    # Find backbone
    backbone = None
    for layer in model_unfreeze.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
            
    # Backbone itself should be trainable=True to allow sub-layers to train
    if not backbone.trainable:
        print("FAIL: Backbone should be trainable=True when unfreezing top N layers.")
        exit(1)
        
    # Check last few layers
    trainable_layers = [l for l in backbone.layers if l.trainable and len(l.trainable_weights) > 0]
    print(f"Backbone has {len(trainable_layers)} trainable layers with weights.")
    
    if len(trainable_layers) == 0:
        print("FAIL: No layers unfrozen in backbone.")
        exit(1)
        
    print("PASS: Partial unfreezing works.")

if __name__ == "__main__":
    test_freezing()
