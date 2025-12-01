# Models & Training

## Available Models

### Classification / Cropper
| Model Name | Config Name | Description |
|------------|-------------|-------------|
| **MobileNetV3** | `cls_mobilenetv3` | Lightweight, fast, suitable for edge devices. |
| **EfficientNetB0** | `cls_efficientnetb0` | Balanced performance and efficiency. |
| **ResNet50V2** | `cls_resnet50v2` | High accuracy, larger footprint. |

### Segmentation
| Model Name | Config Name | Description |
|------------|-------------|-------------|
| **U-Net** | `seg_unet` | Standard medical image segmentation architecture. |
| **ResNet50-UNet** | `seg_resnet50_unet` | U-Net with ResNet50 backbone for better feature extraction. |

## Training Configuration

Training is configured via `configs/training/`. Key parameters:

- `epochs`: Number of training epochs.
- `batch_size`: Batch size.
- `learning_rate`: Initial learning rate.
- `optimizer`: Optimizer name (adam, sgd, etc.).
- `loss`: Loss function (categorical_crossentropy, focal, dice, etc.).

### Advanced Features

- **Mixed Precision**: Enable via `training.mixed_precision.enabled=true`.
- **LR Scheduling**: Configure via `training.lr_schedule`.
- **Early Stopping**: Configure via `training.early_stopping`.

### Advanced Training Features

#### Loss Functions
You can specify advanced loss functions in `configs/training/default.yaml` or your experiment config:
```yaml
training:
  loss:
    type: "focal" # Options: focal, dice, dice_ce, tversky, label_smoothing, iou
    alpha: 0.25
    gamma: 2.0
```

#### Regularization
L1 and L2 regularization can be applied to all model layers:
```yaml
training:
  regularizer:
    enabled: true
    l1: 0.0
    l2: 0.0001
```

#### Warm-up
Gradually increase learning rate at the start of training:
```yaml
training:
  warmup:
    enabled: true
    epochs: 5
```

#### TensorBoard
TensorBoard logging is enabled by default. It supports Cloud TensorBoard (Vertex AI) automatically.
```yaml
training:
  tensorboard:
    enabled: true
    histogram_freq: 1
    write_graph: true
    write_images: false
    update_freq: "epoch"
```

