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
