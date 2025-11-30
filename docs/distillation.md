# Distillation (optional)

Where a multi-model ensemble is too costly on device, distil into one student model.

# Distillation

Where a multi-model ensemble is too costly on device, distil into one student model.

## Implementation
The `Distiller` class in `src/core/training/distillation.py` implements knowledge distillation.
It wraps a student model and a frozen teacher model.

### Loss Function
The loss is a weighted combination of:
1. **Student Loss**: Standard task loss (e.g., CrossEntropy) on ground truth.
2. **Distillation Loss**: KL Divergence between softened teacher logits and student logits.

Formula: `L = alpha * student_loss + (1 - alpha) * distillation_loss`

### Configuration
Enable via `configs/training/distillation.yaml` or override in `config.yaml`:
```yaml
training:
  distillation:
    enabled: true
    teacher_model_path: "path/to/saved_model"
    alpha: 0.1
    temperature: 3.0
```

### Usage
The `StandardTrainer` automatically detects if distillation is enabled and wraps the model.
```bash
python -m src.tasks.classification.entrypoint \
  training.distillation.enabled=true \
  training.distillation.teacher_model_path=artifacts/teacher/saved_model
```
