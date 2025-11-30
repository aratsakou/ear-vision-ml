# Model Distillation Guide

## Overview
Knowledge distillation allows training smaller, faster models (students) from larger, more accurate models (teachers) for on-device deployment.

## When to Use Distillation

- Teacher model is too large for device constraints
- Need faster inference without significant accuracy loss
- Have unlabeled data for soft-label generation

## Distillation Process

### 1. Train Teacher Model
```bash
python -m src.tasks.classification.entrypoint \
  model=cls_resnet50v2 \
  training.epochs=50
```

### 2. Generate Soft Labels
```python
# Use teacher to predict on unlabeled/augmented data
teacher = tf.keras.models.load_model("teacher/saved_model")
soft_labels = teacher.predict(unlabeled_data)
```

### 3. Train Student Model
```python
# Student learns from both hard labels and soft labels
# Loss = alpha * hard_loss + (1-alpha) * distillation_loss
# where distillation_loss = KL(student_logits, teacher_logits)
```

### 4. Compare Performance
- Evaluate both models on test set
- Measure inference latency on target device
- Validate accuracy trade-off is acceptable

## Implementation Notes

Distillation is **not required for MVP** but is documented as a future enhancement option.

To implement:
1. Add `distillation_trainer.py` in `src/core/training/`
2. Create config `training/distillation.yaml`
3. Add temperature parameter for softmax smoothing
4. Implement KL divergence loss

## References
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- TensorFlow Model Optimization Toolkit
