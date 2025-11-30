# Distillation (optional)

Where a multi-model ensemble is too costly on device, distil into one student model.

MVP: documentation only. Implementation requires:
- teacher ensemble inference on training data
- student training with KL divergence to teacher logits + ground-truth loss
- latency and accuracy gates for iPhone XR / iPhone 13
