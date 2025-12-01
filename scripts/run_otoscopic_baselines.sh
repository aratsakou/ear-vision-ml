#!/bin/bash
# Run Otoscopic Baseline Experiments
# Trains all three baseline models with explainability enabled

set -e

EXPERIMENT_DIR="artifacts/otoscopic_experiment"
mkdir -p $EXPERIMENT_DIR

echo "========================================"
echo "Otoscopic Baseline Experiments"
echo "========================================"
echo ""

# Model 1: MobileNetV3 (Lightweight)
echo "1. Training MobileNetV3 (Lightweight Baseline)..."
python -m src.tasks.classification.entrypoint \
  --config-name experiment/otoscopic_baseline \
  model=cls_mobilenetv3 \
  model.input_shape=[512,512,3] \
  run.name=otoscopic_mobilenetv3 \
  run.artifacts_dir=$EXPERIMENT_DIR/mobilenetv3

echo ""
echo "✅ MobileNetV3 training complete"
echo ""

# Model 2: EfficientNetB0 (Balanced)
echo "2. Training EfficientNetB0 (Balanced)..."
python -m src.tasks.classification.entrypoint \
  --config-name experiment/otoscopic_baseline \
  model=cls_efficientnetb0 \
  model.input_shape=[512,512,3] \
  run.name=otoscopic_efficientnetb0 \
  run.artifacts_dir=$EXPERIMENT_DIR/efficientnetb0

echo ""
echo "✅ EfficientNetB0 training complete"
echo ""

# Model 3: ResNet50V2 (High Capacity)
echo "3. Training ResNet50V2 (High Capacity)..."
python -m src.tasks.classification.entrypoint \
  --config-name experiment/otoscopic_baseline \
  model=cls_resnet50v2 \
  model.input_shape=[512,512,3] \
  run.name=otoscopic_resnet50v2 \
  run.artifacts_dir=$EXPERIMENT_DIR/resnet50v2

echo ""
echo "✅ ResNet50V2 training complete"
echo ""

echo "========================================"
echo "All Baseline Models Trained!"
echo "========================================"
echo ""
echo "Results:"
echo "  MobileNetV3:    $EXPERIMENT_DIR/mobilenetv3"
echo "  EfficientNetB0: $EXPERIMENT_DIR/efficientnetb0"
echo "  ResNet50V2:     $EXPERIMENT_DIR/resnet50v2"
echo ""
echo "Next: Compare results and select best model for distillation"
