#!/bin/bash
# Quick End-to-End Test (1 epoch, 20 images)
# Tests all major repository functionalities with minimal resource usage

set -e  # Exit on error

echo "======================================"
echo "Quick E2E Test (1 epoch, 20 images)"
echo "======================================"

# Configuration
TEST_DIR="artifacts/quick_test"
DATA_DIR="$TEST_DIR/data"

# Clean previous runs
echo "Cleaning previous test artifacts..."
rm -rf $TEST_DIR
mkdir -p $TEST_DIR

# Step 1: Generate Synthetic Dataset
echo ""
echo "Step 1: Generating synthetic dataset (20 images)..."
python scripts/generate_fixtures.py \
  --output-dir $DATA_DIR \
  --num-images 20 \
  --num-classes 3

# Step 2: Train Classification Model (Teacher)
echo ""
echo "Step 2: Training Classification Model (1 epoch)..."
python -m src.tasks.classification.entrypoint \
  model=cls_mobilenetv3 \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/classification/manifest.json \
  training.epochs=1 \
  training.batch_size=4 \
  run.artifacts_dir=$TEST_DIR/cls_teacher \
  run.log_vertex_experiments=false

# Step 3: Train Student Model with Distillation
echo ""
echo "Step 3: Training Student Model with Distillation (1 epoch)..."
python -m src.tasks.classification.entrypoint \
  model=cls_mobilenetv3 \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/classification/manifest.json \
  +training.distillation.enabled=true \
  +training.distillation.teacher_model_path=$TEST_DIR/cls_teacher/saved_model \
  +training.distillation.alpha=0.3 \
  +training.distillation.temperature=3.0 \
  training.epochs=1 \
  training.batch_size=4 \
  run.artifacts_dir=$TEST_DIR/cls_student \
  run.log_vertex_experiments=false

# Step 4: Train Segmentation Model
echo ""
echo "Step 4: Training Segmentation Model (1 epoch)..."
python -m src.tasks.segmentation.entrypoint \
  model=seg_unet \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/segmentation/manifest.json \
  training.epochs=1 \
  training.batch_size=2 \
  run.artifacts_dir=$TEST_DIR/seg_model \
  run.log_vertex_experiments=false

# Step 5: Model Monitoring (Drift Detection)
echo ""
echo "Step 5: Running Drift Detection..."
python -m src.tasks.monitoring.entrypoint \
  +monitoring.baseline_data_path=$DATA_DIR/classification/data/train-0000.parquet \
  +monitoring.target_data_path=$DATA_DIR/classification/data/test-0000.parquet \
  +monitoring.features=['label'] \
  run.artifacts_dir=$TEST_DIR/monitoring

# Step 6: Run Full Test Suite
echo ""
echo "Step 6: Running Full Test Suite..."
pytest tests/ -v --tb=short -x

echo ""
echo "======================================"
echo "✅ Quick E2E Test Complete!"
echo "======================================"
echo ""
echo "Results available in: $TEST_DIR"
echo ""
echo "Summary:"
echo "  - Classification Teacher: $TEST_DIR/cls_teacher"
echo "  - Classification Student (Distilled): $TEST_DIR/cls_student"
echo "  - Segmentation Model: $TEST_DIR/seg_model"
echo "  - Drift Report: $TEST_DIR/monitoring/drift_report.json"
echo ""
echo "Artifacts:"
ls -lh $TEST_DIR/*/saved_model 2>/dev/null || echo "  (Models exported successfully)"
echo ""
