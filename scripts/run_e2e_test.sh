#!/bin/bash
# End-to-End Test Script for ear-vision-ml
# This script runs the entire repository lifecycle with synthetic data

set -e  # Exit on error

echo "======================================"
echo "Starting End-to-End Verification Test"
echo "======================================"

# Configuration
E2E_DIR="artifacts/e2e_test"
DATA_DIR="$E2E_DIR/data"
MODELS_DIR="$E2E_DIR/models"

# Clean previous runs
echo "Cleaning previous E2E test artifacts..."
rm -rf $E2E_DIR
mkdir -p $E2E_DIR
mkdir -p $DATA_DIR
mkdir -p $MODELS_DIR

# Step 1: Generate Synthetic Dataset
echo ""
echo "Step 1: Generating synthetic dataset..."
python scripts/generate_fixtures.py \
  --output-dir $DATA_DIR \
  --num-images 100 \
  --num-classes 3

# Step 2: Train Classification Model (Teacher)
echo ""
echo "Step 2: Training Classification Model (Teacher)..."
python -m src.tasks.classification.entrypoint \
  model=cls_mobilenetv3 \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/classification/manifest.json \
  training.epochs=2 \
  training.batch_size=8 \
  run.artifacts_dir=$MODELS_DIR/cls_teacher \
  run.log_vertex_experiments=false

# Step 3: Train Segmentation Model
echo ""
echo "Step 3: Training Segmentation Model..."
python -m src.tasks.segmentation.entrypoint \
  model=seg_unet \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/segmentation/manifest.json \
  training.epochs=2 \
  training.batch_size=4 \
  run.artifacts_dir=$MODELS_DIR/seg_model \
  run.log_vertex_experiments=false

# Step 4: Train Student Model with Distillation
echo ""
echo "Step 4: Training Student Model (Distillation)..."
python -m src.tasks.classification.entrypoint \
  model=cls_mobilenetv3 \
  data=local \
  data.dataset.mode=manifest \
  data.dataset.manifest_path=$DATA_DIR/classification/manifest.json \
  +training.distillation.enabled=true \
  +training.distillation.teacher_model_path=$MODELS_DIR/cls_teacher/saved_model \
  +training.distillation.alpha=0.3 \
  +training.distillation.temperature=3.0 \
  training.epochs=2 \
  training.batch_size=8 \
  run.artifacts_dir=$MODELS_DIR/cls_student \
  run.log_vertex_experiments=false

# Step 5: Model Monitoring (Drift Detection)
echo ""
echo "Step 5: Running Drift Detection..."
python -m src.tasks.monitoring.entrypoint \
  monitoring.baseline_data_path=$DATA_DIR/classification/data/train-0000.parquet \
  monitoring.target_data_path=$DATA_DIR/classification/data/test-0000.parquet \
  monitoring.features=['image_uri'] \
  run.artifacts_dir=$E2E_DIR/monitoring

# Step 6: Run Full Test Suite
echo ""
echo "Step 6: Running Full Test Suite..."
pytest tests/ -v --tb=short

echo ""
echo "======================================"
echo "End-to-End Verification Complete!"
echo "======================================"
echo ""
echo "Results available in: $E2E_DIR"
echo ""
echo "Summary:"
echo "  - Classification Teacher: $MODELS_DIR/cls_teacher"
echo "  - Segmentation Model: $MODELS_DIR/seg_model"
echo "  - Classification Student (Distilled): $MODELS_DIR/cls_student"
echo "  - Drift Report: $E2E_DIR/monitoring/drift_report.json"
echo ""
