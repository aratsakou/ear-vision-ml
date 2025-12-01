# Hydra Simplification Plan

## Current Inventory

We have ~40 config files across 12 directories.

## Simplification Strategy

### 1. Essential Groups (Keep)
- **`task`**: Defines the high-level objective (`classification`, `segmentation`).
- **`model`**: Defines architecture (`cls_mobilenetv3`, `seg_unet`).
- **`data`**: Defines data loading (`local`, `gcs` - keep if code exists).
- **`preprocess`**: Defines ROI/preprocessing pipelines.
- **`training`**: Defines hyperparameters.
- **`export`**: Defines export settings.

### 2. Candidates for Removal/Merge

- **`configs/experiment/`**: `otoscopic_baseline.yaml`. **Action**: Keep as a reference example, but maybe rename to `examples/`.
- **`configs/ensemble/`**: `cloud_soft_voting.yaml`, `device_swift_v1.yaml`. **Action**: Check if `src/ensembles` is used. If not, delete.
- **`configs/monitoring/`**: `default.yaml`. **Action**: Merge into `training` or delete if unused.
- **`configs/tuning/`**: `default.yaml`. **Action**: Merge into `training` (as `tuning` section) to reduce file count.
- **`configs/evaluation/`**: `video.yaml`, `ab_test.yaml`. **Action**: Keep if code exists.

### 3. Proposed Minimal Surface

The user should primarily interact with:
```bash
python src/tasks/classification/entrypoint.py \
  task=classification \
  model=cls_mobilenetv3 \
  data=local
```

### 4. Action Plan

1.  **Verify Ensembles**: Check `src/ensembles`. If dead, delete configs and code.
2.  **Verify Monitoring**: Check `src/tasks/monitoring`.
3.  **Consolidate Tuning**: Move `configs/tuning/default.yaml` content into `configs/training/default.yaml` (disabled by default).
4.  **Consolidate Export**: Ensure `tflite` and `coreml` are sub-nodes of `export` group, or just part of the main config if they are small.

## Execution

1.  [ ] Check `src/ensembles` usage.
2.  [ ] Check `src/tasks/monitoring` usage.
3.  [ ] Merge `tuning` into `training`.
4.  [ ] Delete unused groups.
