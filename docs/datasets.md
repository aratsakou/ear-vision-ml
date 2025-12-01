# Datasets

## Artefact layout
Each dataset folder must contain:
- `data/` Parquet shards (train/val/test)
- `manifest.json` (single source of truth)
- `stats.json`

## Manifest schema
See `src/core/contracts/dataset_manifest_schema.json`.

## Parquet schema
Required columns:
- sample_id
- media_uri
- task_label
- split

Optional columns:
- timestamp_ms
- canonical_labels
- roi_bbox_xyxy_norm

## How to add a new dataset
1. Create a new folder in your data storage (e.g., `data/my_dataset`).
2. Generate `manifest.json` following `src/core/contracts/dataset_manifest_schema.json`.
3. Generate `stats.json`.
4. Create a config file in `configs/data/my_dataset.yaml` pointing to the manifest path.
5. Verify loading with `tests/integration/test_dataset_build_smoke.py` (adapt as needed).

## Handling Unbalanced Datasets

### Class Weighting
Automatically compute class weights based on training data frequency and apply them to the loss function.
```yaml
training:
  class_weights: true
```

### Oversampling
Balance the dataset by oversampling minority classes (random oversampling) during data loading.
```yaml
data:
  dataset:
    sampling:
      strategy: "oversample"
```

