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
