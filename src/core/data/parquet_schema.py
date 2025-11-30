from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDatasetSchema:
    """Canonical task dataset row schema (Parquet).

    This is a logical schema, not a pyarrow schema object.
    """

    required_columns: list[str]
    optional_columns: list[str]


DEFAULT_SCHEMA = TaskDatasetSchema(
    required_columns=["sample_id", "media_uri", "task_label", "split"],
    optional_columns=["timestamp_ms", "canonical_labels", "roi_bbox_xyxy_norm"],
)


def validate_columns(columns: list[str], schema: TaskDatasetSchema = DEFAULT_SCHEMA) -> str | None:
    missing = [c for c in schema.required_columns if c not in columns]
    if missing:
        return f"missing_required_columns: {missing}"
    return None
