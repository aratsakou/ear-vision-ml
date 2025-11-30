import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path("src/core/contracts/dataset_manifest_schema.json")

@pytest.fixture
def manifest_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def test_manifest_schema_valid(manifest_schema: dict) -> None:
    """Test validation of a valid manifest."""
    valid_manifest = {
        "dataset_id": "ds-123",
        "task_name": "classification",
        "version": "v1",
        "status": "active",
        "label_mapping_versions": {
            "ontology": "v1",
            "labelbox_mappings": ["v1"],
            "task_mapping": "v1"
        },
        "sampling_config": {
            "hash": "abc123hash",
            "parameters": {"strategy": "uniform"}
        },
        "preprocess_pipeline": {
            "id": "full_frame",
            "version": "v1"
        },
        "splits": {
            "train": ["p1.parquet"],
            "val": ["p2.parquet"],
            "test": ["p3.parquet"]
        },
        "created_by": "user",
        "created_at": "2023-10-27T10:00:00Z"
    }
    jsonschema.validate(instance=valid_manifest, schema=manifest_schema)

def test_manifest_schema_missing_required(manifest_schema: dict) -> None:
    """Test validation fails when required fields are missing."""
    invalid_manifest = {
        "dataset_id": "ds-123"
        # Missing other required fields
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_manifest, schema=manifest_schema)

def test_manifest_schema_invalid_enum(manifest_schema: dict) -> None:
    """Test validation fails for invalid enum values."""
    invalid_manifest = {
        "dataset_id": "ds-123",
        "task_name": "classification",
        "version": "v1",
        "status": "invalid_status", # Invalid
        "label_mapping_versions": {
            "ontology": "v1",
            "task_mapping": "v1"
        },
        "sampling_config": {"hash": "h"},
        "preprocess_pipeline": {"id": "p", "version": "v"},
        "splits": {},
        "created_by": "u",
        "created_at": "2023-10-27T10:00:00Z"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_manifest, schema=manifest_schema)
