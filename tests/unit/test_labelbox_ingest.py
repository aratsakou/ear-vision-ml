import json
import pandas as pd
import pytest
from pathlib import Path
from src.core.data.labelbox_ingest import load_labelbox_json, normalise_to_dataframe

def test_load_labelbox_json_simple(tmp_path):
    # Create dummy JSON
    data = [
        {"row_data": "gs://bucket/img1.jpg", "annotations": {"cls": "ear"}},
        {"row_data": "gs://bucket/img2.jpg", "annotations": {"cls": "nose"}}
    ]
    p = tmp_path / "export.json"
    p.write_text(json.dumps(data))
    
    loaded = load_labelbox_json(p)
    assert len(loaded) == 2
    assert loaded[0]["row_data"] == "gs://bucket/img1.jpg"

def test_load_labelbox_json_wrapped(tmp_path):
    # Some exports wrap in "data"
    data = {
        "data": [
            {"row_data": "gs://bucket/img1.jpg", "annotations": {"cls": "ear"}}
        ]
    }
    p = tmp_path / "export_wrapped.json"
    p.write_text(json.dumps(data))
    
    loaded = load_labelbox_json(p)
    assert len(loaded) == 1

def test_normalise_to_dataframe():
    items = [
        # Variant 1: row_data at top level
        {
            "row_data": "gs://bucket/img1.jpg",
            "annotations": {"objects": []},
            "metadata": {"split": "train"}
        },
        # Variant 2: data_row.row_data (common in v2 exports)
        {
            "data_row": {"row_data": "gs://bucket/img2.jpg"},
            "projects": {"proj_id": {"labels": []}},
            "metadata": {"split": "val"}
        },
        # Variant 3: media_uri
        {
            "media_uri": "gs://bucket/img3.jpg",
            "annotations": {},
        },
        # Invalid
        {"foo": "bar"} 
    ]
    
    df = normalise_to_dataframe(items)
    
    assert len(df) == 3
    assert df.iloc[0]["media_uri"] == "gs://bucket/img1.jpg"
    assert df.iloc[1]["media_uri"] == "gs://bucket/img2.jpg"
    assert df.iloc[2]["media_uri"] == "gs://bucket/img3.jpg"
    
    assert "annotations" in df.columns
    assert "metadata" in df.columns
