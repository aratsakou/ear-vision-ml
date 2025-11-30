"""Offline Labelbox JSON ingestion (no API calls).

MVP scope:
- Provide a loader that reads a JSON export file and returns a normalised pandas DataFrame.
- Mapping to canonical ontology IDs is configured via YAML files under core/contracts/labelbox_mappings.

This module intentionally does NOT attempt to support all Labelbox export variants in MVP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LabelboxRecord:
    media_uri: str
    annotations: dict[str, Any]
    metadata: dict[str, Any]


def load_labelbox_json(json_path: str | Path) -> list[dict[str, Any]]:
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        # Some exports wrap payloads
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Unsupported Labelbox JSON format: expected list")  # keep strict
    return data


def normalise_to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        # Extremely defensive: Labelbox exports vary.
        media_uri = item.get("data_row", {}).get("row_data") or item.get("row_data") or item.get("media_uri")
        if not media_uri:
            continue
        rows.append(
            {
                "media_uri": media_uri,
                "annotations": item.get("projects") or item.get("annotations") or {},
                "metadata": item.get("metadata") or {},
            }
        )
    return pd.DataFrame(rows)
