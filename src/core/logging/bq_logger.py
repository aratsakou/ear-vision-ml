"""BigQuery logger (optional).

MVP:
- Provide a thin wrapper that can insert JSON rows into tables if configured.
- Must be a no-op if disabled or not authenticated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover
    bigquery = None


@dataclass
class BigQueryLogger:
    enabled: bool
    dataset: str
    table: str

    def insert(self, row: dict[str, Any]) -> None:
        if not self.enabled or bigquery is None:
            return
        try:
            client = bigquery.Client()
            table_id = f"{self.dataset}.{self.table}"
            client.insert_rows_json(table_id, [row])
        except Exception:
            return
