from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalRunContext:
    run_id: str
    artifacts_dir: Path


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def init_local_run(artifacts_root: str, run_name: str) -> LocalRunContext:
    run_id = make_run_id(run_name)
    out = Path(artifacts_root) / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return LocalRunContext(run_id=run_id, artifacts_dir=out)


def write_run_record(ctx: LocalRunContext, record: dict[str, Any]) -> None:
    p = ctx.artifacts_dir / "run.json"
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
