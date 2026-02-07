from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_run_log(logs_dir: Path, entry: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "runs.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def read_recent_logs(logs_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    path = logs_dir / "runs.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    recent = lines[-limit:]
    entries: list[dict[str, Any]] = []
    for line in recent:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def latest_metrics_for_run(logs_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = logs_dir / "runs.log"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("run_id") == run_id:
            return payload
    return None
