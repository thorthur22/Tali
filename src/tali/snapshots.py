from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Snapshot:
    id: str
    path: Path


def create_snapshot(data_dir: Path) -> Snapshot:
    snapshots_dir = data_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshots_dir / snapshot_id
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    shutil.copytree(data_dir, snapshot_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns("snapshots"))
    return Snapshot(id=snapshot_id, path=snapshot_path)


def list_snapshots(data_dir: Path) -> list[Snapshot]:
    snapshots_dir = data_dir / "snapshots"
    if not snapshots_dir.exists():
        return []
    snapshots = []
    for path in sorted(snapshots_dir.iterdir()):
        if path.is_dir():
            snapshots.append(Snapshot(id=path.name, path=path))
    return snapshots


def diff_snapshot(data_dir: Path, snapshot: Snapshot) -> str:
    diffs: list[str] = []
    for current_path in sorted(data_dir.rglob("*")):
        if "snapshots" in current_path.parts or current_path.is_dir():
            continue
        relative = current_path.relative_to(data_dir)
        snapshot_path = snapshot.path / relative
        if not snapshot_path.exists():
            diffs.append(f"Added: {relative}")
            continue
        if current_path.suffix in {".db", ".index"}:
            if current_path.read_bytes() != snapshot_path.read_bytes():
                diffs.append(f"Binary changed: {relative}")
            continue
        current_text = current_path.read_text(errors="ignore").splitlines()
        snapshot_text = snapshot_path.read_text(errors="ignore").splitlines()
        diff = difflib.unified_diff(
            snapshot_text,
            current_text,
            fromfile=str(snapshot_path),
            tofile=str(current_path),
            lineterm="",
        )
        diff_lines = list(diff)
        if diff_lines:
            diffs.extend(diff_lines)
    return "\n".join(diffs) if diffs else "No differences detected."


def rollback_snapshot(data_dir: Path, snapshot: Snapshot) -> None:
    for path in data_dir.iterdir():
        if path.name == "snapshots":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in snapshot.path.iterdir():
        target = data_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
