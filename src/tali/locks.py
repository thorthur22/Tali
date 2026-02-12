from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LockInfo:
    pid: int | None
    created_at: datetime | None
    raw: dict[str, Any]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it.
        return True
    except OSError:
        return False


def _parse_lock_payload(text: str) -> LockInfo:
    text = text.strip()
    if not text:
        return LockInfo(pid=None, created_at=None, raw={})
    try:
        payload = json.loads(text)
        if isinstance(payload, int):
            return LockInfo(pid=payload, created_at=None, raw={"pid": payload})
        if not isinstance(payload, dict):
            return LockInfo(pid=None, created_at=None, raw={})
    except json.JSONDecodeError:
        if text.isdigit():
            pid = int(text)
            return LockInfo(pid=pid, created_at=None, raw={"pid": pid})
        return LockInfo(pid=None, created_at=None, raw={})

    pid = payload.get("pid")
    created_at = None
    created_raw = payload.get("created_at")
    if isinstance(created_raw, str):
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except ValueError:
            created_at = None
    if isinstance(pid, str) and pid.isdigit():
        pid = int(pid)
    if not isinstance(pid, int):
        pid = None
    return LockInfo(pid=pid, created_at=created_at, raw=payload)


@dataclass
class FileLock:
    path: Path
    stale_after_s: float | None = None
    allow_stale_clear: bool = True
    _fd: int | None = None

    def acquire(self) -> bool:
        if self._try_acquire():
            return True
        if self.allow_stale_clear and self._clear_stale():
            return self._try_acquire()
        return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        if self.path.exists():
            self.path.unlink()

    def _try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        payload = {
            "pid": os.getpid(),
            "created_at": _utcnow().isoformat(),
        }
        os.write(self._fd, json.dumps(payload).encode("utf-8"))
        return True

    def _clear_stale(self) -> bool:
        if not self.path.exists():
            return False
        try:
            content = self.path.read_text(encoding="utf-8")
        except OSError:
            return False
        info = _parse_lock_payload(content)
        if not self._is_stale(info):
            return False
        try:
            self.path.unlink()
            return True
        except OSError:
            return False

    def _is_stale(self, info: LockInfo) -> bool:
        if info.pid is not None and _pid_alive(info.pid):
            if self.stale_after_s is None or info.created_at is None:
                return False
            age_s = (_utcnow() - info.created_at).total_seconds()
            return age_s >= self.stale_after_s
        return True
