from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from tali.db import Database
from tali.idle_jobs import IdleJobRunner
from tali.knowledge_sources import KnowledgeSourceRegistry


IDLE_TRIGGER_SECONDS = 300
IDLE_MIN_INTERVAL_SECONDS = 1800
IDLE_LOCK_FILENAME = "idle.lock"
IDLE_LAST_RUN_FILENAME = "idle.last_run"


@dataclass
class IdleLock:
    lock_path: Path
    _fd: int | None = None

    def acquire(self) -> bool:
        try:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        if self.lock_path.exists():
            self.lock_path.unlink()


class IdleScheduler:
    def __init__(
        self,
        data_dir: Path,
        db: Database,
        llm: Any,
        sources: KnowledgeSourceRegistry,
        idle_check_s: int = 30,
        hook_manager: Any | None = None,
        status_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.db = db
        self.llm = llm
        self.sources = sources
        self.idle_check_s = idle_check_s
        self.hook_manager = hook_manager
        self.status_fn = status_fn
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_activity = datetime.utcnow()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def update_activity(self) -> None:
        self._last_activity = datetime.utcnow()
        self._interrupt_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._should_run():
                self._run_idle_jobs()
            time.sleep(self.idle_check_s)

    def _should_run(self) -> bool:
        if self.db.fetch_active_run():
            return False
        idle_time = datetime.utcnow() - self._last_activity
        if idle_time.total_seconds() < IDLE_TRIGGER_SECONDS:
            return False
        last_run = self._last_idle_run()
        if last_run:
            since_last = datetime.utcnow() - last_run
            if since_last.total_seconds() < IDLE_MIN_INTERVAL_SECONDS:
                return False
        return True

    def _run_idle_jobs(self) -> None:
        lock = IdleLock(self.data_dir / IDLE_LOCK_FILENAME)
        if not lock.acquire():
            return
        try:
            self._interrupt_event.clear()
            if self.hook_manager:
                self.hook_manager.run("on_idle", {"timestamp": datetime.utcnow().isoformat()})
            runner = IdleJobRunner(
                db=self.db,
                llm=self.llm,
                sources=self.sources,
                should_stop=lambda: self._interrupt_event.is_set() or self._stop_event.is_set(),
            )
            result = runner.run_cycle()
            if self.status_fn:
                for msg in result.messages:
                    self.status_fn(msg)
            self._write_last_idle_run()
        finally:
            lock.release()

    def _last_idle_run(self) -> datetime | None:
        path = self.data_dir / IDLE_LAST_RUN_FILENAME
        if not path.exists():
            return None
        try:
            content = path.read_text().strip()
            if not content:
                return None
            return datetime.fromisoformat(content)
        except Exception:
            return None

    def _write_last_idle_run(self) -> None:
        path = self.data_dir / IDLE_LAST_RUN_FILENAME
        path.write_text(datetime.utcnow().isoformat())
