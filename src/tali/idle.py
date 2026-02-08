from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tali.config import AutonomyConfig
from tali.db import Database
from tali.idle_jobs import IdleJobRunner
from tali.knowledge_sources import KnowledgeSourceRegistry

if TYPE_CHECKING:
    from tali.task_runner import TaskRunner


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
        task_runner: "TaskRunner | None" = None,
        autonomy_config: AutonomyConfig | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.db = db
        self.llm = llm
        self.sources = sources
        self.idle_check_s = idle_check_s
        self.hook_manager = hook_manager
        self.status_fn = status_fn
        self.task_runner: TaskRunner | None = task_runner
        self.autonomy_config = autonomy_config
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_activity = datetime.utcnow()
        self._autonomous_run_id: str | None = None

    def set_task_runner(self, task_runner: "TaskRunner") -> None:
        """Set the task runner after construction (for deferred initialization)."""
        self.task_runner = task_runner

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
        self._cancel_autonomous_run()

    # ------------------------------------------------------------------
    # Autonomy helpers
    # ------------------------------------------------------------------

    def _autonomy_enabled(self) -> bool:
        return (
            self.autonomy_config is not None
            and self.autonomy_config.enabled
            and self.task_runner is not None
        )

    @property
    def _idle_trigger_seconds(self) -> int:
        if self.autonomy_config:
            return self.autonomy_config.idle_trigger_seconds
        return IDLE_TRIGGER_SECONDS

    @property
    def _idle_min_interval_seconds(self) -> int:
        if self.autonomy_config:
            return self.autonomy_config.idle_min_interval_seconds
        return IDLE_MIN_INTERVAL_SECONDS

    @property
    def _autonomous_idle_delay_seconds(self) -> int:
        if self.autonomy_config:
            return self.autonomy_config.autonomous_idle_delay_seconds
        return 30

    def _idle_seconds(self) -> float:
        return (datetime.utcnow() - self._last_activity).total_seconds()

    @staticmethod
    def _auto_prompt_fn(msg: str) -> str:
        """Non-interactive prompt function for autonomous runs.

        Returns ``"n"`` (deny) so that dangerous tool calls requiring
        explicit approval are rejected while safe tools pass through
        ``auto_approve_safe`` without hitting this callback.
        """
        return "n"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._autonomy_enabled():
                if self._should_auto_continue():
                    self._auto_continue_run()
                elif self._should_execute_commitment():
                    self._execute_commitment()
                elif self._should_run():
                    self._run_idle_jobs()
            elif self._should_run():
                self._run_idle_jobs()
            time.sleep(self.idle_check_s)

    # ------------------------------------------------------------------
    # Housekeeping checks (original behaviour, now with configurable timing)
    # ------------------------------------------------------------------

    def _should_run(self) -> bool:
        if self.db.fetch_active_run():
            return False
        if self._idle_seconds() < self._idle_trigger_seconds:
            return False
        last_run = self._last_idle_run()
        if last_run:
            since_last = datetime.utcnow() - last_run
            if since_last.total_seconds() < self._idle_min_interval_seconds:
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
                hook_manager=self.hook_manager,
            )
            result = runner.run_cycle()
            if self.status_fn:
                for msg in result.messages:
                    self.status_fn(msg)
            self._write_last_idle_run()
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # Autonomous: auto-continue blocked runs
    # ------------------------------------------------------------------

    def _should_auto_continue(self) -> bool:
        if not self.autonomy_config or not self.autonomy_config.auto_continue:
            return False
        if self._idle_seconds() < self._autonomous_idle_delay_seconds:
            return False
        run = self.db.fetch_autonomous_active_run()
        if not run:
            return False
        return run["status"] in {"active", "blocked"}

    def _auto_continue_run(self) -> None:
        if self._interrupt_event.is_set() or self._stop_event.is_set():
            return
        if self.task_runner is None:
            return
        lock = IdleLock(self.data_dir / IDLE_LOCK_FILENAME)
        if not lock.acquire():
            return
        try:
            self._interrupt_event.clear()
            if self.status_fn:
                self.status_fn("Autonomy: auto-continuing blocked run...")
            try:
                result = self.task_runner.run_turn(
                    "continue",
                    prompt_fn=self._auto_prompt_fn,
                    origin="autonomous",
                )
                if self.status_fn:
                    self.status_fn(f"Autonomy: auto-continue complete ({result.llm_calls} LLM calls).")
            except Exception as exc:
                if self.status_fn:
                    self.status_fn(f"Autonomy: auto-continue failed: {exc}")
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # Autonomous: execute pending commitments
    # ------------------------------------------------------------------

    def _should_execute_commitment(self) -> bool:
        if not self.autonomy_config or not self.autonomy_config.execute_commitments:
            return False
        if self._idle_seconds() < self._autonomous_idle_delay_seconds:
            return False
        # Don't start a new commitment if there is already an active run
        if self.db.fetch_active_run():
            return False
        pending = self.db.fetch_pending_commitments()
        return len(pending) > 0

    def _execute_commitment(self) -> None:
        if self._interrupt_event.is_set() or self._stop_event.is_set():
            return
        if self.task_runner is None:
            return

        pending = self.db.fetch_pending_commitments()
        if not pending:
            return
        commitment = pending[0]
        commitment_id = str(commitment["id"])
        description = str(commitment["description"])

        lock = IdleLock(self.data_dir / IDLE_LOCK_FILENAME)
        if not lock.acquire():
            return
        try:
            self._interrupt_event.clear()
            # Mark the commitment as active
            now = datetime.utcnow().isoformat()
            self.db.update_commitment(
                commitment_id=commitment_id,
                description=description,
                status="active",
                priority=int(commitment["priority"]),
                due_date=commitment["due_date"],
                last_touched=now,
            )
            if self.status_fn:
                self.status_fn(f"Autonomy: executing commitment \"{description[:80]}\"...")
            try:
                result = self.task_runner.run_turn(
                    description,
                    prompt_fn=self._auto_prompt_fn,
                    origin="autonomous",
                )
                self._autonomous_run_id = result.run_id

                # Check if interrupted during execution
                if self._interrupt_event.is_set():
                    # Revert commitment to pending so it can be retried
                    self.db.update_commitment(
                        commitment_id=commitment_id,
                        description=description,
                        status="pending",
                        priority=int(commitment["priority"]),
                        due_date=commitment["due_date"],
                        last_touched=datetime.utcnow().isoformat(),
                    )
                    if self.status_fn:
                        self.status_fn("Autonomy: commitment execution interrupted, reverted to pending.")
                    return

                # Determine outcome based on the run status
                run = self.db.fetch_active_run()
                if run and run["status"] == "blocked":
                    # Run is blocked waiting -- leave commitment as active
                    if self.status_fn:
                        self.status_fn("Autonomy: commitment run blocked, will auto-continue later.")
                elif run and run["status"] == "active":
                    # Still in progress
                    if self.status_fn:
                        self.status_fn("Autonomy: commitment in progress, will continue next cycle.")
                else:
                    # Run completed (or no active run means it finished)
                    self.db.update_commitment(
                        commitment_id=commitment_id,
                        description=description,
                        status="done",
                        priority=int(commitment["priority"]),
                        due_date=commitment["due_date"],
                        last_touched=datetime.utcnow().isoformat(),
                    )
                    self._autonomous_run_id = None
                    if self.status_fn:
                        self.status_fn(f"Autonomy: commitment completed ({result.llm_calls} LLM calls).")
            except Exception as exc:
                self.db.update_commitment(
                    commitment_id=commitment_id,
                    description=description,
                    status="failed",
                    priority=int(commitment["priority"]),
                    due_date=commitment["due_date"],
                    last_touched=datetime.utcnow().isoformat(),
                )
                self._autonomous_run_id = None
                if self.status_fn:
                    self.status_fn(f"Autonomy: commitment execution failed: {exc}")
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # Safe interruption
    # ------------------------------------------------------------------

    def _cancel_autonomous_run(self) -> None:
        """Cancel any self-initiated autonomous run when user activity resumes."""
        run = self.db.fetch_autonomous_active_run()
        if not run:
            self._autonomous_run_id = None
            return
        run_id = str(run["id"])
        # Set the autonomous run to blocked so it preserves state
        # but yields control to the user.
        if run["status"] == "active":
            self.db.update_run_status(run_id, status="blocked", last_error="interrupted_by_user")
            if self.status_fn:
                self.status_fn("Autonomy: paused autonomous run for user interaction.")
        self._autonomous_run_id = None

    # ------------------------------------------------------------------
    # Timestamp persistence for housekeeping cycles
    # ------------------------------------------------------------------

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
