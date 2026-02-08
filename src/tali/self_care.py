from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import httpx

from tali.consolidation import SleepPolicy, _is_contradiction, apply_sleep_changes, can_promote_fact
from tali.db import Database
from tali.llm import OllamaClient, OpenAIClient
from tali.models import ProvenanceType
from tali.snapshots import create_snapshot, rollback_snapshot
if TYPE_CHECKING:
    from tali.vector_index import VectorIndex


LOCK_FILENAME = "sleep.lock"
SLEEP_LLM_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class ResolutionOutcome:
    clarification_question: str | None
    applied_fact_id: str | None


class SleepLock:
    def __init__(self, data_dir: Path) -> None:
        self.lock_path = data_dir / LOCK_FILENAME
        self._fd: int | None = None

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


class SleepScheduler:
    def __init__(
        self,
        data_dir: Path,
        db: Database,
        llm: Any,
        vector_index: "VectorIndex",
        sleep_interval_s: int = 60,
        hook_manager: Any | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.db = db
        self.llm = llm
        self.vector_index = vector_index
        self.sleep_interval_s = sleep_interval_s
        self.hook_manager = hook_manager
        self._stop_event = threading.Event()
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

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._should_run():
                run_auto_sleep(
                    self.data_dir,
                    self.db,
                    self.llm,
                    self.vector_index,
                    hook_manager=self.hook_manager,
                )
            time.sleep(self.sleep_interval_s)

    def _should_run(self) -> bool:
        episodes_since = self.db.count_episodes_since_last_sleep()
        if episodes_since >= 25:
            return True
        idle_time = datetime.utcnow() - self._last_activity
        if idle_time >= timedelta(minutes=15):
            return True
        last_run = self.db.last_sleep_run()
        if last_run:
            last_timestamp = datetime.fromisoformat(last_run["timestamp"])
            if datetime.utcnow() - last_timestamp >= timedelta(hours=6):
                return True
        return False


def run_auto_sleep(
    data_dir: Path,
    db: Database,
    llm: Any,
    vector_index: "VectorIndex",
    hook_manager: Any | None = None,
) -> None:
    from tali.sleep import load_sleep_output, run_sleep

    lock = SleepLock(data_dir)
    if not lock.acquire():
        return
    snapshot = None
    log_dir = data_dir / "sleep" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    audit: dict[str, Any] = {"timestamp": datetime.utcnow().isoformat()}
    try:
        output_dir = data_dir / "sleep"
        sleep_llm = _with_sleep_timeout(llm)
        output_path = run_sleep(db, output_dir, llm=sleep_llm)
        snapshot = create_snapshot(data_dir)
        payload = load_sleep_output(output_path)
        result = apply_sleep_changes(db, payload, SleepPolicy(), hook_manager=hook_manager)
        for fact_id in result.inserted_fact_ids:
            facts = db.fetch_facts_by_ids([fact_id])
            if facts:
                vector_index.add(item_type="fact", item_id=fact_id, text=facts[0]["statement"])
        audit.update(
            {
                "snapshot_id": snapshot.id,
                "sleep_output": str(output_path),
                "inserted_fact_ids": result.inserted_fact_ids,
                "staged_item_ids": result.staged_item_ids,
                "contested_fact_ids": result.contested_fact_ids,
            }
        )
        if hook_manager:
            hook_manager.run("on_sleep_complete", audit)
    except Exception as exc:  # noqa: BLE001
        error = _format_sleep_error(exc)
        audit["error"] = error
        if snapshot:
            rollback_snapshot(data_dir, snapshot)
        _stage_sleep_error(db, error)
    finally:
        log_path.write_text(json.dumps(audit, indent=2))
        lock.release()


def _with_sleep_timeout(llm: Any) -> Any:
    if isinstance(llm, (OpenAIClient, OllamaClient)):
        return replace(llm, timeout_s=SLEEP_LLM_TIMEOUT_S)
    return llm


def _format_sleep_error(exc: Exception) -> str:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timed out"
    return str(exc)


def resolve_staged_items(db: Database, user_input: str) -> ResolutionOutcome | None:
    now = datetime.utcnow().isoformat()
    row = db.fetch_next_staged_item(now)
    if not row:
        return None
    item_id = row["id"]
    kind = row["kind"]
    attempts = int(row["attempts"] or 0)
    payload = json.loads(row["payload"])

    if payload.get("note_only"):
        db.update_staged_item(item_id, status="rejected", next_check_at=None, attempts=attempts, last_error=None)
        return None

    if kind == "fact":
        statement = str(payload.get("statement") or "").strip()
        provenance_type = str(payload.get("provenance_type") or row["provenance_type"]).strip()
        source_ref = str(payload.get("source_ref") or row["source_ref"]).strip()
        confidence = float(payload.get("confidence", 0.6) or 0.6)
        if provenance_type in {ProvenanceType.TOOL_VERIFIED.value, ProvenanceType.SYSTEM_OBSERVED.value}:
            applied = _promote_fact(db, statement, provenance_type, source_ref, confidence)
            if applied:
                db.update_staged_item(item_id, status="resolved", next_check_at=None, attempts=attempts, last_error=None)
                return ResolutionOutcome(clarification_question=None, applied_fact_id=applied)
            return _backoff(db, item_id, attempts, "promotion_gate_failed")
        if provenance_type == ProvenanceType.USER_REPORTED.value:
            if payload.get("awaiting_confirmation"):
                response = _parse_confirmation(user_input)
                if response is True:
                    applied = _promote_fact(db, statement, provenance_type, source_ref, confidence)
                    if applied:
                        db.update_staged_item(
                            item_id, status="resolved", next_check_at=None, attempts=attempts, last_error=None
                        )
                        return ResolutionOutcome(clarification_question=None, applied_fact_id=applied)
                    return _backoff(db, item_id, attempts, "promotion_gate_failed")
                if response is False:
                    db.update_staged_item(
                        item_id, status="rejected", next_check_at=None, attempts=attempts, last_error=None
                    )
                    return None
                return _backoff(db, item_id, attempts, "awaiting_confirmation")
            if _is_relevant(user_input, statement):
                question = f"Quick check: is it true that \"{statement}\"?"
                payload["awaiting_confirmation"] = True
                db.update_staged_item_payload(item_id, json.dumps(payload))
                db.update_staged_item(
                    item_id,
                    status="verifying",
                    next_check_at=(datetime.utcnow() + timedelta(hours=1)).isoformat(),
                    attempts=attempts + 1,
                    last_error="asked_confirmation",
                )
                return ResolutionOutcome(clarification_question=question, applied_fact_id=None)
            return _backoff(db, item_id, attempts, "not_relevant")
        if provenance_type == ProvenanceType.INFERRED.value:
            return _backoff(db, item_id, attempts, "inferred_requires_verification")

    if kind == "commitment":
        description = str(payload.get("description") or "").strip()
        if payload.get("awaiting_clarification"):
            response = _parse_confirmation(user_input)
            if response is not None:
                db.update_staged_item(
                    item_id,
                    status="resolved" if response else "rejected",
                    next_check_at=None,
                    attempts=attempts,
                    last_error=None,
                )
                return None
            return _backoff(db, item_id, attempts, "awaiting_clarification")
        if _is_relevant(user_input, description):
            question = f"Should I treat this as an active commitment: \"{description}\"?"
            payload["awaiting_clarification"] = True
            db.update_staged_item_payload(item_id, json.dumps(payload))
            db.update_staged_item(
                item_id,
                status="verifying",
                next_check_at=(datetime.utcnow() + timedelta(hours=1)).isoformat(),
                attempts=attempts + 1,
                last_error="asked_clarification",
            )
            return ResolutionOutcome(clarification_question=question, applied_fact_id=None)
        return _backoff(db, item_id, attempts, "not_relevant")

    if kind == "skill":
        return _backoff(db, item_id, attempts, "needs_more_evidence")

    return _backoff(db, item_id, attempts, "unknown_kind")


def _promote_fact(
    db: Database,
    statement: str,
    provenance_type: str,
    source_ref: str,
    confidence: float,
) -> str | None:
    if not statement or not can_promote_fact(provenance_type, source_ref, confidence):
        return None
    existing = db.search_facts(statement, limit=5)
    if existing and any(row["statement"] == statement for row in existing):
        return None
    contradictory_ids: list[str] = []
    contested = 0
    for row in existing:
        if _is_contradiction(statement, row["statement"]):
            if float(row["confidence"]) >= 0.8:
                return None
            contested = 1
            contradictory_ids.append(str(row["id"]))
    fact_id = str(uuid.uuid4())
    db.insert_fact(
        fact_id=fact_id,
        statement=statement,
        provenance_type=provenance_type,
        source_ref=source_ref,
        confidence=confidence,
        created_at=datetime.utcnow().isoformat(),
        contested=contested,
    )
    if contested:
        for related_id in contradictory_ids:
            db.insert_fact_link(
                link_id=str(uuid.uuid4()),
                fact_id=fact_id,
                related_fact_id=related_id,
                episode_id=source_ref,
                link_type="contradicts",
                created_at=datetime.utcnow().isoformat(),
            )
            db.mark_fact_contested(related_id)
    return fact_id


def _backoff(db: Database, item_id: str, attempts: int, error: str) -> ResolutionOutcome | None:
    schedule = [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24), timedelta(days=3)]
    delay = schedule[min(attempts, len(schedule) - 1)]
    next_check = (datetime.utcnow() + delay).isoformat()
    db.update_staged_item(
        item_id=item_id,
        status="verifying",
        next_check_at=next_check,
        attempts=attempts + 1,
        last_error=error,
    )
    return None


def _is_relevant(user_input: str, statement: str) -> bool:
    if not user_input or not statement:
        return False
    lowered_input = user_input.lower()
    lowered_statement = statement.lower()
    return any(token in lowered_input for token in lowered_statement.split()[:5])


def _parse_confirmation(user_input: str) -> bool | None:
    normalized = user_input.strip().lower()
    if normalized.startswith(("yes", "yep", "y ", "y,", "y.")) or normalized in {
        "yes",
        "y",
        "yep",
        "correct",
        "affirmative",
        "sure",
    }:
        return True
    if normalized.startswith(("no", "nope", "n ", "n,", "n.")) or normalized in {
        "no",
        "n",
        "nope",
        "negative",
    }:
        return False
    return None


def resolve_contradiction_answer(
    db: Database,
    user_input: str,
    reason_json: dict[str, Any],
    source_ref: str,
) -> bool:
    """Resolve a contradiction based on the user's answer.

    The user is expected to have been asked which of two conflicting
    statements is correct.  We parse their answer, lower the confidence
    of the rejected fact, and record a resolution link.

    Returns True if the contradiction was resolved, False otherwise.
    """
    new_fact_id: str | None = reason_json.get("new_fact_id")
    existing_fact_id: str = reason_json.get("existing_fact_id", "")
    new_statement: str = reason_json.get("new_statement", "")
    existing_statement: str = reason_json.get("existing_statement", "")

    if not existing_fact_id:
        return False

    normalized = user_input.strip().lower()

    # Determine which fact the user chose.
    # Heuristic: if the answer contains tokens from one statement or
    # explicit ordinal references ("first", "second", "new", "old"),
    # pick that one.  Fall back to simple yes/no (yes = new is correct).
    keep_new: bool | None = None
    if any(kw in normalized for kw in ("first", "new", "the new")):
        keep_new = True
    elif any(kw in normalized for kw in ("second", "old", "the old", "existing", "original")):
        keep_new = False
    elif new_statement and new_statement.lower() in normalized:
        keep_new = True
    elif existing_statement and existing_statement.lower() in normalized:
        keep_new = False
    else:
        # Fall back to yes/no parsing (yes = the new statement is correct)
        confirmation = _parse_confirmation(user_input)
        if confirmation is True:
            keep_new = True
        elif confirmation is False:
            keep_new = False

    if keep_new is None:
        return False

    now = datetime.utcnow().isoformat()

    if keep_new:
        # Lower confidence of the existing (rejected) fact
        db.update_fact_confidence(existing_fact_id, 0.05)
        db.clear_fact_contested(existing_fact_id)
        if new_fact_id:
            db.clear_fact_contested(new_fact_id)
            db.insert_fact_link(
                link_id=str(uuid.uuid4()),
                fact_id=new_fact_id,
                related_fact_id=existing_fact_id,
                episode_id=source_ref,
                link_type="resolved_by_user",
                created_at=now,
            )
    else:
        # Lower confidence of the new (rejected) fact
        if new_fact_id:
            db.update_fact_confidence(new_fact_id, 0.05)
            db.clear_fact_contested(new_fact_id)
            db.insert_fact_link(
                link_id=str(uuid.uuid4()),
                fact_id=existing_fact_id,
                related_fact_id=new_fact_id,
                episode_id=source_ref,
                link_type="resolved_by_user",
                created_at=now,
            )
        db.clear_fact_contested(existing_fact_id)

    return True


def _stage_sleep_error(db: Database, error: str) -> None:
    payload = {"note_only": True, "error": error}
    db.insert_staged_item(
        item_id=str(uuid.uuid4()),
        kind="commitment",
        payload=json.dumps(payload),
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        source_ref="sleep_error",
        provenance_type=ProvenanceType.SYSTEM_OBSERVED.value,
        next_check_at=datetime.utcnow().isoformat(),
    )
