from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from tali.db import Database


QUESTION_COOLDOWN = timedelta(hours=6)


@dataclass(frozen=True)
class QuestionDecision:
    question_id: str
    question: str
    reason: str | None
    attempts: int


def queue_question(
    db: Database,
    question: str,
    reason: str | None,
    priority: int = 3,
    next_ask_at: str | None = None,
) -> str:
    question_id = str(uuid.uuid4())
    db.insert_user_question(
        question_id=question_id,
        question=question,
        reason=reason,
        created_at=datetime.utcnow().isoformat(),
        status="queued",
        priority=priority,
        next_ask_at=next_ask_at,
        attempts=0,
    )
    return question_id


def select_question_to_ask(db: Database, user_input: str) -> QuestionDecision | None:
    now = datetime.utcnow().isoformat()
    row = db.fetch_next_user_question(now)
    if not row:
        return None
    if not _is_relevant(user_input, row["question"]) and int(row["priority"] or 0) < 4:
        return None
    return QuestionDecision(
        question_id=str(row["id"]),
        question=str(row["question"]),
        reason=str(row["reason"]) if row["reason"] else None,
        attempts=int(row["attempts"] or 0),
    )


def mark_question_asked(db: Database, question_id: str, attempts: int) -> None:
    next_ask_at = (datetime.utcnow() + QUESTION_COOLDOWN).isoformat()
    db.update_user_question_status(
        question_id=question_id,
        status="asked",
        next_ask_at=next_ask_at,
        attempts=attempts + 1,
    )


def resolve_answered_question(
    db: Database,
    user_input: str,
    question_row: dict[str, Any],
    source_ref: str,
) -> dict[str, Any] | None:
    normalized = user_input.strip().lower()
    if normalized in {"skip", "dismiss", "no thanks", "later"}:
        db.update_user_question_status(
            question_id=question_row["id"],
            status="dismissed",
            next_ask_at=None,
            attempts=int(question_row["attempts"] or 0),
        )
        return None
    payload = {
        "statement": user_input.strip(),
        "provenance_type": "USER_REPORTED",
        "source_ref": source_ref,
        "question": question_row["question"],
        "reason": question_row.get("reason"),
    }
    db.update_user_question_status(
        question_id=question_row["id"],
        status="answered",
        next_ask_at=None,
        attempts=int(question_row["attempts"] or 0),
    )
    return payload


def _is_relevant(user_input: str, question: str) -> bool:
    if not user_input or not question:
        return False
    lowered_input = user_input.lower()
    lowered_question = question.lower()
    return any(token in lowered_input for token in lowered_question.split()[:6])
