from __future__ import annotations

import re
import uuid
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from tali.db import Database
from tali.models import FORBIDDEN_FACT_TYPES, ProvenanceType
from tali.questions import queue_question


CONFIDENCE_DEFAULTS: dict[str, float] = {
    ProvenanceType.TOOL_VERIFIED.value: 0.95,
    ProvenanceType.SYSTEM_OBSERVED.value: 0.85,
    ProvenanceType.RETRIEVED_SOURCE.value: 0.75,
    ProvenanceType.USER_REPORTED.value: 0.60,
    ProvenanceType.INFERRED.value: 0.30,
}


@dataclass(frozen=True)
class ConsolidationResult:
    inserted_fact_ids: list[str]
    skipped_candidates: list[str]
    contested_fact_ids: list[str]
    staged_item_ids: list[str]
    applied_commitment_ids: list[str]
    applied_skill_ids: list[str]
    notes: list[str]


@dataclass(frozen=True)
class SleepPolicy:
    allow_fact_provenance: tuple[str, ...] = (
        ProvenanceType.TOOL_VERIFIED.value,
        ProvenanceType.SYSTEM_OBSERVED.value,
    )
    confidence_threshold: float = 0.8
    decay_days: int = 30


def apply_sleep_changes(
    db: Database,
    payload: dict[str, object],
    policy: SleepPolicy | None = None,
    hook_manager: object | None = None,
) -> ConsolidationResult:
    policy = policy or SleepPolicy()
    fact_candidates = payload.get("fact_candidates", [])
    commitment_updates = payload.get("commitment_updates", [])
    skill_candidates = payload.get("skill_candidates", [])
    notes = payload.get("notes", [])

    inserted_fact_ids: list[str] = []
    skipped_candidates: list[str] = []
    contested_fact_ids: list[str] = []
    staged_item_ids: list[str] = []
    applied_commitment_ids: list[str] = []
    applied_skill_ids: list[str] = []
    staged_notes: list[str] = []
    existing_commitments = db.list_commitments()
    existing_skills = db.list_skills()

    if not isinstance(fact_candidates, list):
        raise ValueError("fact_candidates must be a list")
    if not isinstance(commitment_updates, list):
        raise ValueError("commitment_updates must be a list")
    if not isinstance(skill_candidates, list):
        raise ValueError("skill_candidates must be a list")
    if not isinstance(notes, list):
        raise ValueError("notes must be a list")

    cutoff = (datetime.utcnow() - timedelta(days=policy.decay_days)).isoformat()
    db.apply_fact_decay(cutoff)
    db.dedupe_facts()

    for candidate in fact_candidates:
        if not isinstance(candidate, dict):
            skipped_candidates.append("invalid_candidate")
            continue
        statement = str(candidate.get("statement") or "").strip()
        provenance_type = str(candidate.get("provenance_type") or "").strip()
        source_ref = str(candidate.get("source_ref") or "").strip()

        if not statement or not provenance_type or not source_ref:
            skipped_candidates.append("missing_fields")
            continue
        if provenance_type in FORBIDDEN_FACT_TYPES:
            skipped_candidates.append("forbidden_provenance")
            continue
        if provenance_type not in CONFIDENCE_DEFAULTS:
            skipped_candidates.append("unknown_provenance")
            continue
        if not db.episode_exists(str(source_ref)):
            skipped_candidates.append("missing_source")
            continue
        episode = db.fetch_episode(str(source_ref))
        if episode and episode["quarantine"]:
            skipped_candidates.append("quarantined_source")
            continue

        default_confidence = float(CONFIDENCE_DEFAULTS[str(provenance_type)])
        candidate_confidence = _coerce_confidence(candidate.get("confidence"), default_confidence)
        candidate_payload = {
            "statement": statement,
            "provenance_type": provenance_type,
            "source_ref": source_ref,
            "confidence": candidate_confidence,
            "tags": candidate.get("tags", []),
        }

        if provenance_type not in policy.allow_fact_provenance:
            staged_item_ids.append(
                _stage_item(
                    db,
                    kind="fact",
                    payload=candidate_payload,
                    provenance_type=provenance_type,
                    source_ref=source_ref,
                )
            )
            continue
        if not can_promote_fact(provenance_type, source_ref, candidate_confidence):
            staged_item_ids.append(
                _stage_item(
                    db,
                    kind="fact",
                    payload=candidate_payload,
                    provenance_type=provenance_type,
                    source_ref=source_ref,
                )
            )
            continue

        existing = db.search_facts(statement, limit=5)
        if existing and any(row["statement"] == statement for row in existing):
            skipped_candidates.append("duplicate")
            continue

        contested = 0
        contradictory_ids: list[str] = []
        blocked_by_contradiction = False
        for row in existing:
            if _is_contradiction(statement, row["statement"]):
                contradictory_ids.append(str(row["id"]))
                if float(row["confidence"]) >= policy.confidence_threshold:
                    staged_item_ids.append(
                        _stage_item(
                            db,
                            kind="fact",
                            payload={
                                **candidate_payload,
                                "contested": True,
                                "related_fact_ids": contradictory_ids,
                            },
                            provenance_type=provenance_type,
                            source_ref=source_ref,
                        )
                    )
                    _queue_contradiction_question(
                        db,
                        new_statement=statement,
                        existing_statement=str(row["statement"]),
                        new_fact_id=None,
                        existing_fact_id=str(row["id"]),
                    )
                    if hook_manager and hasattr(hook_manager, "run"):
                        hook_manager.run("on_contradiction_detected", {
                            "new_statement": statement,
                            "existing_statement": str(row["statement"]),
                            "new_fact_id": None,
                            "existing_fact_id": str(row["id"]),
                        })
                    skipped_candidates.append("contradiction_high_confidence")
                    blocked_by_contradiction = True
                    break
                contested = 1

        if blocked_by_contradiction:
            continue

        fact_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        db.insert_fact(
            fact_id=fact_id,
            statement=statement,
            provenance_type=provenance_type,
            source_ref=source_ref,
            confidence=candidate_confidence,
            created_at=created_at,
            contested=contested,
        )
        inserted_fact_ids.append(fact_id)
        if contested:
            contested_fact_ids.append(fact_id)
            for related_id in contradictory_ids:
                link_id = str(uuid.uuid4())
                db.insert_fact_link(
                    link_id=link_id,
                    fact_id=fact_id,
                    related_fact_id=related_id,
                    episode_id=source_ref,
                    link_type="contradicts",
                    created_at=datetime.utcnow().isoformat(),
                )
                db.mark_fact_contested(related_id)
                existing_rows = db.fetch_facts_by_ids([related_id])
                existing_stmt = str(existing_rows[0]["statement"]) if existing_rows else ""
                _queue_contradiction_question(
                    db,
                    new_statement=statement,
                    existing_statement=existing_stmt,
                    new_fact_id=fact_id,
                    existing_fact_id=related_id,
                )
                if hook_manager and hasattr(hook_manager, "run"):
                    hook_manager.run("on_contradiction_detected", {
                        "new_statement": statement,
                        "existing_statement": existing_stmt,
                        "new_fact_id": fact_id,
                        "existing_fact_id": related_id,
                    })

    for candidate in commitment_updates:
        if not isinstance(candidate, dict):
            skipped_candidates.append("invalid_commitment")
            continue
        description = str(candidate.get("description") or "").strip()
        status = str(candidate.get("status") or "pending").strip()
        source_ref = str(candidate.get("source_ref") or "").strip()
        commitment_id = candidate.get("commitment_id")
        priority = int(candidate.get("priority", 3) or 3)
        due_date = candidate.get("due_date")
        if not description or not source_ref:
            skipped_candidates.append("commitment_missing_fields")
            continue
        if not db.episode_exists(str(source_ref)):
            skipped_candidates.append("commitment_missing_source")
            continue
        if any(
            str(row["description"]).strip().lower() == description.lower()
            for row in existing_commitments
        ):
            skipped_candidates.append("commitment_duplicate")
            continue
        if not _is_explicit_commitment(db, source_ref, description, candidate):
            staged_item_ids.append(
                _stage_item(
                    db,
                    kind="commitment",
                    payload=candidate,
                    provenance_type="USER_REPORTED",
                    source_ref=source_ref,
                )
            )
            continue
        now = datetime.utcnow().isoformat()
        if commitment_id and db.fetch_commitment(str(commitment_id)):
            db.update_commitment(
                commitment_id=str(commitment_id),
                description=description,
                status=status,
                priority=priority,
                due_date=str(due_date) if due_date else None,
                last_touched=now,
            )
            applied_commitment_ids.append(str(commitment_id))
        else:
            new_id = str(commitment_id or uuid.uuid4())
            db.insert_commitment(
                commitment_id=new_id,
                description=description,
                status=status,
                priority=priority,
                due_date=str(due_date) if due_date else None,
                created_at=now,
                last_touched=now,
                source_ref=source_ref,
            )
            applied_commitment_ids.append(new_id)

    for candidate in skill_candidates:
        if not isinstance(candidate, dict):
            skipped_candidates.append("invalid_skill")
            continue
        name = str(candidate.get("name") or "").strip()
        trigger = str(candidate.get("trigger") or "").strip()
        steps = candidate.get("steps") or []
        source_ref = str(candidate.get("source_ref") or "").strip()
        evidence = candidate.get("success_evidence") or []
        if not name or not trigger or not steps or not source_ref:
            skipped_candidates.append("skill_missing_fields")
            continue
        if not db.episode_exists(str(source_ref)):
            skipped_candidates.append("skill_missing_source")
            continue
        if any(
            str(row["trigger"]).strip().lower() == trigger.lower()
            for row in existing_skills
        ):
            skipped_candidates.append("skill_duplicate_trigger")
            continue
        if not _skill_evidence_strong(evidence) or not _steps_tool_safe(steps):
            staged_item_ids.append(
                _stage_item(
                    db,
                    kind="skill",
                    payload=candidate,
                    provenance_type="USER_REPORTED",
                    source_ref=source_ref,
                )
            )
            continue
        if db.fetch_skill_by_name(name):
            skipped_candidates.append("skill_duplicate")
            continue
        db.insert_skill(
            skill_id=str(uuid.uuid4()),
            name=name,
            trigger=trigger,
            steps=json.dumps(steps),
            created_at=datetime.utcnow().isoformat(),
            source_ref=source_ref,
        )
        applied_skill_ids.append(name)

    for note in notes:
        if isinstance(note, str) and note.strip():
            staged_notes.append(note.strip())

    if staged_notes:
        staged_item_ids.append(
            _stage_item(
                db,
                kind="commitment",
                payload={"note_only": True, "notes": staged_notes},
                provenance_type="SYSTEM_OBSERVED",
                source_ref="sleep_notes",
            )
        )

    return ConsolidationResult(
        inserted_fact_ids=inserted_fact_ids,
        skipped_candidates=skipped_candidates,
        contested_fact_ids=contested_fact_ids,
        staged_item_ids=staged_item_ids,
        applied_commitment_ids=applied_commitment_ids,
        applied_skill_ids=applied_skill_ids,
        notes=staged_notes,
    )


def apply_sleep_output(db: Database, payload: dict[str, object]) -> ConsolidationResult:
    return apply_sleep_changes(db, payload, SleepPolicy())


def can_promote_fact(provenance_type: str, source_ref: str, confidence: float) -> bool:
    if not provenance_type or not source_ref:
        return False
    if provenance_type in FORBIDDEN_FACT_TYPES:
        return False
    if provenance_type == ProvenanceType.INFERRED.value and confidence < 0.5:
        return False
    return True


def _stage_item(
    db: Database,
    kind: str,
    payload: dict[str, object],
    provenance_type: str,
    source_ref: str,
) -> str:
    item_id = str(uuid.uuid4())
    db.insert_staged_item(
        item_id=item_id,
        kind=kind,
        payload=json.dumps(payload),
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        source_ref=source_ref,
        provenance_type=provenance_type,
        next_check_at=datetime.utcnow().isoformat(),
    )
    return item_id


def _is_explicit_commitment(
    db: Database,
    source_ref: str,
    description: str,
    candidate: dict[str, object],
) -> bool:
    explicit_flag = candidate.get("explicit_request")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    episode = db.fetch_episode(source_ref)
    if not episode:
        return False
    user_text = (episode["user_input"] or "").lower()
    description_tokens = description.lower()
    explicit_markers = [
        "please",
        "remind me",
        "i will",
        "i'll",
        "we will",
        "set a reminder",
        "remember to",
        "need to",
        "let's",
    ]
    return description_tokens in user_text or any(marker in user_text for marker in explicit_markers)


def _skill_evidence_strong(evidence: object) -> bool:
    return isinstance(evidence, list) and len(evidence) >= 2


def _steps_tool_safe(steps: object) -> bool:
    if not isinstance(steps, list):
        return False
    unsafe_markers = {"sudo", "rm -rf", "format disk", "delete system"}
    for step in steps:
        if not isinstance(step, str):
            return False
        lowered = step.lower()
        if any(marker in lowered for marker in unsafe_markers):
            return False
    return True


def _coerce_confidence(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _queue_contradiction_question(
    db: Database,
    new_statement: str,
    existing_statement: str,
    new_fact_id: str | None,
    existing_fact_id: str,
) -> str:
    """Queue a high-priority question asking the user to resolve a contradiction."""
    question = (
        f"I found conflicting information: \"{new_statement}\" vs "
        f"\"{existing_statement}\". Which is correct?"
    )
    reason = json.dumps({
        "type": "contradiction",
        "new_statement": new_statement,
        "existing_statement": existing_statement,
        "new_fact_id": new_fact_id,
        "existing_fact_id": existing_fact_id,
    })
    return queue_question(db, question=question, reason=reason, priority=5)


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _is_contradiction(statement: str, existing: str) -> bool:
    normalized_statement = _normalize(statement)
    normalized_existing = _normalize(existing)
    if normalized_statement == normalized_existing:
        return False
    negations = {"not", "never", "no"}
    statement_tokens = set(normalized_statement.split())
    existing_tokens = set(normalized_existing.split())
    if statement_tokens == existing_tokens:
        return False
    if statement_tokens - negations == existing_tokens - negations:
        return ("not" in statement_tokens) != ("not" in existing_tokens) or (
            "never" in statement_tokens
        ) != ("never" in existing_tokens) or ("no" in statement_tokens) != ("no" in existing_tokens)
    return False
