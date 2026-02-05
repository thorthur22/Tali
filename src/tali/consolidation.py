from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from tali.db import Database
from tali.models import FORBIDDEN_FACT_TYPES, ProvenanceType


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


def apply_sleep_output(db: Database, payload: dict[str, object]) -> ConsolidationResult:
    fact_candidates = payload.get("fact_candidates", [])
    inserted_fact_ids: list[str] = []
    skipped_candidates: list[str] = []
    contested_fact_ids: list[str] = []
    last_run = db.last_sleep_run()
    cutoff_date = last_run["timestamp"] if last_run else datetime.utcnow().isoformat()
    db.apply_confidence_decay(cutoff_date)

    if not isinstance(fact_candidates, list):
        raise ValueError("fact_candidates must be a list")

    for candidate in fact_candidates:
        if not isinstance(candidate, dict):
            skipped_candidates.append("invalid_candidate")
            continue
        statement = candidate.get("statement")
        provenance_type = candidate.get("provenance_type")
        source_ref = candidate.get("source_ref")

        if not statement or not provenance_type or not source_ref:
            skipped_candidates.append("missing_fields")
            continue
        if provenance_type in FORBIDDEN_FACT_TYPES:
            skipped_candidates.append("forbidden_provenance")
            continue
        if provenance_type not in CONFIDENCE_DEFAULTS:
            skipped_candidates.append("unknown_provenance")
            continue
        if not can_promote_fact(candidate):
            skipped_candidates.append("promotion_gate")
            continue
        if not db.episode_exists(str(source_ref)):
            skipped_candidates.append("missing_source")
            continue
        episode = db.fetch_episode(str(source_ref))
        if episode and episode["quarantine"]:
            skipped_candidates.append("quarantined_source")
            continue

        existing = db.search_facts(str(statement), limit=5)
        if existing and any(row["statement"] == statement for row in existing):
            skipped_candidates.append("duplicate")
            continue

        contested = 0
        contradictory_ids: list[str] = []
        for row in existing:
            if _is_contradiction(str(statement), row["statement"]):
                contested = 1
                contradictory_ids.append(str(row["id"]))

        fact_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        db.insert_fact(
            fact_id=fact_id,
            statement=str(statement),
            provenance_type=str(provenance_type),
            source_ref=str(source_ref),
            confidence=CONFIDENCE_DEFAULTS[str(provenance_type)],
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
                    episode_id=str(source_ref),
                    link_type="contradicts",
                    created_at=datetime.utcnow().isoformat(),
                )
                db.mark_fact_contested(related_id)

    return ConsolidationResult(
        inserted_fact_ids=inserted_fact_ids,
        skipped_candidates=skipped_candidates,
        contested_fact_ids=contested_fact_ids,
    )


def can_promote_fact(candidate: dict[str, object]) -> bool:
    provenance_type = candidate.get("provenance_type")
    if provenance_type == "AGENT_OUTPUT":
        return False
    if not candidate.get("source_ref"):
        return False
    confidence = candidate.get("confidence")
    confidence_value = CONFIDENCE_DEFAULTS.get(str(provenance_type), 0.0)
    if isinstance(confidence, (int, float)):
        confidence_value = float(confidence)
    elif isinstance(confidence, str) and confidence.strip():
        try:
            confidence_value = float(confidence)
        except ValueError:
            confidence_value = CONFIDENCE_DEFAULTS.get(str(provenance_type), 0.0)
    if provenance_type == ProvenanceType.INFERRED.value and confidence_value < 0.5:
        return False
    return True


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\\s]", "", lowered)
    return re.sub(r"\\s+", " ", lowered).strip()


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
