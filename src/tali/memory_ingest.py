from __future__ import annotations

import json
import uuid
from datetime import datetime

from tali.db import Database
from tali.models import ProvenanceType
from tali.tools.protocol import ToolResult


MAX_STATEMENT_LEN = 240


def stage_tool_result_fact(db: Database, result: ToolResult) -> None:
    if result.status != "ok":
        return
    statement = (result.result_summary or "").strip()
    if not statement:
        return
    statement = _truncate(statement)
    payload = {
        "statement": statement,
        "provenance_type": ProvenanceType.TOOL_VERIFIED.value,
        "source_ref": f"tool_call:{result.id}",
        "confidence": 0.9,
    }
    db.insert_staged_item(
        item_id=str(uuid.uuid4()),
        kind="fact",
        payload=json.dumps(payload),
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        source_ref=payload["source_ref"],
        provenance_type=ProvenanceType.TOOL_VERIFIED.value,
        next_check_at=datetime.utcnow().isoformat(),
    )


def stage_episode_fact(db: Database, episode_id: str, user_input: str, outcome: str) -> None:
    statement = f"User request observed: {user_input}".strip()
    statement = _truncate(statement)
    payload = {
        "statement": statement,
        "provenance_type": ProvenanceType.SYSTEM_OBSERVED.value,
        "source_ref": episode_id,
        "confidence": 0.7,
        "tags": ["episode", outcome],
    }
    db.insert_staged_item(
        item_id=str(uuid.uuid4()),
        kind="fact",
        payload=json.dumps(payload),
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        source_ref=episode_id,
        provenance_type=ProvenanceType.SYSTEM_OBSERVED.value,
        next_check_at=datetime.utcnow().isoformat(),
    )


def _truncate(statement: str) -> str:
    if len(statement) <= MAX_STATEMENT_LEN:
        return statement
    return statement[: MAX_STATEMENT_LEN - 3] + "..."
