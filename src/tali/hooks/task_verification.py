"""Hook that performs lightweight keyword-based verification when a task ends.

Listens to the ``on_task_end`` event.  When a task is marked ``done``, the
hook extracts the ``verification`` string from ``inputs_json`` and checks
whether the key terms appear in the stringified ``outputs_json``.  If fewer
than half the keywords are present, it stages a warning item and returns a
status message so the completion reviewer has an additional signal.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from tali.hooks.core import Hook, HookActions, HookContext

if TYPE_CHECKING:
    pass

_VERIFICATION_TIMEOUT_MS = 500

# Common English stop words to exclude from keyword extraction.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "and", "but", "or",
        "nor", "not", "so", "yet", "for", "with", "from", "into", "that",
        "this", "than", "then", "them", "they", "their", "there", "what",
        "when", "where", "which", "while", "who", "whom", "how", "about",
        "each", "every", "all", "both", "few", "more", "most", "other",
        "some", "such", "only", "same", "also", "just", "because", "very",
        "too", "any", "here", "done", "task", "verify", "check", "ensure",
        "confirm", "validate", "completion", "output", "result",
    }
)

_MIN_KEYWORD_LENGTH = 4


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from the verification string."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= _MIN_KEYWORD_LENGTH and w not in _STOP_WORDS]


def _handle_task_end(context: HookContext) -> HookActions | None:
    payload = context.payload
    status = payload.get("status")
    if status != "done":
        return None

    task_id = payload.get("task_id", "")
    inputs_json_raw = payload.get("inputs_json")
    outputs_json_raw = payload.get("outputs_json")

    # Parse inputs_json to get verification string.
    verification = ""
    if inputs_json_raw:
        try:
            inputs = json.loads(inputs_json_raw) if isinstance(inputs_json_raw, str) else inputs_json_raw
            verification = inputs.get("verification", "")
        except (json.JSONDecodeError, AttributeError):
            return None

    if not verification:
        return None

    keywords = _extract_keywords(verification)
    if not keywords:
        return None

    # Stringify outputs for keyword search.
    outputs_text = ""
    if outputs_json_raw:
        if isinstance(outputs_json_raw, str):
            outputs_text = outputs_json_raw.lower()
        else:
            outputs_text = json.dumps(outputs_json_raw).lower()

    matched = sum(1 for kw in keywords if kw in outputs_text)
    ratio = matched / len(keywords) if keywords else 1.0

    if ratio >= 0.5:
        return None

    warning = (
        f"Task {task_id} marked done but only {matched}/{len(keywords)} "
        f"verification keywords found in outputs (ratio={ratio:.2f})."
    )
    return HookActions(
        messages=[f"Hook: task_verification warning - {warning}"],
        staged_items=[
            {
                "kind": "fact",
                "payload": {
                    "statement": warning,
                    "task_id": task_id,
                    "verification": verification,
                    "keyword_match_ratio": ratio,
                },
                "provenance_type": "SYSTEM_OBSERVED",
                "source_ref": f"task_verification:{task_id}",
            }
        ],
    )


HOOKS = [
    Hook(
        name="task_verification",
        triggers={"on_task_end"},
        handler=_handle_task_end,
        timeout_ms=_VERIFICATION_TIMEOUT_MS,
    ),
]
