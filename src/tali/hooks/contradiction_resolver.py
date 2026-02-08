"""Hook that queues a high-priority user question when a contradiction is detected.

Listens to the ``on_contradiction_detected`` event.  When fired, the hook
queues a priority-5 question asking the user to resolve the conflict between
two facts.

This hook is fast (no LLM call) so the default timeout is sufficient.
"""

from __future__ import annotations

import json

from tali.hooks.core import Hook, HookActions, HookContext


def _handle_contradiction(context: HookContext) -> HookActions | None:
    payload = context.payload
    new_statement = payload.get("new_statement", "")
    existing_statement = payload.get("existing_statement", "")
    new_fact_id = payload.get("new_fact_id")
    existing_fact_id = payload.get("existing_fact_id", "")

    if not new_statement or not existing_statement or not existing_fact_id:
        return None

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
    context.queue_priority_question(question=question, reason=reason)
    return HookActions(messages=["Hook: contradiction question queued."])


HOOKS = [
    Hook(
        name="contradiction_resolver",
        triggers={"on_contradiction_detected"},
        handler=_handle_contradiction,
    ),
]
