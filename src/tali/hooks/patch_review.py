"""Hook that performs a two-agent LLM safety review when a patch is proposed.

Listens to the ``on_patch_proposed`` event.  When fired, the hook uses the
LLM (passed via the event payload) to critique the diff for safety and
correctness.  The review result is stored in the database.

This hook has a generous timeout (120 s) because it makes an LLM call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tali.hooks.core import Hook, HookActions, HookContext

if TYPE_CHECKING:
    pass

_REVIEW_TIMEOUT_MS = 120_000  # 120 seconds for LLM call


def _handle_patch_proposed(context: HookContext) -> HookActions | None:
    payload = context.payload
    proposal_id = payload.get("proposal_id")
    proposal = payload.get("proposal")
    llm = payload.get("llm")
    if not proposal_id or not proposal or not llm:
        return None

    from tali.patches import PatchProposal, review_patch

    if isinstance(proposal, dict):
        proposal = PatchProposal(
            title=proposal.get("title", ""),
            rationale=proposal.get("rationale", ""),
            files=proposal.get("files", []),
            diff_text=proposal.get("diff_text", ""),
            tests=proposal.get("tests", []),
        )

    review_result = review_patch(llm, proposal)
    review_status = "proposed" if review_result.approved else "review_failed"
    context.db.update_patch_review(
        proposal_id=str(proposal_id),
        review_json=json.dumps({
            "approved": review_result.approved,
            "issues": review_result.issues,
            "reviewer_prompt": review_result.reviewer_prompt,
        }),
        status=review_status,
    )
    if review_result.approved:
        return HookActions(messages=["Hook: patch review passed."])
    issues_summary = "; ".join(review_result.issues[:3]) if review_result.issues else "no details"
    return HookActions(messages=[f"Hook: patch review FAILED: {issues_summary}"])


HOOKS = [
    Hook(
        name="patch_review",
        triggers={"on_patch_proposed"},
        handler=_handle_patch_proposed,
        timeout_ms=_REVIEW_TIMEOUT_MS,
    ),
]
