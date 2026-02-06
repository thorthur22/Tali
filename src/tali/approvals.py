from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

try:
    import questionary
except Exception:  # pragma: no cover - optional UI dependency
    questionary = None


@dataclass
class ApprovalOutcome:
    approved: bool
    approval_mode: str
    reason: str | None = None


@dataclass
class ApprovalManager:
    mode: str = "prompt"
    approved_tools: set[str] = field(default_factory=set)
    approved_signatures: set[str] = field(default_factory=set)

    def _prompt(self, prompt_fn: Callable[[str], str], message: str) -> str:
        response = prompt_fn(message).strip().lower()
        return response

    def resolve(
        self,
        prompt_fn: Callable[[str], str],
        tool_name: str,
        signature: str | None,
        requires_approval: bool,
        reason: str | None,
        details: str | None = None,
    ) -> ApprovalOutcome:
        if self.mode == "deny":
            return ApprovalOutcome(approved=False, approval_mode="denied", reason="session denied")
        if signature and signature in self.approved_signatures:
            return ApprovalOutcome(approved=True, approval_mode="auto")
        if tool_name in self.approved_tools and not requires_approval:
            return ApprovalOutcome(approved=True, approval_mode="auto")
        if self.mode == "auto_approve_safe" and not requires_approval:
            return ApprovalOutcome(approved=True, approval_mode="auto")

        reason_line = f"Reason: {reason}" if reason else "Reason: policy requires approval"
        details_line = f"Details:\n{details}" if details else None
        title = f"Approve tool call '{tool_name}'?"
        if questionary:
            choices = [
                {"name": "Approve once", "value": "a"},
                {"name": "Approve tool for session", "value": "t"},
            ]
            if signature:
                choices.append({"name": "Approve command signature for session", "value": "s"})
            choices.extend(
                [
                    {"name": "Approve all safe for session", "value": "y"},
                    {"name": "Deny once", "value": "n"},
                    {"name": "Deny all for session", "value": "d"},
                ]
            )
            instruction = reason_line if not details_line else f"{reason_line}\n{details_line}"
            choice = questionary.select(title, choices=choices, instruction=instruction).ask()
            if choice is None:
                choice = "n"
        else:
            prompt = "\n".join(
                [
                    title,
                    reason_line,
                    details_line or "",
                    "Options: [a]pprove once, approve [t]ool for session, approve [s]ignature for session,",
                    "         approve all safe for session [y], [n]o once, [d]eny all for session",
                    "Choice: ",
                ]
            )
            choice = self._prompt(prompt_fn, prompt)
        if choice in {"a", "approve"}:
            return ApprovalOutcome(approved=True, approval_mode="prompt")
        if choice in {"t", "tool"}:
            self.approved_tools.add(tool_name)
            return ApprovalOutcome(approved=True, approval_mode="prompt")
        if choice in {"s", "sig", "signature"} and signature:
            self.approved_signatures.add(signature)
            return ApprovalOutcome(approved=True, approval_mode="prompt")
        if choice in {"y", "all", "auto"}:
            self.mode = "auto_approve_safe"
            return ApprovalOutcome(approved=True, approval_mode="prompt")
        if choice in {"d", "deny"}:
            self.mode = "deny"
            return ApprovalOutcome(approved=False, approval_mode="denied", reason="session denied")
        return ApprovalOutcome(approved=False, approval_mode="denied", reason="denied by user")
