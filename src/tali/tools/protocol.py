from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tali.models import RetrievalBundle
from tali.prompting import format_retrieval_context
from tali.prompts import SYSTEM_RULES_WITH_PATCH


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    purpose: str | None = None


@dataclass(frozen=True)
class Phase1Plan:
    need_tools: bool
    final_answer_allowed: bool
    tool_calls: list[ToolCall]
    stop_reason: str | None = None


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    status: str
    started_at: str
    ended_at: str
    result_ref: str
    result_summary: str
    result_raw: str


def _validate_tool_call(raw: Any) -> ToolCall | None:
    if not isinstance(raw, dict):
        return None
    call_id = raw.get("id")
    name = raw.get("name")
    args = raw.get("args")
    if not isinstance(call_id, str) or not call_id.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(args, dict):
        return None
    purpose = raw.get("purpose")
    if purpose is not None and not isinstance(purpose, str):
        return None
    return ToolCall(id=call_id, name=name, args=args, purpose=purpose)


def parse_phase1_plan(text: str) -> tuple[Phase1Plan | None, str | None]:
    raw_text = text.strip()
    if not raw_text:
        return None, "empty response"
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "plan must be a json object"
    need_tools = payload.get("need_tools")
    final_answer_allowed = payload.get("final_answer_allowed")
    tool_calls_raw = payload.get("tool_calls")
    if not isinstance(need_tools, bool):
        return None, "need_tools must be boolean"
    if not isinstance(final_answer_allowed, bool):
        return None, "final_answer_allowed must be boolean"
    if not isinstance(tool_calls_raw, list):
        return None, "tool_calls must be a list"
    tool_calls: list[ToolCall] = []
    for entry in tool_calls_raw:
        call = _validate_tool_call(entry)
        if call is None:
            return None, "invalid tool call entry"
        tool_calls.append(call)
    if not need_tools and tool_calls:
        return None, "tool_calls must be empty when need_tools is false"
    stop_reason = payload.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        return None, "stop_reason must be a string"
    return Phase1Plan(
        need_tools=need_tools,
        final_answer_allowed=final_answer_allowed,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
    ), None


def tool_plan_instructions(tool_descriptions: str) -> str:
    return "\n".join(
        [
            "You are a tool planner. Return STRICT JSON ONLY.",
            "Required JSON schema:",
            '{ "need_tools": true|false, "final_answer_allowed": true|false, "tool_calls": [ ... ], "stop_reason": "optional" }',
            "Each tool call must include:",
            '{ "id": "tc_1", "name": "tool.name", "args": { ... }, "purpose": "optional" }',
            "If need_tools is false, tool_calls MUST be empty.",
            "If the user requests actions that require tools (files, shell, web, or python), set need_tools=true.",
            "If you need tools, set final_answer_allowed=false and only output the JSON tool plan.",
            "You have access to the tools listed below. Do NOT claim you lack tool access.",
            "For file creation requests, plan an fs.write call and provide the content yourself.",
            "Do not include any prose or markdown.",
            "",
            "Available tools:",
            tool_descriptions,
        ]
    )


def format_tool_results(tool_results: list[ToolResult]) -> str:
    if not tool_results:
        return "- None"
    lines: list[str] = []
    for result in tool_results:
        lines.append(
            f"- {result.id} {result.name} status={result.status} ref={result.result_ref} summary={result.result_summary}"
        )
    return "\n".join(lines)


def build_phase1_prompt(bundle: RetrievalBundle, user_input: str, tool_descriptions: str) -> str:
    parts: list[str] = []
    parts.append(SYSTEM_RULES_WITH_PATCH)
    parts.append(format_retrieval_context(bundle))
    parts.append("\nUser:")
    parts.append(user_input)
    parts.append("\n[Tool Planning]")
    parts.append(tool_plan_instructions(tool_descriptions))
    return "\n".join(parts)


def build_phase2_prompt(
    bundle: RetrievalBundle, user_input: str, tool_results: list[ToolResult], raw_tool_results: str
) -> str:
    parts: list[str] = []
    parts.append(SYSTEM_RULES_WITH_PATCH)
    parts.append("\n[Tool Results]")
    parts.append("Use tool results only if status=ok. Treat them as ground truth evidence.")
    parts.append("If a tool ran successfully, do NOT claim you lack tool access.")
    parts.append("Acknowledge completed actions and proceed with the final response.")
    parts.append(format_tool_results(tool_results))
    if raw_tool_results:
        parts.append("\n[Tool Outputs]")
        parts.append(raw_tool_results)
    parts.append(format_retrieval_context(bundle))
    parts.append("\nUser:")
    parts.append(user_input)
    return "\n".join(parts)
