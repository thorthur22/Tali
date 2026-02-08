from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tali.prompts import HOLISTIC_PROMPT_PATCH


@dataclass(frozen=True)
class TaskSpec:
    title: str
    description: str
    requires_tools: bool
    verification: str
    dependencies: list[int]


@dataclass(frozen=True)
class ActionPlan:
    next_action_type: str
    message: str | None
    tool_name: str | None
    tool_args: dict[str, Any] | None
    tool_purpose: str | None
    outputs_json: dict[str, Any] | None
    block_reason: str | None
    delegate_to: str | None
    delegate_task: dict[str, Any] | None
    skill_name: str | None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ReviewResult:
    overall_status: str
    checks: list[dict[str, Any]]
    missing_items: list[str]
    assumptions: list[str]
    user_message: str


def build_swarm_context(a2a_client: Any, self_name: str) -> str:
    """Build a description of peer agents from the shared registry.

    Returns an empty string when no A2A client is available or no peers exist.
    """
    if not a2a_client:
        return ""
    try:
        agents = a2a_client.registry.list_agents()
    except Exception:  # noqa: BLE001
        return ""
    peers = [a for a in agents if a.get("agent_name") != self_name]
    if not peers:
        return "No other agents are currently registered in the swarm."
    lines = ["Other agents available for delegation:"]
    for a in peers:
        caps = ", ".join(a.get("capabilities", [])) or "general"
        status = a.get("status", "unknown")
        lines.append(f'- Agent "{a["agent_name"]}": capabilities=[{caps}], status={status}')
    return "\n".join(lines)


def build_decomposition_prompt(
    user_prompt: str,
    tool_descriptions: str,
    agent_context: str,
    memory_context: str,
) -> str:
    return "\n".join(
        [
            "You are an execution planner. Return STRICT JSON ONLY. No reasoning.",
            "You are planning for the Tali agent, which executes tools on your behalf.",
            HOLISTIC_PROMPT_PATCH,
            "Do NOT claim you lack tool access; request tools when needed.",
            "Decompose the user request into specific, actionable, verifiable tasks.",
            "Prefer 3-12 tasks; if the request is tiny, allow 1-3 tasks.",
            "If the request is ambiguous, include an early task to clarify with the user.",
            "Include dependencies as indexes of earlier tasks.",
            "Do NOT use tools for greetings or small talk.",
            "Set requires_tools=true for any task needing files, shell, web, or python tools.",
            "Respect preferences from memory context as constraints.",
            "Some tasks may be delegated to other agents in the swarm during execution if better-suited agents are available. Plan tasks at a level where delegation boundaries are natural.",
            "Required JSON schema:",
            "{",
            '  "tasks": [',
            "    {",
            '      "title": "...",',
            '      "description": "...",',
            '      "requires_tools": true/false,',
            '      "verification": "how to know it is done",',
            '      "dependencies": [0,1]',
            "    }",
            "  ]",
            "}",
            "Available tools:",
            tool_descriptions,
            "Memory context:",
            memory_context or "- None",
            "Agent context:",
            agent_context,
            "User request:",
            user_prompt,
        ]
    )


def parse_decomposition(text: str) -> tuple[list[TaskSpec] | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "payload must be object"
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list):
        return None, "tasks must be list"
    tasks: list[TaskSpec] = []
    for idx, entry in enumerate(tasks_raw):
        if not isinstance(entry, dict):
            return None, f"task {idx} must be object"
        title = entry.get("title")
        description = entry.get("description")
        requires_tools = entry.get("requires_tools")
        verification = entry.get("verification")
        dependencies = entry.get("dependencies", [])
        if not isinstance(title, str) or not title.strip():
            return None, f"task {idx} title missing"
        if not isinstance(description, str):
            return None, f"task {idx} description missing"
        if not isinstance(requires_tools, bool):
            return None, f"task {idx} requires_tools invalid"
        if not isinstance(verification, str):
            return None, f"task {idx} verification missing"
        if not isinstance(dependencies, list) or any(not isinstance(d, int) for d in dependencies):
            return None, f"task {idx} dependencies invalid"
        if any(d >= idx or d < 0 for d in dependencies):
            return None, f"task {idx} dependencies must reference earlier tasks"
        tasks.append(
            TaskSpec(
                title=title.strip(),
                description=description.strip(),
                requires_tools=requires_tools,
                verification=verification.strip(),
                dependencies=dependencies,
            )
        )
    if not tasks:
        return None, "tasks must not be empty"
    if len(tasks) > 12:
        return None, "too many tasks"
    return tasks, None


def build_action_plan_prompt(
    user_prompt: str,
    task_title: str,
    task_description: str,
    task_inputs_json: str | None,
    task_outputs_json: str | None,
    tool_descriptions: str,
    recent_tool_summaries: list[str],
    recent_tool_outputs: list[str],
    agent_context: str,
    memory_context: str,
    working_memory_summary: str,
    run_summary: str | None,
    stuck_context: str | None,
    skill_context: str | None,
    responder_strengths: str | None = None,
) -> str:
    clarifications = ""
    if run_summary:
        for line in run_summary.splitlines():
            if line.strip().lower().startswith("clarifications:"):
                clarifications = line.split(":", 1)[1].strip()
                break
    parts = [
        "You are the task action planner. Return STRICT JSON ONLY. No reasoning.",
        HOLISTIC_PROMPT_PATCH,
        "You are planning for the Tali agent, which executes tools on your behalf.",
        "Do NOT claim you lack tool access; request tools when needed.",
        "Choose the single next action for this task.",
        "Valid next_action_type values:",
        '"respond", "tool_call", "ask_user", "store_output", "mark_done", "block", "fail", "delegate", "execute_skill"',
        "Delegation rules:",
        '- If another agent is better suited for this task (see Team resources in Agent context), use next_action_type="delegate".',
        '- Set delegate_to to the agent\'s name for a specific agent, or to a capability keyword (e.g. "coding") to auto-select an agent with that skill.',
        "- Provide delegate_task with title, description, inputs (dict), and requested_outputs (list of strings).",
        "- After delegating, the task will block until the other agent responds.",
        "- Only delegate when another agent has a clearly relevant capability you lack; prefer doing the work yourself when possible.",
        f"When choosing 'respond', note that the response model excels at: {responder_strengths or 'general conversation'}. Provide detailed factual content and citations in your message so the responder can phrase it well.",
        "If asking the user, keep it to ONE minimal question.",
        "If using a tool, provide tool_name and tool_args.",
        "If executing a skill, provide skill_name and use the skill steps to guide actions.",
        "Always include tool_args as an object; use {} when a tool takes no args.",
        "When work requires files, shell, web, or python, choose tool_call.",
        "If storing outputs, provide outputs_json as an object.",
        "Use recent tool outputs to decide next steps; do NOT repeat a tool call if the output already provides the needed info.",
        "Use last_observations + environment_facts to select the next action.",
        "Do not request tools that are blocked as duplicates.",
        "If stuck signals indicate repeated tool calls, pick a different strategy or tool before asking the user.",
        "Do NOT ask the user for info already answered in Clarifications.",
        "If you reference memory in message, cite it with [fact:ID], [commitment:ID], [preference:KEY], or [episode:ID].",
        "Only use tool_name values that appear in Available tools; never invent tools.",
        "For fs.list, omit the path to list the root; never pass an empty string path.",
        "Avoid shell.run unless using allowed read-only commands (git status/diff/log, ls/dir, cat/type, grep/rg, find, wc, head/tail, which/where, python --version, pip list/show, tree).",
        "To find the Desktop, use fs.list with {} to list the root and locate 'Desktop'. If no Desktop entry exists, ask the user for a path.",
        "Use the OS info from agent context to avoid OS-specific assumptions.",
        "JSON schema:",
        "{",
        '  "next_action_type": "respond|tool_call|ask_user|store_output|mark_done|block|fail|delegate|execute_skill",',
        '  "message": "optional user-visible message",',
        '  "tool_name": "optional",',
        '  "tool_args": { ... },',
        '  "tool_purpose": "optional",',
        '  "outputs_json": { ... },',
        '  "block_reason": "optional",',
        '  "delegate_to": "optional agent name",',
        '  "delegate_task": { "title": "...", "description": "...", "inputs": {...}, "requested_outputs": ["..."] },',
        '  "skill_name": "optional skill name",',
        '  "tool_calls": [{"tool_name": "...", "tool_args": {...}, "tool_purpose": "..."}]  // optional: batch multiple independent tool calls in one step',
        "}",
        'For batched tool calls, use "tool_calls" array instead of single tool_name/tool_args. Use this when multiple independent reads are needed.',
        'Example action (for reference only, do not copy verbatim):',
        '{"next_action_type":"tool_call","tool_name":"fs.search","tool_args":{"query":"TODO"},"tool_purpose":"Find all files containing TODO markers"}',
        "User request:",
        user_prompt,
        "Task:",
        f"Title: {task_title}",
        f"Description: {task_description}",
        f"Inputs JSON: {task_inputs_json or ''}",
        f"Outputs JSON: {task_outputs_json or ''}",
        "Recent tool results:",
        "\n".join(recent_tool_summaries) if recent_tool_summaries else "- None",
        "Recent tool outputs:",
        "\n".join(recent_tool_outputs) if recent_tool_outputs else "- None",
        "Available tools:",
        tool_descriptions,
        "Memory context:",
        memory_context or "- None",
        working_memory_summary or "- None",
        "Run summary:",
        run_summary or "- None",
        "Clarifications:",
        clarifications or "- None",
        "Stuck signals:",
        stuck_context or "- None",
        "Skill details:",
        skill_context or "- None",
        "Agent context:",
        agent_context,
    ]
    return "\n".join(parts)


def build_response_prompt(
    user_prompt: str,
    memory_context: str,
    working_memory_summary: str,
    agent_context: str,
    planner_message: str | None = None,
    recent_tool_summaries: list[str] | None = None,
    recent_tool_outputs: list[str] | None = None,
    run_summary: str | None = None,
) -> str:
    parts = [
        "You are the response model. Be concise, direct, and helpful.",
        HOLISTIC_PROMPT_PATCH,
        "Do not describe internal planning or tools unless asked.",
        "If you reference memory, cite it with [fact:ID], [commitment:ID], [preference:KEY], or [episode:ID].",
        "Memory context:",
        memory_context or "- None",
        working_memory_summary or "- None",
        "Run summary:",
        run_summary or "- None",
        "Recent tool results:",
        "\n".join(recent_tool_summaries or []) or "- None",
        "Recent tool outputs:",
        "\n".join(recent_tool_outputs or []) or "- None",
        "Planner notes:",
        planner_message or "- None",
        "Agent context:",
        agent_context,
        "User request:",
        user_prompt,
    ]
    return "\n".join(parts)


def parse_action_plan(text: str) -> tuple[ActionPlan | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "payload must be object"
    next_action_type = payload.get("next_action_type")
    if not isinstance(next_action_type, str):
        return None, "next_action_type required"
    message = payload.get("message")
    if message is not None and not isinstance(message, str):
        return None, "message must be string"
    tool_name = payload.get("tool_name")
    tool_args = payload.get("tool_args")
    tool_purpose = payload.get("tool_purpose")
    outputs_json = payload.get("outputs_json")
    block_reason = payload.get("block_reason")
    delegate_to = payload.get("delegate_to")
    delegate_task = payload.get("delegate_task")
    skill_name = payload.get("skill_name")
    if tool_name is not None and not isinstance(tool_name, str):
        return None, "tool_name must be string"
    if tool_args is not None and not isinstance(tool_args, dict):
        return None, "tool_args must be object"
    if tool_name == "fs_root":
        tool_name = "fs.list"
        tool_args = {}
    if tool_purpose is not None and not isinstance(tool_purpose, str):
        return None, "tool_purpose must be string"
    if outputs_json is not None and not isinstance(outputs_json, dict):
        return None, "outputs_json must be object"
    if block_reason is not None and not isinstance(block_reason, str):
        return None, "block_reason must be string"
    if delegate_to is not None and not isinstance(delegate_to, str):
        return None, "delegate_to must be string"
    if delegate_task is not None and not isinstance(delegate_task, dict):
        return None, "delegate_task must be object"
    if skill_name is not None and not isinstance(skill_name, str):
        return None, "skill_name must be string"
    # Parse optional batched tool_calls
    tool_calls_raw = payload.get("tool_calls")
    tool_calls: list[dict[str, Any]] | None = None
    if tool_calls_raw is not None:
        if not isinstance(tool_calls_raw, list):
            return None, "tool_calls must be array"
        tool_calls = []
        for idx, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                return None, f"tool_calls[{idx}] must be object"
            tc_name = tc.get("tool_name")
            tc_args = tc.get("tool_args")
            if not isinstance(tc_name, str):
                return None, f"tool_calls[{idx}].tool_name must be string"
            if tc_args is not None and not isinstance(tc_args, dict):
                return None, f"tool_calls[{idx}].tool_args must be object"
            tool_calls.append({
                "tool_name": tc_name,
                "tool_args": tc_args or {},
                "tool_purpose": tc.get("tool_purpose", ""),
            })
    return (
        ActionPlan(
            next_action_type=next_action_type,
            message=message,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_purpose=tool_purpose,
            outputs_json=outputs_json,
            block_reason=block_reason,
            delegate_to=delegate_to,
            delegate_task=delegate_task,
            skill_name=skill_name,
            tool_calls=tool_calls,
        ),
        None,
    )


def build_completion_review_prompt(
    user_prompt: str, task_summaries: list[dict[str, Any]], memory_context: str
) -> str:
    return "\n".join(
        [
            "You are the completion reviewer. Return STRICT JSON ONLY. No reasoning.",
            HOLISTIC_PROMPT_PATCH,
            "Validate whether the tasks fully satisfy the user request.",
            "For each task, compare the 'verification' field in inputs_json against the actual outputs in outputs_json. Mark 'missing' if the verification criteria are not demonstrably met by the outputs.",
            "If you reference memory in user_message, cite it with [fact:ID], [commitment:ID], [preference:KEY], or [episode:ID].",
            "JSON schema:",
            "{",
            '  "overall_status": "complete|incomplete",',
            '  "checks": [ {"task_ordinal": 0, "status": "ok|missing", "note": "..."} ],',
            '  "missing_items": ["..."],',
            '  "assumptions": ["..."],',
            '  "user_message": "concise final response to user"',
            "}",
            "Memory context:",
            memory_context or "- None",
            "User request:",
            user_prompt,
            "Task summaries:",
            json.dumps(task_summaries, indent=2),
        ]
    )


def parse_completion_review(text: str) -> tuple[ReviewResult | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "payload must be object"
    overall_status = payload.get("overall_status")
    checks = payload.get("checks", [])
    missing_items = payload.get("missing_items", [])
    assumptions = payload.get("assumptions", [])
    user_message = payload.get("user_message")
    if overall_status not in {"complete", "incomplete"}:
        return None, "overall_status invalid"
    if not isinstance(checks, list):
        return None, "checks must be list"
    if not isinstance(missing_items, list) or any(not isinstance(i, str) for i in missing_items):
        return None, "missing_items invalid"
    if not isinstance(assumptions, list) or any(not isinstance(i, str) for i in assumptions):
        return None, "assumptions invalid"
    if not isinstance(user_message, str) or not user_message.strip():
        return None, "user_message required"
    return (
        ReviewResult(
            overall_status=overall_status,
            checks=checks,
            missing_items=missing_items,
            assumptions=assumptions,
            user_message=user_message.strip(),
        ),
        None,
    )
