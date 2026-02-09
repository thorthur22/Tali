from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReflectionResult:
    success: bool
    what_worked: str
    what_failed: str
    next_time: str
    metrics: dict[str, Any]
    memory_candidates: list[str]
    preference_candidates: dict[str, str]


def build_reflection_prompt(
    user_prompt: str,
    task_summaries: list[dict[str, Any]],
    memory_context: str,
    outcome: str,
) -> str:
    # Keep prompt compact—local models can be token-limited.
    trimmed_mem = memory_context
    if len(trimmed_mem) > 3500:
        trimmed_mem = trimmed_mem[:3500] + "\n...[truncated]"

    tasks_json = json.dumps(task_summaries, ensure_ascii=False)
    return "\n".join(
        [
            "You are the agent's self-reflection module.",
            "Analyze the latest run and produce concise, actionable learning.",
            "Return STRICT JSON only. No markdown.",
            "Schema:",
            "{\"success\":true|false,\"what_worked\":str,\"what_failed\":str,\"next_time\":str,",
            " \"metrics\":{...},\"memory_candidates\":[str,...],\"preference_candidates\":{\"key\":\"value\",...}}",
            "Guidelines:",
            "- what_worked/what_failed/next_time: <= 500 chars each.",
            "- memory_candidates: facts that are generally useful and stable (short, declarative).",
            "- preference_candidates: only if user expressed a stable preference.",
            "- metrics: include any counters you can infer (e.g., tool_calls, retries, blockers).",
            f"Outcome label: {outcome}",
            "\n[User prompt]",
            user_prompt,
            "\n[Tasks JSON]",
            tasks_json,
            "\n[Retrieved memory context]",
            trimmed_mem,
        ]
    )


def parse_reflection_response(text: str) -> ReflectionResult | None:
    raw = text.strip()
    # Try to extract JSON object if model wrapped it.
    if not raw.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    success = bool(payload.get("success"))
    what_worked = str(payload.get("what_worked") or "").strip()
    what_failed = str(payload.get("what_failed") or "").strip()
    next_time = str(payload.get("next_time") or "").strip()
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    memory_candidates = payload.get("memory_candidates")
    if not isinstance(memory_candidates, list):
        memory_candidates = []
    memory_candidates = [str(x).strip() for x in memory_candidates if str(x).strip()]
    pref_candidates = payload.get("preference_candidates")
    if not isinstance(pref_candidates, dict):
        pref_candidates = {}
    # Ensure string values
    pref_candidates = {
        str(k).strip(): str(v).strip()
        for k, v in pref_candidates.items()
        if str(k).strip() and str(v).strip()
    }
    return ReflectionResult(
        success=success,
        what_worked=what_worked,
        what_failed=what_failed,
        next_time=next_time,
        metrics=metrics,
        memory_candidates=memory_candidates,
        preference_candidates=pref_candidates,
    )
