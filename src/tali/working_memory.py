from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from tali.tools.protocol import ToolResult


_TRUNCATION_SUFFIX = "...[truncated]"
_MAX_CONSTRAINTS_CHARS = 800
_MAX_ENV_FACTS_CHARS = 1200
_MAX_PROGRESS_CHARS = 240
_MAX_OBSERVATION_CHARS = 240


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    args_hash: str
    status: str
    ts: float


@dataclass
class WorkingMemory:
    user_goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    environment_facts: dict[str, Any] = field(default_factory=dict)
    progress: list[dict[str, Any]] = field(default_factory=list)
    last_observations: list[str] = field(default_factory=list)
    recent_tool_calls: deque[ToolCallRecord] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    steps_since_progress: int = 0

    def note_progress(self, kind: str, details: dict[str, Any]) -> None:
        self.progress.append({"kind": kind, "details": details, "ts": time.time()})
        self.steps_since_progress = 0

    def note_observations(
        self, obs: Iterable[str], durable_facts: dict[str, Any] | None
    ) -> None:
        self.last_observations = [item for item in obs if item]
        if durable_facts:
            self._merge_environment_facts(durable_facts)

    def record_tool_call(self, tool_name: str, args_hash: str, status: str) -> None:
        self.recent_tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name, args_hash=args_hash, status=status, ts=time.time()
            )
        )

    def seen_success(self, tool_name: str, args_hash: str) -> bool:
        return any(
            record.tool_name == tool_name
            and record.args_hash == args_hash
            and record.status == "ok"
            for record in self.recent_tool_calls
        )

    @staticmethod
    def args_hash(args: dict[str, Any]) -> str:
        try:
            encoded = json.dumps(
                args or {}, sort_keys=True, default=str, ensure_ascii=True
            ).encode("utf-8")
        except (TypeError, ValueError):
            encoded = repr(args).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def summary_for_prompt(self) -> str:
        recent = [
            f"- {record.tool_name} status={record.status}"
            for record in list(self.recent_tool_calls)[-5:]
        ]
        progress_tail = self.progress[-5:]
        env_facts = _json_compact(self.environment_facts, _MAX_ENV_FACTS_CHARS)
        constraints = _json_compact(self.constraints, _MAX_CONSTRAINTS_CHARS)
        progress_lines = [
            f"- {_json_compact(item, _MAX_PROGRESS_CHARS)}" for item in progress_tail
        ]
        observation_lines = [
            f"- {_truncate_text(item, _MAX_OBSERVATION_CHARS)}"
            for item in self.last_observations
        ]
        parts = [
            "Working memory:",
            f"user_goal: {self.user_goal or '-'}",
            f"constraints: {constraints}",
            f"environment_facts: {env_facts}",
            "progress (last 5):",
            "\n".join(progress_lines) or "- None",
            "last_observations:",
            "\n".join(observation_lines) or "- None",
            "recent_tool_calls:",
            "\n".join(recent) or "- None",
        ]
        return "\n".join(parts)

    def _merge_environment_facts(self, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if key.endswith("_paths") and isinstance(value, list):
                existing = self.environment_facts.get(key)
                if isinstance(existing, list):
                    merged = list(dict.fromkeys(existing + value))
                    self.environment_facts[key] = merged
                else:
                    self.environment_facts[key] = value
            else:
                self.environment_facts[key] = value

    def snapshot(self, max_progress: int = 10) -> dict[str, Any]:
        recent_calls = [
            {
                "tool_name": record.tool_name,
                "args_hash": record.args_hash,
                "status": record.status,
                "ts": record.ts,
            }
            for record in list(self.recent_tool_calls)
        ]
        return {
            "constraints": self.constraints,
            "environment_facts": self.environment_facts,
            "progress": self.progress[-max_progress:],
            "last_observations": self.last_observations,
            "recent_tool_calls": recent_calls,
            "steps_since_progress": self.steps_since_progress,
        }

    @staticmethod
    def from_snapshot(snapshot: dict[str, Any], user_goal: str) -> "WorkingMemory":
        memory = WorkingMemory(user_goal=user_goal)
        if not isinstance(snapshot, dict):
            return memory
        constraints = snapshot.get("constraints")
        if isinstance(constraints, dict):
            memory.constraints = constraints
        environment_facts = snapshot.get("environment_facts")
        if isinstance(environment_facts, dict):
            memory.environment_facts = environment_facts
        progress = snapshot.get("progress")
        if isinstance(progress, list):
            memory.progress = [item for item in progress if isinstance(item, dict)]
        last_observations = snapshot.get("last_observations")
        if isinstance(last_observations, list):
            memory.last_observations = [
                item for item in last_observations if isinstance(item, str)
            ]
        recent_tool_calls = snapshot.get("recent_tool_calls")
        if isinstance(recent_tool_calls, list):
            restored = deque(maxlen=10)
            for entry in recent_tool_calls:
                if not isinstance(entry, dict):
                    continue
                tool_name = entry.get("tool_name")
                args_hash = entry.get("args_hash")
                status = entry.get("status")
                ts = entry.get("ts")
                if not isinstance(tool_name, str) or not isinstance(args_hash, str):
                    continue
                if not isinstance(status, str):
                    continue
                restored.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args_hash=args_hash,
                        status=status,
                        ts=float(ts) if isinstance(ts, (int, float)) else time.time(),
                    )
                )
            memory.recent_tool_calls = restored
        steps_since_progress = snapshot.get("steps_since_progress")
        if isinstance(steps_since_progress, int):
            memory.steps_since_progress = steps_since_progress
        return memory


def summarize_tool_result(
    tool_name: str, args: dict[str, Any], result: ToolResult
) -> tuple[list[str], dict[str, Any]]:
    observations: list[str] = []
    durable: dict[str, Any] = {}
    summary = result.result_summary or ""
    raw = result.result_raw or ""

    if result.status != "ok":
        observations.append(
            f"{tool_name} failed: {result.status} {summary}".strip()
        )
        if "permission" in summary.lower() or "denied" in summary.lower():
            durable["permission_denied"] = True
        return observations[:7], durable

    observations.append(f"{tool_name} ok: {summary}".strip())

    if tool_name == "fs.list":
        path = _extract_path(summary, "Listed ")
        if path:
            observations.append(f"Listed path: {path}")
            durable.setdefault("listed_paths", []).append(path)
        entries = [line for line in raw.splitlines() if line.strip()]
        if entries:
            observations.append(f"Entries: {len(entries)}")
            if "Desktop" in entries:
                durable["desktop_dir"] = "Desktop"
    elif tool_name == "fs.read":
        path = _extract_path(summary, "Read ")
        if path:
            observations.append(f"Read file: {path}")
            durable.setdefault("read_paths", []).append(path)
    elif tool_name == "fs.stat":
        path = _extract_path(summary, "Stat ")
        if path:
            observations.append(f"Stat path: {path}")
            durable.setdefault("stat_paths", []).append(path)
    elif tool_name in {"fs.write", "fs.write_patch", "fs.copy", "fs.move"}:
        path = _extract_path(summary, _leading_verb(summary))
        if path:
            observations.append(f"Created/updated: {path}")
            durable.setdefault("written_paths", []).append(path)
    elif tool_name == "fs.delete":
        path = _extract_path(summary, "Deleted ")
        if path:
            observations.append(f"Deleted: {path}")
            durable.setdefault("deleted_paths", []).append(path)
    elif tool_name in {"web.fetch", "web.fetch_text"}:
        observations.append(f"Fetched URL: {summary}")
        # Store first 500 chars of content as observation
        if raw:
            preview = raw[:500]
            if len(raw) > 500:
                preview += "...[truncated]"
            observations.append(f"Content preview: {preview}")
        url = args.get("url", "")
        if url:
            durable.setdefault("fetched_urls", []).append(url)
    elif tool_name == "web.search":
        query = args.get("query", "")
        observations.append(f"Web search for: {query}")
        if raw and raw != "No results found.":
            # Store search results as observation
            result_lines = [line.strip() for line in raw.splitlines() if line.strip()][:5]
            observations.append(f"Results: {'; '.join(result_lines)}")
        if query:
            durable.setdefault("search_queries", []).append(query)
    elif tool_name == "python.eval":
        if "ok" in summary.lower():
            preview = raw[:500] if raw else "(no output)"
            if len(raw) > 500:
                preview += "...[truncated]"
            observations.append(f"Python output: {preview}")
        else:
            observations.append(f"Python error: {raw[:300]}" if raw else f"Python failed: {summary}")
    elif tool_name == "project.run_tests":
        observations.append(f"Test result: {summary}")
        if "PASSED" in summary:
            durable["last_test_result"] = "pass"
        elif "FAILED" in summary:
            durable["last_test_result"] = "fail"
        if raw:
            # Extract last few lines (usually the summary)
            tail_lines = raw.strip().splitlines()[-5:]
            observations.append(f"Test output tail: {'; '.join(tail_lines)}")
    elif tool_name == "fs.diff":
        observations.append(f"Diff result: {summary}")
        if raw and raw not in {"No uncommitted changes.", "Files are identical."}:
            # Count lines added/removed
            added = sum(1 for line in raw.splitlines() if line.startswith("+") and not line.startswith("+++"))
            removed = sum(1 for line in raw.splitlines() if line.startswith("-") and not line.startswith("---"))
            observations.append(f"Diff stats: +{added} -{removed} lines")
        elif raw:
            observations.append(raw)

    return observations[:7], durable


def is_stateful_progress(tool_name: str, result: ToolResult) -> bool:
    if result.status != "ok":
        return False
    return tool_name in {
        "fs.write",
        "fs.write_patch",
        "fs.copy",
        "fs.move",
        "fs.delete",
        "shell.run",
        "python.eval",
        "project.run_tests",
    }


def _extract_path(summary: str, prefix: str) -> str | None:
    if summary.startswith(prefix):
        return summary[len(prefix) :].strip()
    return None


def _leading_verb(summary: str) -> str:
    if not summary:
        return ""
    parts = summary.split(" ", 1)
    return f"{parts[0]} " if parts else ""


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:max_chars]
    return text[: max_chars - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _json_compact(value: Any, max_chars: int) -> str:
    try:
        rendered = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return _truncate_text(rendered, max_chars)
