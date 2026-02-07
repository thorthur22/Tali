from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from tali.tools.protocol import ToolResult


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
        encoded = json.dumps(args or {}, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def summary_for_prompt(self) -> str:
        recent = [
            f"- {record.tool_name} status={record.status}"
            for record in list(self.recent_tool_calls)[-5:]
        ]
        progress_tail = self.progress[-5:]
        env_facts = json.dumps(self.environment_facts, sort_keys=True)
        parts = [
            "Working memory:",
            f"user_goal: {self.user_goal or '-'}",
            f"constraints: {json.dumps(self.constraints, sort_keys=True)}",
            f"environment_facts: {env_facts}",
            "progress (last 5):",
            "\n".join(f"- {item}" for item in progress_tail) or "- None",
            "last_observations:",
            "\n".join(f"- {item}" for item in self.last_observations) or "- None",
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
