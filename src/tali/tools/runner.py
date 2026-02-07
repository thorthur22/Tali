from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tali.approvals import ApprovalManager, ApprovalOutcome
from tali.config import Paths, ToolSettings
from tali.tools.policy import ToolPolicy
from tali.tools.protocol import ToolCall, ToolResult
from tali.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolRecord:
    id: str
    name: str
    args: dict[str, Any]
    status: str
    result_ref: str | None
    result_summary: str | None
    approval_mode: str | None
    risk_level: str | None
    started_at: str | None
    ended_at: str | None
    result_json: str | None
    result_hash: str | None
    result_path: str | None


class ToolRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy,
        approvals: ApprovalManager,
        settings: ToolSettings,
        paths: Paths,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.approvals = approvals
        self.settings = settings
        self.paths = paths
        self._cache: dict[str, tuple[str, str]] = {}

    def run(
        self, tool_calls: list[ToolCall], prompt_fn: Callable[[str], str]
    ) -> tuple[list[ToolResult], list[ToolRecord]]:
        tool_results: list[ToolResult] = []
        tool_records: list[ToolRecord] = []
        tool_counts: dict[str, int] = {}
        start_time = time.monotonic()

        for call in tool_calls:
            if len(tool_results) >= self.settings.max_tool_calls_per_turn:
                record = self._skipped_record(call, "tool budget exceeded")
                tool_records.append(record)
                tool_results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="skipped",
                        started_at="",
                        ended_at="",
                        result_ref=f"tool_call:{call.id}",
                        result_summary=record.result_summary or "",
                        result_raw="",
                    )
                )
                continue
            if time.monotonic() - start_time > self.settings.max_tool_seconds:
                record = self._skipped_record(call, "tool time budget exceeded")
                tool_records.append(record)
                tool_results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="skipped",
                        started_at="",
                        ended_at="",
                        result_ref=f"tool_call:{call.id}",
                        result_summary=record.result_summary or "",
                        result_raw="",
                    )
                )
                continue
            decision = self.policy.evaluate(call, tool_counts)
            if not decision.allowed:
                record = ToolRecord(
                    id=call.id,
                    name=call.name,
                    args=call.args,
                    status="denied",
                    result_ref=None,
                    result_summary=decision.reason,
                    approval_mode="denied",
                    risk_level=decision.risk_level,
                    started_at=None,
                    ended_at=None,
                    result_json=None,
                    result_hash=None,
                    result_path=None,
                )
                tool_records.append(record)
                tool_results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="denied",
                        started_at="",
                        ended_at="",
                        result_ref=f"tool_call:{call.id}",
                        result_summary=decision.reason or "",
                        result_raw="",
                    )
                )
                continue
            cache_key = self._cache_key(call)
            if cache_key and cache_key in self._cache:
                summary, raw = self._cache[cache_key]
                result_ref = f"tool_cache:{call.id}"
                result_json, result_hash, result_path = self._store_result(call.id, summary, raw)
                tool_records.append(
                    ToolRecord(
                        id=call.id,
                        name=call.name,
                        args=call.args,
                        status="cached",
                        result_ref=result_ref,
                        result_summary=summary,
                        approval_mode="cached",
                        risk_level=decision.risk_level,
                        started_at=None,
                        ended_at=None,
                        result_json=result_json,
                        result_hash=result_hash,
                        result_path=result_path,
                    )
                )
                tool_results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="ok",
                        started_at="",
                        ended_at="",
                        result_ref=result_ref,
                        result_summary=f"[cached] {summary}",
                        result_raw=raw,
                    )
                )
                tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
                continue
            details = self._format_approval_details(call)
            approval: ApprovalOutcome = self.approvals.resolve(
                prompt_fn=prompt_fn,
                tool_name=call.name,
                signature=decision.signature,
                requires_approval=decision.requires_approval,
                reason="; ".join(decision.red_flags) if decision.red_flags else None,
                details=details,
            )
            if not approval.approved:
                record = ToolRecord(
                    id=call.id,
                    name=call.name,
                    args=call.args,
                    status="denied",
                    result_ref=None,
                    result_summary=approval.reason,
                    approval_mode="denied",
                    risk_level=decision.risk_level,
                    started_at=None,
                    ended_at=None,
                    result_json=None,
                    result_hash=None,
                    result_path=None,
                )
                tool_records.append(record)
                tool_results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="denied",
                        started_at="",
                        ended_at="",
                        result_ref=f"tool_call:{call.id}",
                        result_summary=approval.reason or "",
                        result_raw="",
                    )
                )
                continue

            tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
            started_at = datetime.utcnow().isoformat()
            status = "ok"
            summary = ""
            raw = ""
            try:
                if call.name == "python.eval":
                    summary, raw = self._run_python_eval(call.args["code"])
                else:
                    definition = self.registry.get(call.name)
                    if definition is None:
                        raise ValueError("tool not found")
                    summary, raw = definition.handle(call.args)
            except Exception as exc:
                status = "error"
                summary = f"{exc}"
                raw = ""
            ended_at = datetime.utcnow().isoformat()
            result_ref = f"tool_call:{call.id}"
            result_json, result_hash, result_path = self._store_result(call.id, summary, raw)
            tool_records.append(
                ToolRecord(
                    id=call.id,
                    name=call.name,
                    args=call.args,
                    status=status,
                    result_ref=result_ref,
                    result_summary=summary,
                    approval_mode=approval.approval_mode,
                    risk_level=decision.risk_level,
                    started_at=started_at,
                    ended_at=ended_at,
                    result_json=result_json,
                    result_hash=result_hash,
                    result_path=result_path,
                )
            )
            tool_results.append(
                ToolResult(
                    id=call.id,
                    name=call.name,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    result_ref=result_ref,
                    result_summary=summary,
                    result_raw=raw,
                )
            )
            if status == "ok" and cache_key:
                self._cache[cache_key] = (summary, raw)
        return tool_results, tool_records

    def _cache_key(self, call: ToolCall) -> str | None:
        definition = self.registry.get(call.name)
        if not definition:
            return None
        signature = definition.signature(call.args)
        if not signature:
            return None
        return f"{call.name}:{signature}"

    def _format_approval_details(self, call: ToolCall) -> str:
        purpose = call.purpose or ""
        try:
            args_text = json.dumps(call.args, indent=2)
        except (TypeError, ValueError):
            args_text = str(call.args)
        lines = [
            f"Purpose: {purpose}" if purpose else "Purpose: (none provided)",
            "Args:",
            args_text,
        ]
        return "\n".join(lines)

    def _skipped_record(self, call: ToolCall, reason: str) -> ToolRecord:
        return ToolRecord(
            id=call.id,
            name=call.name,
            args=call.args,
            status="skipped",
            result_ref=None,
            result_summary=reason,
            approval_mode="skipped",
            risk_level=None,
            started_at=None,
            ended_at=None,
            result_json=None,
            result_hash=None,
            result_path=None,
        )

    def _store_result(
        self, call_id: str, summary: str, raw: str
    ) -> tuple[str | None, str | None, str | None]:
        payload = {"summary": summary, "raw": raw}
        result_json = json.dumps(payload)
        result_hash = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        if len(result_json) <= self.settings.tool_result_max_bytes:
            return result_json, result_hash, None
        self.paths.tool_results_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.tool_results_dir / f"{call_id}.json"
        path.write_text(result_json, encoding="utf-8")
        truncated = json.dumps({"summary": summary, "raw": "[truncated]"})
        return truncated, result_hash, str(path)

    def _run_python_eval(self, code: str) -> tuple[str, str]:
        if not self.settings.python_enabled:
            raise ValueError("python.eval is disabled")

        def worker(code_text: str, queue: multiprocessing.Queue) -> None:
            import io
            import traceback
            from contextlib import redirect_stderr, redirect_stdout

            safe_builtins = {
                "print": print,
                "len": len,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
            }
            globals_dict = {"__builtins__": safe_builtins}
            locals_dict: dict[str, Any] = {}
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        compiled = compile(code_text, "<python.eval>", "eval")
                        result = eval(compiled, globals_dict, locals_dict)
                        if result is not None:
                            print(result)
                    except SyntaxError:
                        compiled = compile(code_text, "<python.eval>", "exec")
                        exec(compiled, globals_dict, locals_dict)
                queue.put({"ok": True, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()})
            except Exception:
                queue.put(
                    {
                        "ok": False,
                        "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue() + traceback.format_exc(),
                    }
                )

        context = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = context.Queue()
        process = context.Process(target=worker, args=(code, queue))
        process.start()
        process.join(timeout=self.settings.python_timeout_s)
        if process.is_alive():
            process.terminate()
            process.join()
            return "python eval timed out", ""
        if queue.empty():
            return "python eval failed", ""
        result = queue.get()
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        ok = bool(result.get("ok"))
        summary = "python eval ok" if ok else "python eval error"
        raw = stdout + stderr
        return summary, raw
