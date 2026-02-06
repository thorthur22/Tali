from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from tali.db import Database
from tali.guardrails import Guardrails
from tali.hooks import HookManager
from tali.a2a import A2AClient
from tali.llm import LLMClient
from tali.retrieval import Retriever
from tali.tasking import (
    ActionPlan,
    TaskSpec,
    ReviewResult,
    build_action_plan_prompt,
    build_completion_review_prompt,
    build_decomposition_prompt,
    parse_action_plan,
    parse_completion_review,
    parse_decomposition,
)
from tali.tools.protocol import ToolCall, ToolResult
from tali.tools.runner import ToolRecord, ToolRunner


@dataclass(frozen=True)
class TaskRunnerSettings:
    max_tasks_per_turn: int = 5
    max_llm_calls_per_task: int = 3
    max_tool_calls_per_task: int = 5
    max_total_llm_calls_per_run_per_turn: int = 10
    max_total_steps_per_turn: int = 30


@dataclass
class TaskRunnerResult:
    message: str
    tool_records: list[ToolRecord]
    tool_results: list[ToolResult]
    run_id: str | None


class TaskRunner:
    def __init__(
        self,
        db: Database,
        llm: LLMClient,
        retriever: Retriever,
        guardrails: Guardrails,
        tool_runner: ToolRunner,
        tool_descriptions: str,
        hook_manager: HookManager | None = None,
        a2a_client: A2AClient | None = None,
        settings: TaskRunnerSettings | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.retriever = retriever
        self.guardrails = guardrails
        self.tool_runner = tool_runner
        self.tool_descriptions = tool_descriptions
        self.hook_manager = hook_manager
        self.a2a_client = a2a_client
        self.settings = settings or TaskRunnerSettings()

    def run_turn(
        self,
        user_input: str,
        prompt_fn: Callable[[str], str],
    ) -> TaskRunnerResult:
        retrieval_context = self.retriever.retrieve(user_input)
        active_run = self.db.fetch_active_run()
        tool_records: list[ToolRecord] = []
        tool_results: list[ToolResult] = []
        messages: list[str] = []
        llm_calls = 0
        steps = 0
        run_id: str | None = None
        created_run = False

        if active_run is None:
            created_run = True
            run_id = str(uuid.uuid4())
            self.db.insert_run(
                run_id=run_id,
                created_at=datetime.utcnow().isoformat(),
                status="active",
                user_prompt=user_input,
                current_task_id=None,
                last_error=None,
            )
            tasks, error = self._decompose_and_persist(run_id, user_input)
            llm_calls += 1
            if error or not tasks:
                self.db.update_run_status(run_id, status="blocked", last_error=error or "decomposition_failed")
                question = "I couldn't break this into tasks. Could you rephrase or clarify the request?"
                guarded = self.guardrails.enforce(question, retrieval_context.bundle)
                return TaskRunnerResult(
                    message=guarded.safe_output, tool_records=[], tool_results=[], run_id=run_id
                )
            messages.append(self._format_plan(tasks))
        else:
            run_id = str(active_run["id"])
            if active_run["status"] == "blocked":
                self.db.update_run_status(run_id, status="active", last_error=None)

        assert run_id is not None
        tasks_rows = self.db.fetch_tasks_for_run(run_id)
        if not tasks_rows:
            message = "No tasks found for this run."
            guarded = self.guardrails.enforce(message, retrieval_context.bundle)
            return TaskRunnerResult(
                message=guarded.safe_output, tool_records=[], tool_results=[], run_id=run_id
            )

        tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}
        completed_this_turn = 0
        while (
            completed_this_turn < self.settings.max_tasks_per_turn
            and steps < self.settings.max_total_steps_per_turn
            and llm_calls < self.settings.max_total_llm_calls_per_run_per_turn
        ):
            next_task = self._select_next_task(tasks_rows, tasks_by_ordinal)
            if next_task is None:
                break
            task_result = self._run_task(
                run_id=run_id,
                task_row=next_task,
                user_prompt=str(self.db.fetch_run(run_id)["user_prompt"]),
                prompt_fn=prompt_fn,
                llm_calls_remaining=self.settings.max_total_llm_calls_per_run_per_turn - llm_calls,
                steps_remaining=self.settings.max_total_steps_per_turn - steps,
            )
            llm_calls += task_result.llm_calls
            steps += task_result.steps
            tool_records.extend(task_result.tool_records)
            tool_results.extend(task_result.tool_results)
            if task_result.user_message:
                messages.append(task_result.user_message)
            if task_result.blocked:
                self.db.update_run_status(run_id, status="blocked", current_task_id=next_task["id"])
                break
            completed_this_turn += 1
            tasks_rows = self.db.fetch_tasks_for_run(run_id)
            tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}

        if self._all_tasks_complete(tasks_rows):
            review_message, review_llm_calls, review_steps = self._completion_review(
                run_id=run_id,
                user_prompt=str(self.db.fetch_run(run_id)["user_prompt"]),
                tasks_rows=tasks_rows,
            )
            llm_calls += review_llm_calls
            steps += review_steps
            if review_message:
                messages.append(review_message)
        else:
            if steps >= self.settings.max_total_steps_per_turn or llm_calls >= self.settings.max_total_llm_calls_per_run_per_turn:
                messages.append("Progress saved. Continuing next turn.")

        final_message = "\n\n".join(msg for msg in messages if msg.strip())
        if not final_message:
            final_message = "Working on it."
        guarded = self.guardrails.enforce(final_message, retrieval_context.bundle)
        return TaskRunnerResult(
            message=guarded.safe_output,
            tool_records=tool_records,
            tool_results=tool_results,
            run_id=run_id,
        )

    def _decompose_and_persist(
        self, run_id: str, user_prompt: str
    ) -> tuple[list[TaskSpec] | None, str | None]:
        prompt = build_decomposition_prompt(user_prompt, self.tool_descriptions)
        response = self.llm.generate(prompt)
        tasks, error = parse_decomposition(response.content)
        if error or not tasks:
            return None, error
        now = datetime.utcnow().isoformat()
        for ordinal, task in enumerate(tasks):
            inputs_json = json.dumps(
                {"verification": task.verification, "dependencies": task.dependencies}
            )
            self.db.insert_task(
                task_id=str(uuid.uuid4()),
                run_id=run_id,
                parent_task_id=None,
                ordinal=ordinal,
                title=task.title,
                description=task.description,
                status="pending",
                inputs_json=inputs_json,
                outputs_json=None,
                requires_tools=1 if task.requires_tools else 0,
                created_at=now,
                updated_at=now,
            )
        return tasks, None

    def _format_plan(self, tasks: list[TaskSpec]) -> str:
        lines = ["Here’s what I’ll do:"]
        for idx, task in enumerate(tasks, start=1):
            lines.append(f"{idx}) {task.title}")
        return "\n".join(lines)

    def _select_next_task(
        self,
        tasks_rows: list[Any],
        tasks_by_ordinal: dict[int, Any],
    ) -> Any | None:
        for row in tasks_rows:
            status = row["status"]
            if status not in {"pending", "blocked"}:
                continue
            if status == "blocked" and self._is_waiting_on_delegation(row):
                continue
            deps = self._task_dependencies(row)
            if any(tasks_by_ordinal.get(dep, {}).get("status") == "failed" for dep in deps):
                self._skip_task_due_to_dependency(row["id"])
                continue
            if all(tasks_by_ordinal.get(dep, {}).get("status") in {"done", "skipped"} for dep in deps):
                return row
        return None

    def _is_waiting_on_delegation(self, row: Any) -> bool:
        outputs_json = row["outputs_json"]
        if not outputs_json:
            return False
        try:
            payload = json.loads(outputs_json)
        except json.JSONDecodeError:
            return False
        delegation = payload.get("delegation", {})
        return isinstance(delegation, dict) and delegation.get("status") == "sent"

    def _task_dependencies(self, row: Any) -> list[int]:
        inputs_json = row["inputs_json"]
        if not inputs_json:
            return []
        try:
            payload = json.loads(inputs_json)
        except json.JSONDecodeError:
            return []
        deps = payload.get("dependencies", [])
        return deps if isinstance(deps, list) else []

    def _skip_task_due_to_dependency(self, task_id: str) -> None:
        now = datetime.utcnow().isoformat()
        self.db.update_task_status(task_id, status="skipped", outputs_json=None, updated_at=now)
        self._log_task_event(task_id, "note", {"reason": "dependency_failed"})

    def _run_task(
        self,
        run_id: str,
        task_row: Any,
        user_prompt: str,
        prompt_fn: Callable[[str], str],
        llm_calls_remaining: int,
        steps_remaining: int,
    ) -> "TaskExecutionResult":
        task_id = str(task_row["id"])
        now = datetime.utcnow().isoformat()
        self.db.update_task_status(task_id, status="active", outputs_json=task_row["outputs_json"], updated_at=now)
        self.db.update_run_status(run_id, status="active", current_task_id=task_id, last_error=None)
        self._log_task_event(task_id, "start", {"status": "active"})
        if self.hook_manager:
            self.hook_manager.run(
                "on_task_start",
                {"task_id": task_id, "title": task_row["title"], "run_id": run_id},
            )

        llm_calls = 0
        steps = 0
        tool_calls = 0
        tool_records: list[ToolRecord] = []
        tool_results: list[ToolResult] = []
        user_message: str | None = None
        blocked = False

        while (
            llm_calls < self.settings.max_llm_calls_per_task
            and tool_calls < self.settings.max_tool_calls_per_task
            and llm_calls < llm_calls_remaining
            and steps < steps_remaining
        ):
            steps += 1
            plan = self._next_action_plan(user_prompt, task_row, tool_results)
            llm_calls += 1
            if plan is None:
                self._log_task_event(task_id, "fail", {"reason": "action_plan_invalid"})
                self.db.update_task_status(task_id, status="failed", outputs_json=task_row["outputs_json"], updated_at=datetime.utcnow().isoformat())
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message="I hit a planning error and will retry next turn.",
                    blocked=False,
                )
            action = plan.next_action_type
            if action == "tool_call":
                if not plan.tool_name or not plan.tool_args:
                    self._log_task_event(task_id, "fail", {"reason": "tool_call_missing_fields"})
                    self.db.update_task_status(task_id, status="failed", outputs_json=task_row["outputs_json"], updated_at=datetime.utcnow().isoformat())
                    return TaskExecutionResult(
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_records=tool_records,
                        tool_results=tool_results,
                        user_message="Tool call was missing required fields.",
                        blocked=False,
                    )
                tool_call = ToolCall(
                    id=f"tc_{uuid.uuid4().hex}",
                    name=plan.tool_name,
                    args=plan.tool_args,
                    purpose=plan.tool_purpose,
                )
                self._log_task_event(
                    task_id,
                    "tool_call",
                    {"name": tool_call.name, "args": tool_call.args, "purpose": tool_call.purpose},
                )
                results, records = self.tool_runner.run([tool_call], prompt_fn=prompt_fn)
                tool_records.extend(records)
                tool_results.extend(results)
                tool_calls += len(results)
                for result in results:
                    self._log_task_event(
                        task_id,
                        "tool_result",
                        {
                            "name": result.name,
                            "status": result.status,
                            "summary": result.result_summary,
                            "ref": result.result_ref,
                        },
                    )
                    if self.hook_manager:
                        self.hook_manager.run(
                            "on_tool_result",
                            {
                                "task_id": task_id,
                                "run_id": run_id,
                                "tool": result.name,
                                "status": result.status,
                                "summary": result.result_summary,
                            },
                        )
                continue
            if action == "delegate":
                if not self.a2a_client or not plan.delegate_task:
                    self._log_task_event(task_id, "fail", {"reason": "delegate_unavailable"})
                    self.db.update_task_status(
                        task_id,
                        status="failed",
                        outputs_json=task_row["outputs_json"],
                        updated_at=datetime.utcnow().isoformat(),
                    )
                    return TaskExecutionResult(
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_records=tool_records,
                        tool_results=tool_results,
                        user_message="Delegation failed: no A2A client available.",
                        blocked=False,
                    )
                agent = self.a2a_client.select_agent(preferred_name=plan.delegate_to)
                if not agent:
                    self._log_task_event(task_id, "block", {"reason": "no_agent_available"})
                    self.db.update_task_status(
                        task_id,
                        status="blocked",
                        outputs_json=task_row["outputs_json"],
                        updated_at=datetime.utcnow().isoformat(),
                    )
                    return TaskExecutionResult(
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_records=tool_records,
                        tool_results=tool_results,
                        user_message="No eligible agent available for delegation.",
                        blocked=True,
                    )
                correlation_id = str(uuid.uuid4())
                payload = {
                    "type": "task_request",
                    "task": {
                        "title": plan.delegate_task.get("title", task_row["title"]),
                        "description": plan.delegate_task.get("description", task_row["description"]),
                        "inputs": plan.delegate_task.get("inputs", {}),
                    },
                    "constraints": plan.delegate_task.get("constraints", {}),
                    "requested_outputs": plan.delegate_task.get("requested_outputs", []),
                }
                self.a2a_client.send(
                    to_agent_id=agent.get("agent_id"),
                    to_agent_name=agent.get("agent_name"),
                    topic="task",
                    payload=payload,
                    correlation_id=correlation_id,
                )
                now = datetime.utcnow().isoformat()
                self.db.insert_delegation(
                    delegation_id=str(uuid.uuid4()),
                    task_id=task_id,
                    run_id=run_id,
                    correlation_id=correlation_id,
                    to_agent_id=agent.get("agent_id"),
                    to_agent_name=agent.get("agent_name"),
                    status="sent",
                    created_at=now,
                    updated_at=now,
                )
                updated_outputs = self._merge_outputs(
                    task_row["outputs_json"],
                    {
                        "delegation": {
                            "to_agent": agent.get("agent_name"),
                            "correlation_id": correlation_id,
                            "status": "sent",
                        }
                    },
                )
                self.db.update_task_status(
                    task_id=task_id,
                    status="blocked",
                    outputs_json=updated_outputs,
                    updated_at=now,
                )
                self._log_task_event(task_id, "block", {"reason": "delegated"})
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=f"Delegated to {agent.get('agent_name')}.",
                    blocked=True,
                )
            if action == "store_output":
                updated_outputs = self._merge_outputs(task_row["outputs_json"], plan.outputs_json)
                now = datetime.utcnow().isoformat()
                self.db.update_task_outputs(task_id, outputs_json=updated_outputs, updated_at=now)
                task_row = dict(task_row)
                task_row["outputs_json"] = updated_outputs
                self._log_task_event(task_id, "note", {"outputs_json": plan.outputs_json or {}})
                continue
            if action == "respond":
                if plan.message:
                    user_message = plan.message
                self._log_task_event(task_id, "note", {"message": plan.message or ""})
                continue
            if action == "ask_user":
                question = plan.message or "I need one clarification to continue. What should I do next?"
                updated_outputs = self._merge_outputs(
                    task_row["outputs_json"], {"blocked_question": question}
                )
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="blocked", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "block", {"reason": "ask_user", "question": question})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {"task_id": task_id, "run_id": run_id, "status": "blocked"},
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=question,
                    blocked=True,
                )
            if action == "block":
                reason = plan.block_reason or "blocked"
                updated_outputs = self._merge_outputs(task_row["outputs_json"], {"blocked_reason": reason})
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="blocked", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "block", {"reason": reason})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {"task_id": task_id, "run_id": run_id, "status": "blocked"},
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=plan.message or "I’m blocked and need more input.",
                    blocked=True,
                )
            if action == "mark_done":
                updated_outputs = self._merge_outputs(task_row["outputs_json"], plan.outputs_json)
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="done", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "complete", {"outputs_json": plan.outputs_json or {}})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {"task_id": task_id, "run_id": run_id, "status": "done"},
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=plan.message,
                    blocked=False,
                )
            if action == "fail":
                self.db.update_task_status(task_id, status="failed", outputs_json=task_row["outputs_json"], updated_at=datetime.utcnow().isoformat())
                self._log_task_event(task_id, "fail", {"reason": plan.block_reason or "failed"})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {"task_id": task_id, "run_id": run_id, "status": "failed"},
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=plan.message or "Task failed.",
                    blocked=False,
                )

        self._log_task_event(task_id, "note", {"reason": "budget_pause"})
        self.db.update_task_status(task_id, status="pending", outputs_json=task_row["outputs_json"], updated_at=datetime.utcnow().isoformat())
        if self.hook_manager:
            self.hook_manager.run(
                "on_task_end",
                {"task_id": task_id, "run_id": run_id, "status": "pending"},
            )
        return TaskExecutionResult(
            llm_calls=llm_calls,
            steps=steps,
            tool_records=tool_records,
            tool_results=tool_results,
            user_message="Task paused due to budget; continuing next turn.",
            blocked=False,
        )

    def _next_action_plan(
        self,
        user_prompt: str,
        task_row: Any,
        tool_results: list[ToolResult],
    ) -> ActionPlan | None:
        summaries = []
        for result in tool_results[-5:]:
            summaries.append(
                f"- {result.name} status={result.status} summary={result.result_summary}"
            )
        prompt = build_action_plan_prompt(
            user_prompt=user_prompt,
            task_title=str(task_row["title"]),
            task_description=str(task_row["description"] or ""),
            task_inputs_json=task_row["inputs_json"],
            task_outputs_json=task_row["outputs_json"],
            tool_descriptions=self.tool_descriptions,
            recent_tool_summaries=summaries,
        )
        response = self.llm.generate(prompt)
        plan, error = parse_action_plan(response.content)
        if error or not plan:
            return None
        return plan

    def _merge_outputs(self, existing: str | None, update: dict[str, Any] | None) -> str | None:
        if update is None:
            return existing
        base: dict[str, Any] = {}
        if existing:
            try:
                base = json.loads(existing)
            except json.JSONDecodeError:
                base = {}
        base.update(update)
        return json.dumps(base)

    def _log_task_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.db.insert_task_event(
            event_id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            payload=json.dumps(payload),
        )

    def _all_tasks_complete(self, tasks_rows: list[Any]) -> bool:
        for row in tasks_rows:
            if row["status"] not in {"done", "skipped"}:
                return False
        return True

    def _completion_review(
        self, run_id: str, user_prompt: str, tasks_rows: list[Any]
    ) -> tuple[str | None, int, int]:
        summaries: list[dict[str, Any]] = []
        for row in tasks_rows:
            summaries.append(
                {
                    "ordinal": row["ordinal"],
                    "title": row["title"],
                    "status": row["status"],
                    "outputs_json": row["outputs_json"],
                }
            )
        prompt = build_completion_review_prompt(user_prompt, summaries)
        response = self.llm.generate(prompt)
        review, error = parse_completion_review(response.content)
        if error or not review:
            return "Completion review failed; will continue next turn.", 1, 1
        if review.overall_status == "incomplete":
            self._append_missing_tasks(run_id, review.missing_items)
            self.db.update_run_status(run_id, status="active", last_error="review_incomplete")
            return "Missing items found. Continuing with new tasks.", 1, 1
        self.db.update_run_status(run_id, status="done", current_task_id=None, last_error=None)
        review_message = self._format_completion_review(review, tasks_rows)
        return review_message, 1, 1

    def _append_missing_tasks(self, run_id: str, missing_items: list[str]) -> None:
        if not missing_items:
            return
        tasks_rows = self.db.fetch_tasks_for_run(run_id)
        start_ordinal = max(int(row["ordinal"]) for row in tasks_rows) + 1 if tasks_rows else 0
        now = datetime.utcnow().isoformat()
        for idx, item in enumerate(missing_items):
            title = item.strip() or "Address missing item"
            inputs_json = json.dumps({"verification": "Validate completion of missing item", "dependencies": []})
            self.db.insert_task(
                task_id=str(uuid.uuid4()),
                run_id=run_id,
                parent_task_id=None,
                ordinal=start_ordinal + idx,
                title=title,
                description=f"Address missing item: {title}",
                status="pending",
                inputs_json=inputs_json,
                outputs_json=None,
                requires_tools=0,
                created_at=now,
                updated_at=now,
            )

    def _format_completion_review(
        self, review: ReviewResult, tasks_rows: list[Any]
    ) -> str:
        lines = ["Completion Review"]
        for check in review.checks:
            ordinal = check.get("task_ordinal")
            status = check.get("status")
            note = check.get("note", "")
            if ordinal is not None:
                lines.append(f"- Task {int(ordinal) + 1}: {status} {note}".strip())
        outputs: list[str] = []
        for row in tasks_rows:
            if row["outputs_json"]:
                outputs.append(f"- Task {int(row['ordinal']) + 1}: {row['outputs_json']}")
        if outputs:
            lines.append("Outputs")
            lines.extend(outputs)
        if review.assumptions:
            lines.append("Assumptions")
            lines.extend(f"- {item}" for item in review.assumptions)
        lines.append(review.user_message)
        return "\n".join(lines)


@dataclass
class TaskExecutionResult:
    llm_calls: int
    steps: int
    tool_records: list[ToolRecord]
    tool_results: list[ToolResult]
    user_message: str | None
    blocked: bool
