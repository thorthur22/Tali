from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import httpx

from tali.classifier import ScoringResult, classify_request
from tali.db import Database
from tali.guardrails import Guardrails
from tali.hooks.core import HookManager
from tali.a2a import A2AClient
from tali.llm import LLMClient
from tali.models import ProvenanceType
from tali.memory_ingest import stage_tool_result_fact
from tali.retrieval import Retriever
from tali.prompting import format_retrieval_context
from tali.preferences import extract_preferences
from tali.self_reflection import build_reflection_prompt, parse_reflection_response
from tali.tasking import (
    ActionPlan,
    TaskSpec,
    ReviewResult,
    build_action_plan_prompt,
    build_response_prompt,
    build_completion_review_prompt,
    build_decomposition_prompt,
    parse_action_plan,
    parse_completion_review,
    parse_decomposition,
)
from tali.tools.protocol import ToolCall, ToolResult
from tali.tools.runner import ToolRecord, ToolRunner
from tali.working_memory import (
    WorkingMemory,
    is_stateful_progress,
    summarize_tool_result,
)


def _format_llm_http_error(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    detail = ""
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            detail = payload.get("error", {}).get("message") or payload.get("message") or ""
    except Exception:
        detail = exc.response.text.strip()
    detail = detail[:500] if detail else ""
    if status in {401, 403}:
        hint = "Authentication failed. Use an OpenAI API key, or ensure Codex login is valid for this endpoint."
        return f"HTTP {status}. {hint} {detail}".strip()
    return f"HTTP {status}. {detail}".strip()


@dataclass(frozen=True)
class TaskRunnerSettings:
    max_tasks_per_turn: int = 200
    max_llm_calls_per_task: int = 120
    max_tool_calls_per_task: int = 200
    max_total_llm_calls_per_run_per_turn: int = 2000
    max_total_steps_per_turn: int = 2000


@dataclass
class TaskRunnerResult:
    message: str
    tool_records: list[ToolRecord]
    tool_results: list[ToolResult]
    run_id: str | None
    llm_calls: int
    steps: int
    tool_calls: int


class TaskRunner:
    def __init__(
        self,
        db: Database,
        llm: LLMClient,
        retriever: Retriever,
        guardrails: Guardrails,
        tool_runner: ToolRunner,
        tool_descriptions: str,
        planner_llm: LLMClient | None = None,
        responder_llm: LLMClient | None = None,
        hook_manager: HookManager | None = None,
        a2a_client: A2AClient | None = None,
        agent_context: str = "",
        status_fn: Callable[[str], None] | None = None,
        settings: TaskRunnerSettings | None = None,
        responder_strengths: str | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.planner_llm = planner_llm or llm
        self.responder_llm = responder_llm or llm
        self.retriever = retriever
        self.guardrails = guardrails
        self.tool_runner = tool_runner
        self.tool_descriptions = tool_descriptions
        self.hook_manager = hook_manager
        self.a2a_client = a2a_client
        self.agent_context = agent_context
        self.status_fn = status_fn
        self.settings = settings or TaskRunnerSettings()
        self.responder_strengths = responder_strengths
        self._current_classification: ScoringResult | None = None

    def run_turn(
        self,
        user_input: str,
        prompt_fn: Callable[[str], str],
        show_plans: bool = False,
        origin: str = "user",
    ) -> TaskRunnerResult:
        active_run = self.db.fetch_active_run()
        # Detect stale active runs (process may have crashed mid-execution)
        if active_run is not None and active_run["status"] == "active":
            updated_at = active_run["updated_at"]
            if updated_at:
                try:
                    last_update = datetime.fromisoformat(str(updated_at))
                    from datetime import timedelta
                    if datetime.utcnow() - last_update > timedelta(minutes=30):
                        # Stale run: mark as blocked so user can resume or cancel
                        self.db.update_run_status(
                            str(active_run["id"]),
                            status="blocked",
                            last_error="stale_run_recovered",
                        )
                        active_run = self.db.fetch_active_run()
                except (ValueError, TypeError):
                    pass
        resume_intent = self._is_resume_intent(user_input)
        classification = self._classify_request(user_input)
        self._current_classification = classification
        if self.status_fn:
            signals_str = ", ".join(classification.signals[:5]) if classification.signals else "none"
            self.status_fn(
                f"Classification: tier={classification.tier}, "
                f"query_type={classification.query_type}, "
                f"confidence={classification.confidence:.2f}, "
                f"signals=[{signals_str}]"
            )
        effective_prompt = user_input
        if active_run is not None and resume_intent:
            effective_prompt = str(active_run["user_prompt"])
        retrieval_context = self.retriever.retrieve(effective_prompt)
        memory_context = format_retrieval_context(retrieval_context.bundle)
        skill_context = self._build_skill_context(list(retrieval_context.bundle.skills))
        working_memory = WorkingMemory(user_goal=effective_prompt)
        tool_records: list[ToolRecord] = []
        tool_results: list[ToolResult] = []
        messages: list[str] = []
        llm_calls = 0
        steps = 0
        run_id: str | None = None
        created_run = False

        if active_run is not None:
            if (
                active_run["status"] == "active"
                and not resume_intent
                and str(active_run["user_prompt"]).strip() != user_input.strip()
            ):
                if self._is_run_status_query(user_input):
                    return self._respond_with_active_run_status(active_run)
                if not self._is_actionable_followup(user_input):
                    return self._respond_to_unrelated_prompt(
                        user_input=user_input,
                        retrieval_context=retrieval_context,
                        working_memory=working_memory,
                        active_run=active_run,
                    )
                if not self._is_prompt_related_to_run(user_input, active_run):
                    return self._respond_to_unrelated_prompt(
                        user_input=user_input,
                        retrieval_context=retrieval_context,
                        working_memory=working_memory,
                        active_run=active_run,
                    )
            else:
                existing_tasks = self.db.fetch_tasks_for_run(str(active_run["id"]))
                if not existing_tasks:
                    if resume_intent:
                        retry_prompt = self._resolve_resume_prompt(active_run)
                        if not retry_prompt:
                            return TaskRunnerResult(
                                message=(
                                    "I couldn't recover a previous concrete task prompt for this run. "
                                    "Give me a concrete request (for example: 'build the app')."
                                ),
                                tool_records=[],
                                tool_results=[],
                                run_id=str(active_run["id"]),
                                llm_calls=0,
                                steps=0,
                                tool_calls=0,
                            )
                        tasks, error = self._decompose_and_persist(
                            str(active_run["id"]), retry_prompt, memory_context
                        )
                        llm_calls += 1
                        if error or not tasks:
                            self.db.update_run_status(
                                str(active_run["id"]),
                                status="blocked",
                                last_error=error or "decomposition_failed",
                            )
                            question = (
                                f"LLM request failed ({error}). "
                                "Check your LLM server (for Ollama: ensure it is running and the base URL is correct)."
                                if error and error.startswith("llm_error:")
                                else "I couldn't break this into tasks. Could you rephrase or clarify the request?"
                            )
                            guarded = self.guardrails.enforce(question, retrieval_context.bundle)
                            return TaskRunnerResult(
                                message=guarded.safe_output,
                                tool_records=[],
                                tool_results=[],
                                run_id=str(active_run["id"]),
                                llm_calls=llm_calls,
                                steps=steps,
                                tool_calls=0,
                            )
                        if show_plans:
                            messages.append(self._format_plan(tasks))
                        self.db.update_run_status(
                            str(active_run["id"]), status="active", last_error=None
                        )
                    else:
                        self.db.update_run_status(
                            str(active_run["id"]), status="failed", last_error="no_tasks"
                        )
                        active_run = None
                else:
                    self._resolve_blocked_task_response(active_run, user_input)

        if active_run is None and self._should_use_responder_only(classification, user_input):
            response = self._generate_response(
                user_prompt=user_input,
                memory_context=memory_context,
                working_memory_summary=working_memory.summary_for_prompt(),
                tool_results=[],
                run_summary=None,
                planner_message=None,
                response_llm=self._select_response_llm(classification),
                temperature=self._temperature_for_query(classification.query_type),
            )
            guarded = self.guardrails.enforce(response, retrieval_context.bundle)
            return TaskRunnerResult(
                message=guarded.safe_output,
                tool_records=[],
                tool_results=[],
                run_id=None,
                llm_calls=1,
                steps=1,
                tool_calls=0,
            )

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
                origin=origin,
            )
            tasks, error = self._decompose_and_persist(
                run_id, user_input, memory_context
            )
            llm_calls += 1
            if error or not tasks:
                self.db.update_run_status(run_id, status="blocked", last_error=error or "decomposition_failed")
                if error and error.startswith("llm_error:"):
                    question = (
                        f"LLM request failed ({error}). "
                        "Check your LLM server (for Ollama: ensure it is running and the base URL is correct)."
                    )
                else:
                    question = "I couldn't break this into tasks. Could you rephrase or clarify the request?"
                guarded = self.guardrails.enforce(question, retrieval_context.bundle)
                return TaskRunnerResult(
                    message=guarded.safe_output,
                    tool_records=[],
                    tool_results=[],
                    run_id=run_id,
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_calls=0,
                )
            if show_plans:
                messages.append(self._format_plan(tasks))
        else:
            run_id = str(active_run["id"])
            if active_run["status"] == "blocked":
                self.db.update_run_status(run_id, status="active", last_error=None)

        assert run_id is not None
        tasks_rows = self.db.fetch_tasks_for_run(run_id)
        if not tasks_rows:
            message = "No tasks found for this run. Please try again."
            self.db.update_run_status(run_id, status="failed", last_error="no_tasks")
            guarded = self.guardrails.enforce(message, retrieval_context.bundle)
            return TaskRunnerResult(
                message=guarded.safe_output,
                tool_records=[],
                tool_results=[],
                run_id=run_id,
                llm_calls=llm_calls,
                steps=steps,
                tool_calls=0,
            )

        tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}
        if not self._all_tasks_complete(tasks_rows):
            next_task = self._select_next_task(tasks_rows, tasks_by_ordinal)
            if next_task is None:
                waiting = any(
                    row["status"] == "blocked" and self._is_waiting_on_delegation(row)
                    for row in tasks_rows
                )
                if waiting:
                    messages.append(
                        "Waiting on a delegated agent. Try again later or check `inbox`."
                    )
                    guarded = self.guardrails.enforce(
                        "\n\n".join(messages), retrieval_context.bundle
                    )
                    return TaskRunnerResult(
                        message=guarded.safe_output,
                        tool_records=tool_records,
                        tool_results=tool_results,
                        run_id=run_id,
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_calls=len(tool_results),
                    )
                self.db.update_run_status(run_id, status="failed", last_error="no_eligible_tasks")
                run_id = str(uuid.uuid4())
                self.db.insert_run(
                    run_id=run_id,
                    created_at=datetime.utcnow().isoformat(),
                    status="active",
                    user_prompt=user_input,
                    current_task_id=None,
                    last_error=None,
                )
                tasks, error = self._decompose_and_persist(
                    run_id, user_input, memory_context
                )
                llm_calls += 1
                if error or not tasks:
                    self.db.update_run_status(
                        run_id, status="blocked", last_error=error or "decomposition_failed"
                    )
                    question = (
                        f"LLM request failed ({error}). "
                        "Check your LLM server (for Ollama: ensure it is running and the base URL is correct)."
                        if error and error.startswith("llm_error:")
                        else "I couldn't break this into tasks. Could you rephrase or clarify the request?"
                    )
                    guarded = self.guardrails.enforce(question, retrieval_context.bundle)
                    return TaskRunnerResult(
                        message=guarded.safe_output,
                        tool_records=[],
                        tool_results=[],
                        run_id=run_id,
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_calls=0,
                    )
                if show_plans:
                    messages.append(self._format_plan(tasks))
                tasks_rows = self.db.fetch_tasks_for_run(run_id)
        tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}
        if self._apply_user_preferences(run_id, user_input):
            retrieval_context = self.retriever.retrieve(user_input)
            memory_context = format_retrieval_context(retrieval_context.bundle)
            skill_context = self._build_skill_context(list(retrieval_context.bundle.skills))
        run_summary = self._refresh_run_summary(run_id, tasks_rows)
        persist_until_done = self._should_persist_until_done(
            user_input=user_input,
            resume_intent=resume_intent,
            has_active_run=active_run is not None,
        )
        budget_multiplier = 20 if persist_until_done else 1
        max_turn_steps = self.settings.max_total_steps_per_turn * budget_multiplier
        max_turn_llm_calls = self.settings.max_total_llm_calls_per_run_per_turn * budget_multiplier
        max_tasks_this_turn = self.settings.max_tasks_per_turn * budget_multiplier

        # --- Circuit breaker: detect repeated review_incomplete across turns ---
        run_row = self.db.fetch_run(run_id)
        prior_last_error = str(run_row["last_error"] or "") if run_row else ""
        prior_review_incomplete = prior_last_error.startswith("review_incomplete")
        if prior_review_incomplete and not created_run:
            incomplete_count = self._count_consecutive_review_incompletes(run_id)
            if incomplete_count >= 2:
                remaining_msg = self._build_remaining_tasks_message(tasks_rows)
                stuck_message = (
                    "I've attempted this request multiple times but cannot fully complete it. "
                    f"Here's what remains:\n{remaining_msg}\n"
                    "Please provide additional guidance so I can finish."
                )
                self.db.update_run_status(run_id, status="blocked", last_error="stuck_review_incomplete")
                guarded = self.guardrails.enforce(stuck_message, retrieval_context.bundle)
                return TaskRunnerResult(
                    message=guarded.safe_output,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    run_id=run_id,
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_calls=len(tool_results),
                )

        # --- Main task execution loop with auto-continue on incomplete review ---
        max_review_retries = 2
        review_retries = 0

        while True:
            completed_this_turn = 0
            while (
                completed_this_turn < max_tasks_this_turn
                and steps < max_turn_steps
                and llm_calls < max_turn_llm_calls
            ):
                self.db.heartbeat_run(run_id)
                next_task = self._select_next_task(tasks_rows, tasks_by_ordinal)
                if next_task is None:
                    break
                task_result = self._run_task(
                    run_id=run_id,
                    task_row=next_task,
                    user_prompt=str(self.db.fetch_run(run_id)["user_prompt"]),
                    prompt_fn=prompt_fn,
                    memory_context=memory_context,
                    working_memory=working_memory,
                    run_summary=run_summary,
                    skill_context=skill_context,
                    llm_calls_remaining=max_turn_llm_calls - llm_calls,
                    steps_remaining=max_turn_steps - steps,
                )
                llm_calls += task_result.llm_calls
                steps += task_result.steps
                tool_records.extend(task_result.tool_records)
                tool_results.extend(task_result.tool_results)
                if task_result.user_message and not (
                    persist_until_done
                    and task_result.user_message.startswith("Task paused due to budget")
                ):
                    messages.append(task_result.user_message)
                if task_result.blocked:
                    self.db.update_run_status(run_id, status="blocked", current_task_id=next_task["id"])
                    break
                completed_this_turn += 1
                tasks_rows = self.db.fetch_tasks_for_run(run_id)
                tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}
                run_summary = self._refresh_run_summary(run_id, tasks_rows)

            if self._all_tasks_complete(tasks_rows):
                review_message, review_llm_calls, review_steps = self._completion_review(
                    run_id=run_id,
                    user_prompt=str(self.db.fetch_run(run_id)["user_prompt"]),
                    tasks_rows=tasks_rows,
                    memory_context=memory_context,
                )
                llm_calls += review_llm_calls
                steps += review_steps
                if review_message:
                    # Route the planner's review message through the responder for
                    # polishing, ensuring consistent tone and style.
                    cls = getattr(self, "_current_classification", None)
                    resp_llm = self._select_response_llm(cls) if cls else self.responder_llm
                    resp_temp = self._temperature_for_query(cls.query_type) if cls else None
                    polished = self._generate_response(
                        user_prompt=str(self.db.fetch_run(run_id)["user_prompt"]),
                        memory_context=memory_context,
                        working_memory_summary=working_memory.summary_for_prompt(),
                        tool_results=tool_results,
                        run_summary=run_summary,
                        planner_message=review_message,
                        response_llm=resp_llm,
                        temperature=resp_temp,
                    )
                    llm_calls += 1
                    messages.append(polished)
                    break
                else:
                    # Incomplete review -- retry if budget allows
                    review_retries += 1
                    if review_retries > max_review_retries:
                        messages.append(self._build_remaining_tasks_message(tasks_rows))
                        break
                    # Refresh tasks (missing tasks were appended by _completion_review)
                    tasks_rows = self.db.fetch_tasks_for_run(run_id)
                    tasks_by_ordinal = {int(row["ordinal"]): row for row in tasks_rows}
                    run_summary = self._refresh_run_summary(run_id, tasks_rows)
                    # Check if budget allows continuation
                    if (
                        steps >= max_turn_steps
                        or llm_calls >= max_turn_llm_calls
                    ):
                        messages.append(self._build_remaining_tasks_message(tasks_rows))
                        break
                    continue
            else:
                # Budget exhausted or blocked before all tasks complete
                if (
                    steps >= max_turn_steps
                    or llm_calls >= max_turn_llm_calls
                ):
                    messages.append(self._build_remaining_tasks_message(tasks_rows))
                    break
                waiting = any(
                    row["status"] == "blocked" and self._is_waiting_on_delegation(row)
                    for row in tasks_rows
                )
                if waiting:
                    messages.append(
                        "Waiting on a delegated agent. Try again later or check `inbox`."
                    )
                    break
                if completed_this_turn == 0:
                    messages.append(self._build_remaining_tasks_message(tasks_rows))
                    break
                continue

        final_message = "\n\n".join(msg for msg in messages if msg.strip())
        if not final_message:
            remaining = self._build_remaining_tasks_message(tasks_rows)
            final_message = remaining if remaining else "Working on it."
        self._refresh_run_summary(run_id, tasks_rows)
        guarded = self.guardrails.enforce(final_message, retrieval_context.bundle)
        return TaskRunnerResult(
            message=guarded.safe_output,
            tool_records=tool_records,
            tool_results=tool_results,
            run_id=run_id,
            llm_calls=llm_calls,
            steps=steps,
            tool_calls=len(tool_results),
        )

    def _decompose_and_persist(
        self,
        run_id: str,
        user_prompt: str,
        memory_context: str,
    ) -> tuple[list[TaskSpec] | None, str | None]:
        if self.status_fn:
            self.status_fn("Decomposing request into tasks")
        prompt = build_decomposition_prompt(
            user_prompt, self.tool_descriptions, self.agent_context, memory_context
        )
        try:
            response = self.planner_llm.generate(prompt)
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            return None, f"llm_error: {_format_llm_http_error(exc)}"
        except Exception as exc:  # noqa: BLE001
            return None, f"llm_error: {exc}"
        tasks, error = parse_decomposition(response.content)
        if error or not tasks:
            fallback = self._fallback_decomposition_tasks(user_prompt, error)
            if not fallback:
                return None, error
            tasks = fallback
            if self.status_fn:
                self.status_fn("Decomposition parse failed; using fallback task decomposition")
        if self.status_fn:
            self.status_fn(f"Decomposition ok: {len(tasks)} tasks")
        if self.status_fn:
            summary = "; ".join(f"{idx + 1}) {task.title}" for idx, task in enumerate(tasks))
            self.status_fn(f"Plan: {summary}")
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

    def _fallback_decomposition_tasks(
        self, user_prompt: str, parse_error: str | None
    ) -> list[TaskSpec] | None:
        if not self._is_actionable_followup(user_prompt):
            return None
        cleaned = " ".join(user_prompt.strip().split())
        if not cleaned:
            return None
        title = cleaned[:70]
        if len(cleaned) > 70:
            title += "..."
        description = (
            "Execute the user request directly with available tools and produce concrete "
            f"artifacts. Original request: {cleaned}"
        )
        verification = (
            "At least one concrete deliverable is produced or updated, and outputs_json "
            "summarizes changed files and completion evidence."
        )
        if parse_error:
            description += f" (fallback due to decomposition error: {parse_error})"
        return [
            TaskSpec(
                title=f"Execute request: {title}",
                description=description,
                requires_tools=True,
                verification=verification,
                dependencies=[],
            )
        ]

    def _format_plan(self, tasks: list[TaskSpec]) -> str:
        lines = ["Here’s what I’ll do:"]
        for idx, task in enumerate(tasks, start=1):
            lines.append(f"{idx}) {task.title}")
        return "\n".join(lines)

    def _apply_user_preferences(self, run_id: str, user_input: str) -> bool:
        candidates = extract_preferences(user_input)
        if not candidates:
            return False
        now = datetime.utcnow().isoformat()
        for candidate in candidates:
            self.db.upsert_preference(
                key=candidate.key,
                value=candidate.value,
                confidence=candidate.confidence,
                provenance_type=ProvenanceType.USER_REPORTED.value,
                source_ref=f"run:{run_id}",
                updated_at=now,
            )
        return True

    def _build_skill_context(self, skill_names: list[str]) -> str:
        if not skill_names:
            return "- None"
        lines: list[str] = []
        for name in skill_names:
            row = self.db.fetch_skill_by_name(name)
            if not row:
                continue
            steps_raw = row["steps"] or "[]"
            try:
                steps = json.loads(steps_raw)
            except json.JSONDecodeError:
                steps = [steps_raw]
            steps_text = ", ".join(step for step in steps if isinstance(step, str)) or "(no steps)"
            lines.append(
                f"- {row['name']} (trigger={row['trigger']}, success={row['success_count']}, "
                f"failure={row['failure_count']}): {steps_text}"
            )
        return "\n".join(lines) if lines else "- None"

    def _append_skill_use(self, existing: str | None, skill_name: str, steps: list[str]) -> str:
        base: dict[str, Any] = {}
        if existing:
            try:
                base = json.loads(existing)
            except json.JSONDecodeError:
                base = {}
        skills_used = base.get("skills_used")
        if not isinstance(skills_used, list):
            skills_used = []
        if not any(entry.get("name") == skill_name for entry in skills_used if isinstance(entry, dict)):
            skills_used.append(
                {
                    "name": skill_name,
                    "steps": steps,
                    "used_at": datetime.utcnow().isoformat(),
                }
            )
        base["skills_used"] = skills_used
        return json.dumps(base)

    def _apply_skill_outcome(self, outputs_json: str | None, outcome: str) -> None:
        if not outputs_json:
            return
        try:
            payload = json.loads(outputs_json)
        except json.JSONDecodeError:
            return
        skills_used = payload.get("skills_used")
        if not isinstance(skills_used, list):
            return
        for entry in skills_used:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            if outcome == "success":
                self.db.increment_skill_success(name)
            elif outcome == "failure":
                self.db.increment_skill_failure(name)

    def _refresh_run_summary(self, run_id: str, tasks_rows: list[Any]) -> str:
        run_row = self.db.fetch_run(run_id)
        if not run_row:
            return ""
        summary = self._build_run_summary(run_row, tasks_rows)
        self.db.update_run_summary(run_id, summary)
        return summary

    def _build_run_summary(self, run_row: Any, tasks_rows: list[Any]) -> str:
        total = len(tasks_rows)
        counts = {
            "done": 0,
            "active": 0,
            "pending": 0,
            "blocked": 0,
            "failed": 0,
            "skipped": 0,
        }
        current_task_title = None
        current_task_id = run_row["current_task_id"]
        blocked_questions: list[str] = []
        clarifications: list[str] = []
        for row in tasks_rows:
            status = row["status"]
            if status in counts:
                counts[status] += 1
            if current_task_id and str(row["id"]) == str(current_task_id):
                current_task_title = row["title"]
            outputs_json = row["outputs_json"]
            if row["status"] == "blocked" and outputs_json and len(blocked_questions) < 2:
                try:
                    payload = json.loads(outputs_json)
                except json.JSONDecodeError:
                    payload = {}
                question = payload.get("blocked_question")
                if isinstance(question, str) and question.strip():
                    blocked_questions.append(question.strip())
            if outputs_json and len(clarifications) < 3:
                try:
                    payload = json.loads(outputs_json)
                except json.JSONDecodeError:
                    payload = {}
                question = payload.get("blocked_question")
                answer = payload.get("blocked_answer")
                if (
                    isinstance(question, str)
                    and question.strip()
                    and isinstance(answer, str)
                    and answer.strip()
                ):
                    clarifications.append(f"{question.strip()} -> {answer.strip()}")
        lines = [f"Goal: {run_row['user_prompt']}"]
        lines.append(
            "Status: "
            f"done {counts['done']}/{total}; "
            f"active {counts['active']}; "
            f"pending {counts['pending']}; "
            f"blocked {counts['blocked']}; "
            f"failed {counts['failed']}; "
            f"skipped {counts['skipped']}"
        )
        if current_task_title:
            lines.append(f"Current task: {current_task_title}")
        if blocked_questions:
            lines.append("Blocked questions: " + " | ".join(blocked_questions))
        if clarifications:
            lines.append("Clarifications: " + " | ".join(clarifications))
        if run_row["last_error"]:
            lines.append(f"Last error: {run_row['last_error']}")
        # Enriched task outcomes for responder context
        done_rows = [r for r in tasks_rows if r["status"] == "done"]
        if done_rows:
            lines.append("Task outcomes:")
            for row in done_rows[-5:]:
                title = row["title"] or "Untitled"
                outcome = ""
                if row["outputs_json"]:
                    try:
                        payload = json.loads(row["outputs_json"])
                        # Extract meaningful outcome data, skip internal metadata
                        outcome_parts = []
                        for key, val in payload.items():
                            if key in ("blocked_question", "blocked_answer", "skills_used"):
                                continue
                            snippet = str(val)
                            if len(snippet) > 150:
                                snippet = snippet[:150] + "..."
                            outcome_parts.append(f"{key}={snippet}")
                        outcome = "; ".join(outcome_parts[:3])
                    except json.JSONDecodeError:
                        outcome = row["outputs_json"][:200]
                ordinal = int(row["ordinal"]) + 1
                if outcome:
                    lines.append(f"- Task {ordinal} ({title}): {outcome}")
                else:
                    lines.append(f"- Task {ordinal} ({title}): completed")
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
            if any(self._task_status(tasks_by_ordinal.get(dep)) == "failed" for dep in deps):
                self._skip_task_due_to_dependency(row["id"])
                continue
            if all(self._task_status(tasks_by_ordinal.get(dep)) in {"done", "skipped"} for dep in deps):
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

    def _task_status(self, row: Any | None) -> str | None:
        if row is None:
            return None
        return row["status"]

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
        memory_context: str,
        working_memory: WorkingMemory,
        run_summary: str | None,
        skill_context: str | None,
        llm_calls_remaining: int,
        steps_remaining: int,
    ) -> "TaskExecutionResult":
        task_id = str(task_row["id"])
        now = datetime.utcnow().isoformat()
        self.db.update_task_status(task_id, status="active", outputs_json=task_row["outputs_json"], updated_at=now)
        self.db.update_run_status(run_id, status="active", current_task_id=task_id, last_error=None)
        self._log_task_event(task_id, "start", {"status": "active"})
        if self.status_fn:
            self.status_fn(f"Running task: {task_row['title']}")
        if self.hook_manager:
            self.hook_manager.run(
                "on_task_start",
                {"task_id": task_id, "title": task_row["title"], "run_id": run_id},
            )

        llm_calls = 0
        steps = 0
        tool_calls = 0
        store_output_repeats = 0
        repeated_tool_signatures: dict[str, int] = {}
        stuck_replans = 0
        action_replans = 0
        invalid_tool_replans = 0
        planner_failures = 0
        stuck_context: str | None = None
        tool_records: list[ToolRecord] = []
        tool_results: list[ToolResult] = self._hydrate_task_state(task_row, working_memory)
        user_message: str | None = None
        blocked = False

        while (
            llm_calls < self.settings.max_llm_calls_per_task
            and tool_calls < self.settings.max_tool_calls_per_task
            and llm_calls < llm_calls_remaining
            and steps < steps_remaining
        ):
            steps += 1
            plan = self._next_action_plan(
                user_prompt,
                task_row,
                tool_results,
                memory_context,
                working_memory.summary_for_prompt(),
                run_summary,
                stuck_context,
                skill_context,
            )
            llm_calls += 1
            stuck_context = None
            if plan is None:
                planner_failures += 1
                self._log_task_event(
                    task_id,
                    "note",
                    {"reason": "action_plan_invalid", "attempt": planner_failures},
                )
                if self.status_fn:
                    self.status_fn(
                        f"Planning output invalid (attempt {planner_failures}/3); retrying."
                    )
                if planner_failures < 3:
                    stuck_context = (
                        "Previous planning call failed. Retry planning and continue execution."
                    )
                    continue
                if (
                    bool(task_row["requires_tools"])
                    and not self._has_successful_tool_result(tool_results)
                    and self.tool_runner.registry.get("fs.list") is not None
                ):
                    if self.status_fn:
                        self.status_fn(
                            "Planner is stuck; forcing bootstrap tool call: fs.list {}"
                        )
                    plan = ActionPlan(
                        next_action_type="tool_call",
                        message=None,
                        tool_name="fs.list",
                        tool_args={},
                        tool_purpose="Bootstrap workspace context to unblock planning.",
                        outputs_json=None,
                        block_reason=None,
                        delegate_to=None,
                        delegate_task=None,
                        skill_name=None,
                    )
                    planner_failures = 0
                else:
                    break
            planner_failures = 0
            action = plan.next_action_type
            requires_tools = bool(task_row["requires_tools"])
            if (
                action == "respond"
                and plan.message
                and self._response_mentions_tool_action(plan.message)
            ):
                if action_replans < 2:
                    action_replans += 1
                    stuck_context = (
                        "Planner returned respond but message describes a tool action. "
                        "Use next_action_type=tool_call and provide tool_name/tool_args."
                    )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "respond_contains_tool_action", "message": plan.message},
                    )
                    continue
            if (
                action == "respond"
                and self._is_actionable_followup(user_prompt)
                and not self._has_successful_tool_result(tool_results)
                and plan.message
                and self._is_plan_chatter(plan.message)
            ):
                if action_replans < 2:
                    action_replans += 1
                    stuck_context = (
                        "Do not narrate a plan or next action. Perform concrete work now via "
                        "tool_call/store_output/mark_done."
                    )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "respond_plan_chatter", "message": plan.message},
                    )
                    continue
            if (
                action == "respond"
                and requires_tools
                and not self._has_successful_tool_result(tool_results)
            ):
                if action_replans < 2:
                    action_replans += 1
                    stuck_context = (
                        "Task requires tools but no successful tool results exist yet. "
                        "Choose tool_call or ask_user to proceed."
                    )
                    if self.status_fn:
                        self.status_fn(
                            "Replanning: respond chosen before any successful tool result."
                        )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "respond_before_tool_use", "message": plan.message or ""},
                    )
                    continue
                if self.tool_runner.registry.get("fs.list") is not None:
                    if self.status_fn:
                        self.status_fn(
                            "Forcing bootstrap tool call: fs.list {} (respond-before-tools loop)"
                        )
                    plan = ActionPlan(
                        next_action_type="tool_call",
                        message=None,
                        tool_name="fs.list",
                        tool_args={},
                        tool_purpose="Bootstrap workspace context to unblock execution.",
                        outputs_json=None,
                        block_reason=None,
                        delegate_to=None,
                        delegate_task=None,
                        skill_name=None,
                    )
            if (
                action in {"mark_done", "store_output"}
                and requires_tools
                and not self._has_successful_tool_result(tool_results)
            ):
                if action_replans < 2:
                    action_replans += 1
                    stuck_context = (
                        "Task requires tools but no successful tool results exist yet. "
                        "Choose tool_call or ask_user before storing outputs or marking done."
                    )
                    if self.status_fn:
                        self.status_fn(
                            "Replanning: completion chosen before any successful tool result."
                        )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "complete_before_tool_use", "action": action},
                    )
                    continue
                if self.tool_runner.registry.get("fs.list") is not None:
                    if self.status_fn:
                        self.status_fn(
                            "Forcing bootstrap tool call: fs.list {} (complete-before-tools loop)"
                        )
                    plan = ActionPlan(
                        next_action_type="tool_call",
                        message=None,
                        tool_name="fs.list",
                        tool_args={},
                        tool_purpose="Bootstrap workspace context to unblock execution.",
                        outputs_json=None,
                        block_reason=None,
                        delegate_to=None,
                        delegate_task=None,
                        skill_name=None,
                    )
            if action == "tool_call":
                batched_calls: list[dict[str, Any]] | None = None
                if plan.tool_calls:
                    normalized_calls: list[dict[str, Any]] = []
                    for tc in plan.tool_calls:
                        tc_name = tc.get("tool_name")
                        tc_args = tc.get("tool_args") or {}
                        tc_purpose = tc.get("tool_purpose", "")
                        if tc_name == "fs_root":
                            tc_name = "fs.list"
                            tc_args = {}
                        if tc_name == "fs.list":
                            path_value = tc_args.get("path") if isinstance(tc_args, dict) else None
                            if not path_value or (isinstance(path_value, str) and not path_value.strip()):
                                tc_args = {}
                        normalized_calls.append(
                            {"tool_name": tc_name, "tool_args": tc_args, "tool_purpose": tc_purpose}
                        )
                    unknown_tools = [
                        tc["tool_name"]
                        for tc in normalized_calls
                        if self.tool_runner.registry.get(tc["tool_name"]) is None
                    ]
                    if unknown_tools and invalid_tool_replans < 2:
                        invalid_tool_replans += 1
                        stuck_context = (
                            f"Unknown tools {sorted(set(unknown_tools))}. "
                            "Choose valid tools from Available tools."
                        )
                        self._log_task_event(
                            task_id,
                            "note",
                            {"reason": "unknown_tool", "tool": sorted(set(unknown_tools))},
                        )
                        continue
                planned_calls: list[dict[str, Any]] = []
                if plan.tool_calls:
                    for tc in plan.tool_calls:
                        tc_name = tc["tool_name"]
                        tc_args = tc.get("tool_args") or {}
                        if tc_name == "fs.list":
                            path_value = tc_args.get("path")
                            if not path_value or (
                                isinstance(path_value, str) and not path_value.strip()
                            ):
                                tc_args = {}
                        planned_calls.append(
                            {
                                "tool_name": tc_name,
                                "tool_args": tc_args,
                                "tool_purpose": tc.get("tool_purpose", ""),
                            }
                        )
                else:
                    if plan.tool_name == "fs.list" and plan.tool_args is None:
                        plan = ActionPlan(
                            next_action_type=plan.next_action_type,
                            message=plan.message,
                            tool_name=plan.tool_name,
                            tool_args={},
                            tool_purpose=plan.tool_purpose,
                            outputs_json=plan.outputs_json,
                            block_reason=plan.block_reason,
                            delegate_to=plan.delegate_to,
                            delegate_task=plan.delegate_task,
                            skill_name=plan.skill_name,
                        )
                    if not plan.tool_name or (plan.tool_args is None and plan.tool_name != "fs.list"):
                        self._log_task_event(task_id, "fail", {"reason": "tool_call_missing_fields"})
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
                            user_message="Tool call was missing required fields.",
                            blocked=False,
                        )
                    tool_args = plan.tool_args or {}
                    if plan.tool_name == "fs.list":
                        path_value = tool_args.get("path")
                        if not path_value or (
                            isinstance(path_value, str) and not path_value.strip()
                        ):
                            tool_args = {}
                    planned_calls.append(
                        {
                            "tool_name": plan.tool_name,
                            "tool_args": tool_args,
                            "tool_purpose": plan.tool_purpose,
                        }
                    )
                if not planned_calls:
                    self._log_task_event(task_id, "fail", {"reason": "tool_call_missing_fields"})
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
                        user_message="Tool call was missing required fields.",
                        blocked=False,
                    )
                unknown = [
                    call["tool_name"]
                    for call in planned_calls
                    if self.tool_runner.registry.get(call["tool_name"]) is None
                ]
                if unknown:
                    if invalid_tool_replans < 2:
                        invalid_tool_replans += 1
                        stuck_context = (
                            "Unknown tool(s) requested: "
                            f"{', '.join(sorted(set(unknown)))}. "
                            "Choose valid tools from Available tools."
                        )
                        self._log_task_event(
                            task_id,
                            "note",
                            {"reason": "unknown_tool", "tool": sorted(set(unknown))},
                        )
                        continue
                filtered_calls: list[dict[str, Any]] = []
                blocked_dupes: list[dict[str, str]] = []
                for call in planned_calls:
                    args_hash = WorkingMemory.args_hash(call["tool_args"] or {})
                    if working_memory.seen_success(call["tool_name"], args_hash):
                        blocked_dupes.append(
                            {"tool": call["tool_name"], "args_hash": args_hash}
                        )
                        continue
                    filtered_calls.append(call)
                if blocked_dupes and not filtered_calls:
                    stuck_context = (
                        "duplicate_success_blocked: "
                        + ", ".join(
                            f"{item['tool']}:{item['args_hash']}" for item in blocked_dupes
                        )
                    )
                    working_memory.steps_since_progress += 1
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "duplicate_tool_blocked", "blocked": blocked_dupes},
                    )
                    if self.status_fn:
                        self.status_fn(
                            "Blocked duplicate tool call(s): "
                            + ", ".join(
                                f"{item['tool']} {item['args_hash']}" for item in blocked_dupes
                            )
                        )
                    continue
                if blocked_dupes:
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "duplicate_tool_skipped", "blocked": blocked_dupes},
                    )
                for call in filtered_calls:
                    signature = json.dumps(
                        {"name": call["tool_name"], "args": call["tool_args"] or {}},
                        sort_keys=True,
                    )
                    repeated_tool_signatures[signature] = (
                        repeated_tool_signatures.get(signature, 0) + 1
                    )
                    if repeated_tool_signatures[signature] > 2:
                        if stuck_replans < 2:
                            stuck_replans += 1
                            stuck_context = (
                                "Repeated tool call detected. "
                                f"signature={signature} count={repeated_tool_signatures[signature]}. "
                                "Propose a different strategy or tool before asking the user."
                            )
                            self._log_task_event(
                                task_id,
                                "note",
                                {"reason": "stuck_replan", "signature": signature},
                            )
                            break
                        question = (
                            "I'm repeating the same tool call without making progress. "
                            "Can you confirm the correct path or give more details?"
                        )
                        updated_outputs = self._merge_outputs(
                            task_row["outputs_json"], {"blocked_question": question}
                        )
                        now = datetime.utcnow().isoformat()
                        self.db.update_task_status(
                            task_id,
                            status="blocked",
                            outputs_json=updated_outputs,
                            updated_at=now,
                        )
                        self._log_task_event(task_id, "block", {"reason": "repeat_tool_call"})
                        if self.hook_manager:
                            self.hook_manager.run(
                                "on_task_end",
                                {
                                    "task_id": task_id,
                                    "run_id": run_id,
                                    "status": "blocked",
                                    "inputs_json": task_row["inputs_json"],
                                    "outputs_json": updated_outputs,
                                },
                            )
                        return TaskExecutionResult(
                            llm_calls=llm_calls,
                            steps=steps,
                            tool_records=tool_records,
                            tool_results=tool_results,
                            user_message=question,
                            blocked=True,
                        )
                if stuck_context:
                    continue
                # Build tool call list (single or batched)
                calls_to_run: list[ToolCall] = []
                if self.status_fn:
                    if len(filtered_calls) > 1:
                        self.status_fn(f"Executing {len(filtered_calls)} batched tool calls")
                    else:
                        self.status_fn(f"Executing tool call: {filtered_calls[0]['tool_name']}")
                for call in filtered_calls:
                    tc_obj = ToolCall(
                        id=f"tc_{uuid.uuid4().hex}",
                        name=call["tool_name"],
                        args=call.get("tool_args") or {},
                        purpose=call.get("tool_purpose", ""),
                    )
                    calls_to_run.append(tc_obj)
                    self._log_task_event(
                        task_id,
                        "tool_call",
                        {"name": tc_obj.name, "args": tc_obj.args, "purpose": tc_obj.purpose},
                    )
                primary_args_hash = (
                    WorkingMemory.args_hash(calls_to_run[0].args or {})
                    if calls_to_run
                    else ""
                )
                results, records = self.tool_runner.run(calls_to_run, prompt_fn=prompt_fn)
                tool_records.extend(records)
                tool_results.extend(results)
                tool_calls += len(results)
                combined_obs: list[str] = []
                combined_durable: dict[str, Any] = {}
                progress_made = False
                for idx, result in enumerate(results):
                    corresponding_call = calls_to_run[idx] if idx < len(calls_to_run) else calls_to_run[-1]
                    if self.status_fn and result.status == "ok" and result.result_raw:
                        raw = result.result_raw
                        preview = raw if len(raw) <= 200 else f"{raw[:200]}...[truncated]"
                        self.status_fn(f"Tool output {result.name}: {preview}")
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
                    stage_tool_result_fact(self.db, result)
                    observations, durable = summarize_tool_result(
                        corresponding_call.name, corresponding_call.args, result
                    )
                    combined_obs.extend(observations)
                    for key, value in durable.items():
                        if (
                            key.endswith("_paths")
                            and isinstance(value, list)
                            and isinstance(combined_durable.get(key), list)
                        ):
                            combined = combined_durable[key] + value
                            combined_durable[key] = list(dict.fromkeys(combined))
                        else:
                            combined_durable[key] = value
                    call_args_hash = WorkingMemory.args_hash(corresponding_call.args or {})
                    working_memory.record_tool_call(
                        corresponding_call.name, call_args_hash, result.status
                    )
                    if is_stateful_progress(corresponding_call.name, result) or durable:
                        progress_made = True
                working_memory.note_observations(combined_obs, combined_durable)
                primary_call = calls_to_run[0] if calls_to_run else None
                primary_name = primary_call.name if primary_call else plan.tool_name or "unknown"
                primary_args_hash = (
                    WorkingMemory.args_hash(primary_call.args or {})
                    if primary_call
                    else WorkingMemory.args_hash(plan.tool_args or {})
                )
                if progress_made:
                    working_memory.note_progress(
                        "tool_call",
                        {
                            "tool": primary_name,
                            "args_hash": primary_args_hash,
                            "durable_facts": combined_durable,
                        },
                    )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "progress_recorded", "tool": primary_name},
                    )
                else:
                    working_memory.steps_since_progress += 1
                    self._log_task_event(
                        task_id,
                        "note",
                        {
                            "reason": "no_progress",
                            "tool": primary_name,
                            "steps_since_progress": working_memory.steps_since_progress,
                        },
                    )
                self._log_task_event(
                    task_id,
                    "note",
                    {"reason": "memory_update", "observations": combined_obs},
                )
                if working_memory.steps_since_progress >= 2:
                    stuck_context = (
                        "Stuck detected: no progress in 2 tool calls. "
                        "Next action must be stateful progress or a single targeted diagnostic."
                    )
                    self._log_task_event(
                        task_id,
                        "note",
                        {"reason": "stuck_replan", "steps": working_memory.steps_since_progress},
                    )
                continue
            if action == "execute_skill":
                if not plan.skill_name:
                    self._log_task_event(task_id, "fail", {"reason": "skill_missing_name"})
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
                        user_message="Skill execution failed: missing skill name.",
                        blocked=False,
                    )
                skill_row = self.db.fetch_skill_by_name(plan.skill_name)
                if not skill_row:
                    self._log_task_event(task_id, "fail", {"reason": "skill_not_found"})
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
                        user_message=f"Skill not found: {plan.skill_name}.",
                        blocked=False,
                    )
                try:
                    steps_list = json.loads(skill_row["steps"] or "[]")
                except json.JSONDecodeError:
                    steps_list = [skill_row["steps"]]
                steps_list = [step for step in steps_list if isinstance(step, str)]
                updated_outputs = self._append_skill_use(
                    task_row["outputs_json"], plan.skill_name, steps_list
                )
                now = datetime.utcnow().isoformat()
                self.db.update_task_outputs(task_id, outputs_json=updated_outputs, updated_at=now)
                task_row = dict(task_row)
                task_row["outputs_json"] = updated_outputs
                self._log_task_event(
                    task_id,
                    "note",
                    {"skill": plan.skill_name, "steps": steps_list},
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
                if not agent and plan.delegate_to:
                    # delegate_to might be a capability keyword rather than an agent name
                    agent = self.a2a_client.select_agent(capability=plan.delegate_to)
                if not agent:
                    # Graceful fallback: reset task to pending so the planner
                    # can retry with a non-delegate action on the next iteration.
                    capability_hint = (
                        f" with capability '{plan.delegate_to}'" if plan.delegate_to else ""
                    )
                    fallback_note = (
                        f"Delegation failed: no agent available{capability_hint}. "
                        "Handle this task locally or ask the user to create a new agent."
                    )
                    self._log_task_event(
                        task_id, "note", {"delegation_fallback": fallback_note}
                    )
                    updated_outputs = self._merge_outputs(
                        task_row["outputs_json"],
                        {"delegation_fallback": fallback_note},
                    )
                    self.db.update_task_status(
                        task_id,
                        status="pending",
                        outputs_json=updated_outputs,
                        updated_at=datetime.utcnow().isoformat(),
                    )
                    continue
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
                if plan.outputs_json is None:
                    self._log_task_event(task_id, "fail", {"reason": "store_output_missing_fields"})
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
                        user_message="Store_output was missing required fields.",
                        blocked=False,
                    )
                updated_outputs = self._merge_outputs(task_row["outputs_json"], plan.outputs_json)
                if updated_outputs == task_row["outputs_json"]:
                    store_output_repeats += 1
                else:
                    store_output_repeats = 0
                if store_output_repeats >= 2:
                    question = (
                        "I stored the outputs but I'm still not making progress. "
                        "Can you confirm the correct path or clarify what I should do next?"
                    )
                    updated_block = self._merge_outputs(
                        task_row["outputs_json"], {"blocked_question": question}
                    )
                    now = datetime.utcnow().isoformat()
                    self.db.update_task_status(
                        task_id, status="blocked", outputs_json=updated_block, updated_at=now
                    )
                    self._log_task_event(task_id, "block", {"reason": "repeat_store_output"})
                    if self.hook_manager:
                        self.hook_manager.run(
                            "on_task_end",
                            {
                                "task_id": task_id,
                                "run_id": run_id,
                                "status": "blocked",
                                "inputs_json": task_row["inputs_json"],
                                "outputs_json": updated_block,
                            },
                        )
                    return TaskExecutionResult(
                        llm_calls=llm_calls,
                        steps=steps,
                        tool_records=tool_records,
                        tool_results=tool_results,
                        user_message=question,
                        blocked=True,
                    )
                now = datetime.utcnow().isoformat()
                self.db.update_task_outputs(task_id, outputs_json=updated_outputs, updated_at=now)
                task_row = dict(task_row)
                task_row["outputs_json"] = updated_outputs
                self._log_task_event(task_id, "note", {"outputs_json": plan.outputs_json or {}})
                continue
            if action == "respond":
                cls = getattr(self, "_current_classification", None)
                resp_llm = self._select_response_llm(cls) if cls else self.responder_llm
                resp_temp = self._temperature_for_query(cls.query_type) if cls else None
                response = self._generate_response(
                    user_prompt=user_prompt,
                    memory_context=memory_context,
                    working_memory_summary=working_memory.summary_for_prompt(),
                    tool_results=tool_results,
                    run_summary=run_summary,
                    planner_message=plan.message,
                    response_llm=resp_llm,
                    temperature=resp_temp,
                )
                llm_calls += 1
                if response:
                    user_message = response
                self._log_task_event(task_id, "note", {"message": plan.message or ""})
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(
                    task_id,
                    status="done",
                    outputs_json=task_row["outputs_json"],
                    updated_at=now,
                )
                self._apply_skill_outcome(task_row["outputs_json"], "success")
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {
                            "task_id": task_id,
                            "run_id": run_id,
                            "status": "done",
                            "inputs_json": task_row["inputs_json"],
                            "outputs_json": task_row["outputs_json"],
                        },
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=user_message,
                    blocked=False,
                )
            if action == "ask_user":
                question = plan.message or "I need one clarification to continue. What should I do next?"
                if self.status_fn:
                    self.status_fn("Blocking to ask user a question")
                updated_outputs = self._merge_outputs(
                    task_row["outputs_json"], {"blocked_question": question}
                )
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="blocked", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "block", {"reason": "ask_user", "question": question})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {
                            "task_id": task_id,
                            "run_id": run_id,
                            "status": "blocked",
                            "inputs_json": task_row["inputs_json"],
                            "outputs_json": updated_outputs,
                        },
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
                if self.status_fn:
                    self.status_fn(f"Blocking task: {reason}")
                updated_outputs = self._merge_outputs(task_row["outputs_json"], {"blocked_reason": reason})
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="blocked", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "block", {"reason": reason})
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {
                            "task_id": task_id,
                            "run_id": run_id,
                            "status": "blocked",
                            "inputs_json": task_row["inputs_json"],
                            "outputs_json": updated_outputs,
                        },
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
                if self.status_fn:
                    self.status_fn("Marking task done")
                now = datetime.utcnow().isoformat()
                self.db.update_task_status(task_id, status="done", outputs_json=updated_outputs, updated_at=now)
                self._log_task_event(task_id, "complete", {"outputs_json": plan.outputs_json or {}})
                self._apply_skill_outcome(updated_outputs, "success")
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {
                            "task_id": task_id,
                            "run_id": run_id,
                            "status": "done",
                            "inputs_json": task_row["inputs_json"],
                            "outputs_json": updated_outputs,
                        },
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
                if self.status_fn:
                    self.status_fn("Failing task per plan")
                self.db.update_task_status(task_id, status="failed", outputs_json=task_row["outputs_json"], updated_at=datetime.utcnow().isoformat())
                self._log_task_event(task_id, "fail", {"reason": plan.block_reason or "failed"})
                self._apply_skill_outcome(task_row["outputs_json"], "failure")
                if self.hook_manager:
                    self.hook_manager.run(
                        "on_task_end",
                        {
                            "task_id": task_id,
                            "run_id": run_id,
                            "status": "failed",
                            "inputs_json": task_row["inputs_json"],
                            "outputs_json": task_row["outputs_json"],
                        },
                    )
                return TaskExecutionResult(
                    llm_calls=llm_calls,
                    steps=steps,
                    tool_records=tool_records,
                    tool_results=tool_results,
                    user_message=plan.message or "Task failed.",
                    blocked=False,
                )

        if self.status_fn:
            self.status_fn(
                f"Budget pause: steps={steps} llm_calls={llm_calls} tool_calls={tool_calls}"
            )
        self._log_task_event(task_id, "note", {"reason": "budget_pause"})
        snapshot = self._build_task_state_snapshot(working_memory, tool_results)
        updated_outputs = self._merge_outputs(task_row["outputs_json"], snapshot)
        self.db.update_task_status(
            task_id,
            status="pending",
            outputs_json=updated_outputs,
            updated_at=datetime.utcnow().isoformat(),
        )
        if self.hook_manager:
            self.hook_manager.run(
                "on_task_end",
                {
                    "task_id": task_id,
                    "run_id": run_id,
                    "status": "pending",
                    "inputs_json": task_row["inputs_json"],
                    "outputs_json": updated_outputs,
                },
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
        memory_context: str,
        working_memory_summary: str,
        run_summary: str | None,
        stuck_context: str | None,
        skill_context: str | None,
    ) -> ActionPlan | None:
        summaries = []
        for result in tool_results[-5:]:
            summaries.append(
                f"- {result.name} status={result.status} summary={result.result_summary}"
            )
        outputs: list[str] = []
        for result in tool_results[-5:]:
            raw = result.result_raw or ""
            truncated = raw if len(raw) <= 1500 else f"{raw[:1500]}...[truncated]"
            contains_desktop = "true" if "Desktop" in raw else "false"
            outputs.append(
                f"- {result.name} status={result.status} summary={result.result_summary} "
                f"contains_desktop={contains_desktop} raw={truncated}"
            )
        prompt = build_action_plan_prompt(
            user_prompt=user_prompt,
            task_title=str(task_row["title"]),
            task_description=str(task_row["description"] or ""),
            task_inputs_json=task_row["inputs_json"],
            task_outputs_json=task_row["outputs_json"],
            tool_descriptions=self.tool_descriptions,
            recent_tool_summaries=summaries,
            recent_tool_outputs=outputs,
            agent_context=self.agent_context,
            memory_context=memory_context,
            working_memory_summary=working_memory_summary,
            run_summary=run_summary,
            stuck_context=stuck_context,
            skill_context=skill_context,
            responder_strengths=self.responder_strengths,
        )
        if self.status_fn:
            self.status_fn("Planning next action")
        try:
            response = self.planner_llm.generate(prompt)
        except httpx.HTTPStatusError as exc:
            if self.status_fn:
                self.status_fn(f"LLM error: {_format_llm_http_error(exc)}")
            return None
        except Exception as exc:
            if self.status_fn:
                self.status_fn(f"LLM error: {exc}")
            return None
        plan, error = parse_action_plan(response.content)
        if error or not plan:
            if self.status_fn:
                snippet = response.content.strip().replace("\n", " ")
                if len(snippet) > 140:
                    snippet = snippet[:140] + "...[truncated]"
                self.status_fn(
                    f"Planning output invalid: {error or 'unknown parse error'} | raw={snippet}"
                )
            return None
        if self.status_fn:
            tool_line = ""
            if plan.tool_name:
                tool_line = f" tool={plan.tool_name}"
            if plan.delegate_to:
                tool_line += f" delegate_to={plan.delegate_to}"
            self.status_fn(f"Next action: {plan.next_action_type}{tool_line}")
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

    def _response_mentions_tool_action(self, message: str) -> bool:
        lowered = message.lower()
        markers = [
            "action:",
            "next action:",
            "run tool",
            "use tool",
            "tool call",
            "tool_call",
            "run the tool",
            "execute tool",
            "execute the tool",
        ]
        return any(marker in lowered for marker in markers)

    def _is_plan_chatter(self, message: str) -> bool:
        lowered = message.lower()
        markers = [
            "plan:",
            "next action:",
            "i'll",
            "i will",
            "here's what i'll do",
            "here is what i'll do",
            "let's start by",
        ]
        return any(marker in lowered for marker in markers)

    def _has_successful_tool_result(self, results: list[ToolResult]) -> bool:
        for result in results:
            if result.status in {"ok", "cached"}:
                return True
        return False

    def _is_resume_intent(self, user_input: str) -> bool:
        normalized = user_input.strip().lower()
        if not normalized:
            return False
        if normalized in {
            "continue",
            "continue task",
            "continue the task",
            "continue working",
            "continue working on it",
            "resume",
            "keep going",
            "go on",
            "carry on",
        }:
            return True
        resume_prefixes = (
            "continue ",
            "resume ",
            "keep going",
            "go on",
            "carry on",
            "ok continue",
            "okay continue",
        )
        return normalized.startswith(resume_prefixes)

    @staticmethod
    def _is_run_status_query(user_input: str) -> bool:
        normalized = user_input.strip().lower()
        if not normalized:
            return False
        status_markers = (
            "how we doing",
            "how are we doing",
            "status",
            "progress",
            "what's left",
            "what is left",
            "where are we",
            "update on",
            "any update",
        )
        if any(marker in normalized for marker in status_markers):
            return True
        greeting_prefixes = ("hi", "hey", "hello", "yo")
        if normalized.endswith("?") and any(normalized.startswith(prefix) for prefix in greeting_prefixes):
            return True
        return False

    @staticmethod
    def _is_actionable_followup(user_input: str) -> bool:
        normalized = user_input.strip().lower()
        if not normalized:
            return False
        actionable_markers = (
            "do ",
            "make ",
            "work on ",
            "build ",
            "create ",
            "implement ",
            "fix ",
            "update ",
            "add ",
            "remove ",
            "refactor ",
            "change ",
            "continue ",
            "resume ",
            "please ",
            "can you ",
            "could you ",
        )
        return (
            normalized.startswith(actionable_markers)
            or any(f" {marker}" in normalized for marker in actionable_markers)
        )

    @staticmethod
    def _is_smalltalk_message(user_input: str) -> bool:
        normalized = user_input.strip().lower()
        if not normalized:
            return True
        smalltalk = {
            "thanks",
            "thank you",
            "thx",
            "ok",
            "okay",
            "cool",
            "nice",
            "great",
            "got it",
            "hello",
            "hi",
            "hey",
            "sup",
            "yo",
            "bye",
        }
        if normalized in smalltalk:
            return True
        return len(normalized.split()) <= 2 and normalized.endswith(("!", "."))

    def _should_persist_until_done(
        self, user_input: str, resume_intent: bool, has_active_run: bool
    ) -> bool:
        if resume_intent:
            return True
        if self._is_actionable_followup(user_input):
            return True
        if has_active_run and not self._is_run_status_query(user_input):
            return True
        return False

    def _resolve_resume_prompt(self, run_row: Any) -> str | None:
        direct = str(run_row["user_prompt"] or "").strip()
        if direct and not self._is_resume_intent(direct):
            return direct
        run_id = str(run_row["id"])
        for row in self.db.list_runs(limit=50):
            if str(row["id"]) == run_id:
                continue
            candidate = str(row["user_prompt"] or "").strip()
            if not candidate:
                continue
            if self._is_resume_intent(candidate):
                continue
            if self._is_smalltalk_message(candidate):
                continue
            return candidate
        return None

    def _hydrate_task_state(
        self, task_row: Any, working_memory: WorkingMemory
    ) -> list[ToolResult]:
        outputs_json = task_row["outputs_json"]
        if not outputs_json:
            return []
        try:
            payload = json.loads(outputs_json)
        except json.JSONDecodeError:
            return []
        memory_snapshot = payload.get("working_memory")
        if isinstance(memory_snapshot, dict):
            restored = WorkingMemory.from_snapshot(
                memory_snapshot, user_goal=working_memory.user_goal
            )
            working_memory.constraints = restored.constraints
            working_memory.environment_facts = restored.environment_facts
            working_memory.progress = restored.progress
            working_memory.last_observations = restored.last_observations
            working_memory.recent_tool_calls = restored.recent_tool_calls
            working_memory.steps_since_progress = restored.steps_since_progress
        recent_results = payload.get("recent_tool_results")
        if not isinstance(recent_results, list):
            return []
        hydrated: list[ToolResult] = []
        for entry in recent_results:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            status = entry.get("status")
            summary = entry.get("summary", "")
            raw = entry.get("raw", "")
            if not isinstance(name, str) or not isinstance(status, str):
                continue
            hydrated.append(
                ToolResult(
                    id=entry.get("id") if isinstance(entry.get("id"), str) else f"seed_{uuid.uuid4().hex}",
                    name=name,
                    status=status,
                    started_at="",
                    ended_at="",
                    result_ref=entry.get("result_ref") if isinstance(entry.get("result_ref"), str) else "task_state",
                    result_summary=summary if isinstance(summary, str) else "",
                    result_raw=raw if isinstance(raw, str) else "",
                )
            )
        return hydrated

    def _build_task_state_snapshot(
        self, working_memory: WorkingMemory, tool_results: list[ToolResult]
    ) -> dict[str, Any]:
        recent_tool_results: list[dict[str, Any]] = []
        for result in tool_results[-5:]:
            raw = result.result_raw or ""
            truncated = raw if len(raw) <= 1500 else f"{raw[:1500]}...[truncated]"
            recent_tool_results.append(
                {
                    "id": result.id,
                    "name": result.name,
                    "status": result.status,
                    "summary": result.result_summary,
                    "raw": truncated,
                    "result_ref": result.result_ref,
                }
            )
        return {
            "working_memory": working_memory.snapshot(),
            "recent_tool_results": recent_tool_results,
        }

    def _resolve_blocked_task_response(self, run_row: Any, user_input: str) -> None:
        task_id = run_row["current_task_id"]
        if not task_id:
            return
        task = self.db.fetch_task(task_id)
        if not task or task["status"] != "blocked":
            return
        outputs_json = task["outputs_json"]
        if not outputs_json:
            return
        try:
            payload = json.loads(outputs_json)
        except json.JSONDecodeError:
            return
        if not payload.get("blocked_question"):
            return
        if payload.get("blocked_answer"):
            return
        updated_outputs = self._merge_outputs(
            outputs_json, {"blocked_answer": user_input.strip()}
        )
        now = datetime.utcnow().isoformat()
        self.db.update_task_status(task_id, status="done", outputs_json=updated_outputs, updated_at=now)
        self.db.update_run_status(str(run_row["id"]), status="active", current_task_id=None, last_error=None)
        self._log_task_event(task_id, "complete", {"blocked_answer": user_input.strip()})

    def _log_task_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.db.insert_task_event(
            event_id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            payload=json.dumps(payload),
        )

    def _build_remaining_tasks_message(self, tasks_rows: list[Any]) -> str:
        """Build a user-facing message listing tasks that are still pending."""
        pending = [
            row for row in tasks_rows
            if row["status"] not in {"done", "skipped"}
        ]
        if not pending:
            return "Progress saved. Continuing next turn."
        items = "\n".join(
            f"  {i + 1}) {row['title']}" for i, row in enumerate(pending)
        )
        return (
            f"I've made progress but still need to:\n{items}\n"
            "I will continue in the next turn."
        )

    def _count_consecutive_review_incompletes(self, run_id: str) -> int:
        """Return the consecutive review_incomplete count from the run's last_error field.

        The last_error field stores ``review_incomplete:N`` where N is the count.
        Returns 0 if no review_incomplete history exists.
        """
        run_row = self.db.fetch_run(run_id)
        if not run_row:
            return 0
        last_error = str(run_row["last_error"] or "")
        if last_error.startswith("review_incomplete:"):
            try:
                return int(last_error.split(":")[1])
            except (IndexError, ValueError):
                return 1 if "review_incomplete" in last_error else 0
        if last_error == "review_incomplete":
            return 1
        return 0

    def _all_tasks_complete(self, tasks_rows: list[Any]) -> bool:
        for row in tasks_rows:
            if row["status"] not in {"done", "skipped"}:
                return False
        return True

    def _completion_review(
        self, run_id: str, user_prompt: str, tasks_rows: list[Any], memory_context: str
    ) -> tuple[str | None, int, int]:
        summaries: list[dict[str, Any]] = []
        for row in tasks_rows:
            summaries.append(
                {
                    "ordinal": row["ordinal"],
                    "title": row["title"],
                    "description": row["description"],
                    "status": row["status"],
                    "inputs_json": row["inputs_json"],
                    "outputs_json": row["outputs_json"],
                }
            )
        prompt = build_completion_review_prompt(user_prompt, summaries, memory_context)
        try:
            response = self.planner_llm.generate(prompt)
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            return f"Completion review failed: {_format_llm_http_error(exc)}", 1, 1
        except Exception as exc:  # noqa: BLE001
            return f"Completion review failed: {exc}", 1, 1
        review, error = parse_completion_review(response.content)
        if error or not review:
            return "Completion review failed; will continue next turn.", 1, 1
        if review.overall_status == "incomplete":
            self._append_missing_tasks(run_id, review.missing_items)
            # Track consecutive review_incomplete count in last_error field.
            current_count = self._count_consecutive_review_incompletes(run_id)
            self.db.update_run_status(
                run_id, status="active",
                last_error=f"review_incomplete:{current_count + 1}",
            )
            # Record a lightweight failure reflection after repeated incompletes.
            if current_count + 1 >= 2:
                self._run_self_reflection(run_id, user_prompt, tasks_rows, memory_context, outcome="incomplete")
            return None, 1, 1
        # Run self-reflection before marking done so the reflection can reference the final task state.
        self._run_self_reflection(run_id, user_prompt, tasks_rows, memory_context, outcome="complete")
        self.db.update_run_status(run_id, status="done", current_task_id=None, last_error=None)
        return review.user_message, 1, 1

    def _run_self_reflection(
        self,
        run_id: str,
        user_prompt: str,
        tasks_rows: list[Any],
        memory_context: str,
        outcome: str,
    ) -> None:
        """Generate and store a self-reflection entry, plus stage durable learnings."""
        try:
            task_summaries: list[dict[str, Any]] = []
            for row in tasks_rows:
                task_summaries.append(
                    {
                        "ordinal": row.get("ordinal"),
                        "title": row.get("title"),
                        "status": row.get("status"),
                        "description": row.get("description"),
                        "outputs_json": row.get("outputs_json"),
                    }
                )
            prompt = build_reflection_prompt(
                user_prompt=user_prompt,
                task_summaries=task_summaries,
                memory_context=memory_context,
                outcome=outcome,
            )
            response = self.planner_llm.generate(prompt)
            parsed = parse_reflection_response(response.content)
            if not parsed:
                return
            ts = datetime.utcnow().isoformat()
            reflection_id = self.db.insert_reflection(
                run_id=run_id,
                timestamp=ts,
                success=parsed.success,
                what_worked=parsed.what_worked,
                what_failed=parsed.what_failed,
                next_time=parsed.next_time,
                metrics_json=json.dumps(parsed.metrics) if parsed.metrics else None,
            )
            # Vectorize reflection so it can be retrieved later.
            vi = getattr(self.retriever, "vector_index", None)
            if vi is not None:
                try:
                    vi.add(
                        item_type="reflection",
                        item_id=str(reflection_id),
                        text=f"worked: {parsed.what_worked}\nfailed: {parsed.what_failed}\nnext: {parsed.next_time}",
                    )
                except Exception:
                    pass

            # Stage durable memory candidates for later user validation.
            for statement in parsed.memory_candidates[:10]:
                payload = {
                    "statement": statement,
                    "confidence": 0.55,
                    "awaiting_confirmation": True,
                }
                self.db.insert_staged_item(
                    item_id=str(uuid.uuid4()),
                    kind="fact",
                    payload=json.dumps(payload),
                    status="pending",
                    created_at=ts,
                    source_ref=f"reflection:{reflection_id}",
                    provenance_type=ProvenanceType.INFERRED.value,
                    next_check_at=ts,
                )

            # Stage preference candidates.
            for key, value in list(parsed.preference_candidates.items())[:10]:
                pref_payload = {
                    "key": key,
                    "value": value,
                    "confidence": 0.75,
                    "awaiting_confirmation": True,
                }
                self.db.insert_staged_item(
                    item_id=str(uuid.uuid4()),
                    kind="preference",
                    payload=json.dumps(pref_payload),
                    status="pending",
                    created_at=ts,
                    source_ref=f"reflection:{reflection_id}",
                    provenance_type=ProvenanceType.INFERRED.value,
                    next_check_at=ts,
                )
        except Exception:
            return

    def _append_missing_tasks(self, run_id: str, missing_items: list[str]) -> None:
        if not missing_items:
            return
        tasks_rows = self.db.fetch_tasks_for_run(run_id)
        start_ordinal = max(int(row["ordinal"]) for row in tasks_rows) + 1 if tasks_rows else 0
        now = datetime.utcnow().isoformat()
        for idx, item in enumerate(missing_items):
            title = item.strip() or "Address missing item"
            inputs_json = json.dumps({"verification": f"Verify: {title}", "dependencies": []})
            self.db.insert_task(
                task_id=str(uuid.uuid4()),
                run_id=run_id,
                parent_task_id=None,
                ordinal=start_ordinal + idx,
                title=title,
                description=f"Previous attempt missed this requirement: {title}. Address it now.",
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

    def _classify_request(self, user_input: str) -> ScoringResult:
        return classify_request(user_input)

    def _should_use_responder_only(
        self, classification: ScoringResult, user_input: str
    ) -> bool:
        if classification.tier == "SIMPLE":
            if self._is_actionable_followup(user_input):
                return False
            if self._is_smalltalk_message(user_input):
                return True
            # Route non-conversational SIMPLE prompts through tasking so
            # short project updates (e.g. "we are making X") can drive execution.
            return classification.query_type == "conversational"
        # Conversational MEDIUM queries don't need the planner
        if classification.tier == "MEDIUM" and classification.query_type == "conversational":
            return True
        return False

    def _select_response_llm(self, classification: ScoringResult) -> LLMClient:
        """Pick the best LLM for generating the user-facing response."""
        if classification.query_type == "coding":
            return self.planner_llm
        return self.responder_llm

    @staticmethod
    def _temperature_for_query(query_type: str) -> float:
        """Return an appropriate temperature for the query type."""
        if query_type == "coding":
            return 0.15
        if query_type == "conversational":
            return 0.55
        return 0.3

    def _is_prompt_related_to_run(self, user_input: str, run_row: Any) -> bool:
        tasks = self.db.fetch_tasks_for_run(str(run_row["id"]))
        if not tasks:
            return False
        task_lines: list[str] = []
        for row in tasks[:6]:
            title = str(row["title"]) if row["title"] else ""
            description = str(row["description"]) if row["description"] else ""
            if description:
                task_lines.append(f"- {title}: {description}")
            else:
                task_lines.append(f"- {title}")
        prompt = "\n".join(
            [
                "You are routing a user message against an active task run.",
                "Return YES if the message is directly related to continuing or modifying the tasks.",
                "Return NO if it is unrelated.",
                "Return only YES or NO.",
                "Active tasks:",
                *task_lines,
                f"User message: {user_input}",
            ]
        )
        try:
            response = self.responder_llm.generate(prompt)
            answer = response.content.strip().lower()
        except Exception:
            return False
        return answer.startswith("y")

    def _respond_to_unrelated_prompt(
        self,
        user_input: str,
        retrieval_context: Any,
        working_memory: WorkingMemory,
        active_run: Any,
    ) -> TaskRunnerResult:
        cls = getattr(self, "_current_classification", None)
        resp_llm = self._select_response_llm(cls) if cls else self.responder_llm
        resp_temp = self._temperature_for_query(cls.query_type) if cls else None
        response = self._generate_response(
            user_prompt=user_input,
            memory_context=format_retrieval_context(retrieval_context.bundle),
            working_memory_summary=working_memory.summary_for_prompt(),
            tool_results=[],
            run_summary=None,
            planner_message=None,
            response_llm=resp_llm,
            temperature=resp_temp,
        )
        guarded = self.guardrails.enforce(response, retrieval_context.bundle)
        return TaskRunnerResult(
            message=guarded.safe_output,
            tool_records=[],
            tool_results=[],
            run_id=str(active_run["id"]),
            llm_calls=1,
            steps=1,
            tool_calls=0,
        )

    def _respond_with_active_run_status(self, active_run: Any) -> TaskRunnerResult:
        run_id = str(active_run["id"])
        tasks = self.db.fetch_tasks_for_run(run_id)
        done = sum(1 for row in tasks if row["status"] in {"done", "skipped"})
        total = len(tasks)
        current = next((row for row in tasks if row["status"] == "active"), None)
        if current is None and active_run["current_task_id"]:
            current = next(
                (row for row in tasks if str(row["id"]) == str(active_run["current_task_id"])),
                None,
            )
        pending_count = sum(1 for row in tasks if row["status"] in {"pending", "blocked", "active"})
        lines = [f"Run in progress: {done}/{total} tasks complete."]
        if current is not None:
            lines.append(f"Current task: {current['title']}")
        elif pending_count > 0:
            lines.append("Current task: selecting next pending task.")
        lines.append("Say `continue` to keep executing, or give a concrete change request.")
        return TaskRunnerResult(
            message="\n".join(lines),
            tool_records=[],
            tool_results=[],
            run_id=run_id,
            llm_calls=0,
            steps=0,
            tool_calls=0,
        )

    @staticmethod
    def _response_needs_retry(response_text: str, planner_message: str | None) -> bool:
        """Check if the responder output looks incomplete and should be retried."""
        if not planner_message:
            return False
        text = response_text.strip()
        if not text:
            return True
        if len(text) < 20:
            return True
        low = text.lower()
        uncertain_phrases = [
            "i'm not sure",
            "i don't have enough information",
            "i cannot determine",
            "i don't know",
            "unable to provide",
        ]
        return any(phrase in low for phrase in uncertain_phrases)

    def _generate_response(
        self,
        user_prompt: str,
        memory_context: str,
        working_memory_summary: str,
        tool_results: list[ToolResult],
        run_summary: str | None,
        planner_message: str | None,
        response_llm: LLMClient | None = None,
        temperature: float | None = None,
    ) -> str:
        llm = response_llm or self.responder_llm
        summaries = [
            f"- {result.name} status={result.status} summary={result.result_summary}"
            for result in tool_results[-5:]
        ]
        outputs = []
        for result in tool_results[-5:]:
            raw = result.result_raw or ""
            truncated = raw if len(raw) <= 1500 else f"{raw[:1500]}...[truncated]"
            outputs.append(
                f"- {result.name} status={result.status} summary={result.result_summary} raw={truncated}"
            )
        prompt = build_response_prompt(
            user_prompt=user_prompt,
            memory_context=memory_context,
            working_memory_summary=working_memory_summary,
            agent_context=self.agent_context,
            planner_message=planner_message,
            recent_tool_summaries=summaries,
            recent_tool_outputs=outputs,
            run_summary=run_summary,
        )
        try:
            response = llm.generate(prompt, temperature=temperature)
        except httpx.HTTPStatusError as exc:
            error_message = _format_llm_http_error(exc)
            return planner_message or f"I hit a response error: {error_message}"
        except Exception as exc:
            return planner_message or f"I hit a response error: {exc}"

        # Feedback loop: if the response looks incomplete and we have planner
        # context, retry once with an augmented prompt.
        if self._response_needs_retry(response.content, planner_message):
            if self.status_fn:
                self.status_fn("Response quality check failed; retrying with augmented context")
            augmented_prompt = build_response_prompt(
                user_prompt=user_prompt,
                memory_context=memory_context,
                working_memory_summary=working_memory_summary,
                agent_context=self.agent_context,
                planner_message=(
                    f"AUTHORITATIVE PLANNER ANSWER (use as primary source):\n{planner_message}"
                ),
                recent_tool_summaries=summaries,
                recent_tool_outputs=outputs,
                run_summary=run_summary,
            )
            try:
                retry_response = llm.generate(augmented_prompt, temperature=temperature)
                return retry_response.content
            except Exception:
                # Fall through to original response on retry failure
                pass

        return response.content


@dataclass
class TaskExecutionResult:
    llm_calls: int
    steps: int
    tool_records: list[ToolRecord]
    tool_results: list[ToolResult]
    user_message: str | None
    blocked: bool
