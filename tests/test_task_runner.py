import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.guardrails import Guardrails
from tali.classifier import ScoringResult
from tali.llm import LLMResponse
from tali.retrieval import Retriever
from tali.task_runner import TaskRunner, TaskRunnerSettings
from tali.tools.protocol import ToolResult
from tali.working_memory import WorkingMemory, summarize_tool_result


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        if self.calls >= len(self.responses):
            raise AssertionError("LLM called more than expected")
        content = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(content=content, model="fake")


class ScriptedLLM:
    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        if self.calls >= len(self.script):
            raise AssertionError("LLM called more than expected")
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=str(item), model="fake")


class _FakeRegistry:
    """Minimal registry stub that always finds a tool."""

    def get(self, name: str):
        return True  # pretend every tool exists

    def list_tools(self):
        return []


class FakeToolRunner:
    def __init__(self, results_by_call=None) -> None:
        self.results_by_call = results_by_call or []
        self.calls = 0
        self.tool_calls = []
        self.registry = _FakeRegistry()

    def run(self, tool_calls, prompt_fn):
        self.tool_calls.extend(tool_calls)
        if callable(self.results_by_call):
            return self.results_by_call(tool_calls)
        if self.calls < len(self.results_by_call):
            results = self.results_by_call[self.calls]
        else:
            results = []
        self.calls += 1
        return results, []


class TaskRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()
        self.retriever = Retriever(self.db)
        self.guardrails = Guardrails()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_decompose_and_complete_run(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Do first thing",
                    "description": "Step one.",
                    "requires_tools": False,
                    "verification": "First thing done.",
                    "dependencies": [],
                },
                {
                    "title": "Do second thing",
                    "description": "Step two.",
                    "requires_tools": False,
                    "verification": "Second thing done.",
                    "dependencies": [0],
                },
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [
                {"task_ordinal": 0, "status": "ok", "note": ""},
                {"task_ordinal": 1, "status": "ok", "note": ""},
            ],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(action_done),
                json.dumps(action_done),
                json.dumps(review),
                "All set.",  # responder polish of completion review
            ]
        )
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        result = runner.run_turn(
            "First implement the algorithm, then verify it works",
            prompt_fn=lambda _: "",
        )
        self.assertIn("All set", result.message)
        run = self.db.fetch_active_run()
        self.assertIsNone(run)
        tasks = self.db.fetch_tasks_for_run(self.db.fetch_run(self._last_run_id())["id"])
        statuses = [row["status"] for row in tasks]
        self.assertEqual(statuses, ["done", "done"])

    def test_budgets_pause(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Task one",
                    "description": "First.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
                {
                    "title": "Task two",
                    "description": "Second.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Complete.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(action_done),
                json.dumps(action_done),
                json.dumps(review),
            ]
        )
        settings = TaskRunnerSettings(max_tasks_per_turn=1, max_total_llm_calls_per_run_per_turn=2)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        runner._should_persist_until_done = (  # type: ignore[method-assign]
            lambda user_input, resume_intent, has_active_run: False
        )
        runner.run_turn(
            "First implement the algorithm, then optimize the database",
            prompt_fn=lambda _: "",
        )
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        tasks = self.db.fetch_tasks_for_run(run["id"])
        statuses = [row["status"] for row in tasks]
        self.assertEqual(statuses, ["done", "pending"])

    def test_continues_across_batches_without_user_continue(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Task one",
                    "description": "First.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
                {
                    "title": "Task two",
                    "description": "Second.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [
                {"task_ordinal": 0, "status": "ok", "note": ""},
                {"task_ordinal": 1, "status": "ok", "note": ""},
            ],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Complete.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(action_done),
                json.dumps(action_done),
                json.dumps(review),
                "Complete.",
            ]
        )
        settings = TaskRunnerSettings(max_tasks_per_turn=1)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        result = runner.run_turn(
            "First implement the algorithm, then optimize the database",
            prompt_fn=lambda _: "",
        )
        self.assertIn("Complete", result.message)
        run = self.db.fetch_active_run()
        self.assertIsNone(run)
        tasks = self.db.fetch_tasks_for_run(self.db.fetch_run(self._last_run_id())["id"])
        self.assertEqual([row["status"] for row in tasks], ["done", "done"])

    def test_status_check_does_not_advance_active_run(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Task one",
                    "description": "First.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
                {
                    "title": "Task two",
                    "description": "Second.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                },
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        llm = FakeLLM([json.dumps(decomposition), json.dumps(action_done)])
        settings = TaskRunnerSettings(max_tasks_per_turn=1, max_total_llm_calls_per_run_per_turn=2)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        runner.run_turn(
            "First implement the app skeleton, then optimize the data layer",
            prompt_fn=lambda _: "",
        )
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        before = self.db.fetch_tasks_for_run(run["id"])
        self.assertEqual([row["status"] for row in before], ["done", "pending"])

        result = runner.run_turn(
            "hi bob how we doing on making our app?",
            prompt_fn=lambda _: "",
        )
        after = self.db.fetch_tasks_for_run(run["id"])
        self.assertEqual([row["status"] for row in after], ["done", "pending"])
        self.assertIn("Run in progress", result.message)
        self.assertIn("continue", result.message.lower())

    def test_simple_actionable_prompt_starts_task_execution(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Build app skeleton",
                    "description": "Create initial app files.",
                    "requires_tools": False,
                    "verification": "Skeleton exists.",
                    "dependencies": [],
                }
            ]
        }
        llm = FakeLLM([json.dumps(decomposition)])
        settings = TaskRunnerSettings(max_total_llm_calls_per_run_per_turn=1)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        result = runner.run_turn("work on the app", prompt_fn=lambda _: "")
        self.assertIsNotNone(result.run_id)
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        tasks = self.db.fetch_tasks_for_run(run["id"])
        self.assertEqual(len(tasks), 1)

    def test_simple_non_actionable_prompt_uses_responder_only(self) -> None:
        llm = FakeLLM(["You're welcome."])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        result = runner.run_turn("thanks", prompt_fn=lambda _: "")
        self.assertIsNone(result.run_id)
        self.assertIsNone(self.db.fetch_active_run())

    def test_simple_general_prompt_routes_to_tasking(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Start app work",
                    "description": "Initialize the app workstream.",
                    "requires_tools": False,
                    "verification": "Workstream initialized.",
                    "dependencies": [],
                }
            ]
        }
        llm = FakeLLM([json.dumps(decomposition)])
        settings = TaskRunnerSettings(max_total_llm_calls_per_run_per_turn=1)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        runner._classify_request = lambda _text: ScoringResult(
            score=-0.2,
            tier="SIMPLE",
            confidence=0.95,
            signals=["short"],
            query_type="general",
        )
        result = runner.run_turn(
            "we are making an alpaca options trading bot ui using flask",
            prompt_fn=lambda _: "",
        )
        self.assertIsNotNone(result.run_id)
        self.assertIsNotNone(self.db.fetch_active_run())

    def test_actionable_prompt_falls_back_when_decomposition_json_invalid(self) -> None:
        llm = FakeLLM(["not-json"])
        settings = TaskRunnerSettings(max_total_llm_calls_per_run_per_turn=1)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=settings,
        )
        result = runner.run_turn("work on the app", prompt_fn=lambda _: "")
        self.assertIsNotNone(result.run_id)
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        tasks = self.db.fetch_tasks_for_run(run["id"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["requires_tools"], 1)

    def test_continue_retries_blocked_run_decomposition_with_original_prompt(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Build app",
                    "description": "Create app files.",
                    "requires_tools": False,
                    "verification": "App exists.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Done.",
        }
        llm = ScriptedLLM(
            [
                TimeoutError("timed out"),  # first decomposition attempt fails
                json.dumps(decomposition),  # continue retries decomposition
                json.dumps(action_done),
                json.dumps(review),
                "Done.",
            ]
        )
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        first = runner.run_turn("build the app", prompt_fn=lambda _: "")
        self.assertIn("LLM request failed", first.message)
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        self.assertEqual(run["status"], "blocked")

        second = runner.run_turn("continue", prompt_fn=lambda _: "")
        self.assertIn("Done", second.message)
        self.assertIsNone(self.db.fetch_active_run())
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE user_prompt = ?",
                ("continue",),
            ).fetchone()
        self.assertEqual(int(row["count"]), 0)

    def test_resume_recovers_prompt_from_recent_runs_when_current_is_continue(self) -> None:
        older = (datetime.utcnow() - timedelta(minutes=2)).isoformat()
        newer = datetime.utcnow().isoformat()
        self.db.insert_run(
            run_id="run-older",
            created_at=older,
            status="failed",
            user_prompt="build the app",
            current_task_id=None,
            last_error="x",
            origin="user",
        )
        self.db.insert_run(
            run_id="run-continue",
            created_at=newer,
            status="blocked",
            user_prompt="continue",
            current_task_id=None,
            last_error="decomposition_failed",
            origin="user",
        )
        decomposition = {
            "tasks": [
                {
                    "title": "Build app",
                    "description": "Create app files.",
                    "requires_tools": False,
                    "verification": "App exists.",
                    "dependencies": [],
                }
            ]
        }
        llm = FakeLLM([json.dumps(decomposition)])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
            settings=TaskRunnerSettings(max_total_llm_calls_per_run_per_turn=1),
        )
        result = runner.run_turn("resume", prompt_fn=lambda _: "")
        self.assertEqual(result.run_id, "run-continue")
        tasks = self.db.fetch_tasks_for_run("run-continue")
        self.assertEqual(len(tasks), 1)
        run = self.db.fetch_run("run-continue")
        self.assertEqual(run["status"], "active")

    def test_blocked_task_asks_one_question(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Clarify requirement",
                    "description": "Need user input.",
                    "requires_tools": False,
                    "verification": "Answered.",
                    "dependencies": [],
                }
            ]
        }
        ask_user = {
            "next_action_type": "ask_user",
            "message": "Which option should I use?",
        }
        llm = FakeLLM([json.dumps(decomposition), json.dumps(ask_user)])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        result = runner.run_turn(
            "First implement the algorithm, then verify the results",
            prompt_fn=lambda _: "",
        )
        self.assertIn("Which option", result.message)
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        self.assertEqual(run["status"], "blocked")

    def test_memory_injected_into_prompts(self) -> None:
        self.db.insert_fact(
            fact_id="fact-1",
            statement="Project codename is Atlas",
            provenance_type="SYSTEM_OBSERVED",
            source_ref="episode-1",
            confidence=0.9,
            created_at="2024-01-01T00:00:00",
            contested=0,
        )
        decomposition = {
            "tasks": [
                {
                    "title": "Do thing",
                    "description": "Step one.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM([
            json.dumps(decomposition),
            json.dumps(action_done),
            json.dumps(review),
            "All set.",  # responder polish
        ])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm, then configure the database",
            prompt_fn=lambda _: "",
        )
        prompt_blob = "\n".join(llm.prompts)
        self.assertIn("Project codename is Atlas", prompt_blob)

    def test_execute_skill_tracks_success(self) -> None:
        self.db.insert_skill(
            skill_id="skill-1",
            name="TestSkill",
            trigger="do test",
            steps=json.dumps(["Step one", "Step two"]),
            created_at="2024-01-01T00:00:00",
            source_ref="episode-1",
        )
        decomposition = {
            "tasks": [
                {
                    "title": "Use skill",
                    "description": "Apply skill.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                }
            ]
        }
        execute_skill = {"next_action_type": "execute_skill", "skill_name": "TestSkill"}
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(execute_skill),
                json.dumps(action_done),
                json.dumps(review),
                "All set.",  # responder polish
            ]
        )
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm, then apply the testing skill",
            prompt_fn=lambda _: "",
        )
        skill = self.db.fetch_skill_by_name("TestSkill")
        self.assertIsNotNone(skill)
        self.assertEqual(skill["success_count"], 1)

    def test_duplicate_tool_call_blocked(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Inspect",
                    "description": "List files.",
                    "requires_tools": True,
                    "verification": "Listed.",
                    "dependencies": [],
                }
            ]
        }
        tool_call = {
            "next_action_type": "tool_call",
            "tool_name": "fs.list",
            "tool_args": {"path": "C:\\X"},
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(tool_call),
                json.dumps(tool_call),
                json.dumps(action_done),
                json.dumps(review),
                "All set.",  # responder polish
            ]
        )
        result = ToolResult(
            id="tc_1",
            name="fs.list",
            status="ok",
            started_at="",
            ended_at="",
            result_ref="tool_call:tc_1",
            result_summary="Listed C:\\X",
            result_raw="Desktop\nfile.txt",
        )
        tool_runner = FakeToolRunner(results_by_call=[[result]])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=tool_runner,
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm to list files, then verify",
            prompt_fn=lambda _: "",
        )
        self.assertEqual(len(tool_runner.tool_calls), 1)

    def test_batched_tool_calls_execute(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Inspect",
                    "description": "List and read files.",
                    "requires_tools": True,
                    "verification": "Listed and read.",
                    "dependencies": [],
                }
            ]
        }
        tool_call = {
            "next_action_type": "tool_call",
            "tool_calls": [
                {"tool_name": "fs.list", "tool_args": {"path": "C:\\X"}},
                {"tool_name": "fs.read", "tool_args": {"path": "C:\\X\\file.txt"}},
            ],
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(tool_call),
                json.dumps(action_done),
                json.dumps(review),
                "All set.",  # responder polish
            ]
        )

        def _results_for_calls(calls):
            results = []
            for call in calls:
                results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        status="ok",
                        started_at="",
                        ended_at="",
                        result_ref=f"tool_call:{call.id}",
                        result_summary="ok",
                        result_raw="",
                    )
                )
            return results, []

        tool_runner = FakeToolRunner(results_by_call=_results_for_calls)
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=tool_runner,
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm to list and read files, then verify",
            prompt_fn=lambda _: "",
        )
        self.assertEqual(len(tool_runner.tool_calls), 2)
        self.assertEqual(tool_runner.tool_calls[0].name, "fs.list")
        self.assertEqual(tool_runner.tool_calls[1].name, "fs.read")

    def test_batched_duplicate_tool_call_blocked(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Inspect",
                    "description": "List files.",
                    "requires_tools": True,
                    "verification": "Listed.",
                    "dependencies": [],
                }
            ]
        }
        tool_call = {
            "next_action_type": "tool_call",
            "tool_calls": [
                {"tool_name": "fs.list", "tool_args": {"path": "C:\\X"}},
            ],
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(tool_call),
                json.dumps(tool_call),
                json.dumps(action_done),
                json.dumps(review),
                "All set.",  # responder polish
            ]
        )
        result = ToolResult(
            id="tc_1",
            name="fs.list",
            status="ok",
            started_at="",
            ended_at="",
            result_ref="tool_call:tc_1",
            result_summary="Listed C:\\X",
            result_raw="Desktop\nfile.txt",
        )
        tool_runner = FakeToolRunner(results_by_call=[[result]])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=tool_runner,
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm to list files, then verify",
            prompt_fn=lambda _: "",
        )
        self.assertEqual(len(tool_runner.tool_calls), 1)

    def test_stuck_progress_forces_replan(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Inspect twice",
                    "description": "Look around.",
                    "requires_tools": True,
                    "verification": "Done.",
                    "dependencies": [],
                }
            ]
        }
        tool_call = {
            "next_action_type": "tool_call",
            "tool_name": "fs.list",
            "tool_args": {"path": "C:\\X"},
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All set.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                json.dumps(tool_call),
                json.dumps(tool_call),
                json.dumps(action_done),
                json.dumps(review),
                "Inspection completed successfully.",  # responder polish
                "Inspection completed successfully.",  # retry (short response triggers feedback loop)
            ]
        )
        result = ToolResult(
            id="tc_1",
            name="fs.list",
            status="ok",
            started_at="",
            ended_at="",
            result_ref="tool_call:tc_1",
            result_summary="",
            result_raw="",
        )
        tool_runner = FakeToolRunner(results_by_call=[[result], [result]])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=tool_runner,
            tool_descriptions="none",
        )
        runner.run_turn(
            "First implement the algorithm to inspect, then verify the results",
            prompt_fn=lambda _: "",
        )
        prompt_blob = "\n".join(llm.prompts)
        self.assertIn("duplicate_success_blocked", prompt_blob)

    def test_durable_facts_stored(self) -> None:
        memory = WorkingMemory(user_goal="Test")
        result = ToolResult(
            id="tc_1",
            name="fs.list",
            status="ok",
            started_at="",
            ended_at="",
            result_ref="tool_call:tc_1",
            result_summary="Listed C:\\X\\Desktop",
            result_raw="file.txt\nDesktop",
        )
        observations, durable = summarize_tool_result("fs.list", {"path": "C:\\X"}, result)
        memory.note_observations(observations, durable)
        self.assertIn("listed_paths", memory.environment_facts)
        self.assertIn("C:\\X\\Desktop", memory.environment_facts["listed_paths"])

    def test_completion_review_includes_verification(self) -> None:
        """Completion review summaries should include inputs_json with verification."""
        decomposition = {
            "tasks": [
                {
                    "title": "Create file",
                    "description": "Write a file.",
                    "requires_tools": False,
                    "verification": "file_created_successfully",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"created": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "File created.",
        }
        llm = FakeLLM([
            json.dumps(decomposition),
            json.dumps(action_done),
            json.dumps(review),
            "File created.",  # responder polish
        ])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        runner.run_turn(
            "First create the file with proper content, then verify the result",
            prompt_fn=lambda _: "",
        )
        # The completion review prompt (3rd LLM call, index 2) should contain
        # the verification string from inputs_json.
        review_prompt = llm.prompts[2]
        self.assertIn("file_created_successfully", review_prompt)
        self.assertIn("inputs_json", review_prompt)

    def test_auto_continue_on_incomplete_review(self) -> None:
        """When review returns incomplete, the runner should retry within the same turn."""
        decomposition = {
            "tasks": [
                {
                    "title": "Step A",
                    "description": "Do A.",
                    "requires_tools": False,
                    "verification": "A done.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        incomplete_review = {
            "overall_status": "incomplete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": ["Also do B"],
            "assumptions": [],
            "user_message": "Step B is still missing.",
        }
        complete_review = {
            "overall_status": "complete",
            "checks": [
                {"task_ordinal": 0, "status": "ok", "note": ""},
                {"task_ordinal": 1, "status": "ok", "note": ""},
            ],
            "missing_items": [],
            "assumptions": [],
            "user_message": "All done including B.",
        }
        llm = FakeLLM([
            json.dumps(decomposition),       # decompose
            json.dumps(action_done),          # task A mark_done
            json.dumps(incomplete_review),    # first review -> incomplete
            json.dumps(action_done),          # task B (missing item) mark_done
            json.dumps(complete_review),      # second review -> complete
            "All done including B.",          # responder polish
        ])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        result = runner.run_turn(
            "First do step A completely, then also do step B as a follow-up",
            prompt_fn=lambda _: "",
        )
        self.assertIn("All done", result.message)
        # Verify the missing task was created and completed
        run_id = self._last_run_id()
        tasks = self.db.fetch_tasks_for_run(run_id)
        self.assertEqual(len(tasks), 2)
        statuses = [row["status"] for row in tasks]
        self.assertEqual(statuses, ["done", "done"])

    def test_action_planning_error_retries_within_same_turn(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Build app",
                    "description": "Create initial app structure.",
                    "requires_tools": False,
                    "verification": "App skeleton exists.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Done.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                "not-json",  # first action-plan attempt fails
                json.dumps(action_done),  # retry succeeds
                json.dumps(review),
                "Done.",
            ]
        )
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        result = runner.run_turn("build the app", prompt_fn=lambda _: "")
        self.assertNotIn("planning error", result.message.lower())
        self.assertIn("Done", result.message)
        self.assertIsNone(self.db.fetch_active_run())

    def test_planner_invalid_loop_forces_bootstrap_tool_call(self) -> None:
        decomposition = {
            "tasks": [
                {
                    "title": "Build app",
                    "description": "Create initial app structure.",
                    "requires_tools": True,
                    "verification": "App skeleton exists.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        review = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Done.",
        }
        llm = FakeLLM(
            [
                json.dumps(decomposition),
                "bad-json-1",
                "bad-json-2",
                "bad-json-3",
                json.dumps(action_done),
                json.dumps(review),
                "Done.",
            ]
        )
        tool_result = ToolResult(
            id="tc_bootstrap",
            name="fs.list",
            status="ok",
            started_at="",
            ended_at="",
            result_ref="tool_call:tc_bootstrap",
            result_summary="Listed root",
            result_raw="src\nREADME.md",
        )
        tool_runner = FakeToolRunner(results_by_call=[[tool_result]])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=tool_runner,
            tool_descriptions="none",
        )
        result = runner.run_turn("build the app", prompt_fn=lambda _: "")
        self.assertIn("Done", result.message)
        self.assertGreaterEqual(len(tool_runner.tool_calls), 1)
        self.assertEqual(tool_runner.tool_calls[0].name, "fs.list")

    def test_build_remaining_tasks_message(self) -> None:
        """_build_remaining_tasks_message should list pending tasks."""
        llm = FakeLLM([])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        # Simulate task rows with mixed statuses
        fake_rows = [
            {"title": "Done task", "status": "done"},
            {"title": "Pending task one", "status": "pending"},
            {"title": "Pending task two", "status": "active"},
            {"title": "Skipped task", "status": "skipped"},
        ]
        msg = runner._build_remaining_tasks_message(fake_rows)
        self.assertIn("Pending task one", msg)
        self.assertIn("Pending task two", msg)
        self.assertNotIn("Done task", msg)
        self.assertNotIn("Skipped task", msg)
        self.assertIn("still need to", msg)

    def test_build_remaining_tasks_message_all_done(self) -> None:
        """When all tasks are done, fallback message should be returned."""
        llm = FakeLLM([])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        fake_rows = [
            {"title": "Task A", "status": "done"},
            {"title": "Task B", "status": "skipped"},
        ]
        msg = runner._build_remaining_tasks_message(fake_rows)
        self.assertEqual(msg, "Progress saved. Continuing next turn.")

    def test_circuit_breaker_blocks_after_repeated_incompletes(self) -> None:
        """Circuit breaker should block the run after 2 consecutive review_incomplete turns."""
        decomposition = {
            "tasks": [
                {
                    "title": "Do thing",
                    "description": "Step one.",
                    "requires_tools": False,
                    "verification": "Done.",
                    "dependencies": [],
                }
            ]
        }
        action_done = {"next_action_type": "mark_done", "outputs_json": {"ok": True}}
        llm = FakeLLM([
            json.dumps(decomposition),
            json.dumps(action_done),
        ])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        # Simulate a run that already has review_incomplete:2 in last_error
        from datetime import datetime
        run_id = "test-circuit-breaker-run"
        self.db.insert_run(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            user_prompt="Do thing",
            current_task_id=None,
            last_error="review_incomplete:2",
            origin="user",
        )
        self.db.insert_task(
            task_id="task-cb-1",
            run_id=run_id,
            parent_task_id=None,
            ordinal=0,
            title="Remaining task",
            description="Still needs doing.",
            status="pending",
            inputs_json=json.dumps({"verification": "Done.", "dependencies": []}),
            outputs_json=None,
            requires_tools=0,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        result = runner.run_turn("continue", prompt_fn=lambda _: "")
        self.assertIn("attempted this request multiple times", result.message)
        self.assertIn("Remaining task", result.message)
        run = self.db.fetch_run(run_id)
        self.assertEqual(run["status"], "blocked")
        self.assertEqual(run["last_error"], "stuck_review_incomplete")

    def test_count_consecutive_review_incompletes(self) -> None:
        """_count_consecutive_review_incompletes should parse the count from last_error."""
        llm = FakeLLM([])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        from datetime import datetime
        # Run with review_incomplete:3
        run_id = "test-count-run"
        self.db.insert_run(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            user_prompt="test",
            current_task_id=None,
            last_error="review_incomplete:3",
            origin="user",
        )
        self.assertEqual(runner._count_consecutive_review_incompletes(run_id), 3)
        # Run with no error
        run_id2 = "test-count-run-2"
        self.db.insert_run(
            run_id=run_id2,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            user_prompt="test",
            current_task_id=None,
            last_error=None,
            origin="user",
        )
        self.assertEqual(runner._count_consecutive_review_incompletes(run_id2), 0)
        # Run with old-format "review_incomplete" (no colon)
        run_id3 = "test-count-run-3"
        self.db.insert_run(
            run_id=run_id3,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            user_prompt="test",
            current_task_id=None,
            last_error="review_incomplete",
            origin="user",
        )
        self.assertEqual(runner._count_consecutive_review_incompletes(run_id3), 1)

    def test_append_missing_tasks_enriched_description(self) -> None:
        """_append_missing_tasks should create tasks with enriched descriptions."""
        llm = FakeLLM([])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        from datetime import datetime
        run_id = "test-missing-tasks-run"
        self.db.insert_run(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            user_prompt="test",
            current_task_id=None,
            last_error=None,
            origin="user",
        )
        runner._append_missing_tasks(run_id, ["Write unit tests", "Update docs"])
        tasks = self.db.fetch_tasks_for_run(run_id)
        self.assertEqual(len(tasks), 2)
        self.assertIn("Previous attempt missed this requirement", tasks[0]["description"])
        self.assertIn("Write unit tests", tasks[0]["description"])
        # Verification should be specific, not generic
        inputs = json.loads(tasks[0]["inputs_json"])
        self.assertIn("Write unit tests", inputs["verification"])

    def _last_run_id(self) -> str:
        with self.db.connect() as connection:
            cursor = connection.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return str(row["id"])


if __name__ == "__main__":
    unittest.main()
