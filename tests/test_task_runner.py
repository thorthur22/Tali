import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.guardrails import Guardrails
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

    def generate(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        if self.calls >= len(self.responses):
            raise AssertionError("LLM called more than expected")
        content = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(content=content, model="fake")


class FakeToolRunner:
    def __init__(self, results_by_call=None) -> None:
        self.results_by_call = results_by_call or []
        self.calls = 0
        self.tool_calls = []

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
        result = runner.run_turn("Do two things", prompt_fn=lambda _: "")
        self.assertIn("Completion Review", result.message)
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
        runner.run_turn("Do two tasks", prompt_fn=lambda _: "")
        run = self.db.fetch_active_run()
        self.assertIsNotNone(run)
        tasks = self.db.fetch_tasks_for_run(run["id"])
        statuses = [row["status"] for row in tasks]
        self.assertEqual(statuses, ["done", "pending"])

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
        result = runner.run_turn("Do the thing", prompt_fn=lambda _: "")
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
        llm = FakeLLM([json.dumps(decomposition), json.dumps(action_done), json.dumps(review)])
        runner = TaskRunner(
            db=self.db,
            llm=llm,
            retriever=self.retriever,
            guardrails=self.guardrails,
            tool_runner=FakeToolRunner(),
            tool_descriptions="none",
        )
        runner.run_turn("Do thing", prompt_fn=lambda _: "")
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
        runner.run_turn("Use skill", prompt_fn=lambda _: "")
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
        runner.run_turn("List files", prompt_fn=lambda _: "")
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
        runner.run_turn("Inspect", prompt_fn=lambda _: "")
        prompt_blob = "\n".join(llm.prompts)
        self.assertIn("Stuck detected: no progress in 2 tool calls", prompt_blob)

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

    def _last_run_id(self) -> str:
        with self.db.connect() as connection:
            cursor = connection.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return str(row["id"])


if __name__ == "__main__":
    unittest.main()
