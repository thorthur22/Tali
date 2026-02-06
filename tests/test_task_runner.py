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


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        if self.calls >= len(self.responses):
            raise AssertionError("LLM called more than expected")
        content = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(content=content, model="fake")


class FakeToolRunner:
    def run(self, tool_calls, prompt_fn):
        return [], []


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

    def _last_run_id(self) -> str:
        with self.db.connect() as connection:
            cursor = connection.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return str(row["id"])


if __name__ == "__main__":
    unittest.main()
