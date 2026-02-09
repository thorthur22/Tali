import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.tasking import parse_action_plan, parse_completion_review, parse_decomposition


class TaskingParserTests(unittest.TestCase):
    def test_parse_action_plan_accepts_fenced_json(self) -> None:
        payload = {
            "next_action_type": "tool_call",
            "tool_name": "fs.list",
            "tool_args": {},
            "tool_purpose": "List root",
        }
        text = f"```json\n{json.dumps(payload)}\n```"
        plan, error = parse_action_plan(text)
        self.assertIsNone(error)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.tool_name, "fs.list")

    def test_parse_action_plan_accepts_wrapped_json(self) -> None:
        payload = {
            "next_action_type": "mark_done",
            "outputs_json": {"ok": True},
        }
        text = f"Here is the JSON:\n{json.dumps(payload)}\nThanks."
        plan, error = parse_action_plan(text)
        self.assertIsNone(error)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.next_action_type, "mark_done")

    def test_parse_decomposition_accepts_fenced_json(self) -> None:
        payload = {
            "tasks": [
                {
                    "title": "Task",
                    "description": "Do task.",
                    "requires_tools": True,
                    "verification": "Done.",
                    "dependencies": [],
                }
            ]
        }
        text = f"```json\n{json.dumps(payload)}\n```"
        tasks, error = parse_decomposition(text)
        self.assertIsNone(error)
        self.assertIsNotNone(tasks)
        self.assertEqual(len(tasks), 1)

    def test_parse_completion_review_accepts_fenced_json(self) -> None:
        payload = {
            "overall_status": "complete",
            "checks": [{"task_ordinal": 0, "status": "ok", "note": ""}],
            "missing_items": [],
            "assumptions": [],
            "user_message": "Done.",
        }
        text = f"```json\n{json.dumps(payload)}\n```"
        review, error = parse_completion_review(text)
        self.assertIsNone(error)
        self.assertIsNotNone(review)
        self.assertEqual(review.overall_status, "complete")


if __name__ == "__main__":
    unittest.main()
