import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.hooks.core import HookContext
from tali.hooks.task_verification import (
    _extract_keywords,
    _handle_task_end,
)


class TestExtractKeywords(unittest.TestCase):
    def test_extracts_meaningful_words(self) -> None:
        text = "File updated with new content"
        keywords = _extract_keywords(text)
        self.assertIn("file", keywords)
        self.assertIn("updated", keywords)
        self.assertIn("content", keywords)
        # "with" and "new" are too short or stop words
        self.assertNotIn("with", keywords)
        self.assertNotIn("new", keywords)

    def test_excludes_stop_words(self) -> None:
        text = "Verify that the configuration is properly done"
        keywords = _extract_keywords(text)
        self.assertNotIn("that", keywords)
        self.assertNotIn("done", keywords)
        self.assertNotIn("verify", keywords)
        self.assertIn("configuration", keywords)
        self.assertIn("properly", keywords)

    def test_empty_string(self) -> None:
        self.assertEqual(_extract_keywords(""), [])

    def test_all_short_words(self) -> None:
        self.assertEqual(_extract_keywords("a b c d"), [])


class TestTaskVerificationHook(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_context(self, payload: dict) -> HookContext:
        return HookContext(event="on_task_end", payload=payload, db=self.db)

    def test_skips_non_done_status(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "blocked",
            "inputs_json": json.dumps({"verification": "file created"}),
            "outputs_json": json.dumps({"created": True}),
        })
        result = _handle_task_end(context)
        self.assertIsNone(result)

    def test_skips_missing_verification(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({"verification": "", "dependencies": []}),
            "outputs_json": json.dumps({"ok": True}),
        })
        result = _handle_task_end(context)
        self.assertIsNone(result)

    def test_passes_when_keywords_present(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({
                "verification": "database migration completed successfully",
                "dependencies": [],
            }),
            "outputs_json": json.dumps({
                "summary": "Ran database migration. Completed successfully.",
            }),
        })
        result = _handle_task_end(context)
        self.assertIsNone(result)

    def test_flags_when_keywords_missing(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({
                "verification": "database migration completed successfully",
                "dependencies": [],
            }),
            "outputs_json": json.dumps({
                "summary": "Looked at the weather forecast.",
            }),
        })
        result = _handle_task_end(context)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.messages) > 0)
        self.assertIn("warning", result.messages[0].lower())
        self.assertTrue(len(result.staged_items) > 0)
        self.assertEqual(result.staged_items[0]["kind"], "fact")

    def test_handles_inputs_json_as_string(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({
                "verification": "deployment configuration updated",
                "dependencies": [],
            }),
            "outputs_json": "deployment configuration was updated properly",
        })
        result = _handle_task_end(context)
        # Keywords match => should pass
        self.assertIsNone(result)

    def test_handles_none_outputs(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({
                "verification": "important feature implemented",
                "dependencies": [],
            }),
            "outputs_json": None,
        })
        result = _handle_task_end(context)
        # No outputs, so keywords won't be found => should flag
        self.assertIsNotNone(result)
        self.assertTrue(len(result.messages) > 0)

    def test_handles_missing_inputs_json(self) -> None:
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": None,
            "outputs_json": json.dumps({"ok": True}),
        })
        result = _handle_task_end(context)
        self.assertIsNone(result)

    def test_partial_keyword_match_passes_at_threshold(self) -> None:
        """When exactly half the keywords match, it should pass (ratio >= 0.5)."""
        context = self._make_context({
            "task_id": "t1",
            "run_id": "r1",
            "status": "done",
            "inputs_json": json.dumps({
                "verification": "configuration updated deployment ready",
                "dependencies": [],
            }),
            # Only "configuration" and "updated" present (2 of 4 keywords)
            "outputs_json": json.dumps({
                "summary": "configuration was updated",
            }),
        })
        result = _handle_task_end(context)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
