import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.hooks.core import Hook, HookActions, HookManager
from tali.idle import IdleLock, IdleScheduler
from tali.knowledge_sources import KnowledgeSourceRegistry
from tali.questions import mark_question_asked, queue_question, select_question_to_ask
from tali.questions import resolve_answered_question


class DummyLLM:
    def generate(self, prompt):
        return type("Resp", (), {"content": "{}"})


class IdleHooksQuestionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_idle_lock_single_instance(self) -> None:
        lock_path = Path(self.temp_dir.name) / "idle.lock"
        lock = IdleLock(lock_path)
        self.assertTrue(lock.acquire())
        second = IdleLock(lock_path)
        self.assertFalse(second.acquire())
        lock.release()

    def test_idle_lock_clears_stale(self) -> None:
        lock_path = Path(self.temp_dir.name) / "idle.lock"
        lock_path.write_text(
            json.dumps({"pid": 999999, "created_at": "2000-01-01T00:00:00Z"}),
            encoding="utf-8",
        )
        lock = IdleLock(lock_path)
        self.assertTrue(lock.acquire())
        lock.release()

    def test_idle_scheduler_trigger(self) -> None:
        scheduler = IdleScheduler(
            data_dir=Path(self.temp_dir.name),
            db=self.db,
            llm=DummyLLM(),
            sources=KnowledgeSourceRegistry(),
            idle_check_s=1,
        )
        scheduler._last_activity = datetime.utcnow() - timedelta(minutes=10)
        self.assertTrue(scheduler._should_run())

    def test_question_queue_and_answer(self) -> None:
        question_id = queue_question(self.db, "What is the codename?", reason="test", priority=3)
        decision = select_question_to_ask(self.db, "codename question")
        self.assertIsNotNone(decision)
        mark_question_asked(self.db, decision.question_id, decision.attempts)
        row = self.db.fetch_last_asked_question()
        self.assertEqual(row["id"], question_id)
        payload = resolve_answered_question(self.db, "Atlas", dict(row), source_ref="episode-1")
        self.assertEqual(payload["statement"], "Atlas")

    def test_hook_timeout(self) -> None:
        hooks_dir = Path(self.temp_dir.name) / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_file = hooks_dir / "slow_hook.py"
        hook_file.write_text(
            "\n".join(
                [
                    "import time",
                    "from tali.hooks import Hook, HookActions",
                    "",
                    "def handler(ctx):",
                    "    time.sleep(0.3)",
                    "    return HookActions(messages=['done'])",
                    "",
                    "HOOKS = [Hook(name='slow', triggers={'on_turn_start'}, handler=handler)]",
                ]
            ),
            encoding="utf-8",
        )
        manager = HookManager(db=self.db, hooks_dir=hooks_dir)
        manager.load_hooks()
        messages = manager.run("on_turn_start", {"user_input": "hi"})
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
