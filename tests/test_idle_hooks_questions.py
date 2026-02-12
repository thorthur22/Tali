import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.hooks.core import Hook, HookActions, HookManager
from tali.idle import IdleLock, IdleScheduler
from tali.idle_jobs import IdleJobRunner
from tali.knowledge_sources import KnowledgeSourceRegistry
from tali.models import ProvenanceType
from tali.questions import mark_question_asked, queue_question, select_question_to_ask
from tali.questions import resolve_answered_question
from tali.self_care import resolve_staged_confirmation


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

    def test_idle_job_queues_question_updates_staged_item(self) -> None:
        now = datetime.utcnow().isoformat()
        item_id = str(uuid4())
        payload = {
            "statement": "The sky is blue",
            "provenance_type": ProvenanceType.USER_REPORTED.value,
            "source_ref": "episode-1",
            "confidence": 0.7,
        }
        self.db.insert_staged_item(
            item_id=item_id,
            kind="fact",
            payload=json.dumps(payload),
            status="pending",
            created_at=now,
            source_ref="episode-1",
            provenance_type=ProvenanceType.USER_REPORTED.value,
            next_check_at=now,
        )
        runner = IdleJobRunner(
            db=self.db,
            llm=DummyLLM(),
            sources=KnowledgeSourceRegistry(),
            should_stop=lambda: False,
        )
        result = runner._job_clarifying_question()
        self.assertIsNotNone(result)
        question = self.db.fetch_next_user_question(datetime.utcnow().isoformat())
        self.assertIsNotNone(question)
        reason = json.loads(question["reason"])
        self.assertEqual(reason["staged_item_id"], item_id)
        staged = self.db.fetch_staged_item(item_id)
        self.assertEqual(staged["status"], "verifying")
        self.assertEqual(int(staged["attempts"]), 1)
        self.assertEqual(staged["last_error"], "queued_question")
        self.assertIsNotNone(staged["next_check_at"])

    def test_resolve_staged_confirmation_promotes_fact(self) -> None:
        now = datetime.utcnow().isoformat()
        item_id = str(uuid4())
        payload = {
            "statement": "The ocean is salty",
            "provenance_type": ProvenanceType.USER_REPORTED.value,
            "source_ref": "episode-2",
            "confidence": 0.7,
        }
        self.db.insert_staged_item(
            item_id=item_id,
            kind="fact",
            payload=json.dumps(payload),
            status="pending",
            created_at=now,
            source_ref="episode-2",
            provenance_type=ProvenanceType.USER_REPORTED.value,
            next_check_at=now,
        )
        resolution = resolve_staged_confirmation(self.db, item_id, "yes")
        self.assertIsNotNone(resolution)
        self.assertIsNotNone(resolution.applied_fact_id)
        staged = self.db.fetch_staged_item(item_id)
        self.assertEqual(staged["status"], "resolved")
        facts = self.db.list_facts()
        self.assertTrue(any(row["statement"] == "The ocean is salty" for row in facts))

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
