import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.consolidation import SleepPolicy, apply_sleep_changes, _queue_contradiction_question
from tali.db import Database
from tali.questions import queue_question, select_question_to_ask
from tali.self_care import resolve_contradiction_answer


class ContradictionResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _insert_episode(self, episode_id: str) -> None:
        self.db.insert_episode(
            episode_id=episode_id,
            user_input="test input",
            agent_output="test output",
            tool_calls=[],
            outcome="ok",
            quarantine=0,
        )

    def test_contradiction_queues_priority5_question(self) -> None:
        """When a contradiction is found, a priority-5 question should be queued."""
        episode_id = "ep-1"
        self._insert_episode(episode_id)

        # Insert existing fact
        self.db.insert_fact(
            fact_id="fact-existing",
            statement="not blue",
            provenance_type="SYSTEM_OBSERVED",
            source_ref=episode_id,
            confidence=0.7,
            created_at=datetime.utcnow().isoformat(),
            contested=0,
        )

        # Run consolidation with contradictory fact
        payload = {
            "fact_candidates": [
                {
                    "statement": "blue",
                    "provenance_type": "TOOL_VERIFIED",
                    "source_ref": episode_id,
                }
            ],
            "commitment_updates": [],
            "skill_candidates": [],
            "notes": [],
        }
        result = apply_sleep_changes(self.db, payload, SleepPolicy())

        # The new fact should be inserted as contested
        self.assertEqual(len(result.contested_fact_ids), 1)

        # A priority-5 question should have been queued
        now = datetime.utcnow().isoformat()
        question_row = self.db.fetch_next_user_question(now)
        self.assertIsNotNone(question_row)
        self.assertEqual(int(question_row["priority"]), 5)
        self.assertIn("conflicting", question_row["question"])

    def test_contradiction_high_confidence_queues_question(self) -> None:
        """When existing fact has high confidence, new fact is staged AND question queued."""
        episode_id = "ep-2"
        self._insert_episode(episode_id)

        # Insert existing high-confidence fact
        self.db.insert_fact(
            fact_id="fact-high",
            statement="not red",
            provenance_type="TOOL_VERIFIED",
            source_ref=episode_id,
            confidence=0.95,
            created_at=datetime.utcnow().isoformat(),
            contested=0,
        )

        payload = {
            "fact_candidates": [
                {
                    "statement": "red",
                    "provenance_type": "TOOL_VERIFIED",
                    "source_ref": episode_id,
                }
            ],
            "commitment_updates": [],
            "skill_candidates": [],
            "notes": [],
        }
        result = apply_sleep_changes(self.db, payload, SleepPolicy())

        # New fact should be staged (not inserted)
        self.assertEqual(len(result.inserted_fact_ids), 0)
        self.assertIn("contradiction_high_confidence", result.skipped_candidates)

        # A priority-5 question should have been queued
        now = datetime.utcnow().isoformat()
        question_row = self.db.fetch_next_user_question(now)
        self.assertIsNotNone(question_row)
        self.assertEqual(int(question_row["priority"]), 5)

    def test_queue_contradiction_question_directly(self) -> None:
        """Test the helper function directly."""
        question_id = _queue_contradiction_question(
            self.db,
            new_statement="The sky is green",
            existing_statement="The sky is blue",
            new_fact_id="new-1",
            existing_fact_id="existing-1",
        )
        self.assertIsNotNone(question_id)
        now = datetime.utcnow().isoformat()
        row = self.db.fetch_next_user_question(now)
        self.assertIsNotNone(row)
        reason = json.loads(row["reason"])
        self.assertEqual(reason["type"], "contradiction")
        self.assertEqual(reason["new_fact_id"], "new-1")
        self.assertEqual(reason["existing_fact_id"], "existing-1")

    def test_resolve_contradiction_keep_new(self) -> None:
        """User says the new statement is correct."""
        episode_id = "ep-3"
        self._insert_episode(episode_id)

        self.db.insert_fact(
            fact_id="fact-old",
            statement="The default is light mode",
            provenance_type="SYSTEM_OBSERVED",
            source_ref=episode_id,
            confidence=0.7,
            created_at=datetime.utcnow().isoformat(),
            contested=1,
        )
        self.db.insert_fact(
            fact_id="fact-new",
            statement="The default is dark mode",
            provenance_type="USER_REPORTED",
            source_ref=episode_id,
            confidence=0.6,
            created_at=datetime.utcnow().isoformat(),
            contested=1,
        )

        reason_json = {
            "type": "contradiction",
            "new_statement": "The default is dark mode",
            "existing_statement": "The default is light mode",
            "new_fact_id": "fact-new",
            "existing_fact_id": "fact-old",
        }

        resolved = resolve_contradiction_answer(
            self.db, "the new one is correct", reason_json, source_ref=episode_id
        )
        self.assertTrue(resolved)

        # The old fact should have very low confidence
        old_fact = self.db.fetch_facts_by_ids(["fact-old"])
        self.assertAlmostEqual(float(old_fact[0]["confidence"]), 0.05)
        self.assertEqual(int(old_fact[0]["contested"]), 0)

        # The new fact should no longer be contested
        new_fact = self.db.fetch_facts_by_ids(["fact-new"])
        self.assertEqual(int(new_fact[0]["contested"]), 0)

    def test_resolve_contradiction_keep_existing(self) -> None:
        """User says the existing statement is correct."""
        episode_id = "ep-4"
        self._insert_episode(episode_id)

        self.db.insert_fact(
            fact_id="fact-old-2",
            statement="Python version is 3.11",
            provenance_type="TOOL_VERIFIED",
            source_ref=episode_id,
            confidence=0.9,
            created_at=datetime.utcnow().isoformat(),
            contested=1,
        )
        self.db.insert_fact(
            fact_id="fact-new-2",
            statement="Python version is 3.10",
            provenance_type="USER_REPORTED",
            source_ref=episode_id,
            confidence=0.6,
            created_at=datetime.utcnow().isoformat(),
            contested=1,
        )

        reason_json = {
            "type": "contradiction",
            "new_statement": "Python version is 3.10",
            "existing_statement": "Python version is 3.11",
            "new_fact_id": "fact-new-2",
            "existing_fact_id": "fact-old-2",
        }

        resolved = resolve_contradiction_answer(
            self.db, "the old one is correct", reason_json, source_ref=episode_id
        )
        self.assertTrue(resolved)

        # The new fact should have very low confidence
        new_fact = self.db.fetch_facts_by_ids(["fact-new-2"])
        self.assertAlmostEqual(float(new_fact[0]["confidence"]), 0.05)

        # The old fact should no longer be contested
        old_fact = self.db.fetch_facts_by_ids(["fact-old-2"])
        self.assertEqual(int(old_fact[0]["contested"]), 0)

    def test_resolve_contradiction_ambiguous_answer(self) -> None:
        """User gives an ambiguous answer that can't be parsed."""
        reason_json = {
            "type": "contradiction",
            "new_statement": "A",
            "existing_statement": "B",
            "new_fact_id": "f1",
            "existing_fact_id": "f2",
        }
        resolved = resolve_contradiction_answer(
            self.db, "I'm not sure", reason_json, source_ref="ep"
        )
        self.assertFalse(resolved)

    def test_priority5_bypasses_relevance_check(self) -> None:
        """Priority-5 questions should be surfaced regardless of user input relevance."""
        queue_question(
            self.db,
            question="Is the sky blue or green?",
            reason=json.dumps({"type": "contradiction"}),
            priority=5,
        )
        # Use completely unrelated user input
        decision = select_question_to_ask(self.db, "tell me about cooking recipes")
        self.assertIsNotNone(decision)
        self.assertIn("sky", decision.question)


class PersistenceRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_updated_at_set_on_insert(self) -> None:
        """Runs should have updated_at populated on creation."""
        now = datetime.utcnow().isoformat()
        self.db.insert_run(
            run_id="run-1",
            created_at=now,
            status="active",
            user_prompt="test",
        )
        run = self.db.fetch_run("run-1")
        self.assertIsNotNone(run["updated_at"])

    def test_heartbeat_updates_timestamp(self) -> None:
        """Heartbeat should update the updated_at field."""
        old_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        self.db.insert_run(
            run_id="run-2",
            created_at=old_time,
            status="active",
            user_prompt="test",
        )
        self.db.heartbeat_run("run-2")
        run = self.db.fetch_run("run-2")
        # updated_at should be recent (within last minute)
        updated = datetime.fromisoformat(run["updated_at"])
        self.assertGreater(updated, datetime.utcnow() - timedelta(minutes=1))

    def test_fetch_stale_runs(self) -> None:
        """Should return runs that haven't been updated recently."""
        stale_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        self.db.insert_run(
            run_id="stale-run",
            created_at=stale_time,
            status="active",
            user_prompt="stale test",
        )
        # Manually set updated_at to be old
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE runs SET updated_at = ? WHERE id = ?",
                (stale_time, "stale-run"),
            )
        stale = self.db.fetch_stale_runs(timeout_minutes=30)
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["id"], "stale-run")

    def test_fetch_incomplete_runs(self) -> None:
        """Should return all active and blocked runs."""
        now = datetime.utcnow().isoformat()
        self.db.insert_run(run_id="r1", created_at=now, status="active", user_prompt="t1")
        self.db.insert_run(run_id="r2", created_at=now, status="blocked", user_prompt="t2")
        self.db.insert_run(run_id="r3", created_at=now, status="done", user_prompt="t3")
        incomplete = self.db.fetch_incomplete_runs()
        ids = {r["id"] for r in incomplete}
        self.assertIn("r1", ids)
        self.assertIn("r2", ids)
        self.assertNotIn("r3", ids)

    def test_update_run_status_sets_updated_at(self) -> None:
        """Status changes should also update the timestamp."""
        old_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        self.db.insert_run(
            run_id="run-status",
            created_at=old_time,
            status="active",
            user_prompt="test",
        )
        self.db.update_run_status("run-status", status="blocked", last_error="test error")
        run = self.db.fetch_run("run-status")
        updated = datetime.fromisoformat(run["updated_at"])
        self.assertGreater(updated, datetime.utcnow() - timedelta(minutes=1))


if __name__ == "__main__":
    unittest.main()
