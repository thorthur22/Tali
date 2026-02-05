import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.consolidation import SleepPolicy, apply_sleep_changes
from tali.db import Database
from tali.self_care import SleepLock, resolve_staged_items


class SleepPolicyTests(unittest.TestCase):
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
            user_input="Please remember to follow up",
            agent_output="Okay.",
            tool_calls=[],
            outcome="ok",
            quarantine=0,
        )

    def test_tool_verified_auto_promote(self) -> None:
        episode_id = "episode-1"
        self._insert_episode(episode_id)
        payload = {
            "fact_candidates": [
                {
                    "statement": "The project codename is Atlas",
                    "provenance_type": "TOOL_VERIFIED",
                    "source_ref": episode_id,
                }
            ],
            "commitment_updates": [],
            "skill_candidates": [],
            "notes": [],
        }
        result = apply_sleep_changes(self.db, payload, SleepPolicy())
        self.assertEqual(len(result.inserted_fact_ids), 1)
        self.assertEqual(len(self.db.list_facts()), 1)

    def test_inferred_facts_are_staged(self) -> None:
        episode_id = "episode-2"
        self._insert_episode(episode_id)
        payload = {
            "fact_candidates": [
                {
                    "statement": "Likely prefers dark mode.",
                    "provenance_type": "INFERRED",
                    "source_ref": episode_id,
                }
            ],
            "commitment_updates": [],
            "skill_candidates": [],
            "notes": [],
        }
        result = apply_sleep_changes(self.db, payload, SleepPolicy())
        self.assertEqual(len(result.inserted_fact_ids), 0)
        self.assertEqual(self.db.count_staged_items(), 1)

    def test_contested_fact_forks(self) -> None:
        episode_id = "episode-3"
        self._insert_episode(episode_id)
        self.db.insert_fact(
            fact_id="fact-1",
            statement="not blue",
            provenance_type="SYSTEM_OBSERVED",
            source_ref=episode_id,
            confidence=0.7,
            created_at=datetime.utcnow().isoformat(),
            contested=0,
        )
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
        self.assertEqual(len(result.inserted_fact_ids), 1)
        facts = self.db.list_facts()
        contested_flags = {row["statement"]: row["contested"] for row in facts}
        self.assertEqual(contested_flags["blue"], 1)
        self.assertEqual(contested_flags["not blue"], 1)

    def test_staged_item_backoff(self) -> None:
        episode_id = "episode-4"
        self._insert_episode(episode_id)
        payload = {
            "statement": "User prefers short responses.",
            "provenance_type": "USER_REPORTED",
            "source_ref": episode_id,
        }
        self.db.insert_staged_item(
            item_id="stage-1",
            kind="fact",
            payload=json.dumps(payload),
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            source_ref=episode_id,
            provenance_type="USER_REPORTED",
            next_check_at=(datetime.utcnow() - timedelta(minutes=1)).isoformat(),
        )
        resolve_staged_items(self.db, "unrelated question about weather")
        row = self.db.fetch_staged_item("stage-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "verifying")
        self.assertIsNotNone(row["next_check_at"])

    def test_sleep_lock_prevents_concurrent_runs(self) -> None:
        data_dir = Path(self.temp_dir.name)
        lock = SleepLock(data_dir)
        self.assertTrue(lock.acquire())
        second = SleepLock(data_dir)
        self.assertFalse(second.acquire())
        lock.release()


if __name__ == "__main__":
    unittest.main()
