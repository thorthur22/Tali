import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.a2a import A2AClient, A2APoller, AgentProfile
from tali.a2a_bus import A2ABus
from tali.db import Database


class DummyTaskRunner:
    def run_turn(self, user_input: str, prompt_fn):
        return type("Result", (), {"message": "ok"})


class A2ATests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.shared = self.root / "shared"
        self.shared.mkdir(parents=True, exist_ok=True)
        self.bus = A2ABus(self.shared / "a2a.sqlite")
        self.bus.initialize()
        self.db = Database(self.root / "agent.db")
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bus_send_receive(self) -> None:
        message_id = self.bus.send(
            from_agent_id="a1",
            from_agent_name="one",
            to_agent_id="a2",
            to_agent_name="two",
            topic="task",
            payload={"type": "status", "run_state": "idle", "capabilities": [], "load": {}},
        )
        rows = self.bus.fetch_pending(agent_id="a2", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], message_id)

    def test_ttl_expiration(self) -> None:
        self.bus.send(
            from_agent_id="a1",
            from_agent_name="one",
            to_agent_id="a2",
            to_agent_name="two",
            topic="task",
            payload={"type": "status", "run_state": "idle", "capabilities": [], "load": {}},
            ttl_seconds=0,
        )
        expired = self.bus.expire_messages()
        self.assertGreaterEqual(expired, 1)

    def test_bus_connect_context_manager_closes_connection(self) -> None:
        with self.bus.connect() as connection:
            connection.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_task_response_updates_delegation(self) -> None:
        profile = AgentProfile(
            agent_id="a2",
            agent_name="two",
            agent_home=self.root / "two",
            shared_home=self.shared,
            capabilities=[],
        )
        client = A2AClient(db=self.db, profile=profile)
        poller = A2APoller(self.db, client, DummyTaskRunner(), prompt_fn=lambda _: "")
        now = datetime.utcnow().isoformat()
        self.db.insert_run(
            run_id="run-1",
            created_at=now,
            status="active",
            user_prompt="test",
            current_task_id=None,
            last_error=None,
        )
        self.db.insert_task(
            task_id="task-1",
            run_id="run-1",
            parent_task_id=None,
            ordinal=0,
            title="delegate",
            description="",
            status="blocked",
            inputs_json="{}",
            outputs_json=None,
            requires_tools=0,
            created_at=now,
            updated_at=now,
        )
        self.db.insert_delegation(
            delegation_id="del-1",
            task_id="task-1",
            run_id="run-1",
            correlation_id="corr-1",
            to_agent_id="a1",
            to_agent_name="one",
            status="sent",
            created_at=now,
            updated_at=now,
        )
        payload = {
            "type": "task_response",
            "correlation_id": "corr-1",
            "status": "completed",
            "result": {"ok": True},
            "notes": "",
        }
        poller._handle_task_response({}, payload)
        task = self.db.fetch_task("task-1")
        self.assertEqual(task["status"], "done")

    def test_inbound_message_not_fact(self) -> None:
        profile = AgentProfile(
            agent_id="a2",
            agent_name="two",
            agent_home=self.root / "two",
            shared_home=self.shared,
            capabilities=[],
        )
        client = A2AClient(db=self.db, profile=profile)
        poller = A2APoller(self.db, client, DummyTaskRunner(), prompt_fn=lambda _: "")
        row = {
            "id": "m1",
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent_id": "a1",
            "from_agent_name": "one",
            "to_agent_id": "a2",
            "to_agent_name": "two",
            "topic": "status",
            "correlation_id": None,
            "payload": json.dumps({"type": "status", "run_state": "idle", "capabilities": [], "load": {}}),
            "signature": None,
        }
        poller._handle_message(row)
        self.assertEqual(len(self.db.list_facts()), 0)


if __name__ == "__main__":
    unittest.main()
