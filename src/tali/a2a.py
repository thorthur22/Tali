from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from tali.a2a_bus import A2ABus
from tali.a2a_protocol import parse_message, parse_status, parse_task_request, parse_task_response
from tali.a2a_registry import AgentRecord, Registry
from tali.db import Database
if TYPE_CHECKING:
    from tali.task_runner import TaskRunner


POLL_INTERVAL_S = 2
MAX_MESSAGES_PER_POLL = 5


def load_shared_secret(shared_home: Path) -> bytes | None:
    path = shared_home / "secret.key"
    if not path.exists():
        return None
    return path.read_bytes()


def sign_message(secret: bytes, payload: dict[str, Any], headers: dict[str, Any]) -> str:
    body = json.dumps({"headers": headers, "payload": payload}, sort_keys=True).encode("utf-8")
    return hmac.new(secret, body, sha256).hexdigest()


def verify_message(secret: bytes, payload: dict[str, Any], headers: dict[str, Any], signature: str | None) -> bool:
    if not signature:
        return False
    expected = sign_message(secret, payload, headers)
    return hmac.compare_digest(expected, signature)


@dataclass
class AgentProfile:
    agent_id: str
    agent_name: str
    agent_home: Path
    shared_home: Path
    capabilities: list[str]


class A2AClient:
    def __init__(self, db: Database, profile: AgentProfile) -> None:
        self.db = db
        self.profile = profile
        self.bus = A2ABus(profile.shared_home / "a2a.sqlite")
        self.bus.initialize()
        self.secret = load_shared_secret(profile.shared_home)
        self.registry = Registry(profile.shared_home / "registry.json")

    def send(
        self,
        to_agent_id: str | None,
        to_agent_name: str | None,
        topic: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        priority: int = 3,
    ) -> str:
        headers = {
            "from_agent_id": self.profile.agent_id,
            "from_agent_name": self.profile.agent_name,
            "to_agent_id": to_agent_id,
            "to_agent_name": to_agent_name,
            "topic": topic,
            "correlation_id": correlation_id,
        }
        signature = sign_message(self.secret, payload, headers) if self.secret else None
        message_id = self.bus.send(
            from_agent_id=self.profile.agent_id,
            from_agent_name=self.profile.agent_name,
            to_agent_id=to_agent_id,
            to_agent_name=to_agent_name,
            topic=topic,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority,
            signature=signature,
        )
        self.db.insert_agent_message(
            message_id=message_id,
            timestamp=datetime.utcnow().isoformat(),
            direction="outbound",
            from_agent_id=self.profile.agent_id,
            from_agent_name=self.profile.agent_name,
            to_agent_id=to_agent_id,
            to_agent_name=to_agent_name,
            topic=topic,
            correlation_id=correlation_id,
            payload=json.dumps(payload),
            status="sent",
            provenance_type="RECEIVED_AGENT",
        )
        return message_id

    def select_agent(self, preferred_name: str | None = None, capability: str | None = None) -> dict[str, Any] | None:
        agents = self.registry.list_agents()
        if preferred_name:
            for agent in agents:
                if agent.get("agent_name") == preferred_name:
                    return agent
        if capability:
            candidates = [agent for agent in agents if capability in agent.get("capabilities", [])]
        else:
            candidates = agents
        if not candidates:
            return None
        def score(item: dict[str, Any]) -> tuple[int, str]:
            load = item.get("load", {}) or {}
            active = int(load.get("active_tasks", 0) or 0)
            return active, str(item.get("last_seen", ""))

        return sorted(candidates, key=score)[0]


class A2APoller:
    def __init__(
        self,
        db: Database,
        client: A2AClient,
        task_runner: TaskRunner | None,
        prompt_fn: Callable[[str], str],
        status_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.db = db
        self.client = client
        self.task_runner = task_runner
        self.prompt_fn = prompt_fn
        self.status_fn = status_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock_path = client.profile.agent_home / "locks" / "a2a.lock"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not _acquire_lock(self._lock_path):
            return
        try:
            last_heartbeat = 0.0
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now - last_heartbeat > 30:
                    self.client.registry.heartbeat(self.client.profile.agent_id, status="active")
                    run_state = "active" if self.db.fetch_active_run() else "idle"
                    payload = {
                        "type": "status",
                        "run_state": run_state,
                        "capabilities": self.client.profile.capabilities,
                        "load": {"active_tasks": 1 if run_state == "active" else 0},
                    }
                    self.client.send(
                        to_agent_id=None,
                        to_agent_name=None,
                        topic="status",
                        payload=payload,
                        correlation_id=None,
                        priority=5,
                    )
                    last_heartbeat = now
                self.client.bus.expire_messages()
                messages = self.client.bus.fetch_pending(
                    agent_id=self.client.profile.agent_id, limit=MAX_MESSAGES_PER_POLL
                )
                for row in messages:
                    self._handle_message(row)
                    self.client.bus.mark_status(row["id"], "delivered")
                time.sleep(POLL_INTERVAL_S)
        finally:
            _release_lock(self._lock_path)

    def _handle_message(self, row: Any) -> None:
        payload_text = row["payload"]
        message_payload, error = parse_message(payload_text)
        if error or not message_payload:
            self._log_message(row, status="read")
            return
        headers = {
            "from_agent_id": row["from_agent_id"],
            "from_agent_name": row["from_agent_name"],
            "to_agent_id": row["to_agent_id"],
            "to_agent_name": row["to_agent_name"],
            "topic": row["topic"],
            "correlation_id": row["correlation_id"],
        }
        if self.client.secret and not verify_message(
            self.client.secret, message_payload, headers, row["signature"]
        ):
            self._log_message(row, status="read")
            return
        self._log_message(row, status="unread")
        msg_type = message_payload.get("type")
        if msg_type == "task_request":
            self._handle_task_request(row, message_payload)
        elif msg_type == "task_response":
            self._handle_task_response(row, message_payload)
        elif msg_type == "status":
            self._handle_status(row, message_payload)
        self.db.mark_agent_message(row["id"], "read")

    def _handle_task_request(self, row: Any, payload: dict[str, Any]) -> None:
        request, error = parse_task_request(payload)
        if error or not request:
            self._send_response(row, status="rejected", result={"error": error or "invalid"})
            return
        if not self.task_runner:
            self._send_response(row, status="rejected", result={"error": "task runner unavailable"})
            return
        run_id = self._create_delegated_run(request, row["correlation_id"] or str(uuid.uuid4()))
        if self.status_fn:
            self.status_fn(f"A2A: accepted task '{request.title}'")
        result = self.task_runner.run_turn(
            f"Delegated task: {request.title}\n{request.description}",
            prompt_fn=self.prompt_fn,
            show_plans=False,
        )
        run_row = self.db.fetch_run(run_id)
        if run_row and run_row["status"] == "done":
            outputs = [dict(row) for row in self.db.fetch_tasks_for_run(run_id)]
            self._send_response(
                row,
                status="completed",
                result={"run_id": run_id, "tasks": outputs, "message": result.message},
            )
        else:
            self._send_response(
                row,
                status="accepted",
                result={"run_id": run_id, "message": result.message},
            )

    def _handle_task_response(self, row: Any, payload: dict[str, Any]) -> None:
        response, error = parse_task_response(payload)
        if error or not response:
            return
        delegation = self.db.fetch_delegation(response.correlation_id)
        if not delegation:
            return
        now = datetime.utcnow().isoformat()
        self.db.update_delegation_status(response.correlation_id, response.status, now)
        task = self.db.fetch_task(delegation["task_id"])
        if not task:
            return
        outputs = _merge_outputs(
            task["outputs_json"],
            {"delegation_result": response.result, "delegation_status": response.status},
        )
        self.db.update_task_status(
            task_id=task["id"],
            status="done" if response.status == "completed" else "failed",
            outputs_json=outputs,
            updated_at=now,
        )

    def _handle_status(self, row: Any, payload: dict[str, Any]) -> None:
        status_msg, error = parse_status(payload)
        if error or not status_msg:
            return
        self.client.registry.heartbeat(
            agent_id=row["from_agent_id"], status=status_msg.run_state, load=status_msg.load
        )

    def _send_response(self, row: Any, status: str, result: dict[str, Any]) -> None:
        payload = {
            "type": "task_response",
            "correlation_id": row["correlation_id"] or str(uuid.uuid4()),
            "status": status,
            "result": result,
            "notes": "",
        }
        self.client.send(
            to_agent_id=row["from_agent_id"],
            to_agent_name=row["from_agent_name"],
            topic="result",
            payload=payload,
            correlation_id=row["correlation_id"],
        )

    def _create_delegated_run(self, request: Any, correlation_id: str) -> str:
        run_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.db.insert_run(
            run_id=run_id,
            created_at=now,
            status="active",
            user_prompt=f"A2A:{request.title}",
            current_task_id=None,
            last_error=None,
        )
        self.db.insert_task(
            task_id=str(uuid.uuid4()),
            run_id=run_id,
            parent_task_id=None,
            ordinal=0,
            title=request.title,
            description=request.description,
            status="pending",
            inputs_json=json.dumps({"inputs": request.inputs}),
            outputs_json=None,
            requires_tools=0,
            created_at=now,
            updated_at=now,
        )
        return run_id

    def _log_message(self, row: Any, status: str) -> None:
        self.db.insert_agent_message(
            message_id=row["id"],
            timestamp=row["timestamp"],
            direction="inbound",
            from_agent_id=row["from_agent_id"],
            from_agent_name=row["from_agent_name"],
            to_agent_id=row["to_agent_id"],
            to_agent_name=row["to_agent_name"],
            topic=row["topic"],
            correlation_id=row["correlation_id"],
            payload=row["payload"],
            status=status,
            provenance_type="RECEIVED_AGENT",
        )


def _acquire_lock(path: Path) -> bool:
    if path.exists():
        return False
    path.write_text(str(uuid.uuid4()))
    return True


def _release_lock(path: Path) -> None:
    if path.exists():
        path.unlink()


def _merge_outputs(existing: str | None, update: dict[str, Any]) -> str:
    base: dict[str, Any] = {}
    if existing:
        try:
            base = json.loads(existing)
        except json.JSONDecodeError:
            base = {}
    base.update(update)
    return json.dumps(base)
