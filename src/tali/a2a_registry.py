from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    agent_name: str
    home: str
    status: str
    last_seen: str
    capabilities: list[str]
    load: dict[str, Any] | None = None


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"agents": []}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {"agents": []}

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2))
        temp_path.replace(self.path)

    def upsert(self, record: AgentRecord) -> None:
        with self._lock:
            payload = self.load()
            agents = payload.get("agents", [])
            updated = False
            for agent in agents:
                if agent.get("agent_id") == record.agent_id:
                    agent.update(
                        {
                            "agent_name": record.agent_name,
                            "home": record.home,
                            "status": record.status,
                            "last_seen": record.last_seen,
                            "capabilities": record.capabilities,
                            "load": record.load or agent.get("load"),
                        }
                    )
                    updated = True
                    break
            if not updated:
                agents.append(
                    {
                        "agent_id": record.agent_id,
                        "agent_name": record.agent_name,
                        "home": record.home,
                        "status": record.status,
                        "last_seen": record.last_seen,
                        "capabilities": record.capabilities,
                        "load": record.load,
                    }
                )
            payload["agents"] = agents
            self.save(payload)

    def list_agents(self) -> list[dict[str, Any]]:
        payload = self.load()
        return list(payload.get("agents", []))

    def heartbeat(self, agent_id: str, status: str, load: dict[str, Any] | None = None) -> None:
        with self._lock:
            payload = self.load()
            agents = payload.get("agents", [])
            for agent in agents:
                if agent.get("agent_id") == agent_id:
                    agent["status"] = status
                    agent["last_seen"] = datetime.utcnow().isoformat()
                    if load is not None:
                        agent["load"] = load
                    break
            payload["agents"] = agents
            self.save(payload)
