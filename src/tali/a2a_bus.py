from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_messages (
  id TEXT PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  from_agent_id TEXT NOT NULL,
  from_agent_name TEXT NOT NULL,
  to_agent_id TEXT,
  to_agent_name TEXT,
  topic TEXT NOT NULL,
  correlation_id TEXT,
  ttl_seconds INTEGER DEFAULT 86400,
  priority INTEGER DEFAULT 3,
  payload TEXT NOT NULL,
  status TEXT NOT NULL,
  signature TEXT,
  created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_a2a_to_status ON a2a_messages (to_agent_id, status);
CREATE INDEX IF NOT EXISTS idx_a2a_topic_time ON a2a_messages (topic, timestamp);
"""


class _ManagedSQLiteConnection(sqlite3.Connection):
    """SQLite connection that closes at context-manager exit."""

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


@dataclass
class A2ABus:
    path: Path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, factory=_ManagedSQLiteConnection)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def send(
        self,
        from_agent_id: str,
        from_agent_name: str,
        to_agent_id: str | None,
        to_agent_name: str | None,
        topic: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        ttl_seconds: int = 86400,
        priority: int = 3,
        signature: str | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO a2a_messages (
                    id, timestamp, from_agent_id, from_agent_name, to_agent_id, to_agent_name,
                    topic, correlation_id, ttl_seconds, priority, payload, status, signature, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    now,
                    from_agent_id,
                    from_agent_name,
                    to_agent_id,
                    to_agent_name,
                    topic,
                    correlation_id,
                    ttl_seconds,
                    priority,
                    json.dumps(payload),
                    "queued",
                    signature,
                    now,
                ),
            )
        return message_id

    def fetch_pending(
        self,
        agent_id: str,
        limit: int = 10,
        include_broadcast: bool = True,
    ) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if include_broadcast:
                cursor = connection.execute(
                    """
                    SELECT * FROM a2a_messages
                    WHERE status = 'queued'
                      AND (to_agent_id IS NULL OR to_agent_id = ?)
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                    """,
                    (agent_id, limit),
                )
            else:
                cursor = connection.execute(
                    """
                    SELECT * FROM a2a_messages
                    WHERE status = 'queued' AND to_agent_id = ?
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                    """,
                    (agent_id, limit),
                )
            return cursor.fetchall()

    def mark_status(self, message_id: str, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE a2a_messages SET status = ? WHERE id = ?",
                (status, message_id),
            )

    def expire_messages(self) -> int:
        now = datetime.utcnow()
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT id, timestamp, ttl_seconds FROM a2a_messages WHERE status != 'expired'"
            )
            to_expire = []
            for row in cursor.fetchall():
                timestamp = datetime.fromisoformat(row["timestamp"])
                ttl = int(row["ttl_seconds"] or 0)
                if timestamp + timedelta(seconds=ttl) < now:
                    to_expire.append(row["id"])
            for message_id in to_expire:
                connection.execute(
                    "UPDATE a2a_messages SET status = 'expired' WHERE id = ?",
                    (message_id,),
                )
            return len(to_expire)
