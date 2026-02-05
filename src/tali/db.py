from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  user_input TEXT,
  agent_output TEXT,
  tool_calls TEXT,
  outcome TEXT,
  importance REAL DEFAULT 0.0,
  quarantine INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  statement TEXT NOT NULL,
  provenance_type TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at DATETIME NOT NULL,
  last_confirmed DATETIME,
  decay_rate REAL DEFAULT 0.01,
  contested INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fact_links (
  id TEXT PRIMARY KEY,
  fact_id TEXT NOT NULL,
  related_fact_id TEXT,
  episode_id TEXT,
  link_type TEXT NOT NULL,
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
  id TEXT PRIMARY KEY,
  statement TEXT NOT NULL,
  origin_episode_id TEXT NOT NULL,
  confidence REAL DEFAULT 0.3,
  status TEXT DEFAULT 'active',
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS commitments (
  id TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER DEFAULT 3,
  due_date DATETIME,
  created_at DATETIME NOT NULL,
  last_touched DATETIME NOT NULL,
  source_ref TEXT
);

CREATE TABLE IF NOT EXISTS skills (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  trigger TEXT NOT NULL,
  steps TEXT NOT NULL,
  success_count INTEGER DEFAULT 0,
  failure_count INTEGER DEFAULT 0,
  last_used DATETIME,
  created_at DATETIME NOT NULL,
  source_ref TEXT
);

CREATE TABLE IF NOT EXISTS preferences (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  confidence REAL DEFAULT 0.7,
  provenance_type TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  updated_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS sleep_runs (
  id TEXT PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  last_episode_timestamp DATETIME
);

CREATE TABLE IF NOT EXISTS staged_items (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  source_ref TEXT NOT NULL,
  provenance_type TEXT NOT NULL,
  priority INTEGER DEFAULT 3,
  next_check_at DATETIME,
  attempts INTEGER DEFAULT 0,
  last_error TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  args_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  result_hash TEXT,
  result_path TEXT,
  started_at DATETIME,
  ended_at DATETIME,
  risk_level TEXT,
  approval_mode TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_episode ON tool_calls (episode_id);

CREATE INDEX IF NOT EXISTS idx_staged_status_nextcheck ON staged_items (status, next_check_at);
CREATE INDEX IF NOT EXISTS idx_staged_kind ON staged_items (kind);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
  id,
  user_input,
  agent_output
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  id,
  statement
);
"""


@dataclass
class Database:
    path: Path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def insert_episode(
        self,
        episode_id: str,
        user_input: str,
        agent_output: str,
        tool_calls: list[dict[str, Any]],
        outcome: str,
        quarantine: int = 0,
    ) -> None:
        timestamp = datetime.utcnow().isoformat()
        tool_calls_json = json.dumps(tool_calls)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO episodes (id, timestamp, user_input, agent_output, tool_calls, outcome, quarantine)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (episode_id, timestamp, user_input, agent_output, tool_calls_json, outcome, quarantine),
            )
            connection.execute(
                "INSERT INTO episodes_fts (id, user_input, agent_output) VALUES (?, ?, ?)",
                (episode_id, user_input, agent_output),
            )

    def insert_tool_call(
        self,
        tool_call_id: str,
        episode_id: str,
        tool_name: str,
        args: dict[str, Any],
        status: str,
        result_json: str | None,
        result_hash: str | None,
        result_path: str | None,
        started_at: str | None,
        ended_at: str | None,
        risk_level: str | None,
        approval_mode: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_calls (
                    id, episode_id, tool_name, args_json, status,
                    result_json, result_hash, result_path, started_at,
                    ended_at, risk_level, approval_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    episode_id,
                    tool_name,
                    json.dumps(args),
                    status,
                    result_json,
                    result_hash,
                    result_path,
                    started_at,
                    ended_at,
                    risk_level,
                    approval_mode,
                ),
            )

    def list_commitments(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM commitments
                ORDER BY priority ASC, due_date ASC, last_touched DESC
                """
            )
            return cursor.fetchall()

    def list_facts(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM facts
                ORDER BY contested ASC, confidence DESC, created_at DESC
                """
            )
            return cursor.fetchall()

    def dedupe_facts(self) -> int:
        """Remove duplicate facts with identical statement/provenance/source_ref."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT id FROM facts
                WHERE id NOT IN (
                    SELECT MIN(id) FROM facts
                    GROUP BY statement, provenance_type, source_ref
                )
                """
            )
            duplicate_ids = [row["id"] for row in cursor.fetchall()]
            if not duplicate_ids:
                return 0
            placeholders = ", ".join("?" for _ in duplicate_ids)
            connection.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", duplicate_ids)
            connection.execute(
                f"DELETE FROM facts_fts WHERE id IN ({placeholders})",
                duplicate_ids,
            )
            return len(duplicate_ids)

    def insert_fact(
        self,
        fact_id: str,
        statement: str,
        provenance_type: str,
        source_ref: str,
        confidence: float,
        created_at: str,
        contested: int = 0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO facts (id, statement, provenance_type, source_ref, confidence, created_at, contested)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (fact_id, statement, provenance_type, source_ref, confidence, created_at, contested),
            )
            connection.execute(
                "INSERT INTO facts_fts (id, statement) VALUES (?, ?)",
                (fact_id, statement),
            )

    def insert_fact_link(
        self,
        link_id: str,
        fact_id: str,
        related_fact_id: str,
        episode_id: str | None,
        link_type: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO fact_links (id, fact_id, related_fact_id, episode_id, link_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (link_id, fact_id, related_fact_id, episode_id, link_type, created_at),
            )

    def search_facts(self, query: str, limit: int) -> list[sqlite3.Row]:
        safe_query = "".join(ch if ch.isalnum() else " " for ch in query).strip()
        if not safe_query:
            return []
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT facts.* FROM facts
                JOIN facts_fts ON facts.id = facts_fts.id
                WHERE facts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            )
            return cursor.fetchall()

    def episode_exists(self, episode_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT 1 FROM episodes WHERE id = ?",
                (episode_id,),
            )
            return cursor.fetchone() is not None

    def mark_fact_contested(self, fact_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE facts SET contested = 1 WHERE id = ?",
                (fact_id,),
            )

    def fetch_facts_by_ids(self, fact_ids: list[str]) -> list[sqlite3.Row]:
        if not fact_ids:
            return []
        placeholders = ", ".join("?" for _ in fact_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"SELECT * FROM facts WHERE id IN ({placeholders})",
                fact_ids,
            )
            return cursor.fetchall()

    def fetch_episodes_by_ids(self, episode_ids: list[str]) -> list[sqlite3.Row]:
        if not episode_ids:
            return []
        placeholders = ", ".join("?" for _ in episode_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"SELECT * FROM episodes WHERE id IN ({placeholders})",
                episode_ids,
            )
            return cursor.fetchall()

    def list_skills(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM skills
                ORDER BY success_count DESC, name ASC
                """
            )
            return cursor.fetchall()

    def list_preferences(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM preferences ORDER BY confidence DESC, updated_at DESC"
            )
            return cursor.fetchall()

    def fetch_recent_episodes(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

    def search_episodes(self, query: str, limit: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT episodes.* FROM episodes
                JOIN episodes_fts ON episodes.id = episodes_fts.id
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
            return cursor.fetchall()

    def list_episodes_since(
        self, timestamp: str | None, limit: int, exclude_quarantined: bool = True
    ) -> list[sqlite3.Row]:
        with self.connect() as connection:
            where_clause = "WHERE timestamp > ?" if timestamp else "WHERE 1=1"
            params: list[object] = [timestamp] if timestamp else []
            if exclude_quarantined:
                where_clause += " AND quarantine = 0"
            params.append(limit)
            cursor = connection.execute(
                f"""
                SELECT * FROM episodes
                {where_clause}
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                params,
            )
            return cursor.fetchall()

    def fetch_episodes_since_last_sleep(
        self, limit: int, exclude_quarantined: bool = True
    ) -> list[sqlite3.Row]:
        last_run = self.last_sleep_run()
        last_timestamp = last_run["last_episode_timestamp"] if last_run else None
        return self.list_episodes_since(
            last_timestamp, limit=limit, exclude_quarantined=exclude_quarantined
        )

    def fetch_episode(self, episode_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (episode_id,),
            )
            return cursor.fetchone()

    def apply_confidence_decay(self, cutoff_date: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE facts
                SET confidence = MAX(0.0, confidence - decay_rate)
                WHERE COALESCE(last_confirmed, created_at) < ?
                """,
                (cutoff_date,),
            )

    def apply_fact_decay(self, cutoff_timestamp: str) -> int:
        """Apply confidence decay to facts not confirmed since cutoff_timestamp."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE facts
                SET confidence = MAX(0.0, confidence - decay_rate)
                WHERE COALESCE(last_confirmed, created_at) < ?
                """,
                (cutoff_timestamp,),
            )
            return cursor.rowcount

    def insert_commitment(
        self,
        commitment_id: str,
        description: str,
        status: str,
        priority: int,
        due_date: str | None,
        created_at: str,
        last_touched: str,
        source_ref: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO commitments
                (id, description, status, priority, due_date, created_at, last_touched, source_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commitment_id,
                    description,
                    status,
                    priority,
                    due_date,
                    created_at,
                    last_touched,
                    source_ref,
                ),
            )

    def update_commitment(
        self,
        commitment_id: str,
        description: str,
        status: str,
        priority: int,
        due_date: str | None,
        last_touched: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE commitments
                SET description = ?, status = ?, priority = ?, due_date = ?, last_touched = ?
                WHERE id = ?
                """,
                (description, status, priority, due_date, last_touched, commitment_id),
            )

    def fetch_commitment(self, commitment_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM commitments WHERE id = ?",
                (commitment_id,),
            )
            return cursor.fetchone()

    def insert_skill(
        self,
        skill_id: str,
        name: str,
        trigger: str,
        steps: str,
        created_at: str,
        source_ref: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO skills (id, name, trigger, steps, created_at, source_ref)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (skill_id, name, trigger, steps, created_at, source_ref),
            )

    def fetch_skill_by_name(self, name: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM skills WHERE name = ?",
                (name,),
            )
            return cursor.fetchone()

    def insert_staged_item(
        self,
        item_id: str,
        kind: str,
        payload: str,
        status: str,
        created_at: str,
        source_ref: str,
        provenance_type: str,
        priority: int = 3,
        next_check_at: str | None = None,
        attempts: int = 0,
        last_error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO staged_items
                (id, kind, payload, status, created_at, source_ref, provenance_type, priority, next_check_at, attempts, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    kind,
                    payload,
                    status,
                    created_at,
                    source_ref,
                    provenance_type,
                    priority,
                    next_check_at,
                    attempts,
                    last_error,
                ),
            )

    def update_staged_item(
        self,
        item_id: str,
        status: str,
        next_check_at: str | None,
        attempts: int,
        last_error: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE staged_items
                SET status = ?, next_check_at = ?, attempts = ?, last_error = ?
                WHERE id = ?
                """,
                (status, next_check_at, attempts, last_error, item_id),
            )

    def update_staged_item_payload(self, item_id: str, payload: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE staged_items SET payload = ? WHERE id = ?",
                (payload, item_id),
            )

    def fetch_next_staged_item(self, now_timestamp: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM staged_items
                WHERE status IN ('pending', 'verifying')
                  AND (next_check_at IS NULL OR next_check_at <= ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now_timestamp,),
            )
            return cursor.fetchone()

    def fetch_staged_item(self, item_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM staged_items WHERE id = ?",
                (item_id,),
            )
            return cursor.fetchone()

    def count_staged_items(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute("SELECT COUNT(*) as count FROM staged_items")
            row = cursor.fetchone()
            return int(row["count"]) if row else 0

    def oldest_pending_staged_item(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM staged_items
                WHERE status IN ('pending', 'verifying')
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            return cursor.fetchone()

    def count_episodes_since_last_sleep(self) -> int:
        last_run = self.last_sleep_run()
        last_timestamp = last_run["last_episode_timestamp"] if last_run else None
        with self.connect() as connection:
            if last_timestamp:
                cursor = connection.execute(
                    "SELECT COUNT(*) as count FROM episodes WHERE timestamp > ? AND quarantine = 0",
                    (last_timestamp,),
                )
            else:
                cursor = connection.execute(
                    "SELECT COUNT(*) as count FROM episodes WHERE quarantine = 0"
                )
            row = cursor.fetchone()
            return int(row["count"]) if row else 0

    def last_sleep_run(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM sleep_runs ORDER BY timestamp DESC LIMIT 1"
            )
            return cursor.fetchone()

    def insert_sleep_run(self, run_id: str, timestamp: str, last_episode_timestamp: str | None) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sleep_runs (id, timestamp, last_episode_timestamp)
                VALUES (?, ?, ?)
                """,
                (run_id, timestamp, last_episode_timestamp),
            )

    def insert_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        origin_episode_id: str,
        confidence: float,
        status: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO hypotheses (id, statement, origin_episode_id, confidence, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (hypothesis_id, statement, origin_episode_id, confidence, status, created_at),
            )
