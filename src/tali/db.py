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

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at DATETIME NOT NULL,
  updated_at DATETIME,
  status TEXT NOT NULL,
  user_prompt TEXT NOT NULL,
  run_summary TEXT,
  current_task_id TEXT,
  last_error TEXT,
  origin TEXT DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  parent_task_id TEXT,
  ordinal INTEGER NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL,
  inputs_json TEXT,
  outputs_json TEXT,
  requires_tools INTEGER DEFAULT 0,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS task_events (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  timestamp DATETIME NOT NULL,
  event_type TEXT NOT NULL,
  payload TEXT,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_status ON tasks (run_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_run_ordinal ON tasks (run_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_task_events_task_time ON task_events (task_id, timestamp);

CREATE TABLE IF NOT EXISTS user_questions (
  id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  reason TEXT,
  created_at DATETIME NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER DEFAULT 3,
  next_ask_at DATETIME,
  attempts INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS patch_proposals (
  id TEXT PRIMARY KEY,
  created_at DATETIME NOT NULL,
  title TEXT NOT NULL,
  rationale TEXT,
  files_json TEXT NOT NULL,
  diff_text TEXT NOT NULL,
  status TEXT NOT NULL,
  test_results TEXT,
  review_json TEXT,
  rollback_ref TEXT
);

CREATE INDEX IF NOT EXISTS idx_questions_status_next ON user_questions (status, next_ask_at);
CREATE INDEX IF NOT EXISTS idx_patch_status ON patch_proposals (status, created_at);

CREATE TABLE IF NOT EXISTS agent_messages (
  id TEXT PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  direction TEXT NOT NULL,
  from_agent_id TEXT,
  from_agent_name TEXT,
  to_agent_id TEXT,
  to_agent_name TEXT,
  topic TEXT NOT NULL,
  correlation_id TEXT,
  payload TEXT NOT NULL,
  status TEXT NOT NULL,
  provenance_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_status ON agent_messages (status, timestamp);

CREATE TABLE IF NOT EXISTS delegations (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  correlation_id TEXT NOT NULL,
  to_agent_id TEXT,
  to_agent_name TEXT,
  status TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_delegations_corr ON delegations (correlation_id);

CREATE TABLE IF NOT EXISTS reflections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME NOT NULL,
  run_id TEXT NOT NULL,
  success INTEGER DEFAULT 0,
  what_worked TEXT,
  what_failed TEXT,
  next_time TEXT,
  metrics_json TEXT,
  created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflections_run ON reflections (run_id, timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS reflections_fts USING fts5(
  id,
  what_worked,
  what_failed,
  next_time
);

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


class _ManagedSQLiteConnection(sqlite3.Connection):
    """SQLite connection that always closes at context-manager exit."""

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


@dataclass
class Database:
    path: Path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, factory=_ManagedSQLiteConnection)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_run_summary_column(connection)
            self._ensure_run_origin_column(connection)
            self._ensure_run_updated_at_column(connection)
            self._ensure_patch_review_json_column(connection)

    def _ensure_run_summary_column(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("PRAGMA table_info(runs)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "run_summary" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN run_summary TEXT")

    def _ensure_run_origin_column(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("PRAGMA table_info(runs)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "origin" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'user'")

    def _ensure_run_updated_at_column(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("PRAGMA table_info(runs)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "updated_at" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN updated_at DATETIME")

    def _ensure_patch_review_json_column(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute("PRAGMA table_info(patch_proposals)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "review_json" not in columns:
            connection.execute("ALTER TABLE patch_proposals ADD COLUMN review_json TEXT")

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

    def upsert_preference(
        self,
        key: str,
        value: str,
        confidence: float,
        provenance_type: str,
        source_ref: str,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO preferences (key, value, confidence, provenance_type, source_ref, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value = excluded.value,
                  confidence = excluded.confidence,
                  provenance_type = excluded.provenance_type,
                  source_ref = excluded.source_ref,
                  updated_at = excluded.updated_at
                """,
                (key, value, confidence, provenance_type, source_ref, updated_at),
            )

    def delete_preference(self, key: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM preferences WHERE key = ?", (key,))

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

    def list_episodes(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                """
            )
            return cursor.fetchall()

    def search_episodes(self, query: str, limit: int) -> list[sqlite3.Row]:
        safe_query = "".join(ch if ch.isalnum() else " " for ch in query).strip()
        if not safe_query:
            return []
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT episodes.* FROM episodes
                JOIN episodes_fts ON episodes.id = episodes_fts.id
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
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

    def prune_episodes(self, cutoff_timestamp: str, max_importance: float = 0.1) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM episodes
                WHERE timestamp < ?
                  AND quarantine = 0
                  AND importance <= ?
                """,
                (cutoff_timestamp, max_importance),
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

    def increment_skill_success(self, name: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE skills
                SET success_count = success_count + 1,
                    last_used = ?
                WHERE name = ?
                """,
                (datetime.utcnow().isoformat(), name),
            )

    def increment_skill_failure(self, name: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE skills
                SET failure_count = failure_count + 1,
                    last_used = ?
                WHERE name = ?
                """,
                (datetime.utcnow().isoformat(), name),
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

    def count_staged_items_by_status(self) -> dict[str, int]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT status, COUNT(*) as count
                FROM staged_items
                GROUP BY status
                """
            )
            return {row["status"]: int(row["count"]) for row in cursor.fetchall()}

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

    def insert_run(
        self,
        run_id: str,
        created_at: str,
        status: str,
        user_prompt: str,
        run_summary: str | None = None,
        current_task_id: str | None = None,
        last_error: str | None = None,
        origin: str = "user",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, created_at, updated_at, status, user_prompt, run_summary, current_task_id, last_error, origin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, created_at, created_at, status, user_prompt, run_summary, current_task_id, last_error, origin),
            )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        current_task_id: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, current_task_id = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, current_task_id, last_error, datetime.utcnow().isoformat(), run_id),
            )

    def update_run_summary(self, run_id: str, run_summary: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET run_summary = ?
                WHERE id = ?
                """,
                (run_summary, run_id),
            )

    def fetch_run(self, run_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            return cursor.fetchone()

    def fetch_active_run(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM runs
                WHERE status IN ('active', 'blocked')
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            return cursor.fetchone()

    def fetch_autonomous_active_run(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM runs
                WHERE status IN ('active', 'blocked')
                  AND origin = 'autonomous'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            return cursor.fetchone()

    def fetch_pending_commitments(self) -> list[sqlite3.Row]:
        now = datetime.utcnow().isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM commitments
                WHERE status = 'pending'
                   OR (due_date IS NOT NULL AND due_date <= ? AND status NOT IN ('done', 'failed', 'cancelled'))
                ORDER BY priority ASC, due_date ASC, last_touched DESC
                """,
                (now,),
            )
            return cursor.fetchall()

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

    # --- Reflections ---

    def insert_reflection(
        self,
        run_id: str,
        timestamp: str,
        success: bool,
        what_worked: str,
        what_failed: str,
        next_time: str,
        metrics_json: str | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reflections (
                    timestamp, run_id, success, what_worked, what_failed, next_time, metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    run_id,
                    1 if success else 0,
                    what_worked,
                    what_failed,
                    next_time,
                    metrics_json,
                    datetime.utcnow().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def list_reflections(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM reflections
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

    def fetch_recent_reflections(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.list_reflections(limit=limit)

    def fetch_reflections_by_ids(self, reflection_ids: list[str]) -> list[sqlite3.Row]:
        if not reflection_ids:
            return []
        # reflections.id is integer; accept strings that parse to int
        ids: list[int] = []
        for rid in reflection_ids:
            try:
                ids.append(int(str(rid)))
            except ValueError:
                continue
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        with self.connect() as connection:
            cursor = connection.execute(
                f"SELECT * FROM reflections WHERE id IN ({placeholders})",
                tuple(ids),
            )
            return cursor.fetchall()

    def search_reflections(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        q = f"%{query}%"
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM reflections
                WHERE what_worked LIKE ? OR what_failed LIKE ? OR next_time LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (q, q, q, limit),
            )
            return cursor.fetchall()

    def insert_task(
        self,
        task_id: str,
        run_id: str,
        parent_task_id: str | None,
        ordinal: int,
        title: str,
        description: str | None,
        status: str,
        inputs_json: str | None,
        outputs_json: str | None,
        requires_tools: int,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    id, run_id, parent_task_id, ordinal, title, description, status,
                    inputs_json, outputs_json, requires_tools, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    parent_task_id,
                    ordinal,
                    title,
                    description,
                    status,
                    inputs_json,
                    outputs_json,
                    requires_tools,
                    created_at,
                    updated_at,
                ),
            )

    def update_task_status(
        self,
        task_id: str,
        status: str,
        outputs_json: str | None,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, outputs_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, outputs_json, updated_at, task_id),
            )

    def update_task_outputs(
        self,
        task_id: str,
        outputs_json: str | None,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET outputs_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (outputs_json, updated_at, task_id),
            )

    def fetch_task(self, task_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            return cursor.fetchone()

    def fetch_tasks_for_run(self, run_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM tasks
                WHERE run_id = ?
                ORDER BY ordinal ASC
                """,
                (run_id,),
            )
            return cursor.fetchall()

    def insert_task_event(
        self,
        event_id: str,
        task_id: str,
        timestamp: str,
        event_type: str,
        payload: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO task_events (id, task_id, timestamp, event_type, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, task_id, timestamp, event_type, payload),
            )

    def fetch_task_events_for_run(self, run_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT task_events.*
                FROM task_events
                JOIN tasks ON tasks.id = task_events.task_id
                WHERE tasks.run_id = ?
                ORDER BY task_events.timestamp ASC
                """,
                (run_id,),
            )
            return cursor.fetchall()

    def insert_user_question(
        self,
        question_id: str,
        question: str,
        reason: str | None,
        created_at: str,
        status: str,
        priority: int,
        next_ask_at: str | None,
        attempts: int = 0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO user_questions
                (id, question, reason, created_at, status, priority, next_ask_at, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    question,
                    reason,
                    created_at,
                    status,
                    priority,
                    next_ask_at,
                    attempts,
                ),
            )

    def fetch_next_user_question(self, now_timestamp: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM user_questions
                WHERE status = 'queued'
                  AND (next_ask_at IS NULL OR next_ask_at <= ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now_timestamp,),
            )
            return cursor.fetchone()

    def fetch_last_asked_question(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM user_questions
                WHERE status = 'asked'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            return cursor.fetchone()

    def update_user_question_status(
        self,
        question_id: str,
        status: str,
        next_ask_at: str | None,
        attempts: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE user_questions
                SET status = ?, next_ask_at = ?, attempts = ?
                WHERE id = ?
                """,
                (status, next_ask_at, attempts, question_id),
            )

    def count_user_questions_since(self, since_timestamp: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT COUNT(*) as count FROM user_questions
                WHERE created_at >= ?
                """,
                (since_timestamp,),
            )
            row = cursor.fetchone()
            return int(row["count"]) if row else 0

    def insert_patch_proposal(
        self,
        proposal_id: str,
        created_at: str,
        title: str,
        rationale: str | None,
        files_json: str,
        diff_text: str,
        status: str,
        test_results: str | None = None,
        rollback_ref: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO patch_proposals
                (id, created_at, title, rationale, files_json, diff_text, status, test_results, rollback_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    created_at,
                    title,
                    rationale,
                    files_json,
                    diff_text,
                    status,
                    test_results,
                    rollback_ref,
                ),
            )

    def list_patch_proposals(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM patch_proposals
                ORDER BY created_at DESC
                """
            )
            return cursor.fetchall()

    def fetch_patch_proposal(self, proposal_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM patch_proposals WHERE id = ?",
                (proposal_id,),
            )
            return cursor.fetchone()

    def update_patch_proposal(
        self,
        proposal_id: str,
        status: str,
        test_results: str | None,
        rollback_ref: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE patch_proposals
                SET status = ?, test_results = ?, rollback_ref = ?
                WHERE id = ?
                """,
                (status, test_results, rollback_ref, proposal_id),
            )

    def count_patch_proposals_since(self, since_timestamp: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT COUNT(*) as count FROM patch_proposals
                WHERE created_at >= ?
                """,
                (since_timestamp,),
            )
            row = cursor.fetchone()
            return int(row["count"]) if row else 0

    def insert_agent_message(
        self,
        message_id: str,
        timestamp: str,
        direction: str,
        from_agent_id: str | None,
        from_agent_name: str | None,
        to_agent_id: str | None,
        to_agent_name: str | None,
        topic: str,
        correlation_id: str | None,
        payload: str,
        status: str,
        provenance_type: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO agent_messages (
                    id, timestamp, direction, from_agent_id, from_agent_name, to_agent_id, to_agent_name,
                    topic, correlation_id, payload, status, provenance_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    timestamp,
                    direction,
                    from_agent_id,
                    from_agent_name,
                    to_agent_id,
                    to_agent_name,
                    topic,
                    correlation_id,
                    payload,
                    status,
                    provenance_type,
                ),
            )

    def list_unread_agent_messages(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM agent_messages
                WHERE status = 'unread'
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

    def list_staged_items(self, statuses: list[str], limit: int = 10) -> list[sqlite3.Row]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT * FROM staged_items
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (*statuses, limit),
            )
            return cursor.fetchall()

    def mark_agent_message(self, message_id: str, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE agent_messages SET status = ? WHERE id = ?",
                (status, message_id),
            )

    def insert_delegation(
        self,
        delegation_id: str,
        task_id: str,
        run_id: str,
        correlation_id: str,
        to_agent_id: str | None,
        to_agent_name: str | None,
        status: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO delegations (
                    id, task_id, run_id, correlation_id, to_agent_id, to_agent_name,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delegation_id,
                    task_id,
                    run_id,
                    correlation_id,
                    to_agent_id,
                    to_agent_name,
                    status,
                    created_at,
                    updated_at,
                ),
            )

    def update_delegation_status(self, correlation_id: str, status: str, updated_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE delegations
                SET status = ?, updated_at = ?
                WHERE correlation_id = ?
                """,
                (status, updated_at, correlation_id),
            )

    def fetch_delegation(self, correlation_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM delegations WHERE correlation_id = ?",
                (correlation_id,),
            )
            return cursor.fetchone()

    # --- Persistence & Recovery ---

    def heartbeat_run(self, run_id: str) -> None:
        """Touch the updated_at timestamp on a run to signal liveness."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), run_id),
            )

    def fetch_stale_runs(self, timeout_minutes: int = 30) -> list[sqlite3.Row]:
        """Return runs in 'active' status whose updated_at is older than timeout."""
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(minutes=timeout_minutes)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM runs
                WHERE status = 'active'
                  AND updated_at IS NOT NULL
                  AND updated_at < ?
                ORDER BY created_at DESC
                """,
                (cutoff,),
            )
            return cursor.fetchall()

    def fetch_incomplete_runs(self) -> list[sqlite3.Row]:
        """Return all runs with status in ('active', 'blocked')."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM runs
                WHERE status IN ('active', 'blocked')
                ORDER BY created_at DESC
                """
            )
            return cursor.fetchall()

    # --- Patch Review ---

    def update_patch_review(self, proposal_id: str, review_json: str, status: str) -> None:
        """Store the LLM review result and optionally update patch status."""
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE patch_proposals
                SET review_json = ?, status = ?
                WHERE id = ?
                """,
                (review_json, status, proposal_id),
            )

    # --- Fact Confidence Updates ---

    def update_fact_confidence(self, fact_id: str, confidence: float) -> None:
        """Update the confidence score for a fact."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE facts SET confidence = ? WHERE id = ?",
                (confidence, fact_id),
            )

    def clear_fact_contested(self, fact_id: str) -> None:
        """Remove the contested flag from a fact."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE facts SET contested = 0 WHERE id = ?",
                (fact_id,),
            )
