from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Iterable

from tali.config import RetrievalConfig
from tali.db import Database
from tali.models import (
    Commitment,
    EpisodeSummary,
    Fact,
    Preference,
    ProvenanceType,
    RetrievalBundle,
    ReflectionSummary,
)
from tali.vector_index import VectorIndex, VectorItem


PROVENANCE_PRIORITY = {
    ProvenanceType.TOOL_VERIFIED.value: 5,
    ProvenanceType.SYSTEM_OBSERVED.value: 4,
    ProvenanceType.RETRIEVED_SOURCE.value: 3,
    ProvenanceType.USER_REPORTED.value: 2,
    ProvenanceType.INFERRED.value: 1,
}


@dataclass(frozen=True)
class RetrievalContext:
    bundle: RetrievalBundle
    token_budget: int


class Retriever:
    def __init__(
        self,
        db: Database,
        config: RetrievalConfig | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self.db = db
        self.config = config or RetrievalConfig()
        self.vector_index = vector_index
        self._last_vector_sync_at: float | None = None

    def retrieve(self, user_input: str) -> RetrievalContext:
        self._sync_vector_index()
        commitments = self._commitments()
        vector_hits = self._vector_hits(user_input)
        facts = self._facts(user_input, vector_hits)
        episodes = self._episodes(user_input, vector_hits)
        reflections = self._reflections(user_input, vector_hits)
        preferences = self._preferences()
        skills = self._skills(user_input)
        return RetrievalContext(
            bundle=RetrievalBundle(
                commitments=commitments,
                facts=facts,
                preferences=preferences,
                episodes=episodes,
                reflections=reflections,
                skills=skills,
            ),
            token_budget=self.config.token_budget,
        )

    def _reflections(self, user_input: str, vector_hits: list[VectorItem]) -> Iterable[ReflectionSummary]:
        vector_ref_ids = [hit.item_id for hit in vector_hits if hit.item_type == "reflection"]
        rows = self.db.fetch_reflections_by_ids(vector_ref_ids)
        if not rows:
            rows = self.db.search_reflections(user_input, self.config.max_episodes)
        if not rows:
            rows = self.db.fetch_recent_reflections(self.config.max_episodes)
        vector_rank = {item_id: idx for idx, item_id in enumerate(vector_ref_ids)}
        rows = sorted(
            rows,
            key=lambda row: (
                vector_rank.get(str(row["id"]), len(vector_rank) + 1),
                -_timestamp_score(row["timestamp"]),
            ),
        )
        return [
            ReflectionSummary(
                id=str(row["id"]),
                timestamp=str(row["timestamp"]),
                run_id=str(row["run_id"]),
                success=bool(row["success"]),
                what_worked=str(row["what_worked"] or ""),
                what_failed=str(row["what_failed"] or ""),
                next_time=str(row["next_time"] or ""),
            )
            for row in rows
        ]

    def _commitments(self) -> Iterable[Commitment]:
        rows = self.db.list_commitments()
        ranked = sorted(
            rows,
            key=lambda row: (
                int(row["priority"] or 3),
                -_timestamp_score(row["last_touched"]),
            ),
        )
        selected = ranked[: self.config.max_commitments]
        return [
            Commitment(
                id=row["id"],
                description=row["description"],
                status=row["status"],
                priority=row["priority"],
            )
            for row in selected
        ]

    def _facts(self, user_input: str, vector_hits: list[VectorItem]) -> Iterable[Fact]:
        vector_fact_ids = [hit.item_id for hit in vector_hits if hit.item_type == "fact"]
        rows = self.db.fetch_facts_by_ids(vector_fact_ids)
        if not rows:
            rows = self.db.search_facts(user_input, self.config.max_facts)
        if not rows:
            rows = self.db.list_facts()
        direct_ids = {hit.item_id for hit in vector_hits if hit.item_type == "fact"}
        filtered_rows = []
        for row in rows:
            is_direct = row["id"] in direct_ids
            if row["contested"] and not is_direct:
                continue
            if (
                row["provenance_type"] == ProvenanceType.INFERRED.value
                and row["confidence"] < 0.5
                and not is_direct
            ):
                continue
            filtered_rows.append(row)
        rows = filtered_rows
        vector_rank = {item_id: idx for idx, item_id in enumerate(vector_fact_ids)}
        ranked = sorted(
            rows,
            key=lambda row: (
                vector_rank.get(row["id"], len(vector_rank) + 1),
                -PROVENANCE_PRIORITY.get(row["provenance_type"], 0),
                -row["confidence"],
                -_timestamp_score(row["last_confirmed"] or row["created_at"]),
            ),
        )
        selected = _diversify_rows(ranked, self.config.max_facts)
        return [
            Fact(
                id=row["id"],
                statement=row["statement"],
                provenance_type=ProvenanceType(row["provenance_type"]),
                source_ref=row["source_ref"],
                confidence=row["confidence"],
                contested=bool(row["contested"]),
            )
            for row in selected
        ]

    def _episodes(self, user_input: str, vector_hits: list[VectorItem]) -> Iterable[EpisodeSummary]:
        vector_episode_ids = [hit.item_id for hit in vector_hits if hit.item_type == "episode"]
        rows = self.db.fetch_episodes_by_ids(vector_episode_ids)
        if not rows:
            rows = self.db.search_episodes(user_input, self.config.max_episodes)
        if not rows:
            rows = self.db.fetch_recent_episodes(self.config.max_episodes)
        vector_rank = {item_id: idx for idx, item_id in enumerate(vector_episode_ids)}
        rows = sorted(
            rows,
            key=lambda row: (
                vector_rank.get(row["id"], len(vector_rank) + 1),
                -_timestamp_score(row["timestamp"]),
            ),
        )
        return [
            EpisodeSummary(
                id=row["id"],
                timestamp=row["timestamp"],
                user_input=row["user_input"],
                agent_output=row["agent_output"],
            )
            for row in rows
        ]

    def _preferences(self) -> Iterable[Preference]:
        rows = self.db.list_preferences()
        selected = rows[: self.config.max_preferences]
        return [
            Preference(
                key=row["key"],
                value=row["value"],
                confidence=row["confidence"],
                provenance_type=ProvenanceType(row["provenance_type"]),
            )
            for row in selected
        ]

    def _skills(self, user_input: str) -> Iterable[str]:
        rows = self.db.list_skills()
        lowered = user_input.lower()
        matched = [row["name"] for row in rows if row["trigger"].lower() in lowered]
        return matched

    def _vector_hits(self, user_input: str) -> list[VectorItem]:
        if not self.vector_index:
            return []
        return self.vector_index.search(user_input, k=self.config.vector_k)

    def _sync_vector_index(self) -> None:
        if not self.vector_index:
            return
        now = time.monotonic()
        if (
            self._last_vector_sync_at is not None
            and now - self._last_vector_sync_at < self.config.vector_sync_min_interval_s
        ):
            return
        self._last_vector_sync_at = now
        mapping_items = list(self.vector_index.mapping.values())
        if not mapping_items and not self.vector_index.needs_rebuild:
            return
        facts_ids = [item.item_id for item in mapping_items if item.item_type == "fact"]
        episode_ids = [item.item_id for item in mapping_items if item.item_type == "episode"]
        reflection_ids = [item.item_id for item in mapping_items if item.item_type == "reflection"]
        missing = False
        if facts_ids:
            found = {row["id"] for row in self.db.fetch_facts_by_ids(facts_ids)}
            if len(found) < len(facts_ids):
                missing = True
        if episode_ids:
            found = {row["id"] for row in self.db.fetch_episodes_by_ids(episode_ids)}
            if len(found) < len(episode_ids):
                missing = True
        if reflection_ids:
            found = {row["id"] for row in self.db.fetch_reflections_by_ids(reflection_ids)}
            if len(found) < len(reflection_ids):
                missing = True
        if missing or self.vector_index.needs_rebuild:
            items: list[tuple[str, str, str]] = []
            for row in self.db.list_facts():
                items.append(("fact", row["id"], row["statement"]))
            for row in self.db.list_episodes():
                items.append(("episode", row["id"], f"{row['user_input']}\n{row['agent_output']}"))
            for row in self.db.list_reflections(limit=self.config.max_episodes * 10):
                text = "\n".join(
                    [
                        f"success={bool(row['success'])}",
                        str(row.get("what_worked") or ""),
                        str(row.get("what_failed") or ""),
                        str(row.get("next_time") or ""),
                    ]
                ).strip()
                items.append(("reflection", str(row["id"]), text))
            self.vector_index.rebuild_from_items(items)


def _timestamp_score(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _diversify_rows(rows: list[object], limit: int) -> list[object]:
    selected: list[object] = []
    for row in rows:
        if len(selected) >= limit:
            break
        statement = str(row["statement"] or "")
        if not statement:
            continue
        if _is_too_similar(statement, [str(r["statement"] or "") for r in selected]):
            continue
        selected.append(row)
    return selected


def _is_too_similar(statement: str, existing: list[str]) -> bool:
    if not existing:
        return False
    tokens = _tokenize(statement)
    for other in existing:
        other_tokens = _tokenize(other)
        if not tokens or not other_tokens:
            continue
        overlap = len(tokens & other_tokens) / max(len(tokens | other_tokens), 1)
        if overlap >= 0.8:
            return True
    return False


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if len(token) > 2}
