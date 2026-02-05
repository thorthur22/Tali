from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tali.config import RetrievalConfig
from tali.db import Database
from tali.models import Commitment, EpisodeSummary, Fact, Preference, ProvenanceType, RetrievalBundle
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

    def retrieve(self, user_input: str) -> RetrievalContext:
        commitments = self._commitments()
        vector_hits = self._vector_hits(user_input)
        facts = self._facts(user_input, vector_hits)
        episodes = self._episodes(user_input, vector_hits)
        preferences = self._preferences()
        skills = self._skills(user_input)
        return RetrievalContext(
            bundle=RetrievalBundle(
                commitments=commitments,
                facts=facts,
                preferences=preferences,
                episodes=episodes,
                skills=skills,
            ),
            token_budget=self.config.token_budget,
        )

    def _commitments(self) -> Iterable[Commitment]:
        rows = self.db.list_commitments()
        selected = rows[: self.config.max_commitments]
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
        ranked = sorted(
            rows,
            key=lambda row: (
                row["contested"],
                -PROVENANCE_PRIORITY.get(row["provenance_type"], 0),
                -row["confidence"],
            ),
        )
        selected = ranked[: self.config.max_facts]
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
        return self.vector_index.search(user_input, k=5)
