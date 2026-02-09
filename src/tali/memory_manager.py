from __future__ import annotations

"""
Memory manager for Tali.

This module centralizes simple memory operations such as summarizing the top
facts for planning context, reinforcing facts when they are used, and invoking
confidence decay over time. It is intentionally minimal so that it can be
easily imported from CLI or scheduling code without pulling in large
dependencies. The MemoryManager does not replace the existing sleep
consolidation or memory hygiene jobs—it supplements them with real‑time
helpers.
"""

from datetime import datetime, timedelta
from typing import Optional, List

from tali.db import Database
from tali.vector_index import VectorIndex


class MemoryManager:
    """Manage agent memory operations."""

    def __init__(self, db: Database, vector_index: VectorIndex) -> None:
        self.db = db
        self.vector = vector_index

    def reinforce_fact(self, fact_id: str, boost: float = 0.05) -> None:
        """
        Increase the confidence of a fact by a small amount. This can be
        invoked when a fact is cited or otherwise relied upon. Confidence is
        clamped to the range [0.0, 1.0].
        """
        rows = self.db.fetch_facts_by_ids([fact_id])
        if not rows:
            return
        row = rows[0]
        try:
            current_conf = float(row["confidence"])
        except Exception:
            return
        new_conf = min(1.0, current_conf + boost)
        self.db.update_fact_confidence(str(row["id"]), new_conf)

    def decay_facts(self, days: int = 30) -> None:
        """
        Trigger fact confidence decay for facts older than the given number
        of days. This delegates to the database's apply_fact_decay routine.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        self.db.apply_fact_decay(cutoff)

    def summarize_memory(self, limit: int = 5) -> str:
        """
        Return a human‑readable summary of up to `limit` facts with the
        highest confidence. Facts that are contested are sorted to the end
        automatically by the database query.
        """
        facts = self.db.list_facts()
        summary_lines: List[str] = []
        for row in facts[:limit]:
            statement = str(row["statement"])
            try:
                conf = float(row["confidence"])
            except Exception:
                conf = 0.0
            summary_lines.append(f"- {statement} (confidence: {conf:.2f})")
        return "\n".join(summary_lines)

    def promote_episode_to_fact(self, episode_id: str, statement: str, confidence: float = 0.6) -> Optional[str]:
        """
        Create a new fact from an episode statement and add it to the vector
        index. Returns the ID of the new fact, or None on error.
        """
        import uuid

        if not statement:
            return None
        fact_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.db.insert_fact(
            fact_id=fact_id,
            statement=statement,
            provenance_type="AGENT_DERIVED",
            source_ref=episode_id,
            confidence=confidence,
            created_at=now,
            contested=0,
        )
        # Add to vector index so it is retrievable immediately
        try:
            self.vector.add(item_type="fact", item_id=fact_id, text=statement)
        except Exception:
            # Swallow vector index errors; fact will still exist in DB
            pass
        return fact_id
