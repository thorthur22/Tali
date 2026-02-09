from __future__ import annotations

"""
Self reflection support for Tali.

This module provides a simple API to record reflections after a run has
completed. Reflections are stored as opaque JSON strings in a dedicated
database table. They can be retrieved later for analysis or used as a
foundation for self‑improvement heuristics.
"""

import json
from datetime import datetime

from tali.db import Database


class SelfReflection:
    """
    Persist post‑run reflections. Each reflection includes metadata about the
    run, a success flag, free‑form notes, and optional improvement ideas.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def reflect(
        self,
        run_id: str,
        success: bool,
        notes: str,
        improvement: str | None = None,
    ) -> None:
        """
        Record a reflection payload. The payload is stored as a JSON object
        containing the run identifier, success flag, notes, improvement
        suggestions, and a timestamp.
        """
        payload = {
            "run_id": run_id,
            "success": success,
            "notes": notes,
            "improvement": improvement,
            "timestamp": datetime.utcnow().isoformat(),
        }
        # Use the database helper to insert reflections. Each payload is
        # treated as opaque JSON; the database helper assigns a UUID.
        self.db.insert_reflection(json.dumps(payload))
