from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tali.db import Database
from tali.llm import LLMClient
from tali.models import FORBIDDEN_FACT_TYPES


@dataclass(frozen=True)
class SleepResult:
    generated_at: str
    fact_candidates: list[dict[str, str]]
    commitment_updates: list[dict[str, str]]
    skill_candidates: list[dict[str, str]]
    notes: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "fact_candidates": self.fact_candidates,
                "commitment_updates": self.commitment_updates,
                "skill_candidates": self.skill_candidates,
                "notes": self.notes,
            },
            indent=2,
        )


def run_sleep(db: Database, output_dir: Path, llm: LLMClient, episode_limit: int = 50) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    last_run = db.last_sleep_run()
    last_timestamp = last_run["last_episode_timestamp"] if last_run else None
    episodes = db.list_episodes_since(last_timestamp, limit=episode_limit)
    prompt = build_sleep_prompt(episodes)
    response = llm.generate(prompt)
    payload = _parse_json(response.content)
    if not isinstance(payload, dict):
        raise ValueError("Sleep output must be a JSON object.")
    result = SleepResult(
        generated_at=datetime.utcnow().isoformat(),
        fact_candidates=payload.get("fact_candidates", []),
        commitment_updates=payload.get("commitment_updates", []),
        skill_candidates=payload.get("skill_candidates", []),
        notes=payload.get("notes", []),
    )
    output_path = output_dir / f"sleep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(result.to_json())
    db.insert_sleep_run(
        run_id=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        timestamp=datetime.utcnow().isoformat(),
        last_episode_timestamp=episodes[-1]["timestamp"] if episodes else last_timestamp,
    )
    return output_path


def load_sleep_output(path: Path) -> dict[str, object]:
    payload = _parse_json(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Sleep output must be a JSON object.")
    return payload


def build_sleep_prompt(episodes: Iterable[object]) -> str:
    parts = [
        "You are the consolidation engine. Return strict JSON only.",
        "Extract fact_candidates from episodes. Facts must be grounded to episodes.",
        "Never include AGENT_OUTPUT provenance.",
        "Output JSON format:",
        "{",
        '  "fact_candidates": [{"statement": "...", "provenance_type": "USER_REPORTED", "source_ref": "episode_id", "tags": []}],',
        '  "commitment_updates": [],',
        '  "skill_candidates": [],',
        '  "notes": []',
        "}",
        f"Forbidden fact types: {sorted(FORBIDDEN_FACT_TYPES)}",
        "",
        "[Episodes]",
    ]
    for episode in episodes:
        parts.append(
            f"- id={episode['id']} user={episode['user_input']} agent={episode['agent_output']}"
        )
    return "\n".join(parts)


def _parse_json(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Sleep output must be a JSON object.")
    return payload
