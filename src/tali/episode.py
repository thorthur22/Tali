from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from tali.guardrails import GuardrailResult
from tali.models import RetrievalBundle


@dataclass(frozen=True)
class Episode:
    id: str
    timestamp: str
    user_input: str
    agent_output: str
    tool_calls: list[dict[str, str]]
    outcome: str
    quarantine: int


@dataclass(frozen=True)
class EpisodeContext:
    prompt: str
    bundle: RetrievalBundle


def new_episode_id() -> str:
    return str(uuid.uuid4())


def build_prompt(bundle: RetrievalBundle, user_input: str) -> str:
    parts: list[str] = []
    parts.append("System rules: Do not claim memory without citation. Ask if unsure. No recursion.")
    parts.append("\n[Active Commitments]")
    if bundle.commitments:
        for commitment in bundle.commitments:
            parts.append(f"- ({commitment.status}) {commitment.description}")
    else:
        parts.append("- None")
    parts.append("\n[Relevant Facts]")
    if bundle.facts:
        for fact in bundle.facts:
            parts.append(
                f"- {fact.statement} (id={fact.id}, provenance={fact.provenance_type.value}, confidence={fact.confidence:.2f})"
            )
    else:
        parts.append("- None")
    parts.append("\n[Recent Episodes]")
    if bundle.episodes:
        for episode in bundle.episodes:
            parts.append(f"- {episode.timestamp}: {episode.user_input} -> {episode.agent_output}")
    else:
        parts.append("- None")
    parts.append("\n[Preferences]")
    if bundle.preferences:
        for pref in bundle.preferences:
            parts.append(
                f"- {pref.key} = {pref.value} (provenance={pref.provenance_type.value}, confidence={pref.confidence:.2f})"
            )
    else:
        parts.append("- None")
    parts.append("\n[Skills]")
    if bundle.skills:
        for skill in bundle.skills:
            parts.append(f"- {skill}")
    else:
        parts.append("- None")
    parts.append("\nUser:")
    parts.append(user_input)
    return "\n".join(parts)


def build_episode(
    user_input: str,
    guardrail: GuardrailResult,
    tool_calls: list[dict[str, str]],
    outcome: str,
    quarantine: int,
) -> Episode:
    return Episode(
        id=new_episode_id(),
        timestamp=datetime.utcnow().isoformat(),
        user_input=user_input,
        agent_output=guardrail.safe_output,
        tool_calls=tool_calls,
        outcome=outcome,
        quarantine=quarantine,
    )
