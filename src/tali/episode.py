from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from tali.guardrails import GuardrailResult
from tali.models import RetrievalBundle
from tali.prompting import format_retrieval_context
from tali.prompts import SYSTEM_RULES_WITH_PATCH


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
    parts.append(SYSTEM_RULES_WITH_PATCH)
    parts.append(format_retrieval_context(bundle))
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
