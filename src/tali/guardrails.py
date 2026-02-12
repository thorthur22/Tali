from __future__ import annotations

from dataclasses import dataclass
import re

from tali.config import GuardrailConfig
from tali.models import RetrievalBundle


@dataclass(frozen=True)
class GuardrailResult:
    safe_output: str
    flags: list[str]


class Guardrails:
    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()

    def enforce(self, raw_output: str, bundle: RetrievalBundle) -> GuardrailResult:
        flags: list[str] = []
        safe_output = raw_output.strip()
        if not safe_output:
            safe_output = "I don't have a response yet."
            flags.append("empty_output")
        citations = self._extract_citations(safe_output)
        invalid = self._invalid_citations(citations, bundle)
        memory_signal = self._contains_memory_signal(safe_output) or bool(citations)
        if memory_signal and not citations:
            flags.append("missing_memory_citation")
        elif invalid:
            flags.append("invalid_memory_citation")
        return GuardrailResult(safe_output=safe_output, flags=flags)

    def _contains_memory_signal(self, output: str) -> bool:
        lowered = output.lower()
        return any(
            phrase in lowered
            for phrase in (
                "as we discussed",
                "you told me",
                "as noted before",
                "earlier you said",
                "previously you said",
                "remember",
            )
        )

    def _extract_citations(self, output: str) -> list[tuple[str, str]]:
        pattern = re.compile(r"\[(fact|commitment|preference|episode|skill):([^\]]+)\]")
        return [(match.group(1), match.group(2).strip()) for match in pattern.finditer(output)]

    def _invalid_citations(
        self, citations: list[tuple[str, str]], bundle: RetrievalBundle
    ) -> list[tuple[str, str]]:
        fact_ids = {fact.id for fact in bundle.facts}
        commitment_ids = {commitment.id for commitment in bundle.commitments}
        preference_keys = {pref.key for pref in bundle.preferences}
        episode_ids = {episode.id for episode in bundle.episodes}
        skill_names = set(bundle.skills)
        invalid: list[tuple[str, str]] = []
        for kind, value in citations:
            if kind == "fact" and value not in fact_ids:
                invalid.append((kind, value))
            elif kind == "commitment" and value not in commitment_ids:
                invalid.append((kind, value))
            elif kind == "preference" and value not in preference_keys:
                invalid.append((kind, value))
            elif kind == "episode" and value not in episode_ids:
                invalid.append((kind, value))
            elif kind == "skill" and value not in skill_names:
                invalid.append((kind, value))
        return invalid
