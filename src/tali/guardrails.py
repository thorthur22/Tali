from __future__ import annotations

from dataclasses import dataclass

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
        if self._claims_memory(safe_output, bundle):
            safe_output = (
                "I don't have verified memory for that yet. "
                "If you want it stored, please provide a source so it can be added safely."
            )
            flags.append("unverified_memory_claim")
        return GuardrailResult(safe_output=safe_output, flags=flags)

    def _claims_memory(self, output: str, bundle: RetrievalBundle) -> bool:
        lowered = output.lower()
        if "as we discussed" in lowered or "you told me" in lowered:
            return True
        known_ids = {fact.id for fact in bundle.facts}
        return any(fact_id in output for fact_id in known_ids)
