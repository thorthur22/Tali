from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ProvenanceType(str, Enum):
    USER_REPORTED = "USER_REPORTED"
    TOOL_VERIFIED = "TOOL_VERIFIED"
    SYSTEM_OBSERVED = "SYSTEM_OBSERVED"
    RETRIEVED_SOURCE = "RETRIEVED_SOURCE"
    INFERRED = "INFERRED"


FORBIDDEN_FACT_TYPES = {"AGENT_OUTPUT"}


@dataclass(frozen=True)
class Fact:
    id: str
    statement: str
    provenance_type: ProvenanceType
    source_ref: str
    confidence: float
    contested: bool


@dataclass(frozen=True)
class Commitment:
    id: str
    description: str
    status: str
    priority: int


@dataclass(frozen=True)
class Preference:
    key: str
    value: str
    confidence: float
    provenance_type: ProvenanceType


@dataclass(frozen=True)
class EpisodeSummary:
    id: str
    timestamp: str
    user_input: str
    agent_output: str


@dataclass(frozen=True)
class RetrievalBundle:
    commitments: Iterable[Commitment]
    facts: Iterable[Fact]
    preferences: Iterable[Preference]
    episodes: Iterable[EpisodeSummary]
    skills: Iterable[str]
