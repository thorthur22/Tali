from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceCandidate:
    key: str
    value: str
    confidence: float


_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("user_prefer", re.compile(r"\\bprefer(?: to)? (?P<value>[^.?!]+)", re.IGNORECASE), 0.85),
    ("user_use", re.compile(r"\\bplease use (?P<value>[^.?!]+)", re.IGNORECASE), 0.8),
    ("user_always", re.compile(r"\\balways (?:use|do) (?P<value>[^.?!]+)", re.IGNORECASE), 0.8),
    ("user_avoid", re.compile(r"\\bavoid (?P<value>[^.?!]+)", re.IGNORECASE), 0.8),
    ("user_avoid", re.compile(r"\\bdo not (?P<value>[^.?!]+)", re.IGNORECASE), 0.75),
]


def extract_preferences(user_input: str) -> list[PreferenceCandidate]:
    if not user_input:
        return []
    candidates: list[PreferenceCandidate] = []
    for key, pattern, confidence in _PATTERNS:
        match = pattern.search(user_input)
        if not match:
            continue
        value = (match.group("value") or "").strip()
        if not value:
            continue
        candidates.append(PreferenceCandidate(key=key, value=value, confidence=confidence))
    return candidates
