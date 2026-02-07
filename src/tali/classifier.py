from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable
import re

Tier = str


@dataclass(frozen=True)
class ScoringResult:
    score: float
    tier: Tier | None
    confidence: float
    signals: list[str]


@dataclass(frozen=True)
class ScoringConfig:
    token_count_simple: int
    token_count_complex: int
    code_keywords: list[str]
    reasoning_keywords: list[str]
    technical_keywords: list[str]
    creative_keywords: list[str]
    simple_keywords: list[str]
    imperative_verbs: list[str]
    constraint_indicators: list[str]
    output_format_keywords: list[str]
    reference_keywords: list[str]
    negation_keywords: list[str]
    domain_specific_keywords: list[str]
    dimension_weights: dict[str, float]
    simple_medium: float
    medium_complex: float
    complex_reasoning: float
    confidence_steepness: float
    confidence_threshold: float


def default_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        token_count_simple=50,
        token_count_complex=500,
        code_keywords=[
            "function",
            "class",
            "import",
            "def",
            "SELECT",
            "async",
            "await",
            "const",
            "let",
            "var",
            "return",
            "```",
        ],
        reasoning_keywords=[
            "prove",
            "theorem",
            "derive",
            "step by step",
            "chain of thought",
            "formally",
            "mathematical",
            "proof",
            "logically",
        ],
        simple_keywords=[
            "what is",
            "define",
            "translate",
            "hello",
            "yes or no",
            "capital of",
            "how old",
            "who is",
            "when was",
        ],
        technical_keywords=[
            "algorithm",
            "optimize",
            "architecture",
            "distributed",
            "kubernetes",
            "microservice",
            "database",
            "infrastructure",
        ],
        creative_keywords=["story", "poem", "compose", "brainstorm", "creative", "imagine", "write a"],
        imperative_verbs=[
            "build",
            "create",
            "implement",
            "design",
            "develop",
            "construct",
            "generate",
            "deploy",
            "configure",
            "set up",
        ],
        constraint_indicators=[
            "under",
            "at most",
            "at least",
            "within",
            "no more than",
            "o(",
            "maximum",
            "minimum",
            "limit",
            "budget",
        ],
        output_format_keywords=[
            "json",
            "yaml",
            "xml",
            "table",
            "csv",
            "markdown",
            "schema",
            "format as",
            "structured",
        ],
        reference_keywords=[
            "above",
            "below",
            "previous",
            "following",
            "the docs",
            "the api",
            "the code",
            "earlier",
            "attached",
        ],
        negation_keywords=[
            "don't",
            "do not",
            "avoid",
            "never",
            "without",
            "except",
            "exclude",
            "no longer",
        ],
        domain_specific_keywords=[
            "quantum",
            "fpga",
            "vlsi",
            "risc-v",
            "asic",
            "photonics",
            "genomics",
            "proteomics",
            "topological",
            "homomorphic",
            "zero-knowledge",
            "lattice-based",
        ],
        dimension_weights={
            "tokenCount": 0.08,
            "codePresence": 0.15,
            "reasoningMarkers": 0.18,
            "technicalTerms": 0.1,
            "creativeMarkers": 0.05,
            "simpleIndicators": 0.12,
            "multiStepPatterns": 0.12,
            "questionComplexity": 0.05,
            "imperativeVerbs": 0.03,
            "constraintCount": 0.04,
            "outputFormat": 0.03,
            "referenceComplexity": 0.02,
            "negationComplexity": 0.01,
            "domainSpecificity": 0.02,
        },
        simple_medium=0.0,
        medium_complex=0.15,
        complex_reasoning=0.25,
        confidence_steepness=12.0,
        confidence_threshold=0.7,
    )


def classify_request(
    prompt: str,
    system_prompt: str | None = None,
    estimated_tokens: int | None = None,
    config: ScoringConfig | None = None,
) -> ScoringResult:
    cfg = config or default_scoring_config()
    token_estimate = estimated_tokens if estimated_tokens is not None else _estimate_tokens(prompt)
    return _classify_by_rules(prompt, system_prompt, token_estimate, cfg)


def _classify_by_rules(
    prompt: str,
    system_prompt: str | None,
    estimated_tokens: int,
    config: ScoringConfig,
) -> ScoringResult:
    text = f"{system_prompt or ''} {prompt}".lower()
    dimensions = [
        _score_token_count(estimated_tokens, config),
        _score_keyword_match(
            text,
            config.code_keywords,
            "codePresence",
            "code",
            low=1,
            high=2,
            scores=(0, 0.5, 1.0),
        ),
        _score_keyword_match(
            text,
            config.reasoning_keywords,
            "reasoningMarkers",
            "reasoning",
            low=1,
            high=2,
            scores=(0, 0.7, 1.0),
        ),
        _score_keyword_match(
            text,
            config.technical_keywords,
            "technicalTerms",
            "technical",
            low=2,
            high=4,
            scores=(0, 0.5, 1.0),
        ),
        _score_keyword_match(
            text,
            config.creative_keywords,
            "creativeMarkers",
            "creative",
            low=1,
            high=2,
            scores=(0, 0.5, 0.7),
        ),
        _score_keyword_match(
            text,
            config.simple_keywords,
            "simpleIndicators",
            "simple",
            low=1,
            high=2,
            scores=(0, -1.0, -1.0),
        ),
        _score_multi_step(text),
        _score_question_complexity(prompt),
        _score_keyword_match(
            text,
            config.imperative_verbs,
            "imperativeVerbs",
            "imperative",
            low=1,
            high=2,
            scores=(0, 0.3, 0.5),
        ),
        _score_keyword_match(
            text,
            config.constraint_indicators,
            "constraintCount",
            "constraints",
            low=1,
            high=3,
            scores=(0, 0.3, 0.7),
        ),
        _score_keyword_match(
            text,
            config.output_format_keywords,
            "outputFormat",
            "format",
            low=1,
            high=2,
            scores=(0, 0.4, 0.7),
        ),
        _score_keyword_match(
            text,
            config.reference_keywords,
            "referenceComplexity",
            "references",
            low=1,
            high=2,
            scores=(0, 0.3, 0.5),
        ),
        _score_keyword_match(
            text,
            config.negation_keywords,
            "negationComplexity",
            "negation",
            low=2,
            high=3,
            scores=(0, 0.3, 0.5),
        ),
        _score_keyword_match(
            text,
            config.domain_specific_keywords,
            "domainSpecificity",
            "domain-specific",
            low=1,
            high=2,
            scores=(0, 0.5, 0.8),
        ),
    ]
    signals = [d.signal for d in dimensions if d.signal is not None]
    weighted_score = 0.0
    for d in dimensions:
        weight = config.dimension_weights.get(d.name, 0.0)
        weighted_score += d.score * weight
    reasoning_matches = _count_keyword_matches(text, config.reasoning_keywords)
    if reasoning_matches >= 2:
        confidence = _calibrate_confidence(max(weighted_score, 0.3), config.confidence_steepness)
        return ScoringResult(
            score=weighted_score,
            tier="REASONING",
            confidence=max(confidence, 0.85),
            signals=signals,
        )
    tier, distance = _map_to_tier(
        weighted_score,
        config.simple_medium,
        config.medium_complex,
        config.complex_reasoning,
    )
    confidence = _calibrate_confidence(distance, config.confidence_steepness)
    if confidence < config.confidence_threshold:
        return ScoringResult(score=weighted_score, tier=None, confidence=confidence, signals=signals)
    return ScoringResult(score=weighted_score, tier=tier, confidence=confidence, signals=signals)


@dataclass(frozen=True)
class _DimensionScore:
    name: str
    score: float
    signal: str | None


def _score_token_count(estimated_tokens: int, config: ScoringConfig) -> _DimensionScore:
    if estimated_tokens < config.token_count_simple:
        return _DimensionScore("tokenCount", -1.0, f"short ({estimated_tokens} tokens)")
    if estimated_tokens > config.token_count_complex:
        return _DimensionScore("tokenCount", 1.0, f"long ({estimated_tokens} tokens)")
    return _DimensionScore("tokenCount", 0.0, None)


def _score_keyword_match(
    text: str,
    keywords: Iterable[str],
    name: str,
    signal_label: str,
    low: int,
    high: int,
    scores: tuple[float, float, float],
) -> _DimensionScore:
    matches = _keyword_matches(text, keywords)
    if len(matches) >= high:
        return _DimensionScore(name, scores[2], f"{signal_label} ({', '.join(matches[:3])})")
    if len(matches) >= low:
        return _DimensionScore(name, scores[1], f"{signal_label} ({', '.join(matches[:3])})")
    return _DimensionScore(name, scores[0], None)


def _score_multi_step(text: str) -> _DimensionScore:
    patterns = [r"first.*then", r"step \d", r"\d\.\s"]
    if any(re.search(pattern, text) for pattern in patterns):
        return _DimensionScore("multiStepPatterns", 0.5, "multi-step")
    return _DimensionScore("multiStepPatterns", 0.0, None)


def _score_question_complexity(prompt: str) -> _DimensionScore:
    count = prompt.count("?")
    if count > 3:
        return _DimensionScore("questionComplexity", 0.5, f"{count} questions")
    return _DimensionScore("questionComplexity", 0.0, None)


def _map_to_tier(
    weighted_score: float,
    simple_medium: float,
    medium_complex: float,
    complex_reasoning: float,
) -> tuple[Tier, float]:
    if weighted_score < simple_medium:
        return "SIMPLE", simple_medium - weighted_score
    if weighted_score < medium_complex:
        return "MEDIUM", min(weighted_score - simple_medium, medium_complex - weighted_score)
    if weighted_score < complex_reasoning:
        return "COMPLEX", min(weighted_score - medium_complex, complex_reasoning - weighted_score)
    return "REASONING", weighted_score - complex_reasoning


def _calibrate_confidence(distance: float, steepness: float) -> float:
    return 1 / (1 + exp(-steepness * distance))


def _estimate_tokens(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, int(len(cleaned) / 4))


def _keyword_matches(text: str, keywords: Iterable[str]) -> list[str]:
    lowered = text.lower()
    matches = [kw for kw in keywords if kw.lower() in lowered]
    return matches


def _count_keyword_matches(text: str, keywords: Iterable[str]) -> int:
    return len(_keyword_matches(text, keywords))

