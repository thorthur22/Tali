from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from tali.db import Database
from tali.knowledge_sources import KnowledgeSourceRegistry
from tali.models import ProvenanceType
from tali.patches import PatchProposal, parse_patch_proposal, review_patch, store_patch_proposal
from tali.questions import queue_question


MAX_IDLE_LLM_CALLS = 4
MAX_PATCHES_PER_DAY = 1
MAX_QUESTIONS_PER_DAY = 3
IDLE_CYCLE_SECONDS = 60


@dataclass(frozen=True)
class IdleJobResult:
    messages: list[str]
    llm_calls: int


class IdleJobRunner:
    def __init__(
        self,
        db: Database,
        llm: Any,
        sources: KnowledgeSourceRegistry,
        should_stop: Callable[[], bool],
        hook_manager: Any | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.sources = sources
        self.should_stop = should_stop
        self.hook_manager = hook_manager

    def run_cycle(self) -> IdleJobResult:
        messages: list[str] = []
        llm_calls = 0
        start = datetime.utcnow()
        for job in (
            self._job_skill_review,
            self._job_memory_hygiene,
            self._job_knowledge_expansion,
            self._job_clarifying_question,
            self._job_patch_proposal,
        ):
            if self.should_stop():
                break
            if (datetime.utcnow() - start).total_seconds() >= IDLE_CYCLE_SECONDS:
                break
            if llm_calls >= MAX_IDLE_LLM_CALLS:
                break
            job_result = job()
            if job_result:
                messages.extend(job_result.messages)
                llm_calls += job_result.llm_calls
        return IdleJobResult(messages=messages, llm_calls=llm_calls)

    def _has_active_run(self) -> bool:
        run = self.db.fetch_active_run()
        return bool(run and run["status"] in {"active", "blocked"})

    def _has_active_commitments(self) -> bool:
        commitments = self.db.list_commitments()
        return any(row["status"] in {"pending", "active"} for row in commitments)

    def _job_skill_review(self) -> IdleJobResult | None:
        episodes = [row for row in self.db.fetch_recent_episodes(limit=20) if row["quarantine"] == 0]
        skills = self.db.list_skills()
        if not skills or not episodes:
            return None
        prompt = _build_skill_review_prompt(skills, episodes)
        response = self.llm.generate(prompt)
        proposals, error = _parse_skill_review(response.content)
        if error or not proposals:
            return IdleJobResult(messages=["Idle: skill review skipped (invalid response)."], llm_calls=1)
        for proposal in proposals:
            payload = {
                "kind": "skill_improvement",
                "skill_name": proposal.get("skill_name", ""),
                "suggestion": proposal.get("suggestion", ""),
                "reason": proposal.get("reason", ""),
            }
            self.db.insert_staged_item(
                item_id=str(uuid.uuid4()),
                kind="skill",
                payload=json.dumps(payload),
                status="pending",
                created_at=datetime.utcnow().isoformat(),
                source_ref="idle_skill_review",
                provenance_type=ProvenanceType.SYSTEM_OBSERVED.value,
                next_check_at=datetime.utcnow().isoformat(),
            )
        return IdleJobResult(messages=["Idle: skill review staged proposals."], llm_calls=1)

    def _job_memory_hygiene(self) -> IdleJobResult | None:
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        self.db.apply_fact_decay(cutoff)
        self.db.dedupe_facts()
        self.db.prune_episodes(cutoff_timestamp=cutoff, max_importance=0.1)
        return IdleJobResult(messages=["Idle: memory hygiene applied."], llm_calls=0)

    def _job_knowledge_expansion(self) -> IdleJobResult | None:
        sources = list(self.sources.list_sources())
        if not sources:
            return None
        topics: list[str] = []
        for source in sources:
            topics.extend(source.list_topics())
        if not topics:
            return None
        for source in sources:
            for topic in source.list_topics()[:3]:
                for excerpt in source.fetch(topic):
                    payload = {
                        "statement": excerpt.content.strip(),
                        "provenance_type": ProvenanceType.RETRIEVED_SOURCE.value,
                        "source_ref": excerpt.source_ref,
                    }
                    self.db.insert_staged_item(
                        item_id=str(uuid.uuid4()),
                        kind="fact",
                        payload=json.dumps(payload),
                        status="pending",
                        created_at=datetime.utcnow().isoformat(),
                        source_ref=excerpt.source_ref,
                        provenance_type=ProvenanceType.RETRIEVED_SOURCE.value
                        if excerpt.trusted
                        else ProvenanceType.INFERRED.value,
                        next_check_at=datetime.utcnow().isoformat(),
                    )
        return IdleJobResult(messages=["Idle: knowledge expansion staged excerpts."], llm_calls=0)

    def _job_clarifying_question(self) -> IdleJobResult | None:
        if self._has_active_run() or self._has_active_commitments():
            return None
        daily_cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
        if self.db.count_user_questions_since(daily_cutoff) >= MAX_QUESTIONS_PER_DAY:
            return None
        row = self.db.fetch_next_staged_item(datetime.utcnow().isoformat())
        if not row:
            return None
        if row["kind"] not in {"fact", "commitment"}:
            return None
        payload = json.loads(row["payload"])
        if row["kind"] == "fact":
            statement = payload.get("statement", "")
            question = f"Quick check: is it true that \"{statement}\"?"
        else:
            description = payload.get("description", "")
            question = f"Should I treat this as an active commitment: \"{description}\"?"
        queue_question(self.db, question=question, reason=json.dumps(payload), priority=3)
        return IdleJobResult(messages=["Idle: queued a clarifying question."], llm_calls=0)

    def _job_patch_proposal(self) -> IdleJobResult | None:
        if self._has_active_run() or self._has_active_commitments():
            return None
        daily_cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
        if self.db.count_patch_proposals_since(daily_cutoff) >= MAX_PATCHES_PER_DAY:
            return None
        prompt = _build_patch_prompt()
        response = self.llm.generate(prompt)
        proposal, error = parse_patch_proposal(response.content)
        if error or not proposal:
            return IdleJobResult(messages=["Idle: patch proposal skipped (invalid response)."], llm_calls=1)
        proposal_id = store_patch_proposal(self.db, proposal)
        # Two-agent review: use the LLM as a safety reviewer
        review_result = review_patch(self.llm, proposal)
        review_status = "proposed" if review_result.approved else "review_failed"
        self.db.update_patch_review(
            proposal_id=proposal_id,
            review_json=json.dumps({
                "approved": review_result.approved,
                "issues": review_result.issues,
                "reviewer_prompt": review_result.reviewer_prompt,
            }),
            status=review_status,
        )
        # Fire hook event for extensibility
        if self.hook_manager:
            self.hook_manager.run("on_patch_proposed", {
                "proposal_id": proposal_id,
                "proposal": {
                    "title": proposal.title,
                    "rationale": proposal.rationale,
                    "files": proposal.files,
                    "diff_text": proposal.diff_text,
                    "tests": proposal.tests,
                },
                "review_approved": review_result.approved,
                "review_issues": review_result.issues,
                "llm": self.llm,
            })
        if review_result.approved:
            return IdleJobResult(messages=["Idle: patch proposal stored and review passed."], llm_calls=2)
        issues_summary = "; ".join(review_result.issues[:3]) if review_result.issues else "no details"
        return IdleJobResult(
            messages=[f"Idle: patch proposal stored but review FAILED: {issues_summary}"],
            llm_calls=2,
        )


def _build_skill_review_prompt(skills: list[Any], episodes: list[Any]) -> str:
    return "\n".join(
        [
            "You are reviewing skill usage and failures. Return STRICT JSON only.",
            "Schema:",
            "{",
            '  "proposals": [',
            '    {"skill_name": "...", "suggestion": "...", "reason": "..."}',
            "  ]",
            "}",
            "Skills:",
            json.dumps([dict(row) for row in skills], indent=2),
            "Recent episodes:",
            json.dumps(
                [{"id": row["id"], "user_input": row["user_input"], "outcome": row["outcome"]} for row in episodes],
                indent=2,
            ),
        ]
    )


def _parse_skill_review(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    proposals = payload.get("proposals")
    if not isinstance(proposals, list):
        return None, "proposals must be list"
    return proposals, None


def _build_patch_prompt() -> str:
    return "\n".join(
        [
            "You are proposing a safe, minimal code patch. Return STRICT JSON only.",
            "Schema:",
            "{",
            '  "title": "...",',
            '  "rationale": "...",',
            '  "files": ["path1", "path2"],',
            '  "diff_text": "unified diff only",',
            '  "tests": ["pytest ..."]',
            "}",
            "Only propose hooks or safety improvements; do not modify core runtime behavior.",
        ]
    )
