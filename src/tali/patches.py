from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tali.db import Database
from tali.llm import LLMClient


@dataclass(frozen=True)
class PatchReviewResult:
    approved: bool
    issues: list[str]
    reviewer_prompt: str


@dataclass(frozen=True)
class PatchProposal:
    title: str
    rationale: str
    files: list[str]
    diff_text: str
    tests: list[str]


def parse_patch_proposal(text: str) -> tuple[PatchProposal | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "payload must be object"
    title = payload.get("title")
    rationale = payload.get("rationale", "")
    files = payload.get("files", [])
    diff_text = payload.get("diff_text")
    tests = payload.get("tests", [])
    if not isinstance(title, str) or not title.strip():
        return None, "title required"
    if not isinstance(rationale, str):
        return None, "rationale must be string"
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        return None, "files must be list"
    if not isinstance(diff_text, str) or not diff_text.strip():
        return None, "diff_text required"
    if not isinstance(tests, list) or any(not isinstance(item, str) for item in tests):
        return None, "tests must be list"
    return (
        PatchProposal(
            title=title.strip(),
            rationale=rationale.strip(),
            files=files,
            diff_text=diff_text,
            tests=tests,
        ),
        None,
    )


def store_patch_proposal(db: Database, proposal: PatchProposal) -> str:
    proposal_id = str(uuid.uuid4())
    test_payload = json.dumps({"tests": proposal.tests, "results": None})
    db.insert_patch_proposal(
        proposal_id=proposal_id,
        created_at=datetime.utcnow().isoformat(),
        title=proposal.title,
        rationale=proposal.rationale,
        files_json=json.dumps(proposal.files),
        diff_text=proposal.diff_text,
        status="proposed",
        test_results=test_payload,
    )
    return proposal_id


def review_patch(llm: LLMClient, proposal: PatchProposal) -> PatchReviewResult:
    """Run a second-agent LLM review of a patch proposal for safety and correctness."""
    prompt = _build_review_prompt(proposal)
    response = llm.generate(prompt)
    return _parse_review_result(response.content, prompt)


def _build_review_prompt(proposal: PatchProposal) -> str:
    return "\n".join([
        "You are a code safety reviewer. Your ONLY job is to critique the",
        "following code diff for safety, correctness, and adherence to best practices.",
        "You must NOT rubber-stamp changes. Look for:",
        "  - Security risks (credential leaks, injection, unsafe file ops)",
        "  - Logic errors or regressions",
        "  - Violations of the invariant: the LLM must never train itself",
        "  - Modifications to core runtime behavior that could break the agent",
        "  - Unsafe system calls or shell commands",
        "",
        "Return STRICT JSON only with this schema:",
        '{"approved": true/false, "issues": ["issue 1", "issue 2"]}',
        "",
        f"Title: {proposal.title}",
        f"Rationale: {proposal.rationale}",
        f"Files affected: {', '.join(proposal.files)}",
        "",
        "Diff:",
        proposal.diff_text,
    ])


def _parse_review_result(text: str, prompt: str) -> PatchReviewResult:
    raw = text.strip()
    if not raw:
        return PatchReviewResult(approved=False, issues=["empty review response"], reviewer_prompt=prompt)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return PatchReviewResult(approved=False, issues=["invalid json in review response"], reviewer_prompt=prompt)
    if not isinstance(payload, dict):
        return PatchReviewResult(approved=False, issues=["review response must be object"], reviewer_prompt=prompt)
    approved = bool(payload.get("approved", False))
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(i) for i in issues]
    return PatchReviewResult(approved=approved, issues=issues, reviewer_prompt=prompt)


def run_patch_tests(tests: list[str], cwd: Path) -> str:
    results: list[str] = []
    base_cwd = _resolve_repo_root(cwd)
    for cmd in tests:
        cmd = _maybe_use_python_pytest(cmd)
        try:
            completed = subprocess.run(
                cmd, shell=True, cwd=base_cwd, capture_output=True, text=True
            )
            results.append(
                json.dumps(
                    {
                        "command": cmd,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    }
                )
            )
        except Exception as exc:
            results.append(json.dumps({"command": cmd, "error": str(exc)}))
    return "\n".join(results)


def _resolve_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _maybe_use_python_pytest(cmd: str) -> str:
    stripped = cmd.strip()
    if not stripped or not stripped.startswith("pytest"):
        return cmd
    if shutil.which("pytest"):
        return cmd
    suffix = stripped[len("pytest") :]
    return f'"{sys.executable}" -m pytest{suffix}'


def apply_patch(diff_text: str, cwd: Path) -> str | None:
    try:
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            input=diff_text,
            text=True,
            cwd=cwd,
            check=True,
        )
        return None
    except subprocess.CalledProcessError as exc:
        return f"git apply failed: {exc}"


def reverse_patch(diff_text: str, cwd: Path) -> str | None:
    try:
        subprocess.run(
            ["git", "apply", "-R", "--whitespace=nowarn"],
            input=diff_text,
            text=True,
            cwd=cwd,
            check=True,
        )
        return None
    except subprocess.CalledProcessError as exc:
        return f"git apply -R failed: {exc}"
