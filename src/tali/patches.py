from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tali.db import Database


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


def run_patch_tests(tests: list[str], cwd: Path) -> str:
    results: list[str] = []
    for cmd in tests:
        try:
            completed = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
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
