from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from tali.config import Paths


@dataclass(frozen=True)
class WorktreeStatus:
    ok: bool
    conflicted: bool
    message: str | None = None


def resolve_main_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return None


def ensure_agent_worktree(paths: Paths, main_repo: Path) -> tuple[Path, WorktreeStatus]:
    code_dir = paths.code_dir
    if not shutil.which("git"):
        return code_dir, WorktreeStatus(False, False, "Git is required to manage agent worktrees.")
    if code_dir.exists() and not (code_dir / ".git").exists():
        return (
            code_dir,
            WorktreeStatus(
                False,
                False,
                f"Code dir exists but is not a git worktree: {code_dir}",
            ),
        )
    base_ref = _select_base_ref(main_repo)
    if not code_dir.exists():
        cmd = [
            "git",
            "-C",
            str(main_repo),
            "worktree",
            "add",
            "-B",
            _agent_branch(paths.agent_name),
            str(code_dir),
            base_ref,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "git worktree add failed").strip()
            return code_dir, WorktreeStatus(False, False, message)
    return code_dir, sync_agent_worktree(code_dir, base_ref)


def sync_agent_worktree(code_dir: Path, base_ref: str) -> WorktreeStatus:
    if not shutil.which("git"):
        return WorktreeStatus(False, False, "Git is required to sync agent worktrees.")
    fetch = subprocess.run(
        ["git", "-C", str(code_dir), "fetch"],
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        message = (fetch.stderr or fetch.stdout or "git fetch failed").strip()
        return WorktreeStatus(False, False, message)
    merge = subprocess.run(
        ["git", "-C", str(code_dir), "merge", base_ref],
        capture_output=True,
        text=True,
    )
    if merge.returncode == 0:
        return WorktreeStatus(True, False, None)
    conflicts = subprocess.run(
        ["git", "-C", str(code_dir), "diff", "--name-only", "--diff-filter=U"],
        capture_output=True,
        text=True,
    )
    conflicted = bool(conflicts.stdout.strip())
    message = (merge.stderr or merge.stdout or "git merge failed").strip()
    if conflicted:
        message = (
            message
            + "\nMerge conflicts detected in agent worktree. Resolve them and rerun."
        ).strip()
    return WorktreeStatus(False, conflicted, message)


def remove_agent_worktree(paths: Paths, main_repo: Path) -> WorktreeStatus:
    code_dir = paths.code_dir
    if not code_dir.exists():
        return WorktreeStatus(True, False, None)
    if not shutil.which("git"):
        return WorktreeStatus(False, False, "Git is required to remove agent worktrees.")
    result = subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "remove", "--force", str(code_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git worktree remove failed").strip()
        return WorktreeStatus(False, False, message)
    return WorktreeStatus(True, False, None)


def _select_base_ref(repo: Path) -> str:
    if _ref_exists(repo, "refs/heads/main"):
        return "main"
    if _ref_exists(repo, "refs/remotes/origin/main"):
        return "origin/main"
    if _ref_exists(repo, "refs/remotes/origin/master"):
        return "origin/master"
    if _ref_exists(repo, "refs/heads/master"):
        return "master"
    return "HEAD"


def _ref_exists(repo: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", ref],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _agent_branch(agent_name: str) -> str:
    return f"agent/{agent_name}"
