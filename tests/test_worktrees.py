import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tali.config import Paths
from tali.worktrees import ensure_agent_worktree, sync_agent_worktree


class WorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name) / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self._run_git(["init", "-b", "main"], self.repo_dir)
        self._run_git(["config", "user.email", "test@example.com"], self.repo_dir)
        self._run_git(["config", "user.name", "Test User"], self.repo_dir)
        (self.repo_dir / "README.md").write_text("hello\n", encoding="utf-8")
        self._run_git(["add", "README.md"], self.repo_dir)
        self._run_git(["commit", "-m", "init"], self.repo_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_git(self, args: list[str], cwd: Path) -> None:
        subprocess.run(["git", "-C", str(cwd), *args], check=True)

    def test_worktree_create_and_sync(self) -> None:
        agents_root = Path(self.temp_dir.name) / "agents"
        paths = Paths(root_dir=agents_root, agent_name="alpha")
        code_dir, status = ensure_agent_worktree(paths, self.repo_dir)
        self.assertTrue(code_dir.exists())
        self.assertTrue((code_dir / ".git").exists())
        self.assertTrue(status.ok or status.conflicted)

        (self.repo_dir / "README.md").write_text("hello\nworld\n", encoding="utf-8")
        self._run_git(["add", "README.md"], self.repo_dir)
        self._run_git(["commit", "-m", "update"], self.repo_dir)
        sync_status = sync_agent_worktree(code_dir, "main")
        self.assertTrue(sync_status.ok or sync_status.conflicted)


if __name__ == "__main__":
    unittest.main()
