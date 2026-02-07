import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.config import Paths, ToolSettings
from tali.tools.policy import ToolPolicy
from tali.tools.protocol import ToolCall
from tali.tools.registry import build_default_registry


class ToolPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.paths = Paths(root_dir=root, agent_name="agent")
        self.settings = ToolSettings(fs_root=str(root))
        self.registry = build_default_registry(self.paths, self.settings)
        self.policy = ToolPolicy(self.settings, self.registry, self.paths)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_git_show_allowed(self) -> None:
        call = ToolCall(id="1", name="shell.run", args={"command": "git show HEAD --stat"})
        decision = self.policy.evaluate(call, {})
        self.assertTrue(decision.allowed)

    def test_fs_delete_requires_approval(self) -> None:
        call = ToolCall(id="2", name="fs.delete", args={"path": "test.txt"})
        decision = self.policy.evaluate(call, {})
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_approval)


if __name__ == "__main__":
    unittest.main()
