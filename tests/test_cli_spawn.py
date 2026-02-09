import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali import cli


class CliSpawnTests(unittest.TestCase):
    def test_should_not_spawn_when_marked_spawned(self) -> None:
        with patch.dict(cli.os.environ, {"TALI_AGENT_SPAWNED": "1"}, clear=False):
            self.assertFalse(cli._should_spawn_agent_terminal())

    def test_darwin_spawn_injects_spawn_guard_env(self) -> None:
        with (
            patch("tali.cli.platform.system", return_value="Darwin"),
            patch("tali.cli.subprocess.Popen") as popen,
        ):
            spawned = cli._spawn_agent_terminal("agent-a", Path("/tmp/repo"))
        self.assertTrue(spawned)
        args = popen.call_args.args[0]
        self.assertEqual(args[0], "osascript")
        self.assertIn("TALI_AGENT_SPAWNED=1 tali agent chat", args[2])

    def test_linux_spawn_injects_spawn_guard_env(self) -> None:
        def _which(name: str) -> str | None:
            if name == "gnome-terminal":
                return "/usr/bin/gnome-terminal"
            return None

        with (
            patch("tali.cli.platform.system", return_value="Linux"),
            patch("tali.cli.shutil.which", side_effect=_which),
            patch("tali.cli.subprocess.Popen") as popen,
        ):
            spawned = cli._spawn_agent_terminal("agent-a", Path("/tmp/repo"))
        self.assertTrue(spawned)
        args = popen.call_args.args[0]
        self.assertEqual(args[0], "gnome-terminal")
        self.assertIn("TALI_AGENT_SPAWNED=1 tali agent chat", args[-1])


if __name__ == "__main__":
    unittest.main()
