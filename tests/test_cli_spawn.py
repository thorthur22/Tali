import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali import cli
from tali.config import TaskRunnerConfig


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

    def test_relaxed_task_runner_settings_floor_overrides_low_config(self) -> None:
        low = TaskRunnerConfig(
            max_tasks_per_turn=1,
            max_llm_calls_per_task=1,
            max_tool_calls_per_task=1,
            max_total_llm_calls_per_run_per_turn=1,
            max_total_steps_per_turn=1,
        )
        settings = cli._build_relaxed_task_runner_settings(low)
        self.assertGreaterEqual(settings.max_tasks_per_turn, 200)
        self.assertGreaterEqual(settings.max_llm_calls_per_task, 120)
        self.assertGreaterEqual(settings.max_tool_calls_per_task, 200)
        self.assertGreaterEqual(settings.max_total_llm_calls_per_run_per_turn, 2000)
        self.assertGreaterEqual(settings.max_total_steps_per_turn, 2000)


if __name__ == "__main__":
    unittest.main()
