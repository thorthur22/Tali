import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali import cli
from tali.config import TaskRunnerConfig, load_paths


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

    def test_start_service_does_not_use_agent_worktree_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = load_paths(root, "agent-a")
            paths.agent_home.mkdir(parents=True, exist_ok=True)
            paths.code_dir.mkdir(parents=True, exist_ok=True)

            class _Proc:
                pid = 12345

                @staticmethod
                def poll() -> None:
                    return None

            with (
                patch("tali.cli._find_agent_service_pids", return_value=[]),
                patch("tali.cli._read_agent_service_pid", return_value=None),
                patch("tali.cli._is_agent_service_pid", return_value=False),
                patch("tali.cli._clear_agent_service_pid"),
                patch("tali.cli._write_agent_service_pid"),
                patch("tali.cli.time.sleep"),
                patch("tali.cli.subprocess.Popen", return_value=_Proc()) as popen,
            ):
                message = cli._start_agent_service_process(paths, "agent-a")
            self.assertIn("Started agent 'agent-a' (pid=12345).", message)
            self.assertNotEqual(popen.call_args.kwargs["cwd"], str(paths.code_dir))

    def test_start_service_failure_includes_log_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = load_paths(root, "agent-a")
            paths.agent_home.mkdir(parents=True, exist_ok=True)

            class _Proc:
                pid = 12345
                returncode = 0

                @staticmethod
                def poll() -> int:
                    return 0

            with (
                patch("tali.cli._find_agent_service_pids", return_value=[]),
                patch("tali.cli._read_agent_service_pid", return_value=None),
                patch("tali.cli._is_agent_service_pid", return_value=False),
                patch("tali.cli._clear_agent_service_pid"),
                patch("tali.cli.time.sleep"),
                patch("tali.cli.subprocess.Popen", return_value=_Proc()),
                patch("tali.cli._read_log_tail", return_value="OSError: [Errno 24] Too many open files"),
            ):
                message = cli._start_agent_service_process(paths, "agent-a")
            self.assertIn("Failed to start agent 'agent-a' (exit=0).", message)
            self.assertIn("Too many open files", message)
            self.assertIn("ulimit -n", message)

    @patch("tali.cli._is_pid_running", return_value=True)
    @patch("tali.cli.subprocess.run")
    def test_find_agent_service_pids_matches_env_tagged_service(self, mock_run: object, _mock_running: object) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = (
            "111 python some.py TALI_AGENT_SERVICE=1 TALI_AGENT_NAME=agent-a\\n"
            "112 python some.py TALI_AGENT_SERVICE=1 TALI_AGENT_NAME=agent-b\\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = load_paths(Path(tmp), "agent-a")
            pids = cli._find_agent_service_pids(paths)
        self.assertEqual(pids, [111])


if __name__ == "__main__":
    unittest.main()
