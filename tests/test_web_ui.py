import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.config import AppConfig, EmbeddingSettings, LLMSettings, TaskRunnerConfig, ToolSettings, load_paths, save_config
from tali.db import Database
from tali.run_logs import append_run_log
from tali.web_ui import (
    _create_agent,
    _find_agent_service_pids,
    _format_log_entry,
    _load_agent_config,
    _render_activity,
    _render_agent_config,
    _render_agent_memory,
    _render_agents,
    _save_agent_config_from_form,
    _start_agent_service,
)


class WebUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.agent_name = "alpha"
        self.paths = load_paths(self.root, self.agent_name)
        self.paths.agent_home.mkdir(parents=True, exist_ok=True)
        cfg = AppConfig(
            agent_id=str(uuid4()),
            agent_name=self.agent_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            role_description="assistant",
            capabilities=[],
            planner_llm=LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434"),
            responder_llm=LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434"),
            embeddings=EmbeddingSettings(provider="ollama", model="nomic-embed-text", base_url="http://localhost:11434"),
            tools=ToolSettings(fs_root=str(self.root)),
            task_runner=TaskRunnerConfig(),
        )
        save_config(self.paths.config_path, cfg)
        self.db = Database(self.paths.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_format_log_entry_shows_metrics_without_event_message(self) -> None:
        entry = {
            "timestamp": "2026-02-09T01:02:03",
            "run_id": "1234567890abcdef",
            "llm_calls": 8,
            "tool_calls": 3,
            "steps": 15,
            "user_input": "resume this run",
        }
        line = _format_log_entry(entry)
        self.assertIn("run=12345678", line)
        self.assertIn("llm=8", line)
        self.assertIn("tools=3", line)
        self.assertIn("steps=15", line)

    def test_activity_and_memory_render_commitments_and_logs(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.insert_commitment(
            commitment_id=str(uuid4()),
            description="Finish release checklist",
            status="pending",
            priority=2,
            due_date="2026-02-10",
            created_at=now,
            last_touched=now,
            source_ref="test",
        )
        append_run_log(
            self.paths.logs_dir,
            {
                "timestamp": now,
                "run_id": "abcdef123456",
                "llm_calls": 1,
                "tool_calls": 2,
                "steps": 3,
                "user_input": "resume",
            },
        )

        memory_html = _render_agent_memory(self.root, self.agent_name, "", "")
        activity_html = _render_activity(self.root, "", "")

        self.assertIn("Finish release checklist", memory_html)
        self.assertIn("run=abcdef12", activity_html)
        self.assertIn("llm=1", activity_html)

    @patch("tali.web_ui._is_agent_service_pid", return_value=True)
    @patch("tali.web_ui._find_agent_service_pids", return_value=[4242])
    @patch("tali.web_ui._service_pid", return_value=None)
    @patch("tali.cli._start_agent_service_process", return_value="")
    @patch("tali.web_ui._run_cli_args", return_value="(no output)")
    def test_start_service_falls_back_to_pid_message(
        self,
        _mock_run: object,
        _mock_direct_start: object,
        _mock_service_pid: object,
        _mock_find: object,
        _mock_is_pid: object,
    ) -> None:
        result = _start_agent_service(self.root, self.agent_name)
        self.assertIn("Started agent 'alpha' (pid=4242).", result)

    @patch("tali.web_ui._is_agent_service_pid", return_value=False)
    @patch("tali.web_ui._find_agent_service_pids", return_value=[])
    @patch("tali.web_ui._service_pid", return_value=None)
    @patch("tali.cli._start_agent_service_process", return_value="")
    @patch("tali.web_ui._run_cli_args", return_value="(no output)")
    def test_start_service_no_output_reports_log_path(
        self,
        _mock_run: object,
        _mock_direct_start: object,
        _mock_service_pid: object,
        _mock_find: object,
        _mock_is_pid: object,
    ) -> None:
        result = _start_agent_service(self.root, self.agent_name)
        self.assertIn("No CLI output while starting 'alpha'.", result)
        self.assertIn("agent_service.log", result)

    @patch("tali.web_ui._is_pid_running", return_value=True)
    @patch("tali.web_ui.subprocess.run")
    def test_find_agent_service_pids_matches_env_tag(self, mock_run: object, _mock_running: object) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = (
            "211 python tali TALI_AGENT_SERVICE=1 TALI_AGENT_NAME=alpha\\n"
            "212 python tali TALI_AGENT_SERVICE=1 TALI_AGENT_NAME=beta\\n"
        )
        pids = _find_agent_service_pids("alpha")
        self.assertEqual(pids, [211])

    @patch("tali.web_ui._is_pid_running", return_value=True)
    @patch("tali.web_ui.subprocess.run")
    def test_find_agent_service_pids_matches_marker_flag(self, mock_run: object, _mock_running: object) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = (
            "311 python -c runner --tali-service-agent=alpha\\n"
            "312 python -c runner --tali-service-agent=beta\\n"
        )
        pids = _find_agent_service_pids("alpha")
        self.assertEqual(pids, [311])

    @patch("tali.web_ui._repo_root", return_value=None)
    def test_create_agent_persists_provider_and_models(self, _mock_repo_root: object) -> None:
        out = _create_agent(
            self.root,
            "beta",
            "assistant",
            "openai",
            "openai",
            "openai",
            "gpt-4.1-mini",
            "gpt-4.1-mini",
            "text-embedding-3-small",
            "https://api.openai.com/v1",
            str(self.root),
            "prompt",
        )
        self.assertEqual(out, "Agent created.")

        cfg = _load_agent_config(self.root, "beta")
        self.assertEqual(cfg.planner_llm.provider, "openai")
        self.assertEqual(cfg.responder_llm.provider, "openai")
        self.assertEqual(cfg.embeddings.provider, "openai")
        self.assertEqual(cfg.planner_llm.model, "gpt-4.1-mini")
        self.assertEqual(cfg.embeddings.model, "text-embedding-3-small")

    def test_save_agent_config_supports_other_model_fields(self) -> None:
        out = _save_agent_config_from_form(
            self.root,
            self.agent_name,
            {
                "planner_provider": "openai",
                "planner_model": "__other__",
                "planner_model_other": "gpt-5-custom",
                "planner_base_url": "https://api.openai.com/v1",
                "responder_provider": "ollama",
                "responder_model": "llama3.1",
                "responder_base_url": "http://localhost:11434",
                "embedding_provider": "openai",
                "embedding_model": "__other__",
                "embedding_model_other": "text-embedding-3-large",
                "embedding_base_url": "https://api.openai.com/v1",
            },
        )
        self.assertEqual(out, "Agent config updated.")

        cfg = _load_agent_config(self.root, self.agent_name)
        self.assertEqual(cfg.planner_llm.provider, "openai")
        self.assertEqual(cfg.planner_llm.model, "gpt-5-custom")
        self.assertEqual(cfg.responder_llm.provider, "ollama")
        self.assertEqual(cfg.responder_llm.model, "llama3.1")
        self.assertEqual(cfg.embeddings.provider, "openai")
        self.assertEqual(cfg.embeddings.model, "text-embedding-3-large")

    def test_rendered_forms_include_provider_and_model_dropdowns(self) -> None:
        agents_html = _render_agents(self.root, "", "")
        config_html = _render_agent_config(self.root, self.agent_name, "", "")

        self.assertIn("select name='planner_provider'", agents_html)
        self.assertIn("select name='planner_model'", agents_html)
        self.assertIn("option value='__other__'>Other...</option>", agents_html)
        self.assertIn("select name='planner_provider'", config_html)
        self.assertIn("select name='embedding_model'", config_html)


if __name__ == "__main__":
    unittest.main()
