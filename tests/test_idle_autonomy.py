import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.approvals import ApprovalManager
from tali.db import Database
from tali.idle import IdleScheduler
from tali.knowledge_sources import KnowledgeSourceRegistry


class DummyLLM:
    def generate(self, prompt):
        return type("Resp", (), {"content": "{}"})


class DummyResult:
    def __init__(self) -> None:
        self.llm_calls = 1
        self.run_id = "run-1"


class DummyTaskRunner:
    def __init__(self, approvals: ApprovalManager) -> None:
        self.tool_runner = type("ToolRunner", (), {"approvals": approvals})()
        self.seen_mode: str | None = None

    def run_turn(self, *_args, **_kwargs):
        self.seen_mode = self.tool_runner.approvals.mode
        return DummyResult()


class IdleAutonomyApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_autonomous_run_auto_approves_safe_tools(self) -> None:
        approvals = ApprovalManager(mode="prompt")
        runner = DummyTaskRunner(approvals)
        scheduler = IdleScheduler(
            data_dir=Path(self.temp_dir.name),
            db=self.db,
            llm=DummyLLM(),
            sources=KnowledgeSourceRegistry(),
            task_runner=runner,
        )
        scheduler._auto_continue_run()
        self.assertEqual(runner.seen_mode, "auto_approve_safe")
        self.assertEqual(approvals.mode, "prompt")

    def test_autonomous_run_respects_denied_mode(self) -> None:
        approvals = ApprovalManager(mode="deny")
        runner = DummyTaskRunner(approvals)
        scheduler = IdleScheduler(
            data_dir=Path(self.temp_dir.name),
            db=self.db,
            llm=DummyLLM(),
            sources=KnowledgeSourceRegistry(),
            task_runner=runner,
        )
        scheduler._auto_continue_run()
        self.assertEqual(runner.seen_mode, "deny")
        self.assertEqual(approvals.mode, "deny")


if __name__ == "__main__":
    unittest.main()
