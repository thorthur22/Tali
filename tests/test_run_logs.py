import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.run_logs import append_run_log, latest_metrics_for_run, read_recent_logs


class RunLogsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_append_and_read_logs(self) -> None:
        append_run_log(self.logs_dir, {"run_id": "run-1", "llm_calls": 2})
        append_run_log(self.logs_dir, {"run_id": "run-2", "llm_calls": 3})
        entries = read_recent_logs(self.logs_dir, limit=10)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["run_id"], "run-1")

    def test_latest_metrics_for_run(self) -> None:
        append_run_log(self.logs_dir, {"run_id": "run-1", "llm_calls": 1})
        append_run_log(self.logs_dir, {"run_id": "run-1", "llm_calls": 4})
        metrics = latest_metrics_for_run(self.logs_dir, "run-1")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["llm_calls"], 4)


if __name__ == "__main__":
    unittest.main()
