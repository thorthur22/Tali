import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.db import Database
from tali.patches import parse_patch_proposal, store_patch_proposal


class PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tali.db"
        self.db = Database(self.db_path)
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_and_store_patch(self) -> None:
        payload = {
            "title": "Add hook example",
            "rationale": "Improve behavior",
            "files": ["src/tali/hooks/example.py"],
            "diff_text": "--- a/README.md\n+++ b/README.md\n@@\n+Note\n",
            "tests": ["python -m unittest"],
        }
        proposal, error = parse_patch_proposal(json.dumps(payload))
        self.assertIsNone(error)
        self.assertIsNotNone(proposal)
        proposal_id = store_patch_proposal(self.db, proposal)
        row = self.db.fetch_patch_proposal(proposal_id)
        self.assertEqual(row["title"], "Add hook example")


if __name__ == "__main__":
    unittest.main()
