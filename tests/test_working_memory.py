import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.working_memory import WorkingMemory


class WorkingMemoryTests(unittest.TestCase):
    def test_args_hash_handles_non_serializable(self) -> None:
        args = {"path": Path("/tmp"), "obj": object()}
        digest = WorkingMemory.args_hash(args)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 40)

    def test_summary_for_prompt_truncates_env_facts(self) -> None:
        memory = WorkingMemory(user_goal="ship")
        memory.environment_facts = {"big": "a" * 5000}
        summary = memory.summary_for_prompt()
        self.assertIn("...[truncated]", summary)
        self.assertNotIn("a" * 2000, summary)


if __name__ == "__main__":
    unittest.main()
