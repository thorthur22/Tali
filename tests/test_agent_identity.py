import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.agent_identity import resolve_agent, validate_agent_name, write_last_agent


class AgentIdentityTests(unittest.TestCase):
    def test_validate_agent_name(self) -> None:
        self.assertTrue(validate_agent_name("tali-main"))
        self.assertTrue(validate_agent_name("ops1"))
        self.assertFalse(validate_agent_name("BadName"))
        self.assertFalse(validate_agent_name("a"))
        self.assertFalse(validate_agent_name("name with spaces"))

    def test_unique_name_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "shared").mkdir(parents=True, exist_ok=True)
            (root / "alpha").mkdir(parents=True, exist_ok=True)

            prompts = iter(["alpha", "beta"])

            def prompt_fn(msg: str) -> str:
                return next(prompts)

            _, agent_name, _ = resolve_agent(prompt_fn=prompt_fn, root_dir=root, allow_create_config=False)
            self.assertEqual(agent_name, "beta")

    def test_last_agent_selected_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "shared").mkdir(parents=True, exist_ok=True)
            (root / "alpha").mkdir(parents=True, exist_ok=True)
            (root / "beta").mkdir(parents=True, exist_ok=True)
            (root / "alpha" / "config.json").write_text("{}")
            (root / "beta" / "config.json").write_text("{}")
            write_last_agent(root, "beta")

            called = {"count": 0}

            def prompt_fn(msg: str) -> str:
                called["count"] += 1
                return "alpha"

            _, agent_name, _ = resolve_agent(prompt_fn=prompt_fn, root_dir=root, allow_create_config=False)
            self.assertEqual(agent_name, "beta")
            self.assertEqual(called["count"], 0)


if __name__ == "__main__":
    unittest.main()
