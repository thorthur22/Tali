import errno
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tali.a2a_registry import Registry


class A2ARegistryTests(unittest.TestCase):
    def test_load_returns_empty_on_fd_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text('{"agents":[{"agent_id":"a1"}]}', encoding="utf-8")
            registry = Registry(registry_path)
            with patch.object(Path, "read_text", side_effect=OSError(errno.EMFILE, "Too many open files")):
                payload = registry.load()
            self.assertEqual(payload, {"agents": []})

    def test_save_ignores_fd_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(Path(tmp) / "registry.json")
            with patch.object(
                Path,
                "write_text",
                side_effect=OSError(errno.EMFILE, "Too many open files"),
            ):
                registry.save({"agents": []})

    def test_save_raises_non_fd_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(Path(tmp) / "registry.json")
            with patch.object(
                Path,
                "write_text",
                side_effect=OSError(errno.EACCES, "Permission denied"),
            ):
                with self.assertRaises(OSError):
                    registry.save({"agents": []})


if __name__ == "__main__":
    unittest.main()
