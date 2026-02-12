import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tali.llm import OllamaClient


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"message": {"content": self._content}}


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def post(self, *_args, **_kwargs):
        if self.calls >= len(self.outcomes):
            raise AssertionError("post called more times than expected")
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class LLMTests(unittest.TestCase):
    def test_ollama_retries_timeout_and_succeeds(self) -> None:
        client = _FakeClient(
            [
                httpx.ReadTimeout("timed out"),
                httpx.ReadTimeout("timed out"),
                _FakeResponse("ok"),
            ]
        )
        ollama = OllamaClient(
            base_url="http://localhost:11434",
            model="llama3",
            timeout_s=5.0,
            max_retries=2,
            retry_delay_s=0.0,
        )
        with patch("tali.llm._shared_http_client", return_value=client):
            response = ollama.generate("hello")
        self.assertEqual(response.content, "ok")
        self.assertEqual(client.calls, 3)

    def test_ollama_timeout_raises_after_retries(self) -> None:
        client = _FakeClient(
            [
                httpx.ReadTimeout("timed out"),
                httpx.ReadTimeout("timed out"),
                httpx.ReadTimeout("timed out"),
            ]
        )
        ollama = OllamaClient(
            base_url="http://localhost:11434",
            model="llama3",
            timeout_s=5.0,
            max_retries=2,
            retry_delay_s=0.0,
        )
        with patch("tali.llm._shared_http_client", return_value=client):
            with self.assertRaises(httpx.ReadTimeout):
                ollama.generate("hello")
        self.assertEqual(client.calls, 3)


if __name__ == "__main__":
    unittest.main()
