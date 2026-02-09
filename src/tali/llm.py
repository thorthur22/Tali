from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
import time

import httpx


@lru_cache(maxsize=1)
def _shared_http_client() -> httpx.Client:
    return httpx.Client()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str


class LLMClient(Protocol):
    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        ...


@dataclass(frozen=True)
class OpenAIClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    default_temperature: float = 0.2

    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        client = _shared_http_client()
        response = client.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, model=self.model)


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 300.0
    default_temperature: float = 0.2
    max_retries: int = 2
    retry_delay_s: float = 1.0

    def generate(self, prompt: str, *, temperature: float | None = None) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        temp = temperature if temperature is not None else self.default_temperature
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temp},
        }
        client = _shared_http_client()
        attempts = max(0, self.max_retries) + 1
        last_timeout: httpx.TimeoutException | None = None
        for attempt in range(attempts):
            try:
                response = client.post(url, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()
                content = data["message"]["content"]
                return LLMResponse(content=content, model=self.model)
            except httpx.TimeoutException as exc:
                last_timeout = exc
                if attempt >= attempts - 1:
                    raise
                if self.retry_delay_s > 0:
                    time.sleep(self.retry_delay_s * (attempt + 1))
        if last_timeout is not None:
            raise last_timeout
        raise RuntimeError("Ollama request failed without response")
