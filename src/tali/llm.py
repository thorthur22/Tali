from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMResponse:
        ...


@dataclass(frozen=True)
class OpenAIClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0

    def generate(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, model=self.model)


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 60.0

    def generate(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["message"]["content"]
        return LLMResponse(content=content, model=self.model)
