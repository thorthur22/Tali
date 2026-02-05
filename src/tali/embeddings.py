from __future__ import annotations

from dataclasses import dataclass

import httpx


class EmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenAIEmbeddingClient(EmbeddingClient):
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return [item["embedding"] for item in data["data"]]


@dataclass(frozen=True)
class OllamaEmbeddingClient(EmbeddingClient):
    base_url: str
    model: str
    timeout_s: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for text in texts:
                payload = {"model": self.model, "prompt": text}
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
        return embeddings
