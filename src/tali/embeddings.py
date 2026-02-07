from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


@lru_cache(maxsize=1)
def _shared_http_client() -> httpx.Client:
    return httpx.Client()


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
        client = _shared_http_client()
        response = client.post(url, headers=headers, json=payload, timeout=self.timeout_s)
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
        if not texts:
            return []
        client = _shared_http_client()
        results: list[list[float] | None] = [None] * len(texts)

        def _embed_one(index: int, text: str) -> None:
            payload = {"model": self.model, "prompt": text}
            response = client.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            results[index] = data["embedding"]

        max_workers = min(8, len(texts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_embed_one, idx, text) for idx, text in enumerate(texts)]
            for future in as_completed(futures):
                future.result()

        if any(embedding is None for embedding in results):
            raise RuntimeError("Embedding request failed to return all results.")
        return [embedding for embedding in results if embedding is not None]
