from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hnswlib


@dataclass
class VectorStore:
    path: Path
    dim: int
    space: str = "cosine"

    def __post_init__(self) -> None:
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        if self.path.exists():
            self.index.load_index(str(self.path))
        else:
            self.index.init_index(max_elements=10000, ef_construction=200, M=16)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(self.path))

    def add(self, ids: list[int], vectors: list[list[float]]) -> None:
        self.index.add_items(vectors, ids)

    def search(self, vector: list[float], k: int) -> tuple[list[int], list[float]]:
        labels, distances = self.index.knn_query([vector], k=k)
        return labels[0].tolist(), distances[0].tolist()
