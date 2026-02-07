from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tali.embeddings import EmbeddingClient
from tali.vector_store import VectorStore


@dataclass(frozen=True)
class VectorItem:
    item_type: str
    item_id: str


class VectorIndex:
    def __init__(self, path: Path, dim: int, embedder: EmbeddingClient) -> None:
        self.path = path
        self.embedder = embedder
        self.store = VectorStore(path, dim=dim)
        self.mapping_path = path.with_suffix(".json")
        self.mapping: dict[int, VectorItem] = self._load_mapping()
        self.needs_rebuild = False

    def _load_mapping(self) -> dict[int, VectorItem]:
        if not self.mapping_path.exists():
            return {}
        payload = json.loads(self.mapping_path.read_text())
        return {int(k): VectorItem(**v) for k, v in payload.items()}

    def _save_mapping(self) -> None:
        payload = {str(k): {"item_type": v.item_type, "item_id": v.item_id} for k, v in self.mapping.items()}
        self.mapping_path.write_text(json.dumps(payload, indent=2))

    def _reset_store(self, dim: int) -> None:
        if self.path.exists():
            self.path.unlink()
        if self.mapping_path.exists():
            self.mapping_path.unlink()
        self.mapping = {}
        self.store = VectorStore(self.path, dim=dim)

    def _ensure_dim(self, vector_len: int) -> bool:
        if vector_len == self.store.dim:
            return True
        if not self.mapping:
            self._reset_store(vector_len)
            return True
        self._reset_store(vector_len)
        self.needs_rebuild = True
        return True

    def rebuild_from_items(self, items: list[tuple[str, str, str]]) -> None:
        if not items:
            self._reset_store(self.store.dim)
            self.needs_rebuild = False
            return
        first_vector = self.embedder.embed([items[0][2]])[0]
        self._reset_store(len(first_vector))
        for item_type, item_id, text in items:
            self.add(item_type=item_type, item_id=item_id, text=text)
        self.needs_rebuild = False

    def add(self, item_type: str, item_id: str, text: str) -> None:
        vector = self.embedder.embed([text])[0]
        if not self._ensure_dim(len(vector)):
            return
        next_id = max(self.mapping.keys(), default=0) + 1
        self.mapping[next_id] = VectorItem(item_type=item_type, item_id=item_id)
        self.store.add([next_id], [vector])
        self.store.save()
        self._save_mapping()

    def search(self, text: str, k: int) -> list[VectorItem]:
        if not self.mapping:
            return []
        try:
            vector = self.embedder.embed([text])[0]
        except Exception:
            return []
        if not self._ensure_dim(len(vector)):
            return []
        labels, _distances = self.store.search(vector, k=min(k, len(self.mapping)))
        return [self.mapping[label] for label in labels if label in self.mapping]
