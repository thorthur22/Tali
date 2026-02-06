from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class SourceExcerpt:
    source_ref: str
    content: str
    trusted: bool


class KnowledgeSource(Protocol):
    name: str

    def list_topics(self) -> list[str]:
        ...

    def fetch(self, topic: str) -> list[SourceExcerpt]:
        ...


@dataclass
class LocalFileSource:
    root: Path
    name: str = "local_files"
    trusted: bool = True

    def list_topics(self) -> list[str]:
        topics: list[str] = []
        for path in self.root.glob("**/*.md"):
            topics.append(str(path))
        return topics[:10]

    def fetch(self, topic: str) -> list[SourceExcerpt]:
        path = Path(topic)
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        excerpt = text[:1000]
        return [SourceExcerpt(source_ref=str(path), content=excerpt, trusted=self.trusted)]


class KnowledgeSourceRegistry:
    def __init__(self) -> None:
        self._sources: list[KnowledgeSource] = []

    def register(self, source: KnowledgeSource) -> None:
        self._sources.append(source)

    def list_sources(self) -> Iterable[KnowledgeSource]:
        return list(self._sources)
