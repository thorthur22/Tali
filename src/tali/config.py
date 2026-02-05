import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Paths:
    base_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tali.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"


@dataclass(frozen=True)
class RetrievalConfig:
    max_facts: int = 5
    max_episodes: int = 2
    max_preferences: int = 2
    max_commitments: int = 50
    token_budget: int = 1500


@dataclass(frozen=True)
class GuardrailConfig:
    max_replans: int = 1
    max_critiques: int = 1


LLMProvider = Literal["openai", "ollama"]


@dataclass(frozen=True)
class LLMSettings:
    provider: LLMProvider
    model: str
    base_url: str
    api_key: str | None = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: LLMProvider
    model: str
    base_url: str
    api_key: str | None = None
    dim: int = 1536


@dataclass(frozen=True)
class AppConfig:
    llm: LLMSettings
    embeddings: EmbeddingSettings


def load_paths(base_dir: Path | None = None) -> Paths:
    resolved = base_dir or Path.cwd()
    return Paths(base_dir=resolved)


def load_config(path: Path) -> AppConfig:
    payload = json.loads(path.read_text())
    llm = payload.get("llm", {})
    embeddings = payload.get("embeddings", {})
    return AppConfig(
        llm=LLMSettings(
            provider=llm["provider"],
            model=llm["model"],
            base_url=llm["base_url"],
            api_key=llm.get("api_key"),
        ),
        embeddings=EmbeddingSettings(
            provider=embeddings["provider"],
            model=embeddings["model"],
            base_url=embeddings["base_url"],
            api_key=embeddings.get("api_key"),
            dim=int(embeddings.get("dim", 1536)),
        ),
    )


def save_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key": config.llm.api_key,
        },
        "embeddings": {
            "provider": config.embeddings.provider,
            "model": config.embeddings.model,
            "base_url": config.embeddings.base_url,
            "api_key": config.embeddings.api_key,
            "dim": config.embeddings.dim,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
