import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Paths:
    base_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.base_dir

    @property
    def tool_results_dir(self) -> Path:
        return self.base_dir / "tool_results"

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
class ToolSettings:
    approval_mode: str = "prompt"
    fs_root: str | None = None
    fs_max_bytes: int = 1_000_000
    fs_allow_extensions: list[str] | None = None
    web_max_bytes: int = 1_000_000
    web_timeout_s: float = 20.0
    web_max_redirects: int = 5
    python_enabled: bool = False
    python_timeout_s: float = 5.0
    max_tool_calls_per_turn: int = 3
    max_tool_seconds: float = 20.0
    max_calls_per_tool: int = 2
    tool_result_max_bytes: int = 10000


@dataclass(frozen=True)
class AppConfig:
    llm: LLMSettings
    embeddings: EmbeddingSettings
    tools: ToolSettings


def load_paths(base_dir: Path | None = None) -> Paths:
    resolved = base_dir or (Path.home() / ".tali")
    return Paths(base_dir=resolved)


def load_config(path: Path) -> AppConfig:
    payload = json.loads(path.read_text())
    llm = payload.get("llm", {})
    embeddings = payload.get("embeddings", {})
    tools = payload.get("tools", {})
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
        tools=ToolSettings(
            approval_mode=str(tools.get("approval_mode", "prompt")),
            fs_root=tools.get("fs_root") or str(Path.home()),
            fs_max_bytes=int(tools.get("fs_max_bytes", 1_000_000)),
            fs_allow_extensions=tools.get("fs_allow_extensions"),
            web_max_bytes=int(tools.get("web_max_bytes", 1_000_000)),
            web_timeout_s=float(tools.get("web_timeout_s", 20.0)),
            web_max_redirects=int(tools.get("web_max_redirects", 5)),
            python_enabled=bool(tools.get("python_enabled", False)),
            python_timeout_s=float(tools.get("python_timeout_s", 5.0)),
            max_tool_calls_per_turn=int(tools.get("max_tool_calls_per_turn", 3)),
            max_tool_seconds=float(tools.get("max_tool_seconds", 20.0)),
            max_calls_per_tool=int(tools.get("max_calls_per_tool", 2)),
            tool_result_max_bytes=int(tools.get("tool_result_max_bytes", 10000)),
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
        "tools": {
            "approval_mode": config.tools.approval_mode,
            "fs_root": config.tools.fs_root,
            "fs_max_bytes": config.tools.fs_max_bytes,
            "fs_allow_extensions": config.tools.fs_allow_extensions,
            "web_max_bytes": config.tools.web_max_bytes,
            "web_timeout_s": config.tools.web_timeout_s,
            "web_max_redirects": config.tools.web_max_redirects,
            "python_enabled": config.tools.python_enabled,
            "python_timeout_s": config.tools.python_timeout_s,
            "max_tool_calls_per_turn": config.tools.max_tool_calls_per_turn,
            "max_tool_seconds": config.tools.max_tool_seconds,
            "max_calls_per_tool": config.tools.max_calls_per_tool,
            "tool_result_max_bytes": config.tools.tool_result_max_bytes,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
