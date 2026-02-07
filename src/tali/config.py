import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Paths:
    root_dir: Path
    agent_name: str

    @property
    def agent_home(self) -> Path:
        return self.root_dir / self.agent_name

    @property
    def shared_home(self) -> Path:
        return self.root_dir / "shared"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def data_dir(self) -> Path:
        return self.agent_home

    @property
    def tool_results_dir(self) -> Path:
        return self.agent_home / "tool_results"

    @property
    def db_path(self) -> Path:
        return self.agent_home / "db.sqlite"

    @property
    def vector_dir(self) -> Path:
        return self.agent_home / "vectors"

    @property
    def config_path(self) -> Path:
        return self.agent_home / "config.json"

    @property
    def logs_dir(self) -> Path:
        return self.agent_home / "logs"

    @property
    def snapshots_dir(self) -> Path:
        return self.agent_home / "snapshots"

    @property
    def sleep_dir(self) -> Path:
        return self.agent_home / "sleep"

    @property
    def runs_dir(self) -> Path:
        return self.agent_home / "runs"

    @property
    def patches_dir(self) -> Path:
        return self.agent_home / "patches"

    @property
    def code_dir(self) -> Path:
        return self.agent_home / "code"

    @property
    def inbox_dir(self) -> Path:
        return self.agent_home / "inbox"

    @property
    def outbox_dir(self) -> Path:
        return self.agent_home / "outbox"


@dataclass(frozen=True)
class RetrievalConfig:
    max_facts: int = 5
    max_episodes: int = 2
    max_preferences: int = 2
    max_commitments: int = 50
    token_budget: int = 10_000
    vector_k: int = 12
    vector_sync_min_interval_s: float = 30.0


@dataclass(frozen=True)
class GuardrailConfig:
    max_replans: int = 2
    max_critiques: int = 2


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
    python_timeout_s: float = 30.0
    max_tool_calls_per_turn: int = 30
    max_tool_seconds: float = 60.0
    max_calls_per_tool: int = 100
    tool_result_max_bytes: int = 1_000_000


@dataclass(frozen=True)
class TaskRunnerConfig:
    max_tasks_per_turn: int = 50
    max_llm_calls_per_task: int = 30
    max_tool_calls_per_task: int = 50
    max_total_llm_calls_per_run_per_turn: int = 100
    max_total_steps_per_turn: int = 100


@dataclass(frozen=True)
class AppConfig:
    agent_id: str
    agent_name: str
    created_at: str
    role_description: str
    capabilities: list[str]
    planner_llm: LLMSettings | None
    responder_llm: LLMSettings | None
    embeddings: EmbeddingSettings | None
    tools: ToolSettings | None
    task_runner: TaskRunnerConfig | None


@dataclass(frozen=True)
class SharedSettings:
    planner_llm: LLMSettings
    responder_llm: LLMSettings
    embeddings: EmbeddingSettings
    tools: ToolSettings
    task_runner: TaskRunnerConfig | None = None


def load_paths(root_dir: Path, agent_name: str) -> Paths:
    return Paths(root_dir=root_dir, agent_name=agent_name)


def load_config(path: Path) -> AppConfig:
    payload = json.loads(path.read_text())
    agent = payload.get("agent", {})
    planner_llm = payload.get("planner_llm", {})
    responder_llm = payload.get("responder_llm", {})
    embeddings = payload.get("embeddings", {})
    tools = payload.get("tools", {})
    task_runner = payload.get("task_runner", {})
    return AppConfig(
        agent_id=agent.get("agent_id", ""),
        agent_name=agent.get("agent_name", ""),
        created_at=agent.get("created_at", ""),
        role_description=str(agent.get("role_description", "")),
        capabilities=list(agent.get("capabilities", [])),
        planner_llm=LLMSettings(
            provider=planner_llm["provider"],
            model=planner_llm["model"],
            base_url=planner_llm["base_url"],
            api_key=planner_llm.get("api_key"),
        )
        if planner_llm
        else None,
        responder_llm=LLMSettings(
            provider=responder_llm["provider"],
            model=responder_llm["model"],
            base_url=responder_llm["base_url"],
            api_key=responder_llm.get("api_key"),
        )
        if responder_llm
        else None,
        embeddings=EmbeddingSettings(
            provider=embeddings["provider"],
            model=embeddings["model"],
            base_url=embeddings["base_url"],
            api_key=embeddings.get("api_key"),
            dim=int(embeddings.get("dim", 1536)),
        )
        if embeddings
        else None,
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
        )
        if tools
        else None,
        task_runner=TaskRunnerConfig(
            max_tasks_per_turn=int(task_runner.get("max_tasks_per_turn", 5)),
            max_llm_calls_per_task=int(task_runner.get("max_llm_calls_per_task", 3)),
            max_tool_calls_per_task=int(task_runner.get("max_tool_calls_per_task", 5)),
            max_total_llm_calls_per_run_per_turn=int(
                task_runner.get("max_total_llm_calls_per_run_per_turn", 10)
            ),
            max_total_steps_per_turn=int(task_runner.get("max_total_steps_per_turn", 30)),
        )
        if task_runner
        else None,
    )


def load_shared_settings(path: Path) -> SharedSettings | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    planner_llm = payload.get("planner_llm")
    responder_llm = payload.get("responder_llm")
    if not planner_llm or not responder_llm:
        return None
    embeddings = payload.get("embeddings", {})
    tools = payload.get("tools", {})
    task_runner = payload.get("task_runner", {})
    return SharedSettings(
        planner_llm=LLMSettings(
            provider=planner_llm["provider"],
            model=planner_llm["model"],
            base_url=planner_llm["base_url"],
            api_key=planner_llm.get("api_key"),
        ),
        responder_llm=LLMSettings(
            provider=responder_llm["provider"],
            model=responder_llm["model"],
            base_url=responder_llm["base_url"],
            api_key=responder_llm.get("api_key"),
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
        task_runner=TaskRunnerConfig(
            max_tasks_per_turn=int(task_runner.get("max_tasks_per_turn", 5)),
            max_llm_calls_per_task=int(task_runner.get("max_llm_calls_per_task", 3)),
            max_tool_calls_per_task=int(task_runner.get("max_tool_calls_per_task", 5)),
            max_total_llm_calls_per_run_per_turn=int(
                task_runner.get("max_total_llm_calls_per_run_per_turn", 10)
            ),
            max_total_steps_per_turn=int(task_runner.get("max_total_steps_per_turn", 30)),
        )
        if task_runner
        else None,
    )


def save_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": {
            "agent_id": config.agent_id,
            "agent_name": config.agent_name,
            "created_at": config.created_at,
            "role_description": config.role_description,
            "capabilities": config.capabilities,
        }
    }
    if config.planner_llm:
        payload["planner_llm"] = {
            "provider": config.planner_llm.provider,
            "model": config.planner_llm.model,
            "base_url": config.planner_llm.base_url,
            "api_key": config.planner_llm.api_key,
        }
    if config.responder_llm:
        payload["responder_llm"] = {
            "provider": config.responder_llm.provider,
            "model": config.responder_llm.model,
            "base_url": config.responder_llm.base_url,
            "api_key": config.responder_llm.api_key,
        }
    if config.embeddings:
        payload["embeddings"] = {
            "provider": config.embeddings.provider,
            "model": config.embeddings.model,
            "base_url": config.embeddings.base_url,
            "api_key": config.embeddings.api_key,
            "dim": config.embeddings.dim,
        }
    if config.tools:
        payload["tools"] = {
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
        }
    if config.task_runner:
        payload["task_runner"] = {
            "max_tasks_per_turn": config.task_runner.max_tasks_per_turn,
            "max_llm_calls_per_task": config.task_runner.max_llm_calls_per_task,
            "max_tool_calls_per_task": config.task_runner.max_tool_calls_per_task,
            "max_total_llm_calls_per_run_per_turn": config.task_runner.max_total_llm_calls_per_run_per_turn,
            "max_total_steps_per_turn": config.task_runner.max_total_steps_per_turn,
        }
    path.write_text(json.dumps(payload, indent=2))


def save_shared_settings(path: Path, settings: SharedSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner_llm": {
            "provider": settings.planner_llm.provider,
            "model": settings.planner_llm.model,
            "base_url": settings.planner_llm.base_url,
            "api_key": settings.planner_llm.api_key,
        },
        "responder_llm": {
            "provider": settings.responder_llm.provider,
            "model": settings.responder_llm.model,
            "base_url": settings.responder_llm.base_url,
            "api_key": settings.responder_llm.api_key,
        },
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model,
            "base_url": settings.embeddings.base_url,
            "api_key": settings.embeddings.api_key,
            "dim": settings.embeddings.dim,
        },
        "tools": {
            "approval_mode": settings.tools.approval_mode,
            "fs_root": settings.tools.fs_root,
            "fs_max_bytes": settings.tools.fs_max_bytes,
            "fs_allow_extensions": settings.tools.fs_allow_extensions,
            "web_max_bytes": settings.tools.web_max_bytes,
            "web_timeout_s": settings.tools.web_timeout_s,
            "web_max_redirects": settings.tools.web_max_redirects,
            "python_enabled": settings.tools.python_enabled,
            "python_timeout_s": settings.tools.python_timeout_s,
            "max_tool_calls_per_turn": settings.tools.max_tool_calls_per_turn,
            "max_tool_seconds": settings.tools.max_tool_seconds,
            "max_calls_per_tool": settings.tools.max_calls_per_tool,
            "tool_result_max_bytes": settings.tools.tool_result_max_bytes,
        },
    }
    task_runner = settings.task_runner or TaskRunnerConfig()
    payload["task_runner"] = {
        "max_tasks_per_turn": task_runner.max_tasks_per_turn,
        "max_llm_calls_per_task": task_runner.max_llm_calls_per_task,
        "max_tool_calls_per_task": task_runner.max_tool_calls_per_task,
        "max_total_llm_calls_per_run_per_turn": task_runner.max_total_llm_calls_per_run_per_turn,
        "max_total_steps_per_turn": task_runner.max_total_steps_per_turn,
    }
    path.write_text(json.dumps(payload, indent=2))
