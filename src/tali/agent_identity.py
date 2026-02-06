from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from tali.config import (
    AppConfig,
    LLMSettings,
    EmbeddingSettings,
    ToolSettings,
    load_config,
    save_config,
)


AGENT_NAME_PATTERN = re.compile(r"^[a-z0-9-_]{2,32}$")


@dataclass(frozen=True)
class AgentResolution:
    paths: object
    config: AppConfig | None


def validate_agent_name(name: str) -> bool:
    return bool(AGENT_NAME_PATTERN.match(name))


def resolve_agent(
    prompt_fn: Callable[[str], str],
    root_dir: Path | None = None,
    allow_create_config: bool = False,
) -> tuple[Path, str, AppConfig | None]:
    root = root_dir or (Path.home() / ".tali")
    root.mkdir(parents=True, exist_ok=True)
    shared_dir = root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    agent_dirs = _discover_agent_dirs(root)
    legacy_items = _detect_legacy_items(root)

    if legacy_items and not agent_dirs:
        agent_name = _prompt_unique_name(prompt_fn, root)
        agent_home = root / agent_name
        agent_home.mkdir(parents=True, exist_ok=True)
        _migrate_legacy(root, agent_home)
        _migrate_agent_db(agent_home)
        config = _ensure_agent_config(agent_home, agent_name) if allow_create_config else _load_agent_config(agent_home)
        return root, agent_name, config

    if agent_dirs:
        if len(agent_dirs) == 1:
            agent_home = agent_dirs[0]
            _migrate_agent_db(agent_home)
            config_path = agent_home / "config.json"
            config = load_config(config_path) if config_path.exists() else None
            return root, agent_home.name, config
        agent_name = prompt_fn("Choose an agent name")
        while not validate_agent_name(agent_name) or not (root / agent_name).exists():
            agent_name = prompt_fn("Choose an existing agent name")
        config_path = root / agent_name / "config.json"
        _migrate_agent_db(root / agent_name)
        config = load_config(config_path) if config_path.exists() else None
        return root, agent_name, config

    agent_name = _prompt_unique_name(prompt_fn, root)
    agent_home = root / agent_name
    agent_home.mkdir(parents=True, exist_ok=True)
    _migrate_agent_db(agent_home)
    config = _ensure_agent_config(agent_home, agent_name) if allow_create_config else _load_agent_config(agent_home)
    return root, agent_name, config


def _discover_agent_dirs(root: Path) -> list[Path]:
    agent_dirs: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name == "shared":
            continue
        if (path / "config.json").exists():
            agent_dirs.append(path)
    return agent_dirs


def _detect_legacy_items(root: Path) -> list[Path]:
    legacy_names = {"config.json", "tali.db", "vectors", "sleep", "snapshots", "tool_results"}
    items = []
    for path in root.iterdir():
        if path.name in {"shared"}:
            continue
        if path.name in legacy_names:
            items.append(path)
    return items


def _prompt_unique_name(prompt_fn: Callable[[str], str], root: Path) -> str:
    agent_name = prompt_fn("Choose a name for this agent:")
    while True:
        if not validate_agent_name(agent_name):
            agent_name = prompt_fn("Name must be lowercase [a-z0-9-_], 2-32 chars. Try again")
            continue
        if (root / agent_name).exists():
            agent_name = prompt_fn("That name exists. Choose another")
            continue
        return agent_name


def _migrate_legacy(root: Path, agent_home: Path) -> None:
    agent_home.mkdir(parents=True, exist_ok=True)
    for path in list(root.iterdir()):
        if path.name == "shared":
            continue
        if path == agent_home:
            continue
        target = agent_home / path.name
        if path.is_dir():
            shutil.move(str(path), str(target))
        else:
            shutil.move(str(path), str(target))
    legacy_db = agent_home / "tali.db"
    if legacy_db.exists():
        legacy_db.rename(agent_home / "db.sqlite")


def _migrate_agent_db(agent_home: Path) -> None:
    legacy_db = agent_home / "tali.db"
    new_db = agent_home / "db.sqlite"
    if legacy_db.exists() and not new_db.exists():
        legacy_db.rename(new_db)


def _ensure_agent_config(agent_home: Path, agent_name: str) -> AppConfig:
    config_path = agent_home / "config.json"
    if config_path.exists():
        config = load_config(config_path)
        if not config.agent_name or config.agent_name != agent_name:
            updated = AppConfig(
                agent_id=config.agent_id or str(uuid.uuid4()),
                agent_name=agent_name,
                created_at=config.created_at or datetime.utcnow().isoformat(),
                capabilities=config.capabilities or [],
                llm=config.llm,
                embeddings=config.embeddings,
                tools=config.tools,
                task_runner=config.task_runner,
            )
            save_config(config_path, updated)
            return updated
        return config
    config = AppConfig(
        agent_id=str(uuid.uuid4()),
        agent_name=agent_name,
        created_at=datetime.utcnow().isoformat(),
        capabilities=[],
        llm=LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434"),
        embeddings=EmbeddingSettings(provider="ollama", model="nomic-embed-text", base_url="http://localhost:11434"),
        tools=ToolSettings(),
        task_runner=None,
    )
    save_config(config_path, config)
    return config


def _load_agent_config(agent_home: Path) -> AppConfig | None:
    config_path = agent_home / "config.json"
    if not config_path.exists():
        return None
    return load_config(config_path)
