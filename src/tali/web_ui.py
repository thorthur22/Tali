from __future__ import annotations

import html
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import click
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import uuid4
from typer.main import get_command

from tali.a2a_registry import Registry
from tali.config import (
    AppConfig,
    AutonomyConfig,
    EmbeddingSettings,
    LLMSettings,
    TaskRunnerConfig,
    ToolSettings,
    load_config,
    load_paths,
    save_config,
)
from tali.db import Database
from tali.patches import apply_patch, reverse_patch, run_patch_tests
from tali.run_logs import latest_metrics_for_run, read_recent_logs
from tali.worktrees import ensure_agent_worktree, remove_agent_worktree, resolve_main_repo_root


@dataclass
class AgentSummary:
    name: str
    run_status: str
    run_id: str | None
    llm_calls: int | None
    tool_calls: int | None
    steps: int | None
    git_branch: str
    git_changes: int
    git_files: list[str]
    unread_messages: int
    patch_count: int
    service_running: bool
    service_pid: int | None


@dataclass(frozen=True)
class CommandParam:
    name: str
    kind: str  # argument | option
    option: str | None
    required: bool
    is_flag: bool
    default: str


@dataclass(frozen=True)
class CommandSpec:
    path: tuple[str, ...]
    display: str
    group: str
    runnable: bool
    params: tuple[CommandParam, ...]


_COMMAND_CACHE: list[CommandSpec] | None = None
_CACHE_TTL_S = 1.5
_CACHE_LOCK = threading.Lock()
_AGENT_NAMES_CACHE: tuple[float, list[str]] = (0.0, [])
_DASHBOARD_CACHE: tuple[float, dict[str, Any]] = (0.0, {"agents": []})
_AGENT_DATA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_COMMITMENT_STATUSES = ("pending", "in_progress", "blocked", "done", "canceled")
_MODEL_OTHER_VALUE = "__other__"
_SUPPORTED_PROVIDERS = ("ollama", "openai")
_DEFAULT_LLM_MODELS = ("llama3", "llama3.1", "gpt-4.1", "gpt-4.1-mini", "gpt-4o")
_DEFAULT_EMBED_MODELS = ("nomic-embed-text", "text-embedding-3-small", "text-embedding-3-large")


def _q(value: str) -> str:
    return quote(value, safe="")


def _redirect_target(raw: str, fallback: str) -> str:
    target = (raw or "").strip()
    if target.startswith("/") and not target.startswith("//"):
        return target
    return fallback


def _bool_value(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    value = default
    text = (raw or "").strip()
    if text:
        try:
            value = int(text)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _parse_iso_due_date(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        # Accept date-only or full ISO datetime values from form input.
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def _normalize_provider(raw: str, fallback: str = "ollama") -> str:
    provider = (raw or "").strip().lower()
    if provider in _SUPPORTED_PROVIDERS:
        return provider
    fallback_provider = (fallback or "").strip().lower()
    if fallback_provider in _SUPPORTED_PROVIDERS:
        return fallback_provider
    return "ollama"


def _resolve_choice_with_other(selected: str, custom: str) -> str:
    value = (selected or "").strip()
    if value == _MODEL_OTHER_VALUE:
        return (custom or "").strip()
    return value


def _resolve_model_form_value(data: dict[str, str], field_name: str) -> str:
    return _resolve_choice_with_other(data.get(field_name, ""), data.get(f"{field_name}_other", ""))


def _dedupe_nonempty(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _model_options(include_embedding_defaults: bool = False) -> list[str]:
    defaults = list(_DEFAULT_LLM_MODELS)
    if include_embedding_defaults:
        defaults.extend(_DEFAULT_EMBED_MODELS)
    defaults.extend(_list_ollama_models())
    return _dedupe_nonempty(defaults)


def _render_provider_field(label: str, name: str, current: str) -> str:
    current_provider = _normalize_provider(current)
    options = []
    for provider in _SUPPORTED_PROVIDERS:
        marker = " selected" if provider == current_provider else ""
        options.append(f"<option value='{provider}'{marker}>{provider}</option>")
    return f"<label>{label}<select name='{name}'>{''.join(options)}</select></label>"


def _render_model_field(label: str, name: str, current: str, options: list[str]) -> str:
    clean_current = (current or "").strip()
    model_options = _dedupe_nonempty(options)
    use_other = bool(clean_current and clean_current not in model_options)
    selected_value = _MODEL_OTHER_VALUE if use_other else clean_current
    if not selected_value:
        selected_value = model_options[0] if model_options else _MODEL_OTHER_VALUE

    option_tags = []
    for option in model_options:
        marker = " selected" if option == selected_value else ""
        option_tags.append(f"<option value='{html.escape(option, quote=True)}'{marker}>{html.escape(option)}</option>")
    other_marker = " selected" if selected_value == _MODEL_OTHER_VALUE else ""
    option_tags.append(f"<option value='{_MODEL_OTHER_VALUE}'{other_marker}>Other...</option>")

    other_value = clean_current if selected_value == _MODEL_OTHER_VALUE else ""
    other_style = "" if selected_value == _MODEL_OTHER_VALUE else " style='display:none;'"
    return (
        f"<label>{label}"
        f"<select name='{name}' class='model-choice' data-other-input='{name}_other'>{''.join(option_tags)}</select>"
        f"<input name='{name}_other' placeholder='Custom model id' value='{html.escape(other_value, quote=True)}'{other_style} />"
        "</label>"
    )


def _init_db(path: Path) -> Database:
    db = Database(path)
    db.initialize()
    return db


def _invalidate_cache(agent_name: str | None = None) -> None:
    global _DASHBOARD_CACHE, _AGENT_NAMES_CACHE
    with _CACHE_LOCK:
        _DASHBOARD_CACHE = (0.0, {"agents": []})
        if agent_name:
            _AGENT_DATA_CACHE.pop(agent_name, None)
        else:
            _AGENT_DATA_CACHE.clear()
        _AGENT_NAMES_CACHE = (0.0, [])


def _list_agent_names(root: Path) -> list[str]:
    global _AGENT_NAMES_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        cached_at, cached_names = _AGENT_NAMES_CACHE
        if now - cached_at < _CACHE_TTL_S:
            return list(cached_names)
    if not root.exists():
        return []
    names: list[str] = []
    try:
        for entry in root.iterdir():
            if entry.is_dir() and entry.name != "shared" and (entry / "config.json").exists():
                names.append(entry.name)
    except OSError:
        with _CACHE_LOCK:
            return list(_AGENT_NAMES_CACHE[1])
    names_sorted = sorted(names)
    with _CACHE_LOCK:
        _AGENT_NAMES_CACHE = (now, names_sorted)
    return names_sorted


def _paths(root: Path, name: str):
    return load_paths(root, name)


def _repo_root() -> Path | None:
    return resolve_main_repo_root(Path(__file__).resolve())


def _run_git(code_dir: Path, args: list[str]) -> str:
    if not code_dir.exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(code_dir), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return (result.stdout or "").strip()


def _count_unread(db: Database) -> int:
    return len(db.list_unread_agent_messages(limit=1000))


def _count_patches(db: Database) -> int:
    return len(db.list_patch_proposals())


def _service_pid_path(root: Path, name: str) -> Path:
    return _paths(root, name).agent_home / "agent_service.pid"


def _service_pid(root: Path, name: str) -> int | None:
    path = _service_pid_path(root, name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload.get("pid"))
    except Exception:
        return None


def _is_pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    return (result.stdout or "").strip()


def _find_agent_service_pids(name: str) -> list[int]:
    expected = f"agent chat {name}"
    env_name = f"TALI_AGENT_NAME={name}"
    try:
        result = subprocess.run(
            ["ps", "eww", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return []
    pids: list[int] = []
    for raw in (result.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_raw, cmd = parts
        tagged_service = "TALI_AGENT_SERVICE=1" in cmd and env_name in cmd
        legacy_service = expected in cmd and "--service-mode" in cmd
        if not tagged_service and not legacy_service:
            continue
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if _is_pid_running(pid):
            pids.append(pid)
    return sorted(set(pids))


def _is_agent_service_pid(root: Path, name: str, pid: int | None) -> bool:
    if not pid or not _is_pid_running(pid):
        return False
    return pid in _find_agent_service_pids(name)


def _is_run_stale(run: Any) -> bool:
    updated_at = str(run["updated_at"] or "").strip() if run else ""
    if not updated_at:
        return False
    try:
        dt = datetime.fromisoformat(updated_at)
    except ValueError:
        return False
    return (datetime.utcnow() - dt) > timedelta(minutes=30)


def _agent_summary(root: Path, name: str) -> AgentSummary:
    paths = _paths(root, name)
    run = None
    metrics = None
    unread = 0
    patch_count = 0
    try:
        db = _init_db(paths.db_path)
        run = db.fetch_active_run()
        unread = _count_unread(db)
        patch_count = _count_patches(db)
    except (sqlite3.Error, OSError):
        run = None
    pid = _service_pid(root, name)
    live_pids = _find_agent_service_pids(name)
    if live_pids and (pid is None or pid not in live_pids):
        pid = live_pids[0]
    service_running = bool(live_pids)
    run_status = str(run["status"]) if run else "idle"
    if run and run_status in {"active", "blocked"} and not service_running:
        run_status = "stale" if _is_run_stale(run) else "orphaned"
    run_id = str(run["id"]) if run else None
    try:
        metrics = latest_metrics_for_run(paths.logs_dir, run_id) if run_id else None
    except OSError:
        metrics = None
    git_branch = _run_git(paths.code_dir, ["branch", "--show-current"]) or "(no worktree)"
    git_status = _run_git(paths.code_dir, ["status", "--short"])
    git_files = [line.strip() for line in git_status.splitlines() if line.strip()]
    return AgentSummary(
        name=name,
        run_status=run_status,
        run_id=run_id,
        llm_calls=(metrics or {}).get("llm_calls"),
        tool_calls=(metrics or {}).get("tool_calls"),
        steps=(metrics or {}).get("steps"),
        git_branch=git_branch,
        git_changes=len(git_files),
        git_files=git_files[:30],
        unread_messages=unread,
        patch_count=patch_count,
        service_running=service_running,
        service_pid=pid if service_running else None,
    )


def _all_summaries(root: Path) -> list[AgentSummary]:
    return [_agent_summary(root, name) for name in _list_agent_names(root)]


def _collect_dashboard_data(root: Path) -> dict[str, Any]:
    agents = _all_summaries(root)
    return {
        "agents": [
            {
                "name": a.name,
                "run_status": a.run_status,
                "run_id": a.run_id,
                "llm_calls": a.llm_calls,
                "tool_calls": a.tool_calls,
                "steps": a.steps,
                "git_branch": a.git_branch,
                "git_changes": a.git_changes,
                "unread_messages": a.unread_messages,
                "patch_count": a.patch_count,
                "service_running": a.service_running,
                "service_pid": a.service_pid,
            }
            for a in agents
        ]
    }


def _collect_dashboard_data_cached(root: Path) -> dict[str, Any]:
    global _DASHBOARD_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        cached_at, payload = _DASHBOARD_CACHE
        if now - cached_at < _CACHE_TTL_S:
            return dict(payload)
    payload = _collect_dashboard_data(root)
    with _CACHE_LOCK:
        _DASHBOARD_CACHE = (now, payload)
    return dict(payload)


def _collect_agent_data(root: Path, agent_name: str) -> dict[str, Any]:
    paths = _paths(root, agent_name)
    run = None
    tasks: list[dict[str, Any]] = []
    try:
        db = _init_db(paths.db_path)
        run = db.fetch_active_run()
        tasks = db.fetch_tasks_for_run(str(run["id"])) if run else []
    except (sqlite3.Error, OSError):
        run = None
        tasks = []
    a = _agent_summary(root, agent_name)
    return {
        "name": a.name,
        "run_status": a.run_status,
        "run_id": a.run_id,
        "llm_calls": a.llm_calls,
        "tool_calls": a.tool_calls,
        "steps": a.steps,
        "git_branch": a.git_branch,
        "git_changes": a.git_changes,
        "git_files": a.git_files,
        "unread_messages": a.unread_messages,
        "patch_count": a.patch_count,
        "service_running": a.service_running,
        "service_pid": a.service_pid,
        "tasks": [
            {"ordinal": int(r["ordinal"]) + 1, "title": str(r["title"]), "status": str(r["status"])}
            for r in tasks
        ],
    }


def _collect_agent_data_cached(root: Path, agent_name: str) -> dict[str, Any]:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _AGENT_DATA_CACHE.get(agent_name)
        if cached and now - cached[0] < _CACHE_TTL_S:
            return dict(cached[1])
    payload = _collect_agent_data(root, agent_name)
    with _CACHE_LOCK:
        _AGENT_DATA_CACHE[agent_name] = (now, payload)
    return dict(payload)


def _run_cli_args(args: list[str], agent_context: str | None = None, timeout: int = 900) -> str:
    env = os.environ.copy()
    env["TALI_HEADLESS"] = "1"
    if agent_context:
        env["TALI_AGENT_NAME"] = agent_context
    cmd = [sys.executable, "-m", "tali.cli", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except Exception as exc:
        return f"Failed to run command: {exc}"
    out = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    text = out.strip()
    if text:
        if result.returncode != 0:
            return f"[exit {result.returncode}]\n{text}"
        return text
    if result.returncode != 0:
        return f"Command exited with code {result.returncode} and no output."
    return "(no output)"


def _split_notice(text: str) -> tuple[str, str]:
    clean = (text or "").strip()
    if not clean:
        return "No output.", ""
    head, _, tail = clean.partition("\n")
    return head.strip() or clean, tail.strip()


def _load_agent_config(root: Path, agent_name: str) -> AppConfig:
    return load_config(_paths(root, agent_name).config_path)


def _save_agent_config_from_form(root: Path, agent_name: str, data: dict[str, str]) -> str:
    try:
        config = _load_agent_config(root, agent_name)
    except Exception as exc:
        return f"Unable to load config: {exc}"

    planner = config.planner_llm or LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434")
    responder = config.responder_llm or LLMSettings(
        provider=planner.provider, model=planner.model, base_url=planner.base_url
    )
    embeddings = config.embeddings or EmbeddingSettings(
        provider=planner.provider, model="nomic-embed-text", base_url=planner.base_url, dim=1536
    )
    tools = config.tools or ToolSettings()
    task_runner = config.task_runner or TaskRunnerConfig()
    autonomy = config.autonomy or AutonomyConfig()

    role = data.get("role_description", "").strip()
    planner_provider = _normalize_provider(data.get("planner_provider", ""), planner.provider)
    planner_model = _resolve_model_form_value(data, "planner_model")
    planner_base_url = data.get("planner_base_url", "").strip()
    responder_provider = _normalize_provider(data.get("responder_provider", ""), responder.provider)
    responder_model = _resolve_model_form_value(data, "responder_model")
    responder_base_url = data.get("responder_base_url", "").strip()
    embedding_provider = _normalize_provider(data.get("embedding_provider", ""), embeddings.provider)
    embedding_model = _resolve_model_form_value(data, "embedding_model")
    embedding_base_url = data.get("embedding_base_url", "").strip()
    fs_root = data.get("fs_root", "").strip()
    approval_mode = data.get("approval_mode", "").strip() or tools.approval_mode
    if approval_mode not in {"prompt", "auto_approve_safe", "deny"}:
        approval_mode = tools.approval_mode

    planner = replace(
        planner,
        provider=planner_provider,
        model=planner_model or planner.model,
        base_url=planner_base_url or planner.base_url,
    )
    responder = replace(
        responder,
        provider=responder_provider,
        model=responder_model or responder.model,
        base_url=responder_base_url or responder.base_url,
    )
    embeddings = replace(
        embeddings,
        provider=embedding_provider,
        model=embedding_model or embeddings.model,
        base_url=embedding_base_url or embeddings.base_url,
    )
    tools = replace(
        tools,
        approval_mode=approval_mode,
        fs_root=fs_root or tools.fs_root or str(Path.home()),
        python_enabled=_bool_value(data.get("python_enabled", "")),
        max_tool_calls_per_turn=_parse_int(
            data.get("max_tool_calls_per_turn", ""),
            tools.max_tool_calls_per_turn,
            minimum=1,
            maximum=500,
        ),
    )
    task_runner = replace(
        task_runner,
        max_tasks_per_turn=_parse_int(data.get("max_tasks_per_turn", ""), task_runner.max_tasks_per_turn, 1, 500),
        max_total_steps_per_turn=_parse_int(
            data.get("max_total_steps_per_turn", ""),
            task_runner.max_total_steps_per_turn,
            1,
            5000,
        ),
    )
    autonomy = replace(
        autonomy,
        enabled=_bool_value(data.get("autonomy_enabled", "")),
        auto_continue=_bool_value(data.get("auto_continue", "")),
        execute_commitments=_bool_value(data.get("execute_commitments", "")),
        idle_trigger_seconds=_parse_int(
            data.get("idle_trigger_seconds", ""),
            autonomy.idle_trigger_seconds,
            minimum=5,
            maximum=86_400,
        ),
    )
    updated = replace(
        config,
        role_description=role or config.role_description,
        planner_llm=planner,
        responder_llm=responder,
        embeddings=embeddings,
        tools=tools,
        task_runner=task_runner,
        autonomy=autonomy,
    )
    save_config(_paths(root, agent_name).config_path, updated)
    return "Agent config updated."


def _format_log_entry(entry: dict[str, Any]) -> str:
    timestamp = str(entry.get("timestamp") or entry.get("created_at") or "?")
    event = str(entry.get("event") or "").strip()
    message = str(entry.get("message") or "").strip()
    if event or message:
        return f"[{timestamp}] {event} {message}".strip()

    parts: list[str] = []
    run_id = str(entry.get("run_id") or "").strip()
    if run_id:
        parts.append(f"run={run_id[:8]}")
    if "llm_calls" in entry:
        parts.append(f"llm={entry.get('llm_calls')}")
    if "tool_calls" in entry:
        parts.append(f"tools={entry.get('tool_calls')}")
    if "steps" in entry:
        parts.append(f"steps={entry.get('steps')}")
    user_input = str(entry.get("user_input") or "").strip()
    if user_input:
        parts.append(f"input={user_input[:120]}")
    if parts:
        return f"[{timestamp}] {' | '.join(parts)}"
    return f"[{timestamp}] {json.dumps(entry, ensure_ascii=True, sort_keys=True)}"


def _add_commitment(root: Path, agent_name: str, description: str, priority: str, due_date: str) -> str:
    text = description.strip()
    if not text:
        return "Commitment description is required."
    due = _parse_iso_due_date(due_date)
    if due_date.strip() and not due:
        return "Invalid due date. Use YYYY-MM-DD or ISO datetime."
    db = _init_db(_paths(root, agent_name).db_path)
    now = datetime.utcnow().isoformat()
    db.insert_commitment(
        commitment_id=str(uuid4()),
        description=text,
        status="pending",
        priority=_parse_int(priority, default=3, minimum=1, maximum=5),
        due_date=due,
        created_at=now,
        last_touched=now,
        source_ref="web_ui",
    )
    return "Commitment added."


def _update_commitment(root: Path, agent_name: str, commitment_id: str, status: str, priority: str, due_date: str, description: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    row = db.fetch_commitment(commitment_id)
    if not row:
        return "Commitment not found."
    next_status = status.strip() or str(row["status"])
    if next_status not in _COMMITMENT_STATUSES:
        return "Invalid commitment status."
    due = _parse_iso_due_date(due_date)
    if due_date.strip() and not due:
        return "Invalid due date. Use YYYY-MM-DD or ISO datetime."
    next_description = description.strip() or str(row["description"])
    db.update_commitment(
        commitment_id=commitment_id,
        description=next_description,
        status=next_status,
        priority=_parse_int(priority, int(row["priority"] or 3), minimum=1, maximum=5),
        due_date=due,
        last_touched=datetime.utcnow().isoformat(),
    )
    return "Commitment updated."


def _start_all_services(root: Path) -> tuple[str, str]:
    names = _list_agent_names(root)
    if not names:
        return "No agents found.", ""
    lines = [f"{name}: {_start_agent_service(root, name)}" for name in names]
    return "Swarm start requested.", "\n".join(lines)


def _stop_all_services(root: Path) -> tuple[str, str]:
    names = _list_agent_names(root)
    if not names:
        return "No agents found.", ""
    lines = [f"{name}: {_stop_agent_service(root, name)}" for name in names]
    return "Swarm stop requested.", "\n".join(lines)


def _run_swarm_prompt(agent_name: str, prompt: str) -> tuple[str, str]:
    text = prompt.strip()
    if not text:
        return "Swarm prompt is required.", ""
    output = _run_cli_args(["agent", "swarm", text], agent_context=agent_name, timeout=1800)
    return f"Swarm prompt executed via {agent_name}.", output


def _command_specs() -> list[CommandSpec]:
    global _COMMAND_CACHE
    if _COMMAND_CACHE is not None:
        return _COMMAND_CACHE

    from tali.cli import app as cli_app

    root = get_command(cli_app)
    specs: list[CommandSpec] = []
    non_runnable = {
        ("web",),
        ("shell",),
        ("setup",),
        ("agent", "dashboard"),
    }

    def walk(cmd: click.Command, prefix: tuple[str, ...]) -> None:
        if isinstance(cmd, click.Group):
            for name, sub in sorted(cmd.commands.items(), key=lambda item: item[0]):
                walk(sub, prefix + (name,))
            return
        if not prefix:
            return
        params: list[CommandParam] = []
        for param in cmd.params:
            kind = "option" if isinstance(param, click.Option) else "argument"
            option = param.opts[0] if kind == "option" and getattr(param, "opts", None) else None
            is_flag = bool(getattr(param, "is_flag", False))
            default_value = "" if param.default is None else str(param.default)
            params.append(
                CommandParam(
                    name=param.name,
                    kind=kind,
                    option=option,
                    required=bool(getattr(param, "required", False)),
                    is_flag=is_flag,
                    default=default_value,
                )
            )
        group = prefix[0]
        specs.append(
            CommandSpec(
                path=prefix,
                display=" ".join(prefix),
                group=group,
                runnable=prefix not in non_runnable,
                params=tuple(params),
            )
        )

    walk(root, ())
    _COMMAND_CACHE = sorted(specs, key=lambda s: s.display)
    return _COMMAND_CACHE


def _run_agent_chat_turn(agent_name: str, message: str) -> str:
    return _run_cli_args(["agent", "chat", agent_name, message], agent_context=agent_name)


def _start_agent_service(root: Path, agent_name: str) -> str:
    output = _run_cli_args(["agent", "start", agent_name], agent_context=agent_name, timeout=60)
    if output != "(no output)":
        return output
    # Fallback path: run the startup routine directly so web UI isn't blocked by
    # subprocess wrapper edge-cases where stdout/stderr are both empty.
    try:
        from tali.cli import _start_agent_service_process

        direct = _start_agent_service_process(_paths(root, agent_name), agent_name)
        if direct:
            return direct
    except Exception as exc:
        return f"Direct start failed for '{agent_name}': {exc}"
    pid = _service_pid(root, agent_name)
    live_pids = _find_agent_service_pids(agent_name)
    if live_pids and (pid is None or pid not in live_pids):
        pid = live_pids[0]
    if _is_agent_service_pid(root, agent_name, pid):
        return f"Started agent '{agent_name}' (pid={pid})."
    log_path = _paths(root, agent_name).logs_dir / "agent_service.log"
    return f"No CLI output while starting '{agent_name}'. Check log: {log_path}"


def _stop_agent_service(root: Path, agent_name: str) -> str:
    output = _run_cli_args(["agent", "stop", agent_name], agent_context=agent_name, timeout=60)
    if output != "(no output)":
        return output
    try:
        from tali.cli import _stop_agent_service_process

        direct = _stop_agent_service_process(_paths(root, agent_name), agent_name)
        if direct:
            return direct
    except Exception as exc:
        return f"Direct stop failed for '{agent_name}': {exc}"
    pid = _service_pid(root, agent_name)
    if _is_agent_service_pid(root, agent_name, pid):
        return f"No CLI output while stopping '{agent_name}', but service appears to still be running (pid={pid})."
    return f"Stopped agent '{agent_name}'."


def _resume_agent_run(root: Path, agent_name: str) -> str:
    resume_out = _run_cli_args(["run", "resume"], agent_context=agent_name, timeout=90)
    service_out = _start_agent_service(root, agent_name)
    return f"{resume_out}\n{service_out}".strip()


def _cancel_agent_run(root: Path, agent_name: str) -> str:
    try:
        db = _init_db(_paths(root, agent_name).db_path)
    except (sqlite3.Error, OSError) as exc:
        return f"Unable to open agent database: {exc}"
    run = db.fetch_active_run()
    if not run:
        return "No active run to cancel."
    db.update_run_status(str(run["id"]), status="canceled", current_task_id=None, last_error=None)
    return f"Run {run['id']} canceled."


def _create_agent(
    root: Path,
    agent_name: str,
    role: str,
    planner_provider: str,
    responder_provider: str,
    embedding_provider: str,
    planner_model: str,
    responder_model: str,
    embedding_model: str,
    base_url: str,
    fs_root: str,
    approval_mode: str,
) -> str:
    name = agent_name.strip()
    if not name:
        return "Agent name is required."
    if not name.replace("-", "").replace("_", "").isalnum():
        return "Agent name must be alphanumeric plus '-' or '_'."
    paths = _paths(root, name)
    if paths.agent_home.exists():
        return "Agent already exists."

    paths.agent_home.mkdir(parents=True, exist_ok=True)
    planner_provider_value = _normalize_provider(planner_provider, "ollama")
    responder_provider_value = _normalize_provider(responder_provider, planner_provider_value)
    embedding_provider_value = _normalize_provider(embedding_provider, planner_provider_value)
    default_base = "https://api.openai.com/v1" if planner_provider_value == "openai" else "http://localhost:11434"
    base = base_url.strip() or default_base
    planner_default_model = "gpt-4.1" if planner_provider_value == "openai" else "llama3"
    planner = planner_model.strip() or planner_default_model
    responder = responder_model.strip() or planner
    embed_default_model = "text-embedding-3-small" if embedding_provider_value == "openai" else "nomic-embed-text"
    embed = embedding_model.strip() or embed_default_model
    fs_root_value = fs_root.strip() or str(Path.home())
    mode = approval_mode.strip() or "auto_approve_safe"
    if mode not in {"prompt", "auto_approve_safe", "deny"}:
        mode = "auto_approve_safe"
    cfg = AppConfig(
        agent_id=str(uuid4()),
        agent_name=name,
        created_at=datetime.utcnow().isoformat(),
        role_description=role.strip() or "assistant",
        capabilities=[],
        planner_llm=LLMSettings(provider=planner_provider_value, model=planner, base_url=base, api_key=None),
        responder_llm=LLMSettings(provider=responder_provider_value, model=responder, base_url=base, api_key=None),
        embeddings=EmbeddingSettings(
            provider=embedding_provider_value, model=embed, base_url=base, api_key=None, dim=1536
        ),
        tools=ToolSettings(fs_root=fs_root_value, approval_mode=mode),
        task_runner=TaskRunnerConfig(),
    )
    save_config(paths.config_path, cfg)
    for folder in [
        paths.tool_results_dir,
        paths.vector_dir,
        paths.logs_dir,
        paths.snapshots_dir,
        paths.sleep_dir,
        paths.runs_dir,
        paths.patches_dir,
        paths.inbox_dir,
        paths.outbox_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    repo = _repo_root()
    if repo is not None:
        # Repair stale scaffolded code dir from previous failed UI creates.
        if paths.code_dir.exists() and not (paths.code_dir / ".git").exists():
            shutil.rmtree(paths.code_dir, ignore_errors=True)
        _code_dir, status = ensure_agent_worktree(paths, repo)
        if status.message:
            return f"Agent created, but worktree setup failed: {status.message}"
    return "Agent created."


def _delete_agent(root: Path, agent_name: str) -> str:
    target = root / agent_name
    if not target.exists():
        return "Agent not found."
    if target.name in {"shared", ""}:
        return "Invalid agent."

    repo = _repo_root()
    if repo is not None:
        status = remove_agent_worktree(_paths(root, agent_name), repo)
        if status.message:
            return status.message

    Registry(root / "shared" / "registry.json").remove(agent_name)
    shutil.rmtree(target, ignore_errors=True)
    return "Agent deleted."


def _test_patch(root: Path, agent_name: str, proposal_id: str) -> str:
    paths = _paths(root, agent_name)
    repo = _repo_root()
    if repo is not None:
        code_dir, status = ensure_agent_worktree(paths, repo)
        if status.message:
            return status.message
    else:
        code_dir = paths.code_dir
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        return "Patch proposal not found."

    try:
        parsed = json.loads(str(row["test_results"] or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    tests = parsed.get("tests", [])
    if not tests:
        return "No tests specified for this proposal."

    results = run_patch_tests(tests, cwd=code_dir or Path.cwd())
    parsed["results"] = results
    db.update_patch_proposal(proposal_id=proposal_id, status="tested", test_results=json.dumps(parsed), rollback_ref=row["rollback_ref"])
    return results


def _apply_patch(root: Path, agent_name: str, proposal_id: str) -> str:
    paths = _paths(root, agent_name)
    repo = _repo_root()
    if repo is not None:
        code_dir, status = ensure_agent_worktree(paths, repo)
        if status.message:
            return status.message
    else:
        code_dir = paths.code_dir
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        return "Patch proposal not found."
    if row["status"] not in {"tested", "approved"}:
        return "Patch proposal must be tested or approved before applying."

    try:
        parsed = json.loads(str(row["test_results"] or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    if "results" not in parsed:
        return "Test results missing; run tests first."

    error = apply_patch(str(row["diff_text"]), cwd=code_dir or Path.cwd())
    if error:
        db.update_patch_proposal(proposal_id=proposal_id, status="rejected", test_results=row["test_results"], rollback_ref=row["rollback_ref"])
        return error

    db.update_patch_proposal(proposal_id=proposal_id, status="applied", test_results=row["test_results"], rollback_ref="git_apply_reverse")
    return "Patch applied."


def _reject_patch(root: Path, agent_name: str, proposal_id: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        return "Patch proposal not found."
    db.update_patch_proposal(proposal_id=proposal_id, status="rejected", test_results=row["test_results"], rollback_ref=row["rollback_ref"])
    return "Patch rejected."


def _rollback_patch(root: Path, agent_name: str, proposal_id: str) -> str:
    paths = _paths(root, agent_name)
    repo = _repo_root()
    if repo is not None:
        code_dir, status = ensure_agent_worktree(paths, repo)
        if status.message:
            return status.message
    else:
        code_dir = paths.code_dir
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        return "Patch proposal not found."
    if row["status"] != "applied":
        return "Patch proposal is not applied."

    error = reverse_patch(str(row["diff_text"]), cwd=code_dir or Path.cwd())
    if error:
        return error

    db.update_patch_proposal(proposal_id=proposal_id, status="rejected", test_results=row["test_results"], rollback_ref=row["rollback_ref"])
    return "Patch rolled back."


def _list_ollama_models() -> list[str]:
    try:
        completed = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return []
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    out: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            out.append(parts[0])
    return out


def _badge(status: str) -> str:
    cls = "ok" if status in {"idle", "done", "tested", "applied"} else "warn"
    return f"<span class='pill {cls}'>{html.escape(status)}</span>"


def _nav(active: str) -> str:
    items = [
        ("dashboard", "/", "Dashboard"),
        ("agents", "/agents", "Agents"),
        ("swarm", "/swarm", "Swarm"),
        ("activity", "/activity", "Activity"),
        ("operations", "/operations", "Operations"),
        ("models", "/models", "Models"),
    ]
    links = []
    for key, href, label in items:
        cls = "active" if key == active else ""
        links.append(f"<a class='{cls}' href='{href}'>{label}</a>")
    return "".join(links)


def _layout(title: str, body: str, notice: str = "", detail: str = "", active_nav: str = "dashboard") -> str:
    toast_payload = json.dumps({"title": notice, "detail": detail}) if notice else ""
    toast_script = f"<script>window.__TALI_TOAST={toast_payload};</script>" if notice else ""
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg:#eef3fb; --bg2:#d9e7fb; --panel:#ffffff; --text:#0f1f35; --muted:#4b5d7a;
      --line:#c7d6ea; --accent:#0f5ec9; --accent2:#0a3e8a; --ok:#0d7b46; --warn:#8f4c00;
      --danger:#a91f2d;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.48 'IBM Plex Sans', 'Segoe UI', sans-serif; color:var(--text); background:radial-gradient(1200px 600px at -10% -10%, var(--bg2), var(--bg)); }}
    .shell {{ display:grid; grid-template-columns:240px 1fr; min-height:100vh; }}
    .sidebar {{ background:linear-gradient(190deg, #0d2342, #0a3567); color:#e6eefb; padding:24px 16px; border-right:1px solid rgba(255,255,255,0.08); }}
    .brand {{ font-size:20px; font-weight:800; letter-spacing:0.04em; margin-bottom:18px; }}
    .sidebar a {{ display:block; color:#d9e8ff; text-decoration:none; padding:9px 10px; border-radius:8px; margin-bottom:6px; }}
    .sidebar a:hover {{ background:rgba(255,255,255,0.08); }}
    .sidebar a.active {{ background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; }}
    .content {{ padding:24px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 5px 20px rgba(18,38,69,0.04); }}
    h1 {{ margin:0 0 10px; font-size:25px; }}
    h2 {{ margin:0 0 10px; font-size:18px; }}
    h3 {{ margin:0 0 8px; font-size:15px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); margin:12px 0; }}
    .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
    .stat .k {{ font-size:12px; color:var(--muted); }}
    .stat .v {{ font-size:26px; font-weight:700; line-height:1.1; }}
    .grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; max-width:440px; overflow-wrap:anywhere; }}
    th {{ color:#2b4468; font-weight:700; }}
    .pill {{ border-radius:999px; padding:2px 8px; font-size:12px; font-weight:600; }}
    .ok {{ background:#e7f8ef; color:var(--ok); }}
    .warn {{ background:#fff2df; color:var(--warn); }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 12px; }}
    .tabs a {{ border:1px solid var(--line); border-radius:8px; padding:6px 10px; background:#fff; color:#1a3458; }}
    input,textarea,select {{ width:100%; border:1px solid var(--line); border-radius:9px; padding:8px 10px; font:inherit; background:#fff; margin-top:6px; }}
    button {{ width:auto; border:0; background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; font-weight:700; cursor:pointer; border-radius:9px; padding:8px 12px; margin-top:6px; }}
    button:hover {{ filter:brightness(1.05); }}
    button.secondary {{ background:#4f607f; }}
    button.warn {{ background:var(--danger); }}
    .link-btn {{ display:inline-block; text-decoration:none; border-radius:9px; padding:8px 12px; font-weight:700; background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; }}
    .link-btn.secondary {{ background:#4f607f; color:#fff; }}
    .link-btn.warn {{ background:var(--danger); color:#fff; }}
    .link-btn:hover {{ text-decoration:none; filter:brightness(1.05); }}
    form button[type='submit'] {{ min-width:110px; }}
    .actions {{ display:grid; gap:8px; }}
    .actions form button {{ width:100%; }}
    .actions-inline {{ display:flex; gap:6px; flex-wrap:wrap; }}
    .actions-inline form {{ margin:0; }}
    .actions-inline button {{ min-width:74px; margin-top:0; padding:6px 10px; font-size:13px; }}
    .toolbar {{ display:flex; justify-content:flex-end; margin-bottom:10px; }}
    .modal-backdrop {{ position:fixed; inset:0; background:rgba(8,18,33,0.55); display:none; align-items:center; justify-content:center; padding:16px; z-index:50; }}
    .modal {{ width:min(720px, 100%); max-height:90vh; overflow:auto; background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; }}
    .modal-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
    .cmd-grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }}
    .cmd-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }}
    .field-inline {{ display:flex; align-items:center; gap:8px; margin-top:6px; }}
    .field-inline input[type='checkbox'] {{ width:auto; margin:0; }}
    .field-inline label {{ margin:0; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#0f172a; color:#dbeafe; padding:10px; border-radius:8px; max-height:380px; overflow:auto; }}
    .agent-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }}
    .agent-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; }}
    .row {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:6px; }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .toast-stack {{ position:fixed; right:16px; top:16px; z-index:200; display:grid; gap:10px; width:min(460px, calc(100vw - 32px)); }}
    .toast {{ border:1px solid #9eb9e3; background:#f5f9ff; border-radius:10px; padding:10px 12px; box-shadow:0 8px 24px rgba(10,33,66,0.18); animation:toast-in 180ms ease-out; }}
    .toast h3 {{ margin:0; font-size:14px; color:#153a72; }}
    .toast pre {{ margin:6px 0 0; max-height:180px; background:#163154; color:#edf4ff; }}
    .toast-close {{ background:#29466f; font-size:12px; min-width:auto; padding:4px 8px; }}
    @keyframes toast-in {{ from {{ opacity:0; transform:translateY(-8px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @media (max-width: 920px) {{ .shell {{ grid-template-columns:1fr; }} .sidebar {{ position:sticky; top:0; z-index:5; }} }}
    @media (max-width: 760px) {{ .split {{ grid-template-columns:1fr; }} .content {{ padding:16px; }} }}
  </style>
  {toast_script}
</head>
<body>
  <div id='toast-stack' class='toast-stack'></div>
  <div class=\"shell\">
    <aside class=\"sidebar\">
      <div class=\"brand\">TALI Control</div>
      {_nav(active_nav)}
    </aside>
    <main class=\"content\">
      {body}
    </main>
  </div>
  <script>
    function showToast(title, detail) {{
      const stack = document.getElementById('toast-stack');
      if (!stack) return;
      const toast = document.createElement('section');
      toast.className = 'toast';
      const row = document.createElement('div');
      row.className = 'row';
      const heading = document.createElement('h3');
      heading.textContent = String(title || 'Update');
      const close = document.createElement('button');
      close.type = 'button';
      close.className = 'toast-close';
      close.textContent = 'Dismiss';
      close.addEventListener('click', () => toast.remove());
      row.appendChild(heading);
      row.appendChild(close);
      toast.appendChild(row);
      const detailText = String(detail || '').trim();
      if (detailText) {{
        const pre = document.createElement('pre');
        pre.textContent = detailText;
        toast.appendChild(pre);
      }}
      stack.appendChild(toast);
      setTimeout(() => {{
        if (toast.parentNode) {{
          toast.remove();
        }}
      }}, 9000);
    }}
    if (window.__TALI_TOAST && window.__TALI_TOAST.title) {{
      showToast(window.__TALI_TOAST.title, window.__TALI_TOAST.detail || '');
    }}
  </script>
</body>
</html>"""


def _agent_tabs(agent_name: str) -> str:
    safe = _q(agent_name)
    return (
        "<div class='tabs'>"
        f"<a href='/agent/{safe}'>Overview</a>"
        f"<a href='/agent/{safe}/runs'>Runs</a>"
        f"<a href='/agent/{safe}/patches'>Patches</a>"
        f"<a href='/agent/{safe}/memory'>Memory</a>"
        f"<a href='/agent/{safe}/inbox'>Inbox</a>"
        f"<a href='/agent/{safe}/config'>Config</a>"
        "</div>"
    )


def _render_dashboard(root: Path, notice: str, detail: str) -> str:
    agents = _all_summaries(root)
    total = len(agents)
    active = sum(1 for a in agents if a.run_status == "active")
    blocked = sum(1 for a in agents if a.run_status == "blocked")
    changes = sum(a.git_changes for a in agents)
    unread = sum(a.unread_messages for a in agents)

    cards = []
    for a in agents:
        safe = _q(a.name)
        cards.append(
            "<article class='agent-card'>"
            "<div class='agent-head'>"
            f"<h3><a href='/agent/{safe}'>{html.escape(a.name)}</a></h3>{_badge(a.run_status)}"
            "</div>"
            f"<div class='row'><span class='muted'>Run</span><span>{html.escape((a.run_id or '-')[:8])}</span></div>"
            f"<div class='row'><span class='muted'>Branch</span><span>{html.escape(a.git_branch)}</span></div>"
            f"<div class='row'><span class='muted'>Metrics</span><span>LLM {a.llm_calls or 0} · Tools {a.tool_calls or 0} · Steps {a.steps or 0}</span></div>"
            f"<div class='row'><span class='muted'>Git Changes</span><span>{a.git_changes}</span></div>"
            f"<div class='row'><span class='muted'>Service</span><span>{'running' if a.service_running else 'stopped'}</span></div>"
            f"<div class='row'><span class='muted'>Unread Inbox</span><span>{a.unread_messages}</span></div>"
            f"<div class='row'><span class='muted'>Patch Proposals</span><span>{a.patch_count}</span></div>"
            "</article>"
        )
    cards_html = "".join(cards) or "<p class='muted'>No agents yet. Create one from the Agents page.</p>"

    body = f"""
<h1>Dashboard</h1>
<p class='muted'>Multi-agent status board with live state and git visibility.</p>
<section class='cards'>
  <div class='stat'><div class='k'>Agents</div><div class='v'>{total}</div></div>
  <div class='stat'><div class='k'>Active Runs</div><div class='v'>{active}</div></div>
  <div class='stat'><div class='k'>Blocked Runs</div><div class='v'>{blocked}</div></div>
  <div class='stat'><div class='k'>Uncommitted Files</div><div class='v'>{changes}</div></div>
  <div class='stat'><div class='k'>Unread Messages</div><div class='v'>{unread}</div></div>
</section>
<section class='panel'>
  <h2>Agent Board</h2>
  <div class='grid' id='agent-board'>{cards_html}</div>
</section>
<script>
async function refreshDashboard() {{
  try {{
    const r = await fetch('/api/dashboard');
    if (!r.ok) return;
    const data = await r.json();
    const board = document.getElementById('agent-board');
    if (!board) return;
    const cards = (data.agents || []).map((a) => {{
      const cls = (a.run_status === 'idle' || a.run_status === 'done') ? 'ok' : 'warn';
      const run = a.run_id ? a.run_id.slice(0,8) : '-';
      return `<article class="agent-card"><div class="agent-head"><h3><a href="/agent/${{encodeURIComponent(a.name)}}">${{a.name}}</a></h3><span class="pill ${{cls}}">${{a.run_status}}</span></div><div class="row"><span class="muted">Run</span><span>${{run}}</span></div><div class="row"><span class="muted">Branch</span><span>${{a.git_branch}}</span></div><div class="row"><span class="muted">Metrics</span><span>LLM ${{a.llm_calls ?? 0}} · Tools ${{a.tool_calls ?? 0}} · Steps ${{a.steps ?? 0}}</span></div><div class="row"><span class="muted">Git Changes</span><span>${{a.git_changes}}</span></div><div class="row"><span class="muted">Unread Inbox</span><span>${{a.unread_messages ?? 0}}</span></div><div class="row"><span class="muted">Patch Proposals</span><span>${{a.patch_count ?? 0}}</span></div></article>`;
    }}).join('');
    board.innerHTML = cards || '<p class="muted">No agents yet. Create one from the Agents page.</p>';
  }} catch (_err) {{}}
}}
setInterval(refreshDashboard, 2500);
</script>
"""
    return _layout("Tali Dashboard", body, notice, detail, active_nav="dashboard")


def _render_agents(root: Path, notice: str, detail: str) -> str:
    agents = _all_summaries(root)
    rows = []
    for a in agents:
        safe = _q(a.name)
        next_path = "/agents"
        rows.append(
            "<tr>"
            f"<td><a href='/agent/{safe}'>{html.escape(a.name)}</a></td>"
            f"<td>{_badge(a.run_status)}</td>"
            f"<td>{'running' if a.service_running else 'stopped'}{f' (pid {a.service_pid})' if a.service_pid else ''}</td>"
            f"<td>{html.escape((a.run_id or '-')[:8])}</td>"
            f"<td>{html.escape(a.git_branch)}</td>"
            f"<td>{a.git_changes}</td>"
            f"<td>{a.unread_messages}</td>"
            f"<td>{a.patch_count}</td>"
            f"<td><form method='post' action='/agent/{safe}/start'><input type='hidden' name='next' value='{next_path}' /><button type='submit'>Start Service</button></form></td>"
            f"<td><form method='post' action='/agent/{safe}/stop'><input type='hidden' name='next' value='{next_path}' /><button class='warn' type='submit'>Stop</button></form></td>"
            f"<td><form method='post' action='/agent/{safe}/resume'><input type='hidden' name='next' value='{next_path}' /><button class='secondary' type='submit'>Resume Run</button></form></td>"
            f"<td><form method='post' action='/agent/{safe}/cancel'><input type='hidden' name='next' value='{next_path}' /><button class='secondary' type='submit'>Cancel Run</button></form></td>"
            "</tr>"
        )
    table = "".join(rows) or "<tr><td colspan='12' class='muted'>No agents.</td></tr>"
    llm_model_options = _model_options()
    embedding_model_options = _model_options(include_embedding_defaults=True)

    body = f"""
<h1>Agents</h1>
<p class='muted'>Lifecycle controls, runtime status, and direct links into each agent workspace.</p>
<section class='panel'>
  <div class='toolbar' style='gap:8px;'>
    <a href='/swarm' class='link-btn secondary'>Open Swarm Console</a>
    <button type='button' onclick='openCreateModal()'>Create Agent</button>
  </div>
  <h2>Agent List</h2>
  <div class='table-wrap'>
    <table>
      <thead><tr><th>Name</th><th>Run Status</th><th>Service</th><th>Run</th><th>Branch</th><th>Changes</th><th>Unread</th><th>Patches</th><th></th><th></th><th></th><th></th></tr></thead>
      <tbody>{table}</tbody>
    </table>
  </div>
</section>
<div id='create-modal' class='modal-backdrop' onclick='if(event.target===this)closeCreateModal()'>
  <div class='modal'>
    <div class='modal-head'>
      <h2>Create Agent</h2>
      <button type='button' class='secondary' style='width:auto' onclick='closeCreateModal()'>Close</button>
    </div>
    <form id='create-agent-form' method='post' action='/agents/create'>
      <label>Name<input name='name' placeholder='agent-name' required /></label>
      <label>Role<input name='role' placeholder='assistant' /></label>
      {_render_provider_field("Planner Provider", "planner_provider", "ollama")}
      {_render_model_field("Planner Model", "planner_model", "llama3", llm_model_options)}
      {_render_provider_field("Responder Provider", "responder_provider", "ollama")}
      {_render_model_field("Responder Model", "responder_model", "llama3", llm_model_options)}
      {_render_provider_field("Embedding Provider", "embedding_provider", "ollama")}
      {_render_model_field("Embedding Model", "embedding_model", "nomic-embed-text", embedding_model_options)}
      <label>Base URL<input name='base_url' value='http://localhost:11434' /></label>
      <label>File Root<input name='fs_root' value='{html.escape(str(Path.home()))}' /></label>
      <label>Approval Mode
        <select name='approval_mode'>
          <option value='auto_approve_safe' selected>auto_approve_safe</option>
          <option value='prompt'>prompt</option>
          <option value='deny'>deny</option>
        </select>
      </label>
      <button type='submit'>Create Agent</button>
    </form>
  </div>
</div>
<script>
function openCreateModal() {{
  const m = document.getElementById('create-modal');
  if (m) m.style.display = 'flex';
}}
function closeCreateModal() {{
  const m = document.getElementById('create-modal');
  if (m) m.style.display = 'none';
}}
function syncModelChoiceFields(formId) {{
  const form = document.getElementById(formId);
  if (!form) return;
  const selects = form.querySelectorAll('select.model-choice');
  selects.forEach((select) => {{
    const otherName = select.getAttribute('data-other-input');
    if (!otherName) return;
    const other = form.querySelector(`input[name="${{otherName}}"]`);
    if (!other) return;
    const sync = () => {{
      const showOther = select.value === '{_MODEL_OTHER_VALUE}';
      other.style.display = showOther ? 'block' : 'none';
      if (!showOther) other.value = '';
    }};
    select.addEventListener('change', sync);
    sync();
  }});
}}
syncModelChoiceFields('create-agent-form');
</script>
"""
    return _layout("Tali Agents", body, notice, detail, active_nav="agents")


def _render_swarm(root: Path, notice: str, detail: str) -> str:
    local_names = _list_agent_names(root)
    summaries = {item.name: item for item in _all_summaries(root)}
    registry_rows = Registry(root / "shared" / "registry.json").list_agents()

    names = sorted({*local_names, *[str(row.get("agent_name") or "") for row in registry_rows if row.get("agent_name")]})
    table_rows: list[str] = []
    for name in names:
        rec = next((row for row in registry_rows if row.get("agent_name") == name), None) or {}
        summary = summaries.get(name)
        service = "running" if summary and summary.service_running else "stopped"
        run_status = summary.run_status if summary else str(rec.get("status") or "unknown")
        last_seen = str(rec.get("last_seen") or "-")
        capabilities = ", ".join(str(x) for x in (rec.get("capabilities") or [])) or "-"
        load = rec.get("load") or {}
        load_text = ", ".join(f"{k}={v}" for k, v in load.items()) if isinstance(load, dict) and load else "-"
        actions = "<span class='muted'>n/a</span>"
        if name in local_names:
            safe = _q(name)
            actions = (
                "<div class='actions-inline'>"
                f"<form method='post' action='/agent/{safe}/start'><input type='hidden' name='next' value='/swarm' /><button type='submit'>Start</button></form>"
                f"<form method='post' action='/agent/{safe}/stop'><input type='hidden' name='next' value='/swarm' /><button class='warn' type='submit'>Stop</button></form>"
                f"<a href='/agent/{safe}' class='link-btn secondary'>Open</a>"
                "</div>"
            )
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_badge(run_status)}</td>"
            f"<td>{html.escape(service)}</td>"
            f"<td>{html.escape(last_seen)}</td>"
            f"<td>{html.escape(capabilities)}</td>"
            f"<td>{html.escape(load_text)}</td>"
            f"<td>{actions}</td>"
            "</tr>"
        )
    table = "".join(table_rows) or "<tr><td colspan='7' class='muted'>No swarm agents found.</td></tr>"
    coordinator_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>" for name in local_names
    ) or "<option value=''>No local agents</option>"
    coordinator_disabled = " disabled" if not local_names else ""

    body = f"""
<h1>Swarm</h1>
<p class='muted'>Coordinate multi-agent work, run swarm prompts, and manage service fleet state.</p>
<section class='panel'>
  <h2>Fleet Actions</h2>
  <div class='actions-inline'>
    <form method='post' action='/swarm/start-all'><button type='submit'>Start All Services</button></form>
    <form method='post' action='/swarm/stop-all'><button class='warn' type='submit'>Stop All Services</button></form>
  </div>
</section>
<section class='panel' style='margin-top:12px;'>
  <h2>Run Swarm Prompt</h2>
  <form method='post' action='/swarm/run'>
    <div class='split'>
      <label>Coordinator Agent
        <select name='coordinator'{coordinator_disabled}>{coordinator_options}</select>
      </label>
      <label>Prompt
        <input name='prompt' placeholder='Plan and delegate implementation tasks across available agents.' required />
      </label>
    </div>
    <button type='submit'{coordinator_disabled}>Execute Swarm Task</button>
  </form>
</section>
<section class='panel' style='margin-top:12px;'>
  <h2>Registered Agents</h2>
  <div class='table-wrap'>
    <table>
      <thead><tr><th>Name</th><th>Run</th><th>Service</th><th>Last Seen</th><th>Capabilities</th><th>Load</th><th>Actions</th></tr></thead>
      <tbody>{table}</tbody>
    </table>
  </div>
</section>
"""
    return _layout("Tali Swarm", body, notice, detail, active_nav="swarm")


def _render_activity(root: Path, notice: str, detail: str) -> str:
    runs: list[tuple[str, Any]] = []
    log_blocks: list[str] = []
    for name in _list_agent_names(root):
        paths = _paths(root, name)
        try:
            db = _init_db(paths.db_path)
        except (sqlite3.Error, OSError):
            continue
        for run in db.list_runs(limit=12):
            runs.append((name, run))
        try:
            logs = read_recent_logs(paths.logs_dir, limit=8)
        except Exception:
            logs = []
        if logs:
            joined = "\n".join(_format_log_entry(entry) for entry in logs)
            log_blocks.append(f"<h3>{html.escape(name)}</h3><pre>{html.escape(joined)}</pre>")

    runs.sort(key=lambda item: str(item[1]["created_at"]), reverse=True)
    runs_table = "".join(
        "<tr>"
        f"<td><a href='/agent/{_q(name)}'>{html.escape(name)}</a></td>"
        f"<td>{html.escape(str(run['created_at']))}</td>"
        f"<td><code>{html.escape(str(run['id'])[:8])}</code></td>"
        f"<td>{_badge(str(run['status']))}</td>"
        f"<td>{html.escape(str(run['user_prompt'])[:120])}</td>"
        "</tr>"
        for name, run in runs
    ) or "<tr><td colspan='5' class='muted'>No run history yet.</td></tr>"
    logs_html = "".join(log_blocks) or "<p class='muted'>No logs yet.</p>"

    body = f"""
<h1>Activity</h1>
<p class='muted'>Cross-agent run history and recent logs.</p>
<section class='panel'>
  <h2>Recent Runs (All Agents)</h2>
  <div class='table-wrap'>
    <table>
      <thead><tr><th>Agent</th><th>Created</th><th>ID</th><th>Status</th><th>Prompt</th></tr></thead>
      <tbody>{runs_table}</tbody>
    </table>
  </div>
</section>
<section class='panel' style='margin-top:12px;'>
  <h2>Recent Logs</h2>
  {logs_html}
</section>
"""
    return _layout("Tali Activity", body, notice, detail, active_nav="activity")


def _render_models(notice: str, detail: str) -> str:
    models = _list_ollama_models()
    li = "".join(f"<li><code>{html.escape(m)}</code></li>" for m in models) or "<li class='muted'>No local Ollama models detected.</li>"
    body = f"""
<h1>Models</h1>
<section class='panel'>
  <h2>Installed Ollama Models</h2>
  <ul>{li}</ul>
  <p class='muted'>Use terminal for model pulls: <code>ollama pull model-name</code>.</p>
</section>
    """
    return _layout("Tali Models", body, notice, detail, active_nav="models")


def _render_operations(root: Path, notice: str, detail: str, agent_context: str | None) -> str:
    specs = _command_specs()
    by_group: dict[str, list[CommandSpec]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec)

    agents = _list_agent_names(root)
    selected = agent_context or (agents[0] if agents else "")
    options = ["<option value=''>No default agent</option>"]
    for name in agents:
        is_selected = " selected" if name == selected else ""
        options.append(f"<option value='{html.escape(name)}'{is_selected}>{html.escape(name)}</option>")
    context_html = "".join(options)

    groups_html: list[str] = []
    for group_name in sorted(by_group.keys()):
        cards: list[str] = []
        for spec in by_group[group_name]:
            fields: list[str] = []
            for param in spec.params:
                field_name = f"p__{param.name}"
                if param.is_flag:
                    checked = " checked" if param.default.lower() == "true" else ""
                    label = param.option or param.name
                    fields.append(
                        f"<label class='field-inline'><input type='checkbox' name='{html.escape(field_name)}' value='1'{checked} />"
                        f"<span>{html.escape(label)}</span></label>"
                    )
                    continue
                value = param.default
                if param.kind == "argument" and param.name == "agent_name" and selected:
                    value = selected
                req = " required" if param.required else ""
                label = param.option or param.name
                fields.append(
                    f"<label>{html.escape(label)}"
                    f"<input name='{html.escape(field_name)}' value='{html.escape(value, quote=True)}'{req} />"
                    "</label>"
                )
            params_html = "".join(fields) or "<p class='muted'>No parameters.</p>"
            run_button = (
                "<button type='submit'>Run</button>"
                if spec.runnable
                else "<button type='button' class='secondary' disabled>Interactive Only</button>"
            )
            cards.append(
                "<article class='cmd-card'>"
                f"<h3>{html.escape(spec.display)}</h3>"
                f"<p class='muted'>{'Runnable in UI' if spec.runnable else 'Not runnable in UI (interactive command).'}</p>"
                f"<form method='post' action='/operations/run'>"
                f"<input type='hidden' name='cmd_path' value='{html.escape(spec.display, quote=True)}' />"
                f"<input type='hidden' name='agent_context' value='{html.escape(selected, quote=True)}' />"
                f"{params_html}"
                f"{run_button}"
                "</form>"
                "</article>"
            )
        groups_html.append(f"<section class='panel'><h2>{html.escape(group_name.title())}</h2><div class='cmd-grid'>{''.join(cards)}</div></section>")

    body = f"""
<h1>Operations</h1>
<p class='muted'>GUI forms for CLI command parity. Every runnable CLI command is available here.</p>
<section class='panel'>
  <h2>Default Agent Context</h2>
  <form method='post' action='/operations/context'>
    <label>Agent
      <select name='agent_context'>{context_html}</select>
    </label>
    <button type='submit'>Set Context</button>
  </form>
</section>
{''.join(groups_html)}
"""
    return _layout("Tali Operations", body, notice, detail, active_nav="operations")


def _render_agent_overview(root: Path, agent_name: str, notice: str, detail: str) -> str:
    data = _collect_agent_data(root, agent_name)
    paths = _paths(root, agent_name)
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    tasks = db.fetch_tasks_for_run(str(run["id"])) if run else []

    task_rows = "".join(
        f"<tr><td>{int(t['ordinal']) + 1}</td><td>{html.escape(str(t['title']))}</td><td>{html.escape(str(t['status']))}</td></tr>"
        for t in tasks
    ) or "<tr><td colspan='3' class='muted'>No active tasks.</td></tr>"

    files = "".join(f"<li><code>{html.escape(f)}</code></li>" for f in data["git_files"]) or "<li class='muted'>No pending changes.</li>"
    safe = _q(agent_name)

    body = f"""
<p><a href='/agents'>Back to agents</a></p>
<h1>Agent: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<div class='grid'>
  <section class='panel'>
    <h2>Status</h2>
    <div class='row'><span>Run</span><span>{_badge(str(data['run_status']))}</span></div>
    <div class='row'><span>Service</span><span id='service-status'>{'running' if data['service_running'] else 'stopped'}{f" (pid {data['service_pid']})" if data['service_pid'] else ''}</span></div>
    <div class='row'><span>Run ID</span><span>{html.escape(str(data['run_id'] or '-'))}</span></div>
    <div class='row'><span>LLM / Tools / Steps</span><span>{data['llm_calls'] or 0} / {data['tool_calls'] or 0} / {data['steps'] or 0}</span></div>
    <div class='row'><span>Unread Inbox</span><span>{data['unread_messages']}</span></div>
    <div class='row'><span>Patches</span><span>{data['patch_count']}</span></div>
  </section>
  <section class='panel'>
    <h2>Run Controls</h2>
    <div class='actions'>
      <form method='post' action='/agent/{safe}/chat'>
        <input type='hidden' name='next' value='/agent/{safe}' />
        <label>Task Message
          <textarea name='message' rows='4' placeholder='Use `resume` to continue a blocked run, or enter a concrete next action.'></textarea>
        </label>
        <button type='submit'>Send Prompt</button>
      </form>
      <form method='post' action='/agent/{safe}/resume-turn'><input type='hidden' name='next' value='/agent/{safe}' /><button class='secondary' type='submit'>Send `resume`</button></form>
      <form method='post' action='/agent/{safe}/start'><input type='hidden' name='next' value='/agent/{safe}' /><button type='submit'>Start Service</button></form>
      <form method='post' action='/agent/{safe}/stop'><input type='hidden' name='next' value='/agent/{safe}' /><button class='warn' type='submit'>Stop Service</button></form>
      <form method='post' action='/agent/{safe}/resume'><input type='hidden' name='next' value='/agent/{safe}' /><button class='secondary' type='submit'>Resume Run State</button></form>
      <form method='post' action='/agent/{safe}/cancel'><input type='hidden' name='next' value='/agent/{safe}' /><button class='secondary' type='submit'>Cancel Run</button></form>
      <form method='post' action='/agent/{safe}/delete'><input type='hidden' name='next' value='/agents' /><button class='warn' type='submit'>Delete Agent</button></form>
    </div>
  </section>
</div>
<div class='grid' style='margin-top:12px;'>
  <section class='panel'>
    <h2>Current Tasks</h2>
    <table><thead><tr><th>#</th><th>Title</th><th>Status</th></tr></thead><tbody id='tasks-body'>{task_rows}</tbody></table>
  </section>
  <section class='panel'>
    <h2>Git Worktree</h2>
    <div class='row'><span>Branch</span><strong id='git-branch'>{html.escape(str(data['git_branch']))}</strong></div>
    <div class='row'><span>Changed Files</span><strong id='git-count'>{data['git_changes']}</strong></div>
    <ul id='git-files'>{files}</ul>
  </section>
</div>
<script>
async function refreshAgent() {{
  try {{
    const r = await fetch('/api/agent/{safe}');
    if (!r.ok) return;
    const d = await r.json();
    const gitBranch = document.getElementById('git-branch');
    const gitCount = document.getElementById('git-count');
    const gitFiles = document.getElementById('git-files');
    const serviceStatus = document.getElementById('service-status');
    const tasksBody = document.getElementById('tasks-body');
    if (serviceStatus) {{
      const label = d.service_running ? `running${{d.service_pid ? ` (pid ${{d.service_pid}})` : ''}}` : 'stopped';
      serviceStatus.textContent = label;
    }}
    if (gitBranch) gitBranch.textContent = d.git_branch || '(none)';
    if (gitCount) gitCount.textContent = String(d.git_changes ?? 0);
    if (gitFiles) {{
      const rows = (d.git_files || []).map((f) => `<li><code>${{f}}</code></li>`).join('');
      gitFiles.innerHTML = rows || "<li class='muted'>No pending changes.</li>";
    }}
    if (tasksBody) {{
      const rows = (d.tasks || []).map((t) => `<tr><td>${{t.ordinal}}</td><td>${{t.title}}</td><td>${{t.status}}</td></tr>`).join('');
      tasksBody.innerHTML = rows || "<tr><td colspan='3' class='muted'>No active tasks.</td></tr>";
    }}
  }} catch (_err) {{}}
}}
setInterval(refreshAgent, 2500);
</script>
"""
    return _layout(f"Agent {agent_name}", body, notice, detail, active_nav="agents")


def _render_agent_runs(root: Path, agent_name: str, notice: str, detail: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    active = db.fetch_active_run()
    rows = db.list_runs(limit=200)
    table = "".join(
        "<tr>"
        f"<td>{html.escape(str(r['created_at']))}</td>"
        f"<td><code>{html.escape(str(r['id'])[:8])}</code></td>"
        f"<td>{_badge(str(r['status']))}</td>"
        f"<td>{html.escape(str(r['user_prompt'])[:160])}</td>"
        "</tr>"
        for r in rows
    ) or "<tr><td colspan='4' class='muted'>No runs yet.</td></tr>"
    body = f"""
<p><a href='/agent/{_q(agent_name)}'>Back to overview</a></p>
<h1>Runs: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<section class='panel'>
  <h2>Run Controls</h2>
  <div class='row'><span>Active Run</span><strong>{html.escape(str(active['id'])[:8]) if active else '-'}</strong></div>
  <div class='actions'>
    <form method='post' action='/agent/{_q(agent_name)}/resume'><input type='hidden' name='next' value='/agent/{_q(agent_name)}/runs' /><button class='secondary' type='submit'>Resume Run State</button></form>
    <form method='post' action='/agent/{_q(agent_name)}/resume-turn'><input type='hidden' name='next' value='/agent/{_q(agent_name)}/runs' /><button class='secondary' type='submit'>Send `resume` Turn</button></form>
    <form method='post' action='/agent/{_q(agent_name)}/cancel'><input type='hidden' name='next' value='/agent/{_q(agent_name)}/runs' /><button class='secondary' type='submit'>Cancel Active Run</button></form>
  </div>
</section>
<section class='panel' style='margin-top:12px;'>
  <h2>Run History</h2>
  <div class='table-wrap'>
    <table><thead><tr><th>Created</th><th>ID</th><th>Status</th><th>Prompt</th></tr></thead><tbody>{table}</tbody></table>
  </div>
</section>
"""
    return _layout(f"Runs {agent_name}", body, notice, detail, active_nav="agents")


def _render_agent_patches(root: Path, agent_name: str, notice: str, detail: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    rows = db.list_patch_proposals()
    table = "".join(
        "<tr>"
        f"<td><a href='/agent/{_q(agent_name)}/patches/{_q(str(r['id']))}'><code>{html.escape(str(r['id'])[:8])}</code></a></td>"
        f"<td>{html.escape(str(r['title']))}</td>"
        f"<td>{_badge(str(r['status']))}</td>"
        f"<td>{html.escape(str(r['created_at']))}</td>"
        "</tr>"
        for r in rows
    ) or "<tr><td colspan='4' class='muted'>No patch proposals.</td></tr>"
    body = f"""
<p><a href='/agent/{_q(agent_name)}'>Back to overview</a></p>
<h1>Patches: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<section class='panel'>
  <div class='table-wrap'>
    <table><thead><tr><th>ID</th><th>Title</th><th>Status</th><th>Created</th></tr></thead><tbody>{table}</tbody></table>
  </div>
</section>
"""
    return _layout(f"Patches {agent_name}", body, notice, detail, active_nav="agents")


def _render_patch_detail(root: Path, agent_name: str, proposal_id: str, notice: str, detail: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        return _layout("Patch not found", "<h1>Patch not found</h1>", notice, detail, active_nav="agents")

    result_text = ""
    if row["test_results"]:
        try:
            parsed = json.loads(str(row["test_results"]))
            result_text = str(parsed.get("results") or "")
        except json.JSONDecodeError:
            result_text = str(row["test_results"])

    safe_agent = _q(agent_name)
    safe_pid = _q(proposal_id)
    body = f"""
<p><a href='/agent/{safe_agent}/patches'>Back to patches</a></p>
<h1>Patch <code>{html.escape(proposal_id)}</code></h1>
{_agent_tabs(agent_name)}
<div class='grid'>
  <section class='panel'>
    <h2>Metadata</h2>
    <div class='row'><span>Title</span><strong>{html.escape(str(row['title']))}</strong></div>
    <div class='row'><span>Status</span><span>{_badge(str(row['status']))}</span></div>
    <div class='row'><span>Created</span><span>{html.escape(str(row['created_at']))}</span></div>
    <p><strong>Rationale</strong></p>
    <pre>{html.escape(str(row['rationale'] or ''))}</pre>
  </section>
  <section class='panel'>
    <h2>Actions</h2>
    <div class='actions'>
      <form method='post' action='/agent/{safe_agent}/patches/{safe_pid}/test'><button type='submit'>Run Tests</button></form>
      <form method='post' action='/agent/{safe_agent}/patches/{safe_pid}/apply'><button type='submit'>Apply Patch</button></form>
      <form method='post' action='/agent/{safe_agent}/patches/{safe_pid}/reject'><button class='secondary' type='submit'>Reject Patch</button></form>
      <form method='post' action='/agent/{safe_agent}/patches/{safe_pid}/rollback'><button class='warn' type='submit'>Rollback Patch</button></form>
    </div>
  </section>
</div>
<section class='panel' style='margin-top:12px;'><h2>Diff</h2><pre>{html.escape(str(row['diff_text'] or ''))}</pre></section>
<section class='panel' style='margin-top:12px;'><h2>Test Results</h2><pre>{html.escape(result_text or '(none)')}</pre></section>
"""
    return _layout(f"Patch {proposal_id}", body, notice, detail, active_nav="agents")


def _render_agent_memory(root: Path, agent_name: str, notice: str, detail: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    commitments = db.list_commitments()
    facts = db.list_facts()[:100]
    skills = db.list_skills()[:100]
    safe_agent = _q(agent_name)
    commitment_rows = []
    for row in commitments:
        cid = str(row["id"])
        selected = []
        for status in _COMMITMENT_STATUSES:
            marker = " selected" if status == str(row["status"]) else ""
            selected.append(f"<option value='{status}'{marker}>{status}</option>")
        due_val = html.escape(str(row["due_date"] or ""), quote=True)
        commitment_rows.append(
            "<tr>"
            f"<td><code>{html.escape(cid[:8])}</code></td>"
            "<td>"
            f"<form method='post' action='/agent/{safe_agent}/commitments/update'>"
            f"<input type='hidden' name='next' value='/agent/{safe_agent}/memory' />"
            f"<input type='hidden' name='commitment_id' value='{html.escape(cid, quote=True)}' />"
            f"<input name='description' value='{html.escape(str(row['description']), quote=True)}' />"
            "<div class='split'>"
            f"<label>Status<select name='status'>{''.join(selected)}</select></label>"
            f"<label>Priority<input name='priority' type='number' min='1' max='5' value='{int(row['priority'] or 3)}' /></label>"
            "</div>"
            f"<label>Due Date<input name='due_date' placeholder='YYYY-MM-DD' value='{due_val}' /></label>"
            "<button class='secondary' type='submit'>Save</button>"
            "</form>"
            "</td>"
            "</tr>"
        )
    commitments_rows = "".join(commitment_rows) or "<tr><td colspan='2' class='muted'>No commitments.</td></tr>"

    facts_rows = "".join(
        f"<tr><td>{html.escape(str(r['statement']))}</td><td>{float(r['confidence']):.2f}</td><td>{'yes' if int(r['contested']) else 'no'}</td></tr>"
        for r in facts
    ) or "<tr><td colspan='3' class='muted'>No facts.</td></tr>"

    skills_rows = "".join(
        f"<tr><td>{html.escape(str(r['name']))}</td><td>{int(r['success_count'])}</td><td>{int(r['failure_count'])}</td></tr>"
        for r in skills
    ) or "<tr><td colspan='3' class='muted'>No skills.</td></tr>"

    body = f"""
<p><a href='/agent/{_q(agent_name)}'>Back to overview</a></p>
<h1>Memory: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<div class='grid'>
  <section class='panel'>
    <h2>Commitments</h2>
    <form method='post' action='/agent/{safe_agent}/commitments/add'>
      <input type='hidden' name='next' value='/agent/{safe_agent}/memory' />
      <label>Description<input name='description' placeholder='Follow up with API migration checklist.' required /></label>
      <div class='split'>
        <label>Priority (1-5)<input name='priority' type='number' min='1' max='5' value='3' /></label>
        <label>Due Date<input name='due_date' placeholder='YYYY-MM-DD' /></label>
      </div>
      <button type='submit'>Add Commitment</button>
    </form>
    <div class='table-wrap'>
      <table><thead><tr><th>ID</th><th>Manage</th></tr></thead><tbody>{commitments_rows}</tbody></table>
    </div>
  </section>
  <section class='panel'>
    <h2>Skills</h2>
    <div class='table-wrap'>
      <table><thead><tr><th>Name</th><th>Success</th><th>Failure</th></tr></thead><tbody>{skills_rows}</tbody></table>
    </div>
  </section>
</div>
<section class='panel' style='margin-top:12px;'>
  <h2>Facts</h2>
  <div class='table-wrap'>
    <table><thead><tr><th>Statement</th><th>Confidence</th><th>Contested</th></tr></thead><tbody>{facts_rows}</tbody></table>
  </div>
</section>
"""
    return _layout(f"Memory {agent_name}", body, notice, detail, active_nav="agents")


def _render_agent_config(root: Path, agent_name: str, notice: str, detail: str) -> str:
    try:
        config = _load_agent_config(root, agent_name)
    except Exception as exc:
        return _layout(
            f"Config {agent_name}",
            f"<h1>Config: {html.escape(agent_name)}</h1><p class='muted'>Unable to load config: {html.escape(str(exc))}</p>",
            notice,
            detail,
            active_nav="agents",
        )

    planner = config.planner_llm or LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434")
    responder = config.responder_llm or LLMSettings(
        provider=planner.provider, model=planner.model, base_url=planner.base_url
    )
    embeddings = config.embeddings or EmbeddingSettings(
        provider=planner.provider, model="nomic-embed-text", base_url=planner.base_url, dim=1536
    )
    tools = config.tools or ToolSettings()
    task_runner = config.task_runner or TaskRunnerConfig()
    autonomy = config.autonomy or AutonomyConfig()
    safe_agent = _q(agent_name)
    raw_config = ""
    try:
        raw_config = _paths(root, agent_name).config_path.read_text(encoding="utf-8")
    except OSError:
        raw_config = "(unable to read config file)"

    auto_enabled = " checked" if autonomy.enabled else ""
    auto_continue = " checked" if autonomy.auto_continue else ""
    execute_commitments = " checked" if autonomy.execute_commitments else ""
    python_enabled = " checked" if tools.python_enabled else ""
    approval_options = []
    for mode in ("prompt", "auto_approve_safe", "deny"):
        marker = " selected" if mode == tools.approval_mode else ""
        approval_options.append(f"<option value='{mode}'{marker}>{mode}</option>")
    llm_model_options = _model_options()
    embedding_model_options = _model_options(include_embedding_defaults=True)

    body = f"""
<p><a href='/agent/{safe_agent}'>Back to overview</a></p>
<h1>Config: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<form id='agent-config-form' method='post' action='/agent/{safe_agent}/config/save'>
  <input type='hidden' name='next' value='/agent/{safe_agent}/config' />
  <div class='grid'>
    <section class='panel'>
      <h2>Role + Models</h2>
      <label>Role Description<input name='role_description' value='{html.escape(config.role_description, quote=True)}' /></label>
      {_render_provider_field("Planner Provider", "planner_provider", planner.provider)}
      {_render_model_field("Planner Model", "planner_model", planner.model, llm_model_options)}
      <label>Planner Base URL<input name='planner_base_url' value='{html.escape(planner.base_url, quote=True)}' /></label>
      {_render_provider_field("Responder Provider", "responder_provider", responder.provider)}
      {_render_model_field("Responder Model", "responder_model", responder.model, llm_model_options)}
      <label>Responder Base URL<input name='responder_base_url' value='{html.escape(responder.base_url, quote=True)}' /></label>
      {_render_provider_field("Embedding Provider", "embedding_provider", embeddings.provider)}
      {_render_model_field("Embedding Model", "embedding_model", embeddings.model, embedding_model_options)}
      <label>Embedding Base URL<input name='embedding_base_url' value='{html.escape(embeddings.base_url, quote=True)}' /></label>
    </section>
    <section class='panel'>
      <h2>Runtime Controls</h2>
      <label>Approval Mode<select name='approval_mode'>{''.join(approval_options)}</select></label>
      <label>Filesystem Root<input name='fs_root' value='{html.escape(str(tools.fs_root or ''), quote=True)}' /></label>
      <div class='field-inline'><input type='checkbox' name='python_enabled' value='1'{python_enabled} /><span>Python tool enabled</span></div>
      <label>Max Tool Calls / Turn<input type='number' min='1' max='500' name='max_tool_calls_per_turn' value='{tools.max_tool_calls_per_turn}' /></label>
      <label>Max Tasks / Turn<input type='number' min='1' max='500' name='max_tasks_per_turn' value='{task_runner.max_tasks_per_turn}' /></label>
      <label>Max Steps / Turn<input type='number' min='1' max='5000' name='max_total_steps_per_turn' value='{task_runner.max_total_steps_per_turn}' /></label>
      <div class='field-inline'><input type='checkbox' name='autonomy_enabled' value='1'{auto_enabled} /><span>Autonomy enabled</span></div>
      <div class='field-inline'><input type='checkbox' name='auto_continue' value='1'{auto_continue} /><span>Auto-continue runs</span></div>
      <div class='field-inline'><input type='checkbox' name='execute_commitments' value='1'{execute_commitments} /><span>Execute commitments in autonomy</span></div>
      <label>Idle Trigger Seconds<input type='number' min='5' max='86400' name='idle_trigger_seconds' value='{autonomy.idle_trigger_seconds}' /></label>
      <button type='submit'>Save Config</button>
    </section>
  </div>
</form>
<section class='panel' style='margin-top:12px;'>
  <h2>Raw Config</h2>
  <pre>{html.escape(raw_config)}</pre>
</section>
<script>
function syncModelChoiceFields(formId) {{
  const form = document.getElementById(formId);
  if (!form) return;
  const selects = form.querySelectorAll('select.model-choice');
  selects.forEach((select) => {{
    const otherName = select.getAttribute('data-other-input');
    if (!otherName) return;
    const other = form.querySelector(`input[name="${{otherName}}"]`);
    if (!other) return;
    const sync = () => {{
      const showOther = select.value === '{_MODEL_OTHER_VALUE}';
      other.style.display = showOther ? 'block' : 'none';
      if (!showOther) other.value = '';
    }};
    select.addEventListener('change', sync);
    sync();
  }});
}}
syncModelChoiceFields('agent-config-form');
</script>
"""
    return _layout(f"Config {agent_name}", body, notice, detail, active_nav="agents")


def _render_agent_inbox(root: Path, agent_name: str, notice: str, detail: str) -> str:
    db = _init_db(_paths(root, agent_name).db_path)
    rows = db.list_unread_agent_messages(limit=200)
    msg_rows = []
    for row in rows:
        mid = str(row["id"])
        msg_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['timestamp']))}</td>"
            f"<td>{html.escape(str(row['from_agent_name'] or '-'))}</td>"
            f"<td>{html.escape(str(row['topic']))}</td>"
            f"<td><pre>{html.escape(str(row['payload']))}</pre></td>"
            f"<td><form method='post' action='/agent/{_q(agent_name)}/inbox/{_q(mid)}/read'><button class='secondary' type='submit'>Mark Read</button></form></td>"
            "</tr>"
        )
    table = "".join(msg_rows) or "<tr><td colspan='5' class='muted'>No unread messages.</td></tr>"
    body = f"""
<p><a href='/agent/{_q(agent_name)}'>Back to overview</a></p>
<h1>Inbox: {html.escape(agent_name)}</h1>
{_agent_tabs(agent_name)}
<section class='panel'>
  <div class='table-wrap'>
    <table><thead><tr><th>Time</th><th>From</th><th>Topic</th><th>Payload</th><th>Action</th></tr></thead><tbody>{table}</tbody></table>
  </div>
</section>
    """
    return _layout(f"Inbox {agent_name}", body, notice, detail, active_nav="agents")


def _find_command_spec(display: str) -> CommandSpec | None:
    for spec in _command_specs():
        if spec.display == display:
            return spec
    return None


def _build_command_args(spec: CommandSpec, data: dict[str, str], agent_context: str | None) -> tuple[list[str], str | None]:
    args = list(spec.path)
    for param in spec.params:
        key = f"p__{param.name}"
        raw = data.get(key, "")
        value = raw.strip()
        if param.is_flag:
            if value:
                if param.option:
                    args.append(param.option)
            continue
        if param.kind == "argument":
            if not value and param.name == "agent_name" and agent_context:
                value = agent_context
            if not value and param.required:
                return [], f"Missing required argument: {param.name}"
            if value:
                args.append(value)
            continue
        if value:
            if param.option:
                args.extend([param.option, value])
            else:
                args.append(value)
    return args, None


class _UIHandler(BaseHTTPRequestHandler):
    root_dir: Path
    notice: str = ""
    detail: str = ""
    agent_context: str = ""

    def _send_html(self, content: str, status: int = 200) -> None:
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, path: str) -> None:
        self.send_response(303)
        self.send_header("Location", path)
        self.end_headers()

    def _form_data(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)
        return {k: (v[0] if v else "") for k, v in parsed.items()}

    def _set_notice(self, notice: str, detail: str = "") -> None:
        type(self).notice = notice
        type(self).detail = detail

    def _consume_notice(self) -> tuple[str, str]:
        notice = type(self).notice
        detail = type(self).detail
        type(self).notice = ""
        type(self).detail = ""
        return notice, detail

    def _agent_exists(self, name: str) -> bool:
        return name in _list_agent_names(self.root_dir)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path

            if path == "/api/dashboard":
                self._send_json(_collect_dashboard_data_cached(self.root_dir))
                return

            if path.startswith("/api/agent/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) == 3:
                    name = unquote(parts[2])
                    if self._agent_exists(name):
                        self._send_json(_collect_agent_data_cached(self.root_dir, name))
                        return
                    self._send_json({"error": "agent not found"}, status=404)
                    return

            notice, detail = self._consume_notice()
            if path == "/":
                self._send_html(_render_dashboard(self.root_dir, notice, detail))
                return
            if path == "/agents":
                self._send_html(_render_agents(self.root_dir, notice, detail))
                return
            if path == "/swarm":
                self._send_html(_render_swarm(self.root_dir, notice, detail))
                return
            if path == "/activity":
                self._send_html(_render_activity(self.root_dir, notice, detail))
                return
            if path == "/models":
                self._send_html(_render_models(notice, detail))
                return
            if path == "/operations":
                self._send_html(_render_operations(self.root_dir, notice, detail, self.agent_context or None))
                return

            if path.startswith("/agent/"):
                parts = [unquote(p) for p in path.split("/") if p]
                if len(parts) >= 2:
                    name = parts[1]
                    if not self._agent_exists(name):
                        self._send_html(_layout("Not found", "<h1>Agent not found</h1>", notice, detail, active_nav="agents"), status=404)
                        return
                    if len(parts) == 2:
                        self._send_html(_render_agent_overview(self.root_dir, name, notice, detail))
                        return
                    if len(parts) == 3 and parts[2] == "runs":
                        self._send_html(_render_agent_runs(self.root_dir, name, notice, detail))
                        return
                    if len(parts) == 3 and parts[2] == "patches":
                        self._send_html(_render_agent_patches(self.root_dir, name, notice, detail))
                        return
                    if len(parts) == 4 and parts[2] == "patches":
                        self._send_html(_render_patch_detail(self.root_dir, name, parts[3], notice, detail))
                        return
                    if len(parts) == 3 and parts[2] == "memory":
                        self._send_html(_render_agent_memory(self.root_dir, name, notice, detail))
                        return
                    if len(parts) == 3 and parts[2] == "inbox":
                        self._send_html(_render_agent_inbox(self.root_dir, name, notice, detail))
                        return
                    if len(parts) == 3 and parts[2] == "config":
                        self._send_html(_render_agent_config(self.root_dir, name, notice, detail))
                        return

            self._send_html(_layout("Not found", "<h1>Not found</h1>", notice, detail, active_nav="dashboard"), status=404)
        except OSError as exc:
            self._send_html(
                _layout(
                    "Server Busy",
                    "<h1>Server Busy</h1><p class='muted'>Too many open files; retry in a moment.</p>",
                    "Temporary resource limit hit.",
                    str(exc),
                    active_nav="dashboard",
                ),
                status=503,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            data = self._form_data()

            if path == "/swarm/start-all":
                notice, detail = _start_all_services(self.root_dir)
                self._set_notice(notice, detail)
                _invalidate_cache()
                self._redirect("/swarm")
                return
            if path == "/swarm/stop-all":
                notice, detail = _stop_all_services(self.root_dir)
                self._set_notice(notice, detail)
                _invalidate_cache()
                self._redirect("/swarm")
                return
            if path == "/swarm/run":
                coordinator = data.get("coordinator", "").strip()
                if not coordinator:
                    self._set_notice("Coordinator agent is required.")
                elif not self._agent_exists(coordinator):
                    self._set_notice("Coordinator agent not found.")
                else:
                    notice, detail = _run_swarm_prompt(coordinator, data.get("prompt", ""))
                    self._set_notice(notice, detail)
                _invalidate_cache()
                self._redirect("/swarm")
                return

            if path == "/agents/create":
                self._set_notice(
                    _create_agent(
                        self.root_dir,
                        data.get("name", ""),
                        data.get("role", ""),
                        data.get("planner_provider", ""),
                        data.get("responder_provider", ""),
                        data.get("embedding_provider", ""),
                        _resolve_model_form_value(data, "planner_model"),
                        _resolve_model_form_value(data, "responder_model"),
                        _resolve_model_form_value(data, "embedding_model"),
                        data.get("base_url", ""),
                        data.get("fs_root", ""),
                        data.get("approval_mode", ""),
                    )
                )
                _invalidate_cache()
                self._redirect(_redirect_target(data.get("next", ""), "/agents"))
                return
            if path == "/operations/context":
                selected = data.get("agent_context", "").strip()
                if selected and not self._agent_exists(selected):
                    self._set_notice("Selected agent does not exist.")
                else:
                    type(self).agent_context = selected
                    self._set_notice("Agent context updated.", selected or "No default agent context.")
                self._redirect("/operations")
                return
            if path == "/operations/run":
                display = data.get("cmd_path", "").strip()
                agent_context = data.get("agent_context", "").strip() or (self.agent_context or "")
                spec = _find_command_spec(display)
                if spec is None:
                    self._set_notice("Unknown command.")
                    self._redirect("/operations")
                    return
                if not spec.runnable:
                    self._set_notice("Command is interactive-only and cannot run in web UI.")
                    self._redirect("/operations")
                    return
                args, error = _build_command_args(spec, data, agent_context or None)
                if error:
                    self._set_notice(error)
                    self._redirect("/operations")
                    return
                output = _run_cli_args(args, agent_context=agent_context or None)
                self._set_notice(f"Executed: {' '.join(args)}", output)
                self._redirect("/operations")
                return

            if path.startswith("/agent/"):
                parts = [unquote(p) for p in path.split("/") if p]
                if len(parts) >= 2:
                    name = parts[1]
                    if not self._agent_exists(name):
                        self._set_notice("Agent not found.")
                        self._redirect("/agents")
                        return
                    next_path = _redirect_target(data.get("next", ""), f"/agent/{_q(name)}")

                    if len(parts) == 3 and parts[2] == "chat":
                        msg = data.get("message", "").strip()
                        if not msg:
                            self._set_notice("Message is required.")
                        else:
                            out = _run_agent_chat_turn(name, msg)
                            self._set_notice("Agent turn completed.", out)
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "resume-turn":
                        out = _run_agent_chat_turn(name, "resume")
                        self._set_notice("Resume turn sent.", out)
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "resume":
                        self._set_notice(_resume_agent_run(self.root_dir, name))
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "cancel":
                        self._set_notice(_cancel_agent_run(self.root_dir, name))
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "start":
                        self._set_notice(_start_agent_service(self.root_dir, name))
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "stop":
                        self._set_notice(_stop_agent_service(self.root_dir, name))
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 3 and parts[2] == "delete":
                        self._set_notice(_delete_agent(self.root_dir, name))
                        _invalidate_cache()
                        self._redirect(_redirect_target(data.get("next", ""), "/agents"))
                        return

                    if len(parts) == 4 and parts[2] == "commitments" and parts[3] == "add":
                        self._set_notice(
                            _add_commitment(
                                self.root_dir,
                                name,
                                data.get("description", ""),
                                data.get("priority", ""),
                                data.get("due_date", ""),
                            )
                        )
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 4 and parts[2] == "commitments" and parts[3] == "update":
                        self._set_notice(
                            _update_commitment(
                                self.root_dir,
                                name,
                                data.get("commitment_id", "").strip(),
                                data.get("status", ""),
                                data.get("priority", ""),
                                data.get("due_date", ""),
                                data.get("description", ""),
                            )
                        )
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 4 and parts[2] == "config" and parts[3] == "save":
                        self._set_notice(_save_agent_config_from_form(self.root_dir, name, data))
                        _invalidate_cache(name)
                        self._redirect(next_path)
                        return

                    if len(parts) == 5 and parts[2] == "patches":
                        proposal_id = parts[3]
                        action = parts[4]
                        if action == "test":
                            self._set_notice("Patch tests executed.", _test_patch(self.root_dir, name, proposal_id))
                        elif action == "apply":
                            self._set_notice(_apply_patch(self.root_dir, name, proposal_id))
                        elif action == "reject":
                            self._set_notice(_reject_patch(self.root_dir, name, proposal_id))
                        elif action == "rollback":
                            self._set_notice(_rollback_patch(self.root_dir, name, proposal_id))
                        else:
                            self._set_notice("Unknown patch action.")
                        _invalidate_cache(name)
                        self._redirect(_redirect_target(data.get("next", ""), f"/agent/{_q(name)}/patches/{_q(proposal_id)}"))
                        return

                    if len(parts) == 5 and parts[2] == "inbox" and parts[4] == "read":
                        msg_id = parts[3]
                        db = _init_db(_paths(self.root_dir, name).db_path)
                        db.mark_agent_message(msg_id, "read")
                        self._set_notice("Message marked as read.")
                        _invalidate_cache(name)
                        self._redirect(_redirect_target(data.get("next", ""), f"/agent/{_q(name)}/inbox"))
                        return

                self._set_notice("Unknown action.")
                self._redirect("/")
        except OSError as exc:
            self._set_notice("Temporary resource limit hit.", str(exc))
            self._redirect("/")

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def run_web_ui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    root = Path.home() / ".tali"
    root.mkdir(parents=True, exist_ok=True)

    class Handler(_UIHandler):
        root_dir = root
        notice = ""
        detail = ""
        agent_context = ""

    class _UIServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = _UIServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Tali Web UI running at {url}")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
