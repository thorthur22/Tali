from __future__ import annotations

import json
import os
import platform
import subprocess
import re
import shlex
import shutil
import time
from urllib.parse import urlparse
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
import httpx
from rich.markup import escape
from dataclasses import replace

from tali.config import (
    AppConfig,
    AutonomyConfig,
    EmbeddingSettings,
    GuardrailConfig,
    LLMSettings,
    RetrievalConfig,
    ToolSettings,
    TaskRunnerConfig,
    Paths,
    SharedSettings,
    load_config,
    load_paths,
    load_shared_settings,
    save_config,
    save_shared_settings,
    infer_model_strengths,
)
from tali.agent_identity import resolve_agent, validate_agent_name
from tali.a2a import A2AClient, A2APoller, AgentProfile
from tali.a2a_registry import AgentRecord, Registry
from tali.db import Database
from tali.episode import build_episode, build_prompt
from tali.embeddings import OllamaEmbeddingClient, OpenAIEmbeddingClient
from tali.guardrails import GuardrailResult, Guardrails
from tali.memory_ingest import stage_episode_fact
from tali.llm import OllamaClient, OpenAIClient
from tali.retrieval import Retriever
from tali.self_care import SleepScheduler, resolve_contradiction_answer, resolve_staged_items
from tali.snapshots import create_snapshot, diff_snapshot, list_snapshots, rollback_snapshot
from tali.vector_index import VectorIndex
from tali.approvals import ApprovalManager
from tali.task_runner import TaskRunner, TaskRunnerSettings
from tali.tasking import build_swarm_context
from tali.hooks.core import HookManager
from tali.idle import IdleScheduler
from tali.knowledge_sources import KnowledgeSourceRegistry, LocalFileSource
from tali.questions import mark_question_asked, resolve_answered_question, select_question_to_ask
from tali.patches import apply_patch, reverse_patch, run_patch_tests
from tali.worktrees import ensure_agent_worktree, resolve_main_repo_root, remove_agent_worktree
from tali.tools.registry import build_default_registry
from tali.tools.policy import ToolPolicy
from tali.tools.runner import ToolRunner
from tali.model_catalog import list_models as list_catalog_models, download_model as download_catalog_model
from tali.run_logs import append_run_log, read_recent_logs, latest_metrics_for_run
from tali.cli_format import (
    format_inbox,
    format_patches_list,
    format_patch_detail,
    format_run_status,
    format_run_show,
    format_run_list,
    format_timeline,
    format_logs,
    format_agent_name,
    format_agent_list,
    format_commitments,
    format_facts,
    format_skills,
    format_doctor,
    format_preferences,
    format_memory_search,
    format_config,
    format_help_guide,
)

app = typer.Typer(help="Tali agent CLI \u2014 run 'tali --install-completion' for shell tab-completion.")
agent_app = typer.Typer(help="Manage agents")
app.add_typer(agent_app, name="agent")
run_app = typer.Typer(help="Manage task runs")
agent_app.add_typer(run_app, name="run")
patch_app = typer.Typer(help="Manage patch proposals")
agent_app.add_typer(patch_app, name="patches")
config_app = typer.Typer(help="View and edit agent configuration.")
agent_app.add_typer(config_app, name="config")
memory_app = typer.Typer(help="Search and explore agent memory.")
agent_app.add_typer(memory_app, name="memory")
pref_app = typer.Typer(help="Manage user preferences.")
agent_app.add_typer(pref_app, name="preferences")


def _bootstrap_first_agent(start_chat: bool = True) -> None:
    root = Path.home() / ".tali"
    if not _list_agent_dirs(root):
        typer.echo("No agents found. Creating your first agent.")
        create_agent()
    if start_chat:
        chat()


def _management_shell() -> None:
    console = Console()
    root = Path.home() / ".tali"
    agent_dirs = _list_agent_dirs(root)
    console.print(
        "[bold magenta]T A L I[/bold magenta]  "
        "[dim]multi-agent manager[/dim]"
    )
    console.print("[dim]Type commands without the 'tali' prefix. (exit to quit)[/dim]")
    console.print("")
    console.print("[bold]Common commands[/bold]")
    console.print("  agent list                 list agents")
    console.print("  agent create               create a new agent")
    console.print("  agent chat <name>          enter chat with an agent")
    console.print("  agent delete <name>        delete an agent")
    console.print("  agent run status           show active run status")
    console.print("  agent patches list         list patch proposals")
    console.print("  models list                list available models")
    console.print("")
    if agent_dirs:
        console.print("[bold]Recommended next:[/bold] run `agent list` then `agent chat <name>`")
    else:
        console.print("[bold]Recommended next:[/bold] run `agent create` to get started")
    console.print("")
    while True:
        try:
            raw = console.input("[bold cyan]tali>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        line = raw.strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        args = shlex.split(line)
        if args and args[0].lower() == "tali":
            args = args[1:]
        if not args:
            continue
        try:
            app(prog_name="tali", args=args)
        except SystemExit:
            continue


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _bootstrap_first_agent(start_chat=False)
    _sync_all_agent_worktrees()
    _management_shell()


def _init_db(db_path: Path) -> Database:
    db = Database(db_path)
    db.initialize()
    return db


def _sync_all_agent_worktrees() -> None:
    root = Path.home() / ".tali"
    for agent_home in _list_agent_dirs(root):
        paths = load_paths(root, agent_home.name)
        _ensure_agent_worktree(paths)


def _resolve_paths(agent_name: str | None = None) -> tuple[Paths, AppConfig | None]:
    if agent_name:
        root = Path.home() / ".tali"
        agent_home = root / agent_name
        if not agent_home.exists() or not (agent_home / "config.json").exists():
            typer.echo(f"Agent not found: {agent_name}")
            raise typer.Exit(code=1)
        paths = load_paths(root, agent_name)
        config_path = agent_home / "config.json"
        config = load_config(config_path) if config_path.exists() else None
        return paths, config
    root, resolved_name, config = resolve_agent(
        prompt_fn=typer.prompt, allow_create_config=False
    )
    paths = load_paths(root, resolved_name)
    return paths, config


def _list_agent_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    agent_dirs: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name == "shared":
            continue
        if (path / "config.json").exists():
            agent_dirs.append(path)
    return agent_dirs


def _load_or_raise_config(paths: Paths, existing: AppConfig | None = None) -> AppConfig:
    if existing:
        config = existing
    else:
        if not paths.config_path.exists():
            typer.echo("Config not found. Run `agent create` to configure this agent.")
            raise typer.Exit(code=1)
        config = load_config(paths.config_path)
    if not config.agent_name or not config.agent_id or config.agent_name != paths.agent_name:
        updated = AppConfig(
            agent_id=config.agent_id or str(uuid4()),
            agent_name=paths.agent_name,
            created_at=config.created_at or datetime.utcnow().isoformat(),
            role_description=config.role_description or "",
            capabilities=config.capabilities or [],
            planner_llm=config.planner_llm,
            responder_llm=config.responder_llm,
            embeddings=config.embeddings,
            tools=config.tools,
            task_runner=config.task_runner,
        )
        save_config(paths.config_path, updated)
        config = updated
    shared_path = paths.shared_home / "config.json"
    shared = load_shared_settings(shared_path)
    planner_llm = config.planner_llm
    responder_llm = config.responder_llm
    embeddings = config.embeddings
    tools = config.tools
    task_runner = config.task_runner
    if shared:
        planner_llm = planner_llm or shared.planner_llm
        responder_llm = responder_llm or shared.responder_llm
        embeddings = embeddings or shared.embeddings
        tools = tools or shared.tools
        task_runner = task_runner or shared.task_runner
    config = AppConfig(
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        created_at=config.created_at,
        role_description=config.role_description,
        capabilities=config.capabilities,
        planner_llm=planner_llm,
        responder_llm=responder_llm,
        embeddings=embeddings,
        tools=tools,
        task_runner=task_runner,
    )
    if not config.planner_llm or not config.responder_llm or not config.embeddings or not config.tools:
        typer.echo(
            "Missing planner/responder LLM, embeddings, or tool settings. Run `agent create`."
        )
        raise typer.Exit(code=1)
    return config


def _build_llm_client(settings: LLMSettings):
    if settings.provider == "openai":
        if not settings.api_key and not _is_local_base_url(settings.base_url):
            typer.echo("OpenAI API key is required for non-local endpoints.")
            raise typer.Exit(code=1)
        return OpenAIClient(base_url=settings.base_url, api_key=settings.api_key, model=settings.model)
    if settings.provider == "ollama":
        return OllamaClient(base_url=settings.base_url, model=settings.model)
    typer.echo(f"Unsupported LLM provider: {settings.provider}")
    raise typer.Exit(code=1)


def _build_embedder(settings: EmbeddingSettings):
    if settings.provider == "openai":
        if not settings.api_key and not _is_local_base_url(settings.base_url):
            typer.echo("OpenAI API key is required for embeddings on non-local endpoints.")
            raise typer.Exit(code=1)
        return OpenAIEmbeddingClient(
            base_url=settings.base_url, api_key=settings.api_key, model=settings.model
        )
    if settings.provider == "ollama":
        return OllamaEmbeddingClient(base_url=settings.base_url, model=settings.model)
    typer.echo(f"Unsupported embedding provider: {settings.provider}")
    raise typer.Exit(code=1)


def _ensure_dependencies() -> None:
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        subprocess.run(["python", "-m", "pip", "install", "hnswlib"], check=True)
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run(["python", "-m", "pip", "install", "pytest"], check=True)


def _ensure_agent_worktree(paths: Paths) -> Path | None:
    repo_root = resolve_main_repo_root(Path(__file__).resolve())
    if repo_root is None:
        return None
    code_dir, status = ensure_agent_worktree(paths, repo_root)
    if status.message:
        typer.echo(status.message)
    return code_dir


def _should_spawn_agent_terminal() -> bool:
    if os.environ.get("TALI_AGENT_SPAWNED") == "1":
        return False
    if os.environ.get("TALI_HEADLESS") == "1":
        return False
    return True


def _spawn_agent_terminal(agent_name: str, code_dir: Path) -> bool:
    env = os.environ.copy()
    env["TALI_AGENT_SPAWNED"] = "1"
    system = platform.system()
    if system == "Windows":
        code_dir_ps = str(code_dir).replace("'", "''")
        agent_ps = agent_name.replace("'", "''")
        command = (
            f"Set-Location -LiteralPath '{code_dir_ps}'; "
            "$env:TALI_AGENT_SPAWNED='1'; "
            f"tali agent chat '{agent_ps}'"
        )
        subprocess.Popen(
            ["cmd", "/c", "start", "powershell", "-NoExit", "-Command", command],
            env=env,
        )
        return True
    if system == "Darwin":
        command = (
            f"cd {shlex.quote(str(code_dir))}; "
            f"TALI_AGENT_SPAWNED=1 tali agent chat {shlex.quote(agent_name)}"
        )
        escaped = command.replace('"', '\\"')
        subprocess.Popen(
            ["osascript", "-e", f'tell application "Terminal" to do script "{escaped}"'],
            env=env,
        )
        return True
    shell_command = (
        f"cd {shlex.quote(str(code_dir))} && "
        f"TALI_AGENT_SPAWNED=1 tali agent chat {shlex.quote(agent_name)}"
    )
    for term in ["x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal", "xterm"]:
        if not shutil.which(term):
            continue
        if term in {"gnome-terminal", "x-terminal-emulator", "xterm"}:
            subprocess.Popen([term, "-e", "bash", "-lc", shell_command], env=env)
            return True
        if term == "konsole":
            subprocess.Popen([term, "-e", "bash", "-lc", shell_command], env=env)
            return True
        if term == "xfce4-terminal":
            subprocess.Popen([term, "-e", f"bash -lc '{shell_command}'"], env=env)
            return True
    return False


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def _validate_provider_connectivity(provider: str, base_url: str, api_key: str | None) -> bool:
    try:
        if provider == "ollama":
            url = base_url.rstrip("/") + "/api/tags"
            response = httpx.get(url, timeout=5.0)
            return response.status_code < 400
        url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = httpx.get(url, headers=headers, timeout=5.0)
        return response.status_code < 400
    except Exception:
        return False


def _list_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].lower()
    if "name" in header:
        lines = lines[1:]
    models: list[str] = []
    for line in lines:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _prompt_ollama_model(label: str, default: str) -> str:
    models = _list_ollama_models()
    if models:
        typer.echo("Available Ollama models:")
        for idx, name in enumerate(models, start=1):
            typer.echo(f"  {idx}. {name}")
        choice = typer.prompt(f"{label} (name or number)", default=default)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(models):
                return models[index - 1]
        return choice
    return typer.prompt(label, default=default)


def _prompt_ollama_download(model: str) -> None:
    installed = set(_list_ollama_models())
    if model in installed:
        typer.echo(f"Ollama model '{model}' is already installed.")
        return
    if typer.confirm(f"Download Ollama model '{model}' now?", default=True):
        _pull_ollama_model(model)


def _format_agent_label(agent_name: str) -> str:
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", agent_name) if part]
    if not parts:
        return "Agent"
    formatted: list[str] = []
    for part in parts:
        if part.isdigit():
            formatted.append(part)
        else:
            formatted.append(part[:1].upper() + part[1:].lower())
    return "".join(formatted)


def _prompt_llm_settings(
    label: str,
    default_provider: str,
    default_base_url: str,
    default_model: str,
    models_dir: Path,
) -> LLMSettings:
    provider = typer.prompt(f"{label} provider (openai/ollama)", default=default_provider)
    if provider not in {"openai", "ollama"}:
        typer.echo("Provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if provider == "openai":
        base_url, auth_method = _prompt_openai_endpoint(label, default_base_url)
        if auth_method == "local":
            model = _prompt_openai_local_model(models_dir)
            api_key = typer.prompt(
                f"{label} OpenAI API key (leave blank for local server)",
                default="",
                show_default=False,
                hide_input=True,
            )
            if api_key == "":
                api_key = None
        else:
            model_default = "gpt-5-codex" if auth_method == "codex" else default_model
            model = typer.prompt(f"{label} model", default=model_default)
            if auth_method == "codex":
                api_key = _ensure_codex_access_token(label, force_login=False)
            else:
                api_key = typer.prompt(f"{label} OpenAI API key", hide_input=True)
        return LLMSettings(provider=provider, model=model, base_url=base_url, api_key=api_key)
    model = _prompt_ollama_model(f"{label} model", default=default_model)
    base_url = typer.prompt(f"{label} base URL", default="http://localhost:11434")
    _prompt_ollama_download(model)
    return LLMSettings(provider=provider, model=model, base_url=base_url, api_key=None)


def _prompt_openai_endpoint(label: str, default_base_url: str) -> tuple[str, str]:
    typer.echo("OpenAI connection options:")
    typer.echo("  1. Local server (localhost) - use local OpenAI-compatible server.")
    typer.echo("  2. OpenAI API key - standard API access with your key.")
    typer.echo("  3. OpenAI Codex login - use ChatGPT web login token from `codex login`.")
    choice = typer.prompt(f"{label} OpenAI setup (1/2/3)", default="2")
    local_default = (
        default_base_url if _is_local_base_url(default_base_url) else "http://localhost:8000/v1"
    )
    if choice.strip() == "1":
        base_url = typer.prompt(f"{label} base URL", default=local_default)
        return base_url, "local"
    if choice.strip() == "2":
        base_url = typer.prompt(
            f"{label} base URL", default="https://api.openai.com/v1"
        )
        return base_url, "api_key"
    if choice.strip() == "3":
        base_url = typer.prompt(
            f"{label} base URL", default="https://api.openai.com/v1"
        )
        return base_url, "codex"
    typer.echo("Select 1, 2, or 3.")
    raise typer.Exit(code=1)


def _ensure_codex_access_token(label: str, force_login: bool = False) -> str:
    token = _load_codex_access_token()
    if token and not force_login:
        return token
    if token and force_login:
        if not typer.confirm(
            f"{label} already has a Codex token. Re-run ChatGPT login anyway?",
            default=False,
        ):
            return token
    if not shutil.which("codex"):
        typer.echo(
            f"{label} Codex auth not found and `codex` CLI is missing. Install Codex CLI and run `codex login`."
        )
        raise typer.Exit(code=1)
    typer.echo(f"{label} Codex login starting (browser-based ChatGPT login).")
    try:
        subprocess.run(["codex", "login"], check=True)
    except FileNotFoundError:
        typer.echo(
            "Could not find the `codex` CLI. Install it or ensure it's on PATH."
        )
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Codex login failed: {exc}")
        raise typer.Exit(code=1) from exc
    token = _load_codex_access_token()
    if token:
        return token
    typer.echo("Codex login completed but access token was not found.")
    raise typer.Exit(code=1)


def _load_codex_access_token() -> str | None:
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        payload = json.loads(auth_path.read_text())
    except json.JSONDecodeError:
        return None
    return _find_access_token(payload)


def _find_access_token(payload: object) -> str | None:
    if isinstance(payload, dict):
        token = payload.get("access_token")
        if isinstance(token, str) and token.strip():
            return token
        for value in payload.values():
            found = _find_access_token(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_access_token(item)
            if found:
                return found
    return None


def _pull_ollama_model(model: str) -> None:
    typer.echo(f"Downloading Ollama model: {model}")
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        typer.echo("Ollama CLI not found. Install Ollama to download models.")
        raise typer.Exit(code=1) from exc

    assert process.stdout is not None
    for line in process.stdout:
        typer.echo(line.rstrip())
    code = process.wait()
    if code != 0:
        raise typer.Exit(code=code)


def _prompt_unique_agent_name(root: Path) -> str:
    agent_name = typer.prompt("Choose a name for this agent")
    while True:
        if not validate_agent_name(agent_name):
            agent_name = typer.prompt(
                "Name must be lowercase [a-z0-9-_], 2-32 chars. Try again"
            )
            continue
        if (root / agent_name).exists():
            agent_name = typer.prompt("That name exists. Choose another")
            continue
        return agent_name


def _prompt_role_description() -> str:
    roles = [
        ("assistant", "General-purpose assistant."),
        ("developer", "Software engineer focused on implementation and debugging."),
        ("researcher", "Research-focused agent for synthesis and analysis."),
        ("ops", "Operations-focused agent for reliability and tooling."),
        ("custom", "Write a custom role description."),
    ]
    typer.echo("Role presets:")
    for idx, (name, description) in enumerate(roles, start=1):
        typer.echo(f"  {idx}. {name} - {description}")
    choice = typer.prompt("Select a role (number or name)", default="1")
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(roles):
            name, description = roles[index - 1]
            if name == "custom":
                return typer.prompt("Role description")
            return f"{name}: {description}"
    normalized = choice.strip().lower()
    for name, description in roles:
        if normalized == name:
            if name == "custom":
                return typer.prompt("Role description")
            return f"{name}: {description}"
    return choice


def _prompt_openai_local_model(models_dir: Path) -> str:
    catalog = list_catalog_models()
    if not catalog:
        return typer.prompt("LLM model")
    typer.echo("Curated local models (GGUF):")
    for idx, entry in enumerate(catalog, start=1):
        typer.echo(f"  {idx}. {entry.label} ({entry.repo_id})")
    choice = typer.prompt("Select a model (number or 'custom')", default="1")
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(catalog):
            entry = catalog[index - 1]
            models_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Downloading {entry.label} to {models_dir} ...")
            target_path = download_catalog_model(entry.repo_id, entry.pattern, models_dir)
            typer.echo(f"Downloaded model file: {target_path}")
            return target_path.name
    if choice.strip().lower() in {"custom", "other"}:
        return typer.prompt("LLM model")
    return choice


def _ensure_agent_dirs(paths: Paths) -> None:
    paths.agent_home.mkdir(parents=True, exist_ok=True)
    paths.vector_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    paths.sleep_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.inbox_dir.mkdir(parents=True, exist_ok=True)
    paths.outbox_dir.mkdir(parents=True, exist_ok=True)
    paths.shared_home.mkdir(parents=True, exist_ok=True)
    (paths.shared_home / "locks").mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)


@agent_app.command("create")
def create_agent() -> None:
    """Create a new agent with role and LLM settings.

    Walks through an interactive setup to configure the agent's name,
    role, LLM provider, model, and embedding settings.
    """
    root = Path.home() / ".tali"
    root.mkdir(parents=True, exist_ok=True)
    (root / "shared").mkdir(parents=True, exist_ok=True)
    agent_name = _prompt_unique_agent_name(root)
    role_description = _prompt_role_description()
    paths = load_paths(root, agent_name)
    planner_provider = typer.prompt("Planner LLM provider (openai/ollama)", default="openai")
    if planner_provider not in {"openai", "ollama"}:
        typer.echo("Provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if planner_provider == "openai":
        planner_base_url, planner_auth = _prompt_openai_endpoint(
            "Planner", "http://localhost:8000/v1"
        )
        if planner_auth == "local":
            planner_model = _prompt_openai_local_model(paths.models_dir)
        else:
            planner_default_model = (
                "gpt-5-codex" if planner_auth == "codex" else "gpt-4o-mini"
            )
            planner_model = typer.prompt("Planner LLM model", default=planner_default_model)
    else:
        planner_model = _prompt_ollama_model("Planner LLM model", default="llama3")
        planner_base_url = typer.prompt(
            "Planner LLM base URL", default="http://localhost:11434"
        )
    planner_api_key = None
    if planner_provider == "openai":
        if planner_auth == "local":
            planner_api_key = typer.prompt(
                "Planner OpenAI API key (leave blank for local server)",
                default="",
                show_default=False,
                hide_input=True,
            )
            if planner_api_key == "":
                planner_api_key = None
        else:
            if planner_auth == "codex":
                planner_api_key = _ensure_codex_access_token("Planner", force_login=False)
            else:
                planner_api_key = typer.prompt("Planner OpenAI API key", hide_input=True)
    else:
        _prompt_ollama_download(planner_model)

    planner_llm = LLMSettings(
        provider=planner_provider,
        model=planner_model,
        base_url=planner_base_url,
        api_key=planner_api_key,
    )
    if typer.confirm("Use the same LLM settings for the responder?", default=True):
        responder_llm = planner_llm
    else:
        responder_llm = _prompt_llm_settings(
            "Responder LLM",
            default_provider=planner_provider,
            default_base_url=planner_base_url,
            default_model=planner_model,
            models_dir=paths.models_dir,
        )

    embed_provider = typer.prompt(
        "Embedding provider (openai/ollama)", default=planner_provider
    )
    if embed_provider not in {"openai", "ollama"}:
        typer.echo("Embedding provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if embed_provider == "openai":
        embed_base_url = typer.prompt(
            "Embedding base URL",
            default=planner_base_url
            if embed_provider == planner_provider
            else "http://localhost:8000/v1",
        )
        embed_default_model = (
            "nomic-embed-text:latest" if _is_local_base_url(embed_base_url) else "text-embedding-3-small"
        )
        embed_model = typer.prompt("Embedding model", default=embed_default_model)
    else:
        embed_model = _prompt_ollama_model("Embedding model", default="nomic-embed-text")
        embed_base_url = typer.prompt(
            "Embedding base URL",
            default=planner_base_url
            if embed_provider == planner_provider
            else "http://localhost:11434",
        )
    embed_api_key = None
    if embed_provider == "openai":
        if _is_local_base_url(embed_base_url):
            embed_api_key = typer.prompt(
                "OpenAI API key for embeddings (leave blank for local server)",
                default="",
                show_default=False,
                hide_input=True,
            )
            if embed_api_key == "":
                embed_api_key = None
        else:
            embed_api_key = typer.prompt("OpenAI API key for embeddings", hide_input=True)
    else:
        _prompt_ollama_download(embed_model)
    embed_dim = typer.prompt("Embedding dimension", default="1536")
    typer.echo(
        "File access is scoped by tools.fs_root (default: your user profile). "
        "You can change it to narrow or expand the agent's file access for security."
    )
    config = AppConfig(
        agent_id=str(uuid4()),
        agent_name=agent_name,
        created_at=datetime.utcnow().isoformat(),
        role_description=role_description,
        capabilities=[],
        planner_llm=planner_llm,
        responder_llm=responder_llm,
        embeddings=EmbeddingSettings(
            provider=embed_provider,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
            dim=int(embed_dim),
        ),
        tools=ToolSettings(fs_root=str(Path.home())),
        task_runner=TaskRunnerConfig(),
    )
    save_config(paths.config_path, config)
    _ensure_agent_dirs(paths)
    _ensure_agent_worktree(paths)
    typer.echo(f"Agent created: {paths.agent_home}")


@agent_app.command("chat")
def chat(
    agent_name: str = typer.Argument(
        ..., help="Agent name."
    ),
    message: Optional[str] = typer.Argument(
        None, help="Message to send to the agent."
    ),
    verbose_tools: bool = typer.Option(False, "--verbose-tools", help="Show full tool output."),
    show_plans: bool = typer.Option(False, "--show-plans", help="Show task planning steps."),
) -> None:
    """Start a chat loop or run a single turn if message is provided.

    Opens an interactive session with the agent. Provide a message
    argument to run a single turn, or omit it for a persistent chat loop.
    Type 'exit' or 'quit' to leave the chat.
    """
    if isinstance(message, typer.models.ArgumentInfo):
        message = None
    paths, existing_config = _resolve_paths(agent_name)
    code_dir = _ensure_agent_worktree(paths)
    if message is None and code_dir and _should_spawn_agent_terminal():
        spawned = _spawn_agent_terminal(paths.agent_name, code_dir)
        if spawned:
            return
    _ensure_dependencies()
    config = _load_or_raise_config(paths, existing_config)
    db = _init_db(paths.db_path)
    planner_settings = config.planner_llm
    responder_settings = config.responder_llm
    if not planner_settings or not responder_settings:
        typer.echo("Missing planner/responder LLM settings. Run `agent create`.")
        raise typer.Exit(code=1)
    planner_llm = _build_llm_client(planner_settings)
    responder_llm = (
        planner_llm
        if responder_settings == planner_settings
        else _build_llm_client(responder_settings)
    )
    embedder = _build_embedder(config.embeddings)
    vector_index = VectorIndex(paths.vector_dir / "memory.index", config.embeddings.dim, embedder)
    retriever = Retriever(db, RetrievalConfig(), vector_index=vector_index)
    guardrails = Guardrails(GuardrailConfig())
    hooks_dir = Path(__file__).resolve().parent / "hooks"
    hook_manager = HookManager(db=db, hooks_dir=hooks_dir)
    hook_manager.load_hooks()
    scheduler = SleepScheduler(
        paths.data_dir, db, planner_llm, vector_index, hook_manager=hook_manager
    )
    scheduler.start()
    sources = KnowledgeSourceRegistry()
    sources.register(LocalFileSource(paths.agent_home))
    autonomy_config = config.autonomy if config.autonomy else None
    idle_scheduler = IdleScheduler(
        paths.data_dir,
        db,
        planner_llm,
        sources,
        hook_manager=hook_manager,
        status_fn=lambda msg: console.print(f"[dim]{escape(msg)}[/dim]"),
        autonomy_config=autonomy_config,
    )
    idle_scheduler.start()
    console = Console()
    agent_label = _format_agent_label(config.agent_name)
    tool_settings = config.tools if isinstance(config.tools, ToolSettings) else ToolSettings()
    if tool_settings.approval_mode not in {"prompt", "auto_approve_safe", "deny"}:
        tool_settings = replace(tool_settings, approval_mode="prompt")
    if tool_settings.fs_root is None or not str(tool_settings.fs_root).strip():
        tool_settings = replace(tool_settings, fs_root=str(Path.home()))
    registry = build_default_registry(paths, tool_settings)
    policy = ToolPolicy(tool_settings, registry, paths)
    approvals = ApprovalManager(mode=tool_settings.approval_mode)
    runner = ToolRunner(registry, policy, approvals, tool_settings, paths)
    tool_descriptions = registry.describe_tools()
    registry = Registry(paths.shared_home / "registry.json")
    agent_record = AgentRecord(
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        home=str(paths.agent_home),
        status="active",
        last_seen=datetime.utcnow().isoformat(),
        capabilities=config.capabilities,
    )
    registry.upsert(agent_record)
    a2a_client = A2AClient(
        db=db,
        profile=AgentProfile(
            agent_id=config.agent_id,
            agent_name=config.agent_name,
            agent_home=paths.agent_home,
            shared_home=paths.shared_home,
            capabilities=config.capabilities,
        ),
    )
    if a2a_client.secret is None:
        console.print("[dim]A2A: shared secret missing; messages are unsigned.[/dim]")
    planner_strengths = ", ".join(infer_model_strengths(planner_settings))
    responder_strengths = ", ".join(infer_model_strengths(responder_settings))
    agent_context = "\n".join(
        [
            f"Agent name: {config.agent_name}",
            f"Role: {config.role_description or 'assistant'}",
            f"Agent home: {paths.agent_home}",
            f"Config path: {paths.config_path}",
            f"Background model: {planner_settings.model}",
            f"Planner model: {planner_settings.model}",
            f"Responder model: {responder_settings.model}",
            f"Tool fs_root: {tool_settings.fs_root}",
            f"OS: {platform.system()} {platform.release()}",
            "The agent is distinct from the LLM model.",
            f"You are the LLM planner; the {agent_label} agent executes tool calls you request.",
            "Use config.json (not config.txt).",
            "Do not use tools for greetings or small talk.",
            "Model capabilities:",
            f"- Planner ({planner_settings.model}): {planner_strengths}",
            f"- Responder ({responder_settings.model}): {responder_strengths}",
            "When choosing 'respond', provide detailed factual content in your message so the responder can phrase it well for the user.",
        ]
    )
    swarm_info = build_swarm_context(a2a_client, config.agent_name)
    if swarm_info:
        agent_context += "\n\nTeam resources:\n" + swarm_info
    # Append a summary of long‑term memory into the agent context. This
    # provides the planner with the most salient facts at a glance. If
    # summarization fails, we simply proceed without it.
    try:
        from tali.memory_manager import MemoryManager  # type: ignore

        mem_mgr = MemoryManager(db, vector_index)
        summary = mem_mgr.summarize_memory(limit=5)
        if summary:
            agent_context += "\n\nLong-term memory:\n" + summary
    except Exception:
        pass
    task_runner_config = (
        config.task_runner if isinstance(config.task_runner, TaskRunnerConfig) else None
    )
    task_runner_settings = (
        TaskRunnerSettings(
            max_tasks_per_turn=task_runner_config.max_tasks_per_turn,
            max_llm_calls_per_task=task_runner_config.max_llm_calls_per_task,
            max_tool_calls_per_task=task_runner_config.max_tool_calls_per_task,
            max_total_llm_calls_per_run_per_turn=(
                task_runner_config.max_total_llm_calls_per_run_per_turn
            ),
            max_total_steps_per_turn=task_runner_config.max_total_steps_per_turn,
        )
        if task_runner_config
        else None
    )
    task_runner = TaskRunner(
        db=db,
        llm=planner_llm,
        planner_llm=planner_llm,
        responder_llm=responder_llm,
        retriever=retriever,
        guardrails=guardrails,
        tool_runner=runner,
        tool_descriptions=tool_descriptions,
        hook_manager=hook_manager,
        a2a_client=a2a_client,
        agent_context=agent_context,
        status_fn=(lambda msg: console.print(f"[dim]{escape(msg)}[/dim]")),
        settings=task_runner_settings,
        responder_strengths=responder_strengths,
    )
    idle_scheduler.set_task_runner(task_runner)
    poller = A2APoller(
        db=db,
        client=a2a_client,
        task_runner=task_runner,
        prompt_fn=console.input,
        status_fn=lambda msg: console.print(f"[dim]{escape(msg)}[/dim]"),
    )
    poller.start()

    def _is_small_talk(text: str) -> bool:
        lowered = text.strip().lower()
        if re.match(r"^(hi|hello|hey|yo|oi)(\b|$)", lowered):
            return True
        starters = [
            "how are you",
            "how you doing",
            "hows it going",
            "how's it going",
            "what's up",
            "whats up",
        ]
        if any(lowered.startswith(starter) for starter in starters):
            return True
        phrases = [
            "thanks",
            "thank you",
            "appreciate it",
            "good job",
            "good work",
            "great job",
            "great work",
            "nice work",
            "well done",
        ]
        return any(phrase in lowered for phrase in phrases)

    def _is_name_question(text: str) -> bool:
        lowered = text.strip().lower()
        triggers = [
            "what's your name",
            "what is your name",
            "who are you",
            "your name",
            "hi burt",
            "hello burt",
            "hey burt",
        ]
        return any(trigger in lowered for trigger in triggers)

    def _is_open_ended(text: str) -> bool:
        lowered = text.strip().lower()
        starters = [
            "what do you think",
            "what's your take",
            "how do you feel",
            "should i",
            "would you",
            "why",
        ]
        return any(lowered.startswith(starter) for starter in starters) or lowered.endswith("?")

    def _is_action_request(text: str) -> bool:
        lowered = text.strip().lower()
        keywords = [
            "create",
            "make",
            "write",
            "save",
            "add",
            "generate",
            "build",
            "compose",
            "draft",
            "produce",
            "open",
            "read",
            "view",
            "show",
            "display",
            "search",
            "find",
            "fetch",
            "download",
            "upload",
            "edit",
            "change",
            "update",
            "delete",
            "remove",
            "run",
            "execute",
            "list",
            "read file",
            "open file",
            "search",
            "fetch",
            "download",
            "apply",
            "fix",
            "refactor",
            "implement",
            "rename",
            "move",
            "copy",
            "replace",
            "append",
            "insert",
            "modify",
            "configure",
            "set up",
            "setup",
            "install",
            "uninstall",
            "upgrade",
            "downgrade",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _is_short_confirmation(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {
            "yes",
            "y",
            "yep",
            "yeah",
            "sure",
            "correct",
            "no",
            "n",
            "nope",
            "nah",
            "negative",
        }

    def _is_related_to_text(user_input: str, topic: str) -> bool:
        if not topic or not user_input.strip():
            return False
        prompt = "\n".join(
            [
                "Determine if the user message is related to the topic.",
                "Return only YES or NO.",
                f"Topic: {topic}",
                f"User message: {user_input}",
            ]
        )
        try:
            response = responder_llm.generate(prompt).content.strip().lower()
        except Exception:
            return False
        return response.startswith("y")

    def _should_attempt_staged_resolution(user_input: str) -> bool:
        row = db.fetch_next_staged_item(datetime.utcnow().isoformat())
        if not row:
            return False
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            return False
        if payload.get("awaiting_confirmation") or payload.get("awaiting_clarification"):
            return True
        if row["kind"] == "fact":
            topic = str(payload.get("statement") or "").strip()
        elif row["kind"] == "commitment":
            topic = str(payload.get("description") or "").strip()
        else:
            return False
        return _is_related_to_text(user_input, topic)

    def run_turn(user_input: str) -> None:
        scheduler.update_activity()
        idle_scheduler.update_activity()
        hook_messages = hook_manager.run("on_turn_start", {"user_input": user_input})
        for msg in hook_messages:
            console.print(f"[dim]{escape(msg)}[/dim]")
        pending_question_row = db.fetch_last_asked_question()
        has_pending_question = bool(pending_question_row and pending_question_row["status"] == "asked")
        related_to_pending_question = True
        if pending_question_row:
            if _is_short_confirmation(user_input):
                related_to_pending_question = True
            else:
                related_to_pending_question = _is_related_to_text(
                    user_input, str(pending_question_row["question"])
                )
        def _run_task_runner_turn() -> None:
            if tool_settings.approval_mode == "prompt":
                result = task_runner.run_turn(
                    user_input, prompt_fn=console.input, show_plans=show_plans
                )
            else:
                with console.status(
                    "[bold yellow]Working... (planning tasks and tool calls)[/bold yellow]"
                ):
                    result = task_runner.run_turn(
                        user_input, prompt_fn=console.input, show_plans=show_plans
                    )
            tool_calls_log: list[dict[str, str]] = []
            for record in result.tool_records:
                tool_calls_log.append(
                    {
                        "id": record.id,
                        "name": record.name,
                        "status": record.status,
                        "approval_mode": record.approval_mode or "",
                        "result_ref": record.result_ref or "",
                        "summary": record.result_summary or "",
                    }
                )
                console.print(
                    f"[bold magenta]Tool[/bold magenta] {record.name} {record.id} status={record.status}"
                )
                if record.result_summary:
                    console.print(f"[dim]{escape(record.result_summary)}[/dim]")
                if verbose_tools and record.result_json:
                    console.print(f"[dim]{escape(record.result_json)}[/dim]")
            guardrail = GuardrailResult(safe_output=result.message, flags=[])
            episode = build_episode(
                user_input=user_input,
                guardrail=guardrail,
                tool_calls=tool_calls_log,
                outcome="ok",
                quarantine=0,
            )
            db.insert_episode(
                episode_id=episode.id,
                user_input=episode.user_input,
                agent_output=episode.agent_output,
                tool_calls=episode.tool_calls,
                outcome=episode.outcome,
                quarantine=episode.quarantine,
            )
            stage_episode_fact(db, episode.id, episode.user_input, episode.outcome)
            append_run_log(
                paths.logs_dir,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "run_id": result.run_id,
                    "user_input": user_input,
                    "llm_calls": result.llm_calls,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                    "steps_limit": task_runner.settings.max_total_steps_per_turn,
                    "llm_limit": task_runner.settings.max_total_llm_calls_per_run_per_turn,
                    "tool_limit": task_runner.settings.max_tool_calls_per_task,
                },
            )
            console.print(
                f"[dim]Metrics: llm_calls={result.llm_calls} steps={result.steps} tool_calls={result.tool_calls}[/dim]"
            )
            if has_pending_question and related_to_pending_question:
                answer_payload = resolve_answered_question(
                    db, user_input, dict(pending_question_row), source_ref=episode.id
                )
                if answer_payload:
                    # Check if this was a contradiction question
                    _reason_raw = pending_question_row["reason"] if pending_question_row["reason"] else ""
                    _contradiction_resolved = False
                    if _reason_raw:
                        try:
                            _reason_data = json.loads(_reason_raw)
                            if isinstance(_reason_data, dict) and _reason_data.get("type") == "contradiction":
                                _contradiction_resolved = resolve_contradiction_answer(
                                    db, user_input, _reason_data, source_ref=episode.id
                                )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if not _contradiction_resolved:
                        db.insert_staged_item(
                            item_id=str(uuid4()),
                            kind="fact",
                            payload=json.dumps(answer_payload),
                            status="pending",
                            created_at=episode.timestamp,
                            source_ref=episode.id,
                            provenance_type="USER_REPORTED",
                            next_check_at=episode.timestamp,
                        )
            for record in result.tool_records:
                db.insert_tool_call(
                    tool_call_id=record.id,
                    episode_id=episode.id,
                    tool_name=record.name,
                    args=record.args,
                    status=record.status,
                    result_json=record.result_json,
                    result_hash=record.result_hash,
                    result_path=record.result_path,
                    started_at=record.started_at,
                    ended_at=record.ended_at,
                    risk_level=record.risk_level,
                    approval_mode=record.approval_mode,
                )
            vector_index.add(
                item_type="episode",
                item_id=episode.id,
                text=f"{episode.user_input}\n{episode.agent_output}",
            )
            console.print(
                f"[bold green]{agent_label}[/bold green]: {escape(episode.agent_output)}"
            )
            hook_messages = hook_manager.run("on_turn_end", {"episode_id": episode.id, "run_id": result.run_id})
            for msg in hook_messages:
                console.print(f"[dim]{escape(msg)}[/dim]")
            # Record a reflection for this run, noting that a run has completed. This
            # allows the agent to store metadata about its own performance for
            # later self‑improvement. Failures to record reflections are ignored.
            try:
                from tali.self_reflection import SelfReflection  # type: ignore
                reflector = SelfReflection(db)
                if result.run_id:
                    reflector.reflect(
                        run_id=str(result.run_id),
                        success=True,
                        notes="Automatic reflection after run completion",
                        improvement="",
                    )
            except Exception:
                pass
        resolution = resolve_staged_items(db, user_input) if _should_attempt_staged_resolution(user_input) else None
        if resolution and resolution.applied_fact_id:
            facts = db.fetch_facts_by_ids([resolution.applied_fact_id])
            if facts:
                vector_index.add(
                    item_type="fact", item_id=resolution.applied_fact_id, text=facts[0]["statement"]
                )
        if resolution and resolution.clarification_question:
            episode = build_episode(
                user_input=user_input,
                guardrail=guardrails.enforce(resolution.clarification_question, retriever.retrieve(user_input).bundle),
                tool_calls=[],
                outcome="clarification",
                quarantine=0,
            )
            db.insert_episode(
                episode_id=episode.id,
                user_input=episode.user_input,
                agent_output=episode.agent_output,
                tool_calls=episode.tool_calls,
                outcome=episode.outcome,
                quarantine=episode.quarantine,
            )
            stage_episode_fact(db, episode.id, episode.user_input, episode.outcome)
            vector_index.add(
                item_type="episode",
                item_id=episode.id,
                text=f"{episode.user_input}\n{episode.agent_output}",
            )
            console.print(
                f"[bold green]{agent_label}[/bold green]: {escape(episode.agent_output)}"
            )
            return
        active_run = db.fetch_active_run()
        if active_run and active_run["status"] in {"active", "blocked"}:
            _run_task_runner_turn()
            return
        if "facts we need to validate" in user_input.lower() or "staged" in user_input.lower():
            staged = db.list_staged_items(["pending", "verifying"], limit=5)
            if staged:
                lines = ["Staged items to validate:"]
                for row in staged:
                    lines.append(f"- {row['kind']} {row['payload']}")
                message = "\n".join(lines)
            else:
                message = "No staged items pending validation."
            guardrail = guardrails.enforce(message, retriever.retrieve(user_input).bundle)
            episode = build_episode(
                user_input=user_input,
                guardrail=guardrail,
                tool_calls=[],
                outcome="ok",
                quarantine=0,
            )
            db.insert_episode(
                episode_id=episode.id,
                user_input=episode.user_input,
                agent_output=episode.agent_output,
                tool_calls=episode.tool_calls,
                outcome=episode.outcome,
                quarantine=episode.quarantine,
            )
            stage_episode_fact(db, episode.id, episode.user_input, episode.outcome)
            vector_index.add(
                item_type="episode",
                item_id=episode.id,
                text=f"{episode.user_input}\n{episode.agent_output}",
            )
            console.print(
                f"[bold green]{agent_label}[/bold green]: {escape(episode.agent_output)}"
            )
            return
        if not has_pending_question:
            decision = select_question_to_ask(db, user_input)
            if decision:
                if _is_related_to_text(user_input, decision.question):
                    mark_question_asked(db, decision.question_id, decision.attempts)
                    episode = build_episode(
                        user_input=user_input,
                        guardrail=guardrails.enforce(decision.question, retriever.retrieve(user_input).bundle),
                        tool_calls=[],
                        outcome="clarification",
                        quarantine=0,
                    )
                    db.insert_episode(
                        episode_id=episode.id,
                        user_input=episode.user_input,
                        agent_output=episode.agent_output,
                        tool_calls=episode.tool_calls,
                        outcome=episode.outcome,
                        quarantine=episode.quarantine,
                    )
                    stage_episode_fact(db, episode.id, episode.user_input, episode.outcome)
                    vector_index.add(
                        item_type="episode",
                        item_id=episode.id,
                        text=f"{episode.user_input}\n{episode.agent_output}",
                    )
                    console.print(
                        f"[bold green]{agent_label}[/bold green]: {escape(episode.agent_output)}"
                    )
                    return
        _run_task_runner_turn()

    if message:
        run_turn(message)
        scheduler.stop()
        idle_scheduler.stop()
        poller.stop()
        registry.heartbeat(config.agent_id, status="inactive")
        return

    console.print(f"[dim]{agent_label} chat (type 'exit' to quit)[/dim]")
    # Check for incomplete/stale runs on startup
    _startup_incomplete = db.fetch_incomplete_runs()
    if _startup_incomplete:
        for _inc_run in _startup_incomplete:
            _run_created = _inc_run["created_at"] or "unknown"
            _run_status = _inc_run["status"]
            _run_prompt = str(_inc_run["user_prompt"] or "")[:80]
            console.print(
                f"[bold yellow]Incomplete run:[/bold yellow] {_inc_run['id'][:8]}... "
                f"(status={_run_status}, created={_run_created})"
            )
            console.print(f"  [dim]Prompt: {escape(_run_prompt)}[/dim]")
        console.print("[dim]Type 'resume' to continue or 'cancel' to abort.[/dim]")
    while True:
        user_input = console.input("[bold cyan]You[/bold cyan]: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        run_turn(user_input)
    scheduler.stop()
    idle_scheduler.stop()
    poller.stop()
    registry.heartbeat(config.agent_id, status="inactive")


@run_app.command("status")
def run_status(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show the active run and its tasks.

    Displays the current run status, task list, and metrics in a
    human-readable format. Use --json for machine-readable output.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    if not run:
        typer.echo("No active run.")
        return
    tasks = db.fetch_tasks_for_run(run["id"])
    metrics = latest_metrics_for_run(paths.logs_dir, str(run["id"]))
    if as_json:
        payload = {
            "run": dict(run),
            "tasks": [
                {"ordinal": row["ordinal"], "title": row["title"], "status": row["status"]}
                for row in tasks
            ],
            "metrics": metrics or {},
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_run_status(run, tasks, metrics))


@run_app.command("cancel")
def run_cancel() -> None:
    """Cancel the active run.

    Immediately sets the active run's status to 'canceled' and clears
    its current task pointer.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    if not run:
        typer.echo("No active run to cancel.")
        return
    db.update_run_status(run["id"], status="canceled", current_task_id=None, last_error=None)
    typer.echo(f"Run {run['id']} canceled.")


@run_app.command("resume")
def run_resume() -> None:
    """Resume a blocked or stale run.

    Transitions a blocked (or stale active) run back to 'active' status
    so that task execution can continue.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    if not run:
        # Also check for stale runs that might not appear as "active"
        stale = db.fetch_stale_runs(timeout_minutes=30)
        if stale:
            run = stale[0]
        else:
            typer.echo("No active or stale run to resume.")
            return
    if run["status"] == "active":
        # Check if it is stale
        updated_at = run["updated_at"]
        if updated_at:
            from datetime import timedelta
            try:
                last_update = datetime.fromisoformat(str(updated_at))
                if datetime.utcnow() - last_update > timedelta(minutes=30):
                    typer.echo(f"Run {run['id']} appears stale (last updated: {updated_at}). Resuming.")
                    db.update_run_status(run["id"], status="active", current_task_id=run["current_task_id"], last_error=None)
                    typer.echo(f"Run {run['id']} resumed.")
                    return
            except (ValueError, TypeError):
                pass
        typer.echo("Active run is not blocked or stale.")
        return
    if run["status"] != "blocked":
        typer.echo(f"Run {run['id']} has status '{run['status']}' and cannot be resumed.")
        return
    db.update_run_status(run["id"], status="active", current_task_id=run["current_task_id"], last_error=None)
    typer.echo(f"Run {run['id']} resumed.")


@run_app.command("show")
def run_show(
    run_id: str = typer.Argument(..., help="Run ID to display."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show a run and its tasks.

    Displays run details with task list and output summaries.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_run(run_id)
    if not run:
        typer.echo("Run not found.")
        return
    tasks = db.fetch_tasks_for_run(run_id)
    if as_json:
        payload = {
            "run": dict(run),
            "tasks": [
                {
                    "ordinal": row["ordinal"],
                    "title": row["title"],
                    "status": row["status"],
                    "outputs_json": row["outputs_json"],
                }
                for row in tasks
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_run_show(run, tasks))


@run_app.command("list")
def run_list(
    limit: int = typer.Option(20, "--limit", help="Max runs to show."),
    incomplete: bool = typer.Option(False, "--incomplete", help="Show only incomplete (active/blocked) runs."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent runs.

    Shows a table of recent runs with their status and any errors.
    Use --incomplete to filter to only active or blocked runs.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    if incomplete:
        runs = db.fetch_incomplete_runs()
    else:
        runs = db.list_runs(limit=limit)
    if as_json:
        payload = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "status": row["status"],
                "last_error": row["last_error"],
            }
            for row in runs
        ]
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_run_list(runs))


@run_app.command("timeline")
def run_timeline(
    run_id: str = typer.Argument(..., help="Run ID to display timeline."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show task event timeline for a run.

    Displays a chronological list of task events including status
    transitions, tool calls, and LLM interactions.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    events = db.fetch_task_events_for_run(run_id)
    if as_json:
        payload = [
            {
                "timestamp": row["timestamp"],
                "task_id": row["task_id"],
                "event_type": row["event_type"],
                "payload": row["payload"],
            }
            for row in events
        ]
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_timeline(events))


@agent_app.command("logs")
def logs(
    limit: int = typer.Option(10, "--limit", help="Max log entries to show."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show recent structured run logs.

    Displays a table of recent run logs with LLM call counts, tool
    call counts, and step counts. Filter by --run-id to see logs for
    a specific run.
    """
    paths, _ = _resolve_paths()
    entries = read_recent_logs(paths.logs_dir, limit=limit * 5)
    if run_id:
        entries = [entry for entry in entries if entry.get("run_id") == run_id]
    if not entries:
        typer.echo("No logs found.")
        return
    trimmed = entries[-limit:]
    if as_json:
        typer.echo(json.dumps(trimmed, indent=2))
    else:
        console = Console()
        console.print(format_logs(trimmed))


def _render_dashboard(db: Database, paths: Paths, config: AppConfig | None = None) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=2),
        Layout(name="middle", ratio=2),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="agent_state"),
        Layout(name="memory"),
        Layout(name="metrics"),
    )
    run = db.fetch_active_run()
    tasks = db.fetch_tasks_for_run(run["id"]) if run else []

    # --- Agent State panel ---
    agent_name_str = config.agent_name if config else "unknown"
    if run:
        state_label = f"[bold green]{run['status']}[/bold green]"
    else:
        state_label = "[dim]idle[/dim]"
    unread_count = len(db.list_unread_agent_messages(limit=100))
    pending_commitments = len(db.fetch_pending_commitments())
    agent_lines = [
        f"Agent: [bold]{agent_name_str}[/bold]",
        f"State: {state_label}",
        f"Inbox: {unread_count} unread",
        f"Pending commitments: {pending_commitments}",
    ]
    # Autonomy status
    if config and config.autonomy:
        auto = config.autonomy
        auto_status = "[green]on[/green]" if auto.enabled else "[dim]off[/dim]"
        agent_lines.append(f"Autonomy: {auto_status}")
        if auto.enabled:
            agent_lines.append(f"  Idle trigger: {auto.idle_trigger_seconds}s")
            agent_lines.append(f"  Auto-continue: {'yes' if auto.auto_continue else 'no'}")
    layout["top"]["agent_state"].update(Panel("\n".join(agent_lines), title="Agent"))

    # --- Memory panel ---
    staged_by_status = db.count_staged_items_by_status()
    facts_count = len(db.list_facts())
    commitments_count = len(db.list_commitments())
    preferences_count = len(db.list_preferences())
    memory_lines = [
        f"Facts: {facts_count}",
        f"Commitments: {commitments_count}",
        f"Preferences: {preferences_count}",
        f"Staged: {staged_by_status}",
    ]
    layout["top"]["memory"].update(Panel("\n".join(memory_lines), title="Memory"))

    # --- Run Metrics panel ---
    metrics = latest_metrics_for_run(paths.logs_dir, str(run["id"])) if run else None
    metrics_lines = [
        f"Run: {run['id'][:8] if run else '(none)'}",
        f"Status: {run['status']}" if run else "Status: n/a",
        f"LLM calls: {metrics.get('llm_calls')}" if metrics else "LLM calls: n/a",
        f"Tool calls: {metrics.get('tool_calls')}" if metrics else "Tool calls: n/a",
        f"Steps: {metrics.get('steps')}" if metrics else "Steps: n/a",
    ]
    layout["top"]["metrics"].update(Panel("\n".join(metrics_lines), title="Metrics"))

    # --- Tasks table ---
    task_table = Table(title="Tasks")
    task_table.add_column("Ord", justify="right")
    task_table.add_column("Title")
    task_table.add_column("Status")
    for row in tasks:
        task_table.add_row(str(row["ordinal"]), str(row["title"]), str(row["status"]))
    if not tasks:
        task_table.add_row("-", "No active run", "-")
    layout["middle"].update(task_table)

    # --- Logs ---
    recent_logs = read_recent_logs(paths.logs_dir, limit=5)
    log_lines = []
    for entry in recent_logs:
        run_id = entry.get("run_id", "")
        llm_calls = entry.get("llm_calls", "")
        tool_calls = entry.get("tool_calls", "")
        log_lines.append(f"{run_id[:8] if run_id else ''} llm={llm_calls} tools={tool_calls}")
    layout["bottom"].update(Panel("\n".join(log_lines) if log_lines else "No logs.", title="Recent Logs"))
    return layout


@agent_app.command("dashboard")
def dashboard(
    refresh_s: float = typer.Option(1.5, "--refresh", help="Refresh interval (seconds)."),
    duration_s: int = typer.Option(30, "--duration", help="How long to run (seconds). 0 = once."),
) -> None:
    """Show a live dashboard with tasks, memory, and logs.

    Renders a Rich-based live-updating panel showing memory stats,
    run metrics, current tasks, and recent logs. Use --duration 0
    for a single snapshot.
    """
    paths, existing = _resolve_paths()
    db = _init_db(paths.db_path)
    try:
        cfg = _load_or_raise_config(paths, existing)
    except (typer.Exit, SystemExit):
        cfg = None
    if duration_s == 0:
        console = Console()
        console.print(_render_dashboard(db, paths, cfg))
        return
    start = time.time()
    with Live(_render_dashboard(db, paths, cfg), refresh_per_second=max(1, int(1 / refresh_s))) as live:
        while time.time() - start < duration_s:
            live.update(_render_dashboard(db, paths, cfg))
            time.sleep(refresh_s)


@agent_app.command("name")
def agent_name_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show current agent identity.

    Displays the agent's name, unique ID, home directory, and role.
    """
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    if as_json:
        typer.echo(
            json.dumps(
                {"agent_id": config.agent_id, "agent_name": config.agent_name, "home": str(paths.agent_home)},
                indent=2,
            )
        )
    else:
        console = Console()
        console.print(format_agent_name(config, str(paths.agent_home)))


@agent_app.command("list")
def agent_list(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List known local agents.

    Shows all agents registered in the local Tali directory.
    """
    root = Path.home() / ".tali"
    registry = Registry(root / "shared" / "registry.json")
    agents = registry.list_agents()
    if not agents:
        agents = [
            {"agent_name": path.name, "home": str(path)}
            for path in _list_agent_dirs(root)
        ]
    if as_json:
        typer.echo(json.dumps(agents, indent=2))
    else:
        console = Console()
        console.print(format_agent_list(agents))


@agent_app.command("send")
def agent_send(
    to: Optional[str] = typer.Option(None, "--to", help="Agent name to send to (broadcast if omitted)."),
    topic: str = typer.Option("task", "--topic", help="Message topic."),
    payload_json: str = typer.Option(..., "--json", help="JSON payload."),
) -> None:
    """Send an A2A message.

    Sends a message to another agent. If --to is omitted the message
    is broadcast to all registered agents.
    """
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        typer.echo(f"Invalid JSON: {exc}")
        raise typer.Exit(code=1)
    registry = Registry(paths.shared_home / "registry.json")
    target = None
    if to:
        for agent in registry.list_agents():
            if agent.get("agent_name") == to:
                target = agent
                break
        if not target:
            typer.echo("Agent not found.")
            raise typer.Exit(code=1)
    db = _init_db(paths.db_path)
    client = A2AClient(
        db=db,
        profile=AgentProfile(
            agent_id=config.agent_id,
            agent_name=config.agent_name,
            agent_home=paths.agent_home,
            shared_home=paths.shared_home,
            capabilities=config.capabilities,
        ),
    )
    client.send(
        to_agent_id=target.get("agent_id") if target else None,
        to_agent_name=target.get("agent_name") if target else None,
        topic=topic,
        payload=payload,
        correlation_id=payload.get("correlation_id"),
    )
    typer.echo("Message sent.")


@agent_app.command("inbox")
def agent_inbox(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show unread A2A messages.

    Displays sender, topic, timestamp, and a payload summary for each
    unread message in the agent's inbox.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_unread_agent_messages(limit=20)
    if as_json:
        typer.echo(json.dumps([dict(row) for row in rows], indent=2))
    else:
        console = Console()
        console.print(format_inbox(rows))


@agent_app.command("swarm")
def swarm(prompt: str = typer.Argument(..., help="Swarm task prompt.")) -> None:
    """Run a swarm-enabled task turn.

    Executes a task using the multi-agent swarm, coordinating with
    other registered agents via A2A messaging.
    """
    paths, existing = _resolve_paths()
    _ensure_dependencies()
    config = _load_or_raise_config(paths, existing)
    db = _init_db(paths.db_path)
    planner_settings = config.planner_llm
    responder_settings = config.responder_llm
    if not planner_settings or not responder_settings:
        typer.echo("Missing planner/responder LLM settings. Run `agent create`.")
        raise typer.Exit(code=1)
    planner_llm = _build_llm_client(planner_settings)
    responder_llm = (
        planner_llm
        if responder_settings == planner_settings
        else _build_llm_client(responder_settings)
    )
    embedder = _build_embedder(config.embeddings)
    vector_index = VectorIndex(paths.vector_dir / "memory.index", config.embeddings.dim, embedder)
    retriever = Retriever(db, RetrievalConfig(), vector_index=vector_index)
    guardrails = Guardrails(GuardrailConfig())
    tool_settings = config.tools if isinstance(config.tools, ToolSettings) else ToolSettings()
    registry = build_default_registry(paths, tool_settings)
    policy = ToolPolicy(tool_settings, registry, paths)
    approvals = ApprovalManager(mode=tool_settings.approval_mode)
    runner = ToolRunner(registry, policy, approvals, tool_settings, paths)
    tool_descriptions = registry.describe_tools()
    a2a_client = A2AClient(
        db=db,
        profile=AgentProfile(
            agent_id=config.agent_id,
            agent_name=config.agent_name,
            agent_home=paths.agent_home,
            shared_home=paths.shared_home,
            capabilities=config.capabilities,
        ),
    )
    run_planner_strengths = ", ".join(infer_model_strengths(planner_settings))
    run_responder_strengths = ", ".join(infer_model_strengths(responder_settings))
    agent_context = "\n".join(
        [
            f"Agent name: {config.agent_name}",
            f"Role: {config.role_description or 'assistant'}",
            f"Agent home: {paths.agent_home}",
            f"Config path: {paths.config_path}",
            f"Planner model: {planner_settings.model}",
            f"Responder model: {responder_settings.model}",
            f"Tool fs_root: {tool_settings.fs_root}",
            f"OS: {platform.system()} {platform.release()}",
            "The agent is distinct from the LLM model.",
            f"You are the LLM planner; the {config.agent_name} agent executes tool calls you request.",
            "Use config.json (not config.txt).",
            "Do not use tools for greetings or small talk.",
            "Model capabilities:",
            f"- Planner ({planner_settings.model}): {run_planner_strengths}",
            f"- Responder ({responder_settings.model}): {run_responder_strengths}",
            "When choosing 'respond', provide detailed factual content in your message so the responder can phrase it well for the user.",
        ]
    )
    swarm_info = build_swarm_context(a2a_client, config.agent_name)
    if swarm_info:
        agent_context += "\n\nTeam resources:\n" + swarm_info
    # Include a summary of long‑term memory in the agent context. This helps
    # the planner leverage key facts when coordinating multi‑agent work.
    try:
        from tali.memory_manager import MemoryManager  # type: ignore

        mem_mgr = MemoryManager(db, vector_index)
        summary = mem_mgr.summarize_memory(limit=5)
        if summary:
            agent_context += "\n\nLong-term memory:\n" + summary
    except Exception:
        pass
    task_runner_config = (
        config.task_runner if isinstance(config.task_runner, TaskRunnerConfig) else None
    )
    task_runner_settings = (
        TaskRunnerSettings(
            max_tasks_per_turn=task_runner_config.max_tasks_per_turn,
            max_llm_calls_per_task=task_runner_config.max_llm_calls_per_task,
            max_tool_calls_per_task=task_runner_config.max_tool_calls_per_task,
            max_total_llm_calls_per_run_per_turn=(
                task_runner_config.max_total_llm_calls_per_run_per_turn
            ),
            max_total_steps_per_turn=task_runner_config.max_total_steps_per_turn,
        )
        if task_runner_config
        else None
    )
    task_runner = TaskRunner(
        db=db,
        llm=planner_llm,
        planner_llm=planner_llm,
        responder_llm=responder_llm,
        retriever=retriever,
        guardrails=guardrails,
        tool_runner=runner,
        tool_descriptions=tool_descriptions,
        a2a_client=a2a_client,
        agent_context=agent_context,
        settings=task_runner_settings,
        responder_strengths=run_responder_strengths,
    )
    result = task_runner.run_turn(prompt, prompt_fn=typer.prompt)
    typer.echo(result.message)


@agent_app.command("delete")
def delete_agent(agent_name: str = typer.Argument(..., help="Agent name to delete.")) -> None:
    """Delete an agent by name.

    Removes the agent's data directory, worktree, and registry entry.
    If no agents remain, the bootstrap wizard is launched.
    """
    root = Path.home() / ".tali"
    agent_home = root / agent_name
    if not agent_home.exists():
        typer.echo("Agent not found.")
        raise typer.Exit(code=1)
    repo_root = resolve_main_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        paths = load_paths(root, agent_name)
        status = remove_agent_worktree(paths, repo_root)
        if status.message:
            typer.echo(status.message)
    registry = Registry(root / "shared" / "registry.json")
    registry.remove(agent_name)
    typer.echo(f"Deleting agent data at {agent_home} ...")
    shutil.rmtree(agent_home)
    remaining = [p for p in root.iterdir() if p.is_dir() and p.name != "shared" and (p / "config.json").exists()]
    if not remaining:
        typer.echo("No agents remain. Launching agent bootstrap.")
        _bootstrap_first_agent(start_chat=False)


@patch_app.command("list")
def patches_list(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List patch proposals.

    Shows a table of all patch proposals with their ID, creation date,
    title, and current status.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_patch_proposals()
    if as_json:
        payload = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "title": row["title"],
                "status": row["status"],
            }
            for row in rows
        ]
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_patches_list(rows))


@patch_app.command("show")
def patches_show(
    proposal_id: str = typer.Argument(..., help="Patch proposal ID."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show a patch proposal.

    Displays full details of a patch proposal including title, rationale,
    affected files, diff, and test results.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    if as_json:
        typer.echo(json.dumps(dict(row), indent=2))
    else:
        console = Console()
        console.print(format_patch_detail(row))


@patch_app.command("test")
def patches_test(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Run tests for a patch proposal.

    Executes the test suite specified in the patch proposal and records
    the results in the database.
    """
    paths, _ = _resolve_paths()
    _ensure_dependencies()
    code_dir = _ensure_agent_worktree(paths)
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    tests_payload = row["test_results"] or ""
    try:
        parsed = json.loads(tests_payload) if tests_payload else {}
    except json.JSONDecodeError:
        parsed = {}
    tests = parsed.get("tests", [])
    if not tests:
        typer.echo("No tests specified for this proposal.")
        return
    repo_hint = Path(__file__).resolve().parent
    results = run_patch_tests(tests, cwd=code_dir or repo_hint)
    parsed["results"] = results
    db.update_patch_proposal(
        proposal_id=proposal_id,
        status="tested",
        test_results=json.dumps(parsed),
        rollback_ref=row["rollback_ref"],
    )
    typer.echo(results)


@patch_app.command("apply")
def patches_apply(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Apply a patch proposal after tests pass.

    Applies the diff from the proposal to the agent's worktree. The
    patch must be in 'tested' or 'approved' status with recorded results.
    """
    paths, _ = _resolve_paths()
    code_dir = _ensure_agent_worktree(paths)
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    if row["status"] not in {"tested", "approved"}:
        typer.echo("Patch proposal must be tested or approved before applying.")
        return
    tests_payload = row["test_results"] or ""
    try:
        parsed = json.loads(tests_payload) if tests_payload else {}
    except json.JSONDecodeError:
        parsed = {}
    if "results" not in parsed:
        typer.echo("Test results missing; run tests first.")
        return
    error = apply_patch(row["diff_text"], cwd=code_dir or Path.cwd())
    if error:
        db.update_patch_proposal(
            proposal_id=proposal_id,
            status="rejected",
            test_results=row["test_results"],
            rollback_ref=row["rollback_ref"],
        )
        typer.echo(error)
        return
    db.update_patch_proposal(
        proposal_id=proposal_id,
        status="applied",
        test_results=row["test_results"],
        rollback_ref="git_apply_reverse",
    )
    typer.echo("Patch applied.")


@patch_app.command("reject")
def patches_reject(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Reject a patch proposal.

    Marks a patch proposal as 'rejected' without applying changes.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    db.update_patch_proposal(
        proposal_id=proposal_id,
        status="rejected",
        test_results=row["test_results"],
        rollback_ref=row["rollback_ref"],
    )
    typer.echo("Patch rejected.")


@patch_app.command("rollback")
def patches_rollback(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Rollback an applied patch proposal.

    Reverses a previously applied patch, restoring the worktree to its
    pre-patch state.
    """
    paths, _ = _resolve_paths()
    code_dir = _ensure_agent_worktree(paths)
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    if row["status"] != "applied":
        typer.echo("Patch proposal is not applied.")
        return
    error = reverse_patch(row["diff_text"], cwd=code_dir or Path.cwd())
    if error:
        typer.echo(error)
        return
    db.update_patch_proposal(
        proposal_id=proposal_id,
        status="rejected",
        test_results=row["test_results"],
        rollback_ref=row["rollback_ref"],
    )
    typer.echo("Patch rolled back.")


@agent_app.command("commitments")
def commitments_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List commitments.

    Shows all agent commitments with their status, priority, and due date.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_commitments()
    if as_json:
        typer.echo(json.dumps([dict(row) for row in rows], indent=2))
    else:
        console = Console()
        console.print(format_commitments(rows))


@agent_app.command("facts")
def facts_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List facts.

    Shows all stored facts with confidence scores and provenance.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_facts()
    if as_json:
        typer.echo(json.dumps([dict(row) for row in rows], indent=2))
    else:
        console = Console()
        console.print(format_facts(rows))


@agent_app.command("skills")
def skills_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List skills.

    Shows all learned skills with success/failure counts and triggers.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_skills()
    if as_json:
        typer.echo(json.dumps([dict(row) for row in rows], indent=2))
    else:
        console = Console()
        console.print(format_skills(rows))


@agent_app.command("diff")
def diff() -> None:
    """Show differences between the latest snapshot and current data.

    Compares the current data directory against the most recent snapshot
    and displays what has changed.
    """
    paths, _ = _resolve_paths()
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    typer.echo(diff_snapshot(paths.data_dir, latest))


@agent_app.command("rollback")
def rollback() -> None:
    """Rollback data directory to the latest snapshot.

    Restores the agent's data directory to the state captured in the
    most recent snapshot. Use 'agent diff' to preview changes first.
    """
    paths, _ = _resolve_paths()
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    rollback_snapshot(paths.data_dir, latest)
    typer.echo(f"Rolled back data to snapshot {latest.id}.")


@agent_app.command("snapshot")
def snapshot() -> None:
    """Create a snapshot of the data directory.

    Saves a point-in-time backup of the agent's data that can be
    restored later with 'agent rollback'.
    """
    paths, _ = _resolve_paths()
    snapshot = create_snapshot(paths.data_dir)
    typer.echo(f"Snapshot created: {snapshot.id}")


@agent_app.command("doctor")
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Validate core invariants and report staged items.

    Checks fact provenance, confidence ranges, staged item health,
    and vector index integrity. Returns exit code 1 if violations found.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    violations: list[str] = []
    all_facts = db.list_facts()
    for row in all_facts:
        if row["provenance_type"] == "AGENT_OUTPUT":
            violations.append(f"Fact {row['id']} has forbidden provenance.")
        if not row["source_ref"]:
            violations.append(f"Fact {row['id']} missing source_ref.")
        confidence = float(row["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            violations.append(f"Fact {row['id']} has confidence out of range.")
    staged_count = db.count_staged_items()
    staged_by_status = db.count_staged_items_by_status()
    oldest = db.oldest_pending_staged_item()
    mapping_path = (paths.vector_dir / "memory.index").with_suffix(".json")
    vector_report: dict = {"mapping_path": str(mapping_path), "mapping_entries": 0, "missing_facts": 0, "missing_episodes": 0}
    if mapping_path.exists():
        try:
            mapping_payload = json.loads(mapping_path.read_text())
        except json.JSONDecodeError:
            mapping_payload = {}
            violations.append("Vector index mapping JSON is invalid.")
        if isinstance(mapping_payload, dict):
            items = list(mapping_payload.values())
            vector_report["mapping_entries"] = len(items)
            fact_ids = [item.get("item_id") for item in items if item.get("item_type") == "fact"]
            episode_ids = [item.get("item_id") for item in items if item.get("item_type") == "episode"]
            facts_found = {row["id"] for row in db.fetch_facts_by_ids([fid for fid in fact_ids if fid])}
            episodes_found = {row["id"] for row in db.fetch_episodes_by_ids([eid for eid in episode_ids if eid])}
            vector_report["missing_facts"] = len([fid for fid in fact_ids if fid and fid not in facts_found])
            vector_report["missing_episodes"] = len([eid for eid in episode_ids if eid and eid not in episodes_found])
    else:
        vector_report["mapping_entries"] = 0
    report = {
        "violations": violations,
        "staged_items": staged_count,
        "staged_items_by_status": staged_by_status,
        "oldest_pending_staged_item": dict(oldest) if oldest else None,
        "vector_index": vector_report,
    }
    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        console = Console()
        console.print(format_doctor(report))
    if violations:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Config commands
# ---------------------------------------------------------------------------

_CONFIG_SECTIONS = {
    "planner_llm": LLMSettings,
    "responder_llm": LLMSettings,
    "embeddings": EmbeddingSettings,
    "tools": ToolSettings,
    "task_runner": TaskRunnerConfig,
    "autonomy": AutonomyConfig,
}


def _coerce_value(field_type: object, raw: str) -> object:
    """Coerce a CLI string to the expected dataclass field type."""
    import types
    import typing

    # Unwrap union types (e.g. str | None, bool | None) to their first
    # non-None component.
    origin = getattr(field_type, "__origin__", None)
    if isinstance(field_type, types.UnionType) or origin is typing.Union:
        args = typing.get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            field_type = non_none[0]

    if field_type is bool:
        return raw.lower() in ("true", "1", "yes", "on")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return raw

    # Fallback – try int then float then string
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


@config_app.command("show")
def config_show(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show current agent configuration.

    Displays all configuration sections (LLM, embeddings, tools,
    task runner, autonomy) in a human-readable format.
    """
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    if as_json:
        typer.echo(json.dumps(json.loads(paths.config_path.read_text()), indent=2))
    else:
        console = Console()
        console.print(format_config(config, str(paths.agent_home)))


@config_app.command("set")
def config_set(
    field: str = typer.Argument(..., help="Dot-notation path, e.g. responder_llm.model or autonomy.enabled"),
    value: str = typer.Argument(..., help="New value for the field."),
) -> None:
    """Update a configuration field.

    Accepts dot-notation paths such as:
      responder_llm.model, planner_llm.base_url, autonomy.enabled,
      tools.approval_mode, task_runner.max_tasks_per_turn

    Example:
      tali agent config set responder_llm.model gemma-3b-v2
    """
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    parts = field.split(".")
    if len(parts) == 1:
        # Top-level agent field
        top = parts[0]
        if top in ("role_description",):
            old_val = getattr(config, top, None)
            config = replace(config, **{top: value})
            save_config(paths.config_path, config)
            typer.echo(f"Updated {top}: {old_val!r} -> {value!r}")
            return
        typer.echo(f"Unknown or read-only top-level field: {top}")
        typer.echo("Settable fields: role_description, or use section.field (e.g. responder_llm.model)")
        raise typer.Exit(code=1)
    elif len(parts) == 2:
        section_name, attr = parts
        if section_name not in _CONFIG_SECTIONS:
            typer.echo(f"Unknown config section: {section_name}")
            typer.echo(f"Available sections: {', '.join(_CONFIG_SECTIONS.keys())}")
            raise typer.Exit(code=1)
        section_obj = getattr(config, section_name, None)
        section_cls = _CONFIG_SECTIONS[section_name]
        if section_obj is None:
            section_obj = section_cls()
        if not hasattr(section_obj, attr):
            typer.echo(f"Unknown field '{attr}' in section '{section_name}'.")
            valid = [f.name for f in section_cls.__dataclass_fields__.values()]
            typer.echo(f"Available fields: {', '.join(valid)}")
            raise typer.Exit(code=1)
        # Determine target type from dataclass field annotation
        field_meta = section_cls.__dataclass_fields__[attr]
        old_val = getattr(section_obj, attr)
        coerced = _coerce_value(field_meta.type, value)
        # Validate provider fields
        if attr == "provider" and coerced not in ("openai", "ollama"):
            typer.echo("Provider must be 'openai' or 'ollama'.")
            raise typer.Exit(code=1)
        new_section = replace(section_obj, **{attr: coerced})
        config = replace(config, **{section_name: new_section})
        save_config(paths.config_path, config)
        typer.echo(f"Updated {field}: {old_val!r} -> {coerced!r}")
    else:
        typer.echo("Field path too deep. Use section.field format (e.g. responder_llm.model).")
        raise typer.Exit(code=1)


@config_app.command("reset")
def config_reset(
    field: str = typer.Argument(..., help="Dot-notation path to reset, e.g. autonomy.idle_trigger_seconds"),
) -> None:
    """Reset a configuration field to its default value.

    Example:
      tali agent config reset autonomy.idle_trigger_seconds
    """
    parts = field.split(".")
    if len(parts) != 2:
        typer.echo("Use section.field format (e.g. autonomy.idle_trigger_seconds).")
        raise typer.Exit(code=1)
    section_name, attr = parts
    if section_name not in _CONFIG_SECTIONS:
        typer.echo(f"Unknown config section: {section_name}")
        raise typer.Exit(code=1)
    section_cls = _CONFIG_SECTIONS[section_name]
    if attr not in section_cls.__dataclass_fields__:
        typer.echo(f"Unknown field '{attr}' in section '{section_name}'.")
        raise typer.Exit(code=1)
    default_obj = section_cls()
    default_val = getattr(default_obj, attr)
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    section_obj = getattr(config, section_name, None)
    if section_obj is None:
        section_obj = section_cls()
    old_val = getattr(section_obj, attr)
    new_section = replace(section_obj, **{attr: default_val})
    config = replace(config, **{section_name: new_section})
    save_config(paths.config_path, config)
    typer.echo(f"Reset {field}: {old_val!r} -> {default_val!r} (default)")


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------

@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "--limit", help="Max results per category."),
    facts_only: bool = typer.Option(False, "--facts-only", help="Only search facts."),
    episodes_only: bool = typer.Option(False, "--episodes-only", help="Only search episodes."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Search agent memory for facts and episodes.

    Performs a full-text search across the agent's stored facts and
    conversation episodes. Results are ranked by relevance.

    Example:
      tali agent memory search "project deadline"
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    fact_rows = [] if episodes_only else db.search_facts(query, limit)
    episode_rows = [] if facts_only else db.search_episodes(query, limit)
    if as_json:
        payload = {
            "facts": [dict(r) for r in fact_rows],
            "episodes": [dict(r) for r in episode_rows],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(format_memory_search(fact_rows, episode_rows))


# ---------------------------------------------------------------------------
# Commitment add command
# ---------------------------------------------------------------------------

@agent_app.command("commitment-add")
def commitment_add(
    description: str = typer.Argument(..., help="Commitment description."),
    due: Optional[str] = typer.Option(None, "--due", help="Due date (YYYY-MM-DD)."),
    priority: int = typer.Option(3, "--priority", help="Priority (1=highest, 5=lowest)."),
) -> None:
    """Add a new commitment for the agent.

    The agent will pick up this commitment during its next cycle.

    Example:
      tali agent commitment-add "Review pull request #42" --due 2026-03-01
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    now = datetime.utcnow().isoformat()
    cid = str(uuid4())
    db.insert_commitment(
        commitment_id=cid,
        description=description,
        status="pending",
        priority=priority,
        due_date=due,
        created_at=now,
        last_touched=now,
        source_ref="cli",
    )
    typer.echo(f"Commitment added: {cid[:8]} - {description}")


# ---------------------------------------------------------------------------
# Preference commands
# ---------------------------------------------------------------------------

@pref_app.command("list")
def preferences_list(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List user preferences.

    Shows all stored preferences with their confidence scores.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_preferences()
    if as_json:
        typer.echo(json.dumps([dict(row) for row in rows], indent=2))
    else:
        console = Console()
        console.print(format_preferences(rows))


@pref_app.command("set")
def preferences_set(
    key: str = typer.Argument(..., help="Preference key."),
    value: str = typer.Argument(..., help="Preference value."),
    confidence: float = typer.Option(0.9, "--confidence", help="Confidence score (0.0-1.0)."),
) -> None:
    """Set a user preference.

    Stores or updates a preference in the agent's database.

    Example:
      tali agent preferences set humor off
      tali agent preferences set coding_style "functional"
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    now = datetime.utcnow().isoformat()
    db.upsert_preference(
        key=key,
        value=value,
        confidence=confidence,
        provenance_type="USER_EXPLICIT",
        source_ref="cli",
        updated_at=now,
    )
    typer.echo(f"Preference set: {key} = {value} (confidence: {confidence})")


@pref_app.command("remove")
def preferences_remove(
    key: str = typer.Argument(..., help="Preference key to remove."),
) -> None:
    """Remove a user preference.

    Deletes a preference from the agent's database.
    """
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    db.delete_preference(key)
    typer.echo(f"Preference removed: {key}")


# ---------------------------------------------------------------------------
# Help command
# ---------------------------------------------------------------------------

@agent_app.command("help")
def agent_help() -> None:
    """Show common workflows and a quick reference guide.

    Displays a cheat-sheet of the most useful Tali commands grouped
    by category.
    """
    console = Console()
    console.print(format_help_guide())


@app.command("setup")
def setup() -> None:
    """Walk through configuration and install dependencies.

    Interactive wizard that configures LLM provider, model, API keys,
    embedding settings, and verifies connectivity.
    """
    root, agent_name, existing_config = resolve_agent(prompt_fn=typer.prompt, allow_create_config=True)
    paths = load_paths(root, agent_name)
    typer.echo("Setting up Tali configuration.")
    _ensure_dependencies()
    planner_provider = typer.prompt("Planner LLM provider (openai/ollama)", default="openai")
    if planner_provider not in {"openai", "ollama"}:
        typer.echo("Provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if planner_provider == "openai":
        catalog_models = list_catalog_models()
        if catalog_models:
            typer.echo("Catalog models:")
            for name in catalog_models[:10]:
                typer.echo(f"  - {name}")
        planner_base_url, planner_auth = _prompt_openai_endpoint(
            "Planner", "http://localhost:8000/v1"
        )
        if planner_auth == "local":
            default_model = "llama3"
        else:
            default_model = "gpt-5-codex" if planner_auth == "codex" else "gpt-4o-mini"
        planner_model = typer.prompt("Planner LLM model", default=default_model)
    else:
        planner_model = _prompt_ollama_model("Planner LLM model", default="llama3")
        planner_base_url = typer.prompt(
            "Planner LLM base URL", default="http://localhost:11434"
        )
    planner_api_key = None
    if planner_provider == "openai":
        if planner_auth == "local":
            planner_api_key = typer.prompt(
                "Planner OpenAI API key (leave blank for local server)",
                default="",
                show_default=False,
                hide_input=True,
            )
            if planner_api_key == "":
                planner_api_key = None
        else:
            if planner_auth == "codex":
                planner_api_key = _ensure_codex_access_token("Planner", force_login=False)
            else:
                planner_api_key = typer.prompt("Planner OpenAI API key", hide_input=True)
    else:
        _prompt_ollama_download(planner_model)
    if not _validate_provider_connectivity(planner_provider, planner_base_url, planner_api_key):
        typer.echo("Warning: could not validate LLM connectivity. Check base URL or API key.")

    planner_llm = LLMSettings(
        provider=planner_provider,
        model=planner_model,
        base_url=planner_base_url,
        api_key=planner_api_key,
    )
    if typer.confirm("Use the same LLM settings for the responder?", default=True):
        responder_llm = planner_llm
    else:
        responder_llm = _prompt_llm_settings(
            "Responder LLM",
            default_provider=planner_provider,
            default_base_url=planner_base_url,
            default_model=planner_model,
            models_dir=paths.models_dir,
        )

    embed_provider = typer.prompt(
        "Embedding provider (openai/ollama)", default=planner_provider
    )
    if embed_provider not in {"openai", "ollama"}:
        typer.echo("Embedding provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if embed_provider == "openai":
        embed_base_url = typer.prompt(
            "Embedding base URL",
            default=planner_base_url
            if embed_provider == planner_provider
            else "http://localhost:8000/v1",
        )
        embed_default_model = (
            "nomic-embed-text" if _is_local_base_url(embed_base_url) else "text-embedding-3-small"
        )
        typer.echo("Embedding models (catalog):")
        for name in list_catalog_models()[:10]:
            typer.echo(f"  - {name}")
        embed_model = typer.prompt("Embedding model", default=embed_default_model)
    else:
        embed_model = _prompt_ollama_model("Embedding model", default="nomic-embed-text")
        embed_base_url = typer.prompt(
            "Embedding base URL",
            default=planner_base_url
            if embed_provider == planner_provider
            else "http://localhost:11434",
        )
    embed_api_key = None
    if embed_provider == "openai":
        if _is_local_base_url(embed_base_url):
            embed_api_key = typer.prompt(
                "OpenAI API key for embeddings (leave blank for local server)",
                default="",
                show_default=False,
                hide_input=True,
            )
            if embed_api_key == "":
                embed_api_key = None
        else:
            embed_api_key = typer.prompt("OpenAI API key for embeddings", hide_input=True)
    else:
        _prompt_ollama_download(embed_model)
    if not _validate_provider_connectivity(embed_provider, embed_base_url, embed_api_key):
        typer.echo("Warning: could not validate embedding connectivity. Check base URL or API key.")
    embed_dim = typer.prompt("Embedding dimension", default="1536")
    typer.echo(
        "File access is scoped by tools.fs_root (default: your user profile). "
        "You can change it to narrow or expand the agent's file access for security."
    )

    agent_id = existing_config.agent_id if existing_config and existing_config.agent_id else str(uuid4())
    created_at = existing_config.created_at if existing_config and existing_config.created_at else datetime.utcnow().isoformat()
    capabilities = existing_config.capabilities if existing_config else []
    config = AppConfig(
        agent_id=agent_id,
        agent_name=agent_name,
        created_at=created_at,
        role_description=existing_config.role_description if existing_config else "assistant",
        capabilities=capabilities,
        planner_llm=None,
        responder_llm=None,
        embeddings=None,
        tools=None,
        task_runner=TaskRunnerConfig(),
    )
    save_config(paths.config_path, config)
    shared_settings = SharedSettings(
        planner_llm=planner_llm,
        responder_llm=responder_llm,
        embeddings=EmbeddingSettings(
            provider=embed_provider,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
            dim=int(embed_dim),
        ),
        tools=ToolSettings(fs_root=str(Path.home())),
        task_runner=TaskRunnerConfig(),
    )
    save_shared_settings(paths.shared_home / "config.json", shared_settings)
    paths.vector_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    paths.sleep_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.inbox_dir.mkdir(parents=True, exist_ok=True)
    paths.outbox_dir.mkdir(parents=True, exist_ok=True)
    (paths.shared_home / "locks").mkdir(parents=True, exist_ok=True)
    typer.echo(f"Config saved to {paths.config_path}")
    subprocess.run(["python", "-m", "pip", "install", "-e", "."], check=True)
