from __future__ import annotations

import json
import platform
import subprocess
import re
import shutil
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from dataclasses import replace

from tali.config import (
    AppConfig,
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
)
from tali.agent_identity import resolve_agent
from tali.a2a import A2AClient, A2APoller, AgentProfile
from tali.a2a_registry import AgentRecord, Registry
from tali.consolidation import apply_sleep_output
from tali.db import Database
from tali.episode import build_episode, build_prompt
from tali.embeddings import OllamaEmbeddingClient, OpenAIEmbeddingClient
from tali.guardrails import GuardrailResult, Guardrails
from tali.llm import OllamaClient, OpenAIClient
from tali.retrieval import Retriever
from tali.self_care import SleepScheduler, resolve_staged_items
from tali.sleep import load_sleep_output, run_sleep
from tali.snapshots import create_snapshot, diff_snapshot, list_snapshots, rollback_snapshot
from tali.vector_index import VectorIndex
from tali.approvals import ApprovalManager
from tali.task_runner import TaskRunner, TaskRunnerSettings
from tali.hooks.core import HookManager
from tali.idle import IdleScheduler
from tali.knowledge_sources import KnowledgeSourceRegistry
from tali.questions import mark_question_asked, resolve_answered_question, select_question_to_ask
from tali.patches import apply_patch, reverse_patch, run_patch_tests
from tali.tools.registry import build_default_registry
from tali.tools.policy import ToolPolicy
from tali.tools.runner import ToolRunner

app = typer.Typer(help="Tali agent CLI")
run_app = typer.Typer(help="Manage task runs")
app.add_typer(run_app, name="run")
patch_app = typer.Typer(help="Manage patch proposals")
app.add_typer(patch_app, name="patches")


def _init_db(db_path: Path) -> Database:
    db = Database(db_path)
    db.initialize()
    return db


def _resolve_paths() -> tuple[Paths, AppConfig | None]:
    root, agent_name, config = resolve_agent(prompt_fn=typer.prompt, allow_create_config=False)
    paths = load_paths(root, agent_name)
    return paths, config


def _load_or_raise_config(paths: Paths, existing: AppConfig | None = None) -> AppConfig:
    if existing:
        config = existing
    else:
        if not paths.config_path.exists():
            typer.echo("Config not found. Run `agent setup` to configure LLMs.")
            raise typer.Exit(code=1)
        config = load_config(paths.config_path)
    if not config.agent_name or not config.agent_id or config.agent_name != paths.agent_name:
        updated = AppConfig(
            agent_id=config.agent_id or str(uuid4()),
            agent_name=paths.agent_name,
            created_at=config.created_at or datetime.utcnow().isoformat(),
            capabilities=config.capabilities or [],
            llm=config.llm,
            embeddings=config.embeddings,
            tools=config.tools,
            task_runner=config.task_runner,
        )
        save_config(paths.config_path, updated)
        config = updated
    shared_path = paths.shared_home / "config.json"
    shared = load_shared_settings(shared_path)
    if shared:
        config = AppConfig(
            agent_id=config.agent_id,
            agent_name=config.agent_name,
            created_at=config.created_at,
            capabilities=config.capabilities,
            llm=config.llm or shared.llm,
            embeddings=config.embeddings or shared.embeddings,
            tools=config.tools or shared.tools,
            task_runner=config.task_runner,
        )
    if not config.llm or not config.embeddings or not config.tools:
        typer.echo("Shared config missing; run `tali setup` to configure shared settings.")
        raise typer.Exit(code=1)
    return config


def _build_llm_client(settings: LLMSettings):
    if settings.provider == "openai":
        if not settings.api_key:
            typer.echo("OpenAI API key is required.")
            raise typer.Exit(code=1)
        return OpenAIClient(base_url=settings.base_url, api_key=settings.api_key, model=settings.model)
    if settings.provider == "ollama":
        return OllamaClient(base_url=settings.base_url, model=settings.model)
    typer.echo(f"Unsupported LLM provider: {settings.provider}")
    raise typer.Exit(code=1)


def _build_embedder(settings: EmbeddingSettings):
    if settings.provider == "openai":
        if not settings.api_key:
            typer.echo("OpenAI API key is required for embeddings.")
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


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send to the agent."),
    verbose_tools: bool = typer.Option(False, "--verbose-tools", help="Show full tool output."),
    show_plans: bool = typer.Option(False, "--show-plans", help="Show task planning steps."),
) -> None:
    """Start a chat loop or run a single turn if message is provided."""
    paths, existing_config = _resolve_paths()
    _ensure_dependencies()
    config = _load_or_raise_config(paths, existing_config)
    db = _init_db(paths.db_path)
    llm = _build_llm_client(config.llm)
    embedder = _build_embedder(config.embeddings)
    vector_index = VectorIndex(paths.vector_dir / "memory.index", config.embeddings.dim, embedder)
    retriever = Retriever(db, RetrievalConfig(), vector_index=vector_index)
    guardrails = Guardrails(GuardrailConfig())
    hooks_dir = Path(__file__).resolve().parent / "hooks"
    hook_manager = HookManager(db=db, hooks_dir=hooks_dir)
    hook_manager.load_hooks()
    scheduler = SleepScheduler(paths.data_dir, db, llm, vector_index, hook_manager=hook_manager)
    scheduler.start()
    sources = KnowledgeSourceRegistry()
    idle_scheduler = IdleScheduler(
        paths.data_dir,
        db,
        llm,
        sources,
        hook_manager=hook_manager,
        status_fn=lambda msg: console.print(f"[dim]{escape(msg)}[/dim]"),
    )
    idle_scheduler.start()
    console = Console()
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
    agent_context = "\n".join(
        [
            f"Agent name: {config.agent_name}",
            f"Agent home: {paths.agent_home}",
            f"Config path: {paths.config_path}",
            f"LLM model: {config.llm.model}",
            f"Tool fs_root: {tool_settings.fs_root}",
            f"OS: {platform.system()} {platform.release()}",
            "The agent is distinct from the LLM model.",
            "You are the LLM planner; the Tali agent executes tool calls you request.",
            "Use config.json (not config.txt).",
            "Do not use tools for greetings or small talk.",
        ]
    )
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
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        tool_runner=runner,
        tool_descriptions=tool_descriptions,
        hook_manager=hook_manager,
        a2a_client=a2a_client,
        agent_context=agent_context,
        status_fn=(lambda msg: console.print(f"[dim]{escape(msg)}[/dim]")),
        settings=task_runner_settings,
    )
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

    def run_turn(user_input: str) -> None:
        scheduler.update_activity()
        idle_scheduler.update_activity()
        hook_messages = hook_manager.run("on_turn_start", {"user_input": user_input})
        for msg in hook_messages:
            console.print(f"[dim]{escape(msg)}[/dim]")
        pending_question_row = db.fetch_last_asked_question()
        has_pending_question = bool(pending_question_row and pending_question_row["status"] == "asked")
        def _run_task_runner_turn() -> None:
            with console.status(
                "[bold yellow]Working... (planning tasks and tool calls)[/bold yellow]"
            ):
                result = task_runner.run_turn(user_input, prompt_fn=console.input, show_plans=show_plans)
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
            if has_pending_question:
                answer_payload = resolve_answered_question(
                    db, user_input, dict(pending_question_row), source_ref=episode.id
                )
                if answer_payload:
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
                f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
            )
            hook_messages = hook_manager.run("on_turn_end", {"episode_id": episode.id, "run_id": result.run_id})
            for msg in hook_messages:
                console.print(f"[dim]{escape(msg)}[/dim]")
        resolution = resolve_staged_items(db, user_input)
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
            vector_index.add(
                item_type="episode",
                item_id=episode.id,
                text=f"{episode.user_input}\n{episode.agent_output}",
            )
            console.print(
                f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
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
            vector_index.add(
                item_type="episode",
                item_id=episode.id,
                text=f"{episode.user_input}\n{episode.agent_output}",
            )
            console.print(
                f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
            )
            return
        if not has_pending_question:
            decision = select_question_to_ask(db, user_input)
            if decision:
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
                vector_index.add(
                    item_type="episode",
                    item_id=episode.id,
                    text=f"{episode.user_input}\n{episode.agent_output}",
                )
                console.print(
                    f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
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

    console.print("[dim]Tali chat (type 'exit' to quit)[/dim]")
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
def run_status() -> None:
    """Show the active run and its tasks."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    if not run:
        typer.echo("No active run.")
        return
    tasks = db.fetch_tasks_for_run(run["id"])
    payload = {
        "run": dict(run),
        "tasks": [
            {"ordinal": row["ordinal"], "title": row["title"], "status": row["status"]}
            for row in tasks
        ],
    }
    typer.echo(json.dumps(payload, indent=2))


@run_app.command("cancel")
def run_cancel() -> None:
    """Cancel the active run."""
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
    """Resume a blocked run."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_active_run()
    if not run:
        typer.echo("No active run to resume.")
        return
    if run["status"] != "blocked":
        typer.echo("Active run is not blocked.")
        return
    db.update_run_status(run["id"], status="active", current_task_id=run["current_task_id"], last_error=None)
    typer.echo(f"Run {run['id']} resumed.")


@run_app.command("show")
def run_show(run_id: str = typer.Argument(..., help="Run ID to display.")) -> None:
    """Show a run and its tasks."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    run = db.fetch_run(run_id)
    if not run:
        typer.echo("Run not found.")
        return
    tasks = db.fetch_tasks_for_run(run_id)
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


@app.command("name")
def agent_name() -> None:
    """Show current agent identity."""
    paths, existing = _resolve_paths()
    config = _load_or_raise_config(paths, existing)
    typer.echo(
        json.dumps(
            {"agent_id": config.agent_id, "agent_name": config.agent_name, "home": str(paths.agent_home)},
            indent=2,
        )
    )


@app.command("list")
def agent_list() -> None:
    """List known local agents."""
    paths, _ = _resolve_paths()
    registry = Registry(paths.shared_home / "registry.json")
    typer.echo(json.dumps(registry.list_agents(), indent=2))


@app.command("send")
def agent_send(
    to: Optional[str] = typer.Option(None, "--to", help="Agent name to send to (broadcast if omitted)."),
    topic: str = typer.Option("task", "--topic", help="Message topic."),
    payload_json: str = typer.Option(..., "--json", help="JSON payload."),
) -> None:
    """Send an A2A message."""
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


@app.command("inbox")
def agent_inbox() -> None:
    """Show unread A2A messages."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_unread_agent_messages(limit=20)
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command("swarm")
def swarm(prompt: str = typer.Argument(..., help="Swarm task prompt.")) -> None:
    """Run a swarm-enabled task turn."""
    paths, existing = _resolve_paths()
    _ensure_dependencies()
    config = _load_or_raise_config(paths, existing)
    db = _init_db(paths.db_path)
    llm = _build_llm_client(config.llm)
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
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        tool_runner=runner,
        tool_descriptions=tool_descriptions,
        a2a_client=a2a_client,
        settings=task_runner_settings,
    )
    result = task_runner.run_turn(prompt, prompt_fn=typer.prompt, show_plans=show_plans)
    typer.echo(result.message)


@app.command("delete-agent")
def delete_agent(agent_name: str = typer.Argument(..., help="Agent name to delete.")) -> None:
    """Delete an agent by name."""
    root = Path.home() / ".tali"
    agent_home = root / agent_name
    if not agent_home.exists():
        typer.echo("Agent not found.")
        raise typer.Exit(code=1)
    registry = Registry(root / "shared" / "registry.json")
    registry.remove(agent_name)
    typer.echo(f"Deleting agent data at {agent_home} ...")
    shutil.rmtree(agent_home)
    remaining = [p for p in root.iterdir() if p.is_dir() and p.name != "shared" and (p / "config.json").exists()]
    if not remaining:
        typer.echo("No agents remain. Launching setup to create a new agent.")
        setup()


@patch_app.command("list")
def patches_list() -> None:
    """List patch proposals."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_patch_proposals()
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


@patch_app.command("show")
def patches_show(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Show a patch proposal."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    typer.echo(json.dumps(dict(row), indent=2))


@patch_app.command("test")
def patches_test(proposal_id: str = typer.Argument(..., help="Patch proposal ID.")) -> None:
    """Run tests for a patch proposal."""
    paths, _ = _resolve_paths()
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
    results = run_patch_tests(tests, cwd=Path.cwd())
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
    """Apply a patch proposal after tests pass."""
    paths, _ = _resolve_paths()
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
    error = apply_patch(row["diff_text"], cwd=Path.cwd())
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
    """Reject a patch proposal."""
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
    """Rollback an applied patch proposal."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    row = db.fetch_patch_proposal(proposal_id)
    if not row:
        typer.echo("Patch proposal not found.")
        return
    if row["status"] != "applied":
        typer.echo("Patch proposal is not applied.")
        return
    error = reverse_patch(row["diff_text"], cwd=Path.cwd())
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


@app.command()
def sleep(
    output_dir: Optional[Path] = typer.Option(None, help="Directory to write sleep output."),
    apply: Optional[Path] = typer.Option(
        None, "--apply", help="Apply a sleep JSON output file and insert vetted facts."
    ),
) -> None:
    """Run sleep consolidation (offline) or apply a sleep output file."""
    paths, _ = _resolve_paths()
    _ensure_dependencies()
    config = _load_or_raise_config(paths)
    db = _init_db(paths.db_path)
    llm = _build_llm_client(config.llm)
    embedder = _build_embedder(config.embeddings)
    vector_index = VectorIndex(paths.vector_dir / "memory.index", config.embeddings.dim, embedder)
    if apply:
        snapshot = create_snapshot(paths.data_dir)
        try:
            payload = load_sleep_output(apply)
            result = apply_sleep_output(db, payload)
            for fact_id in result.inserted_fact_ids:
                facts = db.fetch_facts_by_ids([fact_id])
                if facts:
                    vector_index.add(item_type="fact", item_id=fact_id, text=facts[0]["statement"])
            typer.echo(
                json.dumps(
                    {
                        "inserted_fact_ids": result.inserted_fact_ids,
                        "skipped_candidates": result.skipped_candidates,
                        "contested_fact_ids": result.contested_fact_ids,
                        "staged_item_ids": result.staged_item_ids,
                        "snapshot_id": snapshot.id,
                    },
                    indent=2,
                )
            )
        except Exception:
            rollback_snapshot(paths.data_dir, snapshot)
            raise
        return
    if output_dir:
        target_dir = output_dir
        output_path = run_sleep(db, target_dir, llm=llm)
        typer.echo(f"Sleep output written to {output_path}")
        return
    last_run = db.last_sleep_run()
    last_time = last_run["timestamp"] if last_run else "never"
    episodes_since = db.count_episodes_since_last_sleep()
    typer.echo(
        json.dumps(
            {
                "last_sleep": last_time,
                "episodes_since_last_sleep": episodes_since,
                "message": "Sleep runs automatically in chat.",
            },
            indent=2,
        )
    )


@app.command()
def commitments() -> None:
    """List commitments."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_commitments()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def facts() -> None:
    """List facts."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_facts()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def skills() -> None:
    """List skills."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    rows = db.list_skills()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def diff() -> None:
    """Show differences between the latest snapshot and current data."""
    paths, _ = _resolve_paths()
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `agent snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    typer.echo(diff_snapshot(paths.data_dir, latest))


@app.command()
def rollback() -> None:
    """Rollback data directory to the latest snapshot."""
    paths, _ = _resolve_paths()
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `agent snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    rollback_snapshot(paths.data_dir, latest)
    typer.echo(f"Rolled back data to snapshot {latest.id}.")


@app.command()
def snapshot() -> None:
    """Create a snapshot of the data directory."""
    paths, _ = _resolve_paths()
    snapshot = create_snapshot(paths.data_dir)
    typer.echo(f"Snapshot created: {snapshot.id}")


@app.command()
def doctor() -> None:
    """Validate core invariants and report staged items."""
    paths, _ = _resolve_paths()
    db = _init_db(paths.db_path)
    violations: list[str] = []
    facts = db.list_facts()
    for row in facts:
        if row["provenance_type"] == "AGENT_OUTPUT":
            violations.append(f"Fact {row['id']} has forbidden provenance.")
        if not row["source_ref"]:
            violations.append(f"Fact {row['id']} missing source_ref.")
        confidence = float(row["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            violations.append(f"Fact {row['id']} has confidence out of range.")
    staged_count = db.count_staged_items()
    oldest = db.oldest_pending_staged_item()
    report = {
        "violations": violations,
        "staged_items": staged_count,
        "oldest_pending_staged_item": dict(oldest) if oldest else None,
    }
    typer.echo(json.dumps(report, indent=2))
    if violations:
        raise typer.Exit(code=1)


@app.command()
def setup() -> None:
    """Walk through configuration and install dependencies."""
    root, agent_name, existing_config = resolve_agent(prompt_fn=typer.prompt, allow_create_config=True)
    paths = load_paths(root, agent_name)
    typer.echo("Setting up Tali configuration.")
    _ensure_dependencies()
    provider = typer.prompt("LLM provider (openai/ollama)", default="ollama")
    if provider not in {"openai", "ollama"}:
        typer.echo("Provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if provider == "openai":
        model = typer.prompt("LLM model", default="gpt-4o-mini")
    else:
        model = _prompt_ollama_model("LLM model", default="llama3")
    base_url = typer.prompt(
        "LLM base URL", default="https://api.openai.com/v1" if provider == "openai" else "http://localhost:11434"
    )
    api_key = None
    if provider == "openai":
        api_key = typer.prompt("OpenAI API key", hide_input=True)
    else:
        _prompt_ollama_download(model)

    embed_provider = typer.prompt("Embedding provider (openai/ollama)", default=provider)
    if embed_provider not in {"openai", "ollama"}:
        typer.echo("Embedding provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    if embed_provider == "openai":
        embed_model = typer.prompt("Embedding model", default="text-embedding-3-small")
    else:
        embed_model = _prompt_ollama_model("Embedding model", default="nomic-embed-text")
    embed_base_url = typer.prompt(
        "Embedding base URL", default=base_url if embed_provider == provider else "http://localhost:11434"
    )
    embed_api_key = None
    if embed_provider == "openai":
        embed_api_key = typer.prompt("OpenAI API key for embeddings", hide_input=True)
    else:
        _prompt_ollama_download(embed_model)
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
        capabilities=capabilities,
        llm=None,
        embeddings=None,
        tools=None,
        task_runner=None,
    )
    save_config(paths.config_path, config)
    shared_settings = SharedSettings(
        llm=LLMSettings(provider=provider, model=model, base_url=base_url, api_key=api_key),
        embeddings=EmbeddingSettings(
            provider=embed_provider,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
            dim=int(embed_dim),
        ),
        tools=ToolSettings(fs_root=str(Path.home())),
    )
    save_shared_settings(paths.shared_home / "config.json", shared_settings)
    paths.vector_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    paths.sleep_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.patches_dir.mkdir(parents=True, exist_ok=True)
    paths.inbox_dir.mkdir(parents=True, exist_ok=True)
    paths.outbox_dir.mkdir(parents=True, exist_ok=True)
    (paths.shared_home / "locks").mkdir(parents=True, exist_ok=True)
    typer.echo(f"Config saved to {paths.config_path}")
    subprocess.run(["python", "-m", "pip", "install", "-e", "."], check=True)
