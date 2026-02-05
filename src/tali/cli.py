from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from dataclasses import replace
from uuid import uuid4

from tali.config import (
    AppConfig,
    EmbeddingSettings,
    GuardrailConfig,
    LLMSettings,
    RetrievalConfig,
    ToolSettings,
    Paths,
    load_config,
    load_paths,
    save_config,
)
from tali.consolidation import apply_sleep_output
from tali.db import Database
from tali.episode import build_episode, build_prompt
from tali.embeddings import OllamaEmbeddingClient, OpenAIEmbeddingClient
from tali.guardrails import Guardrails
from tali.llm import OllamaClient, OpenAIClient
from tali.retrieval import Retriever
from tali.self_care import SleepScheduler, resolve_staged_items
from tali.sleep import load_sleep_output, run_sleep
from tali.snapshots import create_snapshot, diff_snapshot, list_snapshots, rollback_snapshot
from tali.vector_index import VectorIndex
from tali.approvals import ApprovalManager
from tali.tools.protocol import (
    build_phase1_prompt,
    build_phase2_prompt,
    parse_phase1_plan,
)
from tali.tools.registry import build_default_registry
from tali.tools.policy import ToolPolicy
from tali.tools.runner import ToolRunner

app = typer.Typer(help="Tali agent CLI")


def _init_db(db_path: Path) -> Database:
    db = Database(db_path)
    db.initialize()
    return db


def _load_or_raise_config(paths: Paths) -> AppConfig:
    if not paths.config_path.exists():
        typer.echo("Config not found. Run `agent setup` to configure LLMs.")
        raise typer.Exit(code=1)
    return load_config(paths.config_path)


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
) -> None:
    """Start a chat loop or run a single turn if message is provided."""
    paths = load_paths()
    _ensure_dependencies()
    config = _load_or_raise_config(paths)
    db = _init_db(paths.db_path)
    llm = _build_llm_client(config.llm)
    embedder = _build_embedder(config.embeddings)
    vector_index = VectorIndex(paths.vector_dir / "memory.index", config.embeddings.dim, embedder)
    retriever = Retriever(db, RetrievalConfig(), vector_index=vector_index)
    guardrails = Guardrails(GuardrailConfig())
    scheduler = SleepScheduler(paths.data_dir, db, llm, vector_index)
    scheduler.start()
    console = Console()
    tool_settings = config.tools if isinstance(config.tools, ToolSettings) else ToolSettings()
    if tool_settings.approval_mode not in {"prompt", "auto_approve_safe", "deny"}:
        tool_settings = replace(tool_settings, approval_mode="prompt")
    registry = build_default_registry(paths, tool_settings)
    policy = ToolPolicy(tool_settings, registry, paths)
    approvals = ApprovalManager(mode=tool_settings.approval_mode)
    runner = ToolRunner(registry, policy, approvals, tool_settings, paths)
    tool_descriptions = registry.describe_tools()

    def _needs_tools_hint(text: str) -> bool:
        keywords = [
            "file",
            "write",
            "create",
            "save",
            "desktop",
            "folder",
            "list",
            "read",
            "fetch",
            "download",
            "search",
            "web",
            "http",
            "command",
            "shell",
            "run",
            "git",
        ]
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    def run_turn(user_input: str) -> None:
        scheduler.update_activity()
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
        retrieval_context = retriever.retrieve(user_input)
        tool_plan_prompt = build_phase1_prompt(retrieval_context.bundle, user_input, tool_descriptions)
        with console.status("[bold yellow]Planning tools...[/bold yellow]"):
            plan_response = llm.generate(tool_plan_prompt)
        if verbose_tools:
            console.print("[dim]Phase 1 raw response:[/dim]")
            console.print(f"[dim]{escape(plan_response.content)}[/dim]")
        plan, _ = parse_phase1_plan(plan_response.content)

        if plan and not plan.need_tools and _needs_tools_hint(user_input):
            retry_prompt = (
                tool_plan_prompt
                + "\n\nIf the user request requires tools, you MUST return need_tools=true and tool_calls."
            )
            with console.status("[bold yellow]Replanning tools...[/bold yellow]"):
                retry_response = llm.generate(retry_prompt)
            if verbose_tools:
                console.print("[dim]Phase 1 retry raw response:[/dim]")
                console.print(f"[dim]{escape(retry_response.content)}[/dim]")
            plan, _ = parse_phase1_plan(retry_response.content)
            if plan and not plan.need_tools:
                message = "Tool planning failed for a tool-requiring request. Please ask again."
                guardrail = guardrails.enforce(message, retrieval_context.bundle)
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
                console.print(
                    f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
                )
                return

        tool_calls_log: list[dict[str, str]] = []
        tool_results = []
        tool_records = []

        if plan is None:
            warning = "I couldn't form a valid tool plan; ask again."
            prompt = build_prompt(retrieval_context.bundle, user_input)
            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                response = llm.generate(prompt)
            guardrail = guardrails.enforce(f"{warning}\n\n{response.content}", retrieval_context.bundle)
            episode = build_episode(
                user_input=user_input,
                guardrail=guardrail,
                tool_calls=[],
                outcome="ok",
                quarantine=1 if guardrail.flags else 0,
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

        if not plan.need_tools and not plan.final_answer_allowed:
            message = "I cannot answer without tools. Please request verification."
            guardrail = guardrails.enforce(message, retrieval_context.bundle)
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
            console.print(
                f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
            )
            return

        if plan.need_tools and not plan.tool_calls:
            message = "Tool plan required tools but none were provided. Please ask again."
            guardrail = guardrails.enforce(message, retrieval_context.bundle)
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
            console.print(
                f"[bold green]Tali[/bold green]: {escape(episode.agent_output)}"
            )
            return

        if plan.need_tools and plan.tool_calls:
            mapped_calls = []
            for call in plan.tool_calls:
                mapped_calls.append(
                    replace(call, id=f"tc_{uuid4().hex}")
                )
            tool_results, tool_records = runner.run(
                mapped_calls, prompt_fn=console.input
            )
            record_refs: dict[str, str] = {
                record.id: f"tool_call:{record.id}" for record in tool_records
            }
            tool_results = [
                replace(result, result_ref=record_refs.get(result.id, result.result_ref))
                for result in tool_results
            ]
            for record in tool_records:
                record_ref = record_refs.get(record.id, "")
                tool_calls_log.append(
                    {
                        "id": record.id,
                        "name": record.name,
                        "status": record.status,
                        "approval_mode": record.approval_mode or "",
                        "result_ref": record_ref,
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

        if plan.need_tools and plan.tool_calls:
            raw_outputs: list[str] = []
            for result in tool_results:
                if result.status != "ok" or not result.result_raw:
                    continue
                raw = result.result_raw
                if len(raw) > tool_settings.tool_result_max_bytes:
                    raw = raw[: tool_settings.tool_result_max_bytes] + "\n[truncated]"
                raw_outputs.append(f"{result.id} {result.name} {result.status}\n{raw}")
            raw_tool_output = "\n".join(raw_outputs)
            prompt = build_phase2_prompt(
                retrieval_context.bundle, user_input, tool_results, raw_tool_output
            )
        else:
            prompt = build_prompt(retrieval_context.bundle, user_input)

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            response = llm.generate(prompt)
        if tool_results and any(result.status == "ok" for result in tool_results):
            refusal_markers = ["i cannot", "i can't", "do not have the ability", "cannot directly"]
            response_lower = response.content.lower()
            if any(marker in response_lower for marker in refusal_markers):
                summaries = [
                    result.result_summary
                    for result in tool_results
                    if result.status == "ok" and result.result_summary
                ]
                summary_text = "; ".join(summaries) if summaries else "Completed requested tool actions."
                response = replace(
                    response,
                    content=f"{summary_text}\n\nWhat would you like me to do next?",
                )
        guardrail = guardrails.enforce(response.content, retrieval_context.bundle)
        episode = build_episode(
            user_input=user_input,
            guardrail=guardrail,
            tool_calls=tool_calls_log,
            outcome="ok",
            quarantine=1 if guardrail.flags else 0,
        )
        db.insert_episode(
            episode_id=episode.id,
            user_input=episode.user_input,
            agent_output=episode.agent_output,
            tool_calls=episode.tool_calls,
            outcome=episode.outcome,
            quarantine=episode.quarantine,
        )
        for record in tool_records:
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

    if message:
        run_turn(message)
        scheduler.stop()
        return

    console.print("[dim]Tali chat (type 'exit' to quit)[/dim]")
    while True:
        user_input = console.input("[bold cyan]You[/bold cyan]: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        run_turn(user_input)
    scheduler.stop()


@app.command()
def sleep(
    output_dir: Optional[Path] = typer.Option(None, help="Directory to write sleep output."),
    apply: Optional[Path] = typer.Option(
        None, "--apply", help="Apply a sleep JSON output file and insert vetted facts."
    ),
) -> None:
    """Run sleep consolidation (offline) or apply a sleep output file."""
    paths = load_paths()
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
    paths = load_paths()
    db = _init_db(paths.db_path)
    rows = db.list_commitments()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def facts() -> None:
    """List facts."""
    paths = load_paths()
    db = _init_db(paths.db_path)
    rows = db.list_facts()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def skills() -> None:
    """List skills."""
    paths = load_paths()
    db = _init_db(paths.db_path)
    rows = db.list_skills()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def diff() -> None:
    """Show differences between the latest snapshot and current data."""
    paths = load_paths()
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `agent snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    typer.echo(diff_snapshot(paths.data_dir, latest))


@app.command()
def rollback() -> None:
    """Rollback data directory to the latest snapshot."""
    paths = load_paths()
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
    paths = load_paths()
    snapshot = create_snapshot(paths.data_dir)
    typer.echo(f"Snapshot created: {snapshot.id}")


@app.command()
def doctor() -> None:
    """Validate core invariants and report staged items."""
    paths = load_paths()
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
    paths = load_paths()
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

    config = AppConfig(
        llm=LLMSettings(provider=provider, model=model, base_url=base_url, api_key=api_key),
        embeddings=EmbeddingSettings(
            provider=embed_provider,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
            dim=int(embed_dim),
        ),
        tools=ToolSettings(),
    )
    save_config(paths.config_path, config)
    typer.echo(f"Config saved to {paths.config_path}")
    subprocess.run(["python", "-m", "pip", "install", "-e", "."], check=True)
