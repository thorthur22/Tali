from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import typer

from tali.config import (
    AppConfig,
    EmbeddingSettings,
    GuardrailConfig,
    LLMSettings,
    RetrievalConfig,
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


@app.command()
def chat(message: Optional[str] = typer.Argument(None, help="Message to send to the agent.")) -> None:
    """Start a chat loop or run a single turn if message is provided."""
    paths = load_paths(Path.cwd())
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
            typer.echo(episode.agent_output)
            return
        retrieval_context = retriever.retrieve(user_input)
        prompt = build_prompt(retrieval_context.bundle, user_input)
        response = llm.generate(prompt)
        guardrail = guardrails.enforce(response.content, retrieval_context.bundle)
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
        typer.echo(episode.agent_output)

    if message:
        run_turn(message)
        scheduler.stop()
        return

    typer.echo("Tali chat (type 'exit' to quit)")
    while True:
        user_input = typer.prompt("You")
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
    paths = load_paths(Path.cwd())
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
    paths = load_paths(Path.cwd())
    db = _init_db(paths.db_path)
    rows = db.list_commitments()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def facts() -> None:
    """List facts."""
    paths = load_paths(Path.cwd())
    db = _init_db(paths.db_path)
    rows = db.list_facts()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def skills() -> None:
    """List skills."""
    paths = load_paths(Path.cwd())
    db = _init_db(paths.db_path)
    rows = db.list_skills()
    typer.echo(json.dumps([dict(row) for row in rows], indent=2))


@app.command()
def diff() -> None:
    """Show differences between the latest snapshot and current data."""
    paths = load_paths(Path.cwd())
    snapshots = list_snapshots(paths.data_dir)
    if not snapshots:
        typer.echo("No snapshots found. Run `agent snapshot` first.")
        raise typer.Exit(code=1)
    latest = snapshots[-1]
    typer.echo(diff_snapshot(paths.data_dir, latest))


@app.command()
def rollback() -> None:
    """Rollback data directory to the latest snapshot."""
    paths = load_paths(Path.cwd())
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
    paths = load_paths(Path.cwd())
    snapshot = create_snapshot(paths.data_dir)
    typer.echo(f"Snapshot created: {snapshot.id}")


@app.command()
def doctor() -> None:
    """Validate core invariants and report staged items."""
    paths = load_paths(Path.cwd())
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
    paths = load_paths(Path.cwd())
    typer.echo("Setting up Tali configuration.")
    _ensure_dependencies()
    provider = typer.prompt("LLM provider (openai/ollama)", default="ollama")
    if provider not in {"openai", "ollama"}:
        typer.echo("Provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    model = typer.prompt("LLM model", default="gpt-4o-mini" if provider == "openai" else "llama3")
    base_url = typer.prompt(
        "LLM base URL", default="https://api.openai.com/v1" if provider == "openai" else "http://localhost:11434"
    )
    api_key = None
    if provider == "openai":
        api_key = typer.prompt("OpenAI API key", hide_input=True)

    embed_provider = typer.prompt("Embedding provider (openai/ollama)", default=provider)
    if embed_provider not in {"openai", "ollama"}:
        typer.echo("Embedding provider must be 'openai' or 'ollama'.")
        raise typer.Exit(code=1)
    embed_model = typer.prompt(
        "Embedding model", default="text-embedding-3-small" if embed_provider == "openai" else "nomic-embed-text"
    )
    embed_base_url = typer.prompt(
        "Embedding base URL", default=base_url if embed_provider == provider else "http://localhost:11434"
    )
    embed_api_key = None
    if embed_provider == "openai":
        embed_api_key = typer.prompt("OpenAI API key for embeddings", hide_input=True)
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
    )
    save_config(paths.config_path, config)
    typer.echo(f"Config saved to {paths.config_path}")
    subprocess.run(["python", "-m", "pip", "install", "-e", "."], check=True)
