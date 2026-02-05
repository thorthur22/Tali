# Tali

Tali is a persistent, trustworthy, non-self-poisoning agent. It stores memory in
SQLite, keeps a vector index for retrieval, and uses offline consolidation to
promote facts and commitments safely.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run guided setup
agent setup

# Start a chat loop
agent chat
```

The setup flow writes `data/config.json` with the LLM and embedding provider settings.
Supported providers: OpenAI-compatible APIs (`openai`) and Ollama (`ollama`).

## Sleep consolidation

Sleep runs automatically in chat when thresholds are met (episode count, idle
time, or periodic timer). It snapshots before applying safe updates, stages
uncertain items, and resolves them opportunistically.

You can still inspect status or apply a JSON file manually:

```bash
agent sleep
agent sleep --apply data/sleep/<sleep_output.json>
```

## Snapshots

Create a snapshot, inspect differences, or rollback data:

```bash
agent snapshot
agent diff
agent rollback
```

## Core invariants

* The LLM never learns.
* Memory is typed and sourced.
* Agent outputs are not facts.
* Consolidation is offline.
* Contradictions fork, never overwrite.
* Recursion is budgeted.
* Commitments outlive sessions.
