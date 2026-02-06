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

The setup flow writes `~/.tali/<agent_name>/config.json` with the LLM and embedding provider settings.
Supported providers: OpenAI-compatible APIs (`openai`) and Ollama (`ollama`).

## Agent identity + storage

Each agent has its own home directory:

```
~/.tali/
  shared/
  <agent_name>/
```

Per-agent storage is isolated under `~/.tali/<agent_name>/`. Shared A2A infra
and registry live under `~/.tali/shared/`.

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

## Idle self-improvement

When the agent is idle for 5 minutes (and no active run exists), Tali runs a
low-priority, interruptible idle cycle. Idle work is locked to a single
instance and rate-limited.

Idle jobs (in order):

* Skill review: stages improvement ideas as skill proposals (not facts).
* Memory hygiene: confidence decay, dedupe, and low-importance pruning.
* Knowledge expansion: pulls excerpts from trusted sources only (framework).
* Clarifying questions: queues at most one question per idle cycle.
* Patch proposals: drafts gated code changes (never auto-applied).

## Question queue

Idle jobs can queue clarifying questions. At most one question is asked per
user turn, and only when relevant or high priority. Answers are staged as
USER_REPORTED fact candidates (never direct facts).

## Hooks and safe extensions

Hooks are optional plugins loaded from `src/tali/hooks/`. Each hook declares:

* name
* triggers (e.g., `on_turn_start`, `on_turn_end`, `on_idle`, `on_sleep_complete`)
* handler(context) -> actions

Hooks run in a time-bounded sandbox and use safe APIs for staging items or
queuing questions. They cannot directly write facts.

## Patch proposals (gated)

Idle jobs may draft patch proposals, but they are never auto-applied. Proposals
must be tested and explicitly applied or rejected.

```bash
tali patches list
tali patches show <id>
tali patches test <id>
tali patches apply <id>
tali patches reject <id>
tali patches rollback <id>
```

## A2A (agent-to-agent)

Agents coordinate using a shared local message bus and registry.

```bash
tali name
tali list
tali send --to ops --topic task --json '{...}'
tali inbox
tali swarm "Coordinate a multi-step task"
```

All A2A messages are stored as message logs and tagged as `RECEIVED_AGENT`.
They are never promoted to facts without explicit verification.

## Core invariants

* The LLM never learns.
* Memory is typed and sourced.
* Agent outputs are not facts.
* Consolidation is offline.
* Contradictions fork, never overwrite.
* Recursion is budgeted.
* Commitments outlive sessions.
