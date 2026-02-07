# Tali

Tali is a persistent, trustworthy, non-self-poisoning agent. It stores memory in
SQLite, keeps a vector index for retrieval, and uses offline consolidation to
promote facts and commitments safely.

## Quick start

```bash
bash ./quickstart.sh
```

The quickstart script creates a virtualenv, installs Tali, and tells you to run
`tali` to bootstrap your first agent and enter the Tali shell. Commands are
entered without the `tali` prefix once you're inside the shell.

### Manual setup (optional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Bootstrap your first agent (includes guided setup)
tali

# Or create additional agents later (inside the Tali shell)
agent create
```

The setup flow writes `~/.tali/<agent_name>/config.json` with the LLM and embedding provider settings.
Supported providers: OpenAI-compatible APIs (`openai`) and Ollama (`ollama`).

### Local models (Ollama)

Tali talks to a running Ollama server via its local HTTP API (the SDK is just a client).
You still need the Ollama app/daemon installed and running. The CLI is only used
to list or pull models during setup; if you already manage models yourself, you
can enter the model name manually.

### Local models (OpenAI-compatible servers)

You can also use any OpenAI-compatible local server (LocalAI, vLLM, TGI, llama.cpp server).
Set the provider to `openai` and point the base URL at your local server (for example,
`http://localhost:8000/v1`). API keys are optional for local endpoints and can be left blank.
When you pick a curated local model during agent creation, Tali downloads the GGUF file into
`~/.tali/models`. Configure your local server to load that GGUF file and expose a model name
that matches the value stored in your agent config.

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
uncertain items, and resolves them opportunistically without any CLI commands.

## Memory pipeline

Tali uses a two-path memory flow:

* Realtime ingestion: tool results and episodes are staged as fact candidates
  (TOOL_VERIFIED / SYSTEM_OBSERVED) for fast recall.
* Sleep consolidation: deeper extraction, contradiction handling, and promotion
  policies run offline with stronger gates.

When responses reference memory, the agent includes explicit citations like
`[fact:<id>]`, `[commitment:<id>]`, or `[preference:<key>]`.

## Snapshots

Create a snapshot, inspect differences, or rollback data:

```bash
snapshot
diff
rollback
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
patches list
patches show <id>
patches test <id>
patches apply <id>
patches reject <id>
patches rollback <id>
```

## Logs and run browser

Structured logs are written under `~/.tali/<agent>/logs/` for each run.
Quick commands:

```bash
logs --limit 20
dashboard --duration 30
run list
run timeline <run_id>
```

## A2A (agent-to-agent)

Agents coordinate using a shared local message bus and registry.

```bash
name
list
send --to ops --topic task --json '{...}'
inbox
swarm "Coordinate a multi-step task"
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
