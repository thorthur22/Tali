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

Enable shell tab-completion for faster command entry:

```bash
tali --install-completion
```

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

## Agent code worktrees

Each agent gets its own git worktree copy of the Tali codebase at:

```
~/.tali/<agent_name>/code
```

`agent chat <name>` ensures the worktree exists and auto-syncs it from the main
repo before starting the agent. If a merge conflict happens, the agent is told
to resolve the conflict in its worktree.

This worktree is the execution context for:

* Patch proposal tests (`patches test <id>`)
* Patch application/rollback (`patches apply|rollback <id>`)
* Hooks loaded from `src/tali/hooks/` (from the agent's worktree copy)

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

## Autonomy mode

By default agents only act in response to user commands. When autonomy is
enabled, the agent can self-start tasks and continue multi-step runs without
waiting for explicit prompts.

### Enabling autonomy

Use the CLI to enable autonomy without editing JSON:

```bash
agent config set autonomy.enabled true
agent config set autonomy.auto_continue true
```

Or add an `"autonomy"` section to `config.json` directly:

```json
{
  "autonomy": {
    "enabled": true,
    "auto_continue": true,
    "execute_commitments": true,
    "idle_trigger_seconds": 300,
    "idle_min_interval_seconds": 1800,
    "autonomous_idle_delay_seconds": 30,
    "max_autonomous_llm_calls_per_turn": 20
  }
}
```

All fields are optional and default to the values shown above (except
`enabled` and `auto_continue` which default to `false`).

### What autonomy does

* **Commitment execution**: When the agent is idle it checks the commitments
  table for pending items (or commitments whose `due_date` has elapsed). The
  highest-priority commitment is picked up, marked active, and executed via the
  task runner as if the user had typed the commitment description. When the run
  finishes the commitment is marked done; on failure it is marked failed.
* **Auto-continue**: If a self-initiated run is left in a blocked or active
  state the idle scheduler automatically resumes it by sending a `"continue"`
  turn, looping until the run completes or hits budget limits.
* **Configurable timing**: `idle_trigger_seconds` and
  `idle_min_interval_seconds` replace the hardcoded 5-minute and 30-minute
  defaults, letting you tune how quickly the agent starts autonomous work.
  `autonomous_idle_delay_seconds` controls the minimum idle time before the
  agent picks up a commitment or resumes a blocked run (default 30 s).

### Run origin tracking

Runs created by the autonomy system are tagged with `origin = 'autonomous'`
in the `runs` table, distinguishing them from user-initiated runs
(`origin = 'user'`). This allows the idle scheduler to manage only its own
runs without interfering with user work.

### Safety and interruption

* Autonomous runs use `auto_approve_safe` tool approval semantics. Any tool
  call that would normally require explicit user approval is denied
  automatically.
* User input always takes priority. When the user sends a message,
  `update_activity()` sets the interrupt event **and** pauses any in-progress
  autonomous run (setting it to `blocked` with `interrupted_by_user`). The
  commitment is reverted to `pending` if it was interrupted mid-execution.
* All autonomous work is bounded by the same `TaskRunnerSettings` budgets
  (max tasks, LLM calls, steps) as user-initiated runs.

## Question queue

Idle jobs can queue clarifying questions. At most one question is asked per
user turn, and only when relevant or high priority. Answers are staged as
USER_REPORTED fact candidates (never direct facts).

## Request classification and model routing

Each user turn is scored by a lightweight rule-based classifier that maps the
request into a tier (SIMPLE, MEDIUM, COMPLEX, REASONING). The tier influences
how the agent routes work:

* SIMPLE: responder-only for quick replies with no task decomposition.
* MEDIUM/COMPLEX/REASONING: planner + tool runner for multi-step work.

If a task run is already active, the responder checks whether a new prompt is
related to that run before continuing it. Unrelated prompts are answered
directly without canceling the active run.

## Hooks and safe extensions

Hooks are optional plugins loaded from `src/tali/hooks/`. Each hook declares:

* name
* triggers (e.g., `on_turn_start`, `on_turn_end`, `on_idle`, `on_sleep_complete`)
* handler(context) -> actions

Hooks run in a time-bounded sandbox and use safe APIs for staging items or
queuing questions. They cannot directly write facts.

## Configuration management

View and update agent configuration from the CLI without editing JSON files:

```bash
agent config show               # display all settings in grouped panels
agent config show --json        # raw JSON for scripting
agent config set responder_llm.model gemma-3b-v2
agent config set tools.approval_mode auto_approve_safe
agent config set autonomy.idle_trigger_seconds 600
agent config reset autonomy.idle_trigger_seconds   # restore default
```

Dot-notation paths follow the config sections: `planner_llm`, `responder_llm`,
`embeddings`, `tools`, `task_runner`, `autonomy`.

## Memory search

Search across the agent's stored facts and conversation episodes:

```bash
agent memory search "project deadline"
agent memory search "deployment" --facts-only --limit 5
agent memory search "error logs" --episodes-only --json
```

## Commitments

List existing commitments or add new ones from the CLI:

```bash
agent commitments
agent commitment-add "Review pull request #42" --due 2026-03-01 --priority 2
```

Commitments added via CLI are picked up automatically by the agent during
its next cycle (especially when autonomy is enabled).

## Preferences

Manage user preferences that influence agent behaviour:

```bash
agent preferences list
agent preferences set humor off
agent preferences set coding_style "functional" --confidence 0.95
agent preferences remove humor
```

Preferences set via CLI use `USER_EXPLICIT` provenance with configurable
confidence scores.

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
run show <run_id>
run timeline <run_id>
```

The `dashboard` command shows a live-updating Rich panel with agent state
(active/idle), unread inbox count, pending commitments, memory stats, run
metrics, current tasks, and recent logs.

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

## CLI output formatting

All data-listing commands default to human-friendly Rich tables and panels.
Add `--json` to any command for machine-readable JSON output suitable for
scripting or piping:

```bash
agent facts               # Rich table
agent facts --json        # raw JSON
agent inbox --json | jq .
agent run list --json
```

Run `agent help` for a quick-reference guide of common workflows.

## Core invariants

* The LLM never learns.
* Memory is typed and sourced.
* Agent outputs are not facts.
* Consolidation is offline.
* Contradictions fork, never overwrite.
* Recursion is budgeted.
* Commitments outlive sessions.
