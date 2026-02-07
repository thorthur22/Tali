# Tali Improvement Review

**Scope**
- This list is based on a repo walkthrough of core runtime, memory, tools, idle/sleep, and CLI.

**Agentic Behavior**
- Inject retrieval context into every LLM prompt used by planning and execution. Today memory is only used in guardrails, while planning prompts in `/Users/Thor/Documents/GitHub/Tali/src/tali/task_runner.py` and `/Users/Thor/Documents/GitHub/Tali/src/tali/tasking.py` ignore it.
- Unify or remove unused prompt builders. `/Users/Thor/Documents/GitHub/Tali/src/tali/episode.py` and `/Users/Thor/Documents/GitHub/Tali/src/tali/tools/protocol.py` define memory-aware prompts that are never used.
- Store a persistent run summary (goal, constraints, current status) and feed it to action planning so tasks can resume coherently across turns.
- Add a stuck-detector that proposes alternate strategies before asking the user (e.g., if repeated tool signatures exceed a threshold, suggest a different path or tool).

**Memory System**
- Use facts, commitments, preferences, and skills in task decomposition, action planning, and completion review prompts instead of only for memory-claim blocking.
- Introduce explicit memory citations in outputs and update guardrails to validate citations rather than forbidding all memory claims.
- Add a realtime memory ingestion pass after each tool result and episode to stage TOOL_VERIFIED and SYSTEM_OBSERVED facts, instead of waiting only for sleep consolidation.
- Add a preference-extraction pipeline from explicit user directives and apply preferences as constraints during planning.
- Implement skill execution: include skill steps in prompts, allow a skill to be invoked as an action, and track success/failure to tune triggers.
- Improve retrieval ranking with recency, confidence, and commitment priority weighting, plus MMR to avoid duplicates.
- Keep the vector index in sync with DB pruning and deletions; add rebuild/compaction when embedding dimensions change.

**Tooling**
- Add richer file tools: `fs.search`, `fs.glob`, `fs.tree`, `fs.read_lines`, `fs.write_patch`, `fs.copy`, `fs.move`, and `fs.delete` with explicit approvals.
- Expand the safe shell allowlist to cover additional read-only git commands like `git show` and `git ls-files`, and add timeouts to all shell runs.
- Add a web search tool so `web.fetch` is not blind and can discover sources before fetching.
- Cache tool results and surface summaries to planning to reduce repeated tool calls.
- Tag tool outputs with provenance and auto-stage fact candidates directly from tool results.

**Idle and Sleep**
- Register knowledge sources by default, at least `LocalFileSource` rooted to the agent home or configured paths, so knowledge expansion runs.
- Add stronger quality gates for sleep and idle outputs beyond schema checks, including source verification, dedupe for skills and commitments, and contradiction handling.
- Make idle jobs aware of active commitments and run goals to avoid irrelevant patch proposals or questions.

**CLI and UX**
- Build a richer TUI using Rich Live or Textual with panels for tasks, tool activity, memory status, and logs.
- Stream LLM output and show progress bars for long-running tools; display current task and next action in real time.
- Replace the current setup flow with a guided wizard that validates provider connectivity and lists available models before saving config.
- Add an interactive run browser and timeline: run list, task status, tool calls, approvals, and resume/cancel actions.
- Provide a notification area for clarifying questions, staged items, and sleep outputs so the agent feels alive and busy.

**Observability and Ops**
- Write structured logs to `logs_dir` and attach run ids; add a `tali logs` command for quick inspection.
- Emit metrics for LLM calls, tool calls, and time budgets; surface them in `tali run status` and the CLI UI.
- Extend `tali doctor` to include vector index health, orphaned records, and staged item backlog.

**Docs and Tests**
- Document the memory pipeline end-to-end, including when facts/commitments/skills are written and how they are retrieved.
- Add integration tests that cover memory injection into prompts, skill execution, tool policy decisions, and CLI flows.
