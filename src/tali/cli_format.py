"""Rich formatting helpers for Tali CLI output.

Each function accepts raw data (dicts, sqlite3.Row objects, or dataclass
instances) and returns a Rich renderable (Table, Panel, Group, etc.).
"""

from __future__ import annotations

import json
from dataclasses import fields as dc_fields
from typing import Any, Sequence

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def short_id(uuid_str: str | None, length: int = 8) -> str:
    """Return the first *length* characters of a UUID string."""
    if not uuid_str:
        return "-"
    return str(uuid_str)[:length]


_STATUS_COLORS: dict[str, str] = {
    # generic
    "active": "bold green",
    "pending": "yellow",
    "done": "green",
    "completed": "green",
    "failed": "bold red",
    "canceled": "dim",
    "cancelled": "dim",
    "blocked": "bold yellow",
    "skipped": "dim",
    # patches
    "proposed": "cyan",
    "tested": "blue",
    "approved": "bold green",
    "applied": "green",
    "rejected": "red",
    "review_failed": "bold red",
    # messages
    "unread": "bold cyan",
    "read": "dim",
    "processed": "dim",
}


def status_color(status: str | None) -> str:
    """Wrap *status* in Rich markup colour."""
    if not status:
        return "-"
    colour = _STATUS_COLORS.get(status.lower(), "")
    if colour:
        return f"[{colour}]{status}[/{colour}]"
    return status


def truncate(text: str | None, max_len: int = 80) -> str:
    """Safely truncate text with an ellipsis."""
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _row_dict(row: Any) -> dict[str, Any]:
    """Convert a sqlite3.Row (or dict) to a plain dict."""
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Inbox
# ---------------------------------------------------------------------------

def format_inbox(rows: Sequence[Any]) -> Table:
    """Pretty-print unread A2A messages."""
    table = Table(title="Inbox  \u2709", show_lines=True)
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("From", style="cyan")
    table.add_column("Topic", style="bold")
    table.add_column("Timestamp")
    table.add_column("Status")
    table.add_column("Payload", max_width=60)
    if not rows:
        table.add_row("-", "No messages", "", "", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        payload_raw = d.get("payload", "")
        if isinstance(payload_raw, str):
            try:
                payload_raw = json.loads(payload_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        payload_summary = truncate(
            json.dumps(payload_raw) if isinstance(payload_raw, (dict, list)) else str(payload_raw),
            60,
        )
        table.add_row(
            short_id(d.get("id")),
            d.get("from_agent_name") or d.get("from_agent_id") or "unknown",
            d.get("topic", ""),
            d.get("timestamp", ""),
            status_color(d.get("status")),
            payload_summary,
        )
    return table


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def format_patches_list(rows: Sequence[Any]) -> Table:
    """Table of patch proposals."""
    table = Table(title="Patch Proposals")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Created")
    table.add_column("Title", style="bold")
    table.add_column("Status")
    table.add_column("Review")
    if not rows:
        table.add_row("-", "", "No patch proposals", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        review_raw = d.get("review_json")
        review_label = "-"
        if review_raw:
            try:
                review_data = json.loads(review_raw) if isinstance(review_raw, str) else review_raw
                if isinstance(review_data, dict):
                    if review_data.get("approved"):
                        review_label = "[green]passed[/green]"
                    else:
                        issues = review_data.get("issues", [])
                        issue_count = len(issues) if isinstance(issues, list) else 0
                        review_label = f"[red]failed ({issue_count} issues)[/red]"
            except (json.JSONDecodeError, TypeError):
                review_label = "?"
        table.add_row(
            short_id(d.get("id")),
            d.get("created_at", ""),
            d.get("title", ""),
            status_color(d.get("status")),
            review_label,
        )
    return table


def format_patch_detail(row: Any) -> Group:
    """Detailed view of a single patch proposal."""
    d = _row_dict(row)
    header_lines = [
        f"[bold]Title:[/bold]      {d.get('title', '')}",
        f"[bold]ID:[/bold]         {d.get('id', '')}",
        f"[bold]Created:[/bold]    {d.get('created_at', '')}",
        f"[bold]Status:[/bold]     {status_color(d.get('status'))}",
        f"[bold]Rationale:[/bold]  {d.get('rationale') or 'n/a'}",
    ]
    files_raw = d.get("files_json", "[]")
    try:
        files = json.loads(files_raw) if isinstance(files_raw, str) else files_raw
    except (json.JSONDecodeError, TypeError):
        files = []
    if files:
        header_lines.append("[bold]Files:[/bold]")
        for f in files:
            header_lines.append(f"  \u2022 {f}")
    header_panel = Panel("\n".join(header_lines), title="Patch Proposal", border_style="cyan")

    diff_text = d.get("diff_text", "")
    if diff_text:
        diff_panel = Panel(
            Syntax(diff_text, "diff", theme="monokai", line_numbers=False),
            title="Diff",
            border_style="yellow",
        )
    else:
        diff_panel = Panel("(no diff)", title="Diff")

    panels: list[Any] = [header_panel, diff_panel]

    review_raw = d.get("review_json", "")
    if review_raw:
        try:
            review_data = json.loads(review_raw) if isinstance(review_raw, str) else review_raw
            if isinstance(review_data, dict):
                approved = review_data.get("approved", False)
                issues = review_data.get("issues", [])
                review_lines = [
                    f"[bold]Approved:[/bold] {'[green]Yes[/green]' if approved else '[red]No[/red]'}",
                ]
                if issues:
                    review_lines.append("[bold]Issues:[/bold]")
                    for issue in issues:
                        review_lines.append(f"  [red]\u2717[/red] {issue}")
                review_border = "green" if approved else "red"
                panels.append(
                    Panel("\n".join(review_lines), title="LLM Safety Review", border_style=review_border)
                )
        except (json.JSONDecodeError, TypeError):
            pass

    test_raw = d.get("test_results", "")
    if test_raw:
        try:
            test_data = json.loads(test_raw) if isinstance(test_raw, str) else test_raw
            test_text = json.dumps(test_data, indent=2)
        except (json.JSONDecodeError, TypeError):
            test_text = str(test_raw)
        panels.append(Panel(test_text, title="Test Results", border_style="green"))

    return Group(*panels)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def format_run_status(run: Any, tasks: Sequence[Any], metrics: dict | None) -> Group:
    """Active run status with task table and metrics."""
    d = _row_dict(run)
    info_lines = [
        f"[bold]Run:[/bold]     {d.get('id', '')}",
        f"[bold]Status:[/bold]  {status_color(d.get('status'))}",
        f"[bold]Prompt:[/bold]  {truncate(d.get('user_prompt', ''), 100)}",
        f"[bold]Created:[/bold] {d.get('created_at', '')}",
    ]
    if metrics:
        info_lines.append(
            f"[bold]Metrics:[/bold] LLM={metrics.get('llm_calls', 'n/a')}  "
            f"Tools={metrics.get('tool_calls', 'n/a')}  "
            f"Steps={metrics.get('steps', 'n/a')}"
        )
    info_panel = Panel("\n".join(info_lines), title="Active Run", border_style="green")

    task_table = Table(title="Tasks")
    task_table.add_column("#", justify="right", style="dim")
    task_table.add_column("Title", style="bold")
    task_table.add_column("Status")
    for t in tasks:
        td = _row_dict(t)
        task_table.add_row(
            str(td.get("ordinal", "")),
            td.get("title", ""),
            status_color(td.get("status")),
        )
    if not tasks:
        task_table.add_row("-", "No tasks", "-")

    return Group(info_panel, task_table)


def format_run_show(run: Any, tasks: Sequence[Any]) -> Group:
    """Detailed view of a specific run."""
    d = _row_dict(run)
    info_lines = [
        f"[bold]Run:[/bold]     {d.get('id', '')}",
        f"[bold]Status:[/bold]  {status_color(d.get('status'))}",
        f"[bold]Prompt:[/bold]  {truncate(d.get('user_prompt', ''), 120)}",
        f"[bold]Created:[/bold] {d.get('created_at', '')}",
        f"[bold]Summary:[/bold] {d.get('run_summary') or 'n/a'}",
        f"[bold]Error:[/bold]   {d.get('last_error') or 'none'}",
    ]
    info_panel = Panel("\n".join(info_lines), title="Run Detail", border_style="blue")

    task_table = Table(title="Tasks")
    task_table.add_column("#", justify="right", style="dim")
    task_table.add_column("Title", style="bold")
    task_table.add_column("Status")
    task_table.add_column("Outputs", max_width=60)
    for t in tasks:
        td = _row_dict(t)
        outputs = td.get("outputs_json") or ""
        task_table.add_row(
            str(td.get("ordinal", "")),
            td.get("title", ""),
            status_color(td.get("status")),
            truncate(outputs, 60),
        )
    if not tasks:
        task_table.add_row("-", "No tasks", "-", "")

    return Group(info_panel, task_table)


def format_run_list(runs: Sequence[Any]) -> Table:
    """Table of recent runs."""
    table = Table(title="Recent Runs")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Created")
    table.add_column("Status")
    table.add_column("Error", max_width=50)
    if not runs:
        table.add_row("-", "", "No runs", "")
        return table
    for row in runs:
        d = _row_dict(row)
        table.add_row(
            short_id(d.get("id")),
            d.get("created_at", ""),
            status_color(d.get("status")),
            truncate(d.get("last_error") or "", 50),
        )
    return table


def format_timeline(events: Sequence[Any]) -> Table:
    """Task event timeline."""
    table = Table(title="Event Timeline")
    table.add_column("Timestamp")
    table.add_column("Task ID", style="dim", max_width=10)
    table.add_column("Event", style="bold")
    table.add_column("Payload", max_width=60)
    if not events:
        table.add_row("", "-", "No events", "")
        return table
    for row in events:
        d = _row_dict(row)
        table.add_row(
            d.get("timestamp", ""),
            short_id(d.get("task_id")),
            d.get("event_type", ""),
            truncate(d.get("payload") or "", 60),
        )
    return table


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def format_logs(entries: Sequence[dict]) -> Table:
    """Structured run logs."""
    table = Table(title="Run Logs")
    table.add_column("Run ID", style="dim", max_width=10)
    table.add_column("LLM Calls", justify="right")
    table.add_column("Tool Calls", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Timestamp")
    if not entries:
        table.add_row("-", "", "", "", "No logs")
        return table
    for entry in entries:
        table.add_row(
            short_id(entry.get("run_id")),
            str(entry.get("llm_calls", "")),
            str(entry.get("tool_calls", "")),
            str(entry.get("steps", "")),
            entry.get("timestamp", ""),
        )
    return table


# ---------------------------------------------------------------------------
# Agent identity & listing
# ---------------------------------------------------------------------------

def format_agent_name(config: Any, home_path: str) -> Panel:
    """Agent identity panel."""
    lines = [
        f"[bold]Name:[/bold]  {config.agent_name}",
        f"[bold]ID:[/bold]    {config.agent_id}",
        f"[bold]Home:[/bold]  {home_path}",
        f"[bold]Role:[/bold]  {getattr(config, 'role_description', '') or 'n/a'}",
    ]
    return Panel("\n".join(lines), title="Agent Identity", border_style="cyan")


def format_agent_list(agents: Sequence[dict]) -> Table:
    """Table of known local agents."""
    table = Table(title="Local Agents")
    table.add_column("Name", style="bold")
    table.add_column("Home")
    table.add_column("ID", style="dim", max_width=10)
    if not agents:
        table.add_row("No agents found", "", "")
        return table
    for agent in agents:
        table.add_row(
            agent.get("agent_name", ""),
            agent.get("home", ""),
            short_id(agent.get("agent_id")),
        )
    return table


# ---------------------------------------------------------------------------
# Memory: commitments, facts, skills, preferences
# ---------------------------------------------------------------------------

def format_commitments(rows: Sequence[Any]) -> Table:
    """Table of commitments."""
    table = Table(title="Commitments")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Description", max_width=60)
    table.add_column("Status")
    table.add_column("Priority", justify="right")
    table.add_column("Due Date")
    table.add_column("Created")
    if not rows:
        table.add_row("-", "No commitments", "", "", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        table.add_row(
            short_id(d.get("id")),
            truncate(d.get("description", ""), 60),
            status_color(d.get("status")),
            str(d.get("priority", "")),
            d.get("due_date") or "-",
            d.get("created_at", ""),
        )
    return table


def format_facts(rows: Sequence[Any]) -> Table:
    """Table of facts."""
    table = Table(title="Facts")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Statement", max_width=60)
    table.add_column("Confidence", justify="right")
    table.add_column("Provenance")
    table.add_column("Created")
    if not rows:
        table.add_row("-", "No facts", "", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        conf = d.get("confidence")
        conf_str = f"{float(conf):.2f}" if conf is not None else ""
        table.add_row(
            short_id(d.get("id")),
            truncate(d.get("statement", ""), 60),
            conf_str,
            d.get("provenance_type", ""),
            d.get("created_at", ""),
        )
    return table


def format_skills(rows: Sequence[Any]) -> Table:
    """Table of skills."""
    table = Table(title="Skills")
    table.add_column("Name", style="bold")
    table.add_column("Trigger", max_width=40)
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failure", justify="right", style="red")
    table.add_column("Last Used")
    if not rows:
        table.add_row("No skills", "", "", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        table.add_row(
            d.get("name", ""),
            truncate(d.get("trigger", ""), 40),
            str(d.get("success_count", 0)),
            str(d.get("failure_count", 0)),
            d.get("last_used") or "-",
        )
    return table


def format_preferences(rows: Sequence[Any]) -> Table:
    """Table of preferences."""
    table = Table(title="Preferences")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_column("Confidence", justify="right")
    table.add_column("Provenance")
    table.add_column("Updated")
    if not rows:
        table.add_row("No preferences", "", "", "", "")
        return table
    for row in rows:
        d = _row_dict(row)
        conf = d.get("confidence")
        conf_str = f"{float(conf):.2f}" if conf is not None else ""
        table.add_row(
            d.get("key", ""),
            d.get("value", ""),
            conf_str,
            d.get("provenance_type", ""),
            d.get("updated_at", ""),
        )
    return table


# ---------------------------------------------------------------------------
# Doctor report
# ---------------------------------------------------------------------------

def format_doctor(report: dict) -> Group:
    """Structured doctor/health report."""
    violations = report.get("violations", [])
    if violations:
        violation_lines = "\n".join(f"  [red]\u2717[/red] {v}" for v in violations)
        violations_panel = Panel(
            violation_lines,
            title=f"Violations ({len(violations)})",
            border_style="red",
        )
    else:
        violations_panel = Panel(
            "  [green]\u2713[/green] No violations found.",
            title="Violations",
            border_style="green",
        )

    staged = report.get("staged_items", 0)
    staged_by_status = report.get("staged_items_by_status", {})
    oldest = report.get("oldest_pending_staged_item")
    staged_lines = [f"[bold]Total staged items:[/bold] {staged}"]
    if isinstance(staged_by_status, dict):
        for st, count in staged_by_status.items():
            staged_lines.append(f"  {st}: {count}")
    if oldest:
        staged_lines.append(f"[bold]Oldest pending:[/bold] {oldest.get('id', 'n/a')} ({oldest.get('created_at', '')})")
    staged_panel = Panel("\n".join(staged_lines), title="Staged Items", border_style="yellow")

    vec = report.get("vector_index", {})
    vec_lines = [
        f"[bold]Mapping entries:[/bold] {vec.get('mapping_entries', 0)}",
        f"[bold]Missing facts:[/bold]   {vec.get('missing_facts', 0)}",
        f"[bold]Missing episodes:[/bold]{vec.get('missing_episodes', 0)}",
    ]
    vec_panel = Panel("\n".join(vec_lines), title="Vector Index", border_style="blue")

    return Group(violations_panel, staged_panel, vec_panel)


# ---------------------------------------------------------------------------
# Memory search results
# ---------------------------------------------------------------------------

def format_memory_search(facts: Sequence[Any], episodes: Sequence[Any]) -> Group:
    """Merged search results for facts and episodes."""
    fact_table = Table(title="Matching Facts")
    fact_table.add_column("ID", style="dim", max_width=10)
    fact_table.add_column("Statement", max_width=70)
    fact_table.add_column("Confidence", justify="right")
    fact_table.add_column("Source")
    if facts:
        for row in facts:
            d = _row_dict(row)
            conf = d.get("confidence")
            conf_str = f"{float(conf):.2f}" if conf is not None else ""
            fact_table.add_row(
                short_id(d.get("id")),
                truncate(d.get("statement", ""), 70),
                conf_str,
                d.get("source_ref", ""),
            )
    else:
        fact_table.add_row("-", "No matching facts", "", "")

    ep_table = Table(title="Matching Episodes")
    ep_table.add_column("ID", style="dim", max_width=10)
    ep_table.add_column("User Input", max_width=40)
    ep_table.add_column("Agent Output", max_width=40)
    ep_table.add_column("Importance", justify="right")
    if episodes:
        for row in episodes:
            d = _row_dict(row)
            imp = d.get("importance")
            imp_str = f"{float(imp):.2f}" if imp is not None else ""
            ep_table.add_row(
                short_id(d.get("id")),
                truncate(d.get("user_input", ""), 40),
                truncate(d.get("agent_output", ""), 40),
                imp_str,
            )
    else:
        ep_table.add_row("-", "No matching episodes", "", "")

    return Group(fact_table, ep_table)


# ---------------------------------------------------------------------------
# Config display
# ---------------------------------------------------------------------------

def format_config(config: Any, home_path: str) -> Group:
    """Display the full agent configuration as grouped panels."""
    # Agent identity
    agent_lines = [
        f"[bold]Name:[/bold]        {config.agent_name}",
        f"[bold]ID:[/bold]          {config.agent_id}",
        f"[bold]Created:[/bold]     {config.created_at}",
        f"[bold]Role:[/bold]        {config.role_description or 'n/a'}",
        f"[bold]Capabilities:[/bold] {', '.join(config.capabilities) if config.capabilities else 'none'}",
        f"[bold]Home:[/bold]        {home_path}",
    ]
    agent_panel = Panel("\n".join(agent_lines), title="Agent", border_style="cyan")

    panels = [agent_panel]

    def _llm_panel(settings: Any, title: str) -> Panel:
        if not settings:
            return Panel("(not configured)", title=title)
        lines = [
            f"[bold]Provider:[/bold]  {settings.provider}",
            f"[bold]Model:[/bold]     {settings.model}",
            f"[bold]Base URL:[/bold]  {settings.base_url}",
            f"[bold]API Key:[/bold]   {'***' if settings.api_key else '(none)'}",
        ]
        return Panel("\n".join(lines), title=title, border_style="green")

    panels.append(_llm_panel(config.planner_llm, "Planner LLM"))
    panels.append(_llm_panel(config.responder_llm, "Responder LLM"))

    # Embeddings
    emb = config.embeddings
    if emb:
        emb_lines = [
            f"[bold]Provider:[/bold]  {emb.provider}",
            f"[bold]Model:[/bold]     {emb.model}",
            f"[bold]Base URL:[/bold]  {emb.base_url}",
            f"[bold]Dim:[/bold]       {emb.dim}",
        ]
        panels.append(Panel("\n".join(emb_lines), title="Embeddings", border_style="green"))

    # Tools
    tools = config.tools
    if tools:
        tools_lines = [
            f"[bold]Approval mode:[/bold]        {tools.approval_mode}",
            f"[bold]FS root:[/bold]              {tools.fs_root or '(none)'}",
            f"[bold]Python enabled:[/bold]       {tools.python_enabled}",
            f"[bold]Max tool calls/turn:[/bold]  {tools.max_tool_calls_per_turn}",
            f"[bold]Max tool seconds:[/bold]     {tools.max_tool_seconds}",
        ]
        panels.append(Panel("\n".join(tools_lines), title="Tools", border_style="yellow"))

    # Task runner
    tr = config.task_runner
    if tr:
        tr_lines = [
            f"[bold]Max tasks/turn:[/bold]                  {tr.max_tasks_per_turn}",
            f"[bold]Max LLM calls/task:[/bold]              {tr.max_llm_calls_per_task}",
            f"[bold]Max tool calls/task:[/bold]             {tr.max_tool_calls_per_task}",
            f"[bold]Max total LLM calls/run/turn:[/bold]    {tr.max_total_llm_calls_per_run_per_turn}",
            f"[bold]Max total steps/turn:[/bold]            {tr.max_total_steps_per_turn}",
        ]
        panels.append(Panel("\n".join(tr_lines), title="Task Runner", border_style="yellow"))

    # Autonomy
    auto = config.autonomy
    if auto:
        auto_lines = [
            f"[bold]Enabled:[/bold]                  {auto.enabled}",
            f"[bold]Auto continue:[/bold]            {auto.auto_continue}",
            f"[bold]Execute commitments:[/bold]      {auto.execute_commitments}",
            f"[bold]Idle trigger (s):[/bold]         {auto.idle_trigger_seconds}",
            f"[bold]Idle min interval (s):[/bold]    {auto.idle_min_interval_seconds}",
            f"[bold]Autonomous idle delay (s):[/bold]{auto.autonomous_idle_delay_seconds}",
            f"[bold]Max autonomous LLM/turn:[/bold]  {auto.max_autonomous_llm_calls_per_turn}",
        ]
        panels.append(Panel("\n".join(auto_lines), title="Autonomy", border_style="magenta"))

    return Group(*panels)


# ---------------------------------------------------------------------------
# Help / workflow guide
# ---------------------------------------------------------------------------

def format_help_guide() -> Panel:
    """Rich panel with common workflows and tips."""
    text = (
        "[bold underline]Getting Started[/bold underline]\n"
        "  tali agent create          Create a new agent\n"
        "  tali agent chat             Start an interactive chat session\n"
        "  tali setup                  Walk through configuration\n"
        "\n"
        "[bold underline]Managing Runs[/bold underline]\n"
        "  tali agent run status       Show the active run\n"
        "  tali agent run list         List recent runs\n"
        "  tali agent run cancel       Cancel the active run\n"
        "  tali agent run resume       Resume a blocked run\n"
        "\n"
        "[bold underline]Working with Patches[/bold underline]\n"
        "  tali agent patches list     List patch proposals\n"
        "  tali agent patches show ID  Show a specific patch\n"
        "  tali agent patches test ID  Run tests for a patch\n"
        "  tali agent patches apply ID Apply a tested patch\n"
        "\n"
        "[bold underline]Memory and Knowledge[/bold underline]\n"
        "  tali agent memory search Q  Search facts and episodes\n"
        "  tali agent facts            List all facts\n"
        "  tali agent commitments      List commitments\n"
        "  tali agent commitment add   Add a new commitment\n"
        "  tali agent skills           List learned skills\n"
        "\n"
        "[bold underline]Preferences and Config[/bold underline]\n"
        "  tali agent preferences list          List preferences\n"
        "  tali agent preferences set KEY VAL   Set a preference\n"
        "  tali agent config show               Show configuration\n"
        "  tali agent config set FIELD VALUE    Update a config field\n"
        "\n"
        "[bold underline]Monitoring[/bold underline]\n"
        "  tali agent dashboard        Live dashboard with tasks and memory\n"
        "  tali agent logs             Show recent run logs\n"
        "  tali agent doctor           Validate invariants\n"
        "  tali agent inbox            Show unread A2A messages\n"
        "\n"
        "[bold underline]Shell Completion[/bold underline]\n"
        "  tali --install-completion   Enable tab-completion for your shell\n"
    )
    return Panel(text, title="Tali CLI \u2014 Quick Reference", border_style="cyan")
