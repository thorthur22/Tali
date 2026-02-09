from __future__ import annotations

"""
Agentic loop utilities.

This module defines a simple controller for running iterative loops during
autonomous tasks. The AgenticLoop can be used by planners or schedulers to
decide when to continue, retry, or stop a sequence of actions based on a
success flag and budget.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopDecision:
    """Represents a decision about whether to continue an agentic loop."""
    action: str  # 'continue', 'retry', or 'stop'
    reason: str


class AgenticLoop:
    """
    Manage the state of an agentic loop. Each call to `should_continue`
    advances the loop count and returns a LoopDecision indicating what to do
    next. By default the loop will stop after `max_loops` iterations.
    """

    def __init__(self, max_loops: int = 5) -> None:
        self.max_loops = max_loops
        self.iteration = 0

    def should_continue(self, success: bool) -> LoopDecision:
        """
        Given whether the last iteration succeeded, return a decision about
        the next step. On success, the action is 'stop'. On failure the
        decision is 'retry' until the loop budget is exhausted, after which
        the action is 'stop' with a budget exhaustion reason.
        """
        self.iteration += 1
        if success:
            return LoopDecision(action="stop", reason="task completed successfully")
        if self.iteration >= self.max_loops:
            return LoopDecision(action="stop", reason="loop budget exhausted")
        return LoopDecision(action="retry", reason="task incomplete; retrying")
