from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tali.config import Paths, ToolSettings
from tali.tools.protocol import ToolCall
from tali.tools.registry import ToolRegistry


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reason: str | None
    red_flags: list[str]
    risk_level: str | None
    signature: str | None


class ToolPolicy:
    def __init__(self, settings: ToolSettings, registry: ToolRegistry, paths: Paths) -> None:
        self.settings = settings
        self.registry = registry
        self.paths = paths

    def evaluate(self, call: ToolCall, tool_counts: dict[str, int]) -> PolicyDecision:
        definition = self.registry.get(call.name)
        if definition is None:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="unknown tool",
                red_flags=["unknown_tool"],
                risk_level=None,
                signature=None,
            )
        count = tool_counts.get(call.name, 0)
        if count >= self.settings.max_calls_per_tool:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="tool rate limit exceeded",
                red_flags=["rate_limit"],
                risk_level=definition.risk_level,
                signature=None,
            )
        ok, error, red_flags, risk_override = definition.validate_args(call.args)
        if not ok:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=error or "invalid args",
                red_flags=red_flags or ["invalid_args"],
                risk_level=definition.risk_level,
                signature=None,
            )
        risk_level = risk_override or definition.risk_level
        requires_approval = False
        if self._requires_approval(call):
            risk_level = "destructive"
            requires_approval = True
            red_flags = red_flags + ["create_or_delete"]
        if call.name.startswith("fs."):
            fs_error = self._validate_fs_path(call)
            if fs_error:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=fs_error,
                    red_flags=red_flags + ["fs_path"],
                    risk_level=risk_level,
                    signature=None,
                )
        signature = definition.signature(call.args)
        return PolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            reason=None,
            red_flags=red_flags,
            risk_level=risk_level,
            signature=signature,
        )

    def _requires_approval(self, call: ToolCall) -> bool:
        if call.name == "fs.delete":
            return True
        if call.name == "fs.move":
            return True
        if call.name == "fs.write":
            return self._is_fs_create(call, "path")
        if call.name == "fs.write_patch":
            return self._is_fs_create(call, "path")
        if call.name == "fs.copy":
            return self._is_fs_create(call, "dest")
        return False

    def _is_fs_create(self, call: ToolCall, key: str) -> bool:
        root = Path(self.settings.fs_root) if self.settings.fs_root else self.paths.data_dir
        target = call.args.get(key)
        if not isinstance(target, str) or not target:
            return False
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return True
        return not resolved.exists()

    def _validate_fs_path(self, call: ToolCall) -> str | None:
        root = Path(self.settings.fs_root) if self.settings.fs_root else self.paths.data_dir
        target = call.args.get("path")
        if not isinstance(target, str) or not target:
            if call.name == "fs.list":
                return None
            return "path is required"
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            return "path is outside the allowed root"
        if call.name == "fs.write":
            allowlist = self.settings.fs_allow_extensions or []
            if allowlist:
                if resolved.suffix.lower() not in {ext.lower() for ext in allowlist}:
                    return "file extension not allowed"
        return None
