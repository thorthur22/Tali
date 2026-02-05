from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tali.config import Paths, ToolSettings


Validator = Callable[[dict[str, Any]], tuple[bool, str | None, list[str], str | None]]
Handler = Callable[[dict[str, Any]], tuple[str, str]]
SignatureFn = Callable[[dict[str, Any]], str | None]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_schema: dict[str, str]
    risk_level: str
    validate_args: Validator
    handle: Handler
    signature: SignatureFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def describe_tools(self) -> str:
        lines: list[str] = []
        for tool in self.list_tools():
            args_desc = ", ".join(f"{key}: {value}" for key, value in tool.args_schema.items()) or "none"
            lines.append(f"- {tool.name}: {tool.description} (args: {args_desc})")
        return "\n".join(lines)


def _resolve_root(root: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved in resolved.parents:
        return resolved
    raise ValueError("path is outside the allowed root")


def _validate_path_arg(args: dict[str, Any], key: str) -> tuple[bool, str | None]:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        return False, f"{key} must be a non-empty string"
    return True, None


def _validate_fs_read(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    return ok, err, [], None


def _validate_fs_write(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    if not ok:
        return ok, err, [], None
    content = args.get("content")
    if not isinstance(content, str):
        return False, "content must be a string", [], None
    overwrite = args.get("overwrite")
    if overwrite is not None and not isinstance(overwrite, bool):
        return False, "overwrite must be boolean", [], None
    return True, None, [], None


def _validate_fs_list(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    if "path" in args:
        ok, err = _validate_path_arg(args, "path")
        return ok, err, [], None
    return True, None, [], None


def _validate_fs_stat(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    return ok, err, [], None


def _validate_shell_run(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "command must be a non-empty string", [], None
    forbidden = ["|", ">", "<", "&&", ";", "||", "&"]
    for token in forbidden:
        if token in command:
            return False, f"command contains forbidden token: {token}", [], "destructive"
    tokens = shlex.split(command, posix=False)
    if not tokens:
        return False, "command is empty", [], None
    verb = tokens[0].lower()
    if verb == "git":
        if len(tokens) < 2:
            return False, "git subcommand required", [], None
        sub = tokens[1].lower()
        allowed = {"status", "diff", "log"}
        if sub not in allowed:
            return False, f"git subcommand not allowed: {sub}", [], "destructive"
        extra = tokens[2:]
        allowed_flags = {"--oneline", "--stat", "--name-only", "-n", "-s", "-sb", "--short"}
        for token in extra:
            if not token.startswith("-") or token not in allowed_flags:
                return False, f"git arg not allowed: {token}", [], "destructive"
        return True, None, [], "needs_approval"
    if verb in {"ls", "dir"}:
        if len(tokens) > 2:
            return False, "ls/dir allows at most one path argument", [], None
        return True, None, [], "needs_approval"
    if verb in {"cat", "type"}:
        if len(tokens) != 2:
            return False, "cat/type requires a single path argument", [], None
        return True, None, [], "needs_approval"
    return False, f"command not allowed: {verb}", [], "destructive"


def _validate_web_fetch(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        return False, "url must be a non-empty string", [], None
    if not (url.startswith("http://") or url.startswith("https://")):
        return False, "url must start with http:// or https://", [], None
    return True, None, [], None


def _validate_python_eval(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return False, "code must be a non-empty string", [], None
    return True, None, [], "needs_approval"


def _signature_shell(args: dict[str, Any]) -> str | None:
    command = args.get("command")
    return command if isinstance(command, str) else None


def _signature_default(_: dict[str, Any]) -> str | None:
    return None


def build_default_registry(paths: Paths, settings: ToolSettings) -> ToolRegistry:
    registry = ToolRegistry()
    fs_root = Path(settings.fs_root) if settings.fs_root else paths.data_dir

    def fs_read(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        if path.stat().st_size > settings.fs_max_bytes:
            raise ValueError("file exceeds max read size")
        data = path.read_text(encoding="utf-8", errors="replace")
        return f"Read {path}", data

    def fs_write(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        overwrite = bool(args.get("overwrite", False))
        if path.exists() and not overwrite:
            raise ValueError("file exists; set overwrite=true to overwrite")
        if len(args["content"].encode("utf-8")) > settings.fs_max_bytes:
            raise ValueError("content exceeds max write size")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        stat = path.stat()
        return f"Wrote {path}", f"size={stat.st_size} mtime={stat.st_mtime}"

    def fs_list(args: dict[str, Any]) -> tuple[str, str]:
        target = args.get("path", str(fs_root))
        path = _resolve_root(fs_root, target)
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = sorted(os.listdir(path))
        return f"Listed {path}", "\n".join(entries)

    def fs_stat(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        stat = path.stat()
        summary = f"Stat {path}"
        raw = f"size={stat.st_size} mtime={stat.st_mtime}"
        return summary, raw

    def shell_run(args: dict[str, Any]) -> tuple[str, str]:
        command = args["command"]
        tokens = shlex.split(command, posix=False)
        verb = tokens[0].lower()
        if verb == "git":
            import subprocess

            result = subprocess.run(tokens, capture_output=True, text=True, check=False)
            output = result.stdout + result.stderr
            return f"git {tokens[1]} exited {result.returncode}", output.strip()
        if verb in {"ls", "dir"}:
            target = tokens[1] if len(tokens) > 1 else str(paths.data_dir)
            path = _resolve_root(fs_root, target)
            entries = sorted(os.listdir(path))
            return f"Listed {path}", "\n".join(entries)
        if verb in {"cat", "type"}:
            path = _resolve_root(fs_root, tokens[1])
            data = path.read_text(encoding="utf-8", errors="replace")
            return f"Read {path}", data
        raise ValueError("command not allowed")

    def web_fetch(args: dict[str, Any]) -> tuple[str, str]:
        import httpx

        url = args["url"]
        timeout = settings.web_timeout_s
        max_bytes = settings.web_max_bytes
        max_redirects = settings.web_max_redirects
        allowed_types = {"text/html", "application/json", "text/plain"}

        current = url
        for _ in range(max_redirects + 1):
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.get(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    break
                current = location
                continue
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and content_type not in allowed_types:
                raise ValueError(f"content-type not allowed: {content_type}")
            content = response.content[: max_bytes + 1]
            truncated = len(content) > max_bytes
            text = content[:max_bytes].decode("utf-8", errors="replace")
            if truncated:
                text += "\n[truncated]"
            summary = f"Fetched {current} status={response.status_code}"
            return summary, text
        raise ValueError("too many redirects")

    def python_eval(args: dict[str, Any]) -> tuple[str, str]:
        code = args["code"]
        return "Python eval requested", code

    registry.register(
        ToolDefinition(
            name="fs.read",
            description=f"Read a file within root {fs_root}",
            args_schema={"path": "string"},
            risk_level="safe",
            validate_args=_validate_fs_read,
            handle=fs_read,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.write",
            description=f"Write a file within root {fs_root}",
            args_schema={"path": "string", "content": "string", "overwrite": "boolean (optional)"},
            risk_level="destructive",
            validate_args=_validate_fs_write,
            handle=fs_write,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.list",
            description=f"List directory entries within root {fs_root}",
            args_schema={"path": "string (optional)"},
            risk_level="safe",
            validate_args=_validate_fs_list,
            handle=fs_list,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.stat",
            description=f"Stat a path within root {fs_root}",
            args_schema={"path": "string"},
            risk_level="safe",
            validate_args=_validate_fs_stat,
            handle=fs_stat,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="shell.run",
            description="Run a constrained shell command (read-only subcommands)",
            args_schema={"command": "string"},
            risk_level="needs_approval",
            validate_args=_validate_shell_run,
            handle=shell_run,
            signature=_signature_shell,
        )
    )
    registry.register(
        ToolDefinition(
            name="web.fetch",
            description="Fetch a URL over HTTP(S) with limits",
            args_schema={"url": "string"},
            risk_level="safe",
            validate_args=_validate_web_fetch,
            handle=web_fetch,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="python.eval",
            description="Evaluate Python code in a sandbox (disabled by default)",
            args_schema={"code": "string"},
            risk_level="needs_approval",
            validate_args=_validate_python_eval,
            handle=python_eval,
            signature=_signature_default,
        )
    )
    return registry
