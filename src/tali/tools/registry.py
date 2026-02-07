from __future__ import annotations

import os
import re
import shlex
import shutil
from urllib.parse import quote
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


def _validate_fs_search(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return False, "query must be a non-empty string", [], None
    if "path" in args:
        ok, err = _validate_path_arg(args, "path")
        if not ok:
            return ok, err, [], None
    max_results = args.get("max_results")
    if max_results is not None and not isinstance(max_results, int):
        return False, "max_results must be an integer", [], None
    return True, None, [], None


def _validate_fs_glob(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return False, "pattern must be a non-empty string", [], None
    if "path" in args:
        ok, err = _validate_path_arg(args, "path")
        if not ok:
            return ok, err, [], None
    return True, None, [], None


def _validate_fs_tree(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    if "path" in args:
        ok, err = _validate_path_arg(args, "path")
        if not ok:
            return ok, err, [], None
    max_depth = args.get("max_depth")
    if max_depth is not None and not isinstance(max_depth, int):
        return False, "max_depth must be an integer", [], None
    return True, None, [], None


def _validate_fs_read_lines(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    if not ok:
        return ok, err, [], None
    start = args.get("start")
    limit = args.get("limit")
    if start is not None and not isinstance(start, int):
        return False, "start must be an integer", [], None
    if limit is not None and not isinstance(limit, int):
        return False, "limit must be an integer", [], None
    return True, None, [], None


def _validate_fs_copy(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "src")
    if not ok:
        return ok, err, [], None
    ok, err = _validate_path_arg(args, "dest")
    if not ok:
        return ok, err, [], None
    overwrite = args.get("overwrite")
    if overwrite is not None and not isinstance(overwrite, bool):
        return False, "overwrite must be boolean", [], None
    return True, None, [], None


def _validate_fs_move(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    return _validate_fs_copy(args)


def _validate_fs_delete(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    return ok, err, [], None


def _validate_fs_write_patch(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    ok, err = _validate_path_arg(args, "path")
    if not ok:
        return ok, err, [], None
    patch = args.get("patch")
    if not isinstance(patch, str):
        return False, "patch must be a string", [], None
    return True, None, [], None


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
        allowed = {"status", "diff", "log", "show", "ls-files"}
        if sub not in allowed:
            return False, f"git subcommand not allowed: {sub}", [], "destructive"
        extra = tokens[2:]
        allowed_flags = {
            "--oneline",
            "--stat",
            "--name-only",
            "-n",
            "-s",
            "-sb",
            "--short",
            "--no-patch",
            "--pretty",
            "--raw",
            "--decorate",
            "--cached",
            "--others",
            "--exclude-standard",
            "--stage",
        }
        for token in extra:
            if token.startswith("-"):
                if token not in allowed_flags:
                    return False, f"git arg not allowed: {token}", [], "destructive"
                continue
            if not _is_safe_git_arg(token):
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


def _validate_web_search(args: dict[str, Any]) -> tuple[bool, str | None, list[str], str | None]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return False, "query must be a non-empty string", [], None
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


def _is_safe_git_arg(token: str) -> bool:
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-~^@")
    return all(ch in allowed_chars for ch in token)


def _apply_unified_patch(original_text: str, patch_text: str) -> str:
    original_lines = original_text.splitlines()
    patch_lines = patch_text.splitlines()
    output: list[str] = []
    index = 0
    hunk_header = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    line_idx = 0
    while line_idx < len(patch_lines):
        line = patch_lines[line_idx]
        if line.startswith(("---", "+++", "diff --git")):
            line_idx += 1
            continue
        if not line.startswith("@@"):
            line_idx += 1
            continue
        match = hunk_header.match(line)
        if not match:
            raise ValueError("invalid patch hunk header")
        start_old = int(match.group(1))
        if start_old < 1:
            raise ValueError("invalid patch hunk start")
        while index < start_old - 1 and index < len(original_lines):
            output.append(original_lines[index])
            index += 1
        line_idx += 1
        while line_idx < len(patch_lines):
            hunk_line = patch_lines[line_idx]
            if hunk_line.startswith("@@"):
                break
            if not hunk_line:
                marker = " "
                content = ""
            else:
                marker = hunk_line[0]
                content = hunk_line[1:]
            if marker == " ":
                if index >= len(original_lines) or original_lines[index] != content:
                    raise ValueError("patch context mismatch")
                output.append(original_lines[index])
                index += 1
            elif marker == "-":
                if index >= len(original_lines) or original_lines[index] != content:
                    raise ValueError("patch delete mismatch")
                index += 1
            elif marker == "+":
                output.append(content)
            elif marker == "\\":
                pass
            else:
                raise ValueError("invalid patch line")
            line_idx += 1
    output.extend(original_lines[index:])
    return "\n".join(output) + ("\n" if original_text.endswith("\n") else "")


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

    def fs_search(args: dict[str, Any]) -> tuple[str, str]:
        query = args["query"]
        base = args.get("path", str(fs_root))
        pattern = args.get("glob")
        max_results = int(args.get("max_results", 20) or 20)
        base_path = _resolve_root(fs_root, base)
        if not base_path.exists():
            raise ValueError("path does not exist")
        matches: list[str] = []
        iterator = base_path.rglob(pattern) if pattern else base_path.rglob("*")
        for path in iterator:
            if len(matches) >= max_results:
                break
            if path.is_dir():
                continue
            if path.stat().st_size > settings.fs_max_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = path.relative_to(fs_root)
                    matches.append(f"{rel}:{idx}: {line.strip()}")
                    if len(matches) >= max_results:
                        break
        summary = f"Search '{query}' in {base_path}"
        return summary, "\n".join(matches) if matches else "No matches."

    def fs_glob(args: dict[str, Any]) -> tuple[str, str]:
        pattern = args["pattern"]
        base = args.get("path", str(fs_root))
        base_path = _resolve_root(fs_root, base)
        if not base_path.exists():
            raise ValueError("path does not exist")
        matches = [str(path.relative_to(fs_root)) for path in base_path.rglob(pattern)]
        summary = f"Glob {pattern} in {base_path}"
        return summary, "\n".join(sorted(matches)) if matches else "No matches."

    def fs_tree(args: dict[str, Any]) -> tuple[str, str]:
        base = args.get("path", str(fs_root))
        max_depth = int(args.get("max_depth", 3) or 3)
        base_path = _resolve_root(fs_root, base)
        if not base_path.exists():
            raise ValueError("path does not exist")
        lines: list[str] = []
        base_depth = len(base_path.parts)
        for path in sorted(base_path.rglob("*")):
            depth = len(path.parts) - base_depth
            if depth > max_depth:
                continue
            indent = "  " * depth
            label = path.name + ("/" if path.is_dir() else "")
            lines.append(f"{indent}{label}")
        summary = f"Tree {base_path} depth={max_depth}"
        return summary, "\n".join(lines) if lines else "(empty)"

    def fs_read_lines(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        start = int(args.get("start", 1) or 1)
        limit = int(args.get("limit", 100) or 100)
        if start < 1 or limit < 1:
            raise ValueError("start and limit must be positive integers")
        if path.stat().st_size > settings.fs_max_bytes:
            raise ValueError("file exceeds max read size")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        slice_start = start - 1
        slice_end = min(slice_start + limit, len(lines))
        payload = [f"{idx + 1}|{lines[idx]}" for idx in range(slice_start, slice_end)]
        summary = f"Read lines {start}-{slice_end} from {path}"
        return summary, "\n".join(payload)

    def fs_write_patch(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        patch = args["patch"]
        original = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        updated = _apply_unified_patch(original, patch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        return f"Patched {path}", f"bytes={len(updated.encode('utf-8'))}"

    def fs_copy(args: dict[str, Any]) -> tuple[str, str]:
        src = _resolve_root(fs_root, args["src"])
        dest = _resolve_root(fs_root, args["dest"])
        overwrite = bool(args.get("overwrite", False))
        if dest.exists() and not overwrite:
            raise ValueError("destination exists; set overwrite=true to overwrite")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"Copied {src} -> {dest}", ""

    def fs_move(args: dict[str, Any]) -> tuple[str, str]:
        src = _resolve_root(fs_root, args["src"])
        dest = _resolve_root(fs_root, args["dest"])
        overwrite = bool(args.get("overwrite", False))
        if dest.exists() and not overwrite:
            raise ValueError("destination exists; set overwrite=true to overwrite")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return f"Moved {src} -> {dest}", ""

    def fs_delete(args: dict[str, Any]) -> tuple[str, str]:
        path = _resolve_root(fs_root, args["path"])
        if path.is_dir():
            raise ValueError("refusing to delete directory")
        if not path.exists():
            raise ValueError("path does not exist")
        path.unlink()
        return f"Deleted {path}", ""

    def shell_run(args: dict[str, Any]) -> tuple[str, str]:
        command = args["command"]
        tokens = shlex.split(command, posix=False)
        verb = tokens[0].lower()
        if verb == "git":
            import subprocess

            result = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.max_tool_seconds,
            )
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

    def web_search(args: dict[str, Any]) -> tuple[str, str]:
        import httpx

        query = args["query"]
        timeout = settings.web_timeout_s
        url = f"https://duckduckgo.com/html/?q={quote(query)}"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
        if response.status_code != 200:
            raise ValueError(f"search failed status={response.status_code}")
        html = response.text
        pattern = re.compile(r'class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)
        results: list[str] = []
        for match in pattern.finditer(html):
            href = match.group(1)
            title = re.sub(r"<.*?>", "", match.group(2)).strip()
            if not title:
                continue
            results.append(f"- {title} ({href})")
            if len(results) >= 5:
                break
        summary = f"Search results for '{query}'"
        return summary, "\n".join(results) if results else "No results found."

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
            name="fs.search",
            description=f"Search file contents under root {fs_root}",
            args_schema={
                "query": "string",
                "path": "string (optional)",
                "glob": "string (optional)",
                "max_results": "int (optional)",
            },
            risk_level="needs_approval",
            validate_args=_validate_fs_search,
            handle=fs_search,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.glob",
            description=f"Find files matching a glob under root {fs_root}",
            args_schema={"pattern": "string", "path": "string (optional)"},
            risk_level="needs_approval",
            validate_args=_validate_fs_glob,
            handle=fs_glob,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.tree",
            description=f"List a directory tree under root {fs_root}",
            args_schema={"path": "string (optional)", "max_depth": "int (optional)"},
            risk_level="needs_approval",
            validate_args=_validate_fs_tree,
            handle=fs_tree,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.read_lines",
            description=f"Read a range of lines from a file under root {fs_root}",
            args_schema={"path": "string", "start": "int (optional)", "limit": "int (optional)"},
            risk_level="needs_approval",
            validate_args=_validate_fs_read_lines,
            handle=fs_read_lines,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.write_patch",
            description=f"Apply a unified patch to a file under root {fs_root}",
            args_schema={"path": "string", "patch": "string"},
            risk_level="destructive",
            validate_args=_validate_fs_write_patch,
            handle=fs_write_patch,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.copy",
            description=f"Copy a file under root {fs_root}",
            args_schema={"src": "string", "dest": "string", "overwrite": "boolean (optional)"},
            risk_level="destructive",
            validate_args=_validate_fs_copy,
            handle=fs_copy,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.move",
            description=f"Move a file under root {fs_root}",
            args_schema={"src": "string", "dest": "string", "overwrite": "boolean (optional)"},
            risk_level="destructive",
            validate_args=_validate_fs_move,
            handle=fs_move,
            signature=_signature_default,
        )
    )
    registry.register(
        ToolDefinition(
            name="fs.delete",
            description=f"Delete a file under root {fs_root}",
            args_schema={"path": "string"},
            risk_level="destructive",
            validate_args=_validate_fs_delete,
            handle=fs_delete,
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
            name="web.search",
            description="Search the web for sources",
            args_schema={"query": "string"},
            risk_level="safe",
            validate_args=_validate_web_search,
            handle=web_search,
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
