from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tali.db import Database
from tali.questions import queue_question


HOOK_TIME_LIMIT_MS = 200


@dataclass(frozen=True)
class Hook:
    name: str
    triggers: set[str]
    handler: Callable[["HookContext"], "HookActions | None"]


@dataclass
class HookActions:
    messages: list[str] = field(default_factory=list)
    staged_items: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HookContext:
    event: str
    payload: dict[str, Any]
    db: Database

    def stage_item(self, kind: str, payload: dict[str, Any], provenance_type: str, source_ref: str) -> None:
        from datetime import datetime
        import json
        import uuid

        self.db.insert_staged_item(
            item_id=str(uuid.uuid4()),
            kind=kind,
            payload=json.dumps(payload),
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            source_ref=source_ref,
            provenance_type=provenance_type,
            next_check_at=datetime.utcnow().isoformat(),
        )

    def enqueue_question(self, question: str, reason: str | None, priority: int = 3) -> None:
        queue_question(self.db, question=question, reason=reason, priority=priority)


class HookManager:
    def __init__(self, db: Database, hooks_dir: Path) -> None:
        self.db = db
        self.hooks_dir = hooks_dir
        self.hooks: list[Hook] = []

    def load_hooks(self) -> None:
        if not self.hooks_dir.exists():
            return
        for path in sorted(self.hooks_dir.glob("*.py")):
            if path.name.startswith("_") or path.name in {"core.py", "__init__.py"}:
                continue
            module = self._load_module(path)
            if not module:
                continue
            if hasattr(module, "HOOKS"):
                hooks = getattr(module, "HOOKS")
                self._register_hooks(hooks)
            if hasattr(module, "get_hooks"):
                hooks = module.get_hooks()
                self._register_hooks(hooks)

    def run(self, event: str, payload: dict[str, Any]) -> list[str]:
        messages: list[str] = []
        for hook in self.hooks:
            if event not in hook.triggers:
                continue
            context = HookContext(event=event, payload=payload, db=self.db)
            result = self._run_with_timeout(hook, context)
            if result is None:
                continue
            messages.extend(result.messages)
            for item in result.staged_items:
                context.stage_item(
                    kind=item.get("kind", "skill"),
                    payload=item.get("payload", {}),
                    provenance_type=item.get("provenance_type", "SYSTEM_OBSERVED"),
                    source_ref=item.get("source_ref", "hook"),
                )
            for question in result.questions:
                context.enqueue_question(
                    question=question.get("question", ""),
                    reason=question.get("reason"),
                    priority=int(question.get("priority", 3)),
                )
        return messages

    def _run_with_timeout(self, hook: Hook, context: HookContext) -> HookActions | None:
        result_container: dict[str, HookActions | None] = {"result": None}

        def worker() -> None:
            try:
                result_container["result"] = hook.handler(context)
            except Exception:
                result_container["result"] = None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=HOOK_TIME_LIMIT_MS / 1000.0)
        if thread.is_alive():
            return None
        return result_container["result"]

    def _register_hooks(self, hooks: Any) -> None:
        if not isinstance(hooks, list):
            return
        for hook in hooks:
            if isinstance(hook, Hook):
                self.hooks.append(hook)

    def _load_module(self, path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
