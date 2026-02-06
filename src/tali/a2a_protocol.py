from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskRequest:
    title: str
    description: str
    inputs: dict[str, Any]
    constraints: dict[str, Any]
    requested_outputs: list[str]


@dataclass(frozen=True)
class TaskResponse:
    correlation_id: str
    status: str
    result: dict[str, Any]
    notes: str


@dataclass(frozen=True)
class StatusMessage:
    run_state: str
    capabilities: list[str]
    load: dict[str, Any]


def parse_message(payload: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = payload.strip()
    if not raw:
        return None, "empty payload"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(data, dict):
        return None, "payload must be object"
    if "type" not in data:
        return None, "missing type"
    return data, None


def parse_task_request(payload: dict[str, Any]) -> tuple[TaskRequest | None, str | None]:
    task = payload.get("task")
    constraints = payload.get("constraints", {})
    requested_outputs = payload.get("requested_outputs", [])
    if not isinstance(task, dict):
        return None, "task must be object"
    title = task.get("title")
    description = task.get("description")
    inputs = task.get("inputs", {})
    if not isinstance(title, str) or not title.strip():
        return None, "title required"
    if not isinstance(description, str):
        return None, "description required"
    if not isinstance(inputs, dict):
        return None, "inputs must be object"
    if not isinstance(constraints, dict):
        return None, "constraints must be object"
    if not isinstance(requested_outputs, list):
        return None, "requested_outputs must be list"
    return (
        TaskRequest(
            title=title.strip(),
            description=description.strip(),
            inputs=inputs,
            constraints=constraints,
            requested_outputs=[str(item) for item in requested_outputs],
        ),
        None,
    )


def parse_task_response(payload: dict[str, Any]) -> tuple[TaskResponse | None, str | None]:
    correlation_id = payload.get("correlation_id")
    status = payload.get("status")
    result = payload.get("result", {})
    notes = payload.get("notes", "")
    if not isinstance(correlation_id, str) or not correlation_id.strip():
        return None, "correlation_id required"
    if status not in {"accepted", "rejected", "completed", "failed"}:
        return None, "status invalid"
    if not isinstance(result, dict):
        return None, "result must be object"
    if not isinstance(notes, str):
        return None, "notes must be string"
    return (
        TaskResponse(
            correlation_id=correlation_id.strip(),
            status=status,
            result=result,
            notes=notes,
        ),
        None,
    )


def parse_status(payload: dict[str, Any]) -> tuple[StatusMessage | None, str | None]:
    run_state = payload.get("run_state", "")
    capabilities = payload.get("capabilities", [])
    load = payload.get("load", {})
    if not isinstance(run_state, str):
        return None, "run_state must be string"
    if not isinstance(capabilities, list):
        return None, "capabilities must be list"
    if not isinstance(load, dict):
        return None, "load must be object"
    return (
        StatusMessage(
            run_state=run_state,
            capabilities=[str(item) for item in capabilities],
            load=load,
        ),
        None,
    )
