from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import httpx


@dataclass(frozen=True)
class ModelCatalogEntry:
    key: str
    label: str
    repo_id: str
    pattern: str


_CATALOG: list[ModelCatalogEntry] = [
    ModelCatalogEntry(
        key="llama3.1-8b-instruct-q4_k_m",
        label="Llama 3.1 8B Instruct (Q4_K_M)",
        repo_id="unsloth/Llama-3.1-8B-Instruct-GGUF",
        pattern=r"q4_k_m\\.gguf$",
    ),
    ModelCatalogEntry(
        key="qwen2.5-7b-instruct-q4_k_m",
        label="Qwen 2.5 7B Instruct (Q4_K_M)",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        pattern=r"q4_k_m\\.gguf$",
    ),
    ModelCatalogEntry(
        key="qwen2.5-1.5b-instruct-q4_k_m",
        label="Qwen 2.5 1.5B Instruct (Q4_K_M)",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        pattern=r"q4_k_m\\.gguf$",
    ),
]


def list_models() -> list[ModelCatalogEntry]:
    return list(_CATALOG)


def download_model(repo_id: str, pattern: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = _resolve_model_filename(repo_id, pattern)
    target_path = target_dir / filename
    if target_path.exists():
        return target_path
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    timeout = httpx.Timeout(60.0, connect=10.0, read=None)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with target_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    return target_path


def _resolve_model_filename(repo_id: str, pattern: str) -> str:
    url = f"https://huggingface.co/api/models/{repo_id}"
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    siblings = payload.get("siblings", [])
    filenames = [item.get("rfilename", "") for item in siblings if item.get("rfilename")]
    regex = re.compile(pattern, re.IGNORECASE)
    matches = [name for name in filenames if name.endswith(".gguf") and regex.search(name)]
    if not matches:
        raise ValueError(f"No GGUF file matching '{pattern}' in repo {repo_id}.")
    matches.sort(key=lambda name: (len(name), name))
    return matches[0]
