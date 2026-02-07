#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11+ is required but '$PYTHON_BIN' was not found."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required to run Tali.")
PY

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"
python -m pip install -e .

if ! command -v ollama >/dev/null 2>&1; then
  echo "Note: 'ollama' not found in PATH. Local models require Ollama installed and running."
fi

echo "Install complete. Run 'tali' to bootstrap your first agent."
