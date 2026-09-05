#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v ruff >/dev/null 2>&1; then
    RUFF_CMD="ruff"
elif [ -x "${PROJECT_ROOT}/.venv/bin/ruff" ]; then
    RUFF_CMD="${PROJECT_ROOT}/.venv/bin/ruff"
elif command -v uv >/dev/null 2>&1; then
    RUFF_CMD="uv run ruff"
else
    echo "Error: ruff is not installed. Run 'uv pip install ruff' or install dev dependencies." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    ${RUFF_CMD} check .
    ${RUFF_CMD} format --check .
else
    ${RUFF_CMD} check "$@"
fi
