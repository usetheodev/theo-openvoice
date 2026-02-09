#!/usr/bin/env bash
# Demo M8 -- Scheduler Avancado
# Priorizacao | Cancelamento | Batching | Latencia | Shutdown
#
# Uso:
#   ./scripts/demo_m8.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$VENV" ]]; then
    echo "ERROR: Python venv not found at $VENV"
    echo "Run: uv venv --python 3.12 && uv pip install -e '.[dev]'"
    exit 1
fi

exec "$VENV" "$SCRIPT_DIR/demo_m8.py" "$@"
