#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Venv nao encontrado. Execute ./setup.sh primeiro."
    exit 1
fi

# Quando houver servidor (M3): exec $PYTHON -m theo serve "$@"
echo "Theo OpenVoice v$($PYTHON -c 'import theo; print(theo.__version__)')"
echo "Servidor ainda nao implementado (M3)."
