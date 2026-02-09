#!/usr/bin/env bash
# =============================================================================
# Demo M7 -- Segundo Backend STT (WeNet CTC) — validacao model-agnostic
#
# Valida TODOS os componentes do M7:
#   - WeNetBackend (STTBackend para CTC architecture)
#   - Factory _create_backend() com lazy imports (faster-whisper + wenet)
#   - Manifesto theo.yaml para WeNet (architecture: ctc, hot_words: true)
#   - Pipeline Adaptativo: CTC sem LocalAgreement, sem cross-segment context
#   - Hot Words: nativo (WeNet) vs initial_prompt (Whisper)
#   - Contrato identico: mesmos eventos para ambas as engines
#   - Worker CLI: --engine wenet aceito
#
# NAO requer modelo real nem GPU -- usa mocks controlados.
#
# Prerequisitos:
#   - .venv com dependencias instaladas (uv sync)
#
# Uso:
#   ./scripts/demo_m7.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin"
PYTHON="$VENV/python"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
pass()  { echo -e "${GREEN}[PASS]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }

# --- Preconditions ---
info "Checking preconditions..."

if [ ! -f "$PYTHON" ]; then
    fail "Python not found at $PYTHON. Run: uv venv --python 3.12 && uv sync"
    exit 1
fi

# Verify Python version
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Python: $PY_VERSION"

# Verify key imports
if ! $PYTHON -c "import theo" 2>/dev/null; then
    fail "theo package not importable. Run: uv sync"
    exit 1
fi
pass "theo package available"

if ! $PYTHON -c "import numpy" 2>/dev/null; then
    fail "numpy not importable. Run: uv sync"
    exit 1
fi
pass "numpy available"

# --- Run demo ---
echo ""
info "Running M7 demo..."
echo ""

$PYTHON "$SCRIPT_DIR/demo_m7.py"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    pass "M7 Demo completed successfully!"
else
    echo ""
    fail "M7 Demo failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
