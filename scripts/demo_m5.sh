#!/usr/bin/env bash
# =============================================================================
# Demo M5 -- WebSocket Streaming STT (end-to-end validation)
#
# Valida TODOS os componentes do M5:
#   - WS /v1/realtime endpoint
#   - Protocolo de eventos JSON (session.created, vad.*, transcript.*, etc.)
#   - StreamingSession (preprocess -> VAD -> gRPC -> post-process)
#   - BackpressureController (rate limit + frames dropped)
#   - Heartbeat + inactivity timeout
#   - Metricas Prometheus
#   - Error handling (recuperavel / nao-recuperavel)
#
# NAO requer modelo real nem GPU -- usa mocks controlados.
#
# Prerequisitos:
#   - .venv com dependencias instaladas (uv sync)
#
# Uso:
#   ./scripts/demo_m5.sh
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

if ! $PYTHON -c "import starlette.testclient" 2>/dev/null; then
    fail "starlette not importable. Run: uv sync"
    exit 1
fi
pass "starlette TestClient available"

if ! $PYTHON -c "import numpy" 2>/dev/null; then
    fail "numpy not importable. Run: uv sync"
    exit 1
fi
pass "numpy available"

# --- Run demo ---
echo ""
info "Running M5 demo..."
echo ""

$PYTHON "$SCRIPT_DIR/demo_m5.py"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    pass "M5 Demo completed successfully!"
else
    echo ""
    fail "M5 Demo failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
