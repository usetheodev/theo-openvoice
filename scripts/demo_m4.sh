#!/usr/bin/env bash
# =============================================================================
# Demo M4 — Audio Preprocessing + Post-Processing Pipeline
#
# Valida o fluxo completo da Fase 1:
#   audio (qualquer SR) -> preprocessing -> worker -> post-processing -> resposta
#
# Prerequisitos:
#   - .venv com dependencias instaladas (uv sync)
#   - Modelo faster-whisper-tiny instalado em ~/.theo/models/
#   - gTTS instalado (uv pip install gTTS)
#   - ffmpeg instalado (para converter MP3 -> WAV)
#
# Uso:
#   ./scripts/demo_m4.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin"
PYTHON="$VENV/python"
THEO="$VENV/theo"
HOST="127.0.0.1"
PORT="8099"
BASE_URL="http://$HOST:$PORT"
TMPDIR_DEMO=""
SERVER_PID=""

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
step()  { echo -e "\n${CYAN}=== Step $1: $2 ===${NC}"; }

# --- Cleanup ---
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        info "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$TMPDIR_DEMO" ] && [ -d "$TMPDIR_DEMO" ]; then
        rm -rf "$TMPDIR_DEMO"
    fi
}
trap cleanup EXIT

# --- Preconditions ---
info "Checking preconditions..."

if [ ! -f "$PYTHON" ]; then
    fail "Python not found at $PYTHON. Run: uv venv --python 3.12 && uv sync"
    exit 1
fi

if [ ! -f "$THEO" ]; then
    fail "theo CLI not found at $THEO. Run: uv sync"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    fail "ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

# Check if gTTS is installed
if ! $PYTHON -c "import gtts" 2>/dev/null; then
    fail "gTTS not installed. Install with: uv pip install gTTS"
    exit 1
fi

# Check if model is installed
MODELS_DIR="$HOME/.theo/models"
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    fail "No models found in $MODELS_DIR."
    fail "Install a model first. See README for instructions."
    exit 1
fi

info "Models directory: $MODELS_DIR"

# --- Step 0: Generate speech audio files ---
step "0" "Generate speech audio files (gTTS + ffmpeg)"

TMPDIR_DEMO="$(mktemp -d)"
info "Temp dir: $TMPDIR_DEMO"

$PYTHON -c "
from gtts import gTTS
import subprocess, os, wave

tmpdir = '$TMPDIR_DEMO'

# 1. Generate English speech with numbers (tests ITN)
tts_en = gTTS('Hello, the total amount is two thousand and twenty five dollars.', lang='en')
mp3_en = os.path.join(tmpdir, 'speech_en.mp3')
tts_en.save(mp3_en)
print(f'  Generated: speech_en.mp3 ({os.path.getsize(mp3_en)} bytes)')

# 2. Generate Portuguese speech with numbers (tests ITN for pt-BR)
tts_pt = gTTS('Ola, o valor total e dois mil e vinte e cinco reais.', lang='pt')
mp3_pt = os.path.join(tmpdir, 'speech_pt.mp3')
tts_pt.save(mp3_pt)
print(f'  Generated: speech_pt.mp3 ({os.path.getsize(mp3_pt)} bytes)')

# 3. Convert to WAV at different sample rates using ffmpeg
conversions = [
    (mp3_en, 'speech_en_44khz.wav', 44100),   # downsample test
    (mp3_pt, 'speech_pt_8khz.wav', 8000),      # telephony upsample test
    (mp3_en, 'speech_en_16khz.wav', 16000),     # native sample rate
]

for src, name, sr in conversions:
    dst = os.path.join(tmpdir, name)
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', src, '-ar', str(sr), '-ac', '1', '-acodec', 'pcm_s16le', dst],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f'  ERROR converting {name}: {result.stderr[:200]}')
        raise SystemExit(1)
    with wave.open(dst, 'rb') as wf:
        dur = wf.getnframes() / wf.getframerate()
        print(f'  Converted: {name} (sr={wf.getframerate()}, dur={dur:.1f}s)')
"

pass "Speech audio files generated"

# --- Step 1: Start server ---
step "1" "Start theo serve"

$THEO serve --host "$HOST" --port "$PORT" --models-dir "$MODELS_DIR" --log-level WARNING &
SERVER_PID=$!

# Wait for server to be ready
info "Waiting for server to start (PID $SERVER_PID)..."
for i in $(seq 1 60); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        pass "Server started on $BASE_URL (${i}s)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "Server process died during startup"
        exit 1
    fi
    sleep 1
done

if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    fail "Server did not start within 60 seconds"
    exit 1
fi

# Verify health endpoint
HEALTH=$(curl -s "$BASE_URL/health")
info "Health: $HEALTH"

# Wait for worker to be ready (model loading takes time)
info "Waiting for worker to load model..."
WORKER_READY=false
for i in $(seq 1 60); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -F "file=@$TMPDIR_DEMO/speech_en_16khz.wav" \
        -F "model=faster-whisper-tiny" \
        "$BASE_URL/v1/audio/transcriptions")
    if [ "$HTTP_CODE" = "200" ]; then
        pass "Worker ready (model loaded in ${i}s)"
        WORKER_READY=true
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "Server process died while loading model"
        exit 1
    fi
    sleep 1
done

if [ "$WORKER_READY" = "false" ]; then
    fail "Worker did not become ready within 60 seconds"
    exit 1
fi

# --- Helper function for curl requests ---
do_request() {
    local step_num="$1"
    local desc="$2"
    shift 2
    local url="$BASE_URL/v1/audio/transcriptions"

    step "$step_num" "$desc"

    local http_code
    local tmpfile="$TMPDIR_DEMO/response_$step_num.txt"

    http_code=$(curl -s -o "$tmpfile" -w "%{http_code}" "$@" "$url")
    local response
    response=$(cat "$tmpfile")

    if [ "$http_code" = "200" ]; then
        pass "HTTP $http_code"
        info "Response: $response"
    else
        fail "HTTP $http_code (expected 200)"
        info "Response: $response"
        return 1
    fi
}

# --- Step 2: Transcribe 44.1kHz English (preprocessing: downsample 44.1k->16k) ---
do_request "2" "Transcribe 44.1kHz English (preprocessing: downsample 44.1k->16k)" \
    -F "file=@$TMPDIR_DEMO/speech_en_44khz.wav" \
    -F "model=faster-whisper-tiny"

# --- Step 3: Transcribe 8kHz Portuguese (telephony: upsample 8k->16k) ---
do_request "3" "Transcribe 8kHz Portuguese (preprocessing: upsample 8k->16k)" \
    -F "file=@$TMPDIR_DEMO/speech_pt_8khz.wav" \
    -F "model=faster-whisper-tiny" \
    -F "language=pt"

# --- Step 4: Transcribe 16kHz (native, no resample needed) ---
do_request "4" "Transcribe 16kHz English (native sample rate)" \
    -F "file=@$TMPDIR_DEMO/speech_en_16khz.wav" \
    -F "model=faster-whisper-tiny"

# --- Step 5: Transcribe with itn=false ---
do_request "5" "Transcribe with itn=false (post-processing skipped)" \
    -F "file=@$TMPDIR_DEMO/speech_en_44khz.wav" \
    -F "model=faster-whisper-tiny" \
    -F "itn=false"

# --- Step 6: CLI transcribe (default ITN) ---
step "6" "CLI: theo transcribe (default ITN)"

CLI_OUT=$($THEO transcribe "$TMPDIR_DEMO/speech_en_44khz.wav" \
    --model faster-whisper-tiny \
    --server "$BASE_URL" 2>&1) || true
pass "CLI executed"
info "Output: $CLI_OUT"

# --- Step 7: CLI transcribe --no-itn ---
step "7" "CLI: theo transcribe --no-itn"

CLI_OUT=$($THEO transcribe "$TMPDIR_DEMO/speech_en_44khz.wav" \
    --model faster-whisper-tiny \
    --server "$BASE_URL" \
    --no-itn 2>&1) || true
pass "CLI executed"
info "Output: $CLI_OUT"

# --- Step 8: verbose_json format ---
do_request "8" "verbose_json format (segments + metadata)" \
    -F "file=@$TMPDIR_DEMO/speech_en_44khz.wav" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=verbose_json"

# --- Step 9: SRT subtitle format ---
do_request "9" "SRT subtitle format" \
    -F "file=@$TMPDIR_DEMO/speech_en_44khz.wav" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=srt"

# --- Step 10: Translation endpoint (Portuguese -> English) ---
step "10" "Translation endpoint (Portuguese -> English)"

TRANS_CODE=$(curl -s -o "$TMPDIR_DEMO/response_10.txt" -w "%{http_code}" \
    -F "file=@$TMPDIR_DEMO/speech_pt_8khz.wav" \
    -F "model=faster-whisper-tiny" \
    "$BASE_URL/v1/audio/translations")
TRANS_RESP=$(cat "$TMPDIR_DEMO/response_10.txt")

if [ "$TRANS_CODE" = "200" ]; then
    pass "HTTP $TRANS_CODE"
    info "Response: $TRANS_RESP"
else
    fail "HTTP $TRANS_CODE (expected 200)"
    info "Response: $TRANS_RESP"
fi

# --- Step 11: MP3 input (tests audio decode path) ---
do_request "11" "MP3 input (tests audio decode path)" \
    -F "file=@$TMPDIR_DEMO/speech_en.mp3" \
    -F "model=faster-whisper-tiny"

# --- Step 12: Run unit tests ---
step "12" "Run unit tests"

info "Running pytest..."
$PYTHON -m pytest "$PROJECT_DIR/tests/unit/" -v --tb=short 2>&1 | tail -20

pass "Unit tests completed"

# --- Summary ---
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  M4 Demo Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
info "Audio Preprocessing Pipeline: resample, DC remove, gain normalize"
info "  - 44.1kHz -> 16kHz (downsample)"
info "  - 8kHz -> 16kHz (telephony upsample)"
info "  - 16kHz (no resample needed)"
info "Post-Processing Pipeline: ITN (nemo_text_processing)"
info "Config toggles: per-stage enable/disable via PreprocessingConfig/PostProcessingConfig"
info "API control: itn=true/false parameter"
info "CLI control: --no-itn flag"
info "Response formats: json, verbose_json, text, srt, vtt"
info "Translation: /v1/audio/translations endpoint"
echo ""
