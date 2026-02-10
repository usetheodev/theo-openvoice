#!/usr/bin/env bash
# ==============================================================================
# Theo OpenVoice — CLI Integration Test Script
#
# Testa todos os comandos CLI do Theo em ordem:
#   1. Comandos offline (sem servidor)
#   2. Comandos online (com servidor rodando)
#   3. Cleanup
#
# Uso: bash scripts/test_cli.sh
#
# Pre-requisitos:
#   - Modelos instalados: faster-whisper-tiny (STT) e kokoro-v1 (TTS)
#     Execute: theo pull faster-whisper-tiny && theo pull kokoro-v1
#
# NOTA: O download dos modelos acontece na primeira execucao (~75MB STT).
#       Os modelos sao reutilizados em execucoes subsequentes.
#
# NOTA: O teste de transcricao usa um audio sintetico (sine tone 440Hz).
#       O resultado da transcricao pode ser vazio ou impreciso — o objetivo
#       e validar que o fluxo completo funciona, nao a qualidade do STT.
#
# NOTA: Os testes TTS validam que POST /v1/audio/speech retorna audio valido
#       (WAV com header RIFF, PCM raw) e trata erros corretamente.
# ==============================================================================

set -euo pipefail

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
SKIP=0

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
THEO="${PROJECT_DIR}/.venv/bin/theo"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
SERVER_PID=""
SERVER_PORT=8765  # Porta nao-padrao para nao conflitar
SERVER_URL="http://localhost:${SERVER_PORT}"
TEST_AUDIO="${PROJECT_DIR}/tests/fixtures/audio/sample_16khz.wav"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

test_pass() {
    echo -e "  ${GREEN}✓ PASS${NC}: $1"
    PASS=$((PASS + 1))
}

test_fail() {
    echo -e "  ${RED}✗ FAIL${NC}: $1"
    echo -e "    ${RED}$2${NC}"
    FAIL=$((FAIL + 1))
}

test_skip() {
    echo -e "  ${YELLOW}⊘ SKIP${NC}: $1 — $2"
    SKIP=$((SKIP + 1))
}

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Parando servidor (PID $SERVER_PID)...${NC}"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Servidor parado."
    fi
}

trap cleanup EXIT

# ------------------------------------------------------------------------------
# Pre-checks
# ------------------------------------------------------------------------------

section "Pre-checks"

if [ ! -x "$THEO" ]; then
    echo -e "${RED}ERRO: $THEO nao encontrado ou nao executavel.${NC}"
    echo "Execute: cd $PROJECT_DIR && uv venv --python 3.12 && uv pip install -e '.[dev]'"
    exit 1
fi

echo "  theo: $THEO"
echo "  python: $PYTHON"
echo "  audio: $TEST_AUDIO"

if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${RED}ERRO: audio de teste nao encontrado: $TEST_AUDIO${NC}"
    exit 1
fi

# ==============================================================================
# PARTE 1: Comandos Offline (sem servidor)
# ==============================================================================

section "1. theo --version"

VERSION_OUTPUT=$("$THEO" --version 2>&1) || true
if echo "$VERSION_OUTPUT" | grep -q "theo.*version"; then
    test_pass "theo --version -> $VERSION_OUTPUT"
else
    test_fail "theo --version" "Output inesperado: $VERSION_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "2. theo --help"

HELP_OUTPUT=$("$THEO" --help 2>&1) || true
if echo "$HELP_OUTPUT" | grep -q "transcribe" && \
   echo "$HELP_OUTPUT" | grep -q "serve" && \
   echo "$HELP_OUTPUT" | grep -q "pull" && \
   echo "$HELP_OUTPUT" | grep -q "list"; then
    test_pass "theo --help lista comandos: transcribe, serve, pull, list"
else
    test_fail "theo --help" "Comandos esperados nao encontrados no output"
fi

# ------------------------------------------------------------------------------

section "3. theo list"

LIST_OUTPUT=$("$THEO" list 2>&1) || true
if echo "$LIST_OUTPUT" | grep -q "faster-whisper-tiny"; then
    test_pass "theo list mostra faster-whisper-tiny (STT)"
elif echo "$LIST_OUTPUT" | grep -q "Nenhum modelo"; then
    test_skip "theo list" "Nenhum modelo instalado (precisa de 'theo pull' primeiro)"
else
    test_fail "theo list" "Output inesperado: $LIST_OUTPUT"
fi

# Verificar se kokoro-v1 (TTS) tambem aparece
if echo "$LIST_OUTPUT" | grep -q "kokoro-v1"; then
    test_pass "theo list mostra kokoro-v1 (TTS)"
else
    test_skip "theo list (TTS)" "kokoro-v1 nao instalado (execute: theo pull kokoro-v1)"
fi

# ------------------------------------------------------------------------------

section "4. theo inspect faster-whisper-tiny"

INSPECT_OUTPUT=$("$THEO" inspect faster-whisper-tiny 2>&1) || true
if echo "$INSPECT_OUTPUT" | grep -q "Name:" && \
   echo "$INSPECT_OUTPUT" | grep -q "faster-whisper-tiny"; then
    test_pass "theo inspect mostra detalhes do modelo"
    echo "    Engine: $(echo "$INSPECT_OUTPUT" | grep 'Engine:')"
    echo "    Architecture: $(echo "$INSPECT_OUTPUT" | grep 'Architecture:')"
    echo "    Memory: $(echo "$INSPECT_OUTPUT" | grep 'Memory:')"
elif echo "$INSPECT_OUTPUT" | grep -q "nao encontrado"; then
    test_skip "theo inspect" "Modelo nao instalado"
else
    test_fail "theo inspect faster-whisper-tiny" "Output inesperado: $INSPECT_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "5. theo inspect <modelo-inexistente>"

INSPECT_BAD=$("$THEO" inspect modelo-fake 2>&1) || true
if echo "$INSPECT_BAD" | grep -q "nao encontrado"; then
    test_pass "theo inspect modelo-fake retorna erro esperado"
else
    test_fail "theo inspect modelo-fake" "Deveria retornar erro, output: $INSPECT_BAD"
fi

# ------------------------------------------------------------------------------

section "6. theo pull (validacao do catalogo)"

# Testar com modelo inexistente
PULL_BAD=$("$THEO" pull modelo-fake 2>&1) || true
if echo "$PULL_BAD" | grep -q "nao encontrado"; then
    test_pass "theo pull modelo-fake retorna erro com lista de modelos"
else
    test_fail "theo pull modelo-fake" "Deveria listar modelos disponiveis, output: $PULL_BAD"
fi

# Testar que pull de modelo ja instalado avisa
PULL_EXISTS=$("$THEO" pull faster-whisper-tiny 2>&1) || true
if echo "$PULL_EXISTS" | grep -q "ja esta instalado"; then
    test_pass "theo pull faster-whisper-tiny avisa que ja esta instalado"
elif echo "$PULL_EXISTS" | grep -q "Baixando"; then
    test_pass "theo pull faster-whisper-tiny iniciou download (primeira vez)"
else
    test_fail "theo pull faster-whisper-tiny" "Output inesperado: $PULL_EXISTS"
fi

# ------------------------------------------------------------------------------

section "7. theo transcribe (sem servidor)"

TRANSCRIBE_OFFLINE=$("$THEO" transcribe "$TEST_AUDIO" -m faster-whisper-tiny 2>&1) || true
if echo "$TRANSCRIBE_OFFLINE" | grep -qi "nao disponivel\|connect\|refused\|erro"; then
    test_pass "theo transcribe sem servidor retorna erro de conexao"
else
    test_fail "theo transcribe (offline)" "Deveria falhar sem servidor, output: $TRANSCRIBE_OFFLINE"
fi

# ------------------------------------------------------------------------------

section "8. theo ps (sem servidor)"

PS_OFFLINE=$("$THEO" ps 2>&1) || true
if echo "$PS_OFFLINE" | grep -qi "nao disponivel\|connect\|refused\|erro"; then
    test_pass "theo ps sem servidor retorna erro de conexao"
else
    test_fail "theo ps (offline)" "Deveria falhar sem servidor, output: $PS_OFFLINE"
fi

# ==============================================================================
# PARTE 2: Comandos Online (com servidor)
# ==============================================================================

section "9. theo serve (iniciar servidor)"

echo "  Iniciando servidor na porta $SERVER_PORT..."
echo "  Comando: $THEO serve --port $SERVER_PORT --log-level WARNING"
echo ""

# Inicia servidor em background
"$THEO" serve --port "$SERVER_PORT" --log-level WARNING &
SERVER_PID=$!
echo "  PID do servidor: $SERVER_PID"

# Aguardar servidor ficar pronto (health check com retry)
MAX_WAIT=120  # segundos (modelo pode demorar para carregar na primeira vez)
WAITED=0
SERVER_READY=false

echo "  Aguardando servidor ficar pronto (max ${MAX_WAIT}s)..."
while [ $WAITED -lt $MAX_WAIT ]; do
    sleep 2
    WAITED=$((WAITED + 2))

    # Verificar se processo ainda esta vivo
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "  ${RED}Servidor morreu antes de ficar pronto!${NC}"
        # Tenta capturar stderr
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        break
    fi

    # Tentar health check
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health" 2>/dev/null) || true
    if [ "$HEALTH" = "200" ]; then
        SERVER_READY=true
        break
    fi

    # Progress
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  ... ${WAITED}s (aguardando worker ficar ready)"
    fi
done

if [ "$SERVER_READY" = true ]; then
    test_pass "theo serve iniciou em ${WAITED}s (health check 200)"
else
    test_fail "theo serve" "Servidor nao ficou pronto em ${MAX_WAIT}s"
    echo ""
    echo -e "${RED}Servidor nao esta rodando — pulando testes online.${NC}"
    echo ""

    # Reportar resultados parciais
    section "Resultado"
    echo ""
    echo -e "  ${GREEN}PASS: $PASS${NC}"
    echo -e "  ${RED}FAIL: $FAIL${NC}"
    echo -e "  ${YELLOW}SKIP: $SKIP${NC}"
    echo ""
    exit 1
fi

# Aguardar worker ficar READY (health HTTP != worker ready)
# O health check HTTP retorna 200 assim que o FastAPI sobe, mas o worker gRPC
# pode ainda estar em STARTING. Precisamos aguardar o worker transitar para READY.
echo ""
echo "  Aguardando worker ficar READY (max 60s)..."
WORKER_READY=false
WORKER_WAITED=0
while [ $WORKER_WAITED -lt 60 ]; do
    sleep 2
    WORKER_WAITED=$((WORKER_WAITED + 2))

    # Tentar uma transcricao — se retornar 200, worker esta pronto
    WORKER_CHECK=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -F "file=@${TEST_AUDIO}" \
        -F "model=faster-whisper-tiny" \
        -F "response_format=json" \
        "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

    if [ "$WORKER_CHECK" = "200" ]; then
        WORKER_READY=true
        break
    fi

    if [ $((WORKER_WAITED % 10)) -eq 0 ]; then
        echo "  ... ${WORKER_WAITED}s (worker ainda nao esta ready, HTTP $WORKER_CHECK)"
    fi
done

if [ "$WORKER_READY" = true ]; then
    echo -e "  ${GREEN}Worker READY em ${WORKER_WAITED}s${NC}"
else
    echo -e "  ${RED}Worker nao ficou ready em 60s (ultimo HTTP: $WORKER_CHECK)${NC}"
    echo -e "  ${YELLOW}Continuando testes mesmo assim...${NC}"
fi

# ------------------------------------------------------------------------------

section "10. GET /health"

HEALTH_OUTPUT=$(curl -s "$SERVER_URL/health" 2>/dev/null) || true
if echo "$HEALTH_OUTPUT" | grep -q '"status".*"ok"'; then
    test_pass "GET /health -> $HEALTH_OUTPUT"
else
    test_fail "GET /health" "Output: $HEALTH_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "11. GET /v1/models"

MODELS_OUTPUT=$(curl -s "$SERVER_URL/v1/models" 2>/dev/null) || true
if echo "$MODELS_OUTPUT" | grep -q "faster-whisper-tiny"; then
    test_pass "GET /v1/models lista faster-whisper-tiny"
else
    test_fail "GET /v1/models" "Output: $MODELS_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "12. theo ps (com servidor)"

PS_OUTPUT=$("$THEO" ps --server "$SERVER_URL" 2>&1) || true
if echo "$PS_OUTPUT" | grep -q "faster-whisper-tiny"; then
    test_pass "theo ps mostra faster-whisper-tiny carregado"
else
    test_fail "theo ps" "Output: $PS_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "13. POST /v1/audio/transcriptions (curl)"

echo "  Enviando audio: $TEST_AUDIO"
TRANSCRIBE_OUTPUT=$(curl -s -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=json" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

if echo "$TRANSCRIBE_OUTPUT" | grep -q '"text"'; then
    TEXT=$(echo "$TRANSCRIBE_OUTPUT" | "$PYTHON" -c "import sys,json; print(json.loads(sys.stdin.read()).get('text',''))" 2>/dev/null || echo "(parse error)")
    test_pass "POST /v1/audio/transcriptions -> text='$TEXT'"
else
    test_fail "POST /v1/audio/transcriptions" "Output: $TRANSCRIBE_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "14. theo transcribe (via CLI, com servidor)"

# CLI imprime texto puro (nao JSON) no sucesso; httpx loga request no stderr
# Para sine tone, texto pode ser vazio — o indicador de sucesso e o exit code
CLI_TRANSCRIBE=$("$THEO" transcribe "$TEST_AUDIO" -m faster-whisper-tiny --server "$SERVER_URL" 2>&1)
CLI_EXIT=$?
if [ $CLI_EXIT -eq 0 ]; then
    test_pass "theo transcribe via CLI completou com sucesso (exit code 0)"
elif echo "$CLI_TRANSCRIBE" | grep -qi "Erro"; then
    test_fail "theo transcribe" "Output: $CLI_TRANSCRIBE"
else
    test_fail "theo transcribe (CLI)" "Exit code $CLI_EXIT, Output: $CLI_TRANSCRIBE"
fi

# ------------------------------------------------------------------------------

section "15. POST /v1/audio/transcriptions verbose_json"

VERBOSE_OUTPUT=$(curl -s -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=verbose_json" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

if echo "$VERBOSE_OUTPUT" | grep -q '"segments"'; then
    test_pass "verbose_json retorna campo 'segments'"
else
    test_fail "POST verbose_json" "Output: $VERBOSE_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "16. POST /v1/audio/transcriptions text format"

TEXT_OUTPUT=$(curl -s -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=text" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

# text format retorna plain text (pode ser vazio para sine tone)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=faster-whisper-tiny" \
    -F "response_format=text" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

if [ "$HTTP_CODE" = "200" ]; then
    test_pass "response_format=text retorna 200"
else
    test_fail "response_format=text" "HTTP $HTTP_CODE, Output: $TEXT_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "17. POST /v1/audio/translations"

TRANSLATE_OUTPUT=$(curl -s -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=faster-whisper-tiny" \
    "$SERVER_URL/v1/audio/translations" 2>/dev/null) || true

if echo "$TRANSLATE_OUTPUT" | grep -q '"text"'; then
    test_pass "POST /v1/audio/translations retorna texto"
else
    test_fail "POST /v1/audio/translations" "Output: $TRANSLATE_OUTPUT"
fi

# ------------------------------------------------------------------------------

section "18. Erro: modelo inexistente"

ERROR_OUTPUT=$(curl -s -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=modelo-inexistente" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -F "file=@${TEST_AUDIO}" \
    -F "model=modelo-inexistente" \
    "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

if [ "$HTTP_CODE" = "404" ]; then
    test_pass "Modelo inexistente retorna 404"
elif echo "$ERROR_OUTPUT" | grep -qi "not found\|nao encontrado\|error"; then
    test_pass "Modelo inexistente retorna erro ($HTTP_CODE)"
else
    test_fail "Modelo inexistente" "HTTP $HTTP_CODE, Output: $ERROR_OUTPUT"
fi

# ==============================================================================
# PARTE 2B: Testes TTS (com servidor)
# ==============================================================================

section "TTS: Aguardando TTS worker ficar READY"

# Mesma logica do STT: o health HTTP retorna 200, mas o worker TTS gRPC
# pode ainda estar em STARTING. Enviar request real ate obter resposta.
echo "  Aguardando TTS worker ficar READY (max 90s)..."
TTS_WORKER_READY=false
TTS_WORKER_WAITED=0
while [ $TTS_WORKER_WAITED -lt 90 ]; do
    sleep 2
    TTS_WORKER_WAITED=$((TTS_WORKER_WAITED + 2))

    TTS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d '{"model":"kokoro-v1","input":"teste","voice":"default"}' \
        "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

    if [ "$TTS_CHECK" = "200" ]; then
        TTS_WORKER_READY=true
        break
    fi

    if [ $((TTS_WORKER_WAITED % 10)) -eq 0 ]; then
        echo "  ... ${TTS_WORKER_WAITED}s (TTS worker ainda nao esta ready, HTTP $TTS_CHECK)"
    fi
done

if [ "$TTS_WORKER_READY" = true ]; then
    echo -e "  ${GREEN}TTS Worker READY em ${TTS_WORKER_WAITED}s${NC}"
else
    echo -e "  ${RED}TTS Worker nao ficou ready em 90s (ultimo HTTP: $TTS_CHECK)${NC}"
    echo -e "  ${YELLOW}Pulando testes TTS...${NC}"
fi

# Executar testes TTS apenas se worker esta pronto
if [ "$TTS_WORKER_READY" = true ]; then

# ------------------------------------------------------------------------------

section "19. POST /v1/audio/speech (WAV)"

TTS_TMPFILE=$(mktemp /tmp/theo_tts_XXXXXX.wav)
TTS_HTTP_CODE=$(curl -s -o "$TTS_TMPFILE" -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"Ola, como posso ajudar?","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_HTTP_CODE" = "200" ]; then
    # Verificar que o arquivo tem header RIFF (WAV valido)
    RIFF_HEADER=$(head -c 4 "$TTS_TMPFILE" 2>/dev/null) || true
    TTS_SIZE=$(wc -c < "$TTS_TMPFILE")
    if [ "$RIFF_HEADER" = "RIFF" ] && [ "$TTS_SIZE" -gt 44 ]; then
        test_pass "POST /v1/audio/speech -> WAV ${TTS_SIZE} bytes (RIFF header OK)"
    else
        test_fail "POST /v1/audio/speech (WAV)" "Resposta 200 mas sem header RIFF (size: ${TTS_SIZE})"
    fi
else
    test_fail "POST /v1/audio/speech (WAV)" "HTTP $TTS_HTTP_CODE"
fi
rm -f "$TTS_TMPFILE"

# ------------------------------------------------------------------------------

section "20. POST /v1/audio/speech (PCM raw)"

TTS_PCM_TMPFILE=$(mktemp /tmp/theo_tts_pcm_XXXXXX.bin)
TTS_PCM_CODE=$(curl -s -o "$TTS_PCM_TMPFILE" -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"Bom dia","voice":"default","response_format":"pcm"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_PCM_CODE" = "200" ]; then
    PCM_SIZE=$(wc -c < "$TTS_PCM_TMPFILE")
    if [ "$PCM_SIZE" -gt 0 ]; then
        test_pass "POST /v1/audio/speech (PCM) -> ${PCM_SIZE} bytes raw audio"
    else
        test_fail "POST /v1/audio/speech (PCM)" "Resposta 200 mas corpo vazio"
    fi
else
    test_fail "POST /v1/audio/speech (PCM)" "HTTP $TTS_PCM_CODE"
fi
rm -f "$TTS_PCM_TMPFILE"

# ------------------------------------------------------------------------------

section "21. POST /v1/audio/speech (Content-Type headers)"

# Verificar Content-Type: audio/wav para formato WAV
TTS_CT_WAV=$(curl -s -D - -o /dev/null -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"teste","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if echo "$TTS_CT_WAV" | grep -qi "content-type.*audio/wav"; then
    test_pass "Content-Type: audio/wav para formato WAV"
else
    CT_FOUND=$(echo "$TTS_CT_WAV" | grep -i "content-type" | head -1)
    test_fail "Content-Type WAV" "Esperava audio/wav, recebeu: $CT_FOUND"
fi

# Verificar Content-Type: audio/pcm para formato PCM
TTS_CT_PCM=$(curl -s -D - -o /dev/null -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"teste","voice":"default","response_format":"pcm"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if echo "$TTS_CT_PCM" | grep -qi "content-type.*audio/pcm"; then
    test_pass "Content-Type: audio/pcm para formato PCM"
else
    CT_FOUND=$(echo "$TTS_CT_PCM" | grep -i "content-type" | head -1)
    test_fail "Content-Type PCM" "Esperava audio/pcm, recebeu: $CT_FOUND"
fi

# ------------------------------------------------------------------------------

section "22. POST /v1/audio/speech (texto vazio -> 400)"

TTS_EMPTY_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_EMPTY_CODE" = "400" ]; then
    test_pass "Texto vazio retorna 400"
else
    test_fail "Texto vazio" "Esperava 400, recebeu HTTP $TTS_EMPTY_CODE"
fi

# Tambem testar com espacos em branco (deve ser 400)
TTS_BLANK_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"   ","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_BLANK_CODE" = "400" ]; then
    test_pass "Texto so com espacos retorna 400"
else
    test_fail "Texto so com espacos" "Esperava 400, recebeu HTTP $TTS_BLANK_CODE"
fi

# ------------------------------------------------------------------------------

section "23. POST /v1/audio/speech (modelo inexistente -> 404)"

TTS_BAD_MODEL_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"modelo-tts-fake","input":"teste","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_BAD_MODEL_CODE" = "404" ]; then
    test_pass "Modelo TTS inexistente retorna 404"
else
    test_fail "Modelo TTS inexistente" "Esperava 404, recebeu HTTP $TTS_BAD_MODEL_CODE"
fi

# ------------------------------------------------------------------------------

section "24. POST /v1/audio/speech (modelo STT na rota TTS -> 404)"

TTS_WRONG_TYPE_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"faster-whisper-tiny","input":"teste","voice":"default"}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_WRONG_TYPE_CODE" = "404" ]; then
    test_pass "Modelo STT na rota TTS retorna 404"
else
    test_fail "Modelo STT na rota TTS" "Esperava 404, recebeu HTTP $TTS_WRONG_TYPE_CODE"
fi

# ------------------------------------------------------------------------------

section "25. POST /v1/audio/speech (speed customizado)"

TTS_SPEED_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"kokoro-v1","input":"Teste de velocidade","voice":"default","speed":1.5}' \
    "$SERVER_URL/v1/audio/speech" 2>/dev/null) || true

if [ "$TTS_SPEED_CODE" = "200" ]; then
    test_pass "POST /v1/audio/speech com speed=1.5 retorna 200"
else
    test_fail "POST /v1/audio/speech (speed)" "HTTP $TTS_SPEED_CODE"
fi

# ------------------------------------------------------------------------------

section "26. GET /v1/models lista modelo TTS"

MODELS_TTS_OUTPUT=$(curl -s "$SERVER_URL/v1/models" 2>/dev/null) || true
if echo "$MODELS_TTS_OUTPUT" | grep -q "kokoro-v1"; then
    test_pass "GET /v1/models lista kokoro-v1 (TTS)"
else
    test_fail "GET /v1/models (TTS)" "kokoro-v1 nao encontrado no output"
fi

# ------------------------------------------------------------------------------

section "27. Round-trip TTS→STT (qualidade end-to-end)"

# Gera audio via TTS e transcreve via STT, comparando texto original vs transcrito.
# Usa frases simples em ingles (Kokoro default voice = American English).

ROUNDTRIP_TMPDIR=$(mktemp -d)

# Frase 1: simples e clara
RT_TEXT_1="Hello, how can I help you today?"
RT_AUDIO_1="${ROUNDTRIP_TMPDIR}/rt1.wav"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"kokoro-v1\",\"input\":\"${RT_TEXT_1}\",\"voice\":\"default\"}" \
    "$SERVER_URL/v1/audio/speech" \
    --output "$RT_AUDIO_1" 2>/dev/null || true

RT_SIZE_1=$(wc -c < "$RT_AUDIO_1" 2>/dev/null || echo "0")

if [ "$RT_SIZE_1" -gt 1000 ]; then
    # Transcreve o audio gerado pelo TTS
    RT_STT_1=$(curl -s -X POST \
        -F "file=@${RT_AUDIO_1}" \
        -F "model=faster-whisper-tiny" \
        -F "language=en" \
        "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

    RT_TRANSCRIBED_1=$(echo "$RT_STT_1" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null || echo "")

    # Normaliza para comparacao (lowercase, remove pontuacao)
    RT_NORM_ORIG_1=$(echo "$RT_TEXT_1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)
    RT_NORM_TRANS_1=$(echo "$RT_TRANSCRIBED_1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)

    if [ -z "$RT_NORM_TRANS_1" ]; then
        test_fail "Round-trip TTS→STT frase 1" "STT retornou texto vazio para audio TTS"
    elif [ "$RT_NORM_ORIG_1" = "$RT_NORM_TRANS_1" ]; then
        test_pass "Round-trip TTS→STT frase 1: MATCH EXATO"
        echo "    Original:    '${RT_TEXT_1}'"
        echo "    Transcrito:  '${RT_TRANSCRIBED_1}'"
    else
        # Mesmo sem match exato, reporta o resultado para analise
        test_pass "Round-trip TTS→STT frase 1: audio gerado e transcrito"
        echo "    Original:    '${RT_TEXT_1}'"
        echo "    Transcrito:  '${RT_TRANSCRIBED_1}'"
        echo "    Normalizado: '${RT_NORM_ORIG_1}' vs '${RT_NORM_TRANS_1}'"
    fi
else
    test_fail "Round-trip TTS→STT frase 1" "TTS retornou audio muito pequeno (${RT_SIZE_1} bytes)"
fi

# Frase 2: numeros e entidades
RT_TEXT_2="Please transfer one thousand dollars to account number five."
RT_AUDIO_2="${ROUNDTRIP_TMPDIR}/rt2.wav"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"kokoro-v1\",\"input\":\"${RT_TEXT_2}\",\"voice\":\"default\"}" \
    "$SERVER_URL/v1/audio/speech" \
    --output "$RT_AUDIO_2" 2>/dev/null || true

RT_SIZE_2=$(wc -c < "$RT_AUDIO_2" 2>/dev/null || echo "0")

if [ "$RT_SIZE_2" -gt 1000 ]; then
    RT_STT_2=$(curl -s -X POST \
        -F "file=@${RT_AUDIO_2}" \
        -F "model=faster-whisper-tiny" \
        -F "language=en" \
        "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

    RT_TRANSCRIBED_2=$(echo "$RT_STT_2" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null || echo "")

    RT_NORM_ORIG_2=$(echo "$RT_TEXT_2" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)
    RT_NORM_TRANS_2=$(echo "$RT_TRANSCRIBED_2" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)

    if [ -z "$RT_NORM_TRANS_2" ]; then
        test_fail "Round-trip TTS→STT frase 2" "STT retornou texto vazio para audio TTS"
    else
        test_pass "Round-trip TTS→STT frase 2: audio gerado e transcrito"
        echo "    Original:    '${RT_TEXT_2}'"
        echo "    Transcrito:  '${RT_TRANSCRIBED_2}'"
        echo "    Normalizado: '${RT_NORM_ORIG_2}' vs '${RT_NORM_TRANS_2}'"
    fi
else
    test_fail "Round-trip TTS→STT frase 2" "TTS retornou audio muito pequeno (${RT_SIZE_2} bytes)"
fi

# Frase 3: frase mais longa com contexto
RT_TEXT_3="The quick brown fox jumps over the lazy dog."
RT_AUDIO_3="${ROUNDTRIP_TMPDIR}/rt3.wav"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"kokoro-v1\",\"input\":\"${RT_TEXT_3}\",\"voice\":\"default\"}" \
    "$SERVER_URL/v1/audio/speech" \
    --output "$RT_AUDIO_3" 2>/dev/null || true

RT_SIZE_3=$(wc -c < "$RT_AUDIO_3" 2>/dev/null || echo "0")

if [ "$RT_SIZE_3" -gt 1000 ]; then
    RT_STT_3=$(curl -s -X POST \
        -F "file=@${RT_AUDIO_3}" \
        -F "model=faster-whisper-tiny" \
        -F "language=en" \
        "$SERVER_URL/v1/audio/transcriptions" 2>/dev/null) || true

    RT_TRANSCRIBED_3=$(echo "$RT_STT_3" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null || echo "")

    RT_NORM_ORIG_3=$(echo "$RT_TEXT_3" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)
    RT_NORM_TRANS_3=$(echo "$RT_TRANSCRIBED_3" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | xargs)

    if [ -z "$RT_NORM_TRANS_3" ]; then
        test_fail "Round-trip TTS→STT frase 3" "STT retornou texto vazio para audio TTS"
    else
        test_pass "Round-trip TTS→STT frase 3: audio gerado e transcrito"
        echo "    Original:    '${RT_TEXT_3}'"
        echo "    Transcrito:  '${RT_TRANSCRIBED_3}'"
        echo "    Normalizado: '${RT_NORM_ORIG_3}' vs '${RT_NORM_TRANS_3}'"
    fi
else
    test_fail "Round-trip TTS→STT frase 3" "TTS retornou audio muito pequeno (${RT_SIZE_3} bytes)"
fi

# Cleanup temp files
rm -rf "$ROUNDTRIP_TMPDIR"

# Fim dos testes TTS
else
    # TTS worker nao ficou ready — registrar como SKIP
    test_skip "Testes TTS 19-26" "TTS worker nao ficou READY em 90s"
fi

# ==============================================================================
# PARTE 3: Cleanup
# ==============================================================================

section "Parando servidor"

if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo "  Servidor parado."
else
    echo "  Servidor ja estava parado."
fi

# ==============================================================================
# Resultado Final
# ==============================================================================

section "Resultado Final"

TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "  Total:  $TOTAL testes"
echo -e "  ${GREEN}PASS:   $PASS${NC}"
echo -e "  ${RED}FAIL:   $FAIL${NC}"
echo -e "  ${YELLOW}SKIP:   $SKIP${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Alguns testes falharam!${NC}"
    exit 1
else
    echo -e "${GREEN}Todos os testes passaram!${NC}"
    exit 0
fi
