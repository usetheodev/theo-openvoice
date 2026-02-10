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
# NOTA: Este script baixa o modelo faster-whisper-tiny (~75MB) do HuggingFace
#       na primeira execucao. O download leva ~30-60s dependendo da conexao.
#       O modelo e reutilizado em execucoes subsequentes.
#
# NOTA: O teste de transcricao usa um audio sintetico (sine tone 440Hz).
#       O resultado da transcricao pode ser vazio ou impreciso — o objetivo
#       e validar que o fluxo completo funciona, nao a qualidade do STT.
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
    test_pass "theo list mostra faster-whisper-tiny"
elif echo "$LIST_OUTPUT" | grep -q "Nenhum modelo"; then
    test_skip "theo list" "Nenhum modelo instalado (precisa de 'theo pull' primeiro)"
else
    test_fail "theo list" "Output inesperado: $LIST_OUTPUT"
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
