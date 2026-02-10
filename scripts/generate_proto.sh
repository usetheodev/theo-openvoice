#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/src/theo/proto"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  --pyi_out="$PROTO_DIR" \
  "$PROTO_DIR/stt_worker.proto" \
  "$PROTO_DIR/tts_worker.proto"
