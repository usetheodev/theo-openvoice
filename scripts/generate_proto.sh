#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROTO_DIR="${PROJECT_ROOT}/src/theo/proto"

echo "Generating gRPC stubs from ${PROTO_DIR}/stt_worker.proto ..."

${PYTHON:-python} -m grpc_tools.protoc \
  --proto_path="${PROTO_DIR}" \
  --python_out="${PROTO_DIR}" \
  --grpc_python_out="${PROTO_DIR}" \
  "${PROTO_DIR}/stt_worker.proto"

# Fix import path: grpc_tools generates 'import stt_worker_pb2'
# but we need 'from theo.proto import stt_worker_pb2' for package imports.
sed -i 's/^import stt_worker_pb2/from theo.proto import stt_worker_pb2/' \
  "${PROTO_DIR}/stt_worker_pb2_grpc.py"

echo "Generated:"
echo "  ${PROTO_DIR}/stt_worker_pb2.py"
echo "  ${PROTO_DIR}/stt_worker_pb2_grpc.py"
