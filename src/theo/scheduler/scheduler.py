"""Scheduler — roteia requests de transcricao para workers gRPC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc.aio

from theo.exceptions import WorkerCrashError, WorkerTimeoutError, WorkerUnavailableError
from theo.logging import get_logger
from theo.proto.stt_worker_pb2_grpc import STTWorkerStub
from theo.scheduler.converters import build_proto_request, proto_response_to_batch_result

if TYPE_CHECKING:
    from theo._types import BatchResult
    from theo.registry.registry import ModelRegistry
    from theo.server.models.requests import TranscribeRequest
    from theo.workers.manager import WorkerManager

logger = get_logger("scheduler")

# gRPC channel options — defaults do gRPC sao 4MB (insuficiente para audio de 25MB)
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 30 * 1024 * 1024),
    ("grpc.max_receive_message_length", 30 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# Timeout minimo para TranscribeFile (segundos)
_MIN_GRPC_TIMEOUT = 30.0

# Fator aplicado a duracao estimada do audio para calcular timeout
_TIMEOUT_FACTOR = 2.0


class Scheduler:
    """Roteia requests de transcricao para workers gRPC.

    M3: implementacao trivial (1 worker = 1 request).
    M9: evolui para priorizacao, fila, batching.
    """

    def __init__(self, worker_manager: WorkerManager, registry: ModelRegistry) -> None:
        self._worker_manager = worker_manager
        self._registry = registry

    async def transcribe(self, request: TranscribeRequest) -> BatchResult:
        """Envia request ao worker e retorna resultado.

        Raises:
            ModelNotFoundError: Modelo nao existe no registry.
            WorkerUnavailableError: Nenhum worker READY para o modelo.
            WorkerTimeoutError: Worker nao respondeu dentro do timeout.
            WorkerCrashError: Worker retornou erro irrecuperavel.
        """
        # 1. Valida que modelo existe (levanta ModelNotFoundError se nao)
        self._registry.get_manifest(request.model_name)

        # 2. Encontra worker READY para o modelo
        worker = self._worker_manager.get_ready_worker(request.model_name)
        if worker is None:
            raise WorkerUnavailableError(request.model_name)

        logger.info(
            "transcribe_start",
            request_id=request.request_id,
            model=request.model_name,
            worker_id=worker.worker_id,
            audio_bytes=len(request.audio_data),
            task=request.task,
        )

        # 3. Envia gRPC TranscribeFile ao worker
        proto_request = build_proto_request(request)

        # Timeout proporcional ao tamanho do audio (estimativa: 16kHz * 2 bytes/sample)
        audio_duration_estimate = len(request.audio_data) / (16_000 * 2)
        timeout = max(_MIN_GRPC_TIMEOUT, audio_duration_estimate * _TIMEOUT_FACTOR)

        channel = grpc.aio.insecure_channel(
            f"localhost:{worker.port}",
            options=_GRPC_CHANNEL_OPTIONS,
        )
        try:
            stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
            proto_response = await stub.TranscribeFile(
                proto_request,
                timeout=timeout,
            )
        except grpc.aio.AioRpcError as exc:
            _translate_grpc_error(exc, worker.worker_id, timeout)
            raise  # pragma: no cover — _translate_grpc_error sempre levanta
        finally:
            try:
                await channel.close()
            except Exception:
                logger.warning("channel_close_error", worker_id=worker.worker_id)

        # 4. Converte proto -> BatchResult
        result = proto_response_to_batch_result(proto_response)

        logger.info(
            "transcribe_done",
            request_id=request.request_id,
            text_length=len(result.text),
            segments=len(result.segments),
        )

        return result


def _translate_grpc_error(
    exc: grpc.aio.AioRpcError,
    worker_id: str,
    timeout: float,
) -> None:
    """Traduz erros gRPC em exceptions de dominio do Theo.

    Sempre levanta — nunca retorna normalmente.
    """
    code = exc.code()

    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        raise WorkerTimeoutError(worker_id, timeout) from exc

    if code == grpc.StatusCode.UNAVAILABLE:
        raise WorkerCrashError(worker_id) from exc

    if code == grpc.StatusCode.CANCELLED:
        raise WorkerCrashError(worker_id) from exc

    # Demais erros gRPC → WorkerCrashError generico
    logger.error(
        "grpc_error",
        worker_id=worker_id,
        grpc_code=code.name if code else "UNKNOWN",
        grpc_details=exc.details(),
    )
    raise WorkerCrashError(worker_id) from exc
