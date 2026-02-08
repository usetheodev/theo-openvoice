"""gRPC servicer para worker STT.

Implementa o servico STTWorker definido em stt_worker.proto.
Delega transcricao ao STTBackend injetado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from theo.logging import get_logger
from theo.proto.stt_worker_pb2 import (
    TranscribeFileResponse,
)
from theo.workers.stt.converters import (
    batch_result_to_proto_response,
    health_dict_to_proto_response,
    proto_request_to_transcribe_params,
)

if TYPE_CHECKING:
    from theo.proto.stt_worker_pb2 import (
        CancelRequest,
        CancelResponse,
        HealthRequest,
        TranscribeFileRequest,
        TranscriptEvent,
    )
    from theo.workers.stt.interface import STTBackend

from theo.proto.stt_worker_pb2_grpc import STTWorkerServicer as _BaseServicer

logger = get_logger("worker.stt.servicer")


class STTWorkerServicer(_BaseServicer):
    """Implementacao do servico gRPC STTWorker.

    Recebe requests gRPC, delega ao STTBackend, retorna respostas proto.
    """

    def __init__(self, backend: STTBackend, model_name: str, engine: str) -> None:
        self._backend = backend
        self._model_name = model_name
        self._engine = engine

    async def TranscribeFile(  # noqa: N802  # type: ignore[override]
        self,
        request: TranscribeFileRequest,
        context: grpc.aio.ServicerContext[TranscribeFileRequest, TranscribeFileResponse],
    ) -> TranscribeFileResponse:
        """Transcricao batch de arquivo de audio."""
        params = proto_request_to_transcribe_params(request)
        request_id = request.request_id

        logger.info(
            "transcribe_file_start",
            request_id=request_id,
            language=params.get("language"),
            audio_bytes=len(request.audio_data),
        )

        try:
            result = await self._backend.transcribe_file(**params)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("transcribe_file_error", request_id=request_id, error=str(exc))
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            # context.abort levanta AbortError no gRPC real, mas adicionamos
            # return explicito para seguranca — evita `result` indefinida se
            # abort nao levantar (ex: em mocks ou versoes futuras do gRPC).
            return TranscribeFileResponse()  # pragma: no cover — unreachable em gRPC real

        logger.info(
            "transcribe_file_done",
            request_id=request_id,
            text_length=len(result.text),
            segments=len(result.segments),
        )

        return batch_result_to_proto_response(result)

    async def TranscribeStream(  # noqa: N802  # type: ignore[override]
        self,
        request_iterator: object,
        context: grpc.aio.ServicerContext[object, TranscriptEvent],
    ) -> None:
        """Streaming STT — nao implementado nesta milestone."""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "TranscribeStream sera implementado no M5 (Streaming)",
        )

    async def Cancel(  # noqa: N802  # type: ignore[override]
        self,
        request: CancelRequest,
        context: grpc.aio.ServicerContext[CancelRequest, CancelResponse],
    ) -> None:
        """Cancelamento — nao implementado nesta milestone."""
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Cancel sera implementado no M5 (Streaming)",
        )

    async def Health(  # noqa: N802  # type: ignore[override]
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext[HealthRequest, object],
    ) -> object:
        """Health check do worker."""
        health = await self._backend.health()
        return health_dict_to_proto_response(health, self._model_name, self._engine)
