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
    TranscriptEvent,
)
from theo.workers.stt.converters import (
    batch_result_to_proto_response,
    health_dict_to_proto_response,
    proto_request_to_transcribe_params,
    transcript_segment_to_proto_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from theo.proto.stt_worker_pb2 import (
        AudioFrame,
        CancelRequest,
        CancelResponse,
        HealthRequest,
        TranscribeFileRequest,
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
        request_iterator: AsyncIterator[AudioFrame],
        context: grpc.aio.ServicerContext[AudioFrame, TranscriptEvent],
    ) -> AsyncIterator[TranscriptEvent]:
        """Transcricao STT em streaming bidirecional.

        Recebe stream de AudioFrames, delega ao backend via transcribe_stream,
        e retorna stream de TranscriptEvents.

        Metadados (session_id, initial_prompt, hot_words) sao extraidos do
        primeiro AudioFrame antes de iniciar o backend.
        """
        # Ler primeiro frame para extrair metadados da sessao
        first_frame: AudioFrame | None = None
        async for frame in request_iterator:
            first_frame = frame
            break

        if first_frame is None:
            # Stream vazio — nenhum frame recebido
            return

        session_id = first_frame.session_id
        initial_prompt: str | None = (
            first_frame.initial_prompt if first_frame.initial_prompt else None
        )
        hot_words: list[str] | None = (
            list(first_frame.hot_words) if first_frame.hot_words else None
        )

        # Se o primeiro frame ja sinaliza fim, nao ha audio para processar
        if first_frame.is_last:
            logger.info("transcribe_stream_empty", session_id=session_id)
            return

        async def audio_chunk_generator() -> AsyncIterator[bytes]:
            """Extrai PCM bytes do stream de AudioFrames.

            Emite os dados do primeiro frame (ja lido) e depois consome
            o restante do request_iterator.
            """
            # Emitir dados do primeiro frame
            yield bytes(first_frame.data)

            # Consumir frames restantes
            async for frame in request_iterator:
                if context.cancelled():
                    return
                if frame.is_last:
                    return
                yield bytes(frame.data)

        logger.info("transcribe_stream_start", session_id=session_id)

        try:
            # transcribe_stream e implementado como async generator em todos
            # os backends (usa yield), entao retorna AsyncGenerator diretamente.
            # mypy interpreta a assinatura ABC como Coroutine -> AsyncIterator,
            # mas na pratica o resultado e iteravel sem await.
            stream = self._backend.transcribe_stream(
                audio_chunks=audio_chunk_generator(),
                language=None,
                initial_prompt=initial_prompt,
                hot_words=hot_words,
            )
            async for segment in stream:  # type: ignore[attr-defined]
                if context.cancelled():
                    logger.info("transcribe_stream_cancelled", session_id=session_id)
                    return
                event = transcript_segment_to_proto_event(segment, session_id)
                yield event
        except Exception as exc:
            logger.error(
                "transcribe_stream_error",
                session_id=session_id,
                error=str(exc),
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return  # pragma: no cover — unreachable em gRPC real

        logger.info("transcribe_stream_done", session_id=session_id)

    async def Cancel(  # noqa: N802  # type: ignore[override]
        self,
        request: CancelRequest,
        context: grpc.aio.ServicerContext[CancelRequest, CancelResponse],
    ) -> None:
        """Cancelamento unario — nao implementado.

        Cancelamento de streaming usa stream break via gRPC call.cancel()
        no StreamHandle (runtime-side). Este RPC unario e mantido para
        potencial uso futuro em cancelamento de requests batch.
        """
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Cancel unario nao implementado. "
            "Cancelamento de streaming usa stream break via gRPC call.cancel().",
        )

    async def Health(  # noqa: N802  # type: ignore[override]
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext[HealthRequest, object],
    ) -> object:
        """Health check do worker."""
        health = await self._backend.health()
        return health_dict_to_proto_response(health, self._model_name, self._engine)
