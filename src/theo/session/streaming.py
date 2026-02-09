"""StreamingSession — orquestrador central de streaming STT.

Coordena o fluxo: preprocessing -> VAD -> gRPC worker -> post-processing.
Cada sessao WebSocket possui uma instancia de StreamingSession que gerencia
o ciclo de vida completo do streaming.

Estado simplificado (M5): ACTIVE / CLOSED. Maquina de estados completa
(6 estados: INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED)
implementada no M6 (Session Manager).

Regras:
- ITN (post-processing) APENAS em transcript.final, NUNCA em partial.
- Hot words enviados apenas no PRIMEIRO frame de cada segmento de fala.
- Inactivity timeout: 60s sem audio -> CLOSED.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import time
from typing import TYPE_CHECKING

import numpy as np

from theo.exceptions import WorkerCrashError
from theo.logging import get_logger
from theo.server.models.events import (
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
    WordEvent,
)
from theo.session.metrics import (
    HAS_METRICS,
    stt_active_sessions,
    stt_final_delay_seconds,
    stt_ttfb_seconds,
    stt_vad_events_total,
)
from theo.vad.detector import VADEventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from theo.postprocessing.pipeline import PostProcessingPipeline
    from theo.preprocessing.streaming import StreamingPreprocessor
    from theo.scheduler.streaming import StreamHandle, StreamingGRPCClient
    from theo.server.models.events import ServerEvent
    from theo.vad.detector import VADDetector

logger = get_logger("session.streaming")

# Timeout de inatividade: 60s sem receber audio -> sessao fecha
_INACTIVITY_TIMEOUT_S = 60.0


class _SessionState(enum.Enum):
    """Estado simplificado da sessao (M5).

    Full state machine (6 estados) e implementada no M6 (Session Manager).
    """

    ACTIVE = "active"
    CLOSED = "closed"


class StreamingSession:
    """Orquestrador de streaming STT para uma sessao WebSocket.

    Coordena preprocessing, VAD, gRPC streaming com worker e
    post-processing. Emite eventos ao WebSocket handler via callback.

    Lifecycle tipico:
        1. Criar StreamingSession com dependencias injetadas
        2. Chamar process_frame() para cada frame de audio recebido
        3. Eventos sao emitidos via on_event callback
        4. Chamar close() para encerrar a sessao

    Args:
        session_id: Identificador unico da sessao.
        preprocessor: StreamingPreprocessor para normalizar audio.
        vad: VADDetector para detectar fala/silencio.
        grpc_client: StreamingGRPCClient para abrir streams com o worker.
        postprocessor: PostProcessingPipeline para ITN em finals.
        on_event: Callback async para emitir eventos ao WebSocket handler.
        hot_words: Lista de hot words para keyword boosting.
        enable_itn: Se True, aplica ITN nos transcript.final.
    """

    def __init__(
        self,
        session_id: str,
        preprocessor: StreamingPreprocessor,
        vad: VADDetector,
        grpc_client: StreamingGRPCClient,
        postprocessor: PostProcessingPipeline | None,
        on_event: Callable[[ServerEvent], Awaitable[None]],
        hot_words: list[str] | None = None,
        enable_itn: bool = True,
    ) -> None:
        self._session_id = session_id
        self._preprocessor = preprocessor
        self._vad = vad
        self._grpc_client = grpc_client
        self._postprocessor = postprocessor
        self._on_event = on_event
        self._hot_words = hot_words
        self._enable_itn = enable_itn

        # Estado
        self._state = _SessionState.ACTIVE
        self._segment_id = 0
        self._last_audio_time = time.monotonic()

        # gRPC stream handle (aberto durante speech)
        self._stream_handle: StreamHandle | None = None
        self._receiver_task: asyncio.Task[None] | None = None

        # Flag: hot words ja enviados para o segmento de fala atual?
        self._hot_words_sent_for_segment = False

        # Eventos recebidos do worker para o segmento atual
        # Armazenados para processar transcript.final apos SPEECH_END
        self._pending_final_event: TranscriptFinalEvent | None = None

        # Timestamp do inicio do segmento de fala atual (ms)
        self._speech_start_ms: int | None = None

        # Metricas de streaming
        # Timestamp monotonic do speech_start para calculo de TTFB
        self._speech_start_monotonic: float | None = None
        # Timestamp monotonic do speech_end para calculo de final_delay
        self._speech_end_monotonic: float | None = None
        # Flag: TTFB ja foi registrado para este segmento?
        self._ttfb_recorded_for_segment = False

        if HAS_METRICS and stt_active_sessions is not None:
            stt_active_sessions.inc()

    @property
    def session_id(self) -> str:
        """ID da sessao."""
        return self._session_id

    @property
    def is_closed(self) -> bool:
        """True se a sessao esta fechada."""
        return self._state == _SessionState.CLOSED

    @property
    def segment_id(self) -> int:
        """ID do segmento atual."""
        return self._segment_id

    async def process_frame(self, raw_bytes: bytes) -> None:
        """Processa um frame de audio cru do WebSocket.

        Fluxo:
            1. Verifica se sessao esta ativa
            2. Aplica preprocessing (PCM int16 -> float32 16kHz normalizado)
            3. Passa para VAD
            4. Se SPEECH_START: abre gRPC stream, emite evento
            5. Durante speech: envia frames ao worker
            6. Se SPEECH_END: fecha gRPC stream, emite evento

        Args:
            raw_bytes: Bytes PCM 16-bit little-endian mono.

        Raises:
            SessionClosedError: Se a sessao ja esta fechada.
        """
        if self._state == _SessionState.CLOSED:
            return

        self._last_audio_time = time.monotonic()

        # 1. Preprocessing: PCM int16 bytes -> float32 16kHz
        frame = self._preprocessor.process_frame(raw_bytes)
        if len(frame) == 0:
            return

        # 2. VAD
        vad_event = self._vad.process_frame(frame)

        # 3. Atuar conforme evento VAD
        if vad_event is not None:
            if vad_event.type == VADEventType.SPEECH_START:
                await self._handle_speech_start(vad_event.timestamp_ms)
            elif vad_event.type == VADEventType.SPEECH_END:
                await self._handle_speech_end(vad_event.timestamp_ms)

        # 4. Se estamos em fala e temos stream aberto, enviar frame ao worker
        if self._vad.is_speaking and self._stream_handle is not None:
            await self._send_frame_to_worker(frame)

    async def commit(self) -> None:
        """Force commit do segmento atual (manual commit).

        Fecha o stream gRPC atual, fazendo o worker emitir transcript.final
        para o audio acumulado. Incrementa segment_id e reseta estado para
        que o proximo audio abra novo stream.

        No-op se nao ha stream ativo (silencio) ou sessao fechada.
        """
        if self._state == _SessionState.CLOSED:
            return

        if self._stream_handle is None:
            return

        # 1. Fechar stream gRPC (envia is_last=True para flush)
        if not self._stream_handle.is_closed:
            try:
                await self._stream_handle.close()
            except WorkerCrashError:
                logger.warning(
                    "commit_stream_close_worker_crash",
                    session_id=self._session_id,
                )

        # 2. Aguardar receiver task finalizar (worker retorna transcript.final)
        if self._receiver_task is not None and not self._receiver_task.done():
            try:
                await asyncio.wait_for(self._receiver_task, timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "commit_receiver_task_timeout",
                    session_id=self._session_id,
                )
                self._receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver_task
            except asyncio.CancelledError:
                pass

        self._receiver_task = None
        self._stream_handle = None

        # 3. Incrementar segment_id para proximo segmento
        self._segment_id += 1

        # 4. Resetar flag de hot words para que sejam enviados no proximo stream
        self._hot_words_sent_for_segment = False

        logger.debug(
            "manual_commit",
            session_id=self._session_id,
            segment_id=self._segment_id,
        )

    async def close(self) -> None:
        """Fecha a sessao e libera recursos.

        Idempotente: chamadas em sessao ja fechada sao no-op.
        """
        if self._state == _SessionState.CLOSED:
            return

        self._state = _SessionState.CLOSED

        if HAS_METRICS and stt_active_sessions is not None:
            stt_active_sessions.dec()

        # Cancelar receiver task se ativa
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
            self._receiver_task = None

        # Fechar gRPC stream se aberto
        if self._stream_handle is not None:
            try:
                await self._stream_handle.cancel()
            except Exception:
                logger.debug(
                    "stream_cancel_error_on_close",
                    session_id=self._session_id,
                )
            self._stream_handle = None

        logger.info(
            "session_closed",
            session_id=self._session_id,
            segment_id=self._segment_id,
        )

    def check_inactivity(self) -> bool:
        """Verifica se a sessao expirou por inatividade.

        Returns:
            True se a sessao expirou (>60s sem audio).
        """
        if self._state == _SessionState.CLOSED:
            return False
        elapsed = time.monotonic() - self._last_audio_time
        return elapsed > _INACTIVITY_TIMEOUT_S

    async def _cleanup_current_stream(self) -> None:
        """Limpa stream gRPC e receiver task anteriores.

        Defesa contra duplo SPEECH_START: se o VAD emitir SPEECH_START sem
        SPEECH_END anterior (edge case de debounce), o stream/task anteriores
        seriam vazados. Este metodo garante cleanup antes de abrir novo stream.
        """
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
            self._receiver_task = None

        if self._stream_handle is not None and not self._stream_handle.is_closed:
            try:
                await self._stream_handle.cancel()
            except Exception:
                logger.debug(
                    "cleanup_stream_cancel_error",
                    session_id=self._session_id,
                )
            self._stream_handle = None

    async def _handle_speech_start(self, timestamp_ms: int) -> None:
        """Abre gRPC stream e emite vad.speech_start."""
        # Limpar stream anterior se existir (defesa contra duplo SPEECH_START)
        if self._stream_handle is not None or self._receiver_task is not None:
            await self._cleanup_current_stream()

        self._speech_start_ms = timestamp_ms
        self._hot_words_sent_for_segment = False
        self._speech_start_monotonic = time.monotonic()
        self._ttfb_recorded_for_segment = False

        if HAS_METRICS and stt_vad_events_total is not None:
            stt_vad_events_total.labels(event_type="speech_start").inc()

        # Abrir stream gRPC com o worker
        try:
            self._stream_handle = await self._grpc_client.open_stream(
                self._session_id,
            )
        except WorkerCrashError:
            await self._emit_error(
                code="worker_crash",
                message="Worker unavailable, cannot open stream",
                recoverable=True,
            )
            return

        # Iniciar task de recepcao de eventos do worker
        self._receiver_task = asyncio.create_task(
            self._receive_worker_events(),
        )

        # Emitir evento VAD
        await self._on_event(
            VADSpeechStartEvent(timestamp_ms=timestamp_ms),
        )

        logger.debug(
            "speech_start",
            session_id=self._session_id,
            timestamp_ms=timestamp_ms,
            segment_id=self._segment_id,
        )

    async def _handle_speech_end(self, timestamp_ms: int) -> None:
        """Fecha gRPC stream, aguarda eventos finais, emite vad.speech_end.

        Garante que TODOS os transcript.final do worker sejam emitidos ANTES
        do vad.speech_end, respeitando a semantica do protocolo WebSocket
        (PRD secao 9: transcript.final vem antes de vad.speech_end).
        """
        self._speech_end_monotonic = time.monotonic()

        if HAS_METRICS and stt_vad_events_total is not None:
            stt_vad_events_total.labels(event_type="speech_end").inc()

        # Fechar stream gRPC (envia is_last=True)
        if self._stream_handle is not None and not self._stream_handle.is_closed:
            try:
                await self._stream_handle.close()
            except WorkerCrashError:
                logger.warning(
                    "stream_close_worker_crash",
                    session_id=self._session_id,
                )

        # Aguardar receiver task finalizar (com timeout para nao travar).
        # Isso garante que transcript.final seja emitido ANTES de vad.speech_end.
        receiver_ok = True
        if self._receiver_task is not None and not self._receiver_task.done():
            try:
                await asyncio.wait_for(self._receiver_task, timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "receiver_task_timeout",
                    session_id=self._session_id,
                )
                self._receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receiver_task
                receiver_ok = False
            except asyncio.CancelledError:
                receiver_ok = False

        self._receiver_task = None
        self._stream_handle = None

        # Emitir vad.speech_end SOMENTE apos receiver task completar.
        # Se receiver falhou (timeout/cancel), ainda emitimos speech_end
        # para manter o contrato, mas logamos o problema.
        if not receiver_ok:
            logger.warning(
                "speech_end_after_receiver_failure",
                session_id=self._session_id,
                timestamp_ms=timestamp_ms,
            )

        await self._on_event(
            VADSpeechEndEvent(timestamp_ms=timestamp_ms),
        )

        # Incrementar segment_id para o proximo segmento de fala
        self._segment_id += 1
        self._speech_start_ms = None
        self._speech_start_monotonic = None
        self._speech_end_monotonic = None

        logger.debug(
            "speech_end",
            session_id=self._session_id,
            timestamp_ms=timestamp_ms,
            segment_id=self._segment_id,
        )

    async def _send_frame_to_worker(self, frame: np.ndarray) -> None:
        """Converte float32 para PCM int16 bytes e envia ao worker."""
        if self._stream_handle is None or self._stream_handle.is_closed:
            return

        # Converter float32 [-1.0, 1.0] -> int16 bytes
        pcm_int16 = (frame * 32767.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        # Hot words apenas no primeiro frame do segmento
        hot_words: list[str] | None = None
        if not self._hot_words_sent_for_segment and self._hot_words:
            hot_words = self._hot_words
            self._hot_words_sent_for_segment = True

        try:
            await self._stream_handle.send_frame(
                pcm_data=pcm_bytes,
                hot_words=hot_words,
            )
        except WorkerCrashError:
            await self._emit_error(
                code="worker_crash",
                message="Worker crashed during streaming",
                recoverable=True,
            )

    async def _receive_worker_events(self) -> None:
        """Background task: consome eventos do worker via gRPC.

        Converte TranscriptSegment em eventos do protocolo WebSocket
        e emite via callback. Post-processing (ITN) e aplicado APENAS
        em transcript.final.
        """
        if self._stream_handle is None:
            return

        try:
            async for segment in self._stream_handle.receive_events():
                if self._state == _SessionState.CLOSED:
                    break

                # Registrar TTFB no primeiro transcript (partial ou final)
                self._record_ttfb()

                if segment.is_final:
                    # Registrar final_delay se speech_end ja ocorreu
                    self._record_final_delay()

                    # Aplicar post-processing (ITN) em finals
                    text = segment.text
                    if self._enable_itn and self._postprocessor is not None:
                        text = self._postprocessor.process(text)

                    # Converter word timestamps
                    words: list[WordEvent] | None = None
                    if segment.words:
                        words = [
                            WordEvent(
                                word=w.word,
                                start=w.start,
                                end=w.end,
                            )
                            for w in segment.words
                        ]

                    await self._on_event(
                        TranscriptFinalEvent(
                            text=text,
                            segment_id=self._segment_id,
                            start_ms=segment.start_ms or 0,
                            end_ms=segment.end_ms or 0,
                            language=segment.language,
                            confidence=segment.confidence,
                            words=words,
                        ),
                    )
                else:
                    # Partial: emitir sem post-processing
                    await self._on_event(
                        TranscriptPartialEvent(
                            text=segment.text,
                            segment_id=self._segment_id,
                            timestamp_ms=segment.start_ms or 0,
                        ),
                    )
        except WorkerCrashError:
            await self._emit_error(
                code="worker_crash",
                message="Worker crashed during streaming",
                recoverable=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "receiver_unexpected_error",
                session_id=self._session_id,
                error=str(exc),
            )
            await self._emit_error(
                code="internal_error",
                message=f"Unexpected error: {exc}",
                recoverable=False,
            )

    def _record_ttfb(self) -> None:
        """Registra TTFB (Time to First Byte) para o segmento atual.

        TTFB = tempo entre speech_start e o primeiro transcript (partial ou final).
        Registrado uma unica vez por segmento de fala.
        """
        if (
            not HAS_METRICS
            or stt_ttfb_seconds is None
            or self._ttfb_recorded_for_segment
            or self._speech_start_monotonic is None
        ):
            return

        ttfb = time.monotonic() - self._speech_start_monotonic
        stt_ttfb_seconds.observe(ttfb)
        self._ttfb_recorded_for_segment = True

    def _record_final_delay(self) -> None:
        """Registra final_delay para o segmento atual.

        Final delay = tempo entre speech_end e transcript.final.
        So registrado quando speech_end ja ocorreu antes do final.
        """
        if (
            not HAS_METRICS
            or stt_final_delay_seconds is None
            or self._speech_end_monotonic is None
        ):
            return

        delay = time.monotonic() - self._speech_end_monotonic
        stt_final_delay_seconds.observe(delay)

    async def _emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool,
    ) -> None:
        """Emite evento de erro via callback."""
        await self._on_event(
            StreamingErrorEvent(
                code=code,
                message=message,
                recoverable=recoverable,
                resume_segment_id=self._segment_id if recoverable else None,
            ),
        )
