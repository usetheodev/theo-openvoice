"""WS /v1/realtime -- endpoint WebSocket para streaming STT."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from theo._types import STTArchitecture
from theo.exceptions import ModelNotFoundError
from theo.logging import get_logger
from theo.server.models.events import (
    InputAudioBufferCommitCommand,
    SessionCancelCommand,
    SessionCloseCommand,
    SessionClosedEvent,
    SessionConfig,
    SessionConfigureCommand,
    SessionCreatedEvent,
    SessionFramesDroppedEvent,
    SessionRateLimitEvent,
    StreamingErrorEvent,
)
from theo.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)
from theo.session.backpressure import FramesDroppedAction, RateLimitAction

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from theo.server.models.events import ServerEvent
    from theo.session.streaming import StreamingSession

logger = get_logger("server.realtime")

router = APIRouter()

# Defaults (overrideable via app.state para testes com timeouts curtos)
_DEFAULT_HEARTBEAT_INTERVAL_S = 10.0
_DEFAULT_INACTIVITY_TIMEOUT_S = 60.0
_DEFAULT_CHECK_INTERVAL_S = 5.0


def _get_ws_timeouts(websocket: WebSocket) -> tuple[float, float, float]:
    """Retorna (inactivity_timeout, heartbeat_interval, check_interval).

    Valores sao lidos de ``app.state`` se presentes, senao usa defaults.
    Isso permite que testes sobrescrevam com timeouts curtos.
    """
    state = websocket.app.state
    inactivity = getattr(state, "ws_inactivity_timeout_s", _DEFAULT_INACTIVITY_TIMEOUT_S)
    heartbeat = getattr(state, "ws_heartbeat_interval_s", _DEFAULT_HEARTBEAT_INTERVAL_S)
    check = getattr(state, "ws_check_interval_s", _DEFAULT_CHECK_INTERVAL_S)
    return float(inactivity), float(heartbeat), float(check)


async def _send_event(
    websocket: WebSocket,
    event: ServerEvent,
    session_id: str | None = None,
) -> None:
    """Envia evento JSON para o cliente via WebSocket.

    Verifica se a conexao ainda esta ativa antes de enviar.

    Args:
        websocket: Conexao WebSocket.
        event: Evento server->client a enviar.
        session_id: ID da sessao para log correlation (opcional).
    """
    from starlette.websockets import WebSocketState as _WSState

    if websocket.client_state == _WSState.CONNECTED:
        await websocket.send_json(event.model_dump(mode="json"))
    else:
        logger.debug(
            "send_event_skipped_not_connected",
            session_id=session_id,
            event_type=event.type,
        )


async def _inactivity_monitor(
    websocket: WebSocket,
    session_id: str,
    session_start: float,
    last_audio_time_ref: list[float],
    session_ref: list[StreamingSession | None],
) -> str:
    """Background task que monitora inatividade e envia pings WebSocket.

    Executa periodicamente (a cada check_interval segundos) e verifica:
    1. Se nenhum audio frame foi recebido dentro do inactivity_timeout.
    2. Envia WebSocket ping a cada heartbeat_interval (best effort).

    Se inatividade for detectada, emite session.closed e fecha o WebSocket.

    Args:
        websocket: Conexao WebSocket ativa.
        session_id: ID da sessao para logging.
        session_start: Timestamp monotonic de inicio da sessao.
        last_audio_time_ref: Lista com um elemento float, usada como referencia
            mutavel para o ultimo timestamp de audio recebido.
        session_ref: Lista com referencia a StreamingSession (pode ser None).

    Returns:
        Razao de fechamento ("inactivity_timeout" ou "client_disconnect").
    """
    from starlette.websockets import WebSocketState as _WSState

    inactivity_timeout, heartbeat_interval, check_interval = _get_ws_timeouts(websocket)
    last_ping_sent = time.monotonic()

    while True:
        await asyncio.sleep(check_interval)

        if websocket.client_state != _WSState.CONNECTED:
            return "client_disconnect"

        now = time.monotonic()

        # Verificar inatividade (sem audio frames recebidos)
        if now - last_audio_time_ref[0] > inactivity_timeout:
            logger.info(
                "inactivity_timeout",
                session_id=session_id,
                timeout_s=inactivity_timeout,
            )
            # Fechar StreamingSession se existir
            session = session_ref[0]
            segments = session.segment_id if session is not None else 0
            if session is not None and not session.is_closed:
                await session.close()

            total_duration_ms = int((now - session_start) * 1000)
            closed_event = SessionClosedEvent(
                reason="inactivity_timeout",
                total_duration_ms=total_duration_ms,
                segments_transcribed=segments,
            )
            await _send_event(websocket, closed_event, session_id=session_id)

            with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                await websocket.close(code=1000, reason="inactivity_timeout")
            return "inactivity_timeout"

        # Enviar ping periodicamente (best effort)
        if now - last_ping_sent >= heartbeat_interval:
            try:
                if websocket.client_state == _WSState.CONNECTED:
                    await websocket.send({"type": "websocket.ping", "bytes": b""})
                    last_ping_sent = now
            except Exception:
                logger.debug(
                    "heartbeat_ping_failed",
                    session_id=session_id,
                )


def _create_streaming_session(
    websocket: WebSocket,
    session_id: str,
    on_event: Callable[[ServerEvent], Awaitable[None]],
    language: str | None = None,
    architecture: STTArchitecture = STTArchitecture.ENCODER_DECODER,
    engine_supports_hot_words: bool = False,
) -> StreamingSession | None:
    """Cria StreamingSession se streaming_grpc_client esta disponivel.

    Instancia per-session: StreamingPreprocessor, VADDetector
    (EnergyPreFilter + SileroVADClassifier), BackpressureController,
    e StreamingSession.

    Returns None se streaming_grpc_client nao esta configurado
    (ex: testes sem infra de worker).
    """
    from theo.session.streaming import StreamingSession as _StreamingSession

    state = websocket.app.state
    grpc_client = getattr(state, "streaming_grpc_client", None)
    if grpc_client is None:
        return None

    # Obter stages do preprocessing pipeline (batch) para reusar no streaming
    preprocessing_pipeline = getattr(state, "preprocessing_pipeline", None)
    stages = preprocessing_pipeline.stages if preprocessing_pipeline is not None else []

    # Obter postprocessor
    postprocessor = getattr(state, "postprocessing_pipeline", None)

    # Criar preprocessor de streaming
    from theo.preprocessing.streaming import StreamingPreprocessor

    preprocessor = StreamingPreprocessor(stages=stages)

    # Criar VAD (energy pre-filter + silero classifier + detector)
    from theo.vad.detector import VADDetector
    from theo.vad.energy import EnergyPreFilter
    from theo.vad.silero import SileroVADClassifier

    energy_pre_filter = EnergyPreFilter()
    silero_classifier = SileroVADClassifier()
    vad = VADDetector(
        energy_pre_filter=energy_pre_filter,
        silero_classifier=silero_classifier,
    )

    return _StreamingSession(
        session_id=session_id,
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        architecture=architecture,
        engine_supports_hot_words=engine_supports_hot_words,
    )


@router.websocket("/v1/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str | None = None,
    language: str | None = None,
) -> None:
    """Endpoint WebSocket para streaming STT bidirecional.

    Query params:
        model: Nome do modelo STT (obrigatorio).
        language: Codigo ISO 639-1 do idioma (opcional, default: auto-detect).

    Protocolo:
        1. Handshake: valida modelo via registry.
        2. Emite session.created com session_id unico.
        3. Cria StreamingSession (se worker disponivel).
        4. Main loop: recebe binary (audio frames) e text (JSON commands).
        5. Audio frames -> backpressure check -> session.process_frame().
        6. Background task monitora inatividade e envia heartbeat pings.
        7. On disconnect: fecha session e emite session.closed.
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # --- Validacao pre-accept ---
    if model is None:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="invalid_request",
            message="Query parameter 'model' is required",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason="Missing required query parameter: model")
        return

    registry = websocket.app.state.registry
    if registry is None:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="service_unavailable",
            message="No models available",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason="No models available")
        return

    try:
        manifest = registry.get_manifest(model)
    except ModelNotFoundError:
        await websocket.accept()
        error_event = StreamingErrorEvent(
            code="model_not_found",
            message=f"Model '{model}' not found in registry",
            recoverable=False,
        )
        await _send_event(websocket, error_event)
        await websocket.close(code=1008, reason=f"Model not found: {model}")
        return

    # Extrair architecture e hot_words capability do manifesto
    model_architecture = manifest.capabilities.architecture or STTArchitecture.ENCODER_DECODER
    model_supports_hot_words = manifest.capabilities.hot_words or False

    # --- Accept conexao ---
    await websocket.accept()

    session_start = time.monotonic()

    logger.info(
        "session_created",
        session_id=session_id,
        model=model,
        language=language,
    )

    config = SessionConfig(language=language)
    created_event = SessionCreatedEvent(
        session_id=session_id,
        model=model,
        config=config,
    )
    await _send_event(websocket, created_event, session_id=session_id)

    # Referencia mutavel para timestamp do ultimo audio recebido.
    # Usa lista de um elemento para permitir mutacao pela background task e main loop.
    last_audio_time: list[float] = [time.monotonic()]

    # --- Criar StreamingSession (se worker disponivel) ---
    async def _on_session_event(event: ServerEvent) -> None:
        """Callback: emite eventos da StreamingSession para o WebSocket."""
        await _send_event(websocket, event, session_id=session_id)

    session: StreamingSession | None = _create_streaming_session(
        websocket=websocket,
        session_id=session_id,
        on_event=_on_session_event,
        language=language,
        architecture=model_architecture,
        engine_supports_hot_words=model_supports_hot_words,
    )

    # Referencia mutavel para a session (usada pelo monitor de inatividade)
    session_ref: list[StreamingSession | None] = [session]

    # Backpressure controller (per-session)
    from theo.session.backpressure import BackpressureController

    backpressure = BackpressureController()

    monitor_task = asyncio.create_task(
        _inactivity_monitor(
            websocket=websocket,
            session_id=session_id,
            session_start=session_start,
            last_audio_time_ref=last_audio_time,
            session_ref=session_ref,
        ),
    )

    # --- Main loop ---
    closed_reason = "client_disconnect"
    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            result = dispatch_message(message)

            if result is None:
                continue

            if isinstance(result, ErrorResult):
                await _send_event(websocket, result.event, session_id=session_id)
                continue

            if isinstance(result, AudioFrameResult):
                last_audio_time[0] = time.monotonic()
                logger.debug(
                    "audio_frame_received",
                    session_id=session_id,
                    size_bytes=len(result.data),
                )

                # Backpressure check
                bp_action = backpressure.record_frame(len(result.data))
                if isinstance(bp_action, RateLimitAction):
                    await _send_event(
                        websocket,
                        SessionRateLimitEvent(
                            delay_ms=bp_action.delay_ms,
                            message="Client sending faster than real-time, please throttle",
                        ),
                        session_id=session_id,
                    )
                elif isinstance(bp_action, FramesDroppedAction):
                    await _send_event(
                        websocket,
                        SessionFramesDroppedEvent(
                            dropped_ms=bp_action.dropped_ms,
                            message=f"Backlog exceeded, {bp_action.dropped_ms}ms of audio dropped",
                        ),
                        session_id=session_id,
                    )
                    continue  # Frame descartado, nao enviar ao session

                # Enviar ao StreamingSession (se disponivel)
                if session is not None and not session.is_closed:
                    await session.process_frame(result.data)
                continue

            if isinstance(result, CommandResult):
                cmd = result.command

                if isinstance(cmd, SessionConfigureCommand):
                    logger.info(
                        "session_configure",
                        session_id=session_id,
                        language=cmd.language,
                        vad_sensitivity=(
                            cmd.vad_sensitivity.value if cmd.vad_sensitivity else None
                        ),
                    )
                    # Aplicar hot words e ITN settings na session
                    if session is not None:
                        if cmd.hot_words is not None:
                            session._hot_words = cmd.hot_words
                        if cmd.enable_itn is not None:
                            session._enable_itn = cmd.enable_itn

                elif isinstance(cmd, SessionCancelCommand):
                    logger.info("session_cancel", session_id=session_id)
                    closed_reason = "cancelled"
                    segments = session.segment_id if session is not None else 0
                    if session is not None and not session.is_closed:
                        await session.close()
                    closed = SessionClosedEvent(
                        reason="cancelled",
                        total_duration_ms=int(
                            (time.monotonic() - session_start) * 1000,
                        ),
                        segments_transcribed=segments,
                    )
                    await _send_event(websocket, closed, session_id=session_id)
                    break

                elif isinstance(cmd, SessionCloseCommand):
                    logger.info("session_close", session_id=session_id)
                    closed_reason = "client_request"
                    segments = session.segment_id if session is not None else 0
                    if session is not None and not session.is_closed:
                        await session.close()
                    closed = SessionClosedEvent(
                        reason="client_request",
                        total_duration_ms=int(
                            (time.monotonic() - session_start) * 1000,
                        ),
                        segments_transcribed=segments,
                    )
                    await _send_event(websocket, closed, session_id=session_id)
                    break

                elif isinstance(cmd, InputAudioBufferCommitCommand):
                    logger.info(
                        "input_audio_buffer_commit",
                        session_id=session_id,
                    )
                    if session is not None and not session.is_closed:
                        await session.commit()

    except WebSocketDisconnect:
        logger.info(
            "client_disconnected",
            session_id=session_id,
        )
    except Exception:
        logger.exception(
            "session_error",
            session_id=session_id,
        )
        error_event = StreamingErrorEvent(
            code="internal_error",
            message="Internal server error",
            recoverable=False,
        )
        await _send_event(websocket, error_event, session_id=session_id)
    finally:
        # Fechar StreamingSession se ativa
        if session is not None and not session.is_closed:
            await session.close()

        # Cancelar background task de inatividade
        monitor_task.cancel()
        try:
            monitor_result = await monitor_task
            if monitor_result == "inactivity_timeout":
                closed_reason = "inactivity_timeout"
        except asyncio.CancelledError:
            pass

        # Emitir session.closed apenas se nao foi emitido pelo monitor
        # ou por um comando (session.cancel / session.close)
        if closed_reason not in (
            "inactivity_timeout",
            "cancelled",
            "client_request",
        ):
            segments = session.segment_id if session is not None else 0
            total_duration_ms = int((time.monotonic() - session_start) * 1000)
            closed_event = SessionClosedEvent(
                reason=closed_reason,
                total_duration_ms=total_duration_ms,
                segments_transcribed=segments,
            )
            await _send_event(websocket, closed_event, session_id=session_id)

        logger.info(
            "session_closed",
            session_id=session_id,
            reason=closed_reason,
        )
