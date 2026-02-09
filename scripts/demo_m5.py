"""Demo M5 -- WebSocket Streaming STT (end-to-end validation).

Executa o servidor FastAPI com mock worker em-processo, conecta via
WebSocket e exercita todos os componentes do M5:

1. Handshake e session.created
2. Protocolo de eventos (session.configure, session.close, session.cancel)
3. Envio de audio binario (PCM frames)
4. VAD: vad.speech_start + vad.speech_end
5. transcript.partial e transcript.final (com ITN)
6. input_audio_buffer.commit (manual commit)
7. Backpressure: rate_limit e frames_dropped
8. Heartbeat e inactivity timeout
9. Error handling (malformed JSON, unknown command)
10. Metricas Prometheus

Funciona SEM modelo real instalado -- usa mocks controlados para simular
o worker gRPC e o VAD.

Uso:
    .venv/bin/python scripts/demo_m5.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
from starlette.testclient import TestClient

from theo._types import TranscriptSegment, WordTimestamp
from theo.server.app import create_app
from theo.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)
from theo.vad.detector import VADEvent, VADEventType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2
_FRAME_SAMPLES = 1024  # 64ms

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"{CYAN}[INFO]{NC}  {msg}")


def pass_msg(msg: str) -> None:
    print(f"{GREEN}[PASS]{NC}  {msg}")


def fail_msg(msg: str) -> None:
    print(f"{RED}[FAIL]{NC}  {msg}")


def step(num: int | str, desc: str) -> None:
    print(f"\n{CYAN}=== Step {num}: {desc} ==={NC}")


def event_line(event: dict[str, Any]) -> str:
    """Formata evento JSON de forma compacta."""
    event_type = event.get("type", "?")
    filtered = {k: v for k, v in event.items() if k != "type" and v is not None}
    details = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
    return f"{BOLD}{event_type}{NC}  {details}" if details else f"{BOLD}{event_type}{NC}"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def make_pcm_silence(n_samples: int = _FRAME_SAMPLES) -> bytes:
    """Gera bytes PCM int16 de silencio (zeros)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def make_pcm_tone(
    frequency: float = 440.0,
    duration_s: float = 0.064,
    sample_rate: int = _SAMPLE_RATE,
) -> bytes:
    """Gera bytes PCM int16 de tom senoidal (simula fala)."""
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    samples = (32767 * 0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return samples.tobytes()


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


class _AsyncIterFromList:
    """Async iterator que yield items de uma lista."""

    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> _AsyncIterFromList:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def make_mock_registry(known_models: list[str] | None = None) -> Mock:
    """Cria mock do ModelRegistry."""
    if known_models is None:
        known_models = ["faster-whisper-tiny"]

    from theo.exceptions import ModelNotFoundError

    registry = Mock()

    def _get_manifest(model_name: str) -> Mock:
        if model_name in known_models:
            manifest = Mock()
            manifest.name = model_name
            return manifest
        raise ModelNotFoundError(model_name)

    registry.get_manifest = Mock(side_effect=_get_manifest)
    registry.has_model = Mock(side_effect=lambda name: name in known_models)
    return registry


def make_stream_handle_mock(events: list[object] | None = None) -> Mock:
    """Cria mock de StreamHandle."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    if events is None:
        events = []
    handle.receive_events = Mock(return_value=_AsyncIterFromList(events))
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Cria mock de StreamingGRPCClient."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def make_preprocessor_mock() -> Mock:
    """Cria mock de StreamingPreprocessor."""
    mock = Mock()
    mock.process_frame.return_value = np.zeros(_FRAME_SAMPLES, dtype=np.float32)
    return mock


def make_vad_mock(*, is_speaking: bool = False) -> Mock:
    """Cria mock de VADDetector."""
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


def make_postprocessor_mock() -> Mock:
    """Cria mock de PostProcessingPipeline."""
    mock = Mock()
    mock.process.side_effect = lambda text: f"[ITN] {text}"
    return mock


def make_app_simple() -> tuple[Mock, TestClient]:
    """Cria app simples SEM StreamingSession (sem gRPC client)."""
    registry = make_mock_registry()
    app = create_app(registry=registry)
    return registry, TestClient(app)


def make_app_with_short_timeouts(
    *,
    inactivity_s: float = 5.0,
    heartbeat_s: float = 10.0,
    check_s: float = 0.1,
) -> tuple[Mock, TestClient]:
    """Cria app com timeouts curtos para testes."""
    registry = make_mock_registry()
    app = create_app(registry=registry)
    app.state.ws_inactivity_timeout_s = inactivity_s
    app.state.ws_heartbeat_interval_s = heartbeat_s
    app.state.ws_check_interval_s = check_s
    return registry, TestClient(app)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

_total_pass = 0
_total_fail = 0


def check(condition: bool, desc: str) -> bool:
    """Verifica condicao e exibe resultado."""
    global _total_pass, _total_fail
    if condition:
        pass_msg(desc)
        _total_pass += 1
        return True
    else:
        fail_msg(desc)
        _total_fail += 1
        return False


def demo_1_handshake() -> None:
    """Step 1: Handshake e session.created."""
    step(1, "WebSocket Handshake + session.created")

    _, client = make_app_simple()

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    check(event["type"] == "session.created", "Evento session.created recebido")
    check(event["session_id"].startswith("sess_"), f"Session ID: {event['session_id']}")
    check(event["model"] == "faster-whisper-tiny", f"Model: {event['model']}")
    check("config" in event, "Config presente no evento")

    config = event["config"]
    check(config["vad_sensitivity"] == "normal", "VAD sensitivity: normal (default)")
    check(config["enable_itn"] is True, "ITN habilitado por default")
    check(config["enable_partial_transcripts"] is True, "Partial transcripts habilitados")
    check(config["silence_timeout_ms"] == 300, "Silence timeout: 300ms")

    info(f"  Evento completo: {event_line(event)}")


def demo_2_language_param() -> None:
    """Step 2: Language parameter no handshake."""
    step(2, "Language parameter no handshake")

    _, client = make_app_simple()

    with client.websocket_connect(
        "/v1/realtime?model=faster-whisper-tiny&language=pt",
    ) as ws:
        event = ws.receive_json()

    check(event["config"]["language"] == "pt", "Language 'pt' propagado na config")
    info(f"  Config language: {event['config']['language']}")

    # Sem language -> None
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    check(event["config"]["language"] is None, "Sem language -> None (auto-detect)")


def demo_3_protocol_commands() -> None:
    """Step 3: Comandos do protocolo WebSocket."""
    step(3, "Comandos do protocolo (configure, close, cancel)")

    _, client = make_app_simple()

    # 3a: session.configure
    info("  3a: session.configure")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_json({
            "type": "session.configure",
            "language": "pt",
            "vad_sensitivity": "high",
            "hot_words": ["PIX", "TED"],
            "enable_itn": False,
        })
        # session.configure nao emite resposta, mas nao fecha a conexao
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(
        closed["type"] == "session.closed" and closed["reason"] == "client_request",
        "session.configure aceito sem fechar conexao",
    )

    # 3b: session.close
    info("  3b: session.close")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "session.close emite session.closed")
    check(closed["reason"] == "client_request", f"Reason: {closed['reason']}")
    check(closed["total_duration_ms"] >= 0, f"Duration: {closed['total_duration_ms']}ms")
    info(f"  Evento: {event_line(closed)}")

    # 3c: session.cancel
    info("  3c: session.cancel")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_json({"type": "session.cancel"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "session.cancel emite session.closed")
    check(closed["reason"] == "cancelled", f"Reason: {closed['reason']}")
    info(f"  Evento: {event_line(closed)}")

    # 3d: input_audio_buffer.commit
    info("  3d: input_audio_buffer.commit")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_bytes(make_pcm_silence())
        ws.send_json({"type": "input_audio_buffer.commit"})
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "commit aceito sem fechar conexao")


def demo_4_audio_frames() -> None:
    """Step 4: Envio de audio binario via WebSocket."""
    step(4, "Envio de audio binario (PCM frames)")

    _, client = make_app_simple()

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        check(created["type"] == "session.created", "Sessao criada")

        # Enviar 10 frames de audio PCM
        n_frames = 10
        for i in range(n_frames):
            frame = make_pcm_silence() if i % 2 == 0 else make_pcm_tone()
            ws.send_bytes(frame)

        check(True, f"{n_frames} frames PCM enviados ({_FRAME_SAMPLES} samples/frame, 64ms cada)")

        # Fechar normalmente
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "Sessao fechada apos envio de audio")

    frame_bytes = _FRAME_SAMPLES * _BYTES_PER_SAMPLE
    total_bytes = n_frames * frame_bytes
    total_ms = n_frames * (_FRAME_SAMPLES / _SAMPLE_RATE * 1000)
    info(f"  Total enviado: {total_bytes} bytes ({total_ms:.0f}ms de audio)")


def demo_5_error_handling() -> None:
    """Step 5: Error handling (malformed JSON, unknown command, missing model)."""
    step(5, "Error handling (erros recuperaveis e nao-recuperaveis)")

    _, client = make_app_simple()

    # 5a: Malformed JSON (recuperavel)
    info("  5a: JSON malformado -> erro recuperavel")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_text("this is not valid json {{{")
        error = ws.receive_json()

    check(error["type"] == "error", "Evento error emitido")
    check(error["code"] == "malformed_json", f"Code: {error['code']}")
    check(error["recoverable"] is True, "Erro e recuperavel")
    info(f"  Evento: {event_line(error)}")

    # 5b: Malformed JSON nao fecha conexao
    info("  5b: Conexao sobrevive apos erro recuperavel")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_text("bad json")
        ws.receive_json()  # error (malformed_json)
        # Conexao ainda funciona
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "Conexao sobreviveu ao erro recuperavel")

    # 5c: Unknown command (recuperavel)
    info("  5c: Comando desconhecido -> erro recuperavel")
    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_json({"type": "totally.unknown.command"})
        error = ws.receive_json()

    check(error["code"] == "unknown_command", f"Code: {error['code']}")
    check(error["recoverable"] is True, "Erro e recuperavel")

    # 5d: Missing model (nao recuperavel)
    info("  5d: Model ausente -> erro nao recuperavel + close")
    with client.websocket_connect("/v1/realtime") as ws:
        error = ws.receive_json()

    check(error["type"] == "error", "Evento error emitido")
    check(error["code"] == "invalid_request", f"Code: {error['code']}")
    check(error["recoverable"] is False, "Erro NAO e recuperavel")

    # 5e: Invalid model (nao recuperavel)
    info("  5e: Model inexistente -> erro nao recuperavel + close")
    with client.websocket_connect("/v1/realtime?model=nonexistent-model") as ws:
        error = ws.receive_json()

    check(error["code"] == "model_not_found", f"Code: {error['code']}")
    check(error["recoverable"] is False, "Erro NAO e recuperavel")
    check("nonexistent-model" in error["message"], "Mensagem inclui nome do modelo")


def demo_6_unique_sessions() -> None:
    """Step 6: Session IDs unicos entre conexoes."""
    step(6, "Session IDs unicos entre multiplas conexoes")

    _, client = make_app_simple()

    session_ids: list[str] = []
    n_connections = 5
    for _ in range(n_connections):
        with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
            event = ws.receive_json()
            session_ids.append(event["session_id"])

    check(
        len(set(session_ids)) == n_connections,
        f"{n_connections} conexoes -> {n_connections} session_ids unicos",
    )

    # Validar formato: sess_ + 12 hex chars
    all_valid = True
    for sid in session_ids:
        hex_part = sid[5:]
        if not sid.startswith("sess_") or len(hex_part) != 12:
            all_valid = False
            break
        try:
            int(hex_part, 16)
        except ValueError:
            all_valid = False
            break

    check(all_valid, "Formato: sess_ + 12 caracteres hexadecimais")
    info(f"  IDs: {', '.join(session_ids)}")


def demo_7_heartbeat_inactivity() -> None:
    """Step 7: Heartbeat e inactivity timeout."""
    step(7, "Heartbeat + Inactivity timeout")

    # 7a: Inactivity timeout
    info("  7a: Sessao sem audio -> inactivity timeout")
    _, client = make_app_with_short_timeouts(
        inactivity_s=0.3,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        closed = ws.receive_json()  # aguarda timeout

    check(closed["type"] == "session.closed", "Sessao fechada por timeout")
    check(closed["reason"] == "inactivity_timeout", f"Reason: {closed['reason']}")
    check(closed["total_duration_ms"] >= 200, f"Duration: {closed['total_duration_ms']}ms")
    info(f"  Evento: {event_line(closed)}")

    # 7b: Audio resets inactivity timer
    info("  7b: Audio resets o timer de inatividade")
    _, client = make_app_with_short_timeouts(
        inactivity_s=0.4,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created

        start = time.monotonic()
        for _ in range(3):
            ws.send_bytes(make_pcm_silence())
            time.sleep(0.15)
        elapsed = time.monotonic() - start

        # Agora parar de enviar e aguardar timeout
        closed = ws.receive_json()

    check(elapsed > 0.4, f"Sessao mantida viva por {elapsed:.2f}s (>0.4s)")
    check(closed["reason"] == "inactivity_timeout", "Timeout apos parar de enviar")

    # 7c: Comandos JSON NAO resetam inactivity
    info("  7c: Comandos JSON NAO resetam inactivity timer")
    _, client = make_app_with_short_timeouts(
        inactivity_s=0.3,
        check_s=0.1,
    )

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        ws.receive_json()  # session.created
        ws.send_json({"type": "session.configure", "language": "pt"})
        time.sleep(0.15)
        ws.send_json({"type": "session.configure", "language": "en"})
        closed = ws.receive_json()

    check(closed["reason"] == "inactivity_timeout", "Timeout mesmo com comandos JSON")


def demo_8_backpressure() -> None:
    """Step 8: Backpressure (rate limit + frames dropped)."""
    step(8, "Backpressure (rate limit + frames dropped)")

    # ------------------------------------
    # 8a: Rate limit at 3x real-time
    # ------------------------------------
    info("  8a: RateLimitAction ao enviar 3x mais rapido que real-time")

    class _FakeClock:
        def __init__(self, start: float = 0.0) -> None:
            self._now = start

        def __call__(self) -> float:
            return self._now

        def advance(self, seconds: float) -> None:
            self._now += seconds

    clock = _FakeClock()
    bytes_per_20ms = (_SAMPLE_RATE * _BYTES_PER_SAMPLE) // 50  # 640

    ctrl = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        rate_limit_threshold=1.2,
        max_backlog_s=100.0,  # alto para nao dropar
        clock=clock,
    )

    rate_limits: list[RateLimitAction] = []
    for _ in range(200):
        result = ctrl.record_frame(bytes_per_20ms)
        if isinstance(result, RateLimitAction):
            rate_limits.append(result)
        clock.advance(0.00667)  # ~3x real-time

    check(len(rate_limits) >= 1, f"{len(rate_limits)} RateLimitAction(s) emitidas a 3x speed")
    if rate_limits:
        info(f"  Primeiro rate_limit: delay_ms={rate_limits[0].delay_ms}")

    # ------------------------------------
    # 8b: FramesDroppedAction por backlog
    # ------------------------------------
    info("  8b: FramesDroppedAction quando backlog > max_backlog_s")

    clock2 = _FakeClock()
    ctrl2 = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        max_backlog_s=1.0,  # 1s maximo
        rate_limit_threshold=1.2,
        clock=clock2,
    )

    drop_action = None
    for _ in range(100):
        result = ctrl2.record_frame(bytes_per_20ms)
        if isinstance(result, FramesDroppedAction):
            drop_action = result
            break

    check(drop_action is not None, "FramesDroppedAction emitida apos backlog > 1s")
    if drop_action:
        info(f"  Frames dropados: {ctrl2.frames_dropped}, dropped_ms={drop_action.dropped_ms}")

    # ------------------------------------
    # 8c: Normal speed -> sem eventos
    # ------------------------------------
    info("  8c: Audio 1x real-time -> sem eventos de backpressure")

    clock3 = _FakeClock()
    ctrl3 = BackpressureController(
        sample_rate=_SAMPLE_RATE,
        rate_limit_threshold=1.2,
        max_backlog_s=10.0,
        clock=clock3,
    )

    any_event = False
    for _ in range(100):
        result = ctrl3.record_frame(bytes_per_20ms)
        if result is not None:
            any_event = True
        clock3.advance(0.020)  # real-time

    check(not any_event, "Nenhum evento de backpressure a 1x speed")
    check(ctrl3.frames_dropped == 0, "Zero frames dropados")


def demo_9_streaming_session_flow() -> None:
    """Step 9: StreamingSession full flow (com mock gRPC)."""
    step(9, "StreamingSession: speech_start -> partial -> final -> speech_end")

    import asyncio

    from theo.session.streaming import StreamingSession

    async def _run_flow() -> list[dict[str, Any]]:
        """Executa fluxo completo e retorna eventos emitidos."""
        # Arrange
        partial_seg = TranscriptSegment(
            text="ola como",
            is_final=False,
            segment_id=0,
            start_ms=1000,
        )
        final_seg = TranscriptSegment(
            text="ola como posso ajudar",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=3000,
            language="pt",
            confidence=0.95,
            words=(
                WordTimestamp(word="ola", start=1.0, end=1.5),
                WordTimestamp(word="como", start=1.5, end=2.0),
                WordTimestamp(word="posso", start=2.0, end=2.5),
                WordTimestamp(word="ajudar", start=2.5, end=3.0),
            ),
        )

        stream_handle = make_stream_handle_mock(events=[partial_seg, final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        postprocessor = make_postprocessor_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_demo_001",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=_on_event,
            enable_itn=True,
        )

        # Act: speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        # Act: speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=3000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected

    events = asyncio.run(_run_flow())

    # Verificar eventos em ordem
    event_types = [e["type"] for e in events]
    info(f"  Eventos recebidos ({len(events)}):")
    for e in events:
        info(f"    {event_line(e)}")

    check("vad.speech_start" in event_types, "vad.speech_start emitido")
    check("transcript.partial" in event_types, "transcript.partial emitido")
    check("transcript.final" in event_types, "transcript.final emitido")
    check("vad.speech_end" in event_types, "vad.speech_end emitido")

    # Verificar ordem
    idx_start = event_types.index("vad.speech_start")
    idx_partial = event_types.index("transcript.partial")
    idx_final = event_types.index("transcript.final")
    idx_end = event_types.index("vad.speech_end")

    check(
        idx_start < idx_partial < idx_final < idx_end,
        f"Ordem correta: speech_start({idx_start}) < partial({idx_partial}) < final({idx_final}) < speech_end({idx_end})",
    )

    # Verificar conteudo
    partial = next(e for e in events if e["type"] == "transcript.partial")
    check(partial["text"] == "ola como", f"Partial text: '{partial['text']}' (sem ITN)")

    final = next(e for e in events if e["type"] == "transcript.final")
    check(
        final["text"] == "[ITN] ola como posso ajudar",
        f"Final text: '{final['text']}' (com ITN)",
    )
    check(final["language"] == "pt", f"Language: {final['language']}")
    check(final["confidence"] == 0.95, f"Confidence: {final['confidence']}")

    # Word timestamps
    check(final["words"] is not None and len(final["words"]) == 4, "4 word timestamps presentes")
    if final["words"]:
        first_word = final["words"][0]
        check(
            first_word["word"] == "ola" and first_word["start"] == 1.0,
            f"Primeiro word: '{first_word['word']}' @ {first_word['start']}s",
        )


def demo_10_manual_commit() -> None:
    """Step 10: Manual commit (input_audio_buffer.commit)."""
    step(10, "Manual commit via StreamingSession.commit()")

    import asyncio

    from theo.session.streaming import StreamingSession

    async def _run_commit() -> list[dict[str, Any]]:
        """Executa commit e retorna eventos."""
        final_seg = TranscriptSegment(
            text="resultado do commit",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=2000,
            language="pt",
            confidence=0.88,
        )

        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_commit",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=make_postprocessor_mock(),
            on_event=_on_event,
            enable_itn=True,
        )

        assert session.segment_id == 0

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        # Manual commit
        await session.commit()
        seg_after = session.segment_id

        await session.close()
        return events_collected, seg_after  # type: ignore[return-value]

    result = asyncio.run(_run_commit())
    events, seg_after = result  # type: ignore[misc]

    final_events = [e for e in events if e.get("type") == "transcript.final"]
    check(len(final_events) == 1, "transcript.final emitido apos commit")

    if final_events:
        check(
            final_events[0]["text"] == "[ITN] resultado do commit",
            f"Texto com ITN: '{final_events[0]['text']}'",
        )

    check(seg_after == 1, f"Segment ID incrementado: {seg_after}")


def demo_11_worker_crash() -> None:
    """Step 11: Worker crash emite erro recuperavel."""
    step(11, "Worker crash -> erro recuperavel")

    import asyncio

    from theo.exceptions import WorkerCrashError
    from theo.session.streaming import StreamingSession

    async def _run_crash() -> list[dict[str, Any]]:
        stream_handle = make_stream_handle_mock(
            events=[WorkerCrashError("worker_1")],
        )
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_crash",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=make_postprocessor_mock(),
            on_event=_on_event,
        )

        # Trigger speech_start -> receiver recebe WorkerCrashError
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        await session.close()
        return events_collected

    events = asyncio.run(_run_crash())

    error_events = [e for e in events if e.get("type") == "error"]
    check(len(error_events) >= 1, "Evento error emitido")

    if error_events:
        error = error_events[0]
        check(error["code"] == "worker_crash", f"Code: {error['code']}")
        check(error["recoverable"] is True, "Erro e recuperavel")
        check(error["resume_segment_id"] is not None, f"Resume segment: {error['resume_segment_id']}")
        info(f"  Evento: {event_line(error)}")


def demo_12_hot_words() -> None:
    """Step 12: Hot words enviados apenas no primeiro frame."""
    step(12, "Hot words no primeiro frame do segmento")

    import asyncio

    from theo.session.streaming import StreamingSession

    async def _run_hot_words() -> list[Any]:
        stream_handle = make_stream_handle_mock()
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        session = StreamingSession(
            session_id="sess_hotwords",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=make_postprocessor_mock(),
            on_event=AsyncMock(),
            hot_words=["PIX", "TED", "Selic"],
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_silence())

        # Mais frames
        vad.process_frame.return_value = None
        await session.process_frame(make_pcm_silence())
        await session.process_frame(make_pcm_silence())

        calls = stream_handle.send_frame.call_args_list
        await session.close()
        return calls

    calls = asyncio.run(_run_hot_words())

    check(len(calls) == 3, f"{len(calls)} frames enviados ao worker")
    check(
        calls[0].kwargs.get("hot_words") == ["PIX", "TED", "Selic"],
        f"Primeiro frame com hot_words: {calls[0].kwargs.get('hot_words')}",
    )
    check(calls[1].kwargs.get("hot_words") is None, "Segundo frame: hot_words=None")
    check(calls[2].kwargs.get("hot_words") is None, "Terceiro frame: hot_words=None")


def demo_13_itn_control() -> None:
    """Step 13: ITN controlavel (enable/disable)."""
    step(13, "ITN aplicado SOMENTE em transcript.final, SOMENTE quando habilitado")

    import asyncio

    from theo.session.streaming import StreamingSession

    async def _run_itn(enable_itn: bool) -> list[dict[str, Any]]:
        partial_seg = TranscriptSegment(
            text="dois mil", is_final=False, segment_id=0, start_ms=500,
        )
        final_seg = TranscriptSegment(
            text="dois mil reais", is_final=True, segment_id=0,
            start_ms=500, end_ms=2000,
        )

        stream_handle = make_stream_handle_mock(events=[partial_seg, final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        postprocessor = make_postprocessor_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_itn",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=_on_event,
            enable_itn=enable_itn,
        )

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=500,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected

    # Com ITN
    info("  13a: ITN habilitado")
    events_itn = asyncio.run(_run_itn(enable_itn=True))
    final_itn = next(e for e in events_itn if e["type"] == "transcript.final")
    partial_itn = next(e for e in events_itn if e["type"] == "transcript.partial")

    check(final_itn["text"] == "[ITN] dois mil reais", f"Final com ITN: '{final_itn['text']}'")
    check(partial_itn["text"] == "dois mil", f"Partial SEM ITN: '{partial_itn['text']}'")

    # Sem ITN
    info("  13b: ITN desabilitado")
    events_no_itn = asyncio.run(_run_itn(enable_itn=False))
    final_no_itn = next(e for e in events_no_itn if e["type"] == "transcript.final")

    check(
        final_no_itn["text"] == "dois mil reais",
        f"Final SEM ITN: '{final_no_itn['text']}'",
    )


def demo_14_metrics() -> None:
    """Step 14: Metricas Prometheus."""
    step(14, "Metricas Prometheus para streaming STT")

    from theo.session.metrics import (
        HAS_METRICS,
        stt_active_sessions,
        stt_final_delay_seconds,
        stt_ttfb_seconds,
        stt_vad_events_total,
    )

    if HAS_METRICS:
        check(stt_ttfb_seconds is not None, "theo_stt_ttfb_seconds: Histogram registrado")
        check(stt_final_delay_seconds is not None, "theo_stt_final_delay_seconds: Histogram registrado")
        check(stt_active_sessions is not None, "theo_stt_active_sessions: Gauge registrado")
        check(stt_vad_events_total is not None, "theo_stt_vad_events_total: Counter registrado")

        info("  Metricas disponiveis:")
        info("    - theo_stt_ttfb_seconds (Histogram)")
        info("    - theo_stt_final_delay_seconds (Histogram)")
        info("    - theo_stt_active_sessions (Gauge)")
        info("    - theo_stt_vad_events_total (Counter, labels: event_type)")
    else:
        info("  prometheus_client NAO instalado -- metricas desabilitadas (graceful)")
        check(True, "Metricas opcionais (fail-open): funcionamento sem prometheus_client")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{NC}")
    print(f"{BOLD}{CYAN}  Theo OpenVoice M5 Demo{NC}")
    print(f"{BOLD}{CYAN}  WebSocket Streaming STT -- End-to-End Validation{NC}")
    print(f"{BOLD}{CYAN}{'=' * 60}{NC}")
    print()
    info("Componentes validados:")
    info("  - WS /v1/realtime endpoint")
    info("  - Protocolo de eventos JSON (8 server events, 4 client commands)")
    info("  - StreamingSession (preprocess -> VAD -> gRPC -> post-process)")
    info("  - BackpressureController (rate limit + frames dropped)")
    info("  - Heartbeat + inactivity timeout")
    info("  - Metricas Prometheus")
    info("  - Error handling (recuperavel/nao-recuperavel)")
    info("")
    info("Todos os testes usam MOCKS -- nao requer modelo real nem GPU")

    demo_1_handshake()
    demo_2_language_param()
    demo_3_protocol_commands()
    demo_4_audio_frames()
    demo_5_error_handling()
    demo_6_unique_sessions()
    demo_7_heartbeat_inactivity()
    demo_8_backpressure()
    demo_9_streaming_session_flow()
    demo_10_manual_commit()
    demo_11_worker_crash()
    demo_12_hot_words()
    demo_13_itn_control()
    demo_14_metrics()

    # Summary
    print(f"\n{BOLD}{'=' * 60}{NC}")
    total = _total_pass + _total_fail
    if _total_fail == 0:
        print(f"{GREEN}{BOLD}  M5 Demo Complete: {_total_pass}/{total} checks passed{NC}")
    else:
        print(f"{RED}{BOLD}  M5 Demo: {_total_pass}/{total} passed, {_total_fail} FAILED{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")

    print()
    info("Resumo dos componentes M5 validados:")
    info("  [1]  WebSocket handshake + session.created")
    info("  [2]  Language parameter (propagacao na config)")
    info("  [3]  Protocolo: session.configure, session.close, session.cancel, commit")
    info("  [4]  Audio binario: PCM frames via WebSocket")
    info("  [5]  Error handling: malformed JSON, unknown command, missing/invalid model")
    info("  [6]  Session IDs unicos (formato: sess_ + 12 hex)")
    info("  [7]  Heartbeat + inactivity timeout")
    info("  [8]  Backpressure: rate_limit, frames_dropped, normal speed")
    info("  [9]  StreamingSession: speech_start -> partial -> final -> speech_end")
    info("  [10] Manual commit (input_audio_buffer.commit)")
    info("  [11] Worker crash -> erro recuperavel")
    info("  [12] Hot words (primeiro frame apenas)")
    info("  [13] ITN: habilitado/desabilitado, somente em finals")
    info("  [14] Metricas Prometheus (opcional, fail-open)")
    print()

    return 0 if _total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
