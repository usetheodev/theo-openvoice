"""Demo M6 -- Session Manager (end-to-end validation).

Exercita TODOS os componentes do M6 do ponto de vista do usuario:

1.  WebSocket Handshake (confirma compatibilidade com M5)
2.  Maquina de Estados: INIT -> ACTIVE -> SILENCE (visivel via eventos)
3.  Hot Words per Session: session.configure com hot_words
4.  session.hold: silencio prolongado transita para HOLD com evento
5.  Ring Buffer + Force Commit: segmento longo sem silencio gera commit automatico
6.  Manual Commit (input_audio_buffer.commit): usuario forca flush
7.  Crash Recovery: worker crash -> erro recuperavel -> retomada sem duplicacao
8.  Cross-Segment Context: texto do segmento anterior como initial_prompt
9.  WAL Checkpoints: recovery restaura segment_id correto
10. Metricas Prometheus M6: session_duration, force_committed, confidence, recoveries

Funciona SEM modelo real instalado -- usa mocks controlados.

Uso:
    .venv/bin/python scripts/demo_m6.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
from starlette.testclient import TestClient

from theo._types import SessionState, TranscriptSegment
from theo.exceptions import WorkerCrashError
from theo.server.app import create_app
from theo.session.cross_segment import CrossSegmentContext
from theo.session.ring_buffer import RingBuffer
from theo.session.state_machine import SessionStateMachine, SessionTimeouts
from theo.session.streaming import StreamingSession
from theo.session.wal import SessionWAL
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


# ---------------------------------------------------------------------------
# Demo results
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


# ===========================================================================
# Demo Steps
# ===========================================================================


def demo_1_ws_handshake() -> None:
    """Step 1: WebSocket handshake (confirma compatibilidade M5/M6)."""
    step(1, "WebSocket Handshake + session.created")

    _, client = make_app_simple()

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        event = ws.receive_json()

    check(event["type"] == "session.created", "Evento session.created recebido")
    check(event["session_id"].startswith("sess_"), f"Session ID: {event['session_id']}")
    check(event["model"] == "faster-whisper-tiny", f"Model: {event['model']}")
    info(f"  Evento: {event_line(event)}")


def demo_2_state_machine_transitions() -> None:
    """Step 2: Maquina de Estados — INIT -> ACTIVE -> SILENCE via eventos."""
    step(2, "State Machine: INIT -> ACTIVE -> SILENCE (via eventos VAD)")

    async def _run() -> list[dict[str, Any]]:
        partial_seg = TranscriptSegment(
            text="ola mundo",
            is_final=False,
            segment_id=0,
            start_ms=1000,
        )
        final_seg = TranscriptSegment(
            text="ola mundo como vai",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=3000,
            language="pt",
            confidence=0.92,
        )

        stream_handle = make_stream_handle_mock(events=[partial_seg, final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        postprocessor = make_postprocessor_mock()

        # Criar state machine para acompanhar estado
        sm = SessionStateMachine()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_states",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=_on_event,
            state_machine=sm,
        )

        # Estado inicial: INIT
        state_init = session.session_state

        # SPEECH_START -> transita para ACTIVE
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        state_active = session.session_state

        # SPEECH_END -> transita para SILENCE
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=3000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        state_silence = session.session_state

        await session.close()
        state_closed = session.session_state

        return events_collected, state_init, state_active, state_silence, state_closed

    result = asyncio.run(_run())
    events = result[0]
    state_init, state_active, state_silence, state_closed = result[1], result[2], result[3], result[4]

    check(state_init == SessionState.INIT, f"Estado inicial: {state_init.value}")
    check(state_active == SessionState.ACTIVE, f"Apos speech_start: {state_active.value}")
    check(state_silence == SessionState.SILENCE, f"Apos speech_end: {state_silence.value}")
    check(state_closed == SessionState.CLOSED, f"Apos close(): {state_closed.value}")

    event_types = [e["type"] for e in events]
    check("vad.speech_start" in event_types, "Usuario recebe vad.speech_start")
    check("transcript.final" in event_types, "Usuario recebe transcript.final")
    check("vad.speech_end" in event_types, "Usuario recebe vad.speech_end")

    info("  Eventos do usuario:")
    for e in events:
        info(f"    {event_line(e)}")


def demo_3_hot_words_per_session() -> None:
    """Step 3: Hot Words via session.configure — usuario configura termos de dominio."""
    step(3, "Hot Words per Session (session.configure)")

    async def _run() -> tuple[list[dict[str, Any]], list[str]]:
        final_seg = TranscriptSegment(
            text="voce pode fazer um PIX de cem reais",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=2000,
            language="pt",
            confidence=0.95,
        )
        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        postprocessor = make_postprocessor_mock()

        events_collected: list[dict[str, Any]] = []
        initial_prompts_sent: list[str] = []

        async def _capture_send_frame(
            *, pcm_data: bytes, initial_prompt: str | None = None,
            hot_words: list[str] | None = None,
        ) -> None:
            if initial_prompt:
                initial_prompts_sent.append(initial_prompt)

        stream_handle.send_frame = AsyncMock(side_effect=_capture_send_frame)

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_hotwords",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=_on_event,
            hot_words=["PIX", "TED", "Selic"],
            enable_itn=True,
        )

        # Simular: usuario configura hot words via session.configure
        session.update_hot_words(["PIX", "TED", "Selic", "CDI"])

        # speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.01)

        # Enviar frame com fala (hot words enviados no primeiro frame)
        vad.process_frame.return_value = None
        await session.process_frame(make_pcm_tone())

        # speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected, initial_prompts_sent

    events, prompts = asyncio.run(_run())

    check(len(prompts) >= 1, f"initial_prompt enviado ao worker ({len(prompts)} vez(es))")
    if prompts:
        check(
            "PIX" in prompts[0] and "CDI" in prompts[0],
            f"Hot words no prompt: '{prompts[0]}'",
        )

    finals = [e for e in events if e.get("type") == "transcript.final"]
    check(len(finals) >= 1, "transcript.final recebido com hot words aplicados")
    if finals:
        info(f"  Texto final: '{finals[0]['text']}'")


def demo_4_session_hold() -> None:
    """Step 4: session.hold — silencio prolongado transita SILENCE -> HOLD."""
    step(4, "session.hold (silencio prolongado)")

    async def _run() -> list[dict[str, Any]]:
        # State machine com timeout curto de SILENCE para teste
        sm = SessionStateMachine(
            timeouts=SessionTimeouts(
                init_timeout_s=30.0,
                silence_timeout_s=1.0,  # 1s para demo
                hold_timeout_s=300.0,
                closing_timeout_s=2.0,
            ),
        )

        final_seg = TranscriptSegment(
            text="sessao ativa", is_final=True, segment_id=0,
            start_ms=0, end_ms=1000, language="pt", confidence=0.9,
        )
        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_hold",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            state_machine=sm,
        )

        # speech_start -> ACTIVE
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())
        await asyncio.sleep(0.05)

        # speech_end -> SILENCE
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        state_after_silence = session.session_state

        # Esperar timeout de SILENCE (1s) -> transita para HOLD
        await asyncio.sleep(1.2)
        await session.check_timeout()

        state_hold = session.session_state

        await session.close()
        return events_collected, state_after_silence, state_hold

    events, state_silence, state_hold = asyncio.run(_run())

    check(state_silence == SessionState.SILENCE, f"Apos speech_end: {state_silence.value}")
    check(state_hold == SessionState.HOLD, f"Apos timeout de silencio: {state_hold.value}")

    hold_events = [e for e in events if e.get("type") == "session.hold"]
    check(len(hold_events) == 1, "Usuario recebe evento session.hold")
    if hold_events:
        check(
            hold_events[0]["hold_timeout_ms"] == 300000,
            f"hold_timeout_ms={hold_events[0]['hold_timeout_ms']}",
        )
        info(f"  Evento: {event_line(hold_events[0])}")


def demo_5_ring_buffer_force_commit() -> None:
    """Step 5: Ring Buffer force commit — segmento longo sem silencio."""
    step(5, "Ring Buffer: force commit automatico quando buffer > 90%")

    async def _run() -> tuple[list[dict[str, Any]], int]:
        # Ring buffer dimensionado para que 90% seja atingido antes do overrun.
        # Com frames de 2048 bytes (1024 samples * 2 bytes), buffer de 0.7s
        # = 22400 bytes. 90% = 20160 bytes = ~9.8 frames. Apos 10 frames
        # (20480 bytes, 91.4%), force commit dispara.
        rb = RingBuffer(
            duration_s=0.7,  # 22400 bytes
            sample_rate=_SAMPLE_RATE,
            bytes_per_sample=_BYTES_PER_SAMPLE,
        )

        final_seg = TranscriptSegment(
            text="segmento forcado", is_final=True, segment_id=0,
            start_ms=0, end_ms=500, language="pt", confidence=0.88,
        )

        # Stream handle que emite transcript.final quando close() e chamado.
        # Simula worker real: processa audio acumulado e retorna resultado
        # ao fechar o stream (commit fecha o stream).
        #
        # NOTA: __anext__ deve estar no CLASS, nao no instance — Python resolve
        # dunder methods via type(), nao via instance dict.

        class _ForceCommitHandle:
            """Yields final_seg once when close_event fires, then stops."""

            def __init__(self, seg: TranscriptSegment) -> None:
                self.is_closed = False
                self.session_id = "force_commit_test"
                self._seg = seg
                self._close_event = asyncio.Event()
                self._emitted = False

            def receive_events(self) -> _ForceCommitHandle:
                return self

            def __aiter__(self) -> _ForceCommitHandle:
                return self

            async def __anext__(self) -> TranscriptSegment:
                if self._emitted:
                    raise StopAsyncIteration
                await self._close_event.wait()
                self._emitted = True
                return self._seg

            async def send_frame(
                self, *, pcm_data: bytes,
                initial_prompt: str | None = None,
                hot_words: list[str] | None = None,
            ) -> None:
                pass

            async def close(self) -> None:
                self._close_event.set()
                await asyncio.sleep(0)

            async def cancel(self) -> None:
                self._close_event.set()

        class _EmptyHandle:
            """Handle that waits for close, then stops (no events)."""

            def __init__(self) -> None:
                self.is_closed = False
                self.session_id = "force_commit_test_2"
                self._close_event = asyncio.Event()

            def receive_events(self) -> _EmptyHandle:
                return self

            def __aiter__(self) -> _EmptyHandle:
                return self

            async def __anext__(self) -> TranscriptSegment:
                await self._close_event.wait()
                raise StopAsyncIteration

            async def send_frame(
                self, *, pcm_data: bytes,
                initial_prompt: str | None = None,
                hot_words: list[str] | None = None,
            ) -> None:
                pass

            async def close(self) -> None:
                self._close_event.set()
                await asyncio.sleep(0)

            async def cancel(self) -> None:
                self._close_event.set()

        handle1 = _ForceCommitHandle(final_seg)
        handle2 = _EmptyHandle()
        call_count = 0

        async def _open_stream(session_id: str) -> _ForceCommitHandle | _EmptyHandle:
            nonlocal call_count
            call_count += 1
            return handle1 if call_count <= 1 else handle2

        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(side_effect=_open_stream)
        grpc_client.close = AsyncMock()

        vad = make_vad_mock()
        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_force_commit",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            ring_buffer=rb,
        )

        # speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.01)

        # Enviar frames ate preencher >90% do ring buffer
        # (10 frames de 2048 = 20480 bytes > 90% de 22400 = 20160)
        vad.process_frame.return_value = None
        for _ in range(12):
            await session.process_frame(make_pcm_tone())
            await asyncio.sleep(0.005)

        seg_id = session.segment_id
        await session.close()
        return events_collected, seg_id

    events, seg_id = asyncio.run(_run())

    finals = [e for e in events if e.get("type") == "transcript.final"]
    check(
        len(finals) >= 1,
        f"transcript.final emitido por force commit ({len(finals)} recebido(s))",
    )
    check(seg_id >= 1, f"Segment ID incrementado apos force commit: {seg_id}")

    info("  O usuario recebe transcript.final automaticamente quando o buffer enche,")
    info("  sem precisar esperar por silencio (VAD speech_end).")


def demo_6_manual_commit() -> None:
    """Step 6: Manual commit via input_audio_buffer.commit."""
    step(6, "Manual Commit (input_audio_buffer.commit)")

    _, client = make_app_simple()

    with client.websocket_connect("/v1/realtime?model=faster-whisper-tiny") as ws:
        created = ws.receive_json()
        check(created["type"] == "session.created", "Sessao criada")

        # Enviar audio e commit manual (sem StreamingSession, testa protocolo)
        ws.send_bytes(make_pcm_silence())
        ws.send_json({"type": "input_audio_buffer.commit"})

        # Fechar
        ws.send_json({"type": "session.close"})
        closed = ws.receive_json()

    check(closed["type"] == "session.closed", "Sessao fechada apos commit manual")
    info("  Comando input_audio_buffer.commit aceito pelo protocolo WebSocket")

    # Teste mais completo: commit com StreamingSession
    async def _run_commit() -> tuple[list[dict[str, Any]], int]:
        final_seg = TranscriptSegment(
            text="resultado do commit manual",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=1000, language="pt", confidence=0.91,
        )
        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_manual_commit",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=make_postprocessor_mock(),
            on_event=_on_event,
            enable_itn=True,
            wal=wal,
        )

        # speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.05)

        # commit()
        await session.commit()
        seg_after = session.segment_id

        await session.close()
        return events_collected, seg_after

    events, seg = asyncio.run(_run_commit())

    finals = [e for e in events if e.get("type") == "transcript.final"]
    check(len(finals) == 1, "transcript.final emitido apos commit manual")
    if finals:
        check(
            "[ITN]" in finals[0]["text"],
            f"ITN aplicado: '{finals[0]['text']}'",
        )
    check(seg == 1, f"Segment ID incrementado: {seg}")


def demo_7_crash_recovery() -> None:
    """Step 7: Crash Recovery — worker crash, usuario ve erro recuperavel, sessao continua."""
    step(7, "Crash Recovery (worker crash -> retomada sem duplicacao)")

    async def _run() -> tuple[list[dict[str, Any]], int, int]:
        # Worker vai crashar na primeira stream
        crash_handle = make_stream_handle_mock(
            events=[WorkerCrashError("worker_1")],
        )

        # Worker novo apos recovery emite transcript.final
        recovery_seg = TranscriptSegment(
            text="retomada apos crash",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=1000, language="pt", confidence=0.85,
        )
        recovery_handle = make_stream_handle_mock(events=[recovery_seg])

        call_count = 0

        async def _open_stream(session_id: str) -> Mock:
            nonlocal call_count
            call_count += 1
            return crash_handle if call_count <= 1 else recovery_handle

        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(side_effect=_open_stream)
        grpc_client.close = AsyncMock()

        vad = make_vad_mock()

        # WAL com checkpoint para simular estado pre-crash
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=2, buffer_offset=100, timestamp_ms=5000)

        # Ring buffer com dados "nao commitados"
        rb = RingBuffer(duration_s=5.0)

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_recovery",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            wal=wal,
            ring_buffer=rb,
        )

        # speech_start -> abre stream -> worker crasha
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=5000,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.2)

        seg_after = session.segment_id
        wal_seg = wal.last_committed_segment_id

        await session.close()
        return events_collected, seg_after, wal_seg

    events, seg_id, wal_seg = asyncio.run(_run())

    # O usuario ve um erro recuperavel
    error_events = [e for e in events if e.get("type") == "error"]
    check(len(error_events) >= 1, f"Usuario recebe evento de erro ({len(error_events)})")
    if error_events:
        first_error = error_events[0]
        check(
            first_error.get("recoverable") is True,
            f"Erro e recuperavel: recoverable={first_error.get('recoverable')}",
        )
        check(
            "recovery" in first_error.get("message", "").lower()
            or "crash" in first_error.get("message", "").lower(),
            f"Mensagem do erro: '{first_error.get('message')}'",
        )
        info(f"  Evento de erro: {event_line(first_error)}")

    check(seg_id >= 3, f"Segment ID restaurado do WAL: {seg_id} (esperado >= 3)")

    info("  O usuario ve um erro recuperavel e a sessao continua transparentemente.")
    info(f"  WAL last_committed_segment_id: {wal_seg}")


def demo_8_cross_segment_context() -> None:
    """Step 8: Cross-Segment Context — texto anterior como initial_prompt."""
    step(8, "Cross-Segment Context (conditioning entre segmentos)")

    async def _run() -> tuple[list[dict[str, Any]], list[str | None]]:
        # Dois segmentos sequenciais
        seg1_final = TranscriptSegment(
            text="Olá, gostaria de fazer uma transferência",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=2000, language="pt", confidence=0.95,
        )
        seg2_final = TranscriptSegment(
            text="de duzentos reais via PIX",
            is_final=True, segment_id=1,
            start_ms=3000, end_ms=5000, language="pt", confidence=0.93,
        )

        handle1 = make_stream_handle_mock(events=[seg1_final])
        handle2 = make_stream_handle_mock(events=[seg2_final])

        call_count = 0
        initial_prompts_sent: list[str | None] = []

        async def _capture_send_frame(
            *, pcm_data: bytes, initial_prompt: str | None = None,
            hot_words: list[str] | None = None,
        ) -> None:
            initial_prompts_sent.append(initial_prompt)

        handle1.send_frame = AsyncMock(side_effect=_capture_send_frame)
        handle2.send_frame = AsyncMock(side_effect=_capture_send_frame)

        async def _open_stream(session_id: str) -> Mock:
            nonlocal call_count
            call_count += 1
            return handle1 if call_count <= 1 else handle2

        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(side_effect=_open_stream)
        grpc_client.close = AsyncMock()

        vad = make_vad_mock()
        csc = CrossSegmentContext(max_tokens=224)

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_context",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            cross_segment_context=csc,
        )

        # --- Segmento 1 ---
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        context_after_seg1 = csc.get_prompt()

        # --- Segmento 2 ---
        initial_prompts_sent.clear()  # Limpar para capturar apenas seg2

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=3000,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=5000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected, initial_prompts_sent, context_after_seg1

    events, prompts, context_after_seg1 = asyncio.run(_run())

    check(
        context_after_seg1 is not None,
        f"Contexto armazenado apos seg1: '{context_after_seg1}'",
    )

    # No segundo segmento, o initial_prompt deve conter o texto do seg1
    has_context_in_prompt = any(
        p is not None and "transferência" in p.lower()
        for p in prompts
        if p is not None
    )
    check(
        has_context_in_prompt,
        "initial_prompt do seg2 contem contexto do seg1",
    )

    finals = [e for e in events if e.get("type") == "transcript.final"]
    check(len(finals) == 2, f"2 transcript.final recebidos ({len(finals)})")

    info("  O usuario nao ve o cross-segment context diretamente,")
    info("  mas a qualidade da transcricao melhora porque o worker recebe")
    info("  o texto do segmento anterior como contexto.")


def demo_9_wal_checkpoints() -> None:
    """Step 9: WAL Checkpoints — registro de posicao para recovery."""
    step(9, "WAL Checkpoints (registro apos cada transcript.final)")

    async def _run() -> tuple[SessionWAL, int]:
        final_seg = TranscriptSegment(
            text="checkpoint registrado",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=1000, language="pt", confidence=0.94,
        )
        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()
        rb = RingBuffer(duration_s=10.0)

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_wal",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            wal=wal,
            ring_buffer=rb,
        )

        wal_before = wal.last_committed_segment_id

        # speech_start -> ACTIVE -> enviar frames -> speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = None
        for _ in range(5):
            await session.process_frame(make_pcm_tone())

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return wal, wal_before

    wal, wal_before = asyncio.run(_run())

    check(wal_before == 0, f"WAL antes: segment_id={wal_before}")
    check(
        wal.last_committed_segment_id == 0,
        f"WAL apos transcript.final: segment_id={wal.last_committed_segment_id}",
    )
    check(
        wal.last_committed_buffer_offset > 0,
        f"WAL buffer_offset={wal.last_committed_buffer_offset}",
    )
    check(
        wal.last_committed_timestamp_ms > 0,
        f"WAL timestamp_ms={wal.last_committed_timestamp_ms}",
    )

    info("  WAL registra checkpoint apos cada transcript.final:")
    info(f"    segment_id={wal.last_committed_segment_id}")
    info(f"    buffer_offset={wal.last_committed_buffer_offset}")
    info(f"    timestamp_ms={wal.last_committed_timestamp_ms}")
    info("  Em caso de crash, o recovery usa estes valores para retomar")
    info("  sem duplicar segmentos.")


def demo_10_prometheus_metrics() -> None:
    """Step 10: Metricas Prometheus M6."""
    step(10, "Metricas Prometheus (M6)")

    from theo.session import metrics

    info("  Verificando que as metricas M6 estao definidas...")

    # Verificar que metricas existem (podem ser None se prometheus nao instalado)
    metric_names = [
        ("stt_session_duration_seconds", metrics.stt_session_duration_seconds),
        ("stt_segments_force_committed_total", metrics.stt_segments_force_committed_total),
        ("stt_confidence_avg", metrics.stt_confidence_avg),
        ("stt_worker_recoveries_total", metrics.stt_worker_recoveries_total),
        ("stt_active_sessions", metrics.stt_active_sessions),
        ("stt_ttfb_seconds", metrics.stt_ttfb_seconds),
        ("stt_final_delay_seconds", metrics.stt_final_delay_seconds),
        ("stt_vad_events_total", metrics.stt_vad_events_total),
    ]

    if metrics.HAS_METRICS:
        for name, metric in metric_names:
            check(metric is not None, f"Metrica {name} registrada")
    else:
        info("  prometheus_client nao instalado - metricas desativadas (ok para demo)")
        check(True, "Metricas falham graciosamente sem prometheus_client")


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    """Executa todas as demos."""
    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  Demo M6 -- Session Manager (End-to-End){NC}")
    print(f"{BOLD}{'=' * 60}{NC}")
    print()
    info("Componentes demonstrados:")
    info("  - Maquina de Estados (6 estados: INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED)")
    info("  - Ring Buffer (read fence, force commit automatico)")
    info("  - WAL In-Memory (checkpoint apos transcript.final)")
    info("  - Crash Recovery (retomada sem duplicacao de segmentos)")
    info("  - Hot Words per Session (session.configure)")
    info("  - Cross-Segment Context (initial_prompt com texto anterior)")
    info("  - Metricas Prometheus (session_duration, force_committed, confidence, recoveries)")
    print()
    info("Todos os cenarios simulam a experiencia do usuario via protocolo WebSocket.")
    info("Nenhum modelo real e necessario -- mocks controlados simulam o worker.")

    demos = [
        demo_1_ws_handshake,
        demo_2_state_machine_transitions,
        demo_3_hot_words_per_session,
        demo_4_session_hold,
        demo_5_ring_buffer_force_commit,
        demo_6_manual_commit,
        demo_7_crash_recovery,
        demo_8_cross_segment_context,
        demo_9_wal_checkpoints,
        demo_10_prometheus_metrics,
    ]

    for demo_fn in demos:
        try:
            demo_fn()
        except Exception as exc:
            fail_msg(f"EXCEPTION em {demo_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            global _total_fail
            _total_fail += 1

    # --- Summary ---
    print(f"\n{BOLD}{'=' * 60}{NC}")
    total = _total_pass + _total_fail
    print(f"{BOLD}  Resultado: {_total_pass}/{total} checks passaram{NC}")
    if _total_fail > 0:
        print(f"{RED}  {_total_fail} checks falharam{NC}")
    else:
        print(f"{GREEN}  Todos os checks passaram!{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")

    return 0 if _total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
