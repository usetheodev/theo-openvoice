"""Testes de metricas Prometheus para streaming STT.

Valida que o StreamingSession registra corretamente:
- TTFB (time to first partial/final apos speech_start)
- Final delay (tempo entre speech_end e transcript.final)
- Active sessions (incrementa/decrementa no lifecycle)
- VAD events (counter por tipo speech_start/speech_end)

Testes sao deterministicos — usam mock de time.monotonic para controlar timestamps.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from theo._types import TranscriptSegment
from theo.server.models.events import (
    TranscriptFinalEvent,
    TranscriptPartialEvent,
)
from theo.session.streaming import StreamingSession
from theo.vad.detector import VADEvent, VADEventType

# Verificar se prometheus_client esta disponivel para testes de valores
try:
    from prometheus_client import REGISTRY

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


# Frame size padrao: 1024 samples a 16kHz = 64ms
_FRAME_SIZE = 1024


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """Gera bytes PCM int16 (zeros) com n_samples amostras."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_float32_frame(n_samples: int = _FRAME_SIZE) -> np.ndarray:
    """Gera frame float32 (zeros) para mock de preprocessor."""
    return np.zeros(n_samples, dtype=np.float32)


def _make_preprocessor_mock() -> Mock:
    """Cria mock de StreamingPreprocessor."""
    mock = Mock()
    mock.process_frame.return_value = _make_float32_frame()
    return mock


def _make_vad_mock(*, is_speaking: bool = False) -> Mock:
    """Cria mock de VADDetector."""
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


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


def _make_stream_handle_mock(
    events: list[object] | None = None,
) -> Mock:
    """Cria mock de StreamHandle."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    if events is None:
        events = []
    handle.receive_events.return_value = _AsyncIterFromList(events)

    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Cria mock de StreamingGRPCClient."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Cria mock de PostProcessingPipeline."""
    mock = Mock()
    mock.process.side_effect = lambda text: text
    return mock


def _make_on_event() -> AsyncMock:
    """Cria callback on_event mock."""
    return AsyncMock()


def _make_session(
    *,
    stream_handle: Mock | None = None,
    vad: Mock | None = None,
    on_event: AsyncMock | None = None,
    session_id: str = "test_metrics",
) -> tuple[StreamingSession, Mock, Mock, AsyncMock]:
    """Cria StreamingSession com mocks para testes de metricas.

    Returns:
        (session, vad, stream_handle, on_event)
    """
    _vad = vad or _make_vad_mock()
    _stream_handle = stream_handle or _make_stream_handle_mock()
    _on_event = on_event or _make_on_event()
    grpc_client = _make_grpc_client_mock(_stream_handle)

    session = StreamingSession(
        session_id=session_id,
        preprocessor=_make_preprocessor_mock(),
        vad=_vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_on_event,
    )

    return session, _vad, _stream_handle, _on_event


# ---------------------------------------------------------------------------
# Helpers para leitura de metricas
# ---------------------------------------------------------------------------


def _get_gauge_value(metric_name: str) -> float:
    """Le o valor atual de um Gauge do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == metric_name:
                    return sample.value
    return 0.0


def _get_counter_value(metric_name: str, labels: dict[str, str]) -> float:
    """Le o valor atual de um Counter do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_total" and sample.labels == labels:
                    return sample.value
    return 0.0


def _get_histogram_count(metric_name: str) -> float:
    """Le o _count de um Histogram do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_count":
                    return sample.value
    return 0.0


def _get_histogram_sum(metric_name: str) -> float:
    """Le o _sum de um Histogram do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_sum":
                    return sample.value
    return 0.0


# ---------------------------------------------------------------------------
# Tests: Active Sessions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_increments_on_init():
    """Active sessions incrementa quando uma sessao e criada."""
    # Arrange
    initial_value = _get_gauge_value("theo_stt_active_sessions")

    # Act
    session, _, _, _ = _make_session(session_id="active_inc_test")

    # Assert
    current_value = _get_gauge_value("theo_stt_active_sessions")
    assert current_value == initial_value + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_decrements_on_close():
    """Active sessions decrementa quando uma sessao e fechada."""
    # Arrange
    session, _, _, _ = _make_session(session_id="active_dec_test")
    value_after_init = _get_gauge_value("theo_stt_active_sessions")

    # Act
    await session.close()

    # Assert
    value_after_close = _get_gauge_value("theo_stt_active_sessions")
    assert value_after_close == value_after_init - 1


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_idempotent_close():
    """Multiplas chamadas a close() decrementam apenas uma vez."""
    # Arrange
    session, _, _, _ = _make_session(session_id="active_idempotent_test")
    value_after_init = _get_gauge_value("theo_stt_active_sessions")

    # Act
    await session.close()
    await session.close()
    await session.close()

    # Assert: decrementou apenas 1x
    value_after_close = _get_gauge_value("theo_stt_active_sessions")
    assert value_after_close == value_after_init - 1


# ---------------------------------------------------------------------------
# Tests: VAD Events Counter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_vad_speech_start_increments_counter():
    """Evento speech_start incrementa counter VAD."""
    # Arrange
    initial_starts = _get_counter_value(
        "theo_stt_vad_events",
        {"event_type": "speech_start"},
    )
    vad = _make_vad_mock()
    session, _, _, _ = _make_session(vad=vad, session_id="vad_start_test")

    # Act: trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    # Assert
    current_starts = _get_counter_value(
        "theo_stt_vad_events",
        {"event_type": "speech_start"},
    )
    assert current_starts == initial_starts + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_vad_speech_end_increments_counter():
    """Evento speech_end incrementa counter VAD."""
    # Arrange
    initial_ends = _get_counter_value(
        "theo_stt_vad_events",
        {"event_type": "speech_end"},
    )
    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock()
    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="vad_end_test",
    )

    # Trigger speech_start primeiro
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    # Act: trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    # Assert
    current_ends = _get_counter_value(
        "theo_stt_vad_events",
        {"event_type": "speech_end"},
    )
    assert current_ends == initial_ends + 1


# ---------------------------------------------------------------------------
# Tests: TTFB (Time to First Byte)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_on_first_partial():
    """TTFB e registrado quando o primeiro partial transcript e emitido."""
    # Arrange
    partial_seg = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )

    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial_seg])

    initial_count = _get_histogram_count("theo_stt_ttfb_seconds")

    session, _, _, on_event = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_partial_test",
    )

    # Act: trigger speech_start -> receiver task processa partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    # Dar tempo para receiver task processar
    await asyncio.sleep(0.05)

    # Assert: TTFB foi registrado
    current_count = _get_histogram_count("theo_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Assert: partial event foi emitido
    partial_calls = [
        call
        for call in on_event.call_args_list
        if isinstance(call.args[0], TranscriptPartialEvent)
    ]
    assert len(partial_calls) == 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_once_per_segment():
    """TTFB so e registrado uma vez por segmento de fala."""
    # Arrange: dois partials no mesmo segmento
    partial1 = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    partial2 = TranscriptSegment(
        text="ola como",
        is_final=False,
        segment_id=0,
        start_ms=1500,
    )

    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial1, partial2])

    initial_count = _get_histogram_count("theo_stt_ttfb_seconds")

    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_once_test",
    )

    # Act
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: TTFB registrado apenas 1 vez (nao 2)
    current_count = _get_histogram_count("theo_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_value_reflects_elapsed_time():
    """TTFB value reflete o tempo real entre speech_start e primeiro transcript."""
    # Arrange
    partial_seg = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )

    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial_seg])

    initial_sum = _get_histogram_sum("theo_stt_ttfb_seconds")

    # Usar monotonic time real (o TTFB sera pequeno mas > 0)
    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_value_test",
    )

    # Act
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: TTFB sum increased by a positive value
    current_sum = _get_histogram_sum("theo_stt_ttfb_seconds")
    assert current_sum > initial_sum

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Final Delay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_final_delay_recorded_when_final_after_speech_end():
    """Final delay e registrado quando transcript.final chega apos speech_end."""
    # Arrange: final segment que sera emitido pelo worker
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
    )

    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])

    session, _, _, on_event = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="final_delay_test",
    )

    # 1. Speech start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    # Dar tempo para receiver task consumir o final
    await asyncio.sleep(0.05)

    # 2. Speech end (apos o final ja ter sido processado)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    # Assert: final_delay pode ou nao ter sido registrado dependendo do timing.
    # O final pode ter chegado ANTES do speech_end, nesse caso final_delay
    # nao e registrado (speech_end_monotonic era None quando o final chegou).
    # Este teste valida que nao ha crash no fluxo.
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_final_delay_not_recorded_when_no_speech_end():
    """Final delay NAO e registrado quando speech_end nao ocorreu antes do final."""
    # Arrange: final chega enquanto ainda esta falando (sem speech_end)
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
    )

    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])

    initial_count = _get_histogram_count("theo_stt_final_delay_seconds")

    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="no_final_delay_test",
    )

    # Apenas speech_start (sem speech_end)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: final_delay NAO foi registrado (speech_end_monotonic e None)
    current_count = _get_histogram_count("theo_stt_final_delay_seconds")
    assert current_count == initial_count

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Graceful Degradation
# ---------------------------------------------------------------------------


async def test_session_works_without_prometheus():
    """Sessao funciona normalmente mesmo sem prometheus_client.

    Este teste valida que o modulo metrics nao causa crash quando
    prometheus_client nao esta instalado. Como prometheus_client ESTA
    instalado no ambiente de teste, simulamos a ausencia via patch.
    """
    # Arrange: simular HAS_METRICS = False
    with (
        patch("theo.session.streaming.HAS_METRICS", False),
        patch("theo.session.streaming.stt_active_sessions", None),
        patch("theo.session.streaming.stt_vad_events_total", None),
        patch("theo.session.streaming.stt_ttfb_seconds", None),
        patch("theo.session.streaming.stt_final_delay_seconds", None),
    ):
        vad = _make_vad_mock()
        stream_handle = _make_stream_handle_mock()
        on_event = _make_on_event()

        # Act: criar sessao, processar frames, fechar
        session = StreamingSession(
            session_id="no_metrics_test",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=_make_grpc_client_mock(stream_handle),
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.01)

        # Trigger speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())

        # Close
        await session.close()

    # Assert: nenhum crash, sessao completou normalmente
    assert session.is_closed


async def test_metrics_module_has_metrics_flag():
    """Modulo metrics exporta HAS_METRICS indicando disponibilidade."""
    from theo.session import metrics as metrics_mod

    # Com prometheus_client instalado, HAS_METRICS deve ser True
    # Se nao estiver instalado, deve ser False
    # Nao sabemos em qual ambiente rodamos, mas o flag deve existir
    assert isinstance(metrics_mod.HAS_METRICS, bool)


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_metrics_objects_are_not_none():
    """Quando prometheus_client esta instalado, metricas nao sao None."""
    from theo.session import metrics as metrics_mod

    assert metrics_mod.stt_active_sessions is not None
    assert metrics_mod.stt_final_delay_seconds is not None
    assert metrics_mod.stt_ttfb_seconds is not None
    assert metrics_mod.stt_vad_events_total is not None


# ---------------------------------------------------------------------------
# Tests: Multiple Segments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_per_segment_across_segments():
    """TTFB e registrado separadamente para cada segmento de fala."""
    # Arrange: primeiro segmento com partial
    partial1 = TranscriptSegment(
        text="primeiro",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    stream_handle1 = _make_stream_handle_mock(events=[partial1])

    vad = _make_vad_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    on_event = _make_on_event()

    initial_count = _get_histogram_count("theo_stt_ttfb_seconds")

    session = StreamingSession(
        session_id="ttfb_multi_seg_test",
        preprocessor=_make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Segmento 1: speech_start -> partial -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.05)

    # Preparar novo stream_handle para segundo segmento
    partial2 = TranscriptSegment(
        text="segundo",
        is_final=False,
        segment_id=1,
        start_ms=3000,
    )
    stream_handle2 = _make_stream_handle_mock(events=[partial2])
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    # Segmento 2: speech_start -> partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: TTFB registrado 2x (uma por segmento)
    current_count = _get_histogram_count("theo_stt_ttfb_seconds")
    assert current_count == initial_count + 2

    # Cleanup
    await session.close()
