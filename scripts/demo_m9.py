"""Demo M9 -- Full-Duplex STT + TTS — mute-on-speak, TTS backend, REST speech, WS protocol.

Exercita TODOS os componentes do M9 do ponto de vista do usuario:

1.  TTSBackend Factory: _create_backend("kokoro") retorna KokoroBackend
2.  KokoroBackend: load, synthesize, voices, health, unload
3.  TTS gRPC Converters: proto <-> dominio (SynthesizeParams, chunks)
4.  TTS Proto Request: build_tts_proto_request e tts_proto_chunks_to_result
5.  MuteController: mute/unmute/is_muted, idempotencia
6.  StreamingSession mute-on-speak: process_frame descarta frames quando muted
7.  TTS WebSocket Events: TTSSpeakCommand, TTSCancelCommand, TTSSpeakingStart/End
8.  TTS REST Models: SpeechRequest validation, PCM-to-WAV conversion
9.  TTS Metrics: 4 TTS metrics + 1 STT muted_frames counter
10. POST /v1/audio/speech: endpoint end-to-end com worker gRPC leve

Funciona SEM modelo real instalado — usa classes leves no lugar de mocks.

Uso:
    .venv/bin/python scripts/demo_m9.py
"""

from __future__ import annotations

import asyncio
import struct
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from theo._types import (
    ModelType,
    STTArchitecture,
    TTSSpeechResult,
    VoiceInfo,
)
from theo.config.manifest import ModelManifest
from theo.exceptions import ModelLoadError, TTSSynthesisError
from theo.proto.tts_worker_pb2 import (
    HealthResponse,
    SynthesizeChunk,
    SynthesizeRequest,
)
from theo.scheduler.tts_converters import build_tts_proto_request, tts_proto_chunks_to_result
from theo.server.models.events import (
    TTSCancelCommand,
    TTSSpeakCommand,
    TTSSpeakingEndEvent,
    TTSSpeakingStartEvent,
)
from theo.server.models.speech import SpeechRequest
from theo.session.mute import MuteController
from theo.workers.tts.converters import (
    SynthesizeParams,
    audio_chunk_to_proto,
    health_dict_to_proto_response,
    proto_request_to_synthesize_params,
)
from theo.workers.tts.kokoro import (
    KokoroBackend,
    _extract_audio_array,
    _float32_to_pcm16_bytes,
    _resolve_device,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24000

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


def check(condition: bool, desc: str) -> bool:
    if condition:
        pass_msg(desc)
    else:
        fail_msg(desc)
    return condition


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

passes = 0
fails = 0


def record(ok: bool) -> None:
    global passes, fails
    if ok:
        passes += 1
    else:
        fails += 1


# ---------------------------------------------------------------------------
# Lightweight classes (zero unittest.mock)
# ---------------------------------------------------------------------------


class _FakeKokoroLib:
    """Substitui kokoro library para teste de load — sem MagicMock."""

    class _FakeModel:
        pass

    @staticmethod
    def load_model(*_args: object, **_kwargs: object) -> _FakeKokoroLib._FakeModel:
        return _FakeKokoroLib._FakeModel()


class _LightPreprocessor:
    """Preprocessor leve que retorna audio sem transformar."""

    def __init__(self) -> None:
        self.call_count = 0

    def process_frame(self, raw_bytes: bytes) -> bytes:
        self.call_count += 1
        return raw_bytes


class _LightVAD:
    """VAD leve que nunca detecta fala (retorna None)."""

    def process_frame(self, _data: bytes) -> None:
        return None


class _LightStreamHandle:
    """StreamHandle leve para StreamingSession."""

    async def send_frame(self, _frame: bytes) -> None:
        pass

    async def receive_events(self) -> list[object]:
        return []

    async def close(self) -> None:
        pass

    async def cancel(self) -> None:
        pass

    def __aiter__(self) -> _LightStreamHandle:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


class _LightGRPCClient:
    """gRPC client leve que retorna _LightStreamHandle."""

    async def open_stream(self, **_kwargs: object) -> _LightStreamHandle:
        return _LightStreamHandle()

    async def close(self) -> None:
        pass


class _LightWorkerManager:
    """WorkerManager leve que retorna um WorkerHandle fixo."""

    def __init__(self, worker: object) -> None:
        self._worker = worker

    def get_ready_worker(self, _model_name: str) -> object:
        return self._worker


class _LightScheduler:
    """Scheduler leve — nao usado pela rota /v1/audio/speech."""

    pass


class _FakeGRPCChannel:
    """Canal gRPC leve com close async."""

    async def close(self) -> None:
        pass


class _FakeGRPCStub:
    """Stub TTSWorker leve que retorna chunks de audio."""

    def __init__(self, audio_pcm: bytes) -> None:
        self._audio_pcm = audio_pcm

    def Synthesize(self, _request: object, **_kwargs: object) -> _FakeAsyncStream:
        return _FakeAsyncStream(self._audio_pcm)


class _FakeAsyncStream:
    """Async iterator de SynthesizeChunk."""

    def __init__(self, audio_pcm: bytes) -> None:
        self._chunks = [
            SynthesizeChunk(audio_data=audio_pcm[:4096], is_last=False, duration=0.085),
            SynthesizeChunk(audio_data=audio_pcm[4096:], is_last=True, duration=0.5),
        ]
        self._index = 0

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> SynthesizeChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _ObjectWithAudio:
    """Objeto simples com atributo .audio para _extract_audio_array."""

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm_audio(duration_s: float = 0.1, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Gera audio PCM 16-bit mono (sine 440Hz)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32000).astype(np.int16)
    return samples.tobytes()


def _make_float32_audio(duration_s: float = 0.1, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Gera audio float32 normalizado [-1, 1] (sine 440Hz)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------


# 1. TTSBackend Factory
def demo_tts_factory() -> None:
    step(1, "TTSBackend Factory — _create_backend('kokoro')")
    info("Testing TTS factory creates KokoroBackend...")

    from theo.workers.tts.main import _create_backend

    backend = _create_backend("kokoro")
    record(check(isinstance(backend, KokoroBackend), "Factory returns KokoroBackend"))

    info("Testing factory rejects unknown engine...")
    try:
        _create_backend("nonexistent")
        record(check(False, "Should have raised ValueError"))
    except ValueError as e:
        record(check("nao suportada" in str(e), f"ValueError raised: {e}"))


# 2. KokoroBackend: load, synthesize, voices, health, unload
async def demo_kokoro_backend() -> None:
    step(2, "KokoroBackend — load, synthesize, voices, health, unload")

    backend = KokoroBackend()

    # Health when not loaded
    health = await backend.health()
    record(check(health["status"] == "not_loaded", f"Health before load: {health['status']}"))

    # Load raises when kokoro not installed
    info("Testing load raises ModelLoadError when kokoro not installed...")
    import theo.workers.tts.kokoro as kokoro_mod

    original = kokoro_mod.kokoro_lib
    kokoro_mod.kokoro_lib = None  # type: ignore[assignment]
    try:
        try:
            await backend.load("/fake/path", {"device": "cpu"})
            record(check(False, "Should have raised ModelLoadError"))
        except ModelLoadError as e:
            record(check("kokoro" in str(e).lower(), f"ModelLoadError raised: {e}"))
    finally:
        kokoro_mod.kokoro_lib = original

    # Load succeeds with lightweight fake lib
    info("Testing load succeeds with lightweight fake kokoro lib...")
    kokoro_mod.kokoro_lib = _FakeKokoroLib  # type: ignore[assignment]
    try:
        await backend.load("/model/path", {"device": "cpu"})
        health = await backend.health()
        record(check(health["status"] == "ok", f"Health after load: {health['status']}"))
    finally:
        kokoro_mod.kokoro_lib = original

    # Voices
    info("Testing voices()...")
    voices = await backend.voices()
    record(check(len(voices) >= 2, f"voices() returned {len(voices)} voices"))
    record(check(isinstance(voices[0], VoiceInfo), "First voice is VoiceInfo"))

    # Synthesize: empty text raises
    info("Testing synthesize with empty text raises TTSSynthesisError...")
    try:
        async for _chunk in backend.synthesize(""):
            pass
        record(check(False, "Should have raised TTSSynthesisError"))
    except TTSSynthesisError:
        record(check(True, "TTSSynthesisError raised for empty text"))

    # Unload
    info("Testing unload...")
    await backend.unload()
    health = await backend.health()
    record(check(health["status"] == "not_loaded", f"Health after unload: {health['status']}"))


# 3. TTS gRPC Converters
def demo_tts_converters() -> None:
    step(3, "TTS gRPC Converters — proto <-> dominio")

    # proto_request_to_synthesize_params
    info("Testing proto_request_to_synthesize_params...")
    proto_req = SynthesizeRequest(
        request_id="req_1",
        text="Ola mundo",
        voice="default",
        sample_rate=24000,
        speed=1.5,
    )
    params = proto_request_to_synthesize_params(proto_req)
    record(check(isinstance(params, SynthesizeParams), "Returns SynthesizeParams"))
    record(check(params.text == "Ola mundo", f"text: {params.text}"))
    record(check(params.voice == "default", f"voice: {params.voice}"))
    record(check(params.sample_rate == 24000, f"sample_rate: {params.sample_rate}"))
    record(check(params.speed == 1.5, f"speed: {params.speed}"))

    # Defaults for empty/zero values
    info("Testing defaults for empty/zero protobuf values...")
    proto_empty = SynthesizeRequest(request_id="req_2", text="test")
    params_empty = proto_request_to_synthesize_params(proto_empty)
    record(check(params_empty.voice == "default", f"Default voice: {params_empty.voice}"))
    record(check(params_empty.sample_rate == 24000, f"Default sample_rate: {params_empty.sample_rate}"))
    record(check(params_empty.speed == 1.0, f"Default speed: {params_empty.speed}"))

    # audio_chunk_to_proto
    info("Testing audio_chunk_to_proto...")
    audio_bytes = b"\x00" * 4096
    chunk = audio_chunk_to_proto(audio_data=audio_bytes, is_last=False, duration=0.085)
    record(check(isinstance(chunk, SynthesizeChunk), "Returns SynthesizeChunk"))
    record(check(len(chunk.audio_data) == 4096, f"Chunk audio size: {len(chunk.audio_data)}"))
    record(check(chunk.is_last is False, "is_last=False"))

    # health_dict_to_proto_response
    info("Testing health_dict_to_proto_response...")
    health_proto = health_dict_to_proto_response(
        {"status": "ok"}, "kokoro-v1", "kokoro"
    )
    record(check(isinstance(health_proto, HealthResponse), "Returns HealthResponse"))
    record(check(health_proto.status == "ok", f"status: {health_proto.status}"))
    record(check(health_proto.model_name == "kokoro-v1", f"model: {health_proto.model_name}"))


# 4. TTS Proto Request builders (REST converters)
def demo_tts_rest_converters() -> None:
    step(4, "TTS REST Converters — build_tts_proto_request, tts_proto_chunks_to_result")

    # build_tts_proto_request
    info("Testing build_tts_proto_request...")
    proto_req = build_tts_proto_request(
        request_id="req_speech_1",
        text="Bom dia",
        voice="pt_female",
        sample_rate=24000,
        speed=1.0,
    )
    record(check(isinstance(proto_req, SynthesizeRequest), "Returns SynthesizeRequest"))
    record(check(proto_req.request_id == "req_speech_1", f"request_id: {proto_req.request_id}"))
    record(check(proto_req.text == "Bom dia", f"text: {proto_req.text}"))
    record(check(proto_req.voice == "pt_female", f"voice: {proto_req.voice}"))

    # tts_proto_chunks_to_result
    info("Testing tts_proto_chunks_to_result...")
    chunks = [b"\x00" * 4096, b"\x00" * 2048]
    result = tts_proto_chunks_to_result(
        chunks,
        sample_rate=24000,
        voice="default",
        total_duration=0.256,
    )
    record(check(isinstance(result, TTSSpeechResult), "Returns TTSSpeechResult"))
    record(check(len(result.audio_data) == 6144, f"Total audio bytes: {len(result.audio_data)}"))
    record(check(result.sample_rate == 24000, f"sample_rate: {result.sample_rate}"))
    record(check(result.duration == 0.256, f"duration: {result.duration}"))
    record(check(result.voice == "default", f"voice: {result.voice}"))


# 5. MuteController
def demo_mute_controller() -> None:
    step(5, "MuteController — mute/unmute/is_muted, idempotencia")

    mc = MuteController(session_id="sess_001")

    # Initial state
    record(check(mc.is_muted is False, "Initial state: not muted"))

    # Mute
    mc.mute()
    record(check(mc.is_muted is True, "After mute(): is_muted=True"))

    # Idempotent mute
    mc.mute()
    record(check(mc.is_muted is True, "Idempotent mute(): still muted"))

    # Unmute
    mc.unmute()
    record(check(mc.is_muted is False, "After unmute(): is_muted=False"))

    # Idempotent unmute
    mc.unmute()
    record(check(mc.is_muted is False, "Idempotent unmute(): still not muted"))

    # try/finally pattern
    info("Testing try/finally pattern (mute survives exception)...")
    mc2 = MuteController(session_id="sess_002")
    try:
        mc2.mute()
        raise RuntimeError("Simulated TTS crash")
    except RuntimeError:
        pass
    finally:
        mc2.unmute()
    record(check(mc2.is_muted is False, "Unmute in finally: not muted after crash"))


# 6. StreamingSession mute-on-speak
async def demo_streaming_session_mute() -> None:
    step(6, "StreamingSession mute-on-speak — process_frame discards when muted")
    info("Creating StreamingSession with lightweight dependencies...")

    from theo.session.streaming import StreamingSession

    events: list[Any] = []

    async def on_event(event: object) -> None:
        events.append(event)

    preprocessor = _LightPreprocessor()
    vad = _LightVAD()
    grpc_client = _LightGRPCClient()

    session = StreamingSession(
        session_id="sess_mute_test",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=None,
        on_event=on_event,
        architecture=STTArchitecture.ENCODER_DECODER,
    )

    # Process frame normally
    frame = _make_pcm_audio(0.03, 16000)
    await session.process_frame(frame)
    record(check(
        preprocessor.call_count >= 1,
        f"Normal: preprocessor called {preprocessor.call_count} time(s)",
    ))

    # Mute and process frame — should be discarded
    prev_count = preprocessor.call_count
    session.mute()
    record(check(session.is_muted is True, "session.is_muted == True after mute()"))

    await session.process_frame(frame)
    record(check(
        preprocessor.call_count == prev_count,
        "Muted: preprocessor NOT called (frame discarded)",
    ))

    # Unmute and verify processing resumes
    session.unmute()
    record(check(session.is_muted is False, "session.is_muted == False after unmute()"))

    prev_count = preprocessor.call_count
    await session.process_frame(frame)
    record(check(
        preprocessor.call_count > prev_count,
        "Unmuted: preprocessor called again (processing resumed)",
    ))


# 7. TTS WebSocket Events (Pydantic models)
def demo_tts_ws_events() -> None:
    step(7, "TTS WebSocket Events — TTSSpeakCommand, TTSCancelCommand, TTSSpeakingStart/End")

    # TTSSpeakCommand
    info("Testing TTSSpeakCommand...")
    speak_cmd = TTSSpeakCommand(
        text="Ola, como posso ajudar?",
        voice="pt_female",
        request_id="req_tts_1",
    )
    record(check(speak_cmd.type == "tts.speak", f"type: {speak_cmd.type}"))
    record(check(speak_cmd.text == "Ola, como posso ajudar?", "text preserved"))
    record(check(speak_cmd.voice == "pt_female", f"voice: {speak_cmd.voice}"))

    # model_dump for serialization
    dumped = speak_cmd.model_dump()
    record(check(dumped["type"] == "tts.speak", "model_dump preserves type"))

    # TTSSpeakCommand from JSON (parse)
    speak_parsed = TTSSpeakCommand.model_validate(
        {"type": "tts.speak", "text": "Teste"}
    )
    record(check(speak_parsed.voice == "default", f"Default voice: {speak_parsed.voice}"))

    # TTSCancelCommand
    info("Testing TTSCancelCommand...")
    cancel_cmd = TTSCancelCommand(request_id="req_tts_1")
    record(check(cancel_cmd.type == "tts.cancel", f"type: {cancel_cmd.type}"))
    record(check(cancel_cmd.request_id == "req_tts_1", "request_id preserved"))

    # TTSSpeakingStartEvent
    info("Testing TTSSpeakingStartEvent...")
    start_evt = TTSSpeakingStartEvent(
        request_id="req_tts_1",
        timestamp_ms=1500,
    )
    record(check(start_evt.type == "tts.speaking_start", f"type: {start_evt.type}"))
    dumped_start = start_evt.model_dump()
    record(check(dumped_start["timestamp_ms"] == 1500, "timestamp_ms in model_dump"))

    # TTSSpeakingEndEvent
    info("Testing TTSSpeakingEndEvent...")
    end_evt = TTSSpeakingEndEvent(
        request_id="req_tts_1",
        timestamp_ms=3500,
        duration_ms=2000,
        cancelled=False,
    )
    record(check(end_evt.type == "tts.speaking_end", f"type: {end_evt.type}"))
    record(check(end_evt.duration_ms == 2000, f"duration_ms: {end_evt.duration_ms}"))
    record(check(end_evt.cancelled is False, "cancelled=False"))

    # Cancelled variant
    end_cancelled = TTSSpeakingEndEvent(
        request_id="req_tts_2",
        timestamp_ms=2000,
        duration_ms=500,
        cancelled=True,
    )
    record(check(end_cancelled.cancelled is True, "cancelled=True (TTS interrupted)"))


# 8. TTS REST Models + PCM-to-WAV
def demo_tts_rest_models() -> None:
    step(8, "TTS REST Models — SpeechRequest validation, PCM-to-WAV conversion")

    # SpeechRequest validation
    info("Testing SpeechRequest with defaults...")
    req = SpeechRequest(model="kokoro-v1", input="Ola mundo")
    record(check(req.model == "kokoro-v1", f"model: {req.model}"))
    record(check(req.voice == "default", f"Default voice: {req.voice}"))
    record(check(req.response_format == "wav", f"Default format: {req.response_format}"))
    record(check(req.speed == 1.0, f"Default speed: {req.speed}"))

    # SpeechRequest with custom values
    info("Testing SpeechRequest with custom values...")
    req2 = SpeechRequest(
        model="kokoro-v1",
        input="Test",
        voice="pt_female",
        response_format="pcm",
        speed=1.5,
    )
    record(check(req2.response_format == "pcm", f"format: {req2.response_format}"))
    record(check(req2.speed == 1.5, f"speed: {req2.speed}"))

    # Speed validation (0.25 - 4.0)
    info("Testing speed range validation...")
    try:
        SpeechRequest(model="m", input="t", speed=0.1)
        record(check(False, "Should reject speed < 0.25"))
    except Exception:
        record(check(True, "Rejects speed < 0.25"))

    try:
        SpeechRequest(model="m", input="t", speed=5.0)
        record(check(False, "Should reject speed > 4.0"))
    except Exception:
        record(check(True, "Rejects speed > 4.0"))

    # PCM-to-WAV conversion
    info("Testing _pcm_to_wav conversion...")
    from theo.server.routes.speech import _pcm_to_wav

    pcm_data = _make_pcm_audio(0.1, _SAMPLE_RATE)
    wav_data = _pcm_to_wav(pcm_data, _SAMPLE_RATE)

    # Check WAV header
    record(check(wav_data[:4] == b"RIFF", "WAV starts with RIFF"))
    record(check(wav_data[8:12] == b"WAVE", "WAV has WAVE marker"))
    record(check(wav_data[12:16] == b"fmt ", "WAV has fmt subchunk"))

    # Check sample rate in WAV header (bytes 24-28)
    wav_sr = struct.unpack("<I", wav_data[24:28])[0]
    record(check(wav_sr == _SAMPLE_RATE, f"WAV sample rate: {wav_sr}"))

    # Data subchunk size matches PCM data
    data_size = struct.unpack("<I", wav_data[40:44])[0]
    record(check(data_size == len(pcm_data), f"WAV data size matches PCM: {data_size}"))


# 9. TTS Metrics
def demo_tts_metrics() -> None:
    step(9, "TTS Metrics — 4 TTS metrics + 1 STT muted_frames counter")

    info("Checking TTS metrics module...")
    from theo.scheduler import tts_metrics

    record(check(
        hasattr(tts_metrics, "HAS_TTS_METRICS"),
        "tts_metrics has HAS_TTS_METRICS flag",
    ))
    record(check(
        hasattr(tts_metrics, "tts_ttfb_seconds"),
        "tts_ttfb_seconds defined",
    ))
    record(check(
        hasattr(tts_metrics, "tts_synthesis_duration_seconds"),
        "tts_synthesis_duration_seconds defined",
    ))
    record(check(
        hasattr(tts_metrics, "tts_requests_total"),
        "tts_requests_total defined",
    ))
    record(check(
        hasattr(tts_metrics, "tts_active_sessions"),
        "tts_active_sessions defined",
    ))

    info("Checking STT muted_frames counter in session.metrics...")
    from theo.session import metrics as stt_metrics

    record(check(
        hasattr(stt_metrics, "stt_muted_frames_total"),
        "stt_muted_frames_total defined",
    ))

    # Verify lazy import pattern (metrics are None or metric objects)
    info("Checking lazy import pattern...")
    if tts_metrics.HAS_TTS_METRICS:
        record(check(
            tts_metrics.tts_ttfb_seconds is not None,
            "With prometheus_client: tts_ttfb_seconds is not None",
        ))
    else:
        record(check(
            tts_metrics.tts_ttfb_seconds is None,
            "Without prometheus_client: tts_ttfb_seconds is None",
        ))

    # Verify all 4 TTS metric names match spec
    expected_names = {
        "theo_tts_ttfb_seconds",
        "theo_tts_synthesis_duration_seconds",
        "theo_tts_requests",  # Counter._name strips _total suffix
        "theo_tts_active_sessions",
    }
    if tts_metrics.HAS_TTS_METRICS:
        actual_names = set()
        for metric, name in [
            (tts_metrics.tts_ttfb_seconds, "theo_tts_ttfb_seconds"),
            (tts_metrics.tts_synthesis_duration_seconds, "theo_tts_synthesis_duration_seconds"),
            (tts_metrics.tts_requests_total, "theo_tts_requests_total"),
            (tts_metrics.tts_active_sessions, "theo_tts_active_sessions"),
        ]:
            if metric is not None and hasattr(metric, "_name"):
                actual_names.add(metric._name)
        record(check(
            actual_names == expected_names,
            f"All 4 metric names match spec: {actual_names}",
        ))
    else:
        info("  prometheus_client not installed — skipping metric name check")
        record(check(True, "Metric module loaded without prometheus_client (graceful)"))


# 10. POST /v1/audio/speech endpoint e2e
async def demo_speech_endpoint() -> None:
    step(10, "POST /v1/audio/speech — endpoint end-to-end com worker gRPC leve")
    info("Creating FastAPI app with lightweight TTS worker...")

    from httpx import ASGITransport, AsyncClient

    from theo.config.manifest import ModelCapabilities, ModelManifest, ModelResources
    from theo.registry.registry import ModelRegistry
    from theo.server.app import create_app
    from theo.workers.manager import WorkerHandle, WorkerState

    # Create a TTS manifest
    tts_manifest = ModelManifest(
        name="kokoro-v1",
        version="1.0.0",
        engine="kokoro",
        model_type=ModelType.TTS,
        description="Test TTS model",
        capabilities=ModelCapabilities(streaming=True, languages=["en", "pt"]),
        resources=ModelResources(memory_mb=512, gpu_required=False),
        engine_config={"device": "auto"},
    )

    # Create registry with TTS model
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._manifests = {"kokoro-v1": tts_manifest}

    # Create lightweight scheduler and worker manager
    scheduler = _LightScheduler()

    tts_worker = WorkerHandle(
        worker_id="tts-worker-0",
        port=50052,
        model_name="kokoro-v1",
        engine="kokoro",
        state=WorkerState.READY,
    )
    worker_manager = _LightWorkerManager(tts_worker)

    # Create app with all dependencies
    app = create_app(
        registry=registry,
        scheduler=scheduler,  # type: ignore[arg-type]
        worker_manager=worker_manager,  # type: ignore[arg-type]
    )

    # Monkey-patch the speech route's gRPC call with lightweight classes
    audio_pcm = _make_pcm_audio(0.5, _SAMPLE_RATE)
    fake_stub = _FakeGRPCStub(audio_pcm)
    fake_channel = _FakeGRPCChannel()

    import theo.server.routes.speech as speech_mod

    original_insecure_channel = speech_mod.grpc.aio.insecure_channel
    original_stub_class = speech_mod.TTSWorkerStub

    speech_mod.grpc.aio.insecure_channel = lambda *a, **kw: fake_channel  # type: ignore[assignment]
    speech_mod.TTSWorkerStub = lambda *a, **kw: fake_stub  # type: ignore[assignment]

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Test WAV response (default)
            info("Testing POST /v1/audio/speech (WAV response)...")
            response = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "Ola, como posso ajudar?",
                    "voice": "default",
                },
            )
            record(check(response.status_code == 200, f"Status: {response.status_code}"))
            record(check(
                response.headers["content-type"] == "audio/wav",
                f"Content-Type: {response.headers['content-type']}",
            ))
            record(check(
                response.content[:4] == b"RIFF",
                "Response starts with RIFF (valid WAV)",
            ))

            # Test PCM response
            info("Testing POST /v1/audio/speech (PCM response)...")
            response_pcm = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "Teste PCM",
                    "voice": "default",
                    "response_format": "pcm",
                },
            )
            record(check(response_pcm.status_code == 200, f"PCM status: {response_pcm.status_code}"))
            record(check(
                response_pcm.headers["content-type"] == "audio/pcm",
                f"PCM Content-Type: {response_pcm.headers['content-type']}",
            ))

            # Test empty input (400)
            info("Testing POST /v1/audio/speech (empty input -> 400)...")
            response_empty = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-v1",
                    "input": "   ",
                    "voice": "default",
                },
            )
            record(check(
                response_empty.status_code == 400,
                f"Empty input status: {response_empty.status_code}",
            ))

            # Test invalid model (404)
            info("Testing POST /v1/audio/speech (unknown model -> 404)...")
            response_404 = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "nonexistent-model",
                    "input": "test",
                },
            )
            record(check(
                response_404.status_code == 404,
                f"Unknown model status: {response_404.status_code}",
            ))
    finally:
        speech_mod.grpc.aio.insecure_channel = original_insecure_channel  # type: ignore[assignment]
        speech_mod.TTSWorkerStub = original_stub_class  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bonus: Kokoro helper functions (pure functions, no engine needed)
# ---------------------------------------------------------------------------


def demo_bonus_kokoro_helpers() -> None:
    step("B", "Bonus: Kokoro helper functions — _resolve_device, _extract_audio, _float32_to_pcm16")

    # _resolve_device
    info("Testing _resolve_device...")
    record(check(_resolve_device("auto") == "cpu", "auto -> cpu"))
    record(check(_resolve_device("cpu") == "cpu", "cpu -> cpu"))
    record(check(_resolve_device("cuda") == "cuda", "cuda -> cuda"))
    record(check(_resolve_device("cuda:0") == "cuda:0", "cuda:0 -> cuda:0"))

    # _extract_audio_array
    info("Testing _extract_audio_array...")
    arr = _make_float32_audio(0.01)
    record(check(_extract_audio_array(arr) is not None, "Extracts from np.ndarray"))
    record(check(_extract_audio_array({"audio": arr}) is not None, "Extracts from dict"))
    record(check(_extract_audio_array("invalid") is None, "Returns None for invalid"))
    record(check(_extract_audio_array({}) is None, "Returns None for empty dict"))

    obj_with_audio = _ObjectWithAudio(arr)
    record(check(_extract_audio_array(obj_with_audio) is not None, "Extracts from object.audio"))

    # _float32_to_pcm16_bytes
    info("Testing _float32_to_pcm16_bytes...")
    float_audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm_bytes = _float32_to_pcm16_bytes(float_audio)
    record(check(len(pcm_bytes) == 10, f"5 samples -> 10 bytes: {len(pcm_bytes)}"))

    # Verify center sample is 0
    sample_0 = struct.unpack("<h", pcm_bytes[0:2])[0]
    record(check(sample_0 == 0, f"Sample[0]=0.0 -> PCM {sample_0}"))

    # Verify clipping at boundaries
    sample_pos = struct.unpack("<h", pcm_bytes[6:8])[0]
    sample_neg = struct.unpack("<h", pcm_bytes[8:10])[0]
    record(check(sample_pos == 32767, f"Sample[3]=1.0 clipped to {sample_pos}"))
    record(check(sample_neg == -32768, f"Sample[4]=-1.0 clipped to {sample_neg}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> int:
    global passes, fails

    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  Demo M9 -- Full-Duplex STT + TTS{NC}")
    print(f"{BOLD}  TTSBackend | Mute-on-Speak | REST Speech | WS Events | Metrics{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")

    demo_tts_factory()
    await demo_kokoro_backend()
    demo_tts_converters()
    demo_tts_rest_converters()
    demo_mute_controller()
    await demo_streaming_session_mute()
    demo_tts_ws_events()
    demo_tts_rest_models()
    demo_tts_metrics()
    await demo_speech_endpoint()
    demo_bonus_kokoro_helpers()

    # Summary
    total = passes + fails
    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  Summary: {total} checks{NC}")
    print(f"  {GREEN}PASSED: {passes}{NC}")
    if fails > 0:
        print(f"  {RED}FAILED: {fails}{NC}")
    else:
        print(f"  {GREEN}FAILED: 0{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")

    return 0 if fails == 0 else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
