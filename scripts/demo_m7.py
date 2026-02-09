"""Demo M7 -- Segundo Backend STT (WeNet CTC) — validacao model-agnostic.

Exercita TODOS os componentes do M7 do ponto de vista do usuario:

1.  Factory: _create_backend() cria WeNetBackend e FasterWhisperBackend
2.  Arquitetura: WeNet e CTC, Whisper e encoder-decoder
3.  Capabilities: WeNet suporta hot words nativos, nao suporta initial_prompt
4.  Manifesto: theo.yaml com architecture: ctc e hot_words: true
5.  Pipeline Adaptativo: CTC nao usa cross-segment context
6.  Pipeline Adaptativo: CTC nao usa LocalAgreement (partials nativos)
7.  Hot Words: nativo (WeNet) vs initial_prompt (Whisper)
8.  CTC Streaming: partial + final nativos do worker
9.  Contrato identico: mesmos tipos de evento para ambas as engines
10. Worker CLI: --engine wenet aceito no parse_args

Funciona SEM modelo real instalado -- usa mocks controlados.

Uso:
    .venv/bin/python scripts/demo_m7.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np

from theo._types import ModelType, STTArchitecture, TranscriptSegment
from theo.session.cross_segment import CrossSegmentContext
from theo.session.streaming import StreamingSession
from theo.vad.detector import VADEvent, VADEventType
from theo.workers.stt.main import _create_backend, parse_args
from theo.workers.stt.wenet import WeNetBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
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


def demo_1_factory() -> None:
    """Step 1: Factory cria WeNetBackend e FasterWhisperBackend."""
    step(1, "Factory _create_backend() — duas engines registradas")

    from theo.workers.stt.faster_whisper import FasterWhisperBackend

    # WeNet backend
    wenet_backend = _create_backend("wenet")
    check(isinstance(wenet_backend, WeNetBackend), "Factory cria WeNetBackend para 'wenet'")

    # Faster-Whisper backend
    fw_backend = _create_backend("faster-whisper")
    check(isinstance(fw_backend, FasterWhisperBackend), "Factory cria FasterWhisperBackend para 'faster-whisper'")

    # Unknown engine
    try:
        _create_backend("unknown-engine")
        check(False, "Factory deveria rejeitar engine desconhecida")
    except ValueError as e:
        check("nao suportada" in str(e), f"Factory rejeita engine desconhecida: '{e}'")

    info("  Factory usa lazy imports — cada engine so e importada quando solicitada.")


def demo_2_architecture() -> None:
    """Step 2: WeNet e CTC, Whisper e encoder-decoder."""
    step(2, "Arquitetura: CTC vs encoder-decoder")

    wenet = WeNetBackend()
    check(
        wenet.architecture == STTArchitecture.CTC,
        f"WeNet architecture: {wenet.architecture.value}",
    )

    from theo.workers.stt.faster_whisper import FasterWhisperBackend

    fw = FasterWhisperBackend()
    check(
        fw.architecture == STTArchitecture.ENCODER_DECODER,
        f"Faster-Whisper architecture: {fw.architecture.value}",
    )

    check(
        wenet.architecture != fw.architecture,
        "Arquiteturas sao fundamentalmente diferentes",
    )

    info("  O runtime adapta o pipeline automaticamente baseado no campo 'architecture'.")
    info("  - encoder-decoder: LocalAgreement + cross-segment context")
    info("  - CTC: partials nativos, sem LocalAgreement, sem cross-segment context")


def demo_3_capabilities() -> None:
    """Step 3: WeNet suporta hot words nativos, nao suporta initial_prompt."""
    step(3, "Capabilities: WeNet vs Faster-Whisper")

    async def _run() -> tuple[Any, Any]:
        wenet = WeNetBackend()
        wenet_caps = await wenet.capabilities()

        from theo.workers.stt.faster_whisper import FasterWhisperBackend

        fw = FasterWhisperBackend()
        fw_caps = await fw.capabilities()

        return wenet_caps, fw_caps

    wenet_caps, fw_caps = asyncio.run(_run())

    check(wenet_caps.supports_hot_words is True, "WeNet: supports_hot_words=True (keyword boosting nativo)")
    check(wenet_caps.supports_initial_prompt is False, "WeNet: supports_initial_prompt=False (CTC nao usa)")
    check(fw_caps.supports_initial_prompt is True, "Faster-Whisper: supports_initial_prompt=True (conditioning)")

    info(f"  WeNet caps: hot_words={wenet_caps.supports_hot_words}, "
         f"initial_prompt={wenet_caps.supports_initial_prompt}, "
         f"batch={wenet_caps.supports_batch}")
    info(f"  FW caps:    hot_words={fw_caps.supports_hot_words}, "
         f"initial_prompt={fw_caps.supports_initial_prompt}, "
         f"batch={fw_caps.supports_batch}")


def demo_4_manifest() -> None:
    """Step 4: Manifesto theo.yaml para WeNet CTC."""
    step(4, "Manifesto WeNet: architecture=ctc, hot_words=true")

    from theo.config.manifest import ModelManifest

    manifest_path = "tests/fixtures/manifests/valid_stt_wenet.yaml"
    manifest = ModelManifest.from_yaml_path(manifest_path)

    check(manifest.name == "wenet-ctc", f"Manifest name: {manifest.name}")
    check(manifest.engine == "wenet", f"Manifest engine: {manifest.engine}")
    check(manifest.model_type == ModelType.STT, f"Manifest type: {manifest.model_type}")
    check(
        manifest.capabilities.architecture == STTArchitecture.CTC,
        f"Manifest architecture: {manifest.capabilities.architecture}",
    )
    check(
        manifest.capabilities.hot_words is True,
        f"Manifest hot_words: {manifest.capabilities.hot_words}",
    )
    check(
        manifest.capabilities.initial_prompt is False,
        f"Manifest initial_prompt: {manifest.capabilities.initial_prompt}",
    )
    check(
        manifest.engine_config.vad_filter is False,
        "Manifest vad_filter: false (VAD e do runtime, nao da engine)",
    )

    info(f"  Manifesto completo: {manifest_path}")
    info("  O campo 'architecture: ctc' ativa o pipeline adaptativo no runtime.")


def demo_5_pipeline_no_cross_segment() -> None:
    """Step 5: CTC nao usa cross-segment context."""
    step(5, "Pipeline Adaptativo: CTC sem cross-segment context")

    # Cenario: CTC com cross-segment context disponivel
    context = MagicMock()
    context.get_prompt.return_value = "contexto do segmento anterior"

    session_ctc = StreamingSession(
        session_id="sess_ctc_context",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=AsyncMock(),
        postprocessor=None,
        on_event=AsyncMock(),
        architecture=STTArchitecture.CTC,
        cross_segment_context=context,
    )

    prompt_ctc = session_ctc._build_initial_prompt()
    check(prompt_ctc is None, "CTC: _build_initial_prompt() retorna None (sem context)")

    # Cenario: encoder-decoder com cross-segment context
    session_ed = StreamingSession(
        session_id="sess_ed_context",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=AsyncMock(),
        postprocessor=None,
        on_event=AsyncMock(),
        architecture=STTArchitecture.ENCODER_DECODER,
        cross_segment_context=context,
    )

    prompt_ed = session_ed._build_initial_prompt()
    check(prompt_ed is not None, "Encoder-decoder: _build_initial_prompt() retorna prompt")
    if prompt_ed:
        check(
            "contexto do segmento anterior" in prompt_ed,
            f"Encoder-decoder prompt contem context: '{prompt_ed}'",
        )

    info("  CTC ignora cross-segment context porque nao suporta initial_prompt.")
    info("  O runtime detecta architecture=CTC e pula automaticamente.")


def demo_6_pipeline_no_local_agreement() -> None:
    """Step 6: CTC nao usa LocalAgreement — partials nativos."""
    step(6, "Pipeline Adaptativo: CTC sem LocalAgreement (partials nativos)")

    async def _run() -> list[dict[str, Any]]:
        # CTC emite partials diretamente do worker
        partial1 = TranscriptSegment(
            text="ola", is_final=False, segment_id=0, start_ms=100,
        )
        partial2 = TranscriptSegment(
            text="ola como", is_final=False, segment_id=0, start_ms=200,
        )
        final = TranscriptSegment(
            text="ola como posso ajudar",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=2000, language="pt", confidence=0.93,
        )

        stream_handle = make_stream_handle_mock(events=[partial1, partial2, final])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_ctc_partials",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            architecture=STTArchitecture.CTC,
        )

        # speech_start -> abre stream com worker
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.1)

        # speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected

    events = asyncio.run(_run())

    partials = [e for e in events if e.get("type") == "transcript.partial"]
    finals = [e for e in events if e.get("type") == "transcript.final"]

    check(len(partials) >= 1, f"CTC emitiu {len(partials)} partial(s) nativo(s)")
    check(len(finals) == 1, f"CTC emitiu {len(finals)} final")

    if partials:
        info(f"  Partial 1: '{partials[0]['text']}'")
    if len(partials) > 1:
        info(f"  Partial 2: '{partials[1]['text']}'")
    if finals:
        info(f"  Final:     '{finals[0]['text']}'")

    info("  CTC: partials vem diretamente do worker (sem LocalAgreement).")
    info("  Encoder-decoder: partials sao sinteticos via LocalAgreement no runtime.")


def demo_7_hot_words_native_vs_prompt() -> None:
    """Step 7: Hot words — nativo (WeNet) vs initial_prompt (Whisper)."""
    step(7, "Hot Words: nativo (CTC) vs initial_prompt (encoder-decoder)")

    # CTC com suporte nativo a hot words
    session_ctc_native = StreamingSession(
        session_id="sess_ctc_hw_native",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=AsyncMock(),
        postprocessor=None,
        on_event=AsyncMock(),
        architecture=STTArchitecture.CTC,
        engine_supports_hot_words=True,
        hot_words=["PIX", "TED", "Selic"],
    )

    prompt_ctc_native = session_ctc_native._build_initial_prompt()
    check(
        prompt_ctc_native is None,
        "CTC nativo: hot words NAO no prompt (enviados via gRPC hot_words field)",
    )

    # CTC sem suporte nativo (workaround)
    session_ctc_no_native = StreamingSession(
        session_id="sess_ctc_hw_no_native",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=AsyncMock(),
        postprocessor=None,
        on_event=AsyncMock(),
        architecture=STTArchitecture.CTC,
        engine_supports_hot_words=False,
        hot_words=["PIX", "TED"],
    )

    prompt_ctc_no_native = session_ctc_no_native._build_initial_prompt()
    check(
        prompt_ctc_no_native is not None and "PIX" in prompt_ctc_no_native,
        f"CTC sem nativo: hot words no prompt workaround: '{prompt_ctc_no_native}'",
    )

    # Encoder-decoder sem suporte nativo (Whisper usa initial_prompt)
    context = MagicMock()
    context.get_prompt.return_value = "texto anterior"

    session_ed = StreamingSession(
        session_id="sess_ed_hw",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=AsyncMock(),
        postprocessor=None,
        on_event=AsyncMock(),
        architecture=STTArchitecture.ENCODER_DECODER,
        engine_supports_hot_words=False,
        hot_words=["PIX", "Selic"],
        cross_segment_context=context,
    )

    prompt_ed = session_ed._build_initial_prompt()
    check(
        prompt_ed is not None and "PIX" in prompt_ed and "texto anterior" in prompt_ed,
        f"Encoder-decoder: hot words + context no prompt: '{prompt_ed}'",
    )

    info("  WeNet CTC: hot words enviados via campo nativo (keyword boosting).")
    info("  Whisper:   hot words injetados no initial_prompt como workaround semantico.")
    info("  O runtime decide automaticamente com base no manifesto.")


def demo_8_ctc_streaming_events() -> None:
    """Step 8: Streaming CTC — sequencia completa de eventos."""
    step(8, "CTC Streaming: sequencia partial -> final com ITN")

    async def _run() -> list[dict[str, Any]]:
        postprocessor = make_postprocessor_mock()

        partial_seg = TranscriptSegment(
            text="dois mil e vinte e cinco",
            is_final=False, segment_id=0, start_ms=100,
        )
        final_seg = TranscriptSegment(
            text="dois mil e vinte e cinco reais",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=2000, language="pt", confidence=0.95,
        )

        stream_handle = make_stream_handle_mock(events=[partial_seg, final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id="sess_ctc_itn",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=_on_event,
            architecture=STTArchitecture.CTC,
            enable_itn=True,
        )

        # speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.1)

        # speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return events_collected

    events = asyncio.run(_run())

    partials = [e for e in events if e.get("type") == "transcript.partial"]
    finals = [e for e in events if e.get("type") == "transcript.final"]
    speech_starts = [e for e in events if e.get("type") == "vad.speech_start"]
    speech_ends = [e for e in events if e.get("type") == "vad.speech_end"]

    check(len(speech_starts) >= 1, "Evento vad.speech_start emitido")
    check(len(partials) >= 1, f"Partial nativo CTC: '{partials[0]['text']}'" if partials else "Nenhum partial")
    check(len(finals) == 1, "transcript.final emitido")

    if finals:
        # ITN aplicado no final (via mock: prefixo "[ITN]")
        check(
            "[ITN]" in finals[0]["text"],
            f"ITN aplicado no final: '{finals[0]['text']}'",
        )

    if partials:
        # ITN NAO aplicado em partials
        check(
            "[ITN]" not in partials[0]["text"],
            f"ITN NAO aplicado em partial: '{partials[0]['text']}'",
        )

    check(len(speech_ends) >= 1, "Evento vad.speech_end emitido")

    info("  Sequencia de eventos CTC:")
    for e in events:
        info(f"    {event_line(e)}")


def demo_9_contract_comparison() -> None:
    """Step 9: Contrato identico — mesmos eventos para ambas as engines."""
    step(9, "Contrato Identico: CTC e encoder-decoder emitem mesmos tipos de evento")

    async def _run_session(architecture: STTArchitecture) -> list[str]:
        final_seg = TranscriptSegment(
            text="resultado do teste",
            is_final=True, segment_id=0,
            start_ms=0, end_ms=1000, language="pt", confidence=0.90,
        )

        stream_handle = make_stream_handle_mock(events=[final_seg])
        grpc_client = make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        events_collected: list[dict[str, Any]] = []

        async def _on_event(event: Any) -> None:
            events_collected.append(event.model_dump(mode="json"))

        session = StreamingSession(
            session_id=f"sess_contract_{architecture.value}",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=_on_event,
            architecture=architecture,
        )

        # speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START, timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_pcm_tone())
        await asyncio.sleep(0.05)

        # speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END, timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_pcm_silence())

        await session.close()
        return [e["type"] for e in events_collected]

    ctc_types = asyncio.run(_run_session(STTArchitecture.CTC))
    ed_types = asyncio.run(_run_session(STTArchitecture.ENCODER_DECODER))

    # Ambos devem emitir vad.speech_start, transcript.final, vad.speech_end
    check(
        "vad.speech_start" in ctc_types,
        f"CTC emite vad.speech_start ({ctc_types})",
    )
    check(
        "transcript.final" in ctc_types,
        f"CTC emite transcript.final ({ctc_types})",
    )
    check(
        "vad.speech_start" in ed_types,
        f"Encoder-decoder emite vad.speech_start ({ed_types})",
    )
    check(
        "transcript.final" in ed_types,
        f"Encoder-decoder emite transcript.final ({ed_types})",
    )

    # Os tipos de evento sao os mesmos (mesmo contrato para o cliente)
    ctc_event_set = set(ctc_types)
    ed_event_set = set(ed_types)
    check(
        ctc_event_set == ed_event_set,
        f"Mesmos tipos de evento: CTC={ctc_event_set}, ED={ed_event_set}",
    )

    info("  O cliente recebe os mesmos tipos de evento independente da engine.")
    info("  A diferenca e interna: CTC usa partials nativos, ED usa LocalAgreement.")
    info("  Zero mudanca no codigo cliente ao trocar de engine.")


def demo_10_worker_cli() -> None:
    """Step 10: Worker CLI aceita --engine wenet."""
    step(10, "Worker CLI: --engine wenet aceito no parse_args")

    # WeNet args
    args_wenet = parse_args([
        "--port", "50052",
        "--engine", "wenet",
        "--model-path", "/models/wenet-ctc",
        "--model-size", "wenet-ctc",
    ])
    check(args_wenet.engine == "wenet", f"CLI --engine wenet aceito: {args_wenet.engine}")
    check(args_wenet.port == 50052, f"CLI --port 50052: {args_wenet.port}")
    check(args_wenet.model_path == "/models/wenet-ctc", f"CLI --model-path: {args_wenet.model_path}")

    # Faster-Whisper args (default)
    args_fw = parse_args(["--model-path", "/models/fw"])
    check(args_fw.engine == "faster-whisper", f"CLI default engine: {args_fw.engine}")

    info("  O worker aceita --engine wenet e --engine faster-whisper.")
    info("  Cada worker e um subprocess gRPC separado com sua propria engine.")
    info("  O Worker Manager spawna o subprocess com os argumentos corretos.")


# ===========================================================================
# Main
# ===========================================================================


def main() -> int:
    """Executa todas as demos."""
    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  Demo M7 -- Segundo Backend STT (WeNet CTC) — Model-Agnostic{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print()
    info("Componentes demonstrados:")
    info("  - WeNetBackend (STTBackend para CTC architecture)")
    info("  - Factory _create_backend() com lazy imports (faster-whisper + wenet)")
    info("  - Manifesto theo.yaml para WeNet (architecture: ctc, hot_words: true)")
    info("  - Pipeline Adaptativo: CTC sem LocalAgreement, sem cross-segment context")
    info("  - Hot Words: nativo (WeNet) vs initial_prompt (Whisper)")
    info("  - Contrato identico: mesmos eventos para ambas as engines")
    info("  - Worker CLI: --engine wenet aceito")
    print()
    info("Validacao do principio model-agnostic: mesma interface, engines diferentes,")
    info("zero mudancas no runtime core, zero mudancas no cliente.")
    info("Nenhum modelo real e necessario -- mocks controlados simulam os workers.")

    demos = [
        demo_1_factory,
        demo_2_architecture,
        demo_3_capabilities,
        demo_4_manifest,
        demo_5_pipeline_no_cross_segment,
        demo_6_pipeline_no_local_agreement,
        demo_7_hot_words_native_vs_prompt,
        demo_8_ctc_streaming_events,
        demo_9_contract_comparison,
        demo_10_worker_cli,
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
    print(f"\n{BOLD}{'=' * 70}{NC}")
    total = _total_pass + _total_fail
    print(f"{BOLD}  Resultado: {_total_pass}/{total} checks passaram{NC}")
    if _total_fail > 0:
        print(f"{RED}  {_total_fail} checks falharam{NC}")
    else:
        print(f"{GREEN}  Todos os checks passaram!{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print()

    if _total_fail == 0:
        info("M7 validado: Theo OpenVoice e model-agnostic.")
        info("  - WeNet (CTC) e Faster-Whisper (encoder-decoder) funcionam")
        info("    com a mesma interface STTBackend, mesmos eventos WebSocket,")
        info("    e zero mudancas no runtime core.")
        info("  - Para adicionar uma nova engine: docs/ADDING_ENGINE.md (5 passos)")

    return 0 if _total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
