"""Backend STT para Faster-Whisper (CTranslate2).

Implementa STTBackend usando faster-whisper como biblioteca de inferencia.
Faster-whisper e uma dependencia opcional — o import e guardado.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from theo._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    STTArchitecture,
    WordTimestamp,
)
from theo.exceptions import AudioFormatError, ModelLoadError
from theo.logging import get_logger
from theo.workers.stt.interface import STTBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from theo._types import TranscriptSegment

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = get_logger("worker.stt.faster_whisper")


class FasterWhisperBackend(STTBackend):
    """Backend STT usando Faster-Whisper (CTranslate2).

    Arquitetura: encoder-decoder. Streaming via LocalAgreement (M5).
    """

    def __init__(self) -> None:
        self._model: object | None = None

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if WhisperModel is None:
            msg = "faster-whisper nao esta instalado. Instale com: pip install theo-openvoice[faster-whisper]"
            raise ModelLoadError(model_path, msg)

        model_size = str(config.get("model_size", model_path))
        compute_type = str(config.get("compute_type", "float16"))
        device = str(config.get("device", "auto"))

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                ),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        logger.info(
            "model_loaded",
            model_size=model_size,
            compute_type=compute_type,
            device=device,
        )

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_hot_words=False,
            supports_initial_prompt=True,
            supports_batch=True,
            supports_word_timestamps=True,
            max_concurrent_sessions=1,
        )

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        if self._model is None:
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        if not audio_data:
            msg = "Audio vazio"
            raise AudioFormatError(msg)

        audio_array = _audio_bytes_to_numpy(audio_data)

        fw_language: str | None = language
        if fw_language in ("auto", "mixed"):
            fw_language = None

        # Build initial_prompt with hot words if provided
        effective_prompt = initial_prompt
        if hot_words:
            hot_words_text = f"Termos: {', '.join(hot_words)}."
            effective_prompt = (
                f"{effective_prompt} {hot_words_text}" if effective_prompt else hot_words_text
            )

        loop = asyncio.get_running_loop()
        segments_iter, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(  # type: ignore[union-attr]
                audio_array,
                language=fw_language,
                initial_prompt=effective_prompt,
                temperature=temperature,
                word_timestamps=word_timestamps,
                beam_size=5,
                vad_filter=False,
            ),
        )

        # Materialize segments (generator -> list)
        fw_segments = await loop.run_in_executor(None, list, segments_iter)

        segment_details = tuple(
            _fw_segment_to_detail(seg, idx) for idx, seg in enumerate(fw_segments)
        )

        full_text = " ".join(seg.text for seg in segment_details).strip()
        words = _extract_words(fw_segments, word_timestamps)

        return BatchResult(
            text=full_text,
            language=info.language,
            duration=info.duration,
            segments=segment_details,
            words=words,
        )

    async def transcribe_stream(  # type: ignore[override, misc]
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError("Streaming sera implementado no M5 (LocalAgreement)")
        yield  # pragma: no cover - unreachable, makes it an async generator

    async def unload(self) -> None:
        self._model = None
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


def _audio_bytes_to_numpy(audio_data: bytes) -> np.ndarray:
    """Converte bytes PCM 16-bit em numpy float32 normalizado.

    Args:
        audio_data: Audio PCM 16-bit (little-endian).

    Returns:
        Array float32 normalizado para [-1.0, 1.0].

    Raises:
        AudioFormatError: Se os bytes nao tem tamanho par (PCM 16-bit = 2 bytes/sample).
    """
    if len(audio_data) % 2 != 0:
        msg = "Audio PCM 16-bit deve ter numero par de bytes"
        raise AudioFormatError(msg)

    int16_array = np.frombuffer(audio_data, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


def _fw_segment_to_detail(segment: object, index: int) -> SegmentDetail:
    """Converte segmento faster-whisper em SegmentDetail Theo."""
    return SegmentDetail(
        id=index,
        start=segment.start,  # type: ignore[attr-defined]
        end=segment.end,  # type: ignore[attr-defined]
        text=segment.text.strip(),  # type: ignore[attr-defined]
        avg_logprob=segment.avg_logprob,  # type: ignore[attr-defined]
        no_speech_prob=segment.no_speech_prob,  # type: ignore[attr-defined]
        compression_ratio=segment.compression_ratio,  # type: ignore[attr-defined]
    )


def _fw_word_to_timestamp(word: object) -> WordTimestamp:
    """Converte word faster-whisper em WordTimestamp Theo."""
    return WordTimestamp(
        word=word.word.strip(),  # type: ignore[attr-defined]
        start=word.start,  # type: ignore[attr-defined]
        end=word.end,  # type: ignore[attr-defined]
        probability=word.probability,  # type: ignore[attr-defined]
    )


def _extract_words(
    fw_segments: list[object],
    word_timestamps: bool,
) -> tuple[WordTimestamp, ...] | None:
    """Extrai todas as words de todos os segmentos faster-whisper."""
    if not word_timestamps:
        return None

    all_words: list[WordTimestamp] = []
    for seg in fw_segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            all_words.extend(_fw_word_to_timestamp(w) for w in seg_words)

    return tuple(all_words) if all_words else None
