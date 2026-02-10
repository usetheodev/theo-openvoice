"""Backend TTS para Kokoro.

Implementa TTSBackend usando Kokoro como biblioteca de inferencia.
Kokoro e uma dependencia opcional -- o import e guardado.

Kokoro sintetiza texto em audio PCM, retornando chunks iterativamente
para permitir streaming com baixo TTFB.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from theo._types import VoiceInfo
from theo.exceptions import ModelLoadError, TTSSynthesisError
from theo.logging import get_logger
from theo.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    import kokoro as kokoro_lib
except ImportError:
    kokoro_lib = None

logger = get_logger("worker.tts.kokoro")

_DEFAULT_SAMPLE_RATE = 24000

# Tamanho dos chunks de audio retornados pelo synthesize (bytes).
# 4096 bytes = 2048 samples PCM 16-bit = ~85ms a 24kHz.
_CHUNK_SIZE_BYTES = 4096


class KokoroBackend(TTSBackend):
    """Backend TTS usando Kokoro.

    Sintetiza texto em audio PCM 16-bit via Kokoro. A inferencia
    e executada em executor para nao bloquear o event loop.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if kokoro_lib is None:
            msg = "kokoro nao esta instalado. Instale com: pip install theo-openvoice[kokoro]"
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "cpu"))
        device = _resolve_device(device_str)

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: kokoro_lib.load_model(model_path, device=device),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model_path = model_path
        logger.info(
            "model_loaded",
            model_path=model_path,
            device=device,
        )

    async def synthesize(  # type: ignore[override, misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Sintetiza texto em audio, retornando chunks PCM 16-bit.

        Args:
            text: Texto a ser sintetizado.
            voice: Identificador da voz.
            sample_rate: Taxa de amostragem de saida.
            speed: Velocidade da sintese (0.25-4.0).

        Yields:
            Chunks de audio PCM 16-bit.

        Raises:
            ModelLoadError: Se modelo nao esta carregado.
            TTSSynthesisError: Se a sintese falhar.
        """
        if self._model is None:
            msg = "Modelo nao carregado. Chame load() primeiro."
            raise ModelLoadError("unknown", msg)

        if not text.strip():
            raise TTSSynthesisError(self._model_path, "Texto vazio")

        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(
                None,
                lambda: _synthesize_with_model(
                    self._model,
                    text,
                    voice,
                    sample_rate,
                    speed,
                ),
            )
        except TTSSynthesisError:
            raise
        except Exception as exc:
            raise TTSSynthesisError(self._model_path, str(exc)) from exc

        # Yield em chunks para streaming
        for i in range(0, len(audio_data), _CHUNK_SIZE_BYTES):
            yield audio_data[i : i + _CHUNK_SIZE_BYTES]

    async def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                voice_id="default",
                name="Default",
                language="en",
            ),
            VoiceInfo(
                voice_id="pt_female",
                name="Portuguese Female",
                language="pt",
            ),
        ]

    async def unload(self) -> None:
        self._model = None
        self._model_path = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _resolve_device(device_str: str) -> str:
    """Resolve device string para formato Kokoro.

    Args:
        device_str: "auto", "cpu", "cuda", or "cuda:0".

    Returns:
        Device string no formato esperado pelo Kokoro.
    """
    if device_str == "auto":
        return "cpu"
    return device_str


def _synthesize_with_model(
    model: object,
    text: str,
    voice: str,
    sample_rate: int,
    speed: float,
) -> bytes:
    """Sintetiza texto usando modelo Kokoro.

    Chama a API do Kokoro para gerar audio float32 e converte
    para PCM 16-bit bytes.

    Args:
        model: Modelo Kokoro carregado.
        text: Texto a sintetizar.
        voice: Identificador da voz.
        sample_rate: Taxa de amostragem de saida.
        speed: Velocidade da sintese.

    Returns:
        Audio PCM 16-bit como bytes.

    Raises:
        TTSSynthesisError: Se o resultado for vazio.
    """
    result = model.synthesize(  # type: ignore[attr-defined]
        text,
        voice=voice,
        sample_rate=sample_rate,
        speed=speed,
    )

    # Kokoro pode retornar numpy array float32 ou dict com 'audio' key
    audio_array = _extract_audio_array(result)

    if audio_array is None or len(audio_array) == 0:
        msg = "Sintese retornou audio vazio"
        raise TTSSynthesisError("kokoro", msg)

    return _float32_to_pcm16_bytes(audio_array)


def _extract_audio_array(result: object) -> np.ndarray | None:
    """Extrai array de audio do resultado Kokoro.

    Kokoro pode retornar:
    - numpy array float32 diretamente
    - dict com key 'audio' contendo array
    - objeto com atributo 'audio'

    Args:
        result: Resultado da chamada synthesize do Kokoro.

    Returns:
        Array numpy float32 ou None se nao encontrado.
    """
    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, dict):
        audio = result.get("audio")
        if isinstance(audio, np.ndarray):
            return audio
        return None

    if hasattr(result, "audio"):
        audio = result.audio
        if isinstance(audio, np.ndarray):
            return audio

    return None


def _float32_to_pcm16_bytes(audio_array: np.ndarray) -> bytes:
    """Converte array float32 normalizado [-1, 1] para bytes PCM 16-bit.

    Args:
        audio_array: Audio float32 normalizado.

    Returns:
        Bytes PCM 16-bit little-endian.
    """
    int16_data = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)
    return int16_data.tobytes()
