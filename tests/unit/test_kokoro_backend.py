"""Testes para KokoroBackend.

Usa mocks para o modulo kokoro -- nao requer kokoro instalado.
Segue o mesmo padrao de test_wenet_backend.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from theo._types import VoiceInfo
from theo.exceptions import ModelLoadError, TTSSynthesisError
from theo.workers.tts.kokoro import (
    KokoroBackend,
    _extract_audio_array,
    _float32_to_pcm16_bytes,
    _resolve_device,
)


def _make_mock_kokoro_lib(
    audio_output: np.ndarray | None = None,
) -> MagicMock:
    """Cria mock da biblioteca kokoro com resultado configuravel."""
    mock_lib = MagicMock()
    mock_model = MagicMock()

    if audio_output is None:
        audio_output = np.zeros(2400, dtype=np.float32)

    mock_model.synthesize.return_value = audio_output
    mock_lib.load_model.return_value = mock_model
    return mock_lib


class TestHealth:
    async def test_ok_when_model_loaded(self) -> None:
        backend = KokoroBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        health = await backend.health()
        assert health["status"] == "ok"

    async def test_not_loaded_when_model_none(self) -> None:
        backend = KokoroBackend()
        health = await backend.health()
        assert health["status"] == "not_loaded"


class TestLoad:
    async def test_load_succeeds_with_mock(self) -> None:
        mock_kokoro = _make_mock_kokoro_lib()

        import theo.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load("/models/kokoro-v1", {"device": "cpu"})
            assert backend._model is not None
            mock_kokoro.load_model.assert_called_once()
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_stores_model_path(self) -> None:
        mock_kokoro = _make_mock_kokoro_lib()

        import theo.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load("/models/kokoro-v1", {})
            assert backend._model_path == "/models/kokoro-v1"
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_without_library_raises_model_load_error(self) -> None:
        import theo.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = None  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match="nao esta instalado"):
                await backend.load("/models/kokoro-v1", {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_failure_raises_model_load_error(self) -> None:
        import theo.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        mock_kokoro = MagicMock()
        mock_kokoro.load_model.side_effect = RuntimeError("Model file not found")
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            with pytest.raises(ModelLoadError, match="Model file not found"):
                await backend.load("/models/kokoro-v1", {})
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]

    async def test_load_with_auto_device(self) -> None:
        mock_kokoro = _make_mock_kokoro_lib()

        import theo.workers.tts.kokoro as kokoro_mod

        original = kokoro_mod.kokoro_lib
        kokoro_mod.kokoro_lib = mock_kokoro  # type: ignore[assignment]
        try:
            backend = KokoroBackend()
            await backend.load("/models/kokoro-v1", {"device": "auto"})
            # "auto" should be resolved to "cpu"
            call_kwargs = mock_kokoro.load_model.call_args
            assert call_kwargs[1]["device"] == "cpu"
        finally:
            kokoro_mod.kokoro_lib = original  # type: ignore[assignment]


class TestSynthesize:
    def _make_loaded_backend(
        self,
        audio_output: np.ndarray | None = None,
    ) -> KokoroBackend:
        """Cria backend com modelo mock carregado."""
        backend = KokoroBackend()
        mock_model = MagicMock()

        if audio_output is None:
            audio_output = np.zeros(2400, dtype=np.float32)

        mock_model.synthesize.return_value = audio_output
        backend._model = mock_model  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"
        return backend

    async def test_returns_audio_chunks(self) -> None:
        backend = self._make_loaded_backend()
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello world"):
            chunks.append(chunk)
        assert len(chunks) > 0
        # Total bytes should equal 2400 samples * 2 bytes/sample
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 2400 * 2

    async def test_chunk_size_limited(self) -> None:
        # Create large audio output
        large_audio = np.zeros(48000, dtype=np.float32)  # 2 seconds at 24kHz
        backend = self._make_loaded_backend(audio_output=large_audio)
        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Long text"):
            chunks.append(chunk)

        # Every chunk except possibly the last should be exactly 4096 bytes
        for chunk in chunks[:-1]:
            assert len(chunk) == 4096

        # Last chunk may be smaller
        assert len(chunks[-1]) <= 4096

    async def test_empty_text_raises_synthesis_error(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize(""):
                pass

    async def test_whitespace_only_raises_synthesis_error(self) -> None:
        backend = self._make_loaded_backend()
        with pytest.raises(TTSSynthesisError, match="Texto vazio"):
            async for _ in backend.synthesize("   "):
                pass

    async def test_model_not_loaded_raises_error(self) -> None:
        backend = KokoroBackend()
        with pytest.raises(ModelLoadError, match="nao carregado"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_passes_voice_parameter(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", voice="pt_female"):
            pass
        mock_model = backend._model
        assert mock_model is not None
        call_kwargs = mock_model.synthesize.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["voice"] == "pt_female"

    async def test_passes_speed_parameter(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", speed=1.5):
            pass
        mock_model = backend._model
        assert mock_model is not None
        call_kwargs = mock_model.synthesize.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["speed"] == 1.5

    async def test_passes_sample_rate_parameter(self) -> None:
        backend = self._make_loaded_backend()
        async for _ in backend.synthesize("Test", sample_rate=16000):
            pass
        mock_model = backend._model
        assert mock_model is not None
        call_kwargs = mock_model.synthesize.call_args  # type: ignore[union-attr]
        assert call_kwargs[1]["sample_rate"] == 16000

    async def test_inference_error_raises_synthesis_error(self) -> None:
        backend = KokoroBackend()
        mock_model = MagicMock()
        mock_model.synthesize.side_effect = RuntimeError("GPU OOM")
        backend._model = mock_model  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        with pytest.raises(TTSSynthesisError, match="GPU OOM"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_empty_audio_result_raises_synthesis_error(self) -> None:
        empty_audio = np.array([], dtype=np.float32)
        backend = self._make_loaded_backend(audio_output=empty_audio)

        with pytest.raises(TTSSynthesisError, match="audio vazio"):
            async for _ in backend.synthesize("Hello"):
                pass

    async def test_dict_result_with_audio_key(self) -> None:
        backend = KokoroBackend()
        mock_model = MagicMock()
        audio = np.zeros(1200, dtype=np.float32)
        mock_model.synthesize.return_value = {"audio": audio}
        backend._model = mock_model  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello"):
            chunks.append(chunk)

        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 1200 * 2

    async def test_object_result_with_audio_attribute(self) -> None:
        backend = KokoroBackend()
        mock_model = MagicMock()
        audio = np.zeros(1200, dtype=np.float32)

        class AudioResult:
            def __init__(self) -> None:
                self.audio = audio

        mock_model.synthesize.return_value = AudioResult()
        backend._model = mock_model  # type: ignore[assignment]
        backend._model_path = "/models/kokoro-v1"

        chunks: list[bytes] = []
        async for chunk in backend.synthesize("Hello"):
            chunks.append(chunk)

        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 1200 * 2


class TestVoices:
    async def test_returns_voice_info_list(self) -> None:
        backend = KokoroBackend()
        result = await backend.voices()
        assert isinstance(result, list)
        assert len(result) >= 1

    async def test_all_items_are_voice_info(self) -> None:
        backend = KokoroBackend()
        result = await backend.voices()
        for voice in result:
            assert isinstance(voice, VoiceInfo)
            assert voice.voice_id
            assert voice.name
            assert voice.language

    async def test_default_voice_present(self) -> None:
        backend = KokoroBackend()
        result = await backend.voices()
        voice_ids = [v.voice_id for v in result]
        assert "default" in voice_ids


class TestUnload:
    async def test_clears_model(self) -> None:
        backend = KokoroBackend()
        backend._model = MagicMock()  # type: ignore[assignment]
        backend._model_path = "/models/test"
        await backend.unload()
        assert backend._model is None
        assert backend._model_path == ""

    async def test_unload_when_already_none(self) -> None:
        backend = KokoroBackend()
        await backend.unload()
        assert backend._model is None


class TestResolveDevice:
    def test_auto_defaults_to_cpu(self) -> None:
        assert _resolve_device("auto") == "cpu"

    def test_cpu_passthrough(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_cuda_with_id(self) -> None:
        assert _resolve_device("cuda:0") == "cuda:0"


class TestExtractAudioArray:
    def test_extracts_from_numpy_array(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _extract_audio_array(audio)
        assert result is not None
        assert len(result) == 100

    def test_extracts_from_dict_with_audio_key(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _extract_audio_array({"audio": audio})
        assert result is not None
        assert len(result) == 100

    def test_extracts_from_object_with_audio_attr(self) -> None:
        audio = np.zeros(100, dtype=np.float32)

        class FakeResult:
            def __init__(self) -> None:
                self.audio = audio

        result = _extract_audio_array(FakeResult())
        assert result is not None
        assert len(result) == 100

    def test_returns_none_for_dict_without_audio(self) -> None:
        result = _extract_audio_array({"text": "hello"})
        assert result is None

    def test_returns_none_for_string(self) -> None:
        result = _extract_audio_array("not audio")
        assert result is None

    def test_returns_none_for_dict_with_non_array_audio(self) -> None:
        result = _extract_audio_array({"audio": "not_an_array"})
        assert result is None


class TestFloat32ToPcm16Bytes:
    def test_converts_silence(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 200  # 100 samples * 2 bytes/sample
        # All zeros
        assert result == b"\x00\x00" * 100

    def test_converts_max_positive(self) -> None:
        audio = np.ones(1, dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        # Should be clipped to 32767
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == 32767

    def test_converts_max_negative(self) -> None:
        audio = np.array([-1.0], dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 2
        value = int.from_bytes(result, byteorder="little", signed=True)
        assert value == -32768

    def test_clips_beyond_range(self) -> None:
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = _float32_to_pcm16_bytes(audio)
        assert len(result) == 4
        values = np.frombuffer(result, dtype=np.int16)
        assert values[0] == 32767
        assert values[1] == -32768
