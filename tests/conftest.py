"""Fixtures compartilhadas para todos os testes."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
MANIFESTS_DIR = FIXTURES_DIR / "manifests"


@pytest.fixture
def audio_16khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 16kHz, mono."""
    path = AUDIO_DIR / "sample_16khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_8khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 8kHz, mono."""
    path = AUDIO_DIR / "sample_8khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_44khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 44.1kHz, mono."""
    path = AUDIO_DIR / "sample_44khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def valid_stt_manifest_path() -> Path:
    """Caminho para manifesto STT valido."""
    path = MANIFESTS_DIR / "valid_stt.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def valid_tts_manifest_path() -> Path:
    """Caminho para manifesto TTS valido."""
    path = MANIFESTS_DIR / "valid_tts.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def minimal_manifest_path() -> Path:
    """Caminho para manifesto com campos minimos."""
    path = MANIFESTS_DIR / "minimal.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path


@pytest.fixture
def invalid_manifest_path() -> Path:
    """Caminho para manifesto invalido (campos obrigatorios faltando)."""
    path = MANIFESTS_DIR / "invalid_missing.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path
