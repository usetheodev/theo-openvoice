# Adding a New STT Engine to Theo OpenVoice

This document explains how to add a new STT engine to the Theo OpenVoice runtime. The runtime is model-agnostic: every engine plugs in through the same `STTBackend` interface, and the runtime adapts the streaming pipeline automatically based on the engine's `architecture` field.

**Audience**: Developers adding a new engine (e.g., Paraformer, Wav2Vec2, SenseVoice).
**Prerequisite**: Familiarity with the codebase structure described in `docs/ARCHITECTURE.md`.
**Reference implementation**: `WeNetBackend` (CTC architecture) -- added in M7 alongside the existing `FasterWhisperBackend` (encoder-decoder architecture).

---

## Extension Points in the Runtime

The following diagram shows where a new engine connects to the runtime. Only the shaded areas require changes; everything else stays untouched.

```
                   +-----------------------------------------+
                   |         Theo Runtime (unchanged)         |
                   |                                         |
  Client  ------->|  API Server (FastAPI)                    |
  (REST/WS)       |       |                                  |
                   |       v                                  |
                   |  Scheduler                               |
                   |       |                                  |
                   |       v                                  |
                   |  Session Manager                         |
                   |   (state machine, ring buffer, WAL,      |
                   |    LocalAgreement, cross-segment ctx)    |
                   |       |                                  |
                   |       | architecture field               |
                   |       | from manifest controls:          |
                   |       |  - LocalAgreement (enc-dec only) |
                   |       |  - Cross-segment ctx (enc-dec)   |
                   |       |  - Partial strategy (native/LA)  |
                   |       |                                  |
                   |       v                                  |
                   |  Worker Manager  -- spawns subprocess -->|---+
                   |                                         |   |
                   +-----------------------------------------+   |
                                                                 |
      +----------------------------------------------------------+
      |
      v
  +-------------------------------------------------------------+
  |                WORKER SUBPROCESS (gRPC)            [CHANGE]  |
  |                                                              |
  |  +--------------------------------------------------------+ |
  |  | _create_backend() in main.py                   [CHANGE] | |
  |  |   "my-engine" -> MyEngineBackend()                      | |
  |  +--------------------------------------------------------+ |
  |                         |                                    |
  |                         v                                    |
  |  +--------------------------------------------------------+ |
  |  | MyEngineBackend (implements STTBackend)         [NEW]   | |
  |  |   .architecture -> STTArchitecture.XXX                  | |
  |  |   .load() / .unload()                                   | |
  |  |   .transcribe_file()                                    | |
  |  |   .transcribe_stream()                                  | |
  |  |   .capabilities()                                       | |
  |  |   .health()                                             | |
  |  +--------------------------------------------------------+ |
  |                                                              |
  +-------------------------------------------------------------+

  +-------------------------------------------------------------+
  | Model Directory                                     [NEW]   |
  |   models/my-engine-model/                                    |
  |     theo.yaml   <-- manifest declaring engine + architecture |
  |     model files  <-- weights, config, vocab, etc.            |
  +-------------------------------------------------------------+

  +-------------------------------------------------------------+
  | pyproject.toml                                      [CHANGE] |
  |   [project.optional-dependencies]                            |
  |     my-engine = ["my-engine-lib>=1.0,<2.0"]                  |
  +-------------------------------------------------------------+
```

**What you touch (4 files + 1 directory):**

| Item | Type | Description |
|------|------|-------------|
| `src/theo/workers/stt/my_engine.py` | New file | `STTBackend` implementation |
| `src/theo/workers/stt/main.py` | Edit | Add `elif` in `_create_backend()` |
| `models/my-engine-model/theo.yaml` | New file | Model manifest |
| `pyproject.toml` | Edit | Optional dependency |
| `tests/unit/test_my_engine_backend.py` | New file | Unit tests |

**What you do NOT touch:**
- API Server, routes, Pydantic models
- Session Manager, state machine, ring buffer, WAL
- Preprocessing / post-processing pipelines
- Scheduler, Worker Manager
- CLI, Prometheus metrics
- WebSocket protocol, gRPC proto definitions

---

## Step 1: Implement STTBackend

Create a new file at `src/theo/workers/stt/my_engine.py` implementing the `STTBackend` abstract class.

### The interface

The full interface is defined at `src/theo/workers/stt/interface.py`:

```python
class STTBackend(ABC):
    @property
    @abstractmethod
    def architecture(self) -> STTArchitecture: ...

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None: ...

    @abstractmethod
    async def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult: ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def health(self) -> dict[str, str]: ...
```

### Concrete example: WeNetBackend

File: `src/theo/workers/stt/wenet.py` (abbreviated for clarity; full file in the codebase):

```python
"""Backend STT para WeNet (CTC)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from theo._types import (
    BatchResult,
    EngineCapabilities,
    SegmentDetail,
    STTArchitecture,
    TranscriptSegment,
)
from theo.exceptions import AudioFormatError, ModelLoadError
from theo.logging import get_logger
from theo.workers.stt.interface import STTBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Guard the heavy import -- engine is an optional dependency
try:
    import wenet as wenet_lib
except ImportError:
    wenet_lib = None

logger = get_logger("worker.stt.wenet")

_SAMPLE_RATE = 16000


class WeNetBackend(STTBackend):
    """Backend STT using WeNet (CTC/Attention)."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""

    # --- 1. Declare architecture ---
    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.CTC

    # --- 2. Load model into memory ---
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if wenet_lib is None:
            msg = (
                "wenet is not installed. "
                "Install with: pip install theo-openvoice[wenet]"
            )
            raise ModelLoadError(model_path, msg)

        device = str(config.get("device", "cpu"))
        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: wenet_lib.load_model(model_path, device=device),
            )
        except Exception as exc:
            raise ModelLoadError(model_path, str(exc)) from exc
        self._model_path = model_path

    # --- 3. Report runtime capabilities ---
    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_hot_words=True,       # WeNet has native keyword boosting
            supports_initial_prompt=False,  # CTC does not support conditioning
            supports_batch=True,
            supports_word_timestamps=True,
            max_concurrent_sessions=1,
        )

    # --- 4. Batch transcription ---
    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,  # ignored for CTC
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        if self._model is None:
            raise ModelLoadError("unknown", "Model not loaded. Call load() first.")
        if not audio_data:
            raise AudioFormatError("Empty audio")

        # Convert PCM int16 bytes -> float32 numpy array
        audio_array = _audio_bytes_to_numpy(audio_data)
        duration = len(audio_array) / _SAMPLE_RATE

        # Run inference in executor (blocking call)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _transcribe_with_model(self._model, audio_array, hot_words),
        )

        text = _extract_text(result)
        return BatchResult(
            text=text,
            language=language or "zh",
            duration=duration,
            segments=_build_segments(result, duration),
            words=_build_words(result) if word_timestamps else None,
        )

    # --- 5. Streaming transcription ---
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        if self._model is None:
            raise ModelLoadError("unknown", "Model not loaded.")

        # CTC streaming: emit partial after each chunk meeting minimum size
        min_samples = int(0.16 * _SAMPLE_RATE)  # 160ms minimum
        buffer_chunks: list[np.ndarray] = []
        buffer_samples = 0
        segment_id = 0

        async for chunk in audio_chunks:
            if not chunk:
                break
            audio_array = _audio_bytes_to_numpy(chunk)
            buffer_chunks.append(audio_array)
            buffer_samples += len(audio_array)

            if buffer_samples >= min_samples:
                accumulated = np.concatenate(buffer_chunks)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda a=accumulated: _transcribe_with_model(
                        self._model, a, hot_words,
                    ),
                )
                text = _extract_text(result)
                if text:
                    yield TranscriptSegment(
                        text=text,
                        is_final=False,
                        segment_id=segment_id,
                    )

        # Final: transcribe all accumulated audio
        if buffer_chunks:
            all_audio = np.concatenate(buffer_chunks)
            if len(all_audio) > 0:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: _transcribe_with_model(
                        self._model, all_audio, hot_words,
                    ),
                )
                text = _extract_text(result)
                if text:
                    yield TranscriptSegment(
                        text=text,
                        is_final=True,
                        segment_id=segment_id,
                    )

    # --- 6. Unload model ---
    async def unload(self) -> None:
        self._model = None
        self._model_path = ""

    # --- 7. Health check ---
    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}
```

### Key patterns to follow

1. **Guard the engine import** with `try/except ImportError`. The engine library is an optional dependency. If not installed, `load()` must raise `ModelLoadError` with a clear install instruction.

2. **Run blocking inference in an executor**. Engine inference is CPU/GPU-bound. Always use `await loop.run_in_executor(None, ...)` to avoid blocking the asyncio event loop.

3. **Audio input is always PCM 16-bit 16kHz mono bytes**. The runtime preprocessing pipeline normalizes all audio before it reaches the worker. Your backend receives clean, normalized audio.

4. **Use Theo's domain types** (`BatchResult`, `TranscriptSegment`, `SegmentDetail`, `WordTimestamp`, `EngineCapabilities`) from `theo._types`. Do not invent new return types.

5. **Use Theo's typed exceptions** (`ModelLoadError`, `AudioFormatError`) from `theo.exceptions`. Do not raise bare `Exception`.

6. **`transcribe_stream` yields `TranscriptSegment`**. Emit `is_final=False` for partials and `is_final=True` for confirmed segments. An empty chunk (`b""`) signals end of stream.

### Step 1 Checklist

- [ ] File created at `src/theo/workers/stt/<engine_name>.py`
- [ ] Class extends `STTBackend` and implements all 7 abstract methods
- [ ] `architecture` property returns the correct `STTArchitecture` enum value
- [ ] Engine library imported with `try/except ImportError` guard
- [ ] `load()` raises `ModelLoadError` if engine library not installed
- [ ] `load()` runs model loading in executor (non-blocking)
- [ ] `capabilities()` accurately reports what the engine supports
- [ ] `transcribe_file()` converts PCM bytes to engine input, runs in executor, returns `BatchResult`
- [ ] `transcribe_stream()` yields `TranscriptSegment` with `is_final` set correctly
- [ ] `unload()` releases model reference and any GPU resources
- [ ] `health()` returns `{"status": "ok"}` when loaded, `{"status": "not_loaded"}` otherwise
- [ ] `from __future__ import annotations` is the first import
- [ ] Type-only imports are in `if TYPE_CHECKING:` block (ruff TCH rules)

---

## Step 2: Create the Model Manifest

Create a `theo.yaml` file in the model directory. The manifest declares the engine name, architecture, capabilities, resource requirements, and engine-specific configuration.

### Concrete example: WeNet CTC manifest

File: `models/wenet-ctc/theo.yaml`

```yaml
name: wenet-ctc
version: 1.0.0
engine: wenet
type: stt
description: "WeNet CTC - streaming-native STT with keyword boosting"

capabilities:
  streaming: true
  architecture: ctc                 # <-- controls pipeline adaptation
  languages: ["zh", "en"]
  word_timestamps: true
  translation: false
  partial_transcripts: true
  hot_words: true                   # <-- engine supports native keyword boosting
  batch_inference: true
  language_detection: false
  initial_prompt: false             # <-- CTC does not support conditioning

resources:
  memory_mb: 1024
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 5

engine_config:
  device: "auto"
  vad_filter: false                 # VAD is runtime's responsibility, not engine's
  language: "chinese"
```

### For reference: Faster-Whisper manifest (encoder-decoder)

```yaml
name: faster-whisper-large-v3
version: 3.0.0
engine: faster-whisper
type: stt

capabilities:
  streaming: true
  architecture: encoder-decoder     # <-- uses LocalAgreement for partials
  languages: ["auto", "en", "pt", "es"]
  hot_words: false                  # <-- no native boosting, uses initial_prompt
  initial_prompt: true              # <-- supports conditioning via prompt
```

### Critical fields that affect runtime behavior

| Field | Effect on Runtime |
|-------|-------------------|
| `engine` | Must match the string in `_create_backend()` (Step 3) |
| `type` | Must be `stt`. Normalized to `model_type` in `ModelManifest` Python class |
| `capabilities.architecture` | Controls streaming pipeline: `encoder-decoder` uses LocalAgreement for partials; `ctc` and `streaming-native` use native partials |
| `capabilities.hot_words` | If `true`, runtime sends hot words via gRPC `hot_words` field for native boosting. If `false`, runtime injects hot words into `initial_prompt` as workaround |
| `capabilities.initial_prompt` | If `false`, runtime skips cross-segment context (last 224 tokens). Only encoder-decoder engines use this |
| `engine_config.vad_filter` | Must be `false`. VAD is the runtime's responsibility |

### How the registry discovers manifests

The `ModelRegistry` scans a `models/` directory on startup. Each subdirectory containing a `theo.yaml` is loaded:

```
models/
  faster-whisper-large-v3/
    theo.yaml
    model.bin
    ...
  wenet-ctc/
    theo.yaml
    final.zip
    ...
```

The registry indexes by the `name` field in the manifest (not the directory name). The manifest is parsed by `ModelManifest.from_yaml_path()` in `src/theo/config/manifest.py`.

### Step 2 Checklist

- [ ] `theo.yaml` file created in model directory
- [ ] `name` field is unique across all installed models
- [ ] `engine` field matches the string used in `_create_backend()`
- [ ] `type` is `stt`
- [ ] `capabilities.architecture` is one of: `encoder-decoder`, `ctc`, `streaming-native`
- [ ] `capabilities.hot_words` accurately reflects engine's native support
- [ ] `capabilities.initial_prompt` is `false` for CTC and streaming-native engines
- [ ] `engine_config.vad_filter` is `false`
- [ ] `resources` section has reasonable estimates for `memory_mb`, `gpu_required`, `load_time_seconds`
- [ ] A test fixture copy exists at `tests/fixtures/manifests/valid_stt_<engine>.yaml`

---

## Step 3: Register the Factory

Edit `src/theo/workers/stt/main.py` to add an `if` branch in the `_create_backend()` function.

### Current state (with WeNet already registered)

```python
# File: src/theo/workers/stt/main.py

def _create_backend(engine: str) -> STTBackend:
    """Creates the STTBackend instance based on engine name.

    Raises:
        ValueError: If engine is not supported.
    """
    if engine == "faster-whisper":
        from theo.workers.stt.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()

    if engine == "wenet":
        from theo.workers.stt.wenet import WeNetBackend
        return WeNetBackend()

    msg = f"Engine STT nao suportada: {engine}"
    raise ValueError(msg)
```

### Adding a new engine (example: Paraformer)

```python
def _create_backend(engine: str) -> STTBackend:
    if engine == "faster-whisper":
        from theo.workers.stt.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()

    if engine == "wenet":
        from theo.workers.stt.wenet import WeNetBackend
        return WeNetBackend()

    if engine == "paraformer":                                          # <-- NEW
        from theo.workers.stt.paraformer import ParaformerBackend       # <-- NEW
        return ParaformerBackend()                                      # <-- NEW

    msg = f"Engine STT nao suportada: {engine}"
    raise ValueError(msg)
```

### Why lazy imports

The `import` is inside the `if` branch intentionally. This ensures that the heavy engine library (e.g., `wenet`, `faster_whisper`, `funasr`) is only imported when that specific engine is requested. A worker subprocess only loads one engine, so other engine libraries are never imported.

### Step 3 Checklist

- [ ] New `if engine == "<name>":` branch added to `_create_backend()` in `src/theo/workers/stt/main.py`
- [ ] Import is inside the `if` branch (lazy import)
- [ ] The engine string matches the `engine` field in the manifest's `theo.yaml`
- [ ] `ValueError` is still raised for unknown engines (existing behavior preserved)

---

## Step 4: Declare the Optional Dependency

Edit `pyproject.toml` to add the engine library as an optional dependency group.

### Current state

```toml
# File: pyproject.toml

[project.optional-dependencies]
faster-whisper = ["faster-whisper>=1.1,<2.0"]
```

### Adding WeNet

```toml
[project.optional-dependencies]
faster-whisper = ["faster-whisper>=1.1,<2.0"]
wenet = ["wenet>=1.0,<2.0"]                      # <-- NEW
```

Also add a mypy override for the new library (since engine libraries typically lack type stubs):

```toml
[[tool.mypy.overrides]]
module = "wenet.*"
ignore_missing_imports = true
```

### Installation

Users install only the engines they need:

```bash
pip install theo-openvoice[wenet]
pip install theo-openvoice[faster-whisper]
pip install theo-openvoice[faster-whisper,wenet]   # both
```

### Step 4 Checklist

- [ ] New optional dependency group added to `pyproject.toml` under `[project.optional-dependencies]`
- [ ] Version constraints are pinned with lower and upper bounds (e.g., `>=1.0,<2.0`)
- [ ] mypy override added for the engine module under `[[tool.mypy.overrides]]` with `ignore_missing_imports = true`
- [ ] `make check` passes (format, lint, typecheck)

---

## Step 5: Write Tests

Tests must cover the backend without requiring the real engine library installed. Use mocks to simulate the engine.

### Test file location

```
tests/unit/test_<engine>_backend.py
```

### What to test

| Test Category | What to Verify |
|---------------|----------------|
| Architecture | `backend.architecture` returns the correct `STTArchitecture` enum |
| Capabilities | `capabilities()` returns accurate `EngineCapabilities` |
| Load success | `load()` calls engine's model loading function with correct params |
| Load failure (not installed) | `load()` raises `ModelLoadError` when engine library is `None` |
| Load failure (model error) | `load()` raises `ModelLoadError` when engine throws |
| Transcribe file | `transcribe_file()` returns `BatchResult` with correct fields |
| Transcribe file (empty audio) | Raises `AudioFormatError` |
| Transcribe stream (partials) | `transcribe_stream()` yields partial `TranscriptSegment`s |
| Transcribe stream (final) | Last yielded segment has `is_final=True` |
| Unload | `unload()` sets model to `None` |
| Health (loaded) | Returns `{"status": "ok"}` |
| Health (not loaded) | Returns `{"status": "not_loaded"}` |
| Helper functions | Pure helper functions tested in isolation |

### Concrete example: WeNet backend tests

File: `tests/unit/test_wenet_backend.py` (abbreviated):

```python
"""Tests for WeNetBackend. Uses mocks -- does not require wenet installed."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from theo._types import STTArchitecture
from theo.exceptions import AudioFormatError, ModelLoadError
from theo.workers.stt.wenet import WeNetBackend


class TestArchitecture:
    def test_is_ctc(self) -> None:
        backend = WeNetBackend()
        assert backend.architecture == STTArchitecture.CTC


class TestCapabilities:
    async def test_supports_hot_words(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_hot_words is True

    async def test_does_not_support_initial_prompt(self) -> None:
        backend = WeNetBackend()
        caps = await backend.capabilities()
        assert caps.supports_initial_prompt is False


class TestLoad:
    async def test_load_raises_when_wenet_not_installed(self) -> None:
        """When wenet library is not installed, load() raises ModelLoadError."""
        import theo.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        wenet_mod.wenet_lib = None  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            with pytest.raises(ModelLoadError, match="wenet"):
                await backend.load("/fake/path", {"device": "cpu"})
        finally:
            wenet_mod.wenet_lib = original

    async def test_load_succeeds_with_mock(self) -> None:
        """When wenet is available, load() calls load_model()."""
        import theo.workers.stt.wenet as wenet_mod

        original = wenet_mod.wenet_lib
        mock_wenet = MagicMock()
        mock_wenet.load_model.return_value = MagicMock()
        wenet_mod.wenet_lib = mock_wenet  # type: ignore[assignment]
        try:
            backend = WeNetBackend()
            await backend.load("/model/path", {"device": "cpu"})
            assert await backend.health() == {"status": "ok"}
        finally:
            wenet_mod.wenet_lib = original


class TestTranscribeFile:
    async def test_empty_audio_raises(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()
        with pytest.raises(AudioFormatError):
            await backend.transcribe_file(b"")

    async def test_odd_bytes_raises(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()
        with pytest.raises(AudioFormatError, match="par"):
            await backend.transcribe_file(b"\x00\x01\x02")


class TestHealth:
    async def test_not_loaded(self) -> None:
        backend = WeNetBackend()
        assert await backend.health() == {"status": "not_loaded"}

    async def test_loaded(self) -> None:
        backend = WeNetBackend()
        backend._model = MagicMock()
        assert await backend.health() == {"status": "ok"}
```

### Mocking pattern

The mock pattern replaces the module-level engine reference:

```python
import theo.workers.stt.wenet as wenet_mod

original = wenet_mod.wenet_lib       # save original
wenet_mod.wenet_lib = mock_wenet     # replace with mock
try:
    # ... test code ...
finally:
    wenet_mod.wenet_lib = original   # always restore
```

This avoids installing the real engine library in CI while testing all backend logic.

### Running tests

```bash
# Run only your new backend's tests
.venv/bin/python -m pytest tests/unit/test_wenet_backend.py -q

# Run all unit tests (verify you didn't break anything)
make test-unit
```

### Step 5 Checklist

- [ ] Test file created at `tests/unit/test_<engine>_backend.py`
- [ ] Architecture test: `backend.architecture == STTArchitecture.XXX`
- [ ] Capabilities test: all `EngineCapabilities` fields verified
- [ ] Load test (success): mock engine library, verify model loads
- [ ] Load test (not installed): engine library set to `None`, verify `ModelLoadError`
- [ ] Transcribe file test: verify `BatchResult` fields
- [ ] Transcribe file test (empty audio): verify `AudioFormatError`
- [ ] Transcribe stream test: verify partial and final `TranscriptSegment`s
- [ ] Health test: both loaded and not-loaded states
- [ ] Unload test: verify model reference is cleared
- [ ] All tests pass: `make test-unit`
- [ ] Type checking passes: `make check`
- [ ] Manifest fixture exists at `tests/fixtures/manifests/valid_stt_<engine>.yaml`

---

## Architecture-Specific Behavior

The runtime adapts the streaming pipeline based on the `architecture` field. This happens automatically -- you do not need to modify the runtime. But you need to understand what happens so your backend's behavior matches expectations.

### encoder-decoder (e.g., Whisper)

```
Audio -> Preprocessing -> Ring Buffer -> Accumulate window (3-5s)
                                                |
                                                v
                                     Engine.transcribe_stream()
                                                |
                                                v
                                     LocalAgreement compares passes
                                                |
                                 Confirmed tokens -> transcript.partial
                                 VAD silence -> transcript.final
```

- Runtime uses **LocalAgreement** to confirm partial transcripts by comparing consecutive passes.
- **Cross-segment context**: last 224 tokens of the previous `transcript.final` are sent as `initial_prompt` for the next segment.
- **Hot words** (when engine does NOT support native boosting): injected into `initial_prompt` as `"Termos: PIX, TED, Selic."`.

### ctc (e.g., WeNet)

```
Audio -> Preprocessing -> Ring Buffer -> Engine.transcribe_stream()
                                                |
                                    Native partial + final segments
```

- **No LocalAgreement**. The engine produces partials natively, frame-by-frame.
- **No cross-segment context**. CTC does not support `initial_prompt` conditioning.
- **Hot words**: sent via the gRPC `hot_words` field for native keyword boosting (when `capabilities.hot_words: true` in manifest).

### streaming-native (e.g., Paraformer)

```
Audio -> Preprocessing -> Engine.transcribe_stream()
                                    |
                         Engine manages internal state
                         Native partial + final segments
```

- Similar to CTC: native partials, no LocalAgreement.
- Engine manages its own internal streaming state.
- Cross-segment context and initial_prompt support depends on the specific engine.

### Where the adaptation happens in code

The runtime reads the manifest's `architecture` and `hot_words` fields in `src/theo/server/routes/realtime.py`:

```python
model_architecture = manifest.capabilities.architecture or STTArchitecture.ENCODER_DECODER
model_supports_hot_words = manifest.capabilities.hot_words or False
```

These are passed to `StreamingSession` in `src/theo/session/streaming.py`, which uses them in two places:

1. **`_build_initial_prompt()`**: Skips cross-segment context for CTC. Only injects hot words into prompt when engine does NOT support native boosting.

2. **`_receive_worker_events()`**: Skips cross-segment context update after `transcript.final` for CTC.

---

## FAQ

### 1. Do I need to modify the runtime core?

**No.** Adding a new engine requires changes to exactly 4 files (backend implementation, factory registration, manifest, pyproject.toml) plus tests. The runtime (API Server, Session Manager, Scheduler, Preprocessing, Post-processing, WebSocket protocol, gRPC proto) remains untouched.

This is by design. The `STTBackend` interface is the extension point. The runtime interacts with engines exclusively through this interface.

### 2. How does streaming differ by architecture?

The key difference is **who produces partial transcripts**:

| Architecture | Partials | Controlled by |
|---|---|---|
| `encoder-decoder` | Synthetic (LocalAgreement) | Runtime |
| `ctc` | Native (frame-by-frame) | Engine |
| `streaming-native` | Native (engine state) | Engine |

For `encoder-decoder`, the runtime accumulates audio in windows and uses LocalAgreement to compare consecutive inference passes. Your `transcribe_stream()` just needs to yield `TranscriptSegment` objects -- the runtime handles the rest.

For `ctc` and `streaming-native`, your `transcribe_stream()` should yield partial segments (`is_final=False`) as the engine produces them incrementally, and a final segment (`is_final=True`) when the stream ends (empty chunk).

### 3. What if my engine doesn't support hot words?

Set `capabilities.hot_words: false` in the manifest and `supports_hot_words=False` in `EngineCapabilities`. The runtime will automatically inject hot words into the `initial_prompt` parameter as a semantic workaround (e.g., `"Termos: PIX, TED, Selic."`).

If your engine also does not support `initial_prompt` (like CTC), the hot words will be silently skipped. This is expected behavior -- not all engines can boost keywords.

### 4. How do I test without the real model?

Mock the engine library at the module level. Every backend guards the engine import:

```python
try:
    import wenet as wenet_lib
except ImportError:
    wenet_lib = None
```

In tests, replace the module-level reference:

```python
import theo.workers.stt.wenet as wenet_mod
wenet_mod.wenet_lib = mock_wenet  # MagicMock
```

For `transcribe_file` and `transcribe_stream` tests, set `backend._model` to a `MagicMock` and configure its return values to simulate engine output. See `tests/unit/test_wenet_backend.py` for the complete pattern.

### 5. What about GPU/CUDA isolation?

Each worker runs as a **separate subprocess** with its own CUDA context. This is handled by the Worker Manager, not by your backend. Your backend just needs to:

- Accept a `device` parameter in `engine_config` (e.g., `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`)
- Pass it to the engine library's model loading function
- Not worry about sharing GPU memory with other workers

If the worker crashes (GPU OOM, segfault, etc.), the runtime detects it via gRPC stream break and can restart the worker automatically. Your backend does not need crash recovery logic -- that's the Session Manager's job.

### 6. What types should I return from transcribe_file?

`BatchResult` is a frozen dataclass defined in `src/theo/_types.py`:

```python
@dataclass(frozen=True, slots=True)
class BatchResult:
    text: str                                # Full transcription text
    language: str                            # Detected or specified language
    duration: float                          # Audio duration in seconds
    segments: tuple[SegmentDetail, ...]      # Segment-level details
    words: tuple[WordTimestamp, ...] | None   # Word timestamps (if requested)
```

`SegmentDetail` includes `id`, `start`, `end`, `text`, and optional fields like `avg_logprob`, `no_speech_prob`. Fill in what your engine provides; leave defaults for what it doesn't.

### 7. My engine has a capability not covered by EngineCapabilities. What do I do?

`EngineCapabilities` covers the capabilities the runtime actively uses: hot words, initial_prompt, batch, word timestamps, max concurrent sessions. If your engine has a unique capability (e.g., speaker diarization, emotion detection), it does not affect the runtime pipeline today.

Do NOT add fields to `EngineCapabilities` speculatively -- that violates YAGNI. Document the capability in the manifest's `description` field. When the runtime needs to use that capability, `EngineCapabilities` will be extended with a concrete use case.

### 8. Can I add a new STTArchitecture enum value?

Yes, if your engine has a fundamentally different streaming model. Add the new value to the `STTArchitecture` enum in `src/theo/_types.py`:

```python
class STTArchitecture(Enum):
    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"
    MY_NEW_ARCH = "my-new-arch"           # only if truly different
```

However, think carefully before doing this. The three existing architectures cover most STT models. `streaming-native` is intentionally generic for engines that manage their own state. Only add a new value if the runtime needs to behave differently for your architecture in a way that none of the existing values capture.

### 9. How do I verify end-to-end integration?

After completing Steps 1-5, verify the full pipeline:

```bash
# 1. Start the runtime with your model installed
theo serve

# 2. Batch transcription via REST
curl -F file=@audio.wav -F model=<your-model-name> \
  http://localhost:8000/v1/audio/transcriptions

# 3. Streaming via WebSocket (using wscat or similar)
wscat -c "ws://localhost:8000/v1/realtime?model=<your-model-name>"
# Send binary audio frames, observe transcript events

# 4. Verify same contract as Faster-Whisper
curl -F file=@audio.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# Both should return the same JSON structure
```

The response format is identical regardless of engine -- that's the whole point of the `STTBackend` abstraction.

---

## Summary

| Step | What | Where | Lines of Code |
|------|------|-------|---------------|
| 1 | Implement `STTBackend` | `src/theo/workers/stt/<engine>.py` | ~200-400 |
| 2 | Create manifest | `models/<model>/theo.yaml` | ~30 |
| 3 | Register factory | `src/theo/workers/stt/main.py` | ~3 |
| 4 | Declare dependency | `pyproject.toml` | ~2-4 |
| 5 | Write tests | `tests/unit/test_<engine>_backend.py` | ~150-300 |

Total: approximately 400-700 lines of new code, zero changes to the runtime core.
