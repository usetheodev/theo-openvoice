# Full-Duplex STT + TTS -- Developer Integration Guide

**Audience**: Developers integrating Theo OpenVoice full-duplex (STT + TTS) into voice agents.
**Prerequisite**: Familiarity with the WebSocket protocol from M5 (`docs/ARCHITECTURE.md` section 7).
**Status**: M9 (Full-Duplex) -- design finalized, implementation in progress.

---

## 1. Overview

Theo OpenVoice supports **full-duplex voice**: STT and TTS operate simultaneously on the same WebSocket connection (`/v1/realtime`). A single connection handles both directions -- the client sends audio frames for speech-to-text and `tts.speak` commands for text-to-speech; the server responds with transcription events and synthesized audio.

### Single Connection Model

```
+------------------+                          +-------------------+
|     CLIENT       |                          |   THEO RUNTIME    |
|                  |                          |                   |
|  Audio frames    | ----(binary)-----------> |  STT pipeline     |
|  (user speech)   |                          |  (preprocess,     |
|                  |                          |   VAD, worker)    |
|                  |                          |                   |
|  tts.speak cmd   | ----(JSON)-------------> |  TTS pipeline     |
|                  |                          |  (worker, synth)  |
|                  |                          |                   |
|                  | <---(JSON)-------------- |  transcript.*     |
|                  |                          |  vad.*            |
|                  |                          |  tts.*            |
|                  |                          |                   |
|                  | <---(binary)------------ |  TTS audio        |
|  (bot speech)    |                          |  (synthesized)    |
+------------------+                          +-------------------+
```

**Direction is unambiguous**: client-to-server binary messages are STT audio input; server-to-client binary messages are TTS audio output.

### Mute-on-Speak

When TTS is active, STT is automatically muted to prevent the runtime from transcribing the bot's own voice. Audio frames from the client are discarded while TTS is speaking. This eliminates feedback loops without requiring external Acoustic Echo Cancellation (AEC).

### What Theo Provides vs. What the Client Provides

| Theo's Responsibility | Client's Responsibility |
|---|---|
| STT transcription (partial + final) | LLM / dialogue management |
| TTS synthesis (streaming audio) | Sending LLM response to `tts.speak` |
| Mute-on-speak coordination | Audio capture (microphone) |
| VAD (voice activity detection) | Audio playback (speaker) |
| Preprocessing (resample, normalize) | AEC if barge-in is needed |
| Session management and recovery | Turn-taking logic |

---

## 2. WebSocket Protocol (STT + TTS)

### Connection

```
GET /v1/realtime?model=faster-whisper-large-v3&language=pt
Upgrade: websocket
```

The `model` parameter specifies the STT model. The TTS model is configured separately via `session.configure` or the `tts.speak` command.

### Client -> Server Messages

#### Audio Frames (Binary)

Raw PCM 16-bit audio at any sample rate (the runtime resamples to 16kHz automatically).

- **Recommended frame size**: 20ms or 40ms
- **Max message size**: 64KB (~2s of PCM 16kHz mono)
- **Send as**: binary WebSocket messages

#### session.configure

Configures both STT and TTS parameters for the session.

```json
{
  "type": "session.configure",
  "vad_sensitivity": "normal",
  "language": "pt",
  "hot_words": ["PIX", "TED"],
  "enable_itn": true,
  "model_tts": "kokoro-v1",
  "preprocessing": {
    "denoise": false
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `vad_sensitivity` | string | `"normal"` | VAD sensitivity: `high`, `normal`, `low` |
| `language` | string | auto-detect | ISO 639-1 code or `"mixed"` for code-switching |
| `hot_words` | array | `[]` | Domain-specific keywords for boosting |
| `enable_itn` | bool | `true` | Apply Inverse Text Normalization to final transcripts |
| `model_tts` | string | none | TTS model for `tts.speak` commands in this session |
| `preprocessing` | object | defaults | Preprocessing overrides (e.g., `denoise`) |

#### tts.speak

Triggers TTS synthesis. STT is muted automatically when synthesis starts.

```json
{
  "type": "tts.speak",
  "text": "Ola, como posso ajudar?",
  "voice": "default",
  "request_id": "req_abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Plain text to synthesize (no SSML) |
| `voice` | string | no | Voice identifier (default: `"default"`) |
| `request_id` | string | no | Client-generated ID. Auto-generated if omitted |
| `model` | string | no | TTS model override. Falls back to `session.configure.model_tts` |

#### tts.cancel

Cancels active TTS synthesis. STT unmutes immediately.

```json
{
  "type": "tts.cancel",
  "request_id": "req_abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `request_id` | string | no | Cancels specific request. If omitted, cancels any active TTS |

#### Other Commands (unchanged from STT-only protocol)

| Command | Description |
|---|---|
| `session.cancel` | Cancels STT session |
| `input_audio_buffer.commit` | Forces commit of current STT segment |
| `session.close` | Closes session gracefully |

### Server -> Client Messages

#### tts.speaking_start

Emitted when TTS starts producing audio. STT is muted at this point.

```json
{
  "type": "tts.speaking_start",
  "request_id": "req_abc123",
  "timestamp_ms": 1500
}
```

#### TTS Audio (Binary)

Synthesized audio sent as binary WebSocket messages (PCM 16-bit, 16kHz, mono). The client should play these frames through the speaker.

Binary frames from server to client are always TTS audio. There is no ambiguity -- the protocol uses message direction to distinguish STT input audio (client -> server) from TTS output audio (server -> client).

#### tts.speaking_end

Emitted when TTS finishes producing audio. STT unmutes at this point.

```json
{
  "type": "tts.speaking_end",
  "request_id": "req_abc123",
  "timestamp_ms": 2500,
  "duration_ms": 1000,
  "cancelled": false
}
```

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Matches the `tts.speak` request |
| `timestamp_ms` | int | Session-relative timestamp |
| `duration_ms` | int | Total duration of synthesized audio |
| `cancelled` | bool | `true` if TTS was interrupted by `tts.cancel` or a new `tts.speak` |

#### STT Events (unchanged from STT-only protocol)

| Event | Description |
|---|---|
| `session.created` | Session established, includes config |
| `vad.speech_start` | VAD detected speech onset |
| `transcript.partial` | Intermediate transcription hypothesis |
| `transcript.final` | Confirmed transcription segment (with ITN) |
| `vad.speech_end` | VAD detected speech offset |
| `session.hold` | Session transitioned to HOLD (prolonged silence) |
| `session.rate_limit` | Backpressure: client sending faster than real-time |
| `session.frames_dropped` | Frames discarded due to backlog > 10s |
| `error` | Error with `recoverable` flag |
| `session.closed` | Session terminated |

---

## 3. Full-Duplex Flow (Typical Agent Interaction)

The following sequence shows a complete voice agent interaction. The LLM is external to Theo -- the client is responsible for sending the user's transcript to the LLM and the LLM's response to TTS.

```
Client                     Theo Runtime                STT Worker    TTS Worker
  |                             |                           |             |
  |  WS /v1/realtime            |                           |             |
  |  ?model=fw-large-v3         |                           |             |
  |---------------------------->|                           |             |
  |                             |                           |             |
  |  <-- session.created        |                           |             |
  |  {session_id, config}       |                           |             |
  |                             |                           |             |
  |  session.configure          |                           |             |
  |  {model_tts: "kokoro-v1"}   |                           |             |
  |---------------------------->|                           |             |
  |                             |                           |             |
  |  === USER SPEAKS =========================================           |
  |                             |                           |             |
  |  audio frames (binary)      |                           |             |
  |-----(PCM 16-bit)----------->| preprocess -> VAD         |             |
  |                             |-----(gRPC stream)-------->|             |
  |                             |                           |             |
  |  <-- vad.speech_start       |                           |             |
  |                             |                           |             |
  |                             |  <-- TranscriptEvent      |             |
  |  <-- transcript.partial     |      (partial)            |             |
  |  {text: "qual o saldo"}     |                           |             |
  |                             |                           |             |
  |  (VAD detects silence)      |                           |             |
  |                             |  <-- TranscriptEvent      |             |
  |  <-- transcript.final       |      (final)              |             |
  |  {text: "Qual o saldo      |                           |             |
  |   da minha conta?"}         |                           |             |
  |                             |                           |             |
  |  <-- vad.speech_end         |                           |             |
  |                             |                           |             |
  |  === CLIENT CALLS LLM (external, not Theo) =========================|
  |                             |                           |             |
  |  (client sends transcript   |                           |             |
  |   to LLM, receives reply)   |                           |             |
  |                             |                           |             |
  |  === BOT SPEAKS (TTS) =======================================        |
  |                             |                           |             |
  |  tts.speak                  |                           |             |
  |  {text: "Seu saldo e       |                           |             |
  |   R$2.500,00"}              |                           |             |
  |---------------------------->|                           |             |
  |                             |                           |             |
  |                             |----(gRPC Synthesize)----->|             |
  |                             |                           |             |
  |                             |  <-- first audio chunk    |             |
  |                             |                           |             |
  |                             |  [1] MUTE STT             |             |
  |  <-- tts.speaking_start     |                           |             |
  |                             |                           |             |
  |  <-- binary audio frames    |  <-- audio chunks         |             |
  |  (TTS output, play on       |                           |             |
  |   speaker)                  |                           |             |
  |                             |                           |             |
  |  audio frames from client   |                           |             |
  |-----(still sending)-------->|  DISCARDED (muted)        |             |
  |                             |                           |             |
  |                             |  <-- last audio chunk     |             |
  |                             |                           |             |
  |                             |  [2] UNMUTE STT           |             |
  |  <-- tts.speaking_end       |                           |             |
  |  {duration_ms: 1800,        |                           |             |
  |   cancelled: false}         |                           |             |
  |                             |                           |             |
  |  === USER SPEAKS AGAIN ======================================        |
  |                             |                           |             |
  |  audio frames (binary)      |                           |             |
  |-----(PCM 16-bit)----------->| preprocess -> VAD         |             |
  |                             |-----(gRPC stream)-------->|             |
  |                             |                           |             |
  |  <-- vad.speech_start       |                           |             |
  |  <-- transcript.partial     |                           |             |
  |  <-- transcript.final       |                           |             |
  |  <-- vad.speech_end         |                           |             |
  |                             |                           |             |
  |  (cycle repeats)            |                           |             |
```

### Key Points in the Flow

1. **Steps [1] and [2]** are the mute-on-speak mechanism. STT is muted BEFORE the first audio byte reaches the client, and unmuted AFTER the last byte is sent.

2. **Audio frames from the client continue arriving during TTS**. The runtime discards them silently. The client does not need to stop sending audio.

3. **The LLM call is the client's responsibility**. Theo provides STT and TTS. The dialogue management, intent detection, and response generation happen outside the runtime.

4. **Multiple turns** follow the same pattern: user speaks -> STT transcribes -> client calls LLM -> client sends `tts.speak` -> TTS synthesizes -> repeat.

---

## 4. Mute-on-Speak

### How It Works

Mute-on-speak prevents the STT pipeline from transcribing the bot's own voice (TTS audio played through the speaker). Without this mechanism, the microphone would pick up the TTS output, and the STT would transcribe it, creating a feedback loop.

```
Normal state (STT active):
  audio frames -> preprocessing -> VAD -> worker -> transcript.*

tts.speak received:
  1. Runtime sends text to TTS worker
  2. TTS worker starts synthesizing
  3. First audio chunk ready:
     a. MUTE STT (before audio reaches client)
     b. Emit tts.speaking_start
  4. Audio chunks sent to client as binary frames
  5. Last chunk sent:
     a. Emit tts.speaking_end
     b. UNMUTE STT (in finally block)

Muted state (STT paused):
  audio frames -> DISCARDED
  (no preprocessing, no VAD, no worker, no ring buffer writes)
```

### Mute Timing

The mute activates BEFORE the first TTS audio byte is sent to the client. This eliminates the window where the microphone could pick up TTS audio while STT is still active. The sequence is:

1. Mute STT
2. Send `tts.speaking_start` event
3. Send TTS audio frames

### Unmute Guarantees

Unmute happens in a `finally` block, guaranteeing it executes even when:

- TTS worker crashes mid-synthesis
- WebSocket disconnects during TTS
- `tts.cancel` is received
- Any unexpected error occurs

```
try:
    mute_controller.mute()
    # ... stream TTS audio ...
finally:
    mute_controller.unmute()  # always executes
```

### What Happens to Client Audio During Mute

Audio frames from the client are discarded at the `StreamingSession.process_frame()` level. They do not reach:

- Preprocessing pipeline
- VAD detector
- Ring buffer
- STT worker

The session state machine (ACTIVE, SILENCE, etc.) does not change during mute. When unmuted, the session resumes from its previous state, and new audio is processed normally.

### Cancelling TTS

`tts.cancel` immediately:

1. Cancels the gRPC stream to the TTS worker
2. Unmutes STT
3. Emits `tts.speaking_end` with `cancelled: true`

STT resumes processing audio frames immediately after cancel.

### Monitoring Mute

The `theo_stt_muted_frames_total` Prometheus counter tracks how many audio frames were discarded due to mute-on-speak. A persistently high value may indicate:

- TTS is speaking too frequently (adjust dialogue logic)
- TTS synthesis is slow (check `theo_tts_synthesis_duration_seconds`)

---

## 5. REST TTS Endpoint

For batch TTS (non-streaming, non-full-duplex), use the REST endpoint:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Ola, como posso ajudar?", "voice": "default"}' \
  --output speech.wav
```

### Request (JSON body)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | -- | TTS model name from registry |
| `input` | string | yes | -- | Plain text to synthesize |
| `voice` | string | no | `"default"` | Voice identifier |
| `response_format` | string | no | `"wav"` | Output format: `wav`, `pcm` |
| `speed` | float | no | `1.0` | Speech rate: 0.25 to 4.0 |

### Response

Binary audio in the requested format.

| Format | Content-Type | Description |
|---|---|---|
| `wav` | `audio/wav` | WAV file with headers (default) |
| `pcm` | `audio/pcm` | Raw PCM 16-bit, 16kHz, mono |

### Errors

| Status | Meaning |
|---|---|
| `400` | Empty input text or invalid parameters |
| `404` | TTS model not found in registry |
| `503` | TTS model loading (cold start). Includes `Retry-After` header |

### REST vs. WebSocket TTS

| | REST (`POST /v1/audio/speech`) | WebSocket (`tts.speak`) |
|---|---|---|
| Use case | Batch synthesis, pre-generated audio | Real-time voice agents |
| Response | Complete audio in body | Streaming audio frames |
| Latency | Higher (waits for full synthesis) | Lower (streams first chunk immediately) |
| STT coordination | None (separate endpoint) | Automatic mute-on-speak |
| Connection | One request, one response | Persistent, bidirectional |

---

## 6. Limitations

### No Barge-In

Mute-on-speak means the user cannot interrupt the bot mid-speech. Audio from the client is discarded while TTS is active. For barge-in (user interrupts the bot), external Acoustic Echo Cancellation (AEC) is required to separate the user's voice from the TTS audio playing through the speaker.

**Workaround**: The client can send `tts.cancel` to stop TTS programmatically (e.g., based on a button press or external signal), but this is not voice-triggered barge-in.

### No SSML

TTS accepts plain text only. The `text` field in `tts.speak` does not support SSML markup. SSML support is planned for a future release.

### No Streaming TTS via REST

`POST /v1/audio/speech` returns the complete synthesized audio in the response body. It does not support chunked transfer encoding or streaming. For streaming TTS, use the WebSocket protocol (`tts.speak`).

### Single TTS at a Time

Only one TTS synthesis can be active per session at a time. If a new `tts.speak` command arrives while TTS is already active, the previous synthesis is cancelled automatically (the client receives `tts.speaking_end` with `cancelled: true`), and the new synthesis begins.

### LLM is External

Theo provides STT and TTS. The LLM (language model), dialogue management, intent detection, and response generation are the client's responsibility. The typical integration pattern is:

```
User speaks -> Theo STT -> transcript.final -> Client sends to LLM
                                                      |
Client receives LLM response -> tts.speak -> Theo TTS -> Bot speaks
```

Theo does not call any LLM. The client orchestrates the STT -> LLM -> TTS pipeline.

---

## 7. Prometheus Metrics (TTS)

TTS metrics follow the same pattern as STT metrics: lazy import of `prometheus_client`, no-op when Prometheus is not installed.

| Metric | Type | Description |
|---|---|---|
| `theo_tts_ttfb_seconds` | Histogram | Time from `tts.speak` received to first audio chunk sent to client |
| `theo_tts_synthesis_duration_seconds` | Histogram | Total TTS synthesis duration (first chunk to last chunk) |
| `theo_tts_requests_total` | Counter | TTS requests by status label: `ok`, `error`, `cancelled` |
| `theo_tts_active_sessions` | Gauge | Number of sessions with active TTS synthesis |
| `theo_stt_muted_frames_total` | Counter | STT audio frames discarded during mute-on-speak |

### Existing STT Metrics (for reference)

| Metric | Type | Description |
|---|---|---|
| `theo_stt_ttfb_seconds` | Histogram | Time to first partial transcript |
| `theo_stt_final_delay_seconds` | Histogram | Delay of final transcript after end of speech |
| `theo_stt_active_sessions` | Gauge | Active STT streaming sessions |
| `theo_stt_vad_events_total` | Counter | VAD events (speech_start, speech_end) |

### V2V Composite Metric

The runtime exposes `theo_v2v_runtime_latency_seconds`, a composite metric combining:

```
theo_v2v_runtime_latency_seconds = stt_final_delay + tts_ttfb
```

This represents Theo's contribution to the Voice-to-Voice latency. The client's LLM latency is external and not measured by Theo.

---

## 8. V2V Latency Budget

The PRD defines a target of 300ms for end-to-end Voice-to-Voice latency. This budget is shared between Theo and the client's LLM:

```
+-----------------------------------------------------------+
|          LATENCY BUDGET: 300ms total V2V                   |
|                                                            |
|   VAD End-of-Speech ............... 50ms   (Theo)          |
|   ASR Final Transcript ............ 100ms  (Theo)          |
|   LLM Time to First Token ......... 100ms  (External)     |
|   TTS Time to First Byte ........... 50ms  (Theo)          |
|                                     -----                  |
|                              TOTAL: 300ms                  |
+-----------------------------------------------------------+
```

### Theo's Contribution

Theo controls two components of the V2V budget:

| Component | Target | Metric |
|---|---|---|
| ASR final_delay (VAD end + final transcript) | ~100ms | `theo_stt_final_delay_seconds` |
| TTS TTFB (text received to first audio byte) | ~50ms | `theo_tts_ttfb_seconds` |
| **Theo total** | **~150ms** | `theo_v2v_runtime_latency_seconds` |

The remaining ~150ms is the LLM's time to first token, which the client controls.

### Measuring V2V in Practice

```
Total V2V = theo_stt_final_delay + client_llm_latency + theo_tts_ttfb
```

- Monitor `theo_v2v_runtime_latency_seconds` to track Theo's portion.
- If V2V exceeds 300ms, check each component independently:
  - `theo_stt_final_delay_seconds` > 100ms: STT or VAD is the bottleneck.
  - `theo_tts_ttfb_seconds` > 50ms: TTS worker or model loading is the bottleneck.
  - Both within budget: the LLM is the bottleneck.

### Realistic Expectations

The 300ms target is aspirational. In practice:

- 150ms for Theo (ASR + TTS) is achievable with optimized models (e.g., Distil-Whisper + Kokoro on GPU).
- 100ms for LLM TTFT depends on the model, provider, and infrastructure.
- 500ms is a more realistic initial target for the full V2V pipeline.

---

## Appendix: Complete Event Reference

### Client -> Server

| Message | Type | Description |
|---|---|---|
| (audio frames) | Binary | PCM 16-bit, any sample rate, 20-40ms frames |
| `session.configure` | JSON | Configure STT, TTS, VAD, preprocessing |
| `tts.speak` | JSON | Trigger TTS synthesis |
| `tts.cancel` | JSON | Cancel active TTS |
| `session.cancel` | JSON | Cancel STT session |
| `input_audio_buffer.commit` | JSON | Force commit STT segment |
| `session.close` | JSON | Close session gracefully |

### Server -> Client

| Message | Type | Description |
|---|---|---|
| `session.created` | JSON | Session established with config |
| `vad.speech_start` | JSON | VAD detected speech onset |
| `transcript.partial` | JSON | Intermediate STT hypothesis |
| `transcript.final` | JSON | Confirmed STT segment (with ITN) |
| `vad.speech_end` | JSON | VAD detected speech offset |
| `tts.speaking_start` | JSON | TTS started, STT muted |
| (TTS audio) | Binary | Synthesized audio frames (PCM 16-bit) |
| `tts.speaking_end` | JSON | TTS finished, STT unmuted |
| `session.hold` | JSON | Session in HOLD (prolonged silence) |
| `session.rate_limit` | JSON | Backpressure warning |
| `session.frames_dropped` | JSON | Frames dropped due to backlog |
| `error` | JSON | Error with `recoverable` flag |
| `session.closed` | JSON | Session terminated |

---

*This document is part of M9 (Full-Duplex). See also: `docs/ARCHITECTURE.md` for system architecture, `docs/PRD.md` section 18 (Roadmap, Fase 3) for product requirements, `docs/ROADMAP.md` for milestone details.*
