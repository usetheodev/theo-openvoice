# Theo OpenVoice — Documento de Arquitetura

**Versão**: 2.0
**Base**: PRD v2.1
**Status**: Implementado. M1-M9 completos, todas as 3 fases do PRD entregues.

---

## 1. O Que É o Theo OpenVoice

Theo OpenVoice é um **runtime unificado de voz** (STT + TTS) construído do zero em Python. Ele orquestra engines de inferência existentes (Faster-Whisper, Silero VAD, Kokoro, Piper) dentro de um único binário com API compatível com OpenAI e CLI inspirado no Ollama.

**O Theo NÃO é:**
- Fork ou wrapper de projetos existentes (Speaches, Whisper.cpp, LocalAI)
- Uma engine de inferência (não treina modelos, não faz decode)
- Um substituto para PBX (não faz SIP signaling)

**O Theo É:**
- A **camada de runtime** que falta entre as engines de inferência e a produção
- Orquestrador de: session management, preprocessing, post-processing, scheduling, observabilidade, CLI

```
┌─────────────────────────────────────────────────────────────────┐
│                        O QUE O THEO FAZ                         │
│                                                                 │
│   Engine de Inferência ──→ [ THEO RUNTIME ] ──→ Produção        │
│   (Faster-Whisper,         (orquestra,          (API, CLI,      │
│    Kokoro, Piper)           gerencia,            WebSocket,     │
│                             preprocessa,         WebSocket)     │
│                             pós-processa)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Princípio Arquitetural

> **Um binário, um processo, dois tipos de worker.**

Não existe "Theo STT" e "Theo TTS" como produtos separados. Existe **Theo OpenVoice** com capacidades de STT e TTS, habilitadas por quais modelos estão instalados.

```
theo pull faster-whisper-large-v3   → habilita STT
theo pull kokoro-v1                 → habilita TTS
theo serve                          → serve ambos
```

---

## 3. Visão Geral da Arquitetura

### 3.1 Diagrama Principal

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                        CLIENTES                                     │
│   CLI (theo transcribe)  │  REST (curl/SDK)  │  WebSocket           │
│                                                                     │
└──────────────┬────────────┬────────────────────┬──────────┬─────────┘
               │            │                    │          │
               ▼            ▼                    ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     API SERVER (FastAPI)                             │
│                                                                     │
│  POST /v1/audio/transcriptions    (STT batch)                       │
│  POST /v1/audio/translations      (STT tradução)                    │
│  POST /v1/audio/speech            (TTS)                             │
│  WS   /v1/realtime                (STT+TTS full-duplex)            │
│  GET  /health                     (health check)                    │
│  GET  /metrics                    (Prometheus)                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         SCHEDULER                                   │
│                                                                     │
│  • Roteia requests para workers corretos (STT ou TTS)               │
│  • Priorização: realtime (WebSocket) > batch (file)                 │
│  • Cancelamento em ≤50ms                                            │
│  • Orçamento de latência por sessão                                 │
│  • [Fase 3] Dynamic batching                                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                       MODEL REGISTRY                                │
│                                                                     │
│  • Manifesto declarativo (theo.yaml) para STT e TTS                 │
│  • Download sob demanda (theo pull <model>)                         │
│  • Lifecycle: load, evict, unload                                   │
│  • Campo "type" (stt/tts) diferencia modelos                        │
│  • Campo "architecture" informa pipeline adaptativo                 │
│                                                                     │
├──────────────────┬──────────────────────────────────────────────────┤
│                  │                                                   │
│  TTS WORKERS     │              STT WORKERS                          │
│  (subprocessos   │              (subprocessos gRPC)                  │
│   gRPC)          │                                                   │
│                  │  ┌─────────────────────────────────────────────┐  │
│  ┌────────────┐  │  │  Faster-Whisper  (encoder-decoder)          │  │
│  │ Kokoro     │  │  │  WeNet           (CTC) [Fase 2]            │  │
│  │ Piper      │  │  │  Paraformer      (streaming-native) [Fut.] │  │
│  └────────────┘  │  └─────────────────────────────────────────────┘  │
│                  │                                                   │
├──────────────────┴──────────────────────────────────────────────────┤
│              AUDIO PREPROCESSING PIPELINE [v2]                      │
│                                                                     │
│  Ingestão → Resample → DC Remove → Gain Normalize → [Denoise]      │
│             (soxr)     (HPF 20Hz)   (-3dBFS)         (RNNoise)      │
│                                                                     │
│  Input: qualquer sample rate/formato                                │
│  Output: PCM 16-bit, 16kHz, mono, normalizado                      │
│  Custo: <5ms/frame                                                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              POST-PROCESSING PIPELINE [v2]                          │
│                                                                     │
│  Engine Output → ITN → Entity Formatting → Hot Word Correction      │
│                  (NeMo)  (Theo original)    (Theo original)         │
│                                                                     │
│  Input: texto cru da engine                                         │
│  Output: texto formatado ("dois mil" → "2000")                      │
│  Custo: <10ms/segmento                                              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              SESSION MANAGER (apenas STT)                            │
│                                                                     │
│  • 6 estados: INIT → ACTIVE → SILENCE → HOLD → CLOSING → CLOSED    │
│  • Ring Buffer: 60s pré-alocado, read fence, force commit           │
│  • VAD: Silero VAD + energy pre-filter                              │
│  • Recovery: WAL in-memory, retomada sem duplicação                 │
│  • LocalAgreement: partial transcripts para encoder-decoder         │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              OBSERVABILIDADE                                         │
│                                                                     │
│  • Prometheus metrics (/metrics)                                    │
│  • Health check (/health)                                           │
│  • Métricas STT: TTFB, final_delay, sessões ativas, VAD events     │
│  • Métricas de qualidade: confidence_avg, no_speech, hot_word_hits  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 O Que É Compartilhado vs. Específico

| Componente                | Compartilhado STT+TTS | Notas                                    |
|---------------------------|:---------------------:|------------------------------------------|
| API Server (FastAPI)      | ✅                    | Endpoints STT e TTS no mesmo processo    |
| Model Registry            | ✅                    | Mesmo `theo.yaml`, mesmo lifecycle       |
| Scheduler                 | ✅                    | Prioridade por tipo (realtime > batch)   |
| CLI                       | ✅                    | `theo pull/serve/list` servem ambos      |
| Observabilidade           | ✅                    | Mesmas métricas Prometheus               |
| Docker image              | ✅                    | Uma imagem, engines habilitadas por config|
| Audio Preprocessing       | ✅                    | Pipeline compartilhado, stages toggle    |
| Session Manager           | ❌ (só STT)           | TTS é stateless por request              |
| Post-Processing           | ❌ (só STT)           | ITN, entity formatting                   |
| Workers                   | ❌                    | Engines e protobufs diferentes           |

---

## 4. Componentes em Detalhe

### 4.1 API Server (FastAPI)

Processo principal do Theo. Recebe todas as requests e delega para os subsistemas.

**Endpoints:**

| Endpoint                         | Método    | Fase | Descrição                          |
|----------------------------------|-----------|------|------------------------------------|
| `/v1/audio/transcriptions`       | POST      | 1    | Transcrição de arquivo (batch)     |
| `/v1/audio/translations`         | POST      | 1    | Tradução para inglês (batch)       |
| `/v1/audio/speech`               | POST      | 1    | Síntese de voz (TTS)              |
| `/v1/realtime`                   | WebSocket | 2+3  | STT+TTS full-duplex                |
| `/health`                        | GET       | 1    | Health check                       |
| `/metrics`                       | GET       | 1    | Métricas Prometheus                |

**Compatibilidade**: Segue o contrato da OpenAI Audio API para que clientes existentes (SDKs OpenAI, etc.) funcionem sem modificação.

### 4.2 Scheduler

Decide **qual worker** atende cada request e com **qual prioridade**.

```
Request chega
    │
    ▼
┌──────────────────┐
│ Model Registry   │ ──→ Qual engine/worker serve este modelo?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Scheduler        │ ──→ Tem worker disponível?
│                  │     • SIM → roteia
│                  │     • NÃO → enfileira (com prioridade)
│                  │
│  Prioridade:     │
│  1. Streaming    │     (WebSocket = tempo real)
│  2. Batch        │     (upload de arquivo = tolerante a delay)
└──────────────────┘
```

**Cancelamento**: Propaga cancel em ≤50ms via gRPC ao worker.

**[Fase 3] Dynamic Batching**: Acumula requests por até 50ms, faz batch inference, distribui resultados.

### 4.3 Model Registry

Gerencia o **ciclo de vida** dos modelos (STT e TTS).

**Manifesto (`theo.yaml`) — exemplo STT:**

```yaml
name: faster-whisper-large-v3
version: 3.0.0
engine: faster-whisper
type: stt                              # ← diferencia de TTS
architecture: encoder-decoder          # ← informa pipeline adaptativo

capabilities:
  streaming: true
  languages: ["auto", "en", "pt", "es"]
  word_timestamps: true
  translation: true
  partial_transcripts: true
  hot_words: false                     # engine suporta keyword boosting?
  batch_inference: true                # suporta batch mode?

resources:
  memory_mb: 3072
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 8

engine_config:
  model_size: "large-v3"
  compute_type: "float16"
  device: "auto"
  beam_size: 5
  vad_filter: false                    # VAD é feito no runtime, não na engine
```

**Campo `architecture`** — é o que torna o runtime model-agnostic:

| Architecture        | Exemplos               | Como o runtime trata                         |
|---------------------|------------------------|----------------------------------------------|
| `encoder-decoder`   | Whisper, Distil-Whisper| Acumula windows, LocalAgreement para partials|
| `ctc`               | WeNet CTC, Wav2Vec2    | Frame-by-frame, partials nativos             |
| `streaming-native`  | Paraformer             | Streaming verdadeiro, engine gerencia estado |

### 4.4 Workers (Subprocessos gRPC)

Cada worker é um **subprocess separado** que se comunica com o runtime via gRPC.

**Por que subprocessos?**
- Isolamento de falha: crash do worker não derruba o runtime
- Isolamento de GPU: cada worker gerencia seu próprio CUDA context
- Escalabilidade: mais workers = mais capacidade

```
┌──────────────────┐          ┌──────────────────┐
│   THEO RUNTIME   │  gRPC    │     WORKER       │
│   (processo      │ ◄──────► │   (subprocess)   │
│    principal)    │          │                  │
│                  │          │  Engine:         │
│  API Server      │          │  Faster-Whisper  │
│  Scheduler       │          │  (biblioteca)    │
│  Session Manager │          │                  │
│  Preprocessing   │          │  CUDA context    │
│  Post-processing │          │  próprio         │
└──────────────────┘          └──────────────────┘
```

**Protocolo gRPC (original do Theo):**

```protobuf
service STTWorker {
  rpc TranscribeFile   (TranscribeFileRequest)  returns (TranscribeFileResponse);
  rpc TranscribeStream (stream AudioFrame)      returns (stream TranscriptEvent);
  rpc Cancel           (CancelRequest)          returns (CancelResponse);
  rpc Health           (HealthRequest)          returns (HealthResponse);
}
```

**Detecção de crash**: O stream gRPC bidirecional (`TranscribeStream`) funciona como heartbeat implícito. Se o worker crashar, o stream termina com erro → runtime detecta imediatamente.

### 4.4.1 TTS Workers

TTS workers seguem o mesmo modelo de subprocess gRPC dos STT workers. A diferenca principal e que TTS usa **server-streaming** (texto entra, chunks de audio saem) em vez de bidirecional.

**Protocolo gRPC TTS:**

```protobuf
service TTSWorker {
  rpc Synthesize (SynthesizeRequest) returns (stream SynthesizeChunk);
  rpc Health     (HealthRequest)     returns (HealthResponse);
}

message SynthesizeRequest {
  string request_id = 1;
  string text = 2;
  string voice = 3;
  int32 sample_rate = 4;
  float speed = 5;
}

message SynthesizeChunk {
  bytes audio_data = 1;
  bool is_last = 2;
  float duration_seconds = 3;
}
```

**Interface TTSBackend:**

```python
class TTSBackend(ABC):
    async def load(self, model_path: str, config: dict) -> None: ...
    async def synthesize(self, text: str, voice: str = "default", *,
                        sample_rate: int = 24000, speed: float = 1.0
                        ) -> AsyncIterator[bytes]: ...
    async def voices(self) -> list[VoiceInfo]: ...
    async def unload(self) -> None: ...
    async def health(self) -> dict[str, str]: ...
```

**Implementacao:**

| Backend | Arquivo | Engine | Chunk Size | Sample Rate |
|---------|---------|--------|------------|-------------|
| `KokoroBackend` | `src/theo/workers/tts/kokoro.py` | Kokoro | 4096 bytes (~85ms) | 24kHz |

### 4.5 Interface STTBackend

O contrato que **toda engine STT deve implementar** para ser plugável no Theo:

```python
class STTBackend(ABC):
    @property
    def architecture(self) -> STTArchitecture:
        """encoder-decoder | ctc | streaming-native"""

    async def load(self, model_path: str, config: dict) -> None:
        """Carrega modelo em memória."""

    async def capabilities(self) -> EngineCapabilities:
        """O que esta engine suporta em runtime."""

    async def transcribe_file(self, audio_path, language, ...) -> BatchResult:
        """Transcrição batch (arquivo completo)."""

    async def transcribe_stream(self, audio_chunks, language, ...) -> AsyncIterator[TranscriptSegment]:
        """Transcrição streaming (chunks de áudio)."""

    async def unload(self) -> None:
        """Descarrega modelo da memória."""

    async def health(self) -> dict:
        """Status do backend."""
```

**Para adicionar uma nova engine** (ex: Paraformer):
1. Implementar `STTBackend`
2. Criar manifesto `theo.yaml` com `engine: paraformer`
3. Registrar no Model Registry
4. **Zero mudanças no runtime core**

### 4.5.1 Arquitetura Multi-Engine

O Theo e model-agnostic: o runtime adapta seu pipeline automaticamente com base na arquitetura declarada no manifesto do modelo. Dois backends STT estao implementados, validando que a interface `STTBackend` suporta arquiteturas fundamentalmente diferentes.

**Implementacoes:**

| Backend | Arquivo | Arquitetura | Partials | Hot Words | Initial Prompt |
|---------|---------|-------------|----------|-----------|----------------|
| `FasterWhisperBackend` | `src/theo/workers/stt/faster_whisper.py` | `encoder-decoder` | Via LocalAgreement (runtime) | Via `initial_prompt` (workaround) | Sim (conditioning) |
| `WeNetBackend` | `src/theo/workers/stt/wenet.py` | `ctc` | Nativos (frame-by-frame) | Nativos (keyword boosting) | Nao suporta |

**Diagrama de extensibilidade:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        WORKER SUBPROCESS                              │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                    STTBackend (ABC)                           │     │
│  │                                                              │     │
│  │  architecture    -> STTArchitecture enum                     │     │
│  │  load()          -> carrega modelo em memoria                │     │
│  │  capabilities()  -> EngineCapabilities                       │     │
│  │  transcribe_file()   -> BatchResult                          │     │
│  │  transcribe_stream() -> AsyncIterator[TranscriptSegment]     │     │
│  │  unload()        -> libera recursos                          │     │
│  │  health()        -> status                                   │     │
│  └──────────┬────────────────────────┬──────────────────────────┘     │
│             │                        │                                │
│             ▼                        ▼                                │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐  │
│  │ FasterWhisper    │     │ WeNet            │     │ [Paraformer] │  │
│  │ Backend          │     │ Backend          │     │ (futuro)     │  │
│  │                  │     │                  │     │              │  │
│  │ encoder-decoder  │     │ ctc              │     │ streaming-   │  │
│  │ CTranslate2     │     │ LibTorch         │     │ native       │  │
│  │ Whisper models   │     │ WeNet CTC models │     │              │  │
│  └──────────────────┘     └──────────────────┘     └──────────────┘  │
│                                                                       │
│  _create_backend(engine) -> instancia correta                         │
└──────────────────────────────────────────────────────────────────────┘
```

**Factory Pattern — `_create_backend()`:**

No entry point do worker (`src/theo/workers/stt/main.py`), a funcao `_create_backend(engine)` resolve o nome da engine para a implementacao correta:

```python
def _create_backend(engine: str) -> STTBackend:
    if engine == "faster-whisper":
        from theo.workers.stt.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()
    if engine == "wenet":
        from theo.workers.stt.wenet import WeNetBackend
        return WeNetBackend()
    raise ValueError(f"Engine STT nao suportada: {engine}")
```

O import lazy garante que dependencias pesadas (CTranslate2, LibTorch) so sao carregadas quando a engine e realmente utilizada. Cada engine e uma dependencia opcional (`pip install theo-openvoice[faster-whisper]` ou `pip install theo-openvoice[wenet]`).

**Como o manifesto direciona o pipeline:**

O campo `capabilities.architecture` no manifesto `theo.yaml` e lido pelo runtime (WebSocket endpoint, Session Manager) para adaptar o comportamento de streaming:

```
                      ┌──────────────┐
                      │ theo.yaml    │
                      │ architecture │
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     encoder-decoder       ctc       streaming-native
              │              │              │
              ▼              ▼              ▼
     LocalAgreement    Partials        Engine gerencia
     para partials     nativos         estado interno
              │              │              │
              ▼              ▼              ▼
     Cross-segment     Sem cross-      Depende da
     context ativo     segment ctx     engine
              │              │              │
              ▼              ▼              ▼
     Hot words via     Hot words       Engine-specific
     initial_prompt    nativos
```

No endpoint WebSocket (`src/theo/server/routes/realtime.py`), a arquitetura e extraida do manifesto e passada ao `StreamingSession`:

```python
model_architecture = manifest.capabilities.architecture or STTArchitecture.ENCODER_DECODER
```

O `StreamingSession` usa esse campo para:

1. **Cross-segment context**: apenas para `encoder-decoder` (Whisper suporta conditioning via `initial_prompt`). CTC nao suporta — o contexto e ignorado.
2. **Hot words no prompt**: para engines sem keyword boosting nativo (Whisper), hot words sao injetados via `initial_prompt`. Para engines com suporte nativo (WeNet), hot words sao enviados via campo dedicado no `AudioFrame` gRPC.
3. **LocalAgreement**: aplicado apenas para `encoder-decoder`. CTC e `streaming-native` produzem partials nativos — o runtime os repassa diretamente.

**Comparativo de comportamento por arquitetura:**

| Caracteristica | Encoder-Decoder (Whisper) | CTC (WeNet) | Streaming-Native (Futuro) |
|----------------|--------------------------|-------------|---------------------------|
| Partials | Via LocalAgreement (runtime) | Nativos frame-by-frame | Nativos |
| Cross-segment context | Sim (`initial_prompt`) | Nao (ignorado) | Depende da engine |
| Hot words | Via `initial_prompt` | Keyword boosting nativo | Depende da engine |
| TTFB tipico | ~300ms (acumula window 3-5s) | <100ms (per-frame) | <100ms |
| Streaming gRPC | Acumula threshold (5s) | Processa cada chunk | Engine gerencia estado |
| Melhor para | Qualidade maxima, batch | Baixa latencia, streaming | Streaming verdadeiro |

**Principio de validacao (M7):** Se a interface `STTBackend` funciona para Faster-Whisper (encoder-decoder) E WeNet (CTC) sem condicional `if architecture == "ctc"` espalhado pelo runtime core, a abstracao esta correta. Os unicos pontos onde a arquitetura e consultada sao: (1) `StreamingSession._build_initial_prompt()` para decidir se inclui cross-segment context, e (2) `StreamingSession._receive_worker_events()` para decidir se atualiza cross-segment context apos `transcript.final`. Esses sao pontos de variacao legitimos, nao leaky abstractions.

### 4.6 Session Manager (apenas STT)

Gerencia o **estado de cada sessão de streaming**. Componente que **não existe em nenhum projeto open-source de STT**.

**Máquina de estados:**

```
                    ┌──────────────────────────────────┐
                    │                                  │
     ┌──────┐    ┌─┴────┐    ┌─────────┐    ┌──────┐  │   ┌─────────┐    ┌────────┐
     │ INIT │───►│ACTIVE │───►│ SILENCE │───►│ HOLD │──┘   │CLOSING  │───►│ CLOSED │
     └──────┘    └───────┘    └────┬────┘    └──┬───┘      └─────────┘    └────────┘
        │                         │             │              ▲               ▲
        │                         │             │              │               │
        │                         └─────────────┘              │               │
        │                         (nova fala detectada)        │               │
        │                                                      │               │
        └──────────────────────────────────────────────────────┘               │
        (timeout 30s sem áudio)                                                │
                                                                               │
        Qualquer estado com erro irrecuperável ────────────────────────────────┘
```

| Estado      | O que acontece                                   | Timeout               |
|-------------|--------------------------------------------------|-----------------------|
| **INIT**    | Sessão criada, aguardando primeiro áudio          | 30s → CLOSED          |
| **ACTIVE**  | Recebendo áudio com fala detectada                | —                     |
| **SILENCE** | VAD detectou silêncio, aguardando retomada        | 30s → HOLD            |
| **HOLD**    | Silêncio prolongado (ex: chamada em hold)         | 5min → CLOSING        |
| **CLOSING** | Flush de partial transcripts pendentes            | 2s → CLOSED           |
| **CLOSED**  | Sessão encerrada, recursos liberados              | —                     |

**Por que o estado HOLD existe?**
Em call centers, chamadas em hold (música de espera, transferência) duram 1-10 minutos. Sem HOLD, o timeout de 30s do SILENCE fecharia a sessão prematuramente.

**Recovery de falha de worker:**

```
Worker crashou durante ACTIVE
    │
    ▼
1. gRPC stream break detectado (imediato)
2. Evento "error" com recoverable: true → cliente
3. Worker reiniciado
4. WAL in-memory consultado:
   • last_committed_segment_id
   • last_committed_buffer_offset
   • last_committed_timestamp_ms
5. Sessão retomada do último segmento confirmado
6. Áudio no ring buffer após último commit → reprocessado
7. Zero duplicação de segmentos
```

### 4.7 Ring Buffer

Buffer circular pré-alocado por sessão que armazena áudio recente.

```
┌────────────────────────────────────────────────────────┐
│                    RING BUFFER (60s)                     │
│                                                         │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ◄── já commitado ──►│◄── não commitado (protegido) ──►│
│                       │                                 │
│                  READ FENCE                             │
│            (last_committed_offset)                      │
│                                                         │
│  • Dados antes do fence: podem ser sobrescritos         │
│  • Dados depois do fence: protegidos (para recovery)    │
│  • 90% cheio sem commit → FORCE COMMIT do segmento     │
│                                                         │
│  Tamanho: 60s × 16kHz × 2 bytes = 1.9MB por sessão     │
└────────────────────────────────────────────────────────┘
```

**Por que ring buffer fixo?**
- Zero allocations durante streaming → latência previsível
- Tamanho fixo → sem memory leak em sessões longas
- 1.9MB/sessão → custo aceitável

### 4.8 VAD (Voice Activity Detection)

VAD roda no **runtime**, não dentro da engine — garante comportamento consistente entre engines diferentes.

```
Frame de áudio (64ms)
    │
    ▼
┌──────────────────────────┐
│ Energy Pre-filter (Theo) │ ──→ RMS < threshold E spectral flatness > 0.8?
│ Custo: ~0.1ms/frame      │     • SIM → silêncio (não chama Silero)
└───────────┬──────────────┘     • NÃO → continua ▼
            │
            ▼
┌──────────────────────────┐
│ Silero VAD (biblioteca)  │ ──→ Probabilidade de fala > threshold?
│ Custo: ~2ms/frame        │     • SIM → ACTIVE
└──────────────────────────┘     • NÃO → SILENCE
```

**Sensitivity levels:**

| Nível    | Threshold | Energy Pre-filter | Caso de uso                      |
|----------|-----------|-------------------|----------------------------------|
| `high`   | 0.3       | -50dBFS           | Banking confidencial, sussurro   |
| `normal` | 0.5       | -40dBFS           | Conversação normal (default)     |
| `low`    | 0.7       | -30dBFS           | Ambiente ruidoso, call center    |

### 4.9 Audio Preprocessing Pipeline

Normaliza áudio de **qualquer fonte** antes de chegar ao VAD e à engine.

```
┌────────────────────────────────────────────────────────────────────┐
│                   AUDIO PREPROCESSING PIPELINE                      │
│                                                                     │
│   INPUT                                                             │
│   (qualquer SR,    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐│
│    qualquer fmt)──►│ Resample │►│ DC Remove│►│Gain Norm │►│Denoise││
│                    │ (soxr)   │ │(HPF 20Hz)│ │(-3dBFS)  │ │(RNN) ││
│                    │ <1ms     │ │ <0.1ms   │ │ <0.1ms   │ │ ~1ms ││
│                    └──────────┘ └──────────┘ └──────────┘ └──────┘│
│                                                           OUTPUT   │
│                                                    PCM 16-bit     │
│                                                    16kHz, mono    │
│                                                    normalizado    │
│   Custo total: <5ms/frame (com denoise)                           │
└────────────────────────────────────────────────────────────────────┘
```

| Stage              | O que faz                                | Default    | Quando desligar                |
|--------------------|------------------------------------------|------------|--------------------------------|
| **Resample**       | Qualquer SR → 16kHz mono                 | Ativo      | Nunca (engine exige 16kHz)     |
| **DC Remove**      | HPF 20Hz remove DC offset                | Ativo      | Quando fonte já é limpa        |
| **Gain Normalize** | Normaliza pico para -3dBFS               | Ativo      | Quando amplitude já é estável  |
| **Denoise**        | Reduz ruído de fundo                     | Desativado | Habilitar para ambientes ruidosos|

### 4.10 Post-Processing Pipeline

Transforma o texto cru da engine em texto **usável**.

```
┌────────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING PIPELINE                         │
│                                                                     │
│   Engine Output ──►┌──────────┐ ┌──────────────┐ ┌───────────────┐ │
│   "dois mil e      │   ITN    │►│Entity Format │►│Hot Word Correct│ │
│    vinte e cinco"  │  (NeMo)  │ │   (Theo)     │ │    (Theo)     │ │
│                    └──────────┘ └──────────────┘ └───────────────┘ │
│                                                                     │
│   Output ──► "2025"                                                 │
│                                                                     │
│   Aplicado APENAS em transcript.final (nunca em partial)            │
│   Custo: <10ms/segmento                                            │
└────────────────────────────────────────────────────────────────────┘
```

**Exemplos de transformação:**

| Stage               | Input                              | Output          |
|----------------------|------------------------------------|-----------------|
| ITN (NeMo)           | "dois mil e vinte e cinco"         | "2025"          |
| ITN (NeMo)           | "dez por cento"                    | "10%"           |
| Entity (Banking)     | "um dois três ponto quatro..."     | "123.456.789-00"|
| Hot Word Correction  | "pics" (hot word: PIX)             | "PIX"           |

### 4.11 Full-Duplex (STT + TTS)

STT e TTS operam na mesma conexao WebSocket (`/v1/realtime`), coordenados por mute-on-speak.

**Fluxo:**

```
Cliente                     API Server               STT Worker    TTS Worker
  |                              |                        |              |
  | Audio frames (binary) -----> |                        |              |
  |                              | Preprocessing + VAD    |              |
  |                              |---> gRPC stream -----> |              |
  | <--- transcript.partial ---  |                        |              |
  | <--- transcript.final -----  |                        |              |
  |                              |                        |              |
  | {"type":"tts.speak"} ------> |                        |              |
  |                              | session.mute()         |              |
  |                              |----> Synthesize ------> |             |
  | <--- tts.speaking_start ---  |                        |              |
  | <--- binary audio chunks --- |  <--- chunks --------- |              |
  | <--- tts.speaking_end -----  |                        |              |
  |                              | session.unmute()       |              |
  |                              |                        |              |
  | Audio frames (binary) -----> | (STT resumes)         |              |
```

**Mute-on-Speak:**

Enquanto o TTS esta ativo, o STT descarta frames de audio do cliente:

1. `tts.speak` recebido -> `session.mute()`
2. Audio frames do cliente -> descartados (sem preprocessing, sem VAD)
3. TTS completa ou erro -> `session.unmute()` (via `finally`)
4. Audio frames do cliente -> processados normalmente

Metrica `theo_stt_muted_frames_total` conta frames descartados.

**Limitacao**: Mute-on-speak nao suporta barge-in (usuario interromper o bot). Para barge-in, e necessario AEC (Acoustic Echo Cancellation) externo.

---

## 5. Fluxos de Dados

### 5.1 Fluxo Batch (Fase 1) — Transcrição de Arquivo

```
Cliente                API Server              Scheduler         Worker (gRPC)
  │                        │                       │                   │
  │ POST /v1/audio/        │                       │                   │
  │ transcriptions         │                       │                   │
  │ file=audio.wav         │                       │                   │
  │ model=fw-large-v3      │                       │                   │
  │───────────────────────►│                       │                   │
  │                        │                       │                   │
  │                        │  1. Preprocessing     │                   │
  │                        │  (resample, normalize)│                   │
  │                        │                       │                   │
  │                        │  2. Resolve modelo    │                   │
  │                        │─────────────────────► │                   │
  │                        │                       │                   │
  │                        │                       │  3. gRPC          │
  │                        │                       │  TranscribeFile   │
  │                        │                       │─────────────────► │
  │                        │                       │                   │
  │                        │                       │                   │ 4. Faster-Whisper
  │                        │                       │                   │    inference
  │                        │                       │                   │
  │                        │                       │  5. Resultado     │
  │                        │                       │◄─────────────────│
  │                        │                       │                   │
  │                        │  6. Post-processing   │                   │
  │                        │  (ITN, entity format) │                   │
  │                        │                       │                   │
  │  7. JSON response      │                       │                   │
  │◄───────────────────────│                       │                   │
  │  {"text": "2025..."}   │                       │                   │
```

### 5.2 Fluxo Streaming (Fase 2) — WebSocket STT

```
Cliente               API Server         Session Manager        Worker (gRPC)
  │                       │                    │                      │
  │ WS /v1/realtime       │                    │                      │
  │ model=fw-large-v3     │                    │                      │
  │──────────────────────►│                    │                      │
  │                       │                    │                      │
  │ ◄─ session.created    │  Cria sessão       │                      │
  │                       │──────────────────► │                      │
  │                       │                    │ Estado: INIT          │
  │                       │                    │                      │
  │ Audio frames (binary) │                    │                      │
  │──────────────────────►│                    │                      │
  │                       │  Preprocessing     │                      │
  │                       │──────────────────► │                      │
  │                       │                    │ Ring Buffer ← write  │
  │                       │                    │                      │
  │                       │                    │ VAD: fala detectada   │
  │                       │                    │ Estado: ACTIVE        │
  │ ◄─ vad.speech_start   │                    │                      │
  │                       │                    │                      │
  │                       │                    │ LocalAgreement        │
  │                       │                    │ (acumula window)      │
  │                       │                    │───────────────────►  │
  │                       │                    │                      │ inference
  │                       │                    │◄───────────────────  │
  │                       │                    │ compara com pass     │
  │                       │                    │ anterior             │
  │                       │                    │                      │
  │ ◄─ transcript.partial │ tokens confirmados │                      │
  │    "Olá como"         │                    │                      │
  │                       │                    │                      │
  │                       │                    │ VAD: silêncio         │
  │                       │                    │ Estado: SILENCE       │
  │                       │                    │───────────────────►  │
  │                       │                    │                      │ flush
  │                       │                    │◄───────────────────  │
  │                       │                    │                      │
  │                       │                    │ Post-processing      │
  │ ◄─ transcript.final   │                    │ (ITN)                │
  │    "Olá, como posso   │                    │                      │
  │     ajudar?"          │                    │                      │
  │                       │                    │ Ring Buffer fence ►  │
  │ ◄─ vad.speech_end     │                    │                      │
```

---

## 6. Pipeline Adaptativo por Arquitetura

O runtime adapta automaticamente o pipeline de streaming baseado no campo `architecture` do manifesto:

### 6.1 Encoder-Decoder (Whisper) — com LocalAgreement

```
Audio → Preprocessing → Ring Buffer → Acumula window (3-5s)
                                              │
                                              ▼
                                     Engine.transcribe_stream()
                                              │
                                              ▼
                                     Compara com pass anterior
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                    Tokens concordam              Tokens divergem
                    entre passes                  entre passes
                              │                               │
                              ▼                               ▼
                    transcript.partial              Aguarda próxima pass
                    (prefixo confirmado)
                              │
                    VAD silence detectado
                              │
                              ▼
                    transcript.final (flush)
```

**Por que LocalAgreement?**
- Re-processamento naive = 2+ inferências/s por sessão → satura GPU
- LocalAgreement = 1 inferência/window + comparação → ~50% menos custo
- Conceito do [whisper-streaming](https://github.com/ufal/whisper_streaming) (UFAL), implementação própria

### 6.2 CTC (WeNet)

```
Audio → Preprocessing → Ring Buffer → Engine.transcribe_stream() (frame por frame)
                                                    │
                                         TranscriptSegment (partial + final)
                                         Partials nativos, <100ms TTFB
```

### 6.3 Streaming-Native (Paraformer)

```
Audio → Preprocessing → Engine.transcribe_stream() (engine gerencia estado interno)
                                         │
                              TranscriptSegment (partial + final)
                              Partials nativos
```

---

## 7. Protocolo WebSocket (Streaming)

### 7.1 Mensagens Cliente → Servidor

| Mensagem                    | Tipo   | Descrição                                      |
|-----------------------------|--------|-------------------------------------------------|
| (frames de áudio)           | Binary | PCM 16-bit, qualquer SR, 20-40ms/frame, max 64KB|
| `session.configure`         | JSON   | Configura VAD, language, hot words, preprocessing|
| `session.cancel`            | JSON   | Cancela sessão                                  |
| `input_audio_buffer.commit` | JSON   | Força commit de segmento                        |
| `session.close`             | JSON   | Encerra sessão gracefully                       |
| `tts.speak`                 | JSON   | Sintetiza voz e envia audio como binary frames  |
| `tts.cancel`                | JSON   | Cancela sintese TTS ativa                       |

### 7.2 Mensagens Servidor → Cliente

| Mensagem                  | Quando emitido                                       |
|---------------------------|------------------------------------------------------|
| `session.created`         | WebSocket conectado, sessão pronta                   |
| `vad.speech_start`        | VAD detectou início de fala                          |
| `transcript.partial`      | Hipótese intermediária (LocalAgreement ou nativo)    |
| `transcript.final`        | Segmento confirmado (após VAD silence ou force commit)|
| `vad.speech_end`          | VAD detectou fim de fala                             |
| `session.hold`            | Sessão transitou para HOLD (silêncio prolongado)     |
| `session.rate_limit`      | Backpressure: cliente enviando rápido demais         |
| `session.frames_dropped`  | Frames descartados (backlog > 10s)                   |
| `tts.speaking_start`      | TTS comecou a produzir audio                         |
| (binary audio frames)     | Chunks de audio TTS (PCM 16-bit, server->client)     |
| `tts.speaking_end`        | TTS terminou (com flag `cancelled`)                  |
| `error`                   | Erro (com flag `recoverable`)                        |
| `session.closed`          | Sessão encerrada                                     |

### 7.3 Heartbeat

- Server envia **ping WebSocket** a cada 10s
- Se **pong** não recebido em 5s → sessão transita para CLOSING
- Previne connections zombies

---

## 8. CLI

Segue o padrão UX do Ollama (`pull/serve/list`), adaptado para modelos de voz:

```bash
# ─── Gerenciamento de Modelos ───
theo pull faster-whisper-large-v3    # Baixa modelo STT
theo pull kokoro-v1                  # Baixa modelo TTS
theo list                            # Lista modelos instalados
theo ps                              # Lista modelos carregados em memória
theo remove <model>                  # Remove modelo
theo inspect <model>                 # Detalhes do modelo

# ─── Runtime ───
theo serve                           # Inicia API Server (serve tudo instalado)

# ─── STT ───
theo transcribe <file>               # Transcreve arquivo
theo transcribe <file> --format srt  # Gera legenda SRT
theo transcribe --stream             # Streaming do microfone
theo translate <file>                # Traduz para inglês

# ─── STT v2 ───
theo transcribe <file> --hot-words "PIX,TED,Selic"   # Com hot words
theo transcribe <file> --no-itn                        # Sem ITN
```

---

## 9. Observabilidade

### Métricas Prometheus

**Operacionais:**

| Métrica                                  | Tipo       | Descrição                                   |
|------------------------------------------|------------|---------------------------------------------|
| `theo_stt_ttfb_seconds`                  | Histogram  | Tempo até primeiro partial transcript       |
| `theo_stt_final_delay_seconds`           | Histogram  | Delay do final após fim de fala             |
| `theo_stt_active_sessions`               | Gauge      | Sessões ativas agora                        |
| `theo_stt_session_duration_seconds`       | Histogram  | Duração de sessões                          |
| `theo_stt_vad_events_total`              | Counter    | Eventos VAD (speech_start, speech_end)      |
| `theo_stt_worker_errors_total`           | Counter    | Erros por worker                            |
| `theo_stt_preprocessing_duration_seconds` | Histogram  | Tempo de preprocessing por frame            |
| `theo_stt_postprocessing_duration_seconds`| Histogram  | Tempo de post-processing por segmento       |

**Qualidade:**

| Métrica                                        | O que indica                                  |
|------------------------------------------------|-----------------------------------------------|
| `theo_stt_confidence_avg`                      | Proxy para WER (confidence baixa = problema)  |
| `theo_stt_no_speech_segments_total`            | Valor alto = VAD misconfigured                |
| `theo_stt_language_detection_mismatches_total` | Idioma declarado ≠ detectado                  |
| `theo_stt_hot_word_hits_total`                 | Zero persistente = hot words ineficazes       |
| `theo_stt_segments_force_committed_total`      | Valor alto = silêncio não detectado           |

---

## 10. Requisitos Não Funcionais

### Latência

| Cenário                   | Target                                              |
|---------------------------|-----------------------------------------------------|
| Batch (arquivo)           | ≤0.5x duração (30s áudio → ≤15s processamento)     |
| Batch com BatchPipeline   | ≤0.2x duração em GPU (30s → ≤6s)                   |
| Streaming TTFB            | ≤300ms (primeiro partial após receber segmento)     |
| Final transcript delay    | ≤500ms após VAD detectar fim de fala                |
| Preprocessing overhead    | ≤5ms/frame (com denoise)                            |
| Post-processing overhead  | ≤10ms/segmento final                                |
| Cancelamento              | ≤50ms propagação via gRPC                           |

### Latência Voice-to-Voice (V2V) — Target Completo

```
┌─────────────────────────────────────────────────────────┐
│          LATENCY BUDGET: 300ms total V2V                 │
│                                                          │
│   VAD End-of-Speech .......... 50ms                      │
│   ASR Final Transcript ....... 100ms                     │
│   LLM Time to First Token .... 100ms                     │
│   TTS Time to First Byte ...... 50ms                     │
│                                ─────                     │
│                         TOTAL: 300ms                     │
└─────────────────────────────────────────────────────────┘
```

### Estabilidade

| Requisito                           | Target                                  |
|-------------------------------------|-----------------------------------------|
| Sessão contínua sem degradação      | ≥30 minutos                             |
| Memory leak em sessão longa         | Zero (ring buffer fixo)                 |
| Isolamento de falhas                | Worker crash ≠ runtime crash            |
| Recovery de sessão                  | Sem duplicação de segmentos (WAL)       |

### GPU

| Fase   | Concurrency Model                                                       |
|--------|-------------------------------------------------------------------------|
| Fase 1 | 1 worker = 1 sessão ativa. Escala horizontal com mais workers           |
| Fase 2 | Dynamic batching no worker (acumula requests por 50ms, batch inference) |

**Referência**: A10G com Whisper large-v3 → ~4 sessões batch ou ~2 sessões streaming

---

## 11. O Que É Construído do Zero vs. Bibliotecas

### Código Original do Theo (construído do zero)

| Componente                    | Descrição                                              |
|-------------------------------|--------------------------------------------------------|
| API Server                    | FastAPI com endpoints STT + TTS                        |
| Scheduler                     | Priorização, cancelamento, orçamento de latência       |
| Session Manager               | 6 estados, recovery, WAL in-memory                     |
| Model Registry                | Manifesto `theo.yaml`, lifecycle, eviction              |
| Audio Preprocessing Pipeline  | Orquestração dos stages (resample, DC, normalize, denoise) |
| Post-Processing Pipeline      | ITN orchestration, Entity Formatting, Hot Word Correction |
| Ring Buffer                   | Read fence, force commit, zero-copy                    |
| LocalAgreement                | Partial transcripts para encoder-decoder               |
| CLI                           | `theo pull/serve/transcribe/...`                       |
| Protocolo gRPC                | Comunicação runtime ↔ worker                           |
| Protocolo WebSocket           | Eventos de streaming STT                               |
| Energy Pre-filter             | Pre-VAD para reduzir falsos positivos                  |
| Entity Formatting             | CPF, CNPJ, valores monetários por domínio              |
| Hot Word Correction           | Levenshtein distance + boost                           |
| Full-Duplex Coordinator       | Mute-on-speak, co-scheduling STT+TTS na mesma sessao  |

### Bibliotecas (dependências substituíveis)

| Biblioteca             | Papel no Theo                   | Pode ser substituída por           |
|------------------------|---------------------------------|------------------------------------|
| Faster-Whisper         | Engine de inferência STT        | WeNet, Paraformer, qualquer `STTBackend` |
| Silero VAD             | Voice Activity Detection        | Qualquer VAD que aceite PCM 16kHz  |
| nemo_text_processing   | Inverse Text Normalization      | `itn` library, custom regex        |
| RNNoise / NSNet2       | Noise reduction                 | Qualquer denoiser PCM              |
| soxr / scipy           | Resampling, filtros             | Qualquer resampler equivalente     |
| Kokoro / Piper         | Engine de inferência TTS        | Qualquer `TTSBackend`              |
| FastAPI                | HTTP/WebSocket framework        | (dependência estrutural)           |
| gRPC                   | Comunicação runtime ↔ worker    | (dependência estrutural)           |

### Inspirações (zero código compartilhado)

| Projeto           | O que inspirou                                         |
|-------------------|--------------------------------------------------------|
| Ollama            | UX de CLI (`pull/serve/list`), registry local          |
| Speaches          | Contrato de API compatível com OpenAI para STT         |
| whisper-streaming | Conceito de LocalAgreement para partial transcripts    |

---

## 12. Roadmap de Implementação

### Fase 1 — STT Batch + Preprocessing (6 semanas)

**Entregáveis:**
- `POST /v1/audio/transcriptions` e `POST /v1/audio/translations`
- Worker Faster-Whisper (subprocess gRPC)
- Audio Preprocessing Pipeline (resample, DC remove, gain normalize)
- Post-Processing Pipeline (ITN básico via NeMo)
- Model Registry compartilhado (campo `type: stt`)
- CLI: `theo transcribe <file>`, `theo translate <file>`
- Formatos de resposta: json, verbose_json, text, srt, vtt

**Critério de sucesso:**
```bash
curl -F file=@audio.wav -F model=faster-whisper-large-v3 \
  http://localhost:8000/v1/audio/transcriptions
# → Retorna transcrição com números formatados (ITN)
```

### Fase 2 — Streaming Real + Session Manager (8 semanas)

**Entregáveis:**
- `WS /v1/realtime` com protocolo de eventos
- Session Manager (6 estados, recovery, WAL)
- VAD (Silero + energy pre-filter, sensitivity levels)
- Ring Buffer (read fence, force commit)
- LocalAgreement para partial transcripts
- Cross-segment context
- Backpressure e heartbeat WebSocket
- Hot words via `session.configure`
- Segundo backend (WeNet) validando model-agnostic
- CLI: `theo transcribe --stream`

**Critério de sucesso:** Sessão WebSocket de 30 min sem degradação, com recovery sem duplicação.

### Fase 3 — Escala + Full-Duplex (4 semanas) ✅

**Entregáveis:**
- Scheduler com priorização (realtime > batch) ✅
- Dynamic batching no worker ✅
- Co-scheduling STT + TTS (agentes full-duplex) ✅
- Mute-on-speak fallback ✅

---

## 13. Decisões Arquiteturais (ADRs)

| ADR   | Decisão                                          | Status  |
|-------|--------------------------------------------------|---------|
| 001   | Runtime Unificado STT + TTS (um binário)         | Aceito  |
| 002   | Construção From-Scratch com Bibliotecas Maduras   | Aceito  |
| 003   | VAD no Runtime, Não na Engine (Silero como lib)   | Aceito  |
| 004   | Windowing Adaptativo + LocalAgreement             | Aceito  |
| 005   | Ring Buffer Pre-alocado com Read Fence            | Aceito  |
| 006   | Protocolo WebSocket JSON (inspirado OpenAI)       | Aceito  |
| 007   | Audio Preprocessing no Runtime                    | Aceito  |
| 008   | LocalAgreement para Partial Transcripts           | Aceito  |
| 009   | Post-Processing Pipeline Plugável                 | Aceito  |
| 010   | TTS MVP com Engine Placeholder (Kokoro)           | Aceito  |
| 011   | Mute-on-Speak como Fallback (não Barge-in)        | Aceito  |
| 012   | TTS Streaming Unidirecional (server-streaming)    | Aceito  |
| 013   | Protocolo WebSocket Unificado STT + TTS           | Aceito  |

Detalhes completos de cada ADR no [PRD v2.1](./PRD.md#architecture-decision-records-adrs).

---

## 14. Estrutura de Pacotes Proposta (Fase 1)

```
theo-openvoice/
├── src/
│   └── theo/
│       ├── __init__.py
│       ├── __main__.py              # Entry point CLI
│       │
│       ├── server/                  # API Server
│       │   ├── __init__.py
│       │   ├── app.py               # FastAPI application
│       │   ├── routes/
│       │   │   ├── transcriptions.py  # POST /v1/audio/transcriptions
│       │   │   ├── translations.py    # POST /v1/audio/translations
│       │   │   └── health.py          # GET /health, /metrics
│       │   └── models/              # Pydantic request/response models
│       │       ├── transcription.py
│       │       └── common.py
│       │
│       ├── scheduler/               # Request scheduling
│       │   ├── __init__.py
│       │   └── scheduler.py
│       │
│       ├── registry/                # Model Registry
│       │   ├── __init__.py
│       │   ├── registry.py          # Core registry logic
│       │   └── manifest.py          # theo.yaml parsing
│       │
│       ├── workers/                 # Worker management
│       │   ├── __init__.py
│       │   ├── manager.py           # Subprocess lifecycle
│       │   └── stt/
│       │       ├── __init__.py
│       │       ├── interface.py     # STTBackend ABC
│       │       └── faster_whisper.py # Faster-Whisper implementation
│       │
│       ├── preprocessing/           # Audio Preprocessing Pipeline
│       │   ├── __init__.py
│       │   ├── pipeline.py          # Pipeline orchestration
│       │   ├── resample.py
│       │   ├── dc_remove.py
│       │   ├── gain_normalize.py
│       │   └── denoise.py
│       │
│       ├── postprocessing/          # Post-Processing Pipeline
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   ├── itn.py               # Inverse Text Normalization
│       │   ├── entity_formatting.py
│       │   └── hot_word_correction.py
│       │
│       ├── cli/                     # CLI commands
│       │   ├── __init__.py
│       │   ├── main.py              # CLI entry point (click/typer)
│       │   ├── pull.py
│       │   ├── serve.py
│       │   ├── transcribe.py
│       │   └── list.py
│       │
│       └── proto/                   # gRPC definitions
│           ├── stt_worker.proto
│           └── stt_worker_pb2.py    # Generated
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│   ├── PRD.md
│   └── ARCHITECTURE.md              # ← Este documento
│
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 15. Glossário

| Termo                  | Definição                                                                  |
|------------------------|----------------------------------------------------------------------------|
| **Runtime**            | O processo principal do Theo que orquestra tudo                            |
| **Worker**             | Subprocess que executa a engine de inferência via gRPC                     |
| **Engine**             | Biblioteca de inferência (Faster-Whisper, Kokoro, etc.)                    |
| **Session**            | Conexão de streaming STT com estado gerenciado                             |
| **Ring Buffer**        | Buffer circular pré-alocado que armazena áudio da sessão                   |
| **Read Fence**         | Ponteiro no ring buffer que protege dados não commitados                   |
| **Force Commit**       | Flush forçado de segmento quando ring buffer atinge 90%                    |
| **WAL**                | Write-Ahead Log in-memory para recovery sem duplicação                     |
| **LocalAgreement**     | Algoritmo de confirmação de tokens por concordância entre passes           |
| **Partial transcript** | Hipótese intermediária de transcrição (pode mudar)                         |
| **Final transcript**   | Segmento confirmado de transcrição (não muda)                              |
| **ITN**                | Inverse Text Normalization ("dois mil" → "2000")                           |
| **VAD**                | Voice Activity Detection (detecta fala vs silêncio)                        |
| **TTFB**               | Time to First Byte (tempo até primeira resposta)                           |
| **V2V**                | Voice-to-Voice (latência total de voz entrada → voz saída)                 |
| **Barge-in**           | Usuário interrompe o bot enquanto ele fala                                 |
| **AEC**                | Acoustic Echo Cancellation (remover eco do TTS no microfone)               |
| **Manifesto**          | Arquivo `theo.yaml` que descreve capabilities de um modelo                 |
| **CTC**                | Connectionist Temporal Classification (arquitetura de STT)                 |
| **Encoder-decoder**    | Arquitetura de STT que processa chunks (ex: Whisper)                       |
| **Streaming-native**   | Arquitetura de STT com streaming verdadeiro (ex: Paraformer)               |
| **Full-Duplex**        | Modo onde STT e TTS operam simultaneamente na mesma sessao WebSocket       |
| **Mute-on-Speak**      | Pausar ingestao STT enquanto TTS esta ativo (fallback sem AEC)             |
| **TTSBackend**         | Interface que toda engine TTS deve implementar para ser plugavel no Theo   |

---

*Documento baseado no PRD v2.1. Todas as 3 fases (M1-M9) implementadas.*
