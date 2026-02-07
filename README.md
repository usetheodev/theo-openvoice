# Theo OpenVoice STT Runtime

## Product Requirements Document (PRD)

---

## 1. Visão Geral

Theo OpenVoice STT é o módulo de Speech-to-Text do runtime Theo OpenVoice — a mesma infraestrutura que serve TTS. O objetivo é expor uma **API compatível com OpenAI** para transcrição em tempo real, resolvendo lacunas críticas identificadas no projeto Speaches:

- Forte acoplamento ao Whisper.
- Ausência de scheduler temporal consciente de sessões.
- Falta de suporte nativo a cenários de telefonia (ingestão RTP).
- Abstração incompleta para múltiplas arquiteturas de STT.

O STT compartilha com o TTS o mesmo API Server (FastAPI), Model Registry, CLI, observabilidade e infraestrutura de deploy. A diferença está nos workers (engines STT em vez de TTS) e no Session Manager, que é específico de STT.

---

## 2. Relação com o Runtime TTS

### Arquitetura Unificada

```
┌──────────────────────────────────────────────────────────┐
│                     API Server (FastAPI)                   │
│         /v1/audio/speech (TTS)    /v1/audio/* (STT)       │
├──────────────────────────────────────────────────────────┤
│                        Scheduler                          │
│              (routing, queue, cancellation)                │
├──────────────────────────────────────────────────────────┤
│                     Model Registry                        │
│          (manifesto, lifecycle, eviction)                  │
├──────────────┬───────────────────────────────────────────┤
│  TTS Workers │            STT Workers                     │
│  (Kokoro,    │  (Faster-Whisper, WeNet,                   │
│   Piper)     │   Paraformer)                              │
│  subprocess  │  subprocess                                │
├──────────────┴───────────────────────────────────────────┤
│                  Session Manager (STT)                     │
│         (estado, ring buffer, VAD, timeout)                │
└──────────────────────────────────────────────────────────┘
```

### O que é compartilhado

| Componente | Compartilhado | Notas |
|---|---|---|
| API Server (FastAPI) | Sim | Endpoints TTS e STT no mesmo processo |
| Model Registry | Sim | Mesmo manifesto `theo.yaml`, mesmo lifecycle |
| Scheduler | Sim | Mesmo scheduler, prioridade por tipo (realtime > batch) |
| CLI | Sim | `theo pull`, `theo list`, `theo serve` servem ambos |
| Observabilidade | Sim | Mesmas métricas Prometheus, mesmo `/health` |
| Docker image | Sim | Uma imagem, engines habilitadas por config |
| Session Manager | Não | Específico de STT (TTS é stateless por request) |
| Workers | Não | Engines diferentes, protobuf diferente |

### Decisão arquitetural

**Um binário, um processo, dois tipos de worker.** Não existe "Theo STT" e "Theo TTS" como produtos separados. Existe **Theo OpenVoice** com capacidades de TTS e STT, habilitadas por quais modelos estão instalados.

---

## 3. Análise Competitiva

| Projeto | O que faz | Limitação principal |
|---|---|---|
| **Speaches** | Serve Whisper com API OpenAI-compatible | Acoplado ao Whisper, sem scheduler, sem telefonia |
| **Whisper.cpp server** | Serve Whisper via HTTP | Apenas Whisper, sem streaming real, sem session |
| **Vosk Server** | Serve Vosk/Kaldi via WebSocket | API proprietária, sem compatibilidade OpenAI |
| **LocalAI** | Runtime genérico (LLM, TTS, STT) | STT é feature secundária, sem session manager |
| **NVIDIA Riva** | STT/TTS enterprise | Proprietário, vendor lock-in NVIDIA |

**Diferencial do Theo OpenVoice STT:**

- **Model-agnostic real**: interface que abstrai encoder-decoder (Whisper), CTC (WeNet), e streaming-native (Paraformer) sem o core assumir tokenizer ou decoder.
- **Session Manager**: estado explícito por sessão com VAD, timeout e recovery.
- **Runtime unificado**: mesmo produto que serve TTS, compartilhando registry, scheduler e CLI.
- **Streaming-first com contrato claro**: formato de eventos definido, não apenas "partial transcripts".
- **Preparado para telefonia**: ingestão RTP como módulo, não como hack.

---

## 4. Objetivos

**Objetivos principais:**

- Expor API compatível com OpenAI para transcrição (file e streaming).
- Ser model-agnostic: suportar Whisper, CTC e streaming-native sem mudar o core.
- Operar com latência previsível em tempo real (TTFB ≤300ms por segmento).
- Gerenciar sessões de streaming com estado explícito e recovery.
- Compartilhar infraestrutura com o runtime TTS (mesmo binário, registry, scheduler).

**Não objetivos (v1):**

- Treinamento de modelos.
- UI gráfica.
- SIP signaling (apenas ingestão RTP raw na Fase 3).
- Speaker diarization (escopo futuro).
- Billing / autenticação comercial.

---

## 5. Casos de Uso

- **UC-01**: Transcrever arquivo de áudio via REST (`POST /v1/audio/transcriptions`), recebendo texto completo.
- **UC-02**: Transcrever áudio em tempo real via WebSocket, recebendo partial e final transcripts como eventos JSON.
- **UC-03**: Trocar engine STT (ex: Faster-Whisper → WeNet) sem alterar código do cliente — apenas mudar o campo `model`.
- **UC-04**: Manter sessão de streaming por 30+ minutos (call center) com estado gerenciado e recovery de falhas.
- **UC-05**: Receber áudio de um Asterisk via RTP e transcrever em tempo real.
- **UC-06**: Executar múltiplas sessões simultâneas com priorização (telefonia > batch).
- **UC-07**: Traduzir áudio para inglês via endpoint de translation.

---

## 6. Requisitos Funcionais

### RF-01: API OpenAI-Compatible para STT

Endpoints compatíveis com OpenAI Audio API:

- `POST /v1/audio/transcriptions` — transcrição de arquivo.
- `POST /v1/audio/translations` — tradução para inglês.
- `WS /v1/realtime` — streaming bidirecional em tempo real (Fase 2).

### RF-02: Transcrição de Arquivo (Batch)

Upload de arquivo de áudio, processamento completo, retorno de texto. Suporte aos formatos: WAV, MP3, FLAC, OGG, WebM.

### RF-03: Streaming STT

Transcrição em tempo real via WebSocket com:

- Partial transcripts (hipóteses intermediárias).
- Final transcripts (segmentos confirmados).
- Eventos de VAD (voice activity start/end).
- Detecção de silêncio com commit automático de segmento.

**Especificações de streaming de entrada:**

- Formato de áudio aceito: PCM 16-bit, 16kHz, mono (raw).
- Tamanho de window de processamento: 30ms por frame de entrada.
- Envio: cliente envia frames de áudio como mensagens binárias WebSocket.
- Amostra mínima para processamento: 500ms de áudio acumulado.

### RF-04: Voice Activity Detection (VAD)

VAD opera no nível do runtime (não da engine), usando Silero VAD:

- Threshold configurável (default: 0.5).
- Min speech duration: 250ms.
- Min silence duration: 300ms (para commit de segmento).
- Max speech duration: 30s (force commit).

VAD no runtime garante comportamento consistente entre engines.

### RF-05: Seleção Dinâmica de Modelo

Cada request especifica o modelo via campo `model`. O runtime resolve para o backend STT correto e carrega se necessário. Mesmo mecanismo do TTS.

### RF-06: Session Manager

Gerenciamento explícito de sessões de streaming com estados definidos:

```
INIT → ACTIVE → SILENCE → CLOSING → CLOSED
                  ↑          |
                  └──────────┘  (nova fala detectada)
```

**Estados:**

| Estado | Descrição | Timeout |
|---|---|---|
| `INIT` | Sessão criada, aguardando primeiro áudio | 10s (sem áudio → CLOSED) |
| `ACTIVE` | Recebendo áudio com fala detectada | — |
| `SILENCE` | VAD detectou silêncio, aguardando retomada | 30s (configurável, → CLOSING) |
| `CLOSING` | Flush de partial transcripts pendentes | 2s (→ CLOSED) |
| `CLOSED` | Sessão encerrada, recursos liberados | — |

**Recovery:** se worker crashar durante sessão ACTIVE, o Session Manager:

1. Detecta falha via health check (≤1s).
2. Emite evento `error` ao cliente com `recoverable: true`.
3. Reinicia worker.
4. Retoma sessão do último segmento confirmado (final transcript).
5. Áudio no ring buffer entre o último commit e o crash é reprocessado.

### RF-07: Cancelamento

Cliente pode cancelar sessão via WebSocket `close` ou mensagem `cancel`. Runtime propaga cancelamento ao worker em ≤50ms. Partial transcripts pendentes são descartados.

### RF-08: Multi-idioma

Detecção automática de idioma (se engine suportar) ou seleção explícita via campo `language` na request.

---

## 7. Requisitos Não Funcionais

### RNF-01: Latência

- **Batch (arquivo)**: tempo total proporcional à duração do áudio. Fator alvo: ≤0.5x (30s de áudio processado em ≤15s).
- **Streaming TTFB**: ≤300ms do recebimento de segmento com fala até primeiro partial transcript.
- **Final transcript delay**: ≤500ms após VAD detectar fim de fala.
- **Medição**: do momento em que o runtime acumula áudio suficiente até o primeiro evento de transcript ser escrito no WebSocket.

### RNF-02: Estabilidade de Sessão

- Sessões de streaming devem suportar ≥30 minutos contínuos sem degradação de latência ou memory leak.
- Ring buffer com tamanho fixo (default: 60s de áudio, ~1.9MB em PCM 16kHz mono).

### RNF-03: Isolamento de Falhas

- Crash de worker STT não afeta o runtime, TTS workers, nem outras sessões STT.
- Restart automático de worker com recovery de sessão (RF-06).

### RNF-04: Uso de GPU

- Múltiplas sessões compartilham GPU via batching no worker (se engine suportar).
- Sem reallocations de CUDA durante inferência (pre-allocate buffers no load).
- Fallback transparente para CPU.

### RNF-05: Observabilidade

Métricas Prometheus adicionais (além das compartilhadas com TTS):

- `theo_stt_ttfb_seconds` — tempo até primeiro partial transcript.
- `theo_stt_final_delay_seconds` — delay do final transcript após fim de fala.
- `theo_stt_active_sessions` — sessões ativas.
- `theo_stt_session_duration_seconds` — histograma de duração de sessões.
- `theo_stt_vad_events_total` — eventos VAD (speech_start, speech_end).
- `theo_stt_worker_errors_total` — erros por worker.

---

## 8. Contrato de API

### POST /v1/audio/transcriptions

**Request (multipart/form-data):**

| Campo | Tipo | Obrigatório | Default | Descrição |
|---|---|---|---|---|
| `file` | binary | sim | — | Arquivo de áudio |
| `model` | string | sim | — | Identificador do modelo no registry |
| `language` | string | não | auto-detect | Código ISO 639-1 |
| `prompt` | string | não | — | Contexto para guiar transcrição |
| `response_format` | string | não | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | não | `0.0` | Temperatura de sampling (0.0-1.0) |
| `timestamp_granularities` | array | não | `["segment"]` | `["word"]`, `["segment"]`, ou ambos |

**Response (`json`):**

```json
{
  "text": "Olá, como posso ajudar?"
}
```

**Response (`verbose_json`):**

```json
{
  "task": "transcribe",
  "language": "pt",
  "duration": 2.5,
  "text": "Olá, como posso ajudar?",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Olá, como posso ajudar?",
      "tokens": [1234, 5678],
      "temperature": 0.0,
      "avg_logprob": -0.25,
      "compression_ratio": 1.1,
      "no_speech_prob": 0.01
    }
  ],
  "words": [
    { "word": "Olá", "start": 0.0, "end": 0.5 },
    { "word": "como", "start": 0.6, "end": 0.9 },
    { "word": "posso", "start": 1.0, "end": 1.3 },
    { "word": "ajudar", "start": 1.4, "end": 2.5 }
  ]
}
```

**Erros:**

| Status | Significado |
|---|---|
| `400` | Formato de áudio não suportado ou input inválido |
| `404` | Modelo não encontrado no registry |
| `413` | Arquivo excede limite (default: 25MB) |
| `503` | Modelo em loading (cold start) |

### POST /v1/audio/translations

Mesmo contrato que `/transcriptions`, mas output sempre em inglês. Campo `language` indica o idioma do áudio de entrada (opcional, auto-detect).

### WS /v1/realtime (Fase 2)

**Handshake:**

```
GET /v1/realtime?model=faster-whisper-large-v3&language=pt
Upgrade: websocket
```

**Mensagens Client → Server:**

```json
// Enviar áudio (mensagem binária)
// Frames PCM 16-bit, 16kHz, mono
// Enviar como binary WebSocket messages

// Configurar sessão
{
  "type": "session.configure",
  "vad_threshold": 0.5,
  "silence_timeout_ms": 300,
  "max_segment_duration_ms": 30000,
  "language": "pt"
}

// Cancelar
{
  "type": "session.cancel"
}

// Commit manual de segmento
{
  "type": "input_audio_buffer.commit"
}

// Fechar sessão
{
  "type": "session.close"
}
```

**Mensagens Server → Client:**

```json
// Sessão criada
{
  "type": "session.created",
  "session_id": "sess_abc123",
  "model": "faster-whisper-large-v3",
  "config": { "vad_threshold": 0.5, "silence_timeout_ms": 300 }
}

// VAD: fala detectada
{
  "type": "vad.speech_start",
  "timestamp_ms": 1500
}

// Partial transcript
{
  "type": "transcript.partial",
  "text": "Olá como",
  "segment_id": 0,
  "timestamp_ms": 2000
}

// Final transcript
{
  "type": "transcript.final",
  "text": "Olá, como posso ajudar?",
  "segment_id": 0,
  "start_ms": 1500,
  "end_ms": 4000,
  "language": "pt",
  "confidence": 0.95
}

// VAD: silêncio detectado
{
  "type": "vad.speech_end",
  "timestamp_ms": 4000
}

// Erro (recuperável)
{
  "type": "error",
  "code": "worker_crash",
  "message": "Worker restarted, resuming from last segment",
  "recoverable": true
}

// Sessão encerrada
{
  "type": "session.closed",
  "reason": "client_request",
  "total_duration_ms": 45000,
  "segments_transcribed": 12
}
```

---

## 9. CLI (Extensão)

Comandos adicionais ao CLI compartilhado:

```bash
theo transcribe <file> --model faster-whisper-large-v3   # Transcreve arquivo
theo transcribe <file> --model faster-whisper-large-v3 --format srt  # Gera legenda
theo transcribe --stream --model faster-whisper-large-v3  # Streaming do microfone
theo translate <file> --model faster-whisper-large-v3     # Traduz para inglês
```

Comandos compartilhados (`theo pull`, `theo list`, `theo serve`, `theo ps`, `theo remove`, `theo inspect`) funcionam para modelos STT e TTS igualmente.

---

## 10. Model Registry — Manifesto STT

Extensão do mesmo `theo.yaml` do TTS:

```yaml
name: faster-whisper-large-v3
version: 3.0.0
engine: faster-whisper
type: stt
description: "Faster Whisper Large V3 - encoder-decoder STT"

capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["auto", "en", "pt", "es", "ja", "zh"]
  sample_rate: 16000
  word_timestamps: true
  translation: true
  partial_transcripts: true

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
  vad_filter: false  # VAD é feito no runtime, não na engine
```

**Campo `type`**: `stt` ou `tts`. O registry usa esse campo para rotear requests ao tipo correto de worker.

**Campo `architecture`**: informa o runtime sobre o modelo de inferência:

| Architecture | Exemplos | Implicações |
|---|---|---|
| `encoder-decoder` | Whisper, Distil-Whisper | Processa chunks acumulados, gera texto de uma vez |
| `ctc` | WeNet CTC, Wav2Vec2 CTC | Character/token-level output, baixa latência |
| `streaming-native` | Paraformer, WeNet streaming | Streaming verdadeiro, partial transcripts nativos |

O runtime adapta o comportamento de windowing e partial transcript com base na architecture declarada.

---

## 11. Interface de Backend STT

Cada backend STT implementa a seguinte interface Python:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass
from enum import Enum


class STTArchitecture(Enum):
    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"


@dataclass
class TranscriptSegment:
    """Segmento de transcrição (partial ou final)."""
    text: str
    is_final: bool
    segment_id: int
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    confidence: float | None = None
    words: list[dict] | None = None  # [{"word": "olá", "start": 0.0, "end": 0.5}]


@dataclass
class BatchResult:
    """Resultado de transcrição batch (arquivo completo)."""
    text: str
    language: str
    duration: float
    segments: list[dict]
    words: list[dict] | None = None


class STTBackend(ABC):
    """Interface que todo backend STT deve implementar."""

    @property
    @abstractmethod
    def architecture(self) -> STTArchitecture:
        """Retorna a arquitetura do modelo."""
        ...

    @abstractmethod
    async def load(self, model_path: str, config: dict) -> None:
        """Carrega o modelo em memória."""
        ...

    @abstractmethod
    async def transcribe_file(
        self, audio_path: str, language: str | None, **kwargs
    ) -> BatchResult:
        """Transcreve arquivo de áudio completo (batch)."""
        ...

    @abstractmethod
    async def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes], language: str | None, **kwargs
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcreve áudio em streaming.

        Recebe: AsyncIterator de chunks PCM 16-bit 16kHz mono.
        Produz: TranscriptSegment (partial e final).

        Para encoder-decoder: o runtime acumula áudio e chama periodicamente.
        Para streaming-native: a engine processa frame a frame.
        Para CTC: a engine produz character-level output progressivo.
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Descarrega o modelo da memória."""
        ...

    @abstractmethod
    async def health(self) -> dict:
        """Retorna status do backend."""
        ...
```

### Adaptação por Arquitetura

O runtime adapta o pipeline de streaming baseado na `architecture` declarada:

**Encoder-decoder (Whisper):**

```
Audio frames → Ring Buffer → Accumulate (2-5s window) → Engine.transcribe_stream()
                                                              ↓
                                                    TranscriptSegment (final only*)
```

*Partial transcripts são simulados pelo runtime re-processando a window a cada N ms (configurable, default 500ms). Isso é custoso em compute, então partial é best-effort para encoder-decoder.

**CTC:**

```
Audio frames → Ring Buffer → Engine.transcribe_stream() (frame by frame)
                                        ↓
                              TranscriptSegment (partial + final)
```

**Streaming-native:**

```
Audio frames → Engine.transcribe_stream() (frame by frame, engine gerencia estado)
                              ↓
                    TranscriptSegment (partial + final, nativos)
```

---

## 12. Comunicação Runtime ↔ Worker (gRPC)

Extensão do protobuf compartilhado:

```protobuf
service STTWorker {
  // Transcrição batch (arquivo)
  rpc TranscribeFile (TranscribeFileRequest) returns (TranscribeFileResponse);

  // Transcrição streaming
  rpc TranscribeStream (stream AudioFrame) returns (stream TranscriptEvent);

  // Cancelamento
  rpc Cancel (CancelRequest) returns (CancelResponse);

  // Health
  rpc Health (HealthRequest) returns (HealthResponse);
}

message TranscribeFileRequest {
  string request_id = 1;
  bytes audio_data = 2;
  string language = 3;
  string response_format = 4;
  float temperature = 5;
  repeated string timestamp_granularities = 6;
}

message TranscribeFileResponse {
  string text = 1;
  string language = 2;
  float duration = 3;
  repeated Segment segments = 4;
  repeated Word words = 5;
}

message AudioFrame {
  string session_id = 1;
  bytes data = 2;          // PCM 16-bit 16kHz mono
  bool is_last = 3;
}

message TranscriptEvent {
  string session_id = 1;
  string type = 2;         // "partial", "final"
  string text = 3;
  int32 segment_id = 4;
  int64 start_ms = 5;
  int64 end_ms = 6;
  string language = 7;
  float confidence = 8;
  repeated Word words = 9;
}

message Segment {
  int32 id = 1;
  float start = 2;
  float end = 3;
  string text = 4;
  float avg_logprob = 5;
  float no_speech_prob = 6;
}

message Word {
  string word = 1;
  float start = 2;
  float end = 3;
}
```

---

## 13. Ring Buffer e Windowing

### Ring Buffer

Buffer circular de tamanho fixo que armazena áudio recente da sessão:

- **Tamanho default**: 60s de áudio (1,920,000 bytes em PCM 16-bit 16kHz mono).
- **Propósito**: permite reprocessamento após recovery de falha, e acumulação de windows para encoder-decoder.
- **Implementação**: array pré-alocado com ponteiros de read/write. Zero-copy quando possível.

### Windowing por Arquitetura

| Arquitetura | Window size | Overlap | Reprocessa window? |
|---|---|---|---|
| encoder-decoder | 5s (acumulado até VAD silence) | 0 | Sim (para partials) |
| ctc | 30ms (frame a frame) | 0 | Não |
| streaming-native | 30ms (frame a frame) | Engine-defined | Não |

---

## 14. Backends Iniciais

### Faster-Whisper

- Engine: CTranslate2 (C++ com bindings Python).
- Modelos: Whisper tiny → large-v3, Distil-Whisper.
- Arquitetura: encoder-decoder.
- Streaming: simulado via windowing no runtime.
- Ideal para: qualidade máxima de transcrição, batch, GPU.

### WeNet (Fase 2)

- Engine: Python/C++ com LibTorch.
- Modelos: CTC e attention-based.
- Arquitetura: ctc / streaming-native.
- Streaming: nativo.
- Ideal para: latência ultra-baixa, streaming real.

---

## 15. Ingestão RTP (Fase 3)

### Escopo

**Apenas ingestão de RTP raw.** O SIP signaling é responsabilidade do PBX (Asterisk/FreeSWITCH). Theo recebe o stream de áudio já decodificado.

### Fluxo

```
Asterisk ──RTP (G.711/PCM)──→ Theo RTP Listener ──→ Codec Decode ──→ Session Manager
```

### Componente: RTP Listener

- Recebe pacotes UDP com payload RTP.
- Extrai áudio, aplica jitter buffer (20ms, configurável).
- Decodifica G.711 μ-law/A-law para PCM 16-bit 16kHz.
- Alimenta Session Manager como se fosse um WebSocket client.

### O que NÃO faz

- SIP INVITE/BYE (Asterisk gerencia).
- Echo cancellation (Asterisk gerencia).
- DTMF detection (Asterisk gerencia).
- Media negotiation (Asterisk gerencia).

---

## 16. Roadmap

### Fase 1 — STT Batch + Streaming Básico (6 semanas)

**Objetivo**: API funcional para transcrição de arquivo e streaming básico com Faster-Whisper.

**Entregáveis:**

- `POST /v1/audio/transcriptions` com Faster-Whisper como backend.
- `POST /v1/audio/translations` para tradução.
- Formato de resposta: `json`, `verbose_json`, `text`, `srt`, `vtt`.
- Model registry compartilhado com TTS (campo `type: stt`).
- CLI: `theo transcribe <file>`, `theo translate <file>`.
- Worker Faster-Whisper como subprocess com gRPC.
- Testes de latência batch.

**Critério de sucesso**: `curl -F file=@audio.wav -F model=faster-whisper-large-v3 http://localhost:8000/v1/audio/transcriptions` retorna transcrição correta.

**Estratégia**: NÃO reimplementar Speaches. Usar Faster-Whisper diretamente via Python, empacotado como worker Theo. O valor está na integração com o runtime unificado, não em reescrever inference.

### Fase 2 — Streaming Real + Session Manager (8 semanas)

**Entregáveis:**

- `WS /v1/realtime` com protocolo de eventos definido.
- Session Manager com estados (INIT → ACTIVE → SILENCE → CLOSING → CLOSED).
- VAD via Silero VAD no runtime.
- Ring buffer com windowing adaptativo por arquitetura.
- Partial e final transcripts via WebSocket.
- CLI: `theo transcribe --stream`.
- Segundo backend STT (WeNet) demonstrando model-agnostic.

**Critério de sucesso**: sessão WebSocket de 30 minutos sem degradação de latência, com recovery de falha de worker.

### Fase 3 — Telefonia + Scheduler Avançado (8 semanas)

**Entregáveis:**

- RTP Listener com jitter buffer e decode G.711.
- Integração testada com Asterisk (receber áudio de chamada, transcrever em tempo real).
- Scheduler com priorização: realtime (WebSocket/RTP) > batch (file upload).
- Orçamento de latência por sessão no scheduler.
- Co-scheduling STT + TTS (para agentes full-duplex).

---

## Architecture Decision Records (ADRs)

---

### ADR-001 — Runtime Unificado STT + TTS

**Status:** Aceito

**Decisão:** STT e TTS são módulos do mesmo runtime (Theo OpenVoice), não produtos separados.

**Justificativa:**

- Compartilham 70%+ da infraestrutura: API server, registry, scheduler, CLI, observabilidade, Docker.
- Um binário simplifica deploy e operação.
- Agentes de voz precisam de ambos — oferecer num único `theo serve` é o UX correto.
- Model registry com campo `type` diferencia naturalmente.

**Trade-off:** complexidade do binário único aumenta, mas é compensada pela eliminação de duplicação.

---

### ADR-002 — Não Reimplementar Speaches

**Status:** Aceito

**Decisão:** Fase 1 NÃO replica Speaches. Usa Faster-Whisper diretamente como worker, empacotado no runtime Theo.

**Justificativa:**

- Reimplementar funcionalidade existente é desperdício de tempo.
- O valor do Theo está no runtime (registry, scheduler, session manager, CLI unificado), não na inference.
- Faster-Whisper já é maduro e testado.
- Energia deve ir para os diferenciais (session manager, model-agnostic, streaming events protocol).

---

### ADR-003 — VAD no Runtime, Não na Engine

**Status:** Aceito

**Decisão:** Voice Activity Detection roda no runtime (Silero VAD), não dentro da engine STT.

**Justificativa:**

- Comportamento consistente entre engines diferentes.
- Engines como Whisper têm VAD próprio mas com comportamento inconsistente.
- Runtime controla o pipeline de segmentação independente do modelo.
- Silero VAD é leve (~2ms por frame) e roda em CPU sem impacto.

**Trade-off:** engines com VAD nativo superior (se houver) não aproveitam sua vantagem. Configurável via manifesto (`vad_filter: true` na engine_config permite VAD dual).

---

### ADR-004 — Windowing Adaptativo por Arquitetura

**Status:** Aceito

**Decisão:** O runtime adapta o pipeline de streaming baseado na `architecture` declarada no manifesto.

**Justificativa:**

- Encoder-decoder (Whisper) precisa acumular áudio antes de processar. Streaming é simulado.
- CTC processa frame a frame com output incremental.
- Streaming-native (Paraformer) tem pipeline próprio.
- Uma interface única (`transcribe_stream`) com comportamento adaptado é melhor que forçar todas as engines no mesmo modelo.

---

### ADR-005 — Ring Buffer Pre-alocado

**Status:** Aceito

**Decisão:** Cada sessão tem um ring buffer de tamanho fixo (default 60s) pré-alocado.

**Justificativa:**

- Elimina allocations durante streaming (requisito para latência previsível).
- Permite reprocessamento após recovery de falha.
- 60s × 16kHz × 2 bytes = 1.9MB por sessão — custo aceitável.
- Tamanho fixo previne memory leak em sessões longas.

---

### ADR-006 — Protocolo de Eventos WebSocket

**Status:** Aceito

**Decisão:** Protocolo de eventos JSON inspirado na API Realtime da OpenAI, mas simplificado para STT-only.

**Justificativa:**

- Compatibilidade conceitual com OpenAI facilita adoção.
- JSON legível facilita debugging.
- Tipos de evento explícitos (`vad.speech_start`, `transcript.partial`, `transcript.final`) são mais claros que SSE genérico.
- Overhead de JSON parse é negligível comparado à inferência.

**Subset implementado (vs OpenAI Realtime):**

| OpenAI Realtime | Theo | Status |
|---|---|---|
| `session.create` | `session.configure` | Simplificado |
| `input_audio_buffer.append` | Binary WebSocket message | Simplificado |
| `input_audio_buffer.commit` | `input_audio_buffer.commit` | Igual |
| `response.audio_transcript.delta` | `transcript.partial` | Renomeado |
| `conversation.item.created` | Não implementado | Fora de escopo |
| Response/turn management | Não implementado | Fora de escopo (é LLM) |

---

### ADR-007 — Escopo de Telefonia: Apenas RTP Raw

**Status:** Aceito

**Decisão:** Na Fase 3, Theo implementa apenas ingestão RTP raw. SIP signaling fica fora.

**Justificativa:**

- SIP é um protocolo complexo (INVITE, ACK, BYE, re-INVITE, auth, NAT traversal). Implementar é um projeto em si.
- Asterisk/FreeSWITCH já resolvem SIP de forma madura.
- O caso de uso real é: Asterisk recebe chamada, faz bridge de mídia para Theo via RTP.
- Focar em RTP raw mantém escopo controlável e testável.
- Componentes necessários: UDP listener, jitter buffer, G.711 decode — todos bem definidos.

---

## Encerramento

Theo OpenVoice STT completa o runtime unificado de voz. Compartilha infraestrutura com TTS onde faz sentido, e adiciona componentes específicos (Session Manager, VAD, Ring Buffer) onde STT tem requisitos próprios. A estratégia é não reimplementar o que já existe (Faster-Whisper, Silero VAD), mas criar a **camada de runtime** que falta: lifecycle de modelos, session management, streaming protocol, e extensibilidade model-agnostic.