# Theo OpenVoice -- Plano Estrategico do Milestone M5

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Planejado
**Data**: 2026-02-08

**Autora**: Sofia Castellani (Principal Solution Architect)

---

## 1. Objetivo

M5 e o primeiro milestone da Fase 2 (Streaming em Tempo Real). E o milestone de maior complexidade tecnica ate aqui -- introduz duas camadas de streaming concorrentes (WebSocket + gRPC bidirecional), deteccao de atividade vocal (VAD), e um protocolo de eventos JSON que sera o contrato de comunicacao com clientes de streaming.

### O que M5 entrega

Um usuario pode conectar via WebSocket, enviar audio em tempo real, e receber eventos de transcricao (partial e final) com deteccao automatica de fala e silencio. O fluxo completo:

```
Cliente WebSocket                        Theo Runtime                         Worker gRPC
    |                                        |                                    |
    |-- WS connect (/v1/realtime) ---------> |                                    |
    |<- session.created --------------------|                                    |
    |                                        |                                    |
    |-- binary audio frames ---------------> |                                    |
    |                                        |-- preprocessing (frame-by-frame) --|
    |                                        |-- energy pre-filter --------------|
    |                                        |-- Silero VAD --------------------|
    |                                        |                                    |
    |<- vad.speech_start -------------------|                                    |
    |                                        |-- gRPC AudioFrame stream -------->|
    |                                        |                                    |-- inference
    |                                        |<- gRPC TranscriptEvent ----------|
    |<- transcript.partial -----------------|                                    |
    |                                        |                                    |
    |                                        |-- VAD: silencio detectado --------|
    |                                        |<- gRPC TranscriptEvent (final) --|
    |                                        |-- post-processing (ITN) ---------|
    |<- transcript.final -------------------|                                    |
    |<- vad.speech_end ---------------------|                                    |
```

### O que M5 habilita

- **Base para Session Manager (M6)**: O WebSocket handler, VAD e gRPC bidirecional de M5 sao os alicerces sobre os quais M6 constroi a maquina de estados completa, Ring Buffer, WAL e LocalAgreement.
- **Validacao de latencia**: Primeira medicao real de TTFB (Time to First Byte) em streaming. Target: <=300ms.
- **Protocolo de eventos estabilizado**: O contrato JSON (session.created, vad.speech_start, transcript.partial, transcript.final, etc.) e definido e validado aqui. M6+ estende, nao reescreve.

### Criterio de sucesso (Demo Goal)

```
wscat -c ws://localhost:8000/v1/realtime?model=faster-whisper-tiny
# -> {"type": "session.created", "session_id": "sess_abc123", ...}
# (enviar audio binario com fala)
# -> {"type": "vad.speech_start", "timestamp_ms": 1500}
# -> {"type": "transcript.partial", "text": "ola como", "segment_id": 0, ...}
# -> {"type": "transcript.final", "text": "ola como posso ajudar", "segment_id": 0, ...}
# -> {"type": "vad.speech_end", "timestamp_ms": 4000}
```

TTFB <=300ms (primeiro partial apos receber segmento com fala).

### Conexao com o Roadmap

```
PRD Fase 2 (Streaming)
  M5: WebSocket + VAD      [<-- ESTE MILESTONE]
  M6: Session Manager
  M7: Segundo Backend
```

M5 e pre-requisito direto de M6 e M8. A separacao entre M5 (infraestrutura de streaming + VAD) e M6 (maquina de estados + Ring Buffer + LocalAgreement) foi uma decisao deliberada de gestao de risco: cada milestone tem complexidade suficiente para justificar isolamento.

---

## 2. Visao Geral da Arquitetura

### 2.1 Componentes Novos (M5) vs. Reutilizados (M1-M4)

```
+---------------------------------------------------------------------+
|                         FastAPI (existente)                           |
|                                                                      |
|  POST /v1/audio/transcriptions  (M3) -- sem mudanca                 |
|  POST /v1/audio/translations    (M3) -- sem mudanca                 |
|  GET  /health                   (M3) -- sem mudanca                 |
|                                                                      |
|  WS   /v1/realtime              (M5) -- NOVO                        |
|                                                                      |
+------+--------------------------------------------------------------+
       |
       v
+------+--------------------------------------------------------------+
|                    WebSocket Handler (NOVO)                           |
|                                                                      |
|  - Handshake: recebe model, language via query params                |
|  - Recebe frames binarios (audio PCM)                                |
|  - Recebe comandos JSON (session.configure, session.cancel, ...)     |
|  - Envia eventos JSON (session.created, transcript.partial, ...)     |
|  - Thin handler: delega toda logica para StreamingSession            |
|                                                                      |
+------+--------------------------------------------------------------+
       |
       v
+------+--------------------------------------------------------------+
|               StreamingSession (NOVO - simplificada)                  |
|                                                                      |
|  - Estado: ACTIVE / CLOSED (simplificado para M5)                    |
|  - Coordena: preprocessing -> VAD -> gRPC -> post-processing         |
|  - Gera session_id unico                                             |
|  - Tracking de segment_id incremental                                |
|  - Timeout basico (inatividade -> CLOSED)                            |
|  - M6 substitui por SessionManager com 6 estados                     |
|                                                                      |
+------+------+------+------------------------------------------------+
       |      |      |
       v      v      v
+------+--+  ++---------+  +---------+---------------------------+
| Preprocessing |  | VAD        |  | gRPC Bidirecional (NOVO)        |
| (REUTILIZADO) |  | (NOVO)     |  |                                 |
|               |  |            |  | runtime --> stream AudioFrame    |
| AudioStage    |  | Energy     |  | runtime <-- stream TranscriptEvent|
| (frame-by-    |  | pre-filter |  |                                 |
|  frame)       |  | + Silero   |  | Worker servicer: TranscribeStream|
|               |  |            |  | (IMPLEMENTAR - hoje UNIMPLEMENTED)|
+---------------+  +------------+  +---------------------------------+
```

### 2.2 O que ja existe vs. o que M5 cria

| Artefato | Status | Pacote |
|----------|--------|--------|
| `SessionState` enum (6 estados) | Existe (`_types.py`) | `theo._types` |
| `VADSensitivity` enum | Existe (`_types.py`) | `theo._types` |
| `TranscriptSegment` dataclass | Existe (`_types.py`) | `theo._types` |
| `AudioStage` ABC + stages concretos | Existe (M4) | `theo.preprocessing` |
| `AudioPreprocessingPipeline.process()` (batch) | Existe (M4) | `theo.preprocessing.pipeline` |
| `PostProcessingPipeline.process()` | Existe (M4) | `theo.postprocessing.pipeline` |
| `stt_worker.proto` com `TranscribeStream` RPC | Existe (M1) | `theo.proto` |
| `STTWorkerServicer.TranscribeStream` (UNIMPLEMENTED) | Existe como stub (M2) | `theo.workers.stt.servicer` |
| `STTBackend.transcribe_stream()` ABC | Existe como interface (M1) | `theo.workers.stt.interface` |
| `WorkerManager` (spawn, health, crash detection) | Existe (M2) | `theo.workers.manager` |
| `Scheduler` (batch routing) | Existe (M3) | `theo.scheduler` |
| `SessionError`, `SessionNotFoundError`, `SessionClosedError` | Existe (M1) | `theo.exceptions` |
| `session/__init__.py` | Existe (vazio) | `theo.session` |
| `create_app()` factory com DI via `app.state` | Existe (M3/M4) | `theo.server.app` |
| WebSocket endpoint | **NOVO** | `theo.server.routes.realtime` |
| WebSocket protocol (event types, serialization) | **NOVO** | `theo.server.models.events` |
| `StreamingSession` (estado simplificado) | **NOVO** | `theo.session.streaming` |
| `StreamingPreprocessor` (frame-by-frame adapter) | **NOVO** | `theo.preprocessing.streaming` |
| Energy pre-filter | **NOVO** | `theo.vad.energy` |
| Silero VAD integration | **NOVO** | `theo.vad.silero` |
| VAD coordinator | **NOVO** | `theo.vad.detector` |
| gRPC streaming client (runtime-side) | **NOVO** | `theo.scheduler.streaming` |
| `TranscribeStream` servicer implementation | **NOVO** | `theo.workers.stt.servicer` |
| `FasterWhisperBackend.transcribe_stream()` | **NOVO** | `theo.workers.stt.faster_whisper` |
| Backpressure controller | **NOVO** | `theo.session.backpressure` |
| Streaming metrics | **NOVO** | Integrado nos componentes |

### 2.3 Data Flow Detalhado

```
WebSocket frame (binary, PCM any-SR)
    |
    v
StreamingPreprocessor
    |  Reutiliza AudioStage.process() frame-by-frame
    |  Resample -> DC Remove -> Gain Normalize
    |  Output: float32 numpy array, 16kHz mono
    |
    v
EnergyPreFilter
    |  RMS do frame < threshold E spectral flatness > 0.8?
    |  SIM -> silencio (nao chama Silero)
    |  NAO -> continua
    |  Custo: ~0.1ms/frame
    |
    v
SileroVAD
    |  Probabilidade de fala > threshold?
    |  SIM -> speech detected
    |  NAO -> silence detected
    |  Custo: ~2ms/frame
    |
    v
StreamingSession
    |  Gerencia transicoes:
    |  - Speech start -> emite vad.speech_start, inicia gRPC stream
    |  - Speech ongoing -> envia frames ao worker via gRPC
    |  - Silence detected -> emite vad.speech_end, flush worker
    |  - Worker retorna TranscriptEvent -> emite transcript.partial/final
    |
    v
gRPC TranscribeStream (bidirecional)
    |  Runtime envia: stream de AudioFrame (PCM 16-bit 16kHz)
    |  Worker retorna: stream de TranscriptEvent (partial/final)
    |
    v
Post-processing (apenas para transcript.final)
    |  Reutiliza PostProcessingPipeline.process(text)
    |  ITN: "dois mil" -> "2000"
    |
    v
WebSocket event JSON
    {"type": "transcript.final", "text": "2000", ...}
```

---

## 3. Decisoes de Design para M5

### 3.1 Session Manager Simplificada (M5 vs. M6)

M5 usa uma `StreamingSession` simplificada com apenas dois estados logicos: ACTIVE e CLOSED. A maquina de estados completa (INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED) e escopo de M6.

| Aspecto | M5 (Simplificada) | M6 (Completa) |
|---------|-------------------|---------------|
| Estados | ACTIVE, CLOSED | INIT, ACTIVE, SILENCE, HOLD, CLOSING, CLOSED |
| Timeouts | Inatividade geral (60s) | Por estado (INIT: 30s, SILENCE: 30s, HOLD: 5min, CLOSING: 2s) |
| Ring Buffer | Nao -- frames vao direto ao gRPC | Sim -- pre-alocado 60s com read fence |
| WAL / Recovery | Nao | Sim -- recovery sem duplicacao |
| LocalAgreement | Nao -- partials do engine nativo | Sim -- comparacao entre passes |
| Cross-segment context | Nao | Sim -- ultimos 224 tokens como initial_prompt |

**Justificativa**: Separar a complexidade permite validar WebSocket + VAD + gRPC bidirecional isoladamente. Se esses fundamentos funcionam, M6 constroi sobre base solida. Se tentassemos tudo em um milestone, bugs de WebSocket se misturariam com bugs de Ring Buffer, tornando diagnostico impossivel.

### 3.2 Preprocessing em Modo Streaming

Os stages de M4 (`ResampleStage`, `DCRemoveStage`, `GainNormalizeStage`) ja aceitam frames individuais via `process(audio, sample_rate)`. O que M5 precisa e um adapter que:

1. Recebe bytes crus do WebSocket (PCM qualquer SR)
2. Converte para numpy float32
3. Aplica stages sequencialmente
4. Retorna numpy float32 16kHz mono

Nao e necessario reescrever os stages -- apenas criar um adapter fino (`StreamingPreprocessor`) que expoe `process_frame(raw_bytes, input_sample_rate)`.

### 3.3 VAD no Runtime

Conforme ADR-003, VAD roda no runtime (nao na engine). M5 implementa dois componentes:

1. **Energy Pre-filter**: RMS + spectral flatness. Custo ~0.1ms/frame. Filtra 60-70% dos frames silenciosos sem chamar Silero.
2. **Silero VAD**: Modelo neural leve (~2ms/frame CPU). Classifica frame como speech ou silence.

O `VADDetector` coordena ambos e emite eventos (`speech_start`, `speech_end`) que a `StreamingSession` consome.

### 3.4 gRPC Bidirecional

O proto ja define `rpc TranscribeStream(stream AudioFrame) returns (stream TranscriptEvent)`. M5 implementa ambos os lados:

- **Runtime-side** (`StreamingGRPCClient`): Abre stream gRPC para o worker, envia AudioFrames, recebe TranscriptEvents.
- **Worker-side** (`STTWorkerServicer.TranscribeStream`): Recebe AudioFrames, delega ao backend, retorna TranscriptEvents.

Stream break = crash detection (conforme perspectiva Viktor no ROADMAP).

### 3.5 O que NAO fazer (fronteiras YAGNI para M5)

| Item | Por que nao | Quando |
|------|-------------|--------|
| Ring Buffer com read fence | Complexidade de M6. Sem Ring Buffer, frames vao direto ao worker. | M6 |
| WAL in-memory para recovery | Depende do Ring Buffer e Session Manager completo. | M6 |
| Maquina de estados completa (6 estados) | M5 valida a infraestrutura; M6 adiciona a logica de estados. | M6 |
| LocalAgreement para partials | Algoritmo complexo que depende de Ring Buffer. M5 usa partials nativos. | M6 |
| Cross-segment context (initial_prompt) | Depende de tracking de segmentos e Session Manager. | M6 |
| Entity Formatting stage | Post-processing domain-specific, sem caso de uso concreto agora. | Futuro |
| Hot Word Correction stage | Depende de hot words por sessao + Levenshtein. | M6 |
| CLI `theo transcribe --stream` | Depende de Session Manager estavel. | M6 |
| Denoise stage | Desativado por default. Necessario para telefonia. | M8 |

---

## 4. Epics

### Epic 1 -- WebSocket Foundation

Estabelecer o endpoint WebSocket `/v1/realtime`, o protocolo de eventos JSON, o handshake, o heartbeat e a infraestrutura de comunicacao bidirecional com o cliente.

**Racional**: O WebSocket e a porta de entrada do streaming. Sem ele, nenhum outro componente de M5 e acessivel. O protocolo de eventos precisa ser definido e estabilizado aqui -- M6+ estende sem reescrever.

**Tasks**:
- T5-01: Pydantic models para eventos WebSocket (request + response)
- T5-02: Endpoint WebSocket `/v1/realtime` com handshake
- T5-03: Protocol handler (dispatch de comandos JSON + frames binarios)
- T5-04: Heartbeat (ping/pong) e timeout de inatividade

### Epic 2 -- Voice Activity Detection

Implementar energy pre-filter e integracao com Silero VAD para deteccao de fala em tempo real. Emitir eventos `vad.speech_start` e `vad.speech_end`.

**Racional**: VAD e o que transforma um stream de audio bruto em segmentos de fala uteis. Sem VAD, o runtime enviaria silencio ao worker, desperdicando GPU. O energy pre-filter reduz chamadas ao Silero em 60-70% -- essencial para eficiencia.

**Tasks**:
- T5-05: Energy pre-filter (RMS + spectral flatness)
- T5-06: Integracao Silero VAD com sensitivity levels
- T5-07: VAD coordinator (orquestra pre-filter + Silero, emite eventos)

### Epic 3 -- Streaming Pipeline

Conectar preprocessing em modo streaming, gRPC bidirecional e post-processing para que frames de audio do WebSocket resultem em eventos de transcricao.

**Racional**: Este e o "plumbing" critico -- onde WebSocket, VAD, gRPC e worker se encontram. A complexidade esta na coordenacao de streams assincronos concorrentes (audio entrando pelo WS, frames saindo pelo gRPC, transcripts voltando pelo gRPC, eventos saindo pelo WS).

**Tasks**:
- T5-08: Streaming preprocessor (adapter frame-by-frame sobre stages M4)
- T5-09: gRPC bidirecional -- runtime side (StreamingGRPCClient)
- T5-10: gRPC bidirecional -- worker side (TranscribeStream no servicer)
- T5-11: FasterWhisperBackend.transcribe_stream() (implementacao)

### Epic 4 -- Session e Flow Control

Implementar a `StreamingSession` simplificada que coordena o fluxo completo, backpressure, manual commit, e emissao de eventos. Adicionar metricas de streaming.

**Racional**: A `StreamingSession` e o orquestrador central que conecta WebSocket handler, preprocessing, VAD, gRPC client e post-processing. Backpressure e essencial para prevenir OOM quando clientes enviam audio mais rapido que real-time.

**Tasks**:
- T5-12: StreamingSession (estado simplificado, coordenacao do fluxo)
- T5-13: Backpressure (rate_limit, frames_dropped)
- T5-14: Manual commit (input_audio_buffer.commit)
- T5-15: Metricas de streaming (ttfb, final_delay, active_sessions, vad_events)

### Epic 5 -- Integracao e Validacao

Testes end-to-end, teste de estabilidade, e validacao do fluxo completo.

**Racional**: Streaming e inerentemente mais dificil de testar que batch. Precisamos de testes que validem o fluxo completo (WS connect -> audio -> events), backpressure, e estabilidade sob carga.

**Tasks**:
- T5-16: Testes unitarios (cada componente isolado)
- T5-17: Testes de integracao WebSocket (fluxo completo com mock worker)
- T5-18: Teste de estabilidade e documentacao

---

## 5. Tasks (Detalhadas)

### T5-01: Pydantic Models para Eventos WebSocket

**Epic**: E1 -- WebSocket Foundation
**Estimativa**: S (1-2 dias)
**Dependencias**: Nenhuma

**Contexto/Motivacao**: O protocolo WebSocket usa mensagens JSON tipadas. Precisamos de modelos Pydantic para serializar/deserializar eventos de forma type-safe. Esses modelos sao o contrato formal do protocolo de streaming e serao referencia para toda a Fase 2.

**Escopo**:

| In | Out |
|----|-----|
| Modelos para eventos server->client: `SessionCreatedEvent`, `VADSpeechStartEvent`, `VADSpeechEndEvent`, `TranscriptPartialEvent`, `TranscriptFinalEvent`, `SessionRateLimitEvent`, `SessionFramesDroppedEvent`, `StreamingErrorEvent`, `SessionClosedEvent` | Eventos de M6: `session.hold` |
| Modelos para comandos client->server: `SessionConfigureCommand`, `SessionCancelCommand`, `InputAudioBufferCommitCommand`, `SessionCloseCommand` | Comandos de M6: hot words em `session.configure` (pass-through ao worker) |
| `SessionConfig` Pydantic model (vad_sensitivity, silence_timeout_ms, language, hot_words, preprocessing overrides, input_sample_rate) | |
| Base discriminated union com campo `type` para dispatch | |
| Testes unitarios de serialization/deserialization | |

**Entregaveis**:
- `src/theo/server/models/events.py` -- todos os modelos de eventos
- `tests/unit/test_ws_event_models.py`

**DoD**:
1. Todos os event types do PRD secao 9 (WS /v1/realtime) estao modelados
2. Cada modelo serializa para JSON com campo `type` como discriminator
3. Desserializacao de comandos client->server funciona com validacao
4. Modelos sao `frozen=True` (Pydantic `model_config = ConfigDict(frozen=True)`)
5. mypy e ruff passam

---

### T5-02: Endpoint WebSocket `/v1/realtime` com Handshake

**Epic**: E1 -- WebSocket Foundation
**Estimativa**: M (3-4 dias)
**Dependencias**: T5-01

**Contexto/Motivacao**: O endpoint WebSocket e o ponto de entrada para todo o streaming. O handshake valida o modelo solicitado (via query param `model`), cria a sessao, e emite `session.created`. O handler deve ser thin -- toda logica de estado pertence a `StreamingSession`.

**Escopo**:

| In | Out |
|----|-----|
| Rota WebSocket `WS /v1/realtime?model=<name>&language=<lang>` | Autenticacao/auth |
| Validacao de modelo via `ModelRegistry` no handshake | WebSocket over TLS (configuracao de infra) |
| Emissao de `session.created` apos conexao bem-sucedida | |
| Recepcao de frames binarios (audio) e mensagens JSON (comandos) | |
| Emissao de eventos JSON ao cliente | |
| Tratamento de erros (modelo invalido -> close com codigo 1008) | |
| Integracao com `create_app()` via `app.state` | |
| Registro da rota WebSocket no router FastAPI | |

**Entregaveis**:
- `src/theo/server/routes/realtime.py` -- endpoint WebSocket
- Alteracao em `src/theo/server/app.py` -- include router realtime
- `tests/unit/test_realtime_endpoint.py`

**DoD**:
1. `wscat -c ws://localhost:8000/v1/realtime?model=faster-whisper-tiny` retorna `session.created`
2. Modelo invalido retorna close com codigo 1008 e mensagem de erro
3. Modelo ausente (sem query param) retorna close com codigo 1008
4. Multiplas conexoes simultaneas geram session_ids unicos
5. mypy e ruff passam

---

### T5-03: Protocol Handler (Dispatch de Comandos e Frames)

**Epic**: E1 -- WebSocket Foundation
**Estimativa**: M (3-4 dias)
**Dependencias**: T5-02, T5-01

**Contexto/Motivacao**: O WebSocket recebe dois tipos de mensagem: frames binarios (audio) e mensagens de texto (comandos JSON). O protocol handler faz dispatch: identifica o tipo de mensagem, deserializa comandos, e delega a acao correta. E o "roteador" interno do WebSocket.

**Escopo**:

| In | Out |
|----|-----|
| Dispatch de mensagens binarias -> frame de audio para preprocessing | |
| Dispatch de mensagens JSON -> comando tipado (session.configure, session.cancel, session.close, input_audio_buffer.commit) | |
| `session.configure` aplica configuracao na sessao ativa | |
| `session.cancel` cancela sessao e fecha WebSocket | |
| `session.close` fecha sessao gracefully (flush pendente) | |
| Mensagem JSON invalida -> evento `error` com `recoverable: true` | |
| Tipo de comando desconhecido -> evento `error` com detalhes | |

**Entregaveis**:
- Logica de dispatch dentro de `src/theo/server/routes/realtime.py` (ou extraida para `src/theo/server/ws_protocol.py` se ficar grande)
- `tests/unit/test_ws_protocol.py`

**DoD**:
1. Frame binario e identificado e roteado para processamento de audio
2. Comando JSON valido e deserializado e executado
3. Comando JSON invalido (JSON malformado) emite evento `error` sem fechar conexao
4. `session.close` emite `session.closed` e fecha WebSocket limpo (code 1000)
5. mypy e ruff passam

---

### T5-04: Heartbeat (Ping/Pong) e Timeout de Inatividade

**Epic**: E1 -- WebSocket Foundation
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-02

**Contexto/Motivacao**: Conexoes WebSocket podem virar zombies em redes instaveis (mobile, WiFi com handover). Heartbeat via ping/pong detecta conexoes mortas. Timeout de inatividade previne sessions abertas indefinidamente sem audio.

**Escopo**:

| In | Out |
|----|-----|
| Server envia ping WebSocket a cada 10s (configuravel) | Heartbeat customizavel por sessao |
| Se pong nao recebido em 5s -> fechar sessao | |
| Timeout de inatividade: se nenhum frame de audio recebido em 60s -> fechar sessao | |
| Emite `session.closed` com `reason: "timeout"` ou `reason: "heartbeat_timeout"` | |

**Entregaveis**:
- Logica de heartbeat integrada ao WebSocket handler em `realtime.py`
- `tests/unit/test_ws_heartbeat.py`

**DoD**:
1. Ping enviado a cada 10s (verificavel via log ou mock)
2. Conexao fechada se pong nao recebido em 5s
3. Sessao fechada apos 60s sem frames de audio
4. `session.closed` emitido com reason correto antes do close
5. mypy e ruff passam

---

### T5-05: Energy Pre-filter (RMS + Spectral Flatness)

**Epic**: E2 -- Voice Activity Detection
**Estimativa**: S (1-2 dias)
**Dependencias**: Nenhuma (componente isolado)

**Contexto/Motivacao**: O Silero VAD custa ~2ms/frame. Em ambientes ruidosos, muitos frames sao claramente silencio (RMS muito baixo, ou ruido branco com spectral flatness alta). O energy pre-filter descarta esses frames sem chamar Silero, reduzindo custo em 60-70%.

**Escopo**:

| In | Out |
|----|-----|
| `EnergyPreFilter` que recebe frame numpy float32 e retorna `is_silence: bool` | Filtro adaptativo baseado em SNR |
| Calculo de RMS do frame | |
| Calculo de spectral flatness (geometrica/aritmetica de magnitude FFT) | |
| Threshold configuravel por `VADSensitivity` (high: -50dBFS, normal: -40dBFS, low: -30dBFS) | |
| Criterio: `is_silence = (rms_dbfs < threshold) AND (spectral_flatness > 0.8)` | |
| Latencia: <0.1ms/frame | |

**Entregaveis**:
- `src/theo/vad/__init__.py`
- `src/theo/vad/energy.py` -- `EnergyPreFilter`
- `tests/unit/test_vad_energy.py`

**DoD**:
1. Frame de silencio (zeros) classificado como silence
2. Frame de fala (sine wave 440Hz) classificado como non-silence
3. Ruido branco com amplitude baixa classificado como silence (spectral flatness alta)
4. Sensitivity levels alteram thresholds corretamente (high/normal/low)
5. mypy e ruff passam

---

### T5-06: Integracao Silero VAD com Sensitivity Levels

**Epic**: E2 -- Voice Activity Detection
**Estimativa**: M (3-4 dias)
**Dependencias**: Nenhuma (componente isolado)

**Contexto/Motivacao**: Silero VAD e o classificador neural de fala usado pelo Theo. Roda em CPU, ~2ms/frame, e fornece probabilidade de fala que e comparada contra threshold. Os sensitivity levels (high/normal/low) ajustam o threshold e parametros de debounce (min speech/silence duration).

**Escopo**:

| In | Out |
|----|-----|
| `SileroVADClassifier` que recebe frame numpy float32 16kHz e retorna `speech_probability: float` | VAD treinado custom |
| Lazy loading do modelo Silero (torch.jit) | |
| Threshold por sensitivity: high=0.3, normal=0.5, low=0.7 | |
| Min speech duration: 250ms (frames consecutivos acima do threshold) | |
| Min silence duration: 300ms (frames consecutivos abaixo do threshold) | |
| Max speech duration: 30s (force silence event) | |
| Window de 64ms (1024 samples a 16kHz) como frame de analise | |
| Dependencia: `silero-vad` ou `torch` com modelo ONNX | |

**Entregaveis**:
- `src/theo/vad/silero.py` -- `SileroVADClassifier`
- `tests/unit/test_vad_silero.py`

**DoD**:
1. Frame com fala retorna speech_probability > 0.5 (com modelo real ou mock determinista)
2. Frame silencioso retorna speech_probability < 0.2
3. Min speech duration: speech so e reportada apos 250ms consecutivos
4. Min silence duration: silence so e reportada apos 300ms consecutivos
5. Lazy loading: modelo carregado na primeira chamada, nao no init
6. mypy e ruff passam

---

### T5-07: VAD Coordinator (Orquestra Pre-filter + Silero)

**Epic**: E2 -- Voice Activity Detection
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-05, T5-06

**Contexto/Motivacao**: O `VADDetector` coordena energy pre-filter e Silero VAD em sequencia. Emite eventos de alto nivel (`speech_start`, `speech_end`) que a `StreamingSession` consome. Encapsula a logica de debounce e transicao.

**Escopo**:

| In | Out |
|----|-----|
| `VADDetector` que recebe frame numpy float32 16kHz | |
| Chama `EnergyPreFilter` primeiro; se silence, skip Silero | |
| Se non-silence no pre-filter, chama `SileroVADClassifier` | |
| Emite `VADEvent` (speech_start, speech_end, speech_ongoing, silence) | |
| Gerencia estado interno: tracking de speech/silence duration para debounce | |
| Configuravel via `VADSensitivity` | |
| `VADEvent` dataclass com tipo e timestamp | |

**Entregaveis**:
- `src/theo/vad/detector.py` -- `VADDetector`, `VADEvent`, `VADEventType`
- `tests/unit/test_vad_detector.py`

**DoD**:
1. Sequencia [silencio, silencio, fala, fala, fala, silencio, silencio, silencio] produz speech_start e speech_end nos momentos corretos (respeitando min durations)
2. Frame classificado como silence pelo energy pre-filter nao chama Silero (verificavel via mock)
3. Sensitivity level `high` detecta fala com threshold 0.3
4. Sensitivity level `low` requer threshold 0.7 para detectar fala
5. mypy e ruff passam

---

### T5-08: Streaming Preprocessor (Adapter Frame-by-Frame)

**Epic**: E3 -- Streaming Pipeline
**Estimativa**: S (1-2 dias)
**Dependencias**: Nenhuma (reutiliza M4)

**Contexto/Motivacao**: Os stages de preprocessing de M4 ja processam frames individuais (`process(audio, sample_rate)`). O `StreamingPreprocessor` e um adapter fino que recebe bytes crus do WebSocket, converte para numpy, aplica stages, e retorna o frame processado. Nao reinventa -- reutiliza.

**Escopo**:

| In | Out |
|----|-----|
| `StreamingPreprocessor` que recebe `bytes` (PCM raw) e `input_sample_rate` | Decodificacao de formatos (MP3, etc.) -- streaming recebe PCM raw |
| Converte bytes -> numpy float32 | |
| Aplica stages de M4 em sequencia (resample, dc_remove, gain_normalize) | |
| Retorna numpy float32 16kHz mono | |
| Encapsula conversao de `int16` PCM para `float32` normalizado | |

**Entregaveis**:
- `src/theo/preprocessing/streaming.py` -- `StreamingPreprocessor`
- `tests/unit/test_streaming_preprocessor.py`

**DoD**:
1. Frame PCM 16-bit 44.1kHz convertido para float32 16kHz mono
2. Frame PCM 16-bit 16kHz retorna sem resample (skip)
3. DC offset removido do frame
4. Gain normalizado para -3dBFS
5. mypy e ruff passam

---

### T5-09: gRPC Bidirecional -- Runtime Side (StreamingGRPCClient)

**Epic**: E3 -- Streaming Pipeline
**Estimativa**: L (5-7 dias)
**Dependencias**: T5-10 (implementacao pode ser paralela, mas integracao requer ambos)

**Contexto/Motivacao**: O runtime precisa de um cliente gRPC que abre um stream bidirecional com o worker. Envia `AudioFrame` messages, recebe `TranscriptEvent` messages. Deve detectar stream break (worker crash) e propagar cancelamento. Este e o componente de maior complexidade tecnica de M5.

**Escopo**:

| In | Out |
|----|-----|
| `StreamingGRPCClient` que gerencia stream bidirecional com worker | Connection pooling (M9) |
| Metodo `open_stream(session_id, worker_port) -> StreamHandle` | Dynamic batching (M9) |
| `StreamHandle.send_frame(pcm_bytes)` -- envia AudioFrame ao worker | |
| `StreamHandle.receive_events() -> AsyncIterator[TranscriptEvent]` | |
| `StreamHandle.close()` -- fecha stream gracefully (envia is_last=True) | |
| Deteccao de stream break (worker crash) com timeout configuravel | |
| Traducao de erros gRPC para exceptions Theo (WorkerCrashError, WorkerTimeoutError) | |
| Cancel propagation via gRPC context | |
| Propagacao de hot_words e initial_prompt no primeiro AudioFrame | |

**Entregaveis**:
- `src/theo/scheduler/streaming.py` -- `StreamingGRPCClient`, `StreamHandle`
- `tests/unit/test_streaming_grpc_client.py`

**DoD**:
1. Stream aberto com worker, AudioFrame enviado e TranscriptEvent recebido
2. `is_last=True` enviado no close, worker flushes transcricao restante
3. Worker crash (stream break) detectado e traduzido para `WorkerCrashError`
4. Cancelamento propagado via gRPC context
5. mypy e ruff passam

---

### T5-10: gRPC Bidirecional -- Worker Side (TranscribeStream)

**Epic**: E3 -- Streaming Pipeline
**Estimativa**: M (3-4 dias)
**Dependencias**: T5-11

**Contexto/Motivacao**: O `STTWorkerServicer.TranscribeStream` hoje retorna UNIMPLEMENTED. M5 implementa: recebe stream de AudioFrames, delega ao backend `transcribe_stream()`, e retorna stream de TranscriptEvents. O servicer e thin -- toda logica de inferencia esta no backend.

**Escopo**:

| In | Out |
|----|-----|
| Implementacao de `TranscribeStream` no `STTWorkerServicer` | |
| Recebe `stream AudioFrame`, agrupa bytes, passa ao backend | |
| Recebe `AsyncIterator[TranscriptSegment]` do backend, converte para `TranscriptEvent` proto | |
| Detecta `is_last=True` e sinaliza fim do stream ao backend | |
| Tratamento de cancelamento via `context.cancelled()` | |
| Converters: `AudioFrame` proto -> bytes PCM, `TranscriptSegment` -> `TranscriptEvent` proto | |

**Entregaveis**:
- Alteracao em `src/theo/workers/stt/servicer.py` -- implementar `TranscribeStream`
- Alteracao em `src/theo/workers/stt/converters.py` -- converters de streaming
- `tests/unit/test_streaming_servicer.py`

**DoD**:
1. Stream de AudioFrames recebido e delegado ao backend
2. TranscriptEvents (partial e final) retornados ao runtime via stream
3. `is_last=True` no ultimo AudioFrame causa flush do backend
4. Context cancelado encerra stream gracefully
5. mypy e ruff passam

---

### T5-11: FasterWhisperBackend.transcribe_stream()

**Epic**: E3 -- Streaming Pipeline
**Estimativa**: M (3-4 dias)
**Dependencias**: Nenhuma (componente isolado)

**Contexto/Motivacao**: O `FasterWhisperBackend` hoje so implementa `transcribe_file()`. M5 adiciona `transcribe_stream()` que recebe chunks de audio via AsyncIterator e produz TranscriptSegments. Para encoder-decoder (Whisper), a estrategia M5 e simples: acumula audio ate `is_last` ou ate um threshold de duracao, faz inference, retorna resultado. LocalAgreement (partials inteligentes) e escopo de M6.

**Escopo**:

| In | Out |
|----|-----|
| Implementacao de `transcribe_stream()` no `FasterWhisperBackend` | LocalAgreement (M6) |
| Recebe `AsyncIterator[bytes]` de chunks PCM 16-bit 16kHz mono | Cross-segment context (M6) |
| Acumula audio em buffer interno ate `is_last` ou threshold de duracao (5s) | |
| Faz inference com `faster-whisper` no buffer acumulado | |
| Retorna `TranscriptSegment` com `is_final=True` apos cada inference | |
| Retorna `TranscriptSegment` com `is_final=False` (partial) se engine suportar partials nativos | |
| Conversao PCM bytes -> numpy float32 para o faster-whisper | |

**Entregaveis**:
- Alteracao em `src/theo/workers/stt/faster_whisper.py` -- implementar `transcribe_stream()`
- `tests/unit/test_faster_whisper_streaming.py`

**DoD**:
1. Chunks de audio acumulados e transcritos quando buffer atinge threshold ou is_last
2. TranscriptSegment retornado com texto, is_final, segment_id, timestamps
3. Audio curto (<500ms) retorna transcricao apos is_last
4. Multiplos segmentos (threshold atingido multiplas vezes) incrementam segment_id
5. mypy e ruff passam

---

### T5-12: StreamingSession (Estado Simplificado, Coordenacao do Fluxo)

**Epic**: E4 -- Session e Flow Control
**Estimativa**: L (5-7 dias)
**Dependencias**: T5-07 (VAD), T5-08 (preprocessor), T5-09 (gRPC client)

**Contexto/Motivacao**: A `StreamingSession` e o orquestrador central de M5. Coordena o fluxo completo: recebe frames do WebSocket handler, aplica preprocessing, passa pelo VAD, encaminha ao worker via gRPC quando ha fala, recebe transcripts do worker, aplica post-processing nos finals, e emite eventos de volta ao handler. Em M5, usa estado simplificado (ACTIVE/CLOSED). Em M6, sera substituida pelo `SessionManager` completo.

**Escopo**:

| In | Out |
|----|-----|
| `StreamingSession` com estado ACTIVE/CLOSED | Maquina de 6 estados (M6) |
| Coordena: preprocessing -> VAD -> gRPC -> post-processing | Ring Buffer (M6) |
| Gera `session_id` unico (UUID) | WAL / recovery (M6) |
| Tracking de `segment_id` incremental | LocalAgreement (M6) |
| Timeout de inatividade (60s sem audio -> CLOSED) | |
| Metodo `process_frame(raw_bytes)` -- entry point do handler | |
| Callback pattern para emissao de eventos ao handler (`on_event: Callable`) | |
| Abertura e fechamento de stream gRPC baseado em eventos VAD | |
| Post-processing (ITN) aplicado apenas em transcript.final | |
| Hot words passados ao worker via AudioFrame | |
| Cleanup de recursos no close (fechar gRPC stream, cancelar tasks) | |

**Entregaveis**:
- `src/theo/session/streaming.py` -- `StreamingSession`
- `tests/unit/test_streaming_session.py`

**DoD**:
1. Sequencia audio -> vad.speech_start -> transcript.partial -> transcript.final -> vad.speech_end funciona end-to-end (com mock worker)
2. Post-processing aplicado apenas em transcript.final (nunca em partial)
3. Session fecha com timeout de inatividade (60s)
4. Cleanup de recursos no close (gRPC stream fechado, sem leaks)
5. mypy e ruff passam

---

### T5-13: Backpressure (Rate Limit e Frames Dropped)

**Epic**: E4 -- Session e Flow Control
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-12

**Contexto/Motivacao**: Se o cliente enviar audio mais rapido que real-time (ex: arquivo via WebSocket), o runtime pode ficar sobrecarregado. Backpressure protege contra OOM e degradacao de latencia. O PRD define dois mecanismos: `session.rate_limit` (aviso) e `session.frames_dropped` (descarte).

**Escopo**:

| In | Out |
|----|-----|
| `BackpressureController` que monitora taxa de ingestao de frames | Backpressure gRPC (M9) |
| Se taxa > 1.2x real-time, emite `session.rate_limit` com `delay_ms` | |
| Se backlog > 10s de audio, descarta frames e emite `session.frames_dropped` | |
| Integrado a `StreamingSession` | |
| Contadores para metricas (`frames_received`, `frames_dropped`) | |

**Entregaveis**:
- `src/theo/session/backpressure.py` -- `BackpressureController`
- `tests/unit/test_backpressure.py`

**DoD**:
1. Audio enviado a 2x real-time emite `session.rate_limit`
2. Backlog > 10s emite `session.frames_dropped` com contagem de ms descartados
3. Frames descartados nao sao enviados ao worker
4. Audio em velocidade normal (1x) nao dispara nenhum evento
5. mypy e ruff passam

---

### T5-14: Manual Commit (input_audio_buffer.commit)

**Epic**: E4 -- Session e Flow Control
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-12

**Contexto/Motivacao**: O cliente pode forcar commit de segmento a qualquer momento via `input_audio_buffer.commit`. Isso faz o runtime sinalizar ao worker para finalizar o segmento atual (enviar `is_last=True` no gRPC stream), produzindo `transcript.final` independente do VAD.

**Escopo**:

| In | Out |
|----|-----|
| Comando `input_audio_buffer.commit` no protocolo WebSocket | |
| Ao receber, sinaliza ao worker que o segmento atual terminou | |
| Worker retorna transcript.final para o audio acumulado | |
| Novo segmento inicia automaticamente (segment_id incrementa) | |

**Entregaveis**:
- Logica integrada em `StreamingSession` e protocol handler
- `tests/unit/test_manual_commit.py`

**DoD**:
1. `input_audio_buffer.commit` durante fala produz transcript.final
2. segment_id incrementa apos commit
3. Audio posterior ao commit pertence ao novo segmento
4. Commit durante silencio (sem audio acumulado) e no-op sem erro
5. mypy e ruff passam

---

### T5-15: Metricas de Streaming

**Epic**: E4 -- Session e Flow Control
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-12

**Contexto/Motivacao**: Metricas de streaming sao essenciais para validar o target de TTFB <=300ms e monitorar saude do sistema em producao. Sao as metricas definidas no ROADMAP para M5.

**Escopo**:

| In | Out |
|----|-----|
| `theo_stt_ttfb_seconds` (Histogram): tempo ate primeiro partial transcript | Metricas de M6 (session_duration, force_committed, confidence) |
| `theo_stt_final_delay_seconds` (Histogram): delay do final apos fim de fala | |
| `theo_stt_active_sessions` (Gauge): sessoes WebSocket ativas agora | |
| `theo_stt_vad_events_total` (Counter): eventos VAD por tipo (speech_start, speech_end) | |
| Instrumentacao nos componentes (StreamingSession, VADDetector) | |
| Exposicao via `/metrics` (endpoint Prometheus existente) | |

**Entregaveis**:
- `src/theo/session/metrics.py` -- definicao e helpers de metricas
- Instrumentacao adicionada a `StreamingSession` e `VADDetector`
- `tests/unit/test_streaming_metrics.py`

**DoD**:
1. TTFB medido e registrado corretamente (diferenca entre primeiro audio com fala e primeiro partial)
2. Final delay medido (diferenca entre vad.speech_end e transcript.final)
3. Active sessions incrementa/decrementa com open/close de sessao
4. VAD events contabilizados por tipo
5. mypy e ruff passam

---

### T5-16: Testes Unitarios (Cada Componente Isolado)

**Epic**: E5 -- Integracao e Validacao
**Estimativa**: M (3-4 dias)
**Dependencias**: Todos os componentes de T5-01 a T5-15

**Contexto/Motivacao**: Cada componente de M5 tem testes unitarios criados junto com a implementacao (DoD de cada task). T5-16 e o momento de revisao: verificar cobertura, adicionar edge cases faltantes, e garantir que os testes sao determinísticos (sem dependencia de timing real).

**Escopo**:

| In | Out |
|----|-----|
| Review de cobertura de testes de todas as tasks M5 | Testes com modelo real (integracao) |
| Edge cases adicionais: frames vazios, frames gigantes, multiplas sessoes simultaneas | Testes de performance/load |
| Testes de serializacao/desserializacao de todos os event types | |
| Testes de lifecycle: open -> audio -> close, open -> close imediato, open -> timeout | |
| Verificacao de que mocks de Silero VAD sao determinísticos | |

**Entregaveis**:
- Complementos nos arquivos de teste existentes
- Revisao e ajuste de mocks para determinismo

**DoD**:
1. Todos os testes passam com `pytest tests/unit/ -v` (incluindo M1-M4)
2. Nenhum teste depende de timing real (sleep/timeout) -- usar mocks de tempo
3. Edge cases cobertos: frame vazio, frame >64KB, JSON malformado, conexao duplicada
4. Total de testes novos M5: >=80
5. mypy e ruff passam

---

### T5-17: Testes de Integracao WebSocket (Fluxo Completo)

**Epic**: E5 -- Integracao e Validacao
**Estimativa**: M (3-4 dias)
**Dependencias**: T5-12, T5-02

**Contexto/Motivacao**: Os testes unitarios validam componentes isolados com mocks. Os testes de integracao validam o fluxo completo: conectar via WebSocket real do `httpx` ou `websockets`, enviar audio, receber eventos. Usam mock do worker (gRPC) mas WebSocket real do FastAPI.

**Escopo**:

| In | Out |
|----|-----|
| Teste: connect -> session.created -> enviar audio com fala -> vad.speech_start -> transcript.partial -> transcript.final -> vad.speech_end | Teste com worker real (modelo carregado) |
| Teste: connect -> session.configure -> verificar config aplicada | |
| Teste: connect -> session.cancel -> session.closed | |
| Teste: connect -> enviar audio rapido demais -> session.rate_limit | |
| Teste: connect -> input_audio_buffer.commit -> transcript.final | |
| Teste: connect -> nenhum audio por 60s -> session.closed (timeout) | |
| Teste: modelo invalido -> close com codigo 1008 | |
| Teste: post-processing aplicado em final, nao em partial | |
| Audio de teste: sine wave 440Hz (fala simulada) + silencio | |
| Mock worker: retorna TranscriptSegments pre-definidos | |

**Entregaveis**:
- `tests/integration/test_ws_streaming.py`
- Fixtures de audio para streaming (frames curtos 20ms)

**DoD**:
1. Fluxo completo WS -> audio -> events funciona end-to-end
2. Todos os event types verificados em pelo menos um teste
3. Post-processing apenas em finals verificado
4. Backpressure testado com audio enviado a 3x real-time
5. mypy e ruff passam

---

### T5-18: Teste de Estabilidade e Documentacao

**Epic**: E5 -- Integracao e Validacao
**Estimativa**: S (1-2 dias)
**Dependencias**: T5-17

**Contexto/Motivacao**: Streaming e propenso a memory leaks e degradacao de latencia em sessoes longas. Um teste de estabilidade de 5 minutos (mais curto que os 30min de M6, proporcional ao escopo simplificado) valida que o sistema nao degrada. Documentacao atualiza CHANGELOG, ROADMAP e CLAUDE.md.

**Escopo**:

| In | Out |
|----|-----|
| Teste de estabilidade: sessao de 5 minutos com audio continuo (mock) | Teste de 30 minutos (M6) |
| Monitorar RSS do processo a cada 10s -- falhar se crescimento > 10MB | |
| Verificar que latencia (TTFB) nao degrada ao longo da sessao | |
| Atualizar CHANGELOG.md com entradas M5 | |
| Atualizar ROADMAP.md com resultado M5 | |

**Entregaveis**:
- `tests/integration/test_ws_stability.py` (marcado como `@pytest.mark.slow`)
- Atualizacoes em `CHANGELOG.md`, `ROADMAP.md`

**DoD**:
1. Sessao de 5 minutos sem degradacao de latencia (TTFB final <= 1.2x TTFB inicial)
2. Crescimento de memoria <= 10MB ao longo da sessao
3. Zero erros nao esperados nos logs
4. CHANGELOG.md atualizado com entradas M5
5. mypy e ruff passam

---

## 6. Sprint Plan

### Sprint 1 (Semanas 1-2): Foundation + VAD

**Objetivo**: Endpoint WebSocket funcional com handshake e VAD completo.

**Demo Goal**: Conectar via WebSocket, enviar audio, ver eventos `vad.speech_start` e `vad.speech_end` nos logs (sem transcricao ainda).

**Tasks**:

| ID | Task | Estimativa | Prioridade |
|----|------|-----------|------------|
| T5-01 | Pydantic models para eventos WebSocket | S | P0 |
| T5-02 | Endpoint WebSocket `/v1/realtime` | M | P0 |
| T5-03 | Protocol handler (dispatch) | M | P0 |
| T5-04 | Heartbeat e timeout | S | P1 |
| T5-05 | Energy pre-filter | S | P0 |
| T5-06 | Silero VAD integration | M | P0 |
| T5-07 | VAD coordinator | S | P0 |

**Total**: ~17 story points (2x S + 3x M + 2x S = ~14-22 dias de trabalho)

**Checkpoint Sprint 1**:
- WebSocket aceita conexao e retorna `session.created`
- Audio frames recebidos e processados pelo protocol handler
- VAD detecta speech_start e speech_end corretamente
- Heartbeat funciona (ping/pong)
- Todos os testes unitarios de Sprint 1 passam

### Sprint 2 (Semanas 3-4): Streaming Pipeline

**Objetivo**: gRPC bidirecional funcional, transcricao real via streaming.

**Demo Goal**: Conectar via WebSocket, enviar audio, receber `transcript.partial` e `transcript.final` com texto transcrito.

**Tasks**:

| ID | Task | Estimativa | Prioridade |
|----|------|-----------|------------|
| T5-08 | Streaming preprocessor | S | P0 |
| T5-09 | gRPC bidirecional -- runtime side | L | P0 |
| T5-10 | gRPC bidirecional -- worker side | M | P0 |
| T5-11 | FasterWhisperBackend.transcribe_stream() | M | P0 |

**Total**: ~13 story points (1x S + 1x L + 2x M = ~12-17 dias de trabalho)

**Checkpoint Sprint 2**:
- Audio enviado pelo WebSocket chega ao worker via gRPC stream
- Worker transcreve audio e retorna TranscriptEvents
- transcript.partial e transcript.final emitidos ao cliente WebSocket
- Stream break detectado quando worker e terminado

### Sprint 3 (Semanas 5-6): Session + Integracao

**Objetivo**: Fluxo completo end-to-end com backpressure, metricas e testes de estabilidade.

**Demo Goal**: Demo completo conforme criterio de sucesso de M5 (secao 1).

**Tasks**:

| ID | Task | Estimativa | Prioridade |
|----|------|-----------|------------|
| T5-12 | StreamingSession | L | P0 |
| T5-13 | Backpressure | S | P1 |
| T5-14 | Manual commit | S | P1 |
| T5-15 | Metricas de streaming | S | P1 |
| T5-16 | Testes unitarios (review/complemento) | M | P0 |
| T5-17 | Testes de integracao WebSocket | M | P0 |
| T5-18 | Teste de estabilidade e documentacao | S | P1 |

**Total**: ~17 story points (4x S + 2x M + 1x L = ~15-23 dias de trabalho)

**Checkpoint Sprint 3**:
- Fluxo completo: WS -> audio -> vad events -> transcript events
- Backpressure funciona (rate_limit e frames_dropped)
- Manual commit funciona
- Metricas expostas no /metrics
- Teste de estabilidade de 5 minutos passa
- CHANGELOG e ROADMAP atualizados

---

## 7. Grafo de Dependencias

```
T5-01 (Event Models)
  |
  +---> T5-02 (WS Endpoint)
  |       |
  |       +---> T5-03 (Protocol Handler) ---+
  |       |                                  |
  |       +---> T5-04 (Heartbeat)           |
  |                                          |
  +-----------------------------------------+---> T5-12 (StreamingSession) ---+
                                             |                                |
T5-05 (Energy Pre-filter) ---+               |                                |
                              |              |                                |
T5-06 (Silero VAD) ----------+--> T5-07 (VAD Coordinator) --+                |
                                                             |                |
T5-08 (Streaming Preprocessor) -----------------------------+                |
                                                             |                |
T5-11 (FW Backend Stream) ---> T5-10 (Worker Servicer) --+  |                |
                                                          |  |                |
                                T5-09 (gRPC Client) -----+--+                |
                                                                              |
                                                                              |
T5-12 (StreamingSession) <----------------------------------------------------+
  |
  +---> T5-13 (Backpressure)
  +---> T5-14 (Manual Commit)
  +---> T5-15 (Metricas)
  |
  +---> T5-16 (Testes Unitarios Review)
  +---> T5-17 (Testes Integracao WS)
          |
          +---> T5-18 (Estabilidade + Docs)
```

**Caminho critico**: T5-01 -> T5-02 -> T5-03 -> T5-12 -> T5-17 -> T5-18

**Paralelismo maximo**: Sprint 1 permite trabalho paralelo em E1 (WebSocket) e E2 (VAD). Sprint 2 permite paralelismo entre T5-09 (runtime) e T5-10/T5-11 (worker). Sprint 3 permite paralelismo entre T5-13/T5-14/T5-15 apos T5-12.

---

## 8. Estrutura de Arquivos (M5)

```
src/theo/
  vad/                              # NOVO -- pacote completo
    __init__.py                     # Exports publicos
    energy.py                       # EnergyPreFilter                [T5-05]
    silero.py                       # SileroVADClassifier            [T5-06]
    detector.py                     # VADDetector, VADEvent          [T5-07]

  session/                          # EXISTIA (vazio) -- populado
    __init__.py
    streaming.py                    # StreamingSession               [T5-12]
    backpressure.py                 # BackpressureController         [T5-13]
    metrics.py                      # Definicoes de metricas         [T5-15]

  preprocessing/
    streaming.py                    # StreamingPreprocessor          [T5-08]  NOVO

  scheduler/
    streaming.py                    # StreamingGRPCClient            [T5-09]  NOVO

  server/
    models/
      events.py                     # Pydantic event models          [T5-01]  NOVO
    routes/
      realtime.py                   # WS /v1/realtime endpoint       [T5-02]  NOVO
    ws_protocol.py                  # Protocol dispatch (se extraido) [T5-03] NOVO
    app.py                          # Include realtime router        [T5-02]  ALTERADO

  workers/stt/
    servicer.py                     # TranscribeStream implementado  [T5-10]  ALTERADO
    converters.py                   # Converters de streaming        [T5-10]  ALTERADO
    faster_whisper.py               # transcribe_stream()            [T5-11]  ALTERADO

tests/
  unit/
    test_ws_event_models.py         # [T5-01]  NOVO
    test_realtime_endpoint.py       # [T5-02]  NOVO
    test_ws_protocol.py             # [T5-03]  NOVO
    test_ws_heartbeat.py            # [T5-04]  NOVO
    test_vad_energy.py              # [T5-05]  NOVO
    test_vad_silero.py              # [T5-06]  NOVO
    test_vad_detector.py            # [T5-07]  NOVO
    test_streaming_preprocessor.py  # [T5-08]  NOVO
    test_streaming_grpc_client.py   # [T5-09]  NOVO
    test_streaming_servicer.py      # [T5-10]  NOVO
    test_faster_whisper_streaming.py # [T5-11] NOVO
    test_streaming_session.py       # [T5-12]  NOVO
    test_backpressure.py            # [T5-13]  NOVO
    test_manual_commit.py           # [T5-14]  NOVO
    test_streaming_metrics.py       # [T5-15]  NOVO
  integration/
    test_ws_streaming.py            # [T5-17]  NOVO
    test_ws_stability.py            # [T5-18]  NOVO
```

---

## 9. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | gRPC bidirecional + WebSocket = duas camadas de streaming com semanticas diferentes. Bugs de backpressure e ordering podem se manifestar na integracao. | Alta | Alto | Abstrair comunicacao com worker atras de `StreamHandle`. WebSocket handler nao conhece gRPC. Testes de integracao com cenarios de carga desde Sprint 2. |
| R2 | Silero VAD depende de PyTorch (ou ONNX Runtime). Adiciona dependencia pesada ao runtime. | Media | Medio | Usar ONNX Runtime como alternativa mais leve. Declarar como optional dependency (`pip install theo[vad]`). Mock nos testes unitarios. Fallback sem VAD: todo audio enviado ao worker (ineficiente mas funcional). |
| R3 | Latencia de preprocessing + VAD + gRPC round-trip excede target de 300ms TTFB. | Media | Alto | Medir TTFB desde Sprint 2. Se >300ms, identificar gargalo (metricas por componente). Preprocessing e <5ms. VAD e <2ms. gRPC overhead estimado <5ms. O gargalo provavel e a inference do Whisper (~100-200ms). |
| R4 | Memory leak em conexoes WebSocket de longa duracao (asyncio tasks, buffers, gRPC channels). | Media | Alto | Teste de estabilidade de 5 minutos desde Sprint 3. Monitorar RSS. Cleanup explicito de recursos no close da sessao. Usar `async with` para gRPC channels. |
| R5 | FasterWhisper `transcribe_stream()` com estrategia simples (acumular + inference) produz latencia alta para partials. | Alta | Medio | Esperado em M5. Partials serao basicos (resultado completo apos threshold). LocalAgreement em M6 resolve. Documentar como limitacao conhecida de M5. |
| R6 | Backpressure dificil de testar deterministicamente (depende de timing). | Media | Medio | Usar mocks de tempo (`asyncio.sleep` mockado). Enviar audio pre-gravado em velocidade controlada. Nao depender de timing real nos testes. |
| R7 | FastAPI WebSocket nao suporta nativamente ping/pong server-initiated de forma confiavel. | Media | Baixo | FastAPI/Starlette suporta `websocket.send_bytes()` para ping. Alternativa: usar framework-level keepalive via `websockets` library. Testar com client real (wscat). |
| R8 | Testes existentes de M1-M4 (400 testes) quebram com novas dependencias ou alteracoes no `create_app()`. | Baixa | Medio | Novas dependencias (StreamingSession, VAD) sao opcionais no `create_app()`. Router realtime e adicionado ao app, mas nao afeta rotas existentes. Rodar suite completa em cada PR. |

---

## 10. Out of Scope (explicitamente NAO esta em M5)

Estes itens sao mencionados no PRD/ARCHITECTURE.md mas pertencem a milestones futuros:

| Item | Milestone | Justificativa |
|------|-----------|---------------|
| Ring Buffer com read fence e force commit | M6 | Depende de Session Manager completo. M5 envia frames direto ao worker. |
| WAL in-memory para recovery sem duplicacao | M6 | Depende de Ring Buffer e checkpoint tracking. |
| Maquina de estados completa (INIT, ACTIVE, SILENCE, HOLD, CLOSING, CLOSED) | M6 | M5 usa ACTIVE/CLOSED simplificado para validar infraestrutura. |
| LocalAgreement para partial transcripts de encoder-decoder | M6 | Algoritmo complexo que depende de Ring Buffer e comparacao entre passes. |
| Cross-segment context (initial_prompt dos ultimos 224 tokens) | M6 | Depende de tracking de segmentos no Session Manager. |
| Hot Word Correction stage (Levenshtein post-processing) | M6 | Domain-specific, requer configuracao por sessao. |
| Entity Formatting stage (CPF, CNPJ, valores) | Futuro | Sem caso de uso concreto. |
| CLI `theo transcribe --stream` | M6 | Depende de Session Manager estavel para UX confiavel. |
| Denoise stage (RNNoise) | M8 | Necessario para telefonia, desativado por default. |
| Dynamic batching no worker | M9 | Otimizacao de throughput, nao necessaria para M5. |
| RTP Listener | M8 | Telefonia e Fase 3. |
| Segundo backend (WeNet) | M7 | Validacao model-agnostic apos Session Manager completo. |

---

## 11. Dependencias Externas (novas no pyproject.toml)

```toml
# Nova optional dependency group para VAD
[project.optional-dependencies]
vad = [
    "onnxruntime>=1.16,<2.0",     # Runtime para Silero VAD (mais leve que torch)
]

# Silero VAD model sera baixado/embutido como asset
# Alternativa: silero-vad package se estiver disponivel no PyPI
```

**Decisao sobre PyTorch vs ONNX Runtime**: Silero VAD funciona com ambos. ONNX Runtime e ~10x menor que PyTorch (~50MB vs ~500MB). Para M5, recomendar ONNX Runtime como default. Se o worker ja carrega PyTorch (para Faster-Whisper), o runtime NAO precisa de PyTorch porque o VAD roda no processo principal, nao no worker.

**Nota**: `torch` ja e dependencia transitiva do `faster-whisper`. Se o usuario instalar `theo[faster-whisper]`, PyTorch ja esta disponivel. Para instalacoes sem engine (runtime puro + worker remoto), `onnxruntime` e a escolha correta para o VAD.

---

## 12. Criterio de Sucesso do M5

### 12.1 Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | WebSocket connect retorna `session.created` | wscat test |
| 2 | Audio frames recebidos e preprocessados | Log de preprocessing |
| 3 | VAD detecta fala e emite `vad.speech_start` | Evento no WS |
| 4 | VAD detecta silencio e emite `vad.speech_end` | Evento no WS |
| 5 | Worker transcreve via gRPC streaming | TranscriptEvent recebido |
| 6 | `transcript.partial` emitido durante fala | Evento no WS |
| 7 | `transcript.final` emitido apos silencio/commit | Evento no WS |
| 8 | Post-processing (ITN) aplicado em finals | "dois mil" -> "2000" |
| 9 | Post-processing NAO aplicado em partials | Verificar texto cru |
| 10 | Backpressure funciona (rate_limit, frames_dropped) | Teste com audio rapido |
| 11 | Manual commit produz transcript.final | Comando JSON |
| 12 | Heartbeat detecta conexao morta | Timeout test |
| 13 | Timeout de inatividade fecha sessao | 60s sem audio |
| 14 | Worker crash detectado via stream break | Kill worker durante streaming |
| 15 | TTFB <=300ms (target) | Metrica prometheus |

### 12.2 Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `.venv/bin/python -m mypy src/` |
| 2 | ruff check sem warnings | `.venv/bin/python -m ruff check src/ tests/` |
| 3 | ruff format sem diffs | `.venv/bin/python -m ruff format --check src/ tests/` |
| 4 | Todos os testes passam (M1-M5) | `.venv/bin/python -m pytest tests/unit/ -v` |
| 5 | CI verde | GitHub Actions |

### 12.3 Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | Event models (serialization, validation) | 10-15 |
| Unit | WebSocket endpoint (handshake, errors) | 5-8 |
| Unit | Protocol handler (dispatch, errors) | 8-12 |
| Unit | Heartbeat (ping/pong, timeout) | 3-5 |
| Unit | Energy pre-filter (silence, speech, noise) | 5-8 |
| Unit | Silero VAD (speech probability, debounce) | 5-8 |
| Unit | VAD coordinator (pre-filter + silero, events) | 5-8 |
| Unit | Streaming preprocessor (resample, normalize) | 4-6 |
| Unit | gRPC client (open, send, receive, break) | 6-10 |
| Unit | Worker servicer streaming (receive, emit) | 5-8 |
| Unit | FW backend streaming (accumulate, transcribe) | 4-6 |
| Unit | StreamingSession (flow, lifecycle, cleanup) | 8-12 |
| Unit | Backpressure (rate limit, drop) | 4-6 |
| Unit | Manual commit (during speech, during silence) | 3-5 |
| Unit | Metrics (ttfb, final_delay, active_sessions) | 4-6 |
| Integration | WS fluxo completo (7+ cenarios) | 7-10 |
| Integration | Estabilidade (5 min) | 1 |
| **Total** | | **>=80 novos testes** |

### 12.4 Demo Goal

```bash
# 1. Iniciar servidor
theo serve &

# 2. Conectar via WebSocket
wscat -c ws://localhost:8000/v1/realtime?model=faster-whisper-tiny

# 3. Receber session.created
# <- {"type": "session.created", "session_id": "sess_abc123", ...}

# 4. Enviar audio binario com fala (arquivo pre-gravado em PCM 16kHz)
# (wscat nao suporta binario facilmente -- usar script Python ou wscat com --binary)

# 5. Receber eventos
# <- {"type": "vad.speech_start", "timestamp_ms": 1500}
# <- {"type": "transcript.partial", "text": "ola como", ...}
# <- {"type": "transcript.final", "text": "ola como posso ajudar", ...}
# <- {"type": "vad.speech_end", "timestamp_ms": 4000}

# 6. Verificar metricas
curl http://localhost:8000/metrics | grep theo_stt_ttfb
# theo_stt_ttfb_seconds_bucket{le="0.3"} 1

# 7. Verificar que testes anteriores nao quebraram
.venv/bin/python -m pytest tests/unit/ -v
# -> 480+ testes passam (400 M1-M4 + 80+ M5)
```

---

## 13. Transicao M5 -> M6

Ao completar M5, o time deve ter:

1. **WebSocket funcional** -- Endpoint, protocolo, handshake, heartbeat, backpressure. Estabilizado. M6 nao muda o WebSocket handler.

2. **VAD completo** -- Energy pre-filter + Silero VAD com sensitivity levels. Componente reutilizado diretamente em M6 sem mudancas.

3. **gRPC bidirecional validado** -- Runtime e worker se comunicam via stream. Stream break detectado. Este canal e reutilizado por M6 com a mesma interface.

4. **StreamingSession substituivel** -- A `StreamingSession` simplificada de M5 sera substituida pelo `SessionManager` de M6. A interface com o WebSocket handler (callback `on_event`) e mantida. O que muda e interno: Ring Buffer, WAL, maquina de estados, LocalAgreement.

5. **Pontos de extensao para M6**:
   - **Ring Buffer**: Inserido entre preprocessing e gRPC. Frames vao para o Ring Buffer em vez de direto ao gRPC. O Ring Buffer alimenta o worker.
   - **Maquina de estados completa**: Substitui ACTIVE/CLOSED por INIT->ACTIVE->SILENCE->HOLD->CLOSING->CLOSED. Timeouts por estado.
   - **LocalAgreement**: Opera sobre o Ring Buffer. Acumula windows, compara passes, confirma tokens. Emite partials inteligentes em vez dos partials basicos de M5.
   - **WAL**: Checkpoints periodicos do estado da sessao para recovery sem duplicacao.
   - **Cross-segment context**: Ultimos 224 tokens do final anterior como initial_prompt.
   - **Hot Word Correction**: Stage de post-processing com Levenshtein distance.

**O primeiro commit de M6 sera**: extrair `StreamingSession` para `SessionManager` com maquina de estados completa (6 estados com timeouts configuraveis).

---

*Documento gerado por Sofia Castellani (Principal Solution Architect, ARCH). Sera atualizado conforme a implementacao do M5 avanca.*
