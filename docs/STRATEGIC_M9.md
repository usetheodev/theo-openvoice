# M9 -- Full-Duplex (STT + TTS) -- Strategic Roadmap

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Planejado
**Data**: 2026-02-09

**Autores**:
- Sofia Castellani (Principal Solution Architect)
- Viktor Sorokin (Senior Real-Time Engineer)
- Andre Oliveira (Senior Platform Engineer)

**Dependencias**: M8 (Scheduler Avancado com PriorityQueue, cancelamento, dynamic batching)

---

## 1. Objetivo Estrategico

M9 habilita o cenario de **agente de voz full-duplex**: STT e TTS operando simultaneamente na mesma sessao, coordenados pelo runtime. O usuario fala, o STT transcreve, um LLM externo gera resposta, e o TTS sintetiza audio -- tudo na mesma conexao WebSocket.

### O que M9 resolve

O Theo ate M8 e **half-duplex**: o runtime transcreve audio (STT) mas nao sintetiza audio (TTS). Para agentes de voz, o cliente precisa integrar STT e TTS separadamente, lidando com a coordenacao manualmente (quando silenciar o microfone, quando o bot esta falando, como evitar transcrever o audio do proprio bot).

Problemas sem M9:

- **Sem TTS no runtime**: nenhum endpoint TTS, nenhum worker TTS, nenhum proto TTS. O PRD define `POST /v1/audio/speech` e workers TTS (Kokoro, Piper), mas nada esta implementado.
- **Sem coordenacao STT/TTS**: se o cliente usa STT do Theo e TTS externo, nao ha mute-on-speak. O STT transcreve o audio do TTS, gerando loops de feedback.
- **Sem eventos de fala do bot**: o cliente nao sabe quando o TTS comecou/terminou de falar, impossibilitando sincronizacao de UI e logica de turno.

### O que M9 habilita

- **Agente de voz end-to-end**: STT + TTS no mesmo runtime, coordenados pelo Scheduler.
- **Mute-on-speak**: STT pausa automaticamente enquanto TTS esta ativo na mesma sessao. Previne transcricao do audio do bot.
- **Eventos de TTS**: `tts.speaking_start` / `tts.speaking_end` permitem coordenacao de UI e logica de turno.
- **V2V latency tracking**: latencia end-to-end (VAD -> ASR -> [LLM] -> TTS) rastreavel por metricas.

### O que M9 NAO faz

- **LLM**: o Theo nao integra LLM. A cadeia STT -> LLM -> TTS e orquestrada pelo cliente. O Theo fornece STT e TTS; o LLM fica no middleware do cliente.
- **Barge-in completo**: mute-on-speak desabilita STT enquanto TTS fala. Para barge-in real (usuario interrompe o bot), e necessario AEC (Acoustic Echo Cancellation) externo. M9 documenta isso como limitacao.
- **AEC (Acoustic Echo Cancellation)**: remocao de eco do TTS no microfone e responsabilidade do cliente ou de hardware externo.

### Criterio de sucesso (Demo Goal)

```
# Cenario: sessao full-duplex via WebSocket
# 1. Cliente conecta WS /v1/realtime?model=faster-whisper-tiny
# 2. Usuario fala -> STT transcreve (transcript.partial + transcript.final)
# 3. Cliente envia texto para LLM externo (fora do Theo)
# 4. Cliente envia resposta do LLM para TTS via comando tts.speak
# 5. TTS sintetiza -> audio enviado ao cliente via WebSocket binario
# 6. Evento tts.speaking_start emitido -> STT muted automaticamente
# 7. Evento tts.speaking_end emitido -> STT unmuted automaticamente
# 8. Usuario fala novamente -> STT transcreve normalmente
# Resultado: zero transcricao do audio do bot (mute-on-speak efetivo)

# Cenario: TTS batch (REST)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-v1", "input": "Ola, como posso ajudar?", "voice": "default"}'
# -> Audio WAV/PCM retornado no body
```

### Conexao com o Roadmap

```
PRD Fase 3 (Escala + Full-Duplex)
  M8: Scheduler Avancado   (completo)
  M9: Full-Duplex           [<-- ESTE MILESTONE]

Caminho critico: M8 (completo) -> M9
```

M9 e o ultimo milestone do Roadmap atual. Completa a Fase 3 do PRD e entrega o runtime unificado STT + TTS conforme a visao original do projeto.

---

## 2. Pre-Requisitos (de M1-M8)

### O que ja existe e sera REUTILIZADO sem mudancas

| Componente | Pacote | Uso em M9 |
|------------|--------|-----------|
| `Scheduler` (M8: PriorityQueue, CancellationManager, BatchAccumulator, LatencyTracker) | `theo.scheduler.scheduler` | Base para co-scheduling. Sera estendido para TTS. |
| `SchedulerQueue` + `RequestPriority` | `theo.scheduler.queue` | Fila com prioridade. Reutilizada para requests TTS batch. |
| `CancellationManager` | `theo.scheduler.cancel` | Cancelamento de requests TTS batch. Reutilizado. |
| `StreamingSession` | `theo.session.streaming` | Orquestrador STT. Sera estendido com mute-on-speak flag. |
| `SessionStateMachine` + `SessionTimeouts` | `theo.session.state_machine` | Maquina de estados STT. Nao muda. |
| `RingBuffer` + `SessionWAL` | `theo.session.ring_buffer`, `theo.session.wal` | Buffer e WAL do STT. Nao mudam. |
| `StreamingGRPCClient` + `StreamHandle` | `theo.scheduler.streaming` | gRPC streaming STT. Nao muda. |
| `WorkerManager` | `theo.workers.manager` | Gerencia workers STT. Sera reutilizado para workers TTS. |
| `STTBackend` ABC | `theo.workers.stt.interface` | Interface STT. Nao muda. Inspira `TTSBackend`. |
| `ModelRegistry` + `ModelManifest` | `theo.registry.registry`, `theo.config.manifest` | Registry compartilhado STT+TTS. `ModelType.TTS` ja existe no enum. |
| REST endpoints STT | `theo.server.routes.transcriptions`, `theo.server.routes.translations` | Nao mudam. |
| WebSocket endpoint STT | `theo.server.routes.realtime` | Sera estendido com comandos e eventos TTS. |
| Audio Preprocessing Pipeline | `theo.preprocessing` | Nao muda. TTS nao precisa de preprocessing. |
| Post-Processing Pipeline | `theo.postprocessing` | Nao muda. |
| Metricas Prometheus existentes | `theo.session.metrics`, `theo.scheduler.metrics` | Reutilizadas. M9 adiciona metricas TTS. |
| Exceptions tipadas | `theo.exceptions` | Reutilizadas. M9 adiciona `TTSError` e derivadas. |
| `ModelType.TTS` enum | `theo._types` | Ja existe. Registry ja diferencia STT vs TTS. |
| `create_app()` factory | `theo.server.app` | Sera estendido para injetar TTS worker e novo endpoint. |
| CLI (`theo serve`, `theo list`, `theo inspect`) | `theo.cli` | Ja servem modelos STT. Servirao TTS sem mudanca. |

### O que sera MODIFICADO

| Componente | Pacote | O que muda |
|------------|--------|------------|
| `StreamingSession` | `theo.session.streaming` | Adicionar `mute()` / `unmute()` para mute-on-speak. Frames descartados quando muted. |
| WebSocket protocol events | `theo.server.models.events` | Adicionar eventos TTS (`tts.speaking_start`, `tts.speaking_end`) e comando `tts.speak`. |
| WebSocket endpoint | `theo.server.routes.realtime` | Processar comando `tts.speak`; enviar audio TTS e eventos de fala. |
| `ws_protocol.py` | `theo.server.ws_protocol` | Dispatch do novo comando `tts.speak`. |
| `create_app()` | `theo.server.app` | Injetar TTS worker manager e TTS endpoint REST. |
| `Scheduler` | `theo.scheduler.scheduler` | Estender para aceitar requests TTS (via `RequestType` ou reuso de `submit()`). |
| `WorkerManager` | `theo.workers.manager` | Suportar spawn de workers TTS (tipo diferente de worker, mesmo lifecycle). |
| gRPC proto | `theo.proto` | Novo `tts_worker.proto` com service TTSWorker. |
| `CHANGELOG.md` | raiz | Entradas M9. |
| `ROADMAP.md` | docs | Resultado M9, checkpoint Fase 3. |
| `CLAUDE.md` | raiz | Novos componentes e padroes M9. |
| `ARCHITECTURE.md` | docs | Secao TTS workers, full-duplex flow. |

### O que sera CRIADO

| Componente | Pacote | Descricao |
|------------|--------|-----------|
| `TTSBackend` ABC | `theo.workers.tts.interface` | Interface abstrata para engines TTS (espelho de `STTBackend`). |
| `KokoroBackend` (stub) | `theo.workers.tts.kokoro` | Implementacao placeholder para Kokoro (TTS engine). |
| `TTSWorkerServicer` | `theo.workers.tts.servicer` | Servicer gRPC para worker TTS. |
| TTS worker `main.py` | `theo.workers.tts.main` | Entry point do subprocess worker TTS. |
| `tts_worker.proto` | `theo.proto` | Definicao do servico gRPC TTSWorker. |
| `TTSSpeechResult` | `theo._types` | Resultado de sintese TTS (audio bytes + metadata). |
| TTS exceptions | `theo.exceptions` | `TTSError`, `TTSSynthesisError`. |
| REST `POST /v1/audio/speech` | `theo.server.routes.speech` | Endpoint OpenAI-compatible para TTS batch. |
| TTS request/response models | `theo.server.models.speech` | Pydantic models para request/response TTS. |
| `MuteController` | `theo.session.mute` | Coordena mute-on-speak entre TTS e STT na mesma sessao. |
| TTS events/commands | `theo.server.models.events` | `TTSSpeakCommand`, `TTSSpeakingStartEvent`, `TTSSpeakingEndEvent`, `TTSAudioChunkEvent`. |
| TTS metricas | `theo.scheduler.tts_metrics` | `tts_ttfb_seconds`, `tts_synthesis_duration_seconds`, `tts_requests_total`. |
| TTS converters | `theo.scheduler.tts_converters` | Proto <-> dominio para TTS. |

---

## 3. Visao Geral da Arquitetura M9

### 3.1 Full-Duplex Data Flow

```
                    Cliente (WebSocket)
                    |             ^
         audio frames (bin)       | audio TTS (bin) + JSON events
                    |             |
                    v             |
+---------------------------------------------------+
|            WS /v1/realtime (estendido)             |
|                                                     |
|  STT path:                TTS path:                 |
|  audio -> preprocessing   tts.speak cmd ->          |
|        -> VAD             Scheduler.submit_tts() -> |
|        -> StreamingSession  TTS worker gRPC ->      |
|        -> transcript.*      audio chunks ->         |
|                             tts.speaking_start ->   |
|                             MuteController.mute()   |
|                             audio bin frames ->     |
|                             tts.speaking_end ->     |
|                             MuteController.unmute() |
|                                                     |
|  MuteController:                                    |
|    mute() -> StreamingSession.mute()                |
|    unmute() -> StreamingSession.unmute()            |
+---------------------------------------------------+
           |                        |
           v                        v
  STT Worker (gRPC)         TTS Worker (gRPC)
  (Faster-Whisper/WeNet)    (Kokoro/Piper)
```

### 3.2 Mute-on-Speak Mecanismo

```
Estado normal (STT ativo):
  audio frames -> preprocessing -> VAD -> worker -> transcript.*

tts.speak recebido:
  1. Scheduler envia texto ao TTS worker
  2. TTS worker comeca a sintetizar
  3. Primeiro chunk de audio pronto:
     a. Emite tts.speaking_start ao cliente
     b. MuteController.mute() -> StreamingSession.mute()
  4. Audio chunks enviados ao cliente (binary WebSocket)
  5. Ultimo chunk enviado:
     a. Emite tts.speaking_end ao cliente
     b. MuteController.unmute() -> StreamingSession.unmute()

Estado muted (STT pausado):
  audio frames -> preprocessing -> DESCARTADOS (nao vao para VAD nem worker)
  (ring buffer nao recebe frames, state machine nao muda)
```

**Perspectiva Viktor**: O mute e no nivel do `StreamingSession.process_frame()`. Quando muted, frames sao descartados ANTES do ring buffer e VAD. Isso e mais eficiente do que deixar o VAD processar e ignorar o resultado. O mute nao muda o estado da SessionStateMachine -- a sessao continua ACTIVE ou SILENCE, apenas nao processa novos frames.

**Perspectiva Sofia**: O MuteController e o ponto de coordenacao entre o fluxo TTS e o fluxo STT. Ele vive no contexto da sessao WebSocket (nao no Scheduler). Quando o TTS comeca a falar, o handler WebSocket notifica o MuteController, que muta o STT. Quando termina, desmuta. A logica e simples: um flag booleano no StreamingSession. A complexidade esta em garantir que unmute acontece mesmo se o TTS falhar (via try/finally).

### 3.3 TTS Worker (Subprocess gRPC)

```
+---------------------------------------------------------------+
|                   TTS WORKER (subprocess)                      |
|                                                                |
|  +----------------------------------------------------------+ |
|  | TTSWorkerServicer (gRPC)                                 | |
|  |                                                          | |
|  |  Synthesize(SynthesizeRequest) -> stream SynthesizeChunk | |
|  |  Health(HealthRequest) -> HealthResponse                 | |
|  +----------------------------------------------------------+ |
|                         |                                      |
|                         v                                      |
|  +----------------------------------------------------------+ |
|  | TTSBackend (ABC)                                         | |
|  |                                                          | |
|  |  .load() / .unload()                                     | |
|  |  .synthesize(text, voice, ...) -> AsyncIterator[bytes]   | |
|  |  .voices() -> list[VoiceInfo]                            | |
|  |  .health() -> dict                                       | |
|  +----------------------------------------------------------+ |
|                         |                                      |
|              +----------+----------+                           |
|              |                     |                           |
|              v                     v                           |
|      KokoroBackend          [PiperBackend]                     |
|      (MVP: stub)            (futuro)                           |
+---------------------------------------------------------------+
```

### 3.4 TTS Batch Flow (REST)

```
Cliente
  | POST /v1/audio/speech
  | {"model": "kokoro-v1", "input": "Ola", "voice": "default"}
  v
API Server (FastAPI)
  | Valida request
  | Resolve modelo no Registry (type=tts)
  v
Scheduler.submit_tts(request)
  | Enfileira com prioridade BATCH
  v
Worker TTS (gRPC)
  | Synthesize(text, voice, ...)
  | Retorna audio chunks
  v
Concatena audio
  | Content-Type: audio/wav (ou audio/pcm)
  v
Response -> Cliente
```

### 3.5 TTS Streaming Flow (WebSocket — Full-Duplex)

```
Cliente (WS)
  | {"type": "tts.speak", "text": "Ola, como posso ajudar?", "voice": "default"}
  v
WS Handler
  | 1. Valida modelo TTS (session config ou default)
  | 2. Inicia TTS via gRPC streaming ao worker
  | 3. Primeiro chunk pronto:
  |      -> {"type": "tts.speaking_start", "request_id": "..."}
  |      -> MuteController.mute()
  | 4. Audio chunks enviados como binary frames
  | 5. Ultimo chunk:
  |      -> {"type": "tts.speaking_end", "request_id": "..."}
  |      -> MuteController.unmute()
  v
Cliente recebe audio + eventos de coordenacao
```

---

## 4. Epics

### Epic 1: TTS MVP Infrastructure

Criar a infraestrutura minima de TTS: interface `TTSBackend`, proto gRPC, worker subprocess, backend stub (Kokoro). Sem esta base, nenhum outro epic de M9 e possivel.

**Racional**: O PRD define TTS como parte do runtime unificado, mas nenhum componente TTS existe no codebase. M9 precisa de um TTS funcional -- mesmo que MVP -- para habilitar full-duplex. O foco e na infraestrutura (proto, worker, interface), nao na qualidade da sintese.

**Responsavel principal**: Sofia (interfaces, contratos), Viktor (gRPC, worker subprocess)

**Tasks**: M9-01, M9-02, M9-03

### Epic 2: TTS REST Endpoint

Expor `POST /v1/audio/speech` compativel com o contrato OpenAI. Primeiro entregavel usavel de TTS -- um usuario pode sintetizar audio via REST.

**Racional**: O endpoint REST e o caminho mais simples para validar que o TTS worker funciona end-to-end. Tambem serve como entregavel autonomo para usuarios que precisam apenas de TTS batch (sem full-duplex).

**Responsavel principal**: Sofia (endpoint, Pydantic models), Andre (metricas)

**Tasks**: M9-04

### Epic 3: WebSocket Full-Duplex (STT + TTS)

Estender o protocolo WebSocket para suportar comandos TTS (`tts.speak`) e eventos de fala do bot (`tts.speaking_start`, `tts.speaking_end`). Implementar mute-on-speak.

**Racional**: E o core do full-duplex. O cliente pode falar (STT) e ouvir o bot (TTS) na mesma conexao WebSocket. Mute-on-speak previne feedback loops.

**Responsavel principal**: Viktor (streaming, mute-on-speak), Sofia (protocolo, eventos)

**Tasks**: M9-05, M9-06, M9-07

### Epic 4: Metricas, Testes e Finalizacao

Metricas TTS, testes de integracao full-duplex, documentacao e finalizacao do milestone.

**Racional**: Observabilidade TTS e essencial desde o dia 1. Testes de integracao validam o cenario full-duplex completo. Documentacao atualiza o projeto para refletir o runtime unificado STT+TTS.

**Responsavel principal**: Andre (metricas, observabilidade), Todos (testes)

**Tasks**: M9-08, M9-09, M9-10

---

## 5. Tasks (Detalhadas)

### M9-01: TTSBackend Interface + Tipos TTS

**Epic**: E1 -- TTS MVP Infrastructure
**Estimativa**: S (2-3 dias)
**Dependencias**: Nenhuma (componente isolado)
**Desbloqueia**: M9-02, M9-03, M9-04

**Contexto/Motivacao**: A interface `STTBackend` e o contrato que permite engines STT plugaveis. TTS precisa do equivalente: `TTSBackend`. A interface define: `load`, `synthesize`, `voices`, `unload`, `health`. O tipo `TTSSpeechResult` encapsula o resultado da sintese. Exceptions TTS-especificas complementam a hierarquia existente.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `TTSBackend` ABC com metodos: `load`, `synthesize`, `voices`, `unload`, `health` | Implementacao real de engine TTS (M9-02) |
| `TTSSpeechResult` dataclass: audio bytes + sample_rate + duration + voice | SSML parsing (futuro) |
| `VoiceInfo` dataclass: voice_id, name, language, gender | Streaming TTS com word-level timestamps |
| `TTSError`, `TTSSynthesisError` exceptions | Multi-speaker (futuro) |
| `TTSBackend.synthesize()` retorna `AsyncIterator[bytes]` (streaming de audio) | |
| Testes unitarios para tipos e interface (instantiation check) | |

**Entregaveis**:
- `src/theo/workers/tts/__init__.py`
- `src/theo/workers/tts/interface.py` -- `TTSBackend` ABC
- Alteracao em `src/theo/_types.py` -- `TTSSpeechResult`, `VoiceInfo`
- Alteracao em `src/theo/exceptions.py` -- `TTSError`, `TTSSynthesisError`
- `tests/unit/test_tts_types.py`

**DoD**:
- [ ] `TTSBackend` ABC com 5 metodos abstratos: `load`, `synthesize`, `voices`, `unload`, `health`
- [ ] `TTSBackend.synthesize(text, voice, ...)` retorna `AsyncIterator[bytes]` (chunks PCM 16-bit)
- [ ] `TTSBackend.voices()` retorna `list[VoiceInfo]`
- [ ] `TTSSpeechResult` frozen dataclass: `audio_data: bytes`, `sample_rate: int`, `duration: float`, `voice: str`
- [ ] `VoiceInfo` frozen dataclass: `voice_id: str`, `name: str`, `language: str`, `gender: str | None`
- [ ] `TTSError(TheoError)` base para erros TTS
- [ ] `TTSSynthesisError(TTSError)` com `model_name` e `reason`
- [ ] Testes: >=8 testes (tipos frozen, interface nao instanciavel, exception hierarchy)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: A interface `TTSBackend` segue o mesmo padrao de `STTBackend`: ABC com metodos async, tipos de dominio, engine como dependencia opcional. A diferenca fundamental e que `synthesize()` e um gerador (streaming de audio chunks), nao retorna tudo de uma vez. Isso permite TTS streaming com baixo TTFB -- o primeiro chunk de audio pode ser enviado ao cliente antes de toda a sintese estar pronta.

**Perspectiva Viktor**: O `synthesize()` retornando `AsyncIterator[bytes]` e crucial para V2V latency. Se esperassemos a sintese completa, o TTFB de TTS seria proporcional ao tamanho do texto. Com streaming, o primeiro chunk (50-100ms de audio) pode estar pronto em <50ms para engines rapidas como Kokoro.

---

### M9-02: TTS gRPC Proto + Worker Subprocess

**Epic**: E1 -- TTS MVP Infrastructure
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-01
**Desbloqueia**: M9-03, M9-04, M9-05

**Contexto/Motivacao**: Assim como STT usa `stt_worker.proto` para comunicacao runtime-worker, TTS precisa do seu proto: `tts_worker.proto`. O worker TTS e um subprocess gRPC identico em lifecycle ao STT (spawn, health probe, crash detection, auto-restart). O `TTSWorkerServicer` implementa o servico gRPC. O entry point `main.py` usa a mesma factory pattern (`_create_backend`).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `tts_worker.proto` com service `TTSWorker`: `Synthesize`, `Health` | `Cancel` RPC para TTS (TTS e rapido, cancel nao e prioritario) |
| `TTSWorkerServicer` implementando `Synthesize` (server-streaming) e `Health` (unario) | Streaming bidirecional (TTS nao precisa -- e unidirecional: texto in, audio out) |
| TTS worker `main.py` com `_create_backend()` factory | Deploy separado de worker TTS (mesmo binario) |
| Proto compilado com `make proto` (atualizar script de geracao) | |
| Reutilizar `WorkerManager` para spawn de TTS workers | |
| Testes unitarios: servicer, proto roundtrip | |

**Entregaveis**:
- `src/theo/proto/tts_worker.proto` -- definicao gRPC TTSWorker
- `src/theo/proto/tts_worker_pb2.py` + `tts_worker_pb2_grpc.py` (gerados)
- `src/theo/workers/tts/servicer.py` -- `TTSWorkerServicer`
- `src/theo/workers/tts/main.py` -- entry point do worker TTS
- Alteracao em `scripts/generate_proto.sh` -- incluir `tts_worker.proto`
- Alteracao em `Makefile` -- `make proto` gera stubs TTS
- `tests/unit/test_tts_servicer.py`

**DoD**:
- [ ] `tts_worker.proto` com service `TTSWorker`: `Synthesize(SynthesizeRequest) returns (stream SynthesizeChunk)`, `Health(HealthRequest) returns (HealthResponse)`
- [ ] `SynthesizeRequest`: `request_id`, `text`, `voice`, `sample_rate`, `speed`
- [ ] `SynthesizeChunk`: `audio_data` (bytes PCM), `is_last` (bool), `duration` (float)
- [ ] `TTSWorkerServicer` implementa `Synthesize` delegando para `TTSBackend.synthesize()`
- [ ] `TTSWorkerServicer` implementa `Health` delegando para `TTSBackend.health()`
- [ ] Worker `main.py` com `_create_backend(engine)` factory, lazy imports
- [ ] `make proto` gera stubs TTS sem erros
- [ ] `WorkerManager` suporta spawn de worker TTS (via parametro `worker_type` ou deteccao por manifesto `type: tts`)
- [ ] Testes: >=12 testes (servicer Synthesize, Health, proto serialization/deserialization, worker startup, factory)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O `Synthesize` RPC e server-streaming (unidirecional): o cliente envia um request, o servidor retorna um stream de chunks de audio. Nao precisa ser bidirecional como STT (que envia audio continuamente). O servidor produz chunks a medida que a engine sintetiza, e o runtime os envia ao cliente WebSocket como binary frames.

**Perspectiva Andre**: O `WorkerManager` atual spawna workers STT. Para TTS, o lifecycle e identico (spawn subprocess, health probe, crash detection, restart). A forma mais simples e parametrizar o `WorkerManager` com o tipo de worker (STT vs TTS), usando o campo `type` do manifesto. O entry point do subprocess muda (`theo.workers.tts.main` vs `theo.workers.stt.main`), mas o gerenciamento de lifecycle e reutilizado.

---

### M9-03: KokoroBackend (TTS Engine MVP)

**Epic**: E1 -- TTS MVP Infrastructure
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-01
**Desbloqueia**: M9-04, M9-05

**Contexto/Motivacao**: M9 precisa de pelo menos uma engine TTS funcional para validar a infraestrutura. Kokoro e a engine escolhida pelo PRD (lightweight, Python-native, qualidade razoavel). O `KokoroBackend` implementa `TTSBackend` usando a biblioteca Kokoro como dependencia. Se Kokoro nao estiver disponivel/estavel, usar `PiperBackend` como alternativa (Piper e mais maduro mas requer binario externo).

A implementacao segue exatamente o padrao de `FasterWhisperBackend` e `WeNetBackend`: guard de import, run_in_executor, tipos de dominio do Theo, testes com mocks.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `KokoroBackend` implementando `TTSBackend` | Suporte a SSML |
| `synthesize()` retornando chunks PCM via `AsyncIterator[bytes]` | Multi-speaker com voice cloning |
| `voices()` retornando vozes disponiveis do modelo | Ajuste fino de prosody/speed |
| Manifesto `theo.yaml` para Kokoro com `type: tts` | |
| Guard `try/except ImportError` para Kokoro (dependencia opcional) | |
| `pyproject.toml`: extra `[kokoro]` com dependencia | |
| Testes unitarios com mocks (sem engine real) | |
| Fallback: se Kokoro indisponivel, `SimpleTTSBackend` com audio gerado (sine tone) para testes | |

**Entregaveis**:
- `src/theo/workers/tts/kokoro.py` -- `KokoroBackend`
- `models/kokoro-v1/theo.yaml` (ou `tests/fixtures/manifests/valid_tts_kokoro.yaml`)
- Alteracao em `pyproject.toml` -- extra `kokoro`
- `tests/unit/test_kokoro_backend.py`

**DoD**:
- [ ] `KokoroBackend` implementa todos os metodos de `TTSBackend`
- [ ] `KokoroBackend.synthesize(text, voice)` retorna `AsyncIterator[bytes]` (chunks PCM 16-bit)
- [ ] `KokoroBackend.voices()` retorna lista de `VoiceInfo` do modelo
- [ ] `load()` levanta `ModelLoadError` se Kokoro nao instalado
- [ ] Inference roda em `loop.run_in_executor()` (nao bloqueia event loop)
- [ ] Manifesto TTS: `type: tts`, `engine: kokoro`, capabilities minimas
- [ ] `pyproject.toml` extra `kokoro` com dependencia
- [ ] mypy override para modulo kokoro (`ignore_missing_imports`)
- [ ] Testes: >=10 testes (architecture, capabilities, load success/failure, synthesize, voices, health, unload)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O backend TTS segue o mesmo padrao do STT: guard de import, tipos do Theo, testes com mocks. A diferenca e que `synthesize()` e um gerador de audio, nao um consumidor. Kokoro e a escolha MVP por ser Python-native (sem binario externo). Se Kokoro nao estiver maduro o suficiente, Piper (via `piper-tts` Python wrapper) e a alternativa. A interface `TTSBackend` e a mesma -- trocar engine e trocar a implementacao.

---

### M9-04: REST Endpoint POST /v1/audio/speech

**Epic**: E2 -- TTS REST Endpoint
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-02, M9-03
**Desbloqueia**: M9-05, M9-08

**Contexto/Motivacao**: O PRD define `POST /v1/audio/speech` como endpoint TTS batch compativel com OpenAI. O contrato aceita texto e retorna audio sintetizado. Este e o primeiro entregavel usavel de TTS -- valida que worker, proto e backend funcionam end-to-end. O endpoint usa o Scheduler existente (M8) para enfileirar a request TTS.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `POST /v1/audio/speech` com contrato OpenAI (model, input, voice, response_format, speed) | Streaming response (chunked transfer) -- TTS retorna audio completo no body |
| Pydantic models para request/response TTS | SSML no campo `input` |
| Roteamento via `ModelRegistry` (resolve `type: tts`) | Multiplos formatos de audio (apenas WAV e PCM inicialmente) |
| Envio ao TTS worker via Scheduler | |
| Error handling: 400, 404, 503 com formato OpenAI-compatible | |
| Testes end-to-end com mock de TTS worker | |

**Entregaveis**:
- `src/theo/server/routes/speech.py` -- endpoint TTS
- `src/theo/server/models/speech.py` -- Pydantic request/response models
- `src/theo/scheduler/tts_converters.py` -- proto <-> dominio TTS
- Alteracao em `src/theo/server/app.py` -- registrar rota speech
- `tests/unit/test_speech_endpoint.py`

**DoD**:
- [ ] `POST /v1/audio/speech` aceita JSON body: `model`, `input` (texto), `voice`, `response_format` (wav/pcm), `speed`
- [ ] Response: audio binary com Content-Type `audio/wav` ou `audio/pcm`
- [ ] `model` resolve para TTS worker via `ModelRegistry.get_manifest()` com `type: tts`
- [ ] Se modelo nao encontrado: 404 com formato OpenAI error
- [ ] Se worker nao disponivel: 503 com `Retry-After`
- [ ] Se texto vazio: 400 com erro claro
- [ ] Converters TTS: `build_tts_proto_request()`, `tts_proto_chunks_to_result()`
- [ ] Testes: >=15 testes (happy path, erro 400/404/503, formatos, conversores, validacao de input)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O contrato OpenAI para TTS (`POST /v1/audio/speech`) e simples: texto entra, audio sai. Campos: `model` (obrigatorio), `input` (texto, obrigatorio), `voice` (default: "default"), `response_format` (wav, pcm, mp3 -- iniciar com wav/pcm), `speed` (0.25-4.0, default 1.0). A resposta e binary audio no body, nao JSON. Isso difere do contrato STT (que retorna JSON).

**Perspectiva Andre**: O endpoint TTS batch usa o Scheduler da mesma forma que batch STT. A request TTS e enfileirada na PriorityQueue com prioridade BATCH. O worker TTS e resolvido via `ModelRegistry` pelo campo `type: tts`. O Scheduler nao precisa de mudanca na interface -- `submit(request, priority)` e generico o suficiente. O unico ajuste e garantir que o TTS worker e resolvido pelo `WorkerManager` baseado no tipo do modelo.

---

### M9-05: Protocolo WebSocket Full-Duplex -- Comandos e Eventos TTS

**Epic**: E3 -- WebSocket Full-Duplex
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-02, M9-04
**Desbloqueia**: M9-06, M9-07

**Contexto/Motivacao**: O protocolo WebSocket atual (M5) suporta apenas STT. Para full-duplex, o cliente precisa enviar comandos TTS (`tts.speak`, `tts.cancel`) e receber eventos TTS (`tts.speaking_start`, `tts.speaking_end`). O audio sintetizado e enviado como frames binarios de output (server -> client), distinguidos dos frames de input (client -> server) pelo sentido da mensagem.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Comando `tts.speak`: texto + voice + model_tts (opcional) | SSML no campo text |
| Comando `tts.cancel`: interrompe sintese em andamento | TTS streaming bidirecional (texto incremental) |
| Evento `tts.speaking_start`: TTS comecou a produzir audio | Word-level timestamps durante sintese |
| Evento `tts.speaking_end`: TTS terminou de produzir audio | |
| Audio TTS enviado como binary frames server -> client | |
| `session.configure` estendido com `model_tts` (modelo TTS para sessao) | |
| Dispatch de novos comandos no `ws_protocol.py` | |

**Entregaveis**:
- Alteracao em `src/theo/server/models/events.py` -- novos modelos TTS
- Alteracao em `src/theo/server/ws_protocol.py` -- dispatch de `tts.speak` e `tts.cancel`
- Alteracao em `src/theo/server/routes/realtime.py` -- processar comandos TTS
- `tests/unit/test_ws_tts_protocol.py`

**DoD**:
- [ ] `TTSSpeakCommand` Pydantic model: `type="tts.speak"`, `text`, `voice`, `request_id` (auto-gerado se omitido)
- [ ] `TTSCancelCommand` Pydantic model: `type="tts.cancel"`, `request_id` (opcional, cancela sintese ativa)
- [ ] `TTSSpeakingStartEvent`: `type="tts.speaking_start"`, `request_id`, `timestamp_ms`
- [ ] `TTSSpeakingEndEvent`: `type="tts.speaking_end"`, `request_id`, `timestamp_ms`, `duration_ms`
- [ ] `dispatch_message()` reconhece e despacha `tts.speak` e `tts.cancel`
- [ ] `SessionConfigureCommand` aceita campo `model_tts` (string, opcional)
- [ ] Audio TTS enviado como binary frames via `websocket.send_bytes(chunk)`
- [ ] `session.created` inclui `model_tts` no config (se configurado)
- [ ] Atualizacao dos union types `ServerEvent` e `ClientCommand`
- [ ] Testes: >=15 testes (dispatch, serialization de eventos, roundtrip, configuracao, cancel)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O protocolo distingue audio de input (client -> server) de audio de output (server -> client) pelo sentido da mensagem WebSocket. Nao precisa de header adicional -- o cliente sabe que binary frames que ele RECEBE sao audio TTS. Se for necessario distinguir (ex: cenario futuro com multiplos streams de audio), podemos adicionar um header de 4 bytes no futuro. KISS por enquanto.

**Perspectiva Viktor**: O `tts.speak` e assincrono: o cliente envia o comando e recebe `tts.speaking_start` quando o primeiro chunk esta pronto. O fluxo TTS roda em uma task asyncio separada no handler WebSocket, permitindo que o cliente continue enviando audio STT (que sera descartado pelo mute). O `tts.cancel` cancela a task TTS e emite `tts.speaking_end` com flag `cancelled: true`.

---

### M9-06: MuteController + Mute-on-Speak

**Epic**: E3 -- WebSocket Full-Duplex
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-05
**Desbloqueia**: M9-07, M9-08

**Contexto/Motivacao**: Mute-on-speak e o mecanismo que previne o STT de transcrever o audio do proprio bot (TTS). Quando o TTS comeca a sintetizar audio, o STT e silenciado. Quando termina, o STT e retomado. O `MuteController` coordena essa interacao entre o fluxo TTS e a `StreamingSession` STT.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `MuteController` com `mute()` / `unmute()` / `is_muted` | AEC (Acoustic Echo Cancellation) |
| `StreamingSession.mute()` / `unmute()` que descarta frames quando muted | Barge-in (usuario interrompe bot) -- requer AEC externo |
| Integracao no handler WebSocket: mute ao comecar TTS, unmute ao terminar | Unmute parcial (reduzir sensibilidade em vez de mutar completamente) |
| Garantia de unmute mesmo em caso de erro TTS (try/finally) | |
| Metrica `theo_stt_muted_frames_total` (frames descartados por mute) | |
| Testes: mute/unmute, mute durante fala ativa, unmute apos erro | |

**Entregaveis**:
- `src/theo/session/mute.py` -- `MuteController`
- Alteracao em `src/theo/session/streaming.py` -- `mute()`, `unmute()`, check no `process_frame()`
- Alteracao em `src/theo/server/routes/realtime.py` -- integracao mute com fluxo TTS
- `tests/unit/test_mute_controller.py`
- `tests/unit/test_streaming_session_mute.py`

**DoD**:
- [ ] `MuteController` com `mute()`, `unmute()`, `is_muted` property
- [ ] `MuteController` e safe para chamadas idempotentes: mute quando ja muted = no-op
- [ ] `StreamingSession.mute()` seta flag `_muted = True`
- [ ] `StreamingSession.unmute()` seta flag `_muted = False`
- [ ] `StreamingSession.process_frame()`: se `_muted`, descarta frame SEM processar (nao vai para preprocessing, VAD, nem worker)
- [ ] Handler WebSocket: `try: mute() ... tts_stream ... finally: unmute()` garante unmute em erro
- [ ] Metrica `theo_stt_muted_frames_total` (Counter) incrementada por frame descartado
- [ ] Se TTS e cancelado via `tts.cancel`, unmute e imediato
- [ ] Se WebSocket desconecta durante TTS, unmute e cleanup acontecem
- [ ] Testes: >=15 testes (mute/unmute basic, frames descartados, unmute after error, unmute after cancel, idempotent mute, metrica, concurrent mute/unmute)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O ponto critico e o timing do mute. O mute deve acontecer ANTES do primeiro byte de audio TTS ser enviado ao cliente. Se o audio chega ao alto-falante e o STT ainda nao foi mutado, ha uma janela onde o microfone captura o TTS. A sequencia deve ser: (1) mute STT, (2) enviar `tts.speaking_start`, (3) enviar audio. A janela entre mute e audio no alto-falante e de 0ms (local) -- o mute e efetivo porque acontece no runtime, nao no dispositivo.

**Perspectiva Sofia**: O `MuteController` e uma abstracao simples (flag booleano + integracao). A complexidade esta nos edge cases: o que acontece se `tts.speak` e chamado durante uma fala ativa do usuario? O STT esta emitindo partials -- mutamos e descartamos o partial atual? Sim: mutamos imediatamente. O partial atual e descartado. Quando o TTS terminar e o STT desmutar, o VAD vai detectar a fala do usuario (se ele ainda estiver falando) como um novo segmento. Isso e aceitavel e documentado.

---

### M9-07: Integracao Full-Duplex End-to-End

**Epic**: E3 -- WebSocket Full-Duplex
**Estimativa**: L (5-7 dias)
**Dependencias**: M9-05, M9-06
**Desbloqueia**: M9-08, M9-09

**Contexto/Motivacao**: Integrar todos os componentes TTS no fluxo WebSocket: resolver modelo TTS, conectar ao TTS worker via gRPC streaming, enviar audio ao cliente, coordenar mute-on-speak, e lidar com erros. Esta task conecta tudo que foi criado nas tasks anteriores.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Handler completo de `tts.speak` no WebSocket: resolve modelo -> gRPC -> chunks -> client | Co-scheduling sofisticado (STT e TTS no mesmo GPU) |
| Handler de `tts.cancel`: interrompe gRPC stream TTS, unmute, emite `tts.speaking_end` | V2V latency optimization (e feito via metricas e tuning, nao nesta task) |
| Fluxo TTS roda como background task (nao bloqueia o main loop STT) | |
| Multiplos `tts.speak` em sequencia: segundo cancela primeiro (ou enfileira) | |
| Error handling: TTS worker crash -> emite error event, unmute | |
| Modelo TTS resolvido por `session.configure` `model_tts` OU parametro no `tts.speak` | |

**Entregaveis**:
- Alteracao em `src/theo/server/routes/realtime.py` -- handler TTS completo
- Testes de integracao: sessao WebSocket com STT + TTS simultaneos
- `tests/unit/test_full_duplex.py`

**DoD**:
- [ ] `tts.speak` resolve modelo TTS via `ModelRegistry` (type=tts)
- [ ] `tts.speak` abre gRPC stream `Synthesize` com TTS worker
- [ ] Primeiro chunk TTS pronto: emite `tts.speaking_start` + mute STT
- [ ] Chunks TTS enviados como binary frames ao cliente
- [ ] Ultimo chunk: emite `tts.speaking_end` + unmute STT
- [ ] `tts.cancel` cancela gRPC stream + emite `tts.speaking_end(cancelled=true)` + unmute
- [ ] Se TTS worker crashar: emite error event + unmute
- [ ] Se segundo `tts.speak` chega durante TTS ativo: cancela primeiro, inicia segundo
- [ ] Modelo TTS configurado via `session.configure.model_tts` ou parametro `model` no `tts.speak`
- [ ] TTS roda em background task: nao bloqueia processamento de audio STT
- [ ] Testes: >=20 testes (happy path, cancel, worker crash, sequential speaks, mute/unmute lifecycle, modelo nao encontrado, sessao fechada durante TTS)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O TTS roda em uma `asyncio.Task` separada dentro do handler WebSocket. O main loop continua recebendo frames de audio (que sao descartados pelo mute). Quando o TTS termina (normalmente ou por cancel), a task resolve e o main loop checa o resultado. A coordenacao e via `MuteController` (compartilhado entre main loop e TTS task). Nao ha race condition porque tudo roda no mesmo event loop.

**Perspectiva Andre**: O TTS worker e resolvido pelo `WorkerManager` assim como o STT. O `WorkerManager` mantem workers STT e TTS. A resolucao e por `model_name` via `ModelRegistry`, que ja diferencia por `type`. O canal gRPC para o TTS worker e criado sob demanda (como o pool de canais STT no Scheduler). Para M9, um canal por worker TTS e suficiente.

---

### M9-08: Metricas TTS + Latency Budget V2V

**Epic**: E4 -- Metricas, Testes e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-04, M9-06, M9-07
**Desbloqueia**: M9-09

**Contexto/Motivacao**: TTS precisa de metricas desde o dia 1, seguindo o mesmo padrao de STT (M5/M6) e Scheduler (M8). Alem das metricas TTS basicas, o V2V latency budget do PRD (300ms target) precisa ser rastreavel: VAD (50ms) + ASR (100ms) + [LLM externo] + TTS TTFB (50ms).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `theo_tts_ttfb_seconds` (Histogram): tempo ate primeiro chunk de audio | Dashboard Grafana |
| `theo_tts_synthesis_duration_seconds` (Histogram): duracao total da sintese | Alerting automatico |
| `theo_tts_requests_total` (Counter, labels: status) | Metricas de qualidade TTS (MOS score) |
| `theo_stt_muted_frames_total` (Counter): frames descartados por mute-on-speak | |
| `theo_tts_active_sessions` (Gauge): sessoes com TTS ativo | |
| V2V latency: metrica composta (STT final_delay + TTS TTFB) | |
| Lazy import `prometheus_client`, no-op sem prometheus | |

**Entregaveis**:
- `src/theo/scheduler/tts_metrics.py` -- metricas TTS
- Alteracao em `src/theo/session/metrics.py` -- metrica de mute
- Instrumentacao em routes/realtime.py e session/streaming.py
- `tests/unit/test_tts_metrics.py`

**DoD**:
- [ ] `theo_tts_ttfb_seconds` (Histogram) observado entre envio de `tts.speak` e primeiro chunk
- [ ] `theo_tts_synthesis_duration_seconds` (Histogram) observado entre primeiro e ultimo chunk
- [ ] `theo_tts_requests_total` (Counter, labels: `status` = ok/error/cancelled) incrementado em conclusao
- [ ] `theo_stt_muted_frames_total` (Counter) incrementado por frame descartado em mute
- [ ] `theo_tts_active_sessions` (Gauge) incrementado em `tts.speaking_start`, decrementado em `tts.speaking_end`
- [ ] Lazy import + `HAS_METRICS` flag (pattern de M5/M6)
- [ ] Testes: >=10 testes (cada metrica observa corretamente, no-op sem prometheus)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: O V2V latency e composto: `stt_final_delay + client_llm_time + tts_ttfb`. O Theo controla `stt_final_delay` e `tts_ttfb`. O `client_llm_time` e externo e nao pode ser medido pelo Theo. Podemos oferecer `theo_v2v_runtime_latency_seconds = stt_final_delay + tts_ttfb` como metrica composta que mostra a contribuicao do Theo no budget de 300ms. O cliente deve adicionar seu tempo de LLM para ter o total.

---

### M9-09: Testes de Integracao Full-Duplex

**Epic**: E4 -- Metricas, Testes e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-07, M9-08
**Desbloqueia**: M9-10

**Contexto/Motivacao**: Validacao end-to-end do cenario full-duplex: sessao WebSocket com STT e TTS simultaneos, mute-on-speak efetivo, cancelamento de TTS, erro de worker TTS. Estes testes usam mocks de worker (nao requerem engine real) mas exercitam o fluxo completo.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Teste: STT transcrevendo + TTS sintetizando na mesma sessao | Load test (centenas de sessoes full-duplex) |
| Teste: mute-on-speak efetivo (zero transcricoes durante TTS) | Teste com engine TTS real (requer GPU) |
| Teste: cancel de TTS unmuta STT | Teste V2V de 30 minutos |
| Teste: TTS worker crash -> error + unmute | |
| Teste: multiplos tts.speak em sequencia | |
| Teste: REST TTS endpoint end-to-end | |

**Entregaveis**:
- `tests/unit/test_full_duplex_integration.py`
- `tests/unit/test_speech_e2e.py`

**DoD**:
- [ ] Teste: sessao WS com audio STT -> `transcript.final` -> `tts.speak` -> audio TTS recebido
- [ ] Teste: durante TTS, frames STT sao descartados (zero `transcript.*` events emitidos)
- [ ] Teste: apos `tts.speaking_end`, STT retoma e produz `transcript.*` normalmente
- [ ] Teste: `tts.cancel` interrompe sintese, emite `tts.speaking_end`, STT desmutado
- [ ] Teste: TTS worker crash emite error event com `recoverable: true`, STT desmutado
- [ ] Teste: segundo `tts.speak` cancela primeiro automaticamente
- [ ] Teste: `POST /v1/audio/speech` retorna audio com Content-Type correto
- [ ] Teste: modelo TTS nao encontrado retorna 404
- [ ] Todos os testes M1-M8 continuam passando (regressao)
- [ ] Testes: >=20 testes novos
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: Os testes de integracao usam a mesma abordagem de M5/M6: `_Light*` mock classes (sem unittest.mock.Mock) para evitar inflacao de memoria. O mock de TTS worker retorna chunks de audio pre-gerados (sine tone PCM, como os fixtures de audio STT). O timing e controlado via `asyncio.Event` para garantir determinismo.

---

### M9-10: Finalizacao -- Documentacao, CHANGELOG, ROADMAP

**Epic**: E4 -- Metricas, Testes e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M9-09
**Desbloqueia**: Nenhuma (leaf task, ultima do milestone)

**Contexto/Motivacao**: Atualizacao de toda a documentacao do projeto para refletir o runtime unificado STT+TTS. Inclui CHANGELOG, ROADMAP, CLAUDE.md, ARCHITECTURE.md e guia de integracao full-duplex.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `CHANGELOG.md` com entradas M9 | Guia de deploy em producao (futuro) |
| `ROADMAP.md` com resultado M9, checkpoint Fase 3 completa | Tutorial de integracao com LLM especifico |
| `CLAUDE.md` atualizado com componentes M9 | |
| `ARCHITECTURE.md` com secao TTS workers e full-duplex flow | |
| Guia de integracao full-duplex para desenvolvedores | |
| `make ci` verde com todos os testes M1-M9 | |

**Entregaveis**:
- Alteracao em `CHANGELOG.md`
- Alteracao em `docs/ROADMAP.md`
- Alteracao em `CLAUDE.md`
- Alteracao em `docs/ARCHITECTURE.md`
- `docs/FULL_DUPLEX.md` -- guia de integracao full-duplex

**DoD**:
- [ ] `CHANGELOG.md` com entradas M9 na secao `[Unreleased]`
- [ ] `ROADMAP.md` com resultado M9 e checkpoint Fase 3 completa
- [ ] `CLAUDE.md` atualizado com componentes M9, padroes TTS, coisas que mordem
- [ ] `ARCHITECTURE.md` com diagrama TTS worker, full-duplex flow, protocolo estendido
- [ ] `docs/FULL_DUPLEX.md` com: visao geral, protocolo WebSocket completo (STT+TTS), exemplos de integracao, limitacoes (mute-on-speak vs barge-in), metricas V2V
- [ ] `make ci` verde
- [ ] Total de testes novos M9: >=130
- [ ] Total acumulado (M1-M9): >=1540
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

## 6. Grafo de Dependencias

```
M9-01 (TTSBackend + Tipos)
  |
  +---> M9-02 (TTS Proto + Worker) -----+
  |          |                           |
  +---> M9-03 (KokoroBackend) ----------+
               |                         |
               +---> M9-04 (REST /v1/audio/speech) ---+
               |                                       |
               +---> M9-05 (WS Protocol TTS) ---------+
                          |                            |
                          +---> M9-06 (MuteController) |
                          |          |                  |
                          +----------+                  |
                                     |                  |
                                     v                  |
                          M9-07 (Full-Duplex E2E) ------+
                                     |                  |
                                     v                  v
                          M9-08 (Metricas TTS) ---------+
                                     |
                                     v
                          M9-09 (Testes Integracao)
                                     |
                                     v
                          M9-10 (Finalizacao)
```

### Caminho critico

```
M9-01 -> M9-02 -> M9-05 -> M9-06 -> M9-07 -> M9-08 -> M9-09 -> M9-10
```

### Paralelismo maximo

- **Sprint 1**: M9-01 (TTSBackend + tipos) -- ponto de partida isolado.
- **Sprint 2**: M9-02 (Proto + worker) e M9-03 (KokoroBackend) em paralelo -- ambos dependem de M9-01.
- **Sprint 3**: M9-04 (REST endpoint) e M9-05 (WS protocol) em paralelo -- dependem de Sprint 2.
- **Sprint 4**: M9-06 (MuteController) -- depende de M9-05.
- **Sprint 5**: M9-07 (Full-Duplex E2E) -- depende de M9-05, M9-06.
- **Sprint 6**: M9-08 (Metricas), M9-09 (Testes integracao) -- sequenciais.
- **Sprint 7**: M9-10 (Finalizacao) -- ultima task.

---

## 7. Estrutura de Arquivos (M9)

```
src/theo/
  workers/
    tts/
      __init__.py                          (NOVO) [M9-01]
      interface.py                         (NOVO) [M9-01] -- TTSBackend ABC
      kokoro.py                            (NOVO) [M9-03] -- KokoroBackend
      servicer.py                          (NOVO) [M9-02] -- TTSWorkerServicer
      main.py                              (NOVO) [M9-02] -- entry point worker TTS
    stt/
      (todos inalterados)
    manager.py                             (ALTERADO) [M9-02] -- suporte a workers TTS

  proto/
    tts_worker.proto                       (NOVO) [M9-02]
    tts_worker_pb2.py                      (NOVO, gerado) [M9-02]
    tts_worker_pb2_grpc.py                 (NOVO, gerado) [M9-02]
    stt_worker.proto                       (SEM MUDANCA)
    stt_worker_pb2.py                      (SEM MUDANCA)

  server/
    routes/
      speech.py                            (NOVO) [M9-04] -- POST /v1/audio/speech
      realtime.py                          (ALTERADO) [M9-05, M9-07] -- comandos TTS
      transcriptions.py                    (SEM MUDANCA)
      translations.py                      (SEM MUDANCA)
      health.py                            (SEM MUDANCA)
    models/
      speech.py                            (NOVO) [M9-04] -- Pydantic TTS models
      events.py                            (ALTERADO) [M9-05] -- eventos TTS
    ws_protocol.py                         (ALTERADO) [M9-05] -- dispatch TTS commands
    app.py                                 (ALTERADO) [M9-04] -- registrar rota speech

  session/
    mute.py                                (NOVO) [M9-06] -- MuteController
    streaming.py                           (ALTERADO) [M9-06] -- mute/unmute
    state_machine.py                       (SEM MUDANCA)
    ring_buffer.py                         (SEM MUDANCA)
    wal.py                                 (SEM MUDANCA)
    local_agreement.py                     (SEM MUDANCA)
    cross_segment.py                       (SEM MUDANCA)
    backpressure.py                        (SEM MUDANCA)
    metrics.py                             (ALTERADO) [M9-08] -- metrica mute

  scheduler/
    scheduler.py                           (ALTERADO) [M9-04] -- submit TTS requests
    tts_converters.py                      (NOVO) [M9-04] -- proto <-> dominio TTS
    tts_metrics.py                         (NOVO) [M9-08] -- metricas TTS
    queue.py                               (SEM MUDANCA)
    cancel.py                              (SEM MUDANCA)
    batching.py                            (SEM MUDANCA)
    latency.py                             (SEM MUDANCA)
    metrics.py                             (SEM MUDANCA)
    streaming.py                           (SEM MUDANCA)
    converters.py                          (SEM MUDANCA)

  _types.py                                (ALTERADO) [M9-01] -- TTSSpeechResult, VoiceInfo
  exceptions.py                            (ALTERADO) [M9-01] -- TTSError, TTSSynthesisError

tests/
  unit/
    test_tts_types.py                      [M9-01]          NOVO
    test_tts_servicer.py                   [M9-02]          NOVO
    test_kokoro_backend.py                 [M9-03]          NOVO
    test_speech_endpoint.py                [M9-04]          NOVO
    test_ws_tts_protocol.py                [M9-05]          NOVO
    test_mute_controller.py                [M9-06]          NOVO
    test_streaming_session_mute.py         [M9-06]          NOVO
    test_full_duplex.py                    [M9-07]          NOVO
    test_tts_metrics.py                    [M9-08]          NOVO
    test_full_duplex_integration.py        [M9-09]          NOVO
    test_speech_e2e.py                     [M9-09]          NOVO

docs/
    FULL_DUPLEX.md                         [M9-10]          NOVO
    ARCHITECTURE.md                        [M9-10]          ALTERADO
    ROADMAP.md                             [M9-10]          ALTERADO
```

### Contagem de arquivos impactados

| Tipo | Novos | Alterados | Inalterados |
|------|-------|-----------|-------------|
| Source | 12 (`interface.py`, `kokoro.py`, `servicer.py`, `main.py`, `tts_worker.proto`, 2x pb2, `speech.py` route, `speech.py` models, `mute.py`, `tts_converters.py`, `tts_metrics.py`) | 9 (`_types.py`, `exceptions.py`, `events.py`, `ws_protocol.py`, `realtime.py`, `app.py`, `manager.py`, `scheduler.py`, `metrics.py`) | 25+ |
| Tests | 11 arquivos | 0 | -- |
| Docs | 1 (`FULL_DUPLEX.md`) | 3 (`CHANGELOG.md`, `ROADMAP.md`, `CLAUDE.md`, `ARCHITECTURE.md`) | -- |

---

## 8. Criterios de Saida do M9

### Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `TTSBackend` interface definida e implementada (KokoroBackend) | Testes unitarios |
| 2 | `tts_worker.proto` compilado, TTS worker funcional como subprocess | Testes unitarios |
| 3 | `POST /v1/audio/speech` retorna audio sintetizado | Testes E2E |
| 4 | WebSocket suporta `tts.speak` e retorna audio + eventos | Testes E2E |
| 5 | Mute-on-speak efetivo: zero transcricoes durante TTS | Testes integracao |
| 6 | `tts.cancel` interrompe sintese e desmuta STT | Testes integracao |
| 7 | Unmute garantido mesmo em caso de erro TTS | Testes integracao |
| 8 | STT funciona normalmente apos TTS terminar (unmute) | Testes integracao |
| 9 | Multiplos `tts.speak` sequenciais funcionam (cancel automatico do anterior) | Testes integracao |
| 10 | TTS worker crash: error event + unmute + sessao continua | Testes integracao |

### Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `make check` |
| 2 | ruff check sem warnings | `make check` |
| 3 | ruff format sem diffs | `make check` |
| 4 | Todos os testes passam (M1-M9) | `make test-unit` |
| 5 | CI verde | GitHub Actions |

### Metricas Prometheus (verificaveis)

| Metrica | Tipo | Labels |
|---------|------|--------|
| `theo_tts_ttfb_seconds` | Histogram | -- |
| `theo_tts_synthesis_duration_seconds` | Histogram | -- |
| `theo_tts_requests_total` | Counter | `status` |
| `theo_tts_active_sessions` | Gauge | -- |
| `theo_stt_muted_frames_total` | Counter | -- |

### Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | TTSBackend + tipos | 8 |
| Unit | TTS proto + servicer | 12 |
| Unit | KokoroBackend | 10 |
| Unit | REST /v1/audio/speech | 15 |
| Unit | WS protocol TTS | 15 |
| Unit | MuteController + session mute | 15 |
| Unit | Full-duplex E2E | 20 |
| Unit | Metricas TTS | 10 |
| Unit | Integracao full-duplex | 20 |
| Unit | Finalizacao (regression) | 5 |
| **Total novos M9** | | **>=130** |
| **Total acumulado (M1-M9)** | | **>=1540** |

---

## 9. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Kokoro (engine TTS) nao esta maduro o suficiente para producao | Media | Medio | Usar Kokoro como MVP (stub). Se instavel, substituir por Piper (mais maduro, binario externo). Interface `TTSBackend` garante swappability. |
| R2 | Mute-on-speak elimina barge-in (usuario nao pode interromper o bot) | Certa | Medio | Documentar como limitacao explicita. Barge-in requer AEC externo. Oferecer campo `barge_in_enabled: false` no `session.configure` para quando AEC estiver disponivel (nao implementar, apenas reservar). |
| R3 | Coordenacao STT/TTS adiciona latencia ao pipeline | Media | Medio | Mute e unmute sao flags booleanos (latencia ~0ms). O overhead real e o gRPC para TTS worker. Medir via `theo_tts_ttfb_seconds`. |
| R4 | Latencia V2V de 300ms e agressiva para ponta a ponta | Alta | Alto | 300ms e o target do PRD incluindo LLM (que esta fora do Theo). O budget do Theo (ASR final_delay + TTS TTFB) e ~150ms. Medir por componente e aceitar 500ms como target inicial realista. |
| R5 | TTS worker e STT worker competem por GPU | Media | Alto | Workers sao subprocessos separados com CUDA contexts isolados. Em GPU compartilhada, configurar 1 worker STT + 1 worker TTS. Para producao, workers STT e TTS em GPUs separadas (ou TTS em CPU, que e rapido o suficiente para Kokoro). |
| R6 | Binary frames de saida (TTS) e entrada (STT) confusos no protocolo WebSocket | Baixa | Medio | WebSocket tem direcao: client->server e server->client. Frames binarios server->client sao SEMPRE TTS. Client->server sao SEMPRE STT. Sem ambiguidade. Documentar no guia. |
| R7 | Escopo de M9 e maior que estimado (TTS MVP + full-duplex + mute-on-speak) | Alta | Alto | O TTS MVP (E1) e o maior risco. Se necessario, priorizar E1+E2 (TTS REST) como M9a e E3+E4 (full-duplex) como M9b. Cada epic e entregavel autonomo. |

---

## 10. Out of Scope (explicitamente NAO esta em M9)

| Item | Justificativa |
|------|---------------|
| Barge-in completo (usuario interrompe bot) | Requer AEC externo. Mute-on-speak e o fallback funcional. |
| AEC (Acoustic Echo Cancellation) | Responsabilidade do cliente ou hardware. Fora do runtime. |
| SSML parsing | TTS MVP aceita texto plano. SSML e evolucao futura. |
| Voice cloning / multi-speaker | Kokoro/Piper tem vozes pre-treinadas. Cloning e futuro. |
| Streaming TTS incremental (texto parcial) | `tts.speak` recebe texto completo. Texto incremental (chunked) e evolucao futura. |
| TTS Cancel RPC no proto | TTS e rapido (<500ms para frases curtas). Cancel via gRPC stream break e suficiente. |
| Co-scheduling sofisticado STT+TTS (shared GPU batching) | Workers sao isolados. Scheduling por tipo de worker. |
| LLM integration | O Theo fornece STT e TTS. LLM e responsabilidade do middleware do cliente. |
| PiperBackend | Backend alternativo. Implementavel no futuro seguindo o guia de ADDING_ENGINE.md (adaptado para TTS). |
| TTS streaming response no REST (chunked transfer) | `POST /v1/audio/speech` retorna audio completo. Streaming response e otimizacao futura. |

---

## 11. Decisoes Arquiteturais de M9

### ADR-M9-01: TTS MVP com Engine Placeholder

**Decisao**: M9 implementa infraestrutura TTS completa (interface, proto, worker, endpoint) com KokoroBackend como engine MVP. Se Kokoro nao estiver maduro, usar um SimpleTTSBackend que gera audio sintetizado basico (sine tone / buzzer) para validar a infraestrutura.

**Justificativa**: O valor de M9 esta na infraestrutura de full-duplex (mute-on-speak, coordenacao, protocolo), nao na qualidade da sintese. Uma engine placeholder valida 100% do pipeline. A qualidade do TTS e melhorada depois, trocando a engine via `TTSBackend` interface.

**Alternativa rejeitada**: Implementar TTS com qualidade de producao desde M9. Rejeitada porque dobraria o escopo sem beneficio para o objetivo principal (full-duplex).

**Consequencias**: O TTS em M9 pode ter qualidade de audio inferior a producao. Documentado como "MVP" no CHANGELOG e README.

### ADR-M9-02: Mute-on-Speak como Fallback (nao Barge-in)

**Decisao**: Mute-on-speak e o mecanismo de coordenacao STT/TTS. O STT e silenciado enquanto o TTS fala. Barge-in (usuario interrompe o bot) NAO e suportado em M9.

**Justificativa**: Barge-in requer AEC (Acoustic Echo Cancellation) para remover o eco do TTS do sinal do microfone. AEC e complexo, domain-specific (depende do hardware/ambiente), e fora do escopo do runtime. Mute-on-speak e simples, eficaz, e resolve 90% dos cenarios (o bot fala, o usuario espera, o usuario fala).

**Alternativa rejeitada**: Implementar AEC no runtime. Rejeitada por complexidade e por ser responsabilidade do cliente/hardware.

**Consequencias**: Em cenarios onde o usuario tenta interromper o bot, a interrupcao e ignorada ate o TTS terminar. Documentado como limitacao. O campo `barge_in_enabled` e reservado no protocolo para futuro (quando AEC externo estiver disponivel).

### ADR-M9-03: TTS Streaming Unidirecional (nao Bidirecional)

**Decisao**: O gRPC `Synthesize` RPC e server-streaming (request-response com stream de chunks de output), nao bidirecional.

**Justificativa**: TTS e fundamentalmente unidirecional: texto entra, audio sai. Nao ha necessidade de enviar texto incremental (como audio em STT). O texto completo e enviado no request; o servidor retorna chunks de audio a medida que sintetiza. Isso e mais simples, mais eficiente, e alinhado com o modelo de TTS de todas as engines existentes.

**Alternativa rejeitada**: Bidirecional streaming para TTS (enviar tokens de texto incrementalmente). Rejeitada porque nenhuma engine TTS suporta isso nativamente, e o ganho de latencia e negligivel (o texto completo e normalmente uma frase curta, <100 tokens).

### ADR-M9-04: Protocolo WebSocket Unificado (STT + TTS no Mesmo WS)

**Decisao**: STT e TTS compartilham a mesma conexao WebSocket (`/v1/realtime`). Nao existe endpoint WebSocket separado para TTS.

**Justificativa**: Full-duplex significa STT e TTS na mesma sessao. Separar em dois WebSockets adiciona complexidade de coordenacao para o cliente (sincronizar dois sockets, lidar com reconexao independente). Um unico WebSocket com comandos tipados (`tts.speak`, `session.configure`) e mais simples e alinhado com o modelo da OpenAI Realtime API.

**Alternativa rejeitada**: WebSocket separado para TTS (`/v1/tts/realtime`). Rejeitada porque mute-on-speak requer coordenacao instantanea entre STT e TTS -- mais facil com uma unica conexao.

**Consequencias**: O protocolo WebSocket fica mais complexo (mais tipos de evento/comando). Documentacao e essencial. O `dispatch_message()` cresce, mas cada handler e isolado.

---

## 12. Perspectivas Cruzadas

### Onde as Perspectivas se Encontram

| Intersecao | Sofia (Arquitetura) | Viktor (Real-Time) | Andre (Platform) |
|------------|--------------------|--------------------|------------------|
| **TTSBackend** | Design: ABC espelha STTBackend, synthesize como AsyncIterator | Performance: streaming chunks minimiza TTFB | Observabilidade: metricas desde dia 1 |
| **TTS Proto** | Contrato: server-streaming, nao bidirecional (KISS) | Latencia: primeiro chunk em <50ms para Kokoro | Deploy: make proto gera ambos (STT+TTS) |
| **Mute-on-speak** | Coordenacao: MuteController como ponto unico | Timing: mute ANTES do audio chegar ao client | Metrica: muted_frames_total para detectar excesso |
| **Full-Duplex WS** | Protocolo: unificado STT+TTS, comandos tipados | Concorrencia: TTS como background task, nao bloqueia STT | Metricas: V2V latency composta (STT + TTS) |
| **REST TTS** | Contrato: OpenAI-compatible, audio no body | Latencia: batch TTS nao tem constraint de real-time | Endpoint: reutiliza Scheduler existente |

### Consenso do Time

- **Sofia**: M9 e o milestone que completa a visao do PRD: runtime unificado STT+TTS. O risco principal e o escopo (TTS MVP + full-duplex + mute-on-speak e grande). A mitigacao e priorizar epics: E1 (infra TTS) e autonomo e entregavel sozinho. E2 (REST) valida TTS batch. E3 (full-duplex) e o core. Se o timeline aperta, E1+E2 viram M9a e E3+E4 viram M9b. A interface TTSBackend garante que trocar engine e trivial -- o Kokoro MVP nao bloqueia ninguem.

- **Viktor**: O mute-on-speak e surpreendentemente simples de implementar (um flag booleano). A complexidade real esta na integracao: garantir que unmute sempre acontece (try/finally), que TTS como background task nao bloqueia o event loop, e que o timing do mute e correto (antes do audio chegar ao cliente). O V2V latency budget de 300ms e o target aspiracional -- na pratica, 150ms do Theo (ASR + TTS) e realista, com o restante dependendo do LLM do cliente.

- **Andre**: Para operadores, M9 adiciona 5 metricas TTS que complementam as existentes de STT e Scheduler. A metrica mais importante e `tts_ttfb_seconds` -- ela mostra a contribuicao do TTS ao V2V latency e e o primeiro lugar a olhar quando latencia degrada. O `muted_frames_total` e um indicador de saude: valor muito alto indica que o TTS esta falando demais ou o STT esta lento em desmutar.

---

## 13. Transicao M9 -> Alem

Ao completar M9, o Theo OpenVoice tera:

1. **Runtime unificado STT + TTS** -- um binario que serve transcricao e sintese, conforme a visao original do PRD.

2. **Full-Duplex funcional** -- STT e TTS na mesma sessao WebSocket, coordenados por mute-on-speak.

3. **API OpenAI-compatible** -- `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `POST /v1/audio/speech`, `WS /v1/realtime`.

4. **Model-agnostic para STT e TTS** -- `STTBackend` (Faster-Whisper, WeNet) e `TTSBackend` (Kokoro) como interfaces plugaveis.

5. **Scheduler avancado** -- priorizacao, cancelamento, dynamic batching, latency tracking.

6. **Observabilidade completa** -- metricas STT, TTS, Scheduler, V2V latency.

### Evolucoes possiveis alem de M9

| Evolucao | Descricao | Prerequisito |
|----------|-----------|-------------|
| Barge-in com AEC | Usuario interrompe bot. Requer AEC externo + integracao | M9 (mute-on-speak como base) |
| TTS streaming response (REST) | Chunked transfer no `POST /v1/audio/speech` | M9 (infra TTS) |
| SSML support | Texto com markup de prosody, pausa, enfase | M9 (TTSBackend interface) |
| PiperBackend | Segunda engine TTS, validando TTSBackend como STTBackend foi validado em M7 | M9 (TTSBackend interface) |
| Multi-speaker / voice cloning | Vozes customizadas por sessao | M9 (TTSBackend interface) |
| V2V latency optimization | TTS TTFB <30ms, ASR final_delay <80ms | M9 (metricas V2V) |
| SIP/RTP ingestao direta | Ingestao de audio via SIP/RTP sem WebSocket intermediario | M9 (full-duplex funcional) |
| Auto-scaling de workers | HPA baseado em `active_sessions` e `queue_depth` | M8+M9 (metricas) |

---

*Documento gerado pelo Time de Arquitetura (ARCH) -- Sofia Castellani, Viktor Sorokin, Andre Oliveira. Sera atualizado conforme a implementacao do M9 avanca.*
