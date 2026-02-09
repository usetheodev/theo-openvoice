# M6 -- Session Manager + Ring Buffer -- Strategic Roadmap

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Planejado
**Data**: 2026-02-09

**Autores**:
- Sofia Castellani (Principal Solution Architect)
- Viktor Sorokin (Senior Real-Time Engineer)
- Andre Oliveira (Senior Platform Engineer)

---

## 1. Objetivo Estrategico

M6 e o milestone que transforma o streaming STT de M5 em um sistema de producao com gerenciamento completo de estado. E o componente mais original do Theo OpenVoice -- o Session Manager com maquina de estados de 6 estados, Ring Buffer com read fence, WAL in-memory para recovery, e LocalAgreement para partial transcripts. Nenhum projeto open-source de STT possui equivalente.

### O que M6 entrega

Um usuario pode manter sessoes de streaming de longa duracao (30+ minutos) com:
- Transicoes de estado explicitas e observaveis (INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED)
- Partial transcripts inteligentes via LocalAgreement (confirmacao de tokens entre passes)
- Recovery automatico apos crash de worker, sem duplicacao de segmentos
- Contexto entre segmentos para melhorar continuidade de transcricao
- Hot words configuraveis por sessao
- Ring Buffer com protecao de dados (read fence) e force commit automatico
- Streaming via microfone pelo CLI

### O que M6 habilita

- **Base para Multi-Engine (M7)**: O pipeline adaptativo por arquitetura (encoder-decoder vs CTC) depende do Session Manager para orquestrar LocalAgreement vs partials nativos.
- **Resiliencia de producao**: Recovery sem duplicacao permite uso em cenarios criticos (call center, banking).
- **Sessoes de longa duracao**: Estado HOLD permite chamadas em espera sem perder sessao.
- **Validacao de qualidade**: Partial transcripts via LocalAgreement e cross-segment context melhoram percepcion de qualidade pelo usuario.

### Criterio de sucesso (Demo Goal)

```
# 1. Sessao WebSocket de 30 minutos sem degradacao
wscat -c ws://localhost:8000/v1/realtime?model=faster-whisper-tiny

# 2. Partial transcripts inteligentes (LocalAgreement)
# <- {"type": "transcript.partial", "text": "ola como", ...}  (tokens confirmados)
# <- {"type": "transcript.final", "text": "ola como posso ajudar", ...}

# 3. Transicao para HOLD apos silencio prolongado
# <- {"type": "session.hold", "timestamp_ms": 34000, "hold_timeout_ms": 300000}

# 4. Recovery apos kill do worker
# (matar worker com kill -9)
# <- {"type": "error", "code": "worker_crash", "recoverable": true, "resume_segment_id": 5}
# (worker reinicia, sessao retoma do segmento 5 sem duplicacao)

# 5. CLI streaming do microfone
theo transcribe --stream --model faster-whisper-tiny

# 6. Metricas
curl http://localhost:8000/metrics | grep theo_stt_session_duration
curl http://localhost:8000/metrics | grep theo_stt_segments_force_committed
```

### Conexao com o Roadmap

```
PRD Fase 2 (Streaming)
  M5: WebSocket + VAD           [COMPLETO]
  M6: Session Manager           [<-- ESTE MILESTONE]
  M7: Segundo Backend

Caminho critico: M5 -> M6 -> M7
                         \-> M8 (RTP Listener)
```

M6 completa o core da Fase 2. Apos M6, o Theo tem streaming de producao com recovery, partial transcripts inteligentes e gerenciamento de sessoes. M7 valida que a abstracão e model-agnostic; M8 extende para telefonia.

---

## 2. Pre-requisitos (de M5)

### O que ja existe e sera REUTILIZADO sem mudancas

| Componente | Pacote | Uso em M6 |
|------------|--------|-----------|
| WebSocket endpoint `/v1/realtime` | `theo.server.routes.realtime` | Handler principal. M6 integra Session Manager ao handler existente. |
| Protocolo de eventos (Pydantic models) | `theo.server.models.events` | Todos os event types. M6 adiciona uso de `SessionHoldEvent` (ja definido). |
| Protocol handler (dispatch) | `theo.server.ws_protocol` | Dispatch de comandos e frames. Sem mudancas. |
| Heartbeat e inactivity monitor | `theo.server.routes.realtime` | Mantido. Timeouts serao gerenciados pela state machine. |
| Energy Pre-filter | `theo.vad.energy` | Sem mudancas. |
| Silero VAD Classifier | `theo.vad.silero` | Sem mudancas. |
| VAD Detector | `theo.vad.detector` | Sem mudancas. Emite VADEvents que a state machine consome. |
| Streaming Preprocessor | `theo.preprocessing.streaming` | Sem mudancas. |
| gRPC streaming (StreamingGRPCClient + StreamHandle) | `theo.scheduler.streaming` | Sem mudancas na interface. M6 reutiliza. |
| Worker STT (servicer + FasterWhisperBackend) | `theo.workers.stt.*` | Sem mudancas. |
| Backpressure Controller | `theo.session.backpressure` | Sem mudancas. |
| Streaming Metrics (TTFB, final_delay, active_sessions, vad_events) | `theo.session.metrics` | Extendido com metricas novas. |
| `SessionState` enum (6 estados) | `theo._types` | Ja definido com transicoes documentadas. |

### O que sera EVOLUIDO

| Componente | Pacote | O que muda |
|------------|--------|------------|
| `StreamingSession` | `theo.session.streaming` | Estado simplificado (ACTIVE/CLOSED) substituido pela `SessionStateMachine`. Ring Buffer inserido entre preprocessing e gRPC. LocalAgreement inserido entre Ring Buffer e worker. Cross-segment context adicionado. Hot words por sessao. |
| `_send_frame_to_worker()` | `theo.session.streaming` | Frames vao para Ring Buffer em vez de direto ao gRPC. |
| `_handle_speech_start/end()` | `theo.session.streaming` | Delegam transicoes para `SessionStateMachine`. |
| Inactivity check | `theo.session.streaming` | Substituido por timeouts da state machine (por estado). |
| WebSocket endpoint | `theo.server.routes.realtime` | `_create_streaming_session()` passa config de timeouts, hot words, e cria dependencias adicionais (Ring Buffer, WAL). |
| Metricas | `theo.session.metrics` | Novas metricas: `session_duration_seconds`, `segments_force_committed_total`, `confidence_avg`. |

---

## 3. Visao Geral da Arquitetura M6

### 3.1 Decomposicao de Componentes

A perspectiva da Sofia: resistir a tentacao de God class. O Session Manager e decomposto em sub-componentes testáveis isoladamente.

```
+------------------------------------------------------------------+
|                      StreamingSession (EVOLUIDO)                   |
|                                                                    |
|  Orquestrador central. Delega para sub-componentes:               |
|                                                                    |
|  +-----------------------+  +-------------------+                  |
|  | SessionStateMachine   |  | RingBuffer        |                  |
|  |                       |  |                   |                  |
|  | 6 estados com         |  | 60s pre-alocado   |                  |
|  | transicoes validas    |  | read/write ptrs   |                  |
|  | timeouts por estado   |  | read fence        |                  |
|  | callbacks on_enter/   |  | force commit @90% |                  |
|  | on_exit               |  |                   |                  |
|  +-----------------------+  +-------------------+                  |
|                                                                    |
|  +-----------------------+  +-------------------+                  |
|  | SessionWAL            |  | LocalAgreement    |                  |
|  |                       |  | Policy            |                  |
|  | last_committed_seg_id |  |                   |                  |
|  | last_committed_offset |  | Comparacao entre  |                  |
|  | last_committed_ts_ms  |  | passes para       |                  |
|  | Recovery sem          |  | confirmar tokens  |                  |
|  | duplicacao            |  | Partials          |                  |
|  +-----------------------+  | inteligentes      |                  |
|                              +-------------------+                  |
|                                                                    |
|  +-----------------------+                                         |
|  | CrossSegmentContext   |                                         |
|  |                       |                                         |
|  | Ultimos 224 tokens    |                                         |
|  | do final anterior     |                                         |
|  | como initial_prompt   |                                         |
|  +-----------------------+                                         |
+------------------------------------------------------------------+
```

### 3.2 Data Flow com Ring Buffer e LocalAgreement

```
WebSocket frame (binary, PCM)
    |
    v
StreamingPreprocessor (reutilizado M5)
    |  float32 16kHz mono
    v
VADDetector (reutilizado M5)
    |  VADEvent (speech_start / speech_end)
    v
SessionStateMachine (NOVO)
    |  Transicoes de estado: INIT->ACTIVE->SILENCE->HOLD->CLOSING->CLOSED
    |  Timeouts por estado
    |  Emite eventos: session.hold
    v
RingBuffer (NOVO)
    |  Armazena frames float32
    |  Read fence protege dados nao commitados
    |  Force commit em 90% de capacidade
    v
LocalAgreementPolicy (NOVO)
    |  Acumula window (3-5s)
    |  Compara com pass anterior
    |  Tokens confirmados -> transcript.partial
    v
gRPC StreamHandle (reutilizado M5)
    |  AudioFrame com initial_prompt (CrossSegmentContext)
    |  AudioFrame com hot_words (primeira vez por segmento)
    v
Worker (reutilizado M5)
    |  TranscriptEvent (partial / final)
    v
LocalAgreementPolicy
    |  Compara com output anterior
    |  Confirma tokens -> emite partial
    v
Post-processing (ITN) -- apenas em final
    v
WebSocket event JSON
    |
    v
SessionWAL (NOVO)
    |  Registra checkpoint apos final emitido
    |  last_committed_segment_id, offset, timestamp
```

### 3.3 Recovery Flow (Crash de Worker)

```
Worker crashou durante ACTIVE
    |
    v
1. gRPC stream break detectado (imediato - StreamHandle raises WorkerCrashError)
    |
    v
2. SessionStateMachine: estado permanece ACTIVE (nao transita para CLOSED)
    |
    v
3. Emite "error" com recoverable: true + resume_segment_id ao cliente
    |
    v
4. WorkerManager reinicia worker (ja implementado em M2)
    |
    v
5. Consulta SessionWAL:
   - last_committed_segment_id = 5
   - last_committed_buffer_offset = 48000
   - last_committed_timestamp_ms = 15000
    |
    v
6. Abre novo gRPC stream
    |
    v
7. RingBuffer: lê dados apos last_committed_offset (protegidos pelo read fence)
    |
    v
8. Reenvia dados nao commitados ao novo worker
    |
    v
9. Sessao retomada a partir do segmento 6 -- zero duplicacao
```

---

## 4. Epics

### Epic 1: Session State Machine

Implementar a maquina de estados completa com 6 estados, transicoes validas, timeouts configuraveis por estado, e integracao com o fluxo de streaming existente.

**Racional**: A maquina de estados e o nucleo do Session Manager. Sem ela, o runtime nao distingue entre fala ativa, silencio transitorio e hold prolongado. E o componente que permite sessoes de 30+ minutos em call center sem degradacao.

**Responsavel principal**: Sofia (design da state machine, transicoes, contratos)

**Tasks**: M6-01, M6-02, M6-03

### Epic 2: Ring Buffer + WAL

Implementar o buffer circular pre-alocado com read fence, force commit, e o Write-Ahead Log in-memory para recovery sem duplicacao.

**Racional**: O Ring Buffer e o que permite recovery apos crash de worker. Sem ele, dados de audio sao perdidos quando o worker morre. O WAL garante que a retomada nao duplica segmentos. E tambem pre-requisito para LocalAgreement (que opera sobre windows no Ring Buffer).

**Responsavel principal**: Viktor (implementacao de baixo nivel, zero allocations, performance)

**Tasks**: M6-04, M6-05, M6-06, M6-07

### Epic 3: LocalAgreement + Cross-Segment + Hot Words

Implementar partial transcripts inteligentes via LocalAgreement, contexto entre segmentos via initial_prompt, e hot words configuraveis por sessao.

**Racional**: LocalAgreement e o que diferencia os partials do Theo de qualquer outro projeto. Em vez de reprocessar todo o audio a cada 500ms (custo proibitivo), compara output entre passes consecutivas e confirma tokens que concordam. Cross-segment context melhora continuidade em frases cortadas no limite de segmentos.

**Responsavel principal**: Sofia (algoritmo LocalAgreement, cross-segment), Viktor (integracao com Ring Buffer)

**Tasks**: M6-08, M6-09, M6-10

### Epic 4: Integracao, CLI, Metricas e Estabilidade

Integrar todos os componentes no fluxo end-to-end, implementar CLI `theo transcribe --stream`, adicionar metricas de M6, e validar estabilidade com testes de 30 minutos e recovery.

**Racional**: Os componentes de E1-E3 sao testados isoladamente. E4 valida que funcionam juntos: state machine coordenando Ring Buffer e LocalAgreement, recovery via WAL, metricas expostas, e CLI usavel.

**Responsavel principal**: Andre (CLI, metricas, testes de estabilidade), com suporte de Sofia e Viktor

**Tasks**: M6-11, M6-12, M6-13, M6-14, M6-15, M6-16

---

## 5. Tasks (Detalhadas)

### M6-01: SessionStateMachine -- Core da Maquina de Estados

**Epic**: E1 -- Session State Machine
**Estimativa**: M (3-5 dias)
**Dependencias**: Nenhuma (componente isolado, usa `SessionState` enum de `_types.py`)
**Desbloqueia**: M6-02, M6-03

**Contexto/Motivacao**: A maquina de estados e o componente mais critico de M6. Define 6 estados com transicoes validas e timeouts configuraveis. O `SessionState` enum ja existe em `theo._types` com transicoes documentadas. A `SessionStateMachine` implementa a logica de transicao, validacao de transicoes invalidas, e callbacks.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `SessionStateMachine` com 6 estados | Integracao com StreamingSession (M6-03) |
| Transicoes validas conforme docstring de `SessionState` | Integracao com Ring Buffer (M6-05) |
| Timeouts configuraveis por estado (INIT: 30s, SILENCE: 30s, HOLD: 5min, CLOSING: 2s) | WAL (M6-06) |
| Metodo `transition(target_state)` com validacao | |
| `InvalidTransitionError` para transicoes invalidas | |
| Callbacks `on_enter` e `on_exit` por estado | |
| Metodo `check_timeout() -> SessionState | None` que retorna estado alvo se timeout expirou | |
| Propriedade `state` (estado atual), `elapsed_in_state_ms` | |
| Clock injetavel para testes deterministicos | |

**Entregaveis**:
- `src/theo/session/state_machine.py` -- `SessionStateMachine`
- Nova exception `InvalidTransitionError` em `src/theo/exceptions.py`
- `tests/unit/test_session_state_machine.py`

**DoD**:
- [ ] Todas as transicoes validas do `SessionState` docstring funcionam
- [ ] Transicoes invalidas (ex: INIT->HOLD, CLOSED->ACTIVE) levantam `InvalidTransitionError`
- [ ] Timeout de INIT (30s) transita para CLOSED
- [ ] Timeout de SILENCE (30s) transita para HOLD
- [ ] Timeout de HOLD (5min) transita para CLOSING
- [ ] Timeout de CLOSING (2s) transita para CLOSED
- [ ] Callbacks `on_enter`/`on_exit` sao chamados na transicao
- [ ] Estado CLOSED e terminal -- nenhuma transicao e aceita
- [ ] Qualquer estado pode transitar para CLOSED (erro irrecuperavel)
- [ ] Testes: >=20 testes cobrindo todos os estados, transicoes validas, transicoes invalidas, timeouts, callbacks, e estado terminal
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O `check_timeout()` deve ser O(1) -- apenas comparar elapsed time vs timeout configurado. Nao usar timer tasks por estado (overhead de asyncio.Task por sessao e inaceitavel em escala). O caller (StreamingSession) chama `check_timeout()` periodicamente.

---

### M6-02: Timeouts Configuraveis via session.configure

**Epic**: E1 -- Session State Machine
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-01
**Desbloqueia**: M6-03

**Contexto/Motivacao**: Os timeouts da state machine devem ser configuraveis via `session.configure` do protocolo WebSocket. O `SessionConfigureCommand` ja tem campos `silence_timeout_ms`, `hold_timeout_ms`, `max_segment_duration_ms`. M6-02 conecta esses campos a `SessionStateMachine`.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Dataclass `SessionTimeouts` com defaults do PRD | |
| Metodo `update_timeouts(timeouts)` na `SessionStateMachine` | |
| Validacao: timeout minimo de 1s para qualquer estado | |
| Conversao dos campos de `SessionConfigureCommand` para `SessionTimeouts` | |

**Entregaveis**:
- `SessionTimeouts` dataclass em `src/theo/session/state_machine.py`
- Logica de update em `SessionStateMachine`
- `tests/unit/test_session_timeouts.py`

**DoD**:
- [ ] Defaults: INIT=30s, SILENCE=30s, HOLD=5min, CLOSING=2s
- [ ] `session.configure` com `silence_timeout_ms: 5000` reduz timeout de SILENCE para 5s
- [ ] Timeout minimo de 1s validado -- valores menores levantam `ValueError`
- [ ] Mudanca de timeout em runtime afeta estado atual (se em SILENCE e timeout muda, recalcula)
- [ ] Testes: >=8 testes cobrindo defaults, override, validacao, runtime update
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-03: Integracao State Machine com StreamingSession

**Epic**: E1 -- Session State Machine
**Estimativa**: L (5-7 dias)
**Dependencias**: M6-01, M6-02
**Desbloqueia**: M6-05, M6-08, M6-11

**Contexto/Motivacao**: A `StreamingSession` de M5 usa estado simplificado (`_SessionState.ACTIVE/CLOSED`). M6-03 substitui esse estado pela `SessionStateMachine`, alterando o fluxo de eventos VAD para transitar entre estados. A integracao e o ponto critico: VAD events disparam transicoes, timeouts sao verificados periodicamente, e o estado determina o comportamento (ex: frames em HOLD nao sao enviados ao worker).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Substituir `_SessionState` por `SessionStateMachine` no `StreamingSession` | Ring Buffer (M6-05) -- frames continuam indo direto ao gRPC |
| VAD SPEECH_START: INIT->ACTIVE ou SILENCE->ACTIVE ou HOLD->ACTIVE | LocalAgreement (M6-08) |
| VAD SPEECH_END: ACTIVE->SILENCE | WAL (M6-06) |
| Timeout de SILENCE: SILENCE->HOLD com emissao de `SessionHoldEvent` | |
| Timeout de HOLD: HOLD->CLOSING | |
| CLOSING: flush de pendentes, CLOSING->CLOSED | |
| Timeout de INIT: INIT->CLOSED (sem audio recebido) | |
| Substituir `_INACTIVITY_TIMEOUT_S` pelo timeout de INIT da state machine | |
| Emissao de `SessionHoldEvent` ao transitar para HOLD | |
| Verificacao periodica de timeouts (integrado ao monitor de inatividade existente) | |
| Comportamento por estado: HOLD nao envia frames ao worker (economia de GPU) | |

**Entregaveis**:
- Alteracao em `src/theo/session/streaming.py` -- `StreamingSession` evoluido
- `tests/unit/test_session_state_integration.py`

**DoD**:
- [ ] Estado inicial e INIT (nao ACTIVE como em M5)
- [ ] Primeiro frame com fala transita INIT->ACTIVE
- [ ] VAD speech_end transita ACTIVE->SILENCE
- [ ] 30s de silencio transita SILENCE->HOLD com emissao de `SessionHoldEvent`
- [ ] Fala durante HOLD transita HOLD->ACTIVE (sessao retoma)
- [ ] 5min em HOLD transita HOLD->CLOSING
- [ ] CLOSING flush pendentes e transita para CLOSED em <=2s
- [ ] 30s sem audio em INIT transita para CLOSED
- [ ] Frames recebidos em HOLD nao sao enviados ao worker
- [ ] `SessionHoldEvent` emitido com `hold_timeout_ms` correto
- [ ] Testes M5 de StreamingSession continuam passando (regressao zero)
- [ ] Testes: >=15 testes cobrindo transicoes, timeouts, comportamento por estado, hold event
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: A integracao deve ser feita preservando a interface publica de `StreamingSession` (`process_frame`, `commit`, `close`, `is_closed`). O WebSocket handler (`realtime.py`) nao deve precisar mudar. Toda a complexidade da state machine e interna.

**Perspectiva Viktor**: A verificacao de timeouts deve ser integrada ao loop de monitor existente (`_inactivity_monitor` em `realtime.py`) em vez de criar uma nova asyncio.Task por sessao. O monitor verifica `session.check_timeout()` e atua conforme o estado retornado.

---

### M6-04: Ring Buffer -- Buffer Circular Pre-alocado

**Epic**: E2 -- Ring Buffer + WAL
**Estimativa**: M (3-5 dias)
**Dependencias**: Nenhuma (componente isolado)
**Desbloqueia**: M6-05, M6-06

**Contexto/Motivacao**: O Ring Buffer e um array pre-alocado de 60s de audio que armazena frames recentes. E essencial para recovery (reprocessar audio apos crash de worker) e para LocalAgreement (acumular windows de 3-5s). Implementado com `bytearray` pre-alocado e indices circulares. Zero allocations durante streaming.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `RingBuffer` com array pre-alocado (60s = 1,920,000 bytes PCM 16kHz mono) | Integracao com StreamingSession (M6-05) |
| Ponteiros `write_pos` e `read_pos` circulares | LocalAgreement (M6-08) |
| Metodo `write(data: bytes) -> int` que retorna offset absoluto | |
| Metodo `read(offset: int, length: int) -> bytes` que le a partir de offset | |
| Metodo `read_from_offset(offset: int) -> bytes` que le tudo de offset ate write_pos | |
| Propriedade `capacity_bytes`, `used_bytes`, `usage_percent` | |
| Suporte a wrap-around (escrita circular quando atinge fim do array) | |
| Tamanho configuravel via parametro `duration_s` e `sample_rate` | |
| Sem locking (single-threaded, roda no event loop asyncio) | |

**Entregaveis**:
- `src/theo/session/ring_buffer.py` -- `RingBuffer`
- `tests/unit/test_ring_buffer.py`

**DoD**:
- [ ] Buffer pre-alocado com tamanho correto (60s * 16000 * 2 = 1,920,000 bytes)
- [ ] Write: dados escritos corretamente, offset absoluto retornado
- [ ] Write com wrap-around: dados que excedem capacidade sobrescrevem do inicio
- [ ] Read: dados lidos corretamente a partir de offset absoluto
- [ ] Read com wrap-around: le corretamente quando dados cruzam a borda
- [ ] `usage_percent` calcula corretamente (incluindo wrap-around)
- [ ] Zero allocations apos init (verificavel via tracemalloc ou inspection do bytearray)
- [ ] Tamanho configuravel: `RingBuffer(duration_s=30)` cria buffer de 30s
- [ ] Testes: >=15 testes cobrindo escrita, leitura, wrap-around, capacidade, edge cases (buffer cheio, leitura alem do disponivel)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: Usar `bytearray` pre-alocado com `memoryview` para zero-copy onde possivel. NAO usar `deque`, `list`, ou `io.BytesIO` -- todos fazem allocations. O offset absoluto (nao circular) e essencial para o WAL: o offset 1,920,001 significa que ja deu uma volta completa, e isso deve ser rastreavel.

---

### M6-05: Read Fence + Force Commit

**Epic**: E2 -- Ring Buffer + WAL
**Estimativa**: M (3-5 dias)
**Dependencias**: M6-04, M6-03
**Desbloqueia**: M6-06, M6-07, M6-08

**Contexto/Motivacao**: O read fence (`last_committed_offset`) protege dados que ainda nao foram confirmados via `transcript.final`. Sem read fence, o wrap-around do Ring Buffer poderia sobrescrever audio que precisaria ser reprocessado em caso de crash. O force commit e o mecanismo de seguranca: se o buffer atinge 90% sem commit, forca o segmento atual a ser finalizado.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Propriedade `read_fence` (offset do ultimo commit) no `RingBuffer` | WAL (M6-06) -- read fence e atualizado, mas WAL registra separadamente |
| Metodo `commit(offset: int)` que avanca o read fence | Recovery (M6-07) |
| Protecao: write nao sobrescreve dados apos read fence | |
| Metodo `available_for_write_bytes` que considera read fence | |
| Propriedade `uncommitted_bytes` (dados entre fence e write_pos) | |
| Callback `on_force_commit` chamado quando usage > 90% | |
| Integracao com `StreamingSession`: frames escritos no Ring Buffer | |
| `StreamingSession` alimenta worker a partir do Ring Buffer (nao direto do preprocessing) | |
| Force commit: ao atingir 90%, StreamingSession fecha stream gRPC atual | |

**Entregaveis**:
- Alteracao em `src/theo/session/ring_buffer.py` -- read fence e force commit
- Alteracao em `src/theo/session/streaming.py` -- integracao com Ring Buffer
- `tests/unit/test_ring_buffer_fence.py`

**DoD**:
- [ ] `read_fence` inicia em 0 (nenhum dado commitado)
- [ ] `commit(offset)` avanca fence para o offset especificado
- [ ] Write que sobrescreveria dados apos fence e bloqueado (retorna -1 ou levanta exception)
- [ ] `uncommitted_bytes` retorna tamanho correto entre fence e write_pos
- [ ] Force commit disparado quando `usage_percent > 90` e `uncommitted_bytes > 0`
- [ ] Callback `on_force_commit` chamado com offset do dados uncommitted
- [ ] StreamingSession.process_frame: frame vai para Ring Buffer, nao direto ao gRPC
- [ ] StreamingSession alimenta worker lendo do Ring Buffer durante fala
- [ ] Testes: >=12 testes cobrindo fence, commit, protecao de sobrescrita, force commit, integracao
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O force commit NAO deve bloquear a escrita de novos frames. Se atingir 90%, a callback `on_force_commit` e chamada assincronamente e os novos frames continuam sendo escritos. O commit libera espaco para que o buffer nao atinja 100%.

---

### M6-06: WAL In-Memory (Write-Ahead Log)

**Epic**: E2 -- Ring Buffer + WAL
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-05
**Desbloqueia**: M6-07

**Contexto/Motivacao**: O WAL registra checkpoints da sessao em memoria para permitir recovery sem duplicacao. Cada `transcript.final` emitido gera um checkpoint no WAL com segment_id, buffer_offset e timestamp. No recovery, o Session Manager consulta o WAL para saber de onde retomar.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `SessionWAL` com registro de checkpoints | Persistencia em disco (WAL e in-memory) |
| Metodo `checkpoint(segment_id, buffer_offset, timestamp_ms)` | Recovery automatico (M6-07) |
| Propriedade `last_committed_segment_id` | |
| Propriedade `last_committed_buffer_offset` | |
| Propriedade `last_committed_timestamp_ms` | |
| Escrita atomica (um registro por vez, sem race condition) | |
| Integracao com StreamingSession: checkpoint apos emitir transcript.final | |

**Entregaveis**:
- `src/theo/session/wal.py` -- `SessionWAL`
- `tests/unit/test_session_wal.py`

**DoD**:
- [ ] `SessionWAL` inicializa com segment_id=0, offset=0, timestamp=0
- [ ] `checkpoint()` atualiza todos os campos atomicamente
- [ ] `last_committed_segment_id` retorna o ultimo segment_id commitado
- [ ] `last_committed_buffer_offset` retorna o ultimo offset commitado
- [ ] `last_committed_timestamp_ms` retorna o ultimo timestamp commitado
- [ ] Multiplos checkpoints: cada um sobrescreve o anterior (WAL in-memory, nao append-only)
- [ ] Integracao: apos `transcript.final` emitido, StreamingSession chama `wal.checkpoint()`
- [ ] Integracao: apos `checkpoint()`, StreamingSession chama `ring_buffer.commit(offset)`
- [ ] Testes: >=8 testes cobrindo init, checkpoint, propriedades, multiplos checkpoints, integracao
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O WAL e deliberadamente simples. In-memory, um registro, sobrescreve. Nao e um log append-only -- e um ponteiro para "onde estamos". Complexidade minima para maximo valor.

---

### M6-07: Recovery de Crash -- Retomada sem Duplicacao

**Epic**: E2 -- Ring Buffer + WAL
**Estimativa**: L (5-7 dias)
**Dependencias**: M6-05, M6-06
**Desbloqueia**: M6-15 (teste de recovery)

**Contexto/Motivacao**: Quando o worker crasha durante uma sessao ACTIVE, o runtime deve detectar (via gRPC stream break, ja implementado em M5), reiniciar o worker, e retomar a sessao de onde parou. Os dados no Ring Buffer apos o `last_committed_offset` sao reprocessados. O segment_id do WAL garante que nenhum segmento e duplicado.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Metodo `recover()` na `StreamingSession` que orquestra o recovery | Recovery de crash do runtime inteiro (apenas worker) |
| Deteccao de crash (ja existe via WorkerCrashError em StreamHandle) | Recuperacao de estado da state machine apos restart do runtime |
| Apos detectar crash: manter state machine em ACTIVE (nao transitar para CLOSED) | |
| Emitir `StreamingErrorEvent(recoverable=True, resume_segment_id=N)` | |
| Aguardar WorkerManager reiniciar worker (ja implementado em M2) | |
| Abrir novo stream gRPC com o worker reiniciado | |
| Ler dados nao commitados do Ring Buffer (apos fence) | |
| Reenviar dados ao novo worker via gRPC | |
| Restaurar segment_id do WAL (evitar duplicacao) | |
| Timeout de recovery: se worker nao volta em 10s, transitar para CLOSED | |

**Entregaveis**:
- Metodo `recover()` em `src/theo/session/streaming.py`
- Alteracao em `_receive_worker_events()` para chamar `recover()` ao detectar crash
- `tests/unit/test_session_recovery.py`

**DoD**:
- [ ] Worker crash durante ACTIVE detectado e recovery iniciado automaticamente
- [ ] `StreamingErrorEvent(recoverable=True)` emitido ao cliente com `resume_segment_id` correto
- [ ] Novo stream gRPC aberto apos worker reiniciar
- [ ] Dados nao commitados do Ring Buffer reenviados ao novo worker
- [ ] `segment_id` retomado do WAL (sem duplicacao)
- [ ] Sessao continua normalmente apos recovery (novos frames processados)
- [ ] Se worker nao reinicia em 10s, sessao transita para CLOSED com error irrecuperavel
- [ ] Testes: >=10 testes cobrindo crash detection, recovery com dados no buffer, recovery sem dados, timeout de recovery, duplicacao evitada
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O recovery deve ser testavel deterministicamente. Injetar `WorkerManager` mock que simula crash e restart com timing controlado. O timeout de 10s deve usar clock injetavel. Testar cenario: matar worker (mock crash), verificar que dados apos fence sao reenviados, e que segment_id nao volta atras.

---

### M6-08: LocalAgreement -- Partial Transcripts Inteligentes

**Epic**: E3 -- LocalAgreement + Cross-Segment + Hot Words
**Estimativa**: L (5-7 dias)
**Dependencias**: M6-05 (Ring Buffer integrado ao fluxo)
**Desbloqueia**: M6-09, M6-11

**Contexto/Motivacao**: Para engines encoder-decoder (Whisper), partials nativos nao existem. M5 emite partial apenas quando o worker retorna resultado apos acumular audio. LocalAgreement melhora significativamente: compara output entre passes consecutivas sobre windows incrementais, e confirma tokens que concordam entre passes. Conceito adaptado do whisper-streaming (UFAL), implementacao propria integrada ao Ring Buffer.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `LocalAgreementPolicy` | Suporte a CTC/streaming-native (partials nativos, nao precisam de LocalAgreement) |
| Acumula window de audio do Ring Buffer (configuravel, default 3s) | Hot Word Correction post-processing (M6-10 trata hot words no worker, nao pos) |
| Envia window ao worker via gRPC | |
| Recebe output do worker (tokens com timestamps) | |
| Compara com output da pass anterior | |
| Tokens que concordam entre 2+ passes sao confirmados -> `transcript.partial` | |
| Tokens novos/divergentes sao retidos (aguardam proxima pass) | |
| Ao receber VAD speech_end, flush: emite todos os tokens retidos como `transcript.final` | |
| Configuravel: `min_confirm_passes` (default: 2), `window_ms` (default: 3000) | |
| Desabilitavel via config: `partial_strategy: disabled` (emite apenas finals) | |
| Flag `enable_partial_transcripts` do `SessionConfigureCommand` controla ativacao | |

**Entregaveis**:
- `src/theo/session/local_agreement.py` -- `LocalAgreementPolicy`
- Alteracao em `src/theo/session/streaming.py` -- integrar LocalAgreement no fluxo
- `tests/unit/test_local_agreement.py`

**DoD**:
- [ ] Window de 3s acumulada do Ring Buffer e enviada ao worker
- [ ] Output do worker comparado com pass anterior token-a-token
- [ ] Tokens confirmados (concordancia em 2+ passes) emitidos como `transcript.partial`
- [ ] Tokens divergentes retidos ate proxima pass
- [ ] VAD speech_end causa flush: todos os tokens (confirmados + retidos) viram `transcript.final`
- [ ] `partial_strategy: disabled` emite apenas `transcript.final` (sem partials)
- [ ] Custo: ~1 inferencia por window (nao 2/s como naive)
- [ ] Latencia de partial: <=300ms apos window acumulada (target TTFB)
- [ ] Testes: >=15 testes cobrindo confirmacao de tokens, divergencia, flush, disabled mode, edge cases (primeira pass, audio curto, tokens vazios)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O `LocalAgreementPolicy` e puro -- recebe tokens, retorna decisao (confirmar/reter). Nao conhece gRPC, Ring Buffer ou WebSocket. A `StreamingSession` orquestra: le do Ring Buffer, envia ao worker, passa output ao LocalAgreement, emite evento conforme decisao. Separacao de concerns e essencial para testabilidade.

**Perspectiva Viktor**: A comparacao de tokens deve ser eficiente. Usar comparacao posicional (offset no texto), nao Levenshtein. Se token N na pass atual e igual ao token N da pass anterior, confirmar. Se token N difere, reter a partir de N. Custo da comparacao: O(n) onde n e numero de tokens na window.

---

### M6-09: Cross-Segment Context (initial_prompt)

**Epic**: E3 -- LocalAgreement + Cross-Segment + Hot Words
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-08
**Desbloqueia**: M6-11

**Contexto/Motivacao**: Em sessoes longas, o Whisper pode perder contexto entre segmentos consecutivos. Cross-segment context envia os ultimos 224 tokens do `transcript.final` anterior como `initial_prompt` do proximo segmento. Melhora continuidade em frases cortadas no limite de segmentos.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `CrossSegmentContext` que armazena ultimos N tokens | Entity Formatting / Hot Word Correction stages |
| Metodo `update(text: str)` apos cada `transcript.final` | |
| Metodo `get_prompt() -> str | None` que retorna contexto atual | |
| Configuracao: `max_tokens` (default: 224, metade do context window do Whisper) | |
| Integracao com StreamingSession: passa `initial_prompt` ao gRPC via `StreamHandle.send_frame()` | |
| Aplicavel apenas para engines com `supports_initial_prompt: true` | |

**Entregaveis**:
- `src/theo/session/cross_segment.py` -- `CrossSegmentContext`
- Alteracao em `src/theo/session/streaming.py` -- integracao
- `tests/unit/test_cross_segment_context.py`

**DoD**:
- [ ] Apos `transcript.final` com texto "Ola, como posso ajudar?", `get_prompt()` retorna esse texto
- [ ] Texto acumulado respeita limite de 224 tokens (trunca do inicio se exceder)
- [ ] `initial_prompt` enviado ao worker no primeiro frame de cada novo segmento
- [ ] Para engine sem `supports_initial_prompt`, `get_prompt()` retorna None
- [ ] Testes: >=8 testes cobrindo acumulacao, truncamento, limpeza, integracao
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-10: Hot Words por Sessao via session.configure

**Epic**: E3 -- LocalAgreement + Cross-Segment + Hot Words
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-03 (state machine integrada)
**Desbloqueia**: M6-11

**Contexto/Motivacao**: M5 ja suporta hot words passados ao worker via `AudioFrame.hot_words`. O que falta e a integracao completa: `session.configure` com `hot_words` e `hot_word_boost` deve ser persistido na sessao, e os hot words devem ser combinados com cross-segment context no `initial_prompt` para Whisper (ex: "Termos: PIX, TED, Selic. Contexto anterior: Ola como posso ajudar?").

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Hot words persistidos na sessao (ja parcialmente implementado em M5) | Hot Word Correction post-processing (Levenshtein) -- futuro |
| Hot words combinados com cross-segment context no `initial_prompt` para Whisper | |
| Formato de injecao: "Termos: PIX, TED, Selic." como prefixo do initial_prompt | |
| `session.configure` atualiza hot words e boost em runtime | |

**Entregaveis**:
- Alteracao em `src/theo/session/streaming.py` -- combinacao hot words + context
- `tests/unit/test_hot_words_session.py`

**DoD**:
- [ ] `session.configure(hot_words=["PIX", "TED"])` persiste hot words na sessao
- [ ] Hot words combinados com cross-segment context: "Termos: PIX, TED. Ola como posso ajudar?"
- [ ] Hot words enviados como `initial_prompt` (nao `hot_words` field) para Whisper
- [ ] Hot words atualizaveis em runtime via `session.configure`
- [ ] Sem hot words configurados, apenas cross-segment context e enviado
- [ ] Testes: >=6 testes cobrindo configuracao, combinacao com context, atualizacao, sem hot words
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-11: Integracao End-to-End -- Todos os Componentes

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: L (5-7 dias)
**Dependencias**: M6-03, M6-05, M6-08, M6-09, M6-10
**Desbloqueia**: M6-12, M6-13, M6-14

**Contexto/Motivacao**: Os componentes de E1-E3 sao desenvolvidos e testados isoladamente. M6-11 e o momento de conectar tudo: state machine, Ring Buffer, LocalAgreement, cross-segment context, hot words, WAL, e recovery. O WebSocket handler (`realtime.py`) precisa ser atualizado para criar e injetar todas as novas dependencias.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Atualizar `_create_streaming_session()` em `realtime.py` para injetar Ring Buffer, WAL, LocalAgreement, CrossSegmentContext | CLI (M6-12) |
| Verificar que `session.configure` propaga corretamente para todos os sub-componentes | Metricas (M6-13) |
| Verificar que o fluxo completo funciona: WS -> preprocessing -> VAD -> state machine -> Ring Buffer -> LocalAgreement -> gRPC -> post-processing -> WS | Testes de estabilidade (M6-15/M6-16) |
| Verificar que recovery funciona end-to-end (crash de worker -> retomada) | |
| Testes de integracao WebSocket com todos os componentes | |

**Entregaveis**:
- Alteracao em `src/theo/server/routes/realtime.py` -- factory atualizada
- `tests/integration/test_m6_e2e.py`

**DoD**:
- [ ] Fluxo completo funciona: WS connect -> audio -> vad.speech_start -> transcript.partial (LocalAgreement) -> transcript.final -> vad.speech_end
- [ ] State machine transita corretamente: INIT -> ACTIVE -> SILENCE -> (timeout) -> HOLD -> (fala) -> ACTIVE
- [ ] `session.hold` emitido quando transita para HOLD
- [ ] Force commit disparado quando Ring Buffer atinge 90%
- [ ] Cross-segment context enviado no initial_prompt do proximo segmento
- [ ] Hot words configurados via session.configure e enviados ao worker
- [ ] Recovery apos crash de worker funciona (com mock de WorkerManager)
- [ ] Todos os testes de M5 continuam passando (regressao zero)
- [ ] Testes: >=12 testes de integracao cobrindo fluxo completo, transicoes, recovery, force commit
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-12: CLI -- `theo transcribe --stream`

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: M (3-5 dias)
**Dependencias**: M6-11
**Desbloqueia**: Nenhuma (leaf task)

**Contexto/Motivacao**: O CLI `theo transcribe` atualmente suporta apenas arquivo (batch via HTTP). M6-12 adiciona `--stream` que captura audio do microfone, envia via WebSocket, e exibe transcricoes no terminal em tempo real. E a primeira experiencia interativa do Theo para desenvolvedores.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Flag `--stream` no comando `theo transcribe` | Audio input de arquivo via WebSocket (existe via REST) |
| Captura de audio do microfone via `sounddevice` ou `pyaudio` | Interface grafica |
| Conexao WebSocket ao servidor local | |
| Envio de frames PCM 16-bit 16kHz mono | |
| Exibicao de transcript.partial (overwrite na mesma linha) e transcript.final (nova linha) | |
| Exibicao de eventos VAD (speech_start, speech_end) com indicador visual | |
| Ctrl+C para encerrar sessao gracefully (session.close) | |
| Dependencia `sounddevice` como optional (`pip install theo[stream]`) | |

**Entregaveis**:
- `src/theo/cli/stream.py` -- comando de streaming
- Alteracao em `src/theo/cli/transcribe.py` -- flag `--stream`
- `tests/unit/test_cli_stream.py` (testes com mock de audio input e WebSocket)

**DoD**:
- [ ] `theo transcribe --stream --model faster-whisper-tiny` conecta via WebSocket
- [ ] Audio do microfone capturado e enviado como frames PCM
- [ ] Partials exibidos na mesma linha (overwrite via `\r`)
- [ ] Finals exibidos em nova linha
- [ ] Ctrl+C envia `session.close` e encerra gracefully
- [ ] Sem microfone disponivel: mensagem de erro clara
- [ ] Testes: >=6 testes com mocks de audio input e WebSocket
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: `sounddevice` e a escolha certa -- e cross-platform (macOS, Linux, Windows via PortAudio), leve, e funciona em modo callback ou blocking. `pyaudio` e alternativa mas tem problemas de instalacao em macOS ARM. Declarar como optional dependency.

---

### M6-13: Metricas de M6

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-11
**Desbloqueia**: M6-14

**Contexto/Motivacao**: M5 adicionou metricas de TTFB, final_delay, active_sessions e vad_events. M6 adiciona metricas especificas do Session Manager: duracao de sessoes, force commits, e confidence media.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `theo_stt_session_duration_seconds` (Histogram): duracao total de sessoes encerradas | Metricas de M7+ (por engine, por architecture) |
| `theo_stt_segments_force_committed_total` (Counter): segmentos commitados por force commit (90% buffer) | |
| `theo_stt_confidence_avg` (Histogram): confidence media dos transcript.final | |
| `theo_stt_worker_recoveries_total` (Counter): recuperacoes de crash de worker | |
| Instrumentacao nos componentes: SessionStateMachine, RingBuffer, StreamingSession | |

**Entregaveis**:
- Alteracao em `src/theo/session/metrics.py` -- novas metricas
- Instrumentacao em componentes de M6
- `tests/unit/test_m6_metrics.py`

**DoD**:
- [ ] `session_duration_seconds` registrado ao fechar sessao (CLOSED)
- [ ] `segments_force_committed_total` incrementado a cada force commit
- [ ] `confidence_avg` registrado a cada `transcript.final` com confidence
- [ ] `worker_recoveries_total` incrementado a cada recovery bem-sucedido
- [ ] Metricas opcionais (try/except se prometheus_client nao instalado, padrao de M5)
- [ ] Testes: >=6 testes verificando instrumentacao correta
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-14: Testes Unitarios -- Review e Complemento

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: M (3-5 dias)
**Dependencias**: M6-01 a M6-13 (todos os componentes)
**Desbloqueia**: M6-15, M6-16

**Contexto/Motivacao**: Cada task de M6-01 a M6-13 cria testes junto com a implementacao. M6-14 e o momento de revisao: verificar cobertura, adicionar edge cases faltantes, garantir determinismo, e verificar que TODOS os testes de M1-M5 continuam passando.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Review de cobertura de todos os testes M6 | Testes de performance/load |
| Edge cases: transicoes rapidas (INIT -> ACTIVE -> SILENCE em 500ms), buffer cheio, WAL com dados corrompidos | Testes com modelo real |
| Verificar que mocks de worker sao determiniscos (sem timing real) | |
| Verificar que testes de M1-M5 (743 testes) continuam passando | |
| Testes de regressao para interface publica de `StreamingSession` | |

**Entregaveis**:
- Complementos nos arquivos de teste M6
- Revisao e ajuste de mocks para determinismo

**DoD**:
- [ ] Todos os testes passam: `pytest tests/unit/ -v` (M1-M6)
- [ ] Nenhum teste depende de timing real (clock injetavel em todos os componentes)
- [ ] Edge cases cobertos: transicao rapida, buffer cheio com force commit, recovery com buffer vazio, LocalAgreement com primeira pass, cross-segment com texto longo
- [ ] Total de testes novos M6: >=120
- [ ] Total acumulado (M1-M6): >=860
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M6-15: Teste de Estabilidade -- 30 Minutos

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: M (3-5 dias)
**Dependencias**: M6-14
**Desbloqueia**: M6-16

**Contexto/Motivacao**: O criterio de sucesso do M6 e uma sessao de 30 minutos sem degradacao de latencia ou memory leak. O Ring Buffer pre-alocado deveria prevenir memory leak, mas sessoes longas podem revelar vazamentos em asyncio tasks, gRPC channels, ou estado do LocalAgreement.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Teste de sessao de 30 minutos com audio simulado (mock worker) | Teste com modelo real carregado |
| Audio com multiplos segmentos de fala e silencio (simula conversa) | Teste com multiplas sessoes simultaneas (M9) |
| Monitorar RSS do processo a cada 10s -- falhar se crescimento > 5MB | |
| Verificar que TTFB nao degrada (final <= 1.2x inicial) | |
| Verificar transicoes de estado ao longo de 30 min (ACTIVE -> SILENCE -> HOLD -> ACTIVE) | |
| Verificar que force commit ocorre se segmento > 30s | |
| Marcado como `@pytest.mark.slow` | |

**Entregaveis**:
- `tests/integration/test_m6_stability.py`
- Fixtures de audio de 30 minutos (geradas programaticamente com intervalos de fala/silencio)

**DoD**:
- [ ] Sessao de 30 minutos completa sem erros
- [ ] Crescimento de memoria <= 5MB ao longo da sessao (Ring Buffer fixo deve prevenir leak)
- [ ] TTFB final <= 1.2x TTFB dos primeiros 5 minutos
- [ ] Pelo menos 1 transicao para HOLD durante o teste (periodo de silencio longo)
- [ ] Pelo menos 1 force commit durante o teste (segmento longo)
- [ ] Zero erros nao esperados nos logs
- [ ] Teste passa consistentemente (nao flaky -- usar clock controlavel)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: Este teste deve rodar no CI como job separado (schedule noturno ou pre-release, nao em cada push). Marcado como `@pytest.mark.slow`. Tempo de execucao real do teste: ~2-3 minutos (audio simulado em velocidade acelerada com clock controlavel), nao 30 minutos reais.

---

### M6-16: Teste de Recovery e Documentacao

**Epic**: E4 -- Integracao, CLI, Metricas e Estabilidade
**Estimativa**: S (1-2 dias)
**Dependencias**: M6-15
**Desbloqueia**: Nenhuma (leaf task, ultima do milestone)

**Contexto/Motivacao**: Teste dedicado ao cenario de recovery: matar worker durante sessao ACTIVE e verificar retomada sem duplicacao. Tambem inclui atualizacao de documentacao (CHANGELOG, ROADMAP, CLAUDE.md).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Teste de recovery: crash de worker durante speech -> recovery -> retomada | Teste de crash do runtime inteiro |
| Verificar: segment_id nao duplica, dados do Ring Buffer reprocessados | |
| Verificar: transcript.final pos-recovery tem conteudo correto | |
| Verificar: recovery com Ring Buffer vazio (crash logo apos commit) | |
| Atualizar CHANGELOG.md com entradas M6 | |
| Atualizar ROADMAP.md com resultado M6 | |
| Atualizar CLAUDE.md com novos componentes M6 | |

**Entregaveis**:
- `tests/integration/test_m6_recovery.py`
- Atualizacoes em `CHANGELOG.md`, `docs/ROADMAP.md`, `CLAUDE.md`

**DoD**:
- [ ] Worker crash durante ACTIVE -> recovery -> sessao retomada sem duplicacao de segment_id
- [ ] Dados do Ring Buffer apos fence reenviados ao worker novo
- [ ] Recovery com buffer vazio (crash imediato apos commit) funciona sem erro
- [ ] Recovery falha gracefully se worker nao reinicia em 10s (CLOSED com erro irrecuperavel)
- [ ] CHANGELOG.md atualizado com entradas M6
- [ ] ROADMAP.md atualizado com resultado e checkpoint M6
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

## 6. Grafo de Dependencias

```
M6-01 (StateMachine Core)
  |
  +---> M6-02 (Timeouts Configuraveis)
  |       |
  +-------+--> M6-03 (Integracao SM + StreamingSession) --+
                                                            |
M6-04 (Ring Buffer Core)                                   |
  |                                                         |
  +---> M6-05 (Read Fence + Force Commit) <----- M6-03 ---+
  |       |                                        |
  |       +---> M6-06 (WAL In-Memory)             |
  |       |       |                                |
  |       |       +---> M6-07 (Recovery)          |
  |       |                                        |
  |       +---> M6-08 (LocalAgreement) <----------+
  |               |
  |               +---> M6-09 (Cross-Segment Context)
  |                                        |
  +----------------------------------------+
                                           |
M6-10 (Hot Words Session) <--- M6-03 -----+
  |                                        |
  +----------------------------------------+
                                           |
                                           v
                                    M6-11 (Integracao E2E)
                                      |
                                      +---> M6-12 (CLI --stream)
                                      +---> M6-13 (Metricas)
                                      |       |
                                      +-------+--> M6-14 (Testes Review)
                                                      |
                                                      +---> M6-15 (Estabilidade 30min)
                                                              |
                                                              +---> M6-16 (Recovery Test + Docs)
```

### Caminho critico

```
M6-01 -> M6-03 -> M6-05 -> M6-08 -> M6-11 -> M6-14 -> M6-15 -> M6-16
```

### Paralelismo maximo

- **Sprint 1**: M6-01 (state machine) e M6-04 (ring buffer) em paralelo
- **Sprint 2**: M6-06 (WAL) e M6-08 (LocalAgreement) em paralelo apos M6-05
- **Sprint 3**: M6-12 (CLI), M6-13 (metricas) e M6-10 (hot words) em paralelo apos M6-11

---

## 7. Sprint Plan

### Sprint 1 (Semanas 1-2): State Machine + Ring Buffer

**Objetivo**: State machine integrada e Ring Buffer funcional.

**Demo Goal**: Sessao WebSocket com transicoes de estado observaveis (INIT -> ACTIVE -> SILENCE -> HOLD -> ACTIVE) e `session.hold` emitido.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M6-01 | SessionStateMachine core | M | Sofia |
| M6-02 | Timeouts configuraveis | S | Sofia |
| M6-03 | Integracao SM + StreamingSession | L | Sofia + Viktor |
| M6-04 | Ring Buffer core | M | Viktor |

**Checkpoint Sprint 1**:
- State machine funciona com todas as transicoes
- Ring Buffer le/escreve com wrap-around
- StreamingSession usa state machine em vez de ACTIVE/CLOSED
- `session.hold` emitido apos silencio prolongado
- Testes de M5 continuam passando

### Sprint 2 (Semanas 3-4): Ring Buffer Avancado + LocalAgreement

**Objetivo**: Read fence, WAL, recovery, e LocalAgreement parcialmente funcional.

**Demo Goal**: Recovery apos crash de worker (simulado). Partials inteligentes via LocalAgreement.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M6-05 | Read Fence + Force Commit | M | Viktor |
| M6-06 | WAL In-Memory | S | Viktor |
| M6-07 | Recovery de Crash | L | Viktor + Sofia |
| M6-08 | LocalAgreement | L | Sofia |

**Checkpoint Sprint 2**:
- Ring Buffer com read fence protege dados nao commitados
- Force commit disparado em 90% de capacidade
- WAL registra checkpoints apos cada transcript.final
- Recovery apos crash de worker funciona (dados reprocessados, sem duplicacao)
- LocalAgreement emite partials confirmados

### Sprint 3 (Semanas 5-6): Integracao + CLI + Estabilidade

**Objetivo**: Tudo integrado, CLI streaming, testes de estabilidade e recovery.

**Demo Goal**: Demo completo conforme criterio de sucesso de M6 (secao 1). Sessao de 30 minutos sem degradacao.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M6-09 | Cross-Segment Context | S | Sofia |
| M6-10 | Hot Words por Sessao | S | Sofia |
| M6-11 | Integracao End-to-End | L | Sofia + Viktor |
| M6-12 | CLI --stream | M | Andre |
| M6-13 | Metricas M6 | S | Andre |
| M6-14 | Testes Review | M | Todos |
| M6-15 | Estabilidade 30min | M | Viktor + Andre |
| M6-16 | Recovery Test + Docs | S | Andre |

**Checkpoint Sprint 3**:
- Fluxo completo funciona end-to-end
- `theo transcribe --stream` captura e transcreve audio do microfone
- Sessao de 30 minutos sem degradacao
- Recovery funciona end-to-end
- Metricas expostas no /metrics
- CHANGELOG e ROADMAP atualizados
- >=860 testes passando

---

## 8. Estrutura de Arquivos (M6)

```
src/theo/
  session/                              # EVOLUIDO -- novos componentes
    __init__.py
    streaming.py                        # StreamingSession (EVOLUIDO)    [M6-03, M6-05, M6-07, M6-08]
    state_machine.py                    # SessionStateMachine (NOVO)     [M6-01, M6-02]
    ring_buffer.py                      # RingBuffer (NOVO)              [M6-04, M6-05]
    wal.py                              # SessionWAL (NOVO)              [M6-06]
    local_agreement.py                  # LocalAgreementPolicy (NOVO)    [M6-08]
    cross_segment.py                    # CrossSegmentContext (NOVO)     [M6-09]
    backpressure.py                     # BackpressureController         (sem mudanca)
    metrics.py                          # Metricas (EVOLUIDO)            [M6-13]

  cli/
    stream.py                           # CLI streaming (NOVO)           [M6-12]
    transcribe.py                       # Flag --stream (ALTERADO)       [M6-12]

  server/
    routes/
      realtime.py                       # Factory atualizada (ALTERADO)  [M6-11]

  exceptions.py                         # InvalidTransitionError (NOVO)  [M6-01]

tests/
  unit/
    test_session_state_machine.py       # [M6-01]  NOVO
    test_session_timeouts.py            # [M6-02]  NOVO
    test_session_state_integration.py   # [M6-03]  NOVO
    test_ring_buffer.py                 # [M6-04]  NOVO
    test_ring_buffer_fence.py           # [M6-05]  NOVO
    test_session_wal.py                 # [M6-06]  NOVO
    test_session_recovery.py            # [M6-07]  NOVO
    test_local_agreement.py             # [M6-08]  NOVO
    test_cross_segment_context.py       # [M6-09]  NOVO
    test_hot_words_session.py           # [M6-10]  NOVO
    test_m6_metrics.py                  # [M6-13]  NOVO
    test_cli_stream.py                  # [M6-12]  NOVO
  integration/
    test_m6_e2e.py                      # [M6-11]  NOVO
    test_m6_stability.py                # [M6-15]  NOVO
    test_m6_recovery.py                 # [M6-16]  NOVO
```

---

## 9. Checkpoints de Validacao

### Checkpoint 1 (apos Sprint 1): State Machine Funcional

```
Validacao:
  [ ] State machine com 6 estados e transicoes validas
  [ ] Timeouts configuraveis por estado
  [ ] StreamingSession integrada com state machine
  [ ] session.hold emitido ao transitar para HOLD
  [ ] Ring Buffer pre-alocado com read/write
  [ ] Testes M5 continuam passando (regressao zero)
```

### Checkpoint 2 (apos Sprint 2): Recovery e LocalAgreement

```
Validacao:
  [ ] Read fence protege dados nao commitados
  [ ] Force commit em 90% de capacidade do buffer
  [ ] WAL registra checkpoints
  [ ] Recovery de crash de worker funciona sem duplicacao
  [ ] LocalAgreement emite partials confirmados
  [ ] Custo de inferencia: ~1 por window (nao 2/s)
```

### Checkpoint 3 (apos Sprint 3): Milestone Completo

```
Validacao:
  [ ] Fluxo completo end-to-end com todos os componentes
  [ ] theo transcribe --stream funciona com microfone
  [ ] Sessao de 30 minutos sem degradacao
  [ ] Recovery end-to-end funciona
  [ ] >=860 testes passando
  [ ] Metricas M6 no /metrics
  [ ] CHANGELOG, ROADMAP, CLAUDE.md atualizados
```

---

## 10. Criterios de Saida do M6

### Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | State machine com 6 estados e transicoes validas | Testes unitarios |
| 2 | Timeouts configuraveis por estado via session.configure | Teste de integracao |
| 3 | Ring Buffer pre-alocado 60s com zero allocations | Teste com tracemalloc |
| 4 | Read fence protege dados nao commitados | Teste unitario |
| 5 | Force commit em 90% de capacidade | Teste unitario |
| 6 | WAL in-memory registra checkpoints | Teste unitario |
| 7 | Recovery de crash sem duplicacao de segmentos | Teste de integracao |
| 8 | LocalAgreement emite partials confirmados | Teste unitario |
| 9 | Cross-segment context via initial_prompt | Teste unitario |
| 10 | Hot words por sessao via session.configure | Teste unitario |
| 11 | session.hold emitido ao transitar para HOLD | Teste de integracao |
| 12 | CLI theo transcribe --stream funciona | Teste com mock |
| 13 | Sessao de 30 minutos sem degradacao | Teste de estabilidade |
| 14 | TTFB <=300ms (target) | Metrica Prometheus |
| 15 | Metricas M6 expostas no /metrics | Teste de integracao |

### Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `make check` |
| 2 | ruff check sem warnings | `make check` |
| 3 | ruff format sem diffs | `make check` |
| 4 | Todos os testes passam (M1-M6) | `make test-unit` |
| 5 | CI verde | GitHub Actions |

### Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | State machine (estados, transicoes, timeouts) | 20 |
| Unit | Timeouts configuraveis | 8 |
| Unit | State machine + StreamingSession integracao | 15 |
| Unit | Ring Buffer (write, read, wrap-around) | 15 |
| Unit | Read fence + force commit | 12 |
| Unit | WAL (checkpoint, propriedades) | 8 |
| Unit | Recovery (crash, retomada, timeout) | 10 |
| Unit | LocalAgreement (confirmacao, retencao, flush) | 15 |
| Unit | Cross-segment context | 8 |
| Unit | Hot words por sessao | 6 |
| Unit | Metricas M6 | 6 |
| Unit | CLI --stream | 6 |
| Integration | End-to-end | 12 |
| Integration | Estabilidade 30min | 1 |
| Integration | Recovery | 3 |
| **Total novos M6** | | **>=120** |
| **Total acumulado (M1-M6)** | | **>=860** |

---

## 11. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | LocalAgreement produz partials de baixa qualidade para pt-BR (otimizado para en) | Media | Medio | Testar com corpus pt-BR. Parametro `min_confirm_passes` ajustavel. Fallback: `partial_strategy: disabled`. Monitorar confidence nos partials vs finals. |
| R2 | Race condition no WAL durante crash recovery | Media | Alto | WAL e escrita atomica (um registro por vez, single-threaded). Clock injetavel. Testar com chaos engineering (mock de crash em momentos aleatorios). |
| R3 | Ring Buffer force commit interrompe fala no meio de frase | Baixa | Medio | 60s de buffer e suficiente para 99%+ dos segmentos. Monitorar `segments_force_committed_total`. Se valor alto, aumentar buffer ou ajustar max_segment_duration. |
| R4 | State machine com transicoes invalidas em edge cases (VAD emite speech_start duplo, etc.) | Media | Alto | Testar TODAS as transicoes possiveis (validas e invalidas). Estado CLOSED e terminal e imutavel. Guard contra duplo speech_start ja implementado em M5 (`_cleanup_current_stream`). |
| R5 | Memory leak no LocalAgreement (acumula tokens/windows indefinidamente) | Media | Alto | LocalAgreement limpa estado a cada flush (speech_end). Window size configuravel. Teste de estabilidade de 30 minutos monitora RSS. |
| R6 | Recovery de crash leva mais que 10s (worker pesado, GPU allocation lenta) | Media | Medio | Timeout de recovery configuravel (default 10s). Se timeout, sessao fecha com erro irrecuperavel. WorkerManager ja tem health probe com backoff (M2). |
| R7 | Complexidade do Session Manager cresce alem do gerenciavel | Media | Alto | Decomposicao em sub-componentes (StateMachine, RingBuffer, WAL, LocalAgreement). Cada um testavel isoladamente. StreamingSession e orquestrador, nao God class. |
| R8 | `sounddevice` (CLI --stream) nao funciona em CI (sem device de audio) | Alta | Baixo | Testes de CLI usam mock de audio input. `sounddevice` como optional dependency. CI nao testa captura real de microfone. |

---

## 12. Out of Scope (explicitamente NAO esta em M6)

| Item | Milestone | Justificativa |
|------|-----------|---------------|
| Segundo backend STT (WeNet) | M7 | Validacao model-agnostic apos Session Manager completo |
| Pipeline adaptativo por arquitetura (CTC vs encoder-decoder) | M7 | Depende de WeNet para validar |
| Hot Word Correction post-processing (Levenshtein) | Futuro | Domain-specific, sem caso concreto |
| Entity Formatting (CPF, CNPJ) | Futuro | Domain-specific, sem caso concreto |
| RTP Listener | M8 | Telefonia e Fase 3 |
| Denoise stage | M8 | Necessario para telefonia |
| Dynamic batching no worker | M9 | Otimizacao de throughput |
| Priorizacao no Scheduler (realtime > batch) | M9 | Requer multiplos tipos de input |
| Co-scheduling STT + TTS | M10 | Full-duplex |
| Persistencia de WAL em disco | Futuro | WAL in-memory e suficiente para recovery de worker |
| Recovery de crash do runtime inteiro | Futuro | Apenas worker crash e coberto |

---

## 13. Transicao M6 -> M7

Ao completar M6, o time deve ter:

1. **Session Manager completo** -- 6 estados, timeouts, transicoes validadas, recovery funcional. Componente mais original do Theo, sem equivalente open-source.

2. **Ring Buffer estavel** -- Pre-alocado, read fence, force commit. Zero allocations durante streaming. Provado por 30 minutos de estabilidade.

3. **LocalAgreement funcional** -- Partials inteligentes para Whisper. Confirmacao de tokens entre passes. Desabilitavel.

4. **Pontos de extensao para M7**:
   - **Pipeline adaptativo**: `StreamingSession` verifica `architecture` do manifesto. Para `encoder-decoder` (Whisper): usa LocalAgreement. Para `ctc` (WeNet): usa partials nativos (nao chama LocalAgreement). O `if architecture` deve estar em um unico ponto (nao espalhado).
   - **WeNetBackend**: Implementa `STTBackend` com `transcribe_stream()` que retorna partials nativos. O `SessionStateMachine`, `RingBuffer` e `WAL` sao reutilizados sem mudanca.
   - **Hot words por engine**: Whisper usa initial_prompt, WeNet usa keyword boosting nativo. A escolha e feita no `StreamingSession` baseada na capability da engine.

5. **Pontos de extensao para M8**:
   - **RTP Listener**: Cria sessao via `StreamingSession` (mesmo contrato que WebSocket). Ring Buffer e Session Manager sao reutilizados.
   - **Denoise**: Adicionado como stage no preprocessing pipeline. Ativado por default para audio RTP.

**O primeiro commit de M7 sera**: implementar `WeNetBackend` com `transcribe_stream()` e validar que o pipeline adaptativo trata `architecture: ctc` sem LocalAgreement.

---

## 14. Demo Goal Final -- Validacao Completa do M6

Este e o roteiro completo de demonstracao que prova que o M6 esta 100% entregue. Cada passo e verificavel e reproduzivel. A demo deve ser executavel por qualquer membro do time sem contexto previo.

### Pre-requisitos

```bash
# 1. Build e testes passando
make ci
# -> format OK, lint OK, typecheck OK, >=860 testes passando

# 2. Servidor rodando (com modelo tiny para demo)
theo serve &
# -> API Server rodando em http://localhost:8000
# -> Worker Faster-Whisper carregado
```

### Demo 1: State Machine -- Transicoes de Estado

**Objetivo**: Provar que a maquina de 6 estados funciona com transicoes corretas e timeouts.

```bash
# Conectar via WebSocket
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# 1. INIT: sessao criada, aguardando audio
# <- {"type": "session.created", "session_id": "sess_...", ...}
# Estado: INIT

# 2. INIT -> ACTIVE: enviar audio com fala
# (enviar frames PCM binarios com fala)
# <- {"type": "vad.speech_start", "timestamp_ms": ...}
# Estado: ACTIVE

# 3. ACTIVE -> SILENCE: parar de enviar audio com fala
# <- {"type": "transcript.final", "text": "...", ...}
# <- {"type": "vad.speech_end", "timestamp_ms": ...}
# Estado: SILENCE

# 4. SILENCE -> HOLD: aguardar 30s sem fala
# <- {"type": "session.hold", "timestamp_ms": ..., "hold_timeout_ms": 300000}
# Estado: HOLD

# 5. HOLD -> ACTIVE: enviar audio com fala novamente
# <- {"type": "vad.speech_start", "timestamp_ms": ...}
# Estado: ACTIVE (sessao retomou)

# 6. Fechar sessao
# -> {"type": "session.close"}
# <- {"type": "session.closed", "reason": "client_request", "total_duration_ms": ..., "segments_transcribed": ...}
# Estado: CLOSED
```

**Verificacao**: Todos os 6 estados foram visitados. Transicoes na ordem correta. `session.hold` emitido com `hold_timeout_ms`.

### Demo 2: Partial Transcripts via LocalAgreement

**Objetivo**: Provar que partials inteligentes funcionam -- tokens confirmados entre passes consecutivas.

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# Configurar partials habilitados (default)
# -> {"type": "session.configure", "enable_partial_transcripts": true}

# Enviar audio com fala continua (~5 segundos de "Olá, como posso ajudar você hoje?")
# (enviar frames PCM binarios)

# Partials confirmados via LocalAgreement (tokens que concordam entre passes):
# <- {"type": "vad.speech_start", ...}
# <- {"type": "transcript.partial", "text": "Ola", "segment_id": 0, ...}
# <- {"type": "transcript.partial", "text": "Ola como", "segment_id": 0, ...}
# <- {"type": "transcript.partial", "text": "Ola como posso", "segment_id": 0, ...}

# Apos silencio, final com post-processing (ITN):
# <- {"type": "transcript.final", "text": "Ola, como posso ajudar voce hoje?", "segment_id": 0, "confidence": 0.95, ...}
# <- {"type": "vad.speech_end", ...}
```

**Verificacao**: Partials sao incrementais (tokens confirmados crescem). Final inclui texto completo com confidence. Partials NAO tem ITN, final TEM ITN.

### Demo 3: Cross-Segment Context

**Objetivo**: Provar que o contexto do segmento anterior e enviado como initial_prompt ao proximo segmento.

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# Segmento 1: "O valor do PIX e de dois mil reais"
# (enviar audio -> silencio -> final)
# <- {"type": "transcript.final", "text": "O valor do PIX e de R$2.000,00", "segment_id": 0, ...}

# Segmento 2: "E o TED de quinhentos" (frase curta que depende do contexto anterior)
# (enviar audio -> silencio -> final)
# <- {"type": "transcript.final", "text": "E o TED de R$500,00", "segment_id": 1, ...}
```

**Verificacao**: O segmento 2 transcreve corretamente porque recebe contexto do segmento 1 via initial_prompt. Sem context, "quinhentos" poderia ser transcrito sem formato monetario.

### Demo 4: Hot Words por Sessao

**Objetivo**: Provar que hot words configurados via session.configure melhoram a transcricao de termos de dominio.

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# Configurar hot words
# -> {"type": "session.configure", "hot_words": ["PIX", "TED", "Selic", "CDI", "IPCA"]}

# Enviar audio com termos de dominio
# (enviar audio: "Qual a taxa Selic atual?")
# <- {"type": "transcript.final", "text": "Qual a taxa Selic atual?", ...}
```

**Verificacao**: Termos de dominio (PIX, Selic, etc.) transcritos corretamente. Hot words enviados ao worker como parte do initial_prompt.

### Demo 5: Force Commit do Ring Buffer

**Objetivo**: Provar que o ring buffer forca commit quando atinge 90% de capacidade.

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# Enviar audio continuo SEM silencio por >50 segundos (90% de 60s)
# O VAD detecta fala continua, nao emite speech_end
# Ring Buffer atinge 90% -> force commit

# <- {"type": "vad.speech_start", ...}
# <- {"type": "transcript.partial", ...}  (varios)
# <- {"type": "transcript.final", ...}    (force commit -- segmento forçado)
# Audio continua sendo processado no proximo segmento

# Verificar metrica:
curl -s http://localhost:8000/metrics | grep theo_stt_segments_force_committed
# -> theo_stt_segments_force_committed_total 1
```

**Verificacao**: `transcript.final` emitido por force commit (sem speech_end). Metrica `segments_force_committed_total` incrementada. Audio continua sendo processado.

### Demo 6: Recovery apos Crash de Worker

**Objetivo**: Provar que apos crash do worker, a sessao retoma sem duplicacao de segmentos.

```bash
wscat -c "ws://localhost:8000/v1/realtime?model=faster-whisper-tiny"

# 1. Iniciar transcricao normal
# (enviar audio com fala)
# <- {"type": "transcript.final", "text": "Primeiro segmento", "segment_id": 0, ...}

# 2. Durante fala ativa, matar o worker
kill -9 $(pgrep -f "theo.workers.stt")

# 3. Runtime detecta crash e inicia recovery
# <- {"type": "error", "code": "worker_crash", "message": "Worker restarted, resuming from segment 1", "recoverable": true, "resume_segment_id": 1}

# 4. Worker reinicia automaticamente, sessao retoma
# (continuar enviando audio)
# <- {"type": "transcript.final", "text": "Segundo segmento apos recovery", "segment_id": 1, ...}

# Verificar: segment_id nao duplicou (0 -> 1, nao 0 -> 0)
# Verificar metrica:
curl -s http://localhost:8000/metrics | grep theo_stt_worker_recoveries
# -> theo_stt_worker_recoveries_total 1
```

**Verificacao**: `segment_id` nao duplica. Dados do Ring Buffer apos fence foram reprocessados. Sessao continua normalmente apos recovery. Metrica `worker_recoveries_total` incrementada.

### Demo 7: Sessao de 30 Minutos sem Degradacao

**Objetivo**: Provar estabilidade de longa duracao -- sem memory leak, sem degradacao de latencia.

```bash
# Executar teste de estabilidade automatizado
.venv/bin/python -m pytest tests/integration/test_m6_stability.py -v

# Ou manualmente:
# 1. Conectar e enviar audio simulado por 30 minutos
# 2. Monitorar RSS do processo:
while true; do ps -o rss= -p $(pgrep -f "theo serve"); sleep 10; done

# 3. Verificar que:
#    - Crescimento de memoria <= 5MB ao longo de 30 min
#    - TTFB no final <= 1.2x TTFB dos primeiros 5 min
#    - Pelo menos 1 transicao ACTIVE -> SILENCE -> HOLD -> ACTIVE
#    - Pelo menos 1 force commit
#    - Zero erros inesperados
```

**Verificacao**: Memoria estavel (Ring Buffer pre-alocado previne leak). Latencia estavel. Todas as transicoes de estado exercitadas.

### Demo 8: CLI -- `theo transcribe --stream`

**Objetivo**: Provar que o CLI captura audio do microfone e exibe transcricoes em tempo real.

```bash
# Iniciar streaming do microfone
theo transcribe --stream --model faster-whisper-tiny

# Terminal exibe:
# [VAD] Speech started
# > Ola como posso_                    (partial, overwrite na mesma linha)
# > Ola como posso ajudar_             (partial, overwrite)
# Ola, como posso ajudar?              (final, nova linha)
# [VAD] Speech ended
#
# [VAD] Speech started
# > O valor e de dois_                 (partial)
# O valor e de R$2.000,00              (final)
# [VAD] Speech ended

# Ctrl+C para encerrar
# -> Sessao encerrada. 2 segmentos transcritos em 15.3s.
```

**Verificacao**: Partials aparecem na mesma linha (overwrite). Finals aparecem em nova linha. Eventos VAD visiveis. Ctrl+C encerra gracefully.

### Demo 9: Metricas Prometheus

**Objetivo**: Provar que todas as metricas de M6 estao expostas e com valores corretos.

```bash
# Apos executar algumas sessoes (demos anteriores):
curl -s http://localhost:8000/metrics | grep theo_stt

# Metricas M5 (ja existentes):
# theo_stt_ttfb_seconds_bucket{...}
# theo_stt_final_delay_seconds_bucket{...}
# theo_stt_active_sessions
# theo_stt_vad_events_total{event="speech_start"}
# theo_stt_vad_events_total{event="speech_end"}

# Metricas M6 (novas):
# theo_stt_session_duration_seconds_bucket{...}      <- duracao de sessoes encerradas
# theo_stt_segments_force_committed_total             <- force commits por buffer cheio
# theo_stt_confidence_avg_bucket{...}                 <- confidence media dos finals
# theo_stt_worker_recoveries_total                    <- recoveries de crash
```

**Verificacao**: 4 metricas novas expostas. Valores consistentes com as demos executadas.

### Demo 10: Testes Automatizados

**Objetivo**: Provar que toda a suite de testes (M1-M6) passa sem erros.

```bash
# Suite completa
make ci
# -> ruff format: OK
# -> ruff check: OK
# -> mypy --strict: OK
# -> pytest: >=860 tests passed, 0 failed, 0 errors

# Detalhamento por area M6:
.venv/bin/python -m pytest tests/unit/test_session_state_machine.py -v      # >=20 testes
.venv/bin/python -m pytest tests/unit/test_ring_buffer.py -v                 # >=15 testes
.venv/bin/python -m pytest tests/unit/test_ring_buffer_fence.py -v           # >=12 testes
.venv/bin/python -m pytest tests/unit/test_session_wal.py -v                 # >=8 testes
.venv/bin/python -m pytest tests/unit/test_session_recovery.py -v            # >=10 testes
.venv/bin/python -m pytest tests/unit/test_local_agreement.py -v             # >=15 testes
.venv/bin/python -m pytest tests/unit/test_cross_segment_context.py -v       # >=8 testes
.venv/bin/python -m pytest tests/integration/test_m6_e2e.py -v               # >=12 testes
.venv/bin/python -m pytest tests/integration/test_m6_stability.py -v         # estabilidade
.venv/bin/python -m pytest tests/integration/test_m6_recovery.py -v          # recovery

# Regressao zero:
.venv/bin/python -m pytest tests/ -v --tb=short
# -> Todos os testes de M1-M5 (743) + M6 (>=120) passam
```

**Verificacao**: `make ci` verde. >=860 testes. Zero regressao em M1-M5. mypy strict e ruff limpos.

### Checklist Final da Demo

```
Demo Completa M6 -- Checklist de Validacao

Estado e Transicoes:
  [ ] State machine com 6 estados funcionando
  [ ] INIT -> ACTIVE apos primeiro frame com fala
  [ ] ACTIVE -> SILENCE apos VAD speech_end
  [ ] SILENCE -> HOLD apos 30s com session.hold emitido
  [ ] HOLD -> ACTIVE ao detectar fala (sessao retoma)
  [ ] CLOSING -> CLOSED apos flush
  [ ] Timeouts configuraveis via session.configure

Partial Transcripts:
  [ ] LocalAgreement emite partials incrementais (tokens confirmados)
  [ ] Partials sem ITN, finals com ITN
  [ ] partial_strategy: disabled funciona (sem partials)

Ring Buffer:
  [ ] Buffer pre-alocado 60s, zero allocations durante streaming
  [ ] Read fence protege dados nao commitados
  [ ] Force commit em 90% de capacidade
  [ ] Metrica segments_force_committed_total incrementada

Recovery:
  [ ] Crash de worker detectado via gRPC stream break
  [ ] Error event com recoverable: true emitido
  [ ] Sessao retomada sem duplicacao de segment_id
  [ ] Dados do Ring Buffer apos fence reprocessados
  [ ] Metrica worker_recoveries_total incrementada

Contexto e Hot Words:
  [ ] Cross-segment context enviado via initial_prompt
  [ ] Hot words configuraveis via session.configure
  [ ] Hot words combinados com cross-segment context

CLI:
  [ ] theo transcribe --stream captura audio do microfone
  [ ] Partials overwrite na mesma linha, finals em nova linha
  [ ] Ctrl+C encerra gracefully

Metricas:
  [ ] session_duration_seconds registrada
  [ ] segments_force_committed_total registrada
  [ ] confidence_avg registrada
  [ ] worker_recoveries_total registrada

Estabilidade:
  [ ] Sessao de 30 minutos sem degradacao
  [ ] Crescimento de memoria <= 5MB
  [ ] TTFB final <= 1.2x TTFB inicial

Qualidade de Codigo:
  [ ] make ci verde (format + lint + typecheck + testes)
  [ ] >=860 testes (743 M1-M5 + >=120 M6)
  [ ] Zero regressao em testes M1-M5
  [ ] CHANGELOG.md atualizado
  [ ] ROADMAP.md atualizado
```

---

*Documento gerado pelo Time de Arquitetura (ARCH) -- Sofia Castellani, Viktor Sorokin, Andre Oliveira. Sera atualizado conforme a implementacao do M6 avanca.*
