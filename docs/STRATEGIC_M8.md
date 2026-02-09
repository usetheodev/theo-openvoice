# M8 -- Scheduler Avancado -- Strategic Roadmap

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Planejado
**Data**: 2026-02-09

**Autores**:
- Sofia Castellani (Principal Solution Architect)
- Viktor Sorokin (Senior Real-Time Engineer)
- Andre Oliveira (Senior Platform Engineer)

**Dependencias**: M7 (segundo backend para testar scheduling multi-engine), M5 (WebSocket como input real-time)

---

## 1. Objetivo Estrategico

M8 evolui o Scheduler de um roteador trivial (round-robin, 1 worker = 1 request) para um componente inteligente que:
1. **Prioriza** requests de streaming (realtime via WebSocket) sobre batch (file upload).
2. **Cancela** requests em <=50ms via propagacao gRPC.
3. **Rastreia** orcamento de latencia por sessao (TTFB, final_delay).
4. **Agrupa** requests batch no worker para throughput maximo (dynamic batching).

### O que M8 resolve

O Scheduler atual (`src/theo/scheduler/scheduler.py`) e uma implementacao M3 -- funcional mas minimal:
- **Sem fila**: se o worker esta ocupado, a request falha com `WorkerUnavailableError`.
- **Sem prioridade**: batch e streaming sao tratados igualmente.
- **Sem cancelamento real**: a propagacao de cancel nao e implementada end-to-end.
- **Sem batching**: cada request abre um canal gRPC independente.
- **Sem tracking de latencia**: nao ha orcamento de latencia por sessao.

Em cenarios de producao com multiplas sessoes streaming e requests batch simultaneas, o scheduler atual nao garante que sessoes real-time mantenham latencia aceitavel enquanto batch requests sao processadas.

### O que M8 habilita

- **Qualidade de servico diferenciada**: sessoes streaming nunca esperam por batch.
- **Cancelamento rapido**: usuario pode cancelar transcrição batch antes de completar.
- **Dynamic batching**: worker processa multiplas requests batch em paralelo (2-3x throughput).
- **Observabilidade de scheduling**: metricas de profundidade de fila, inversoes de prioridade, latencia de cancelamento.
- **Base para M9**: co-scheduling STT+TTS requer scheduler com consciencia de tipo e prioridade.

### Criterio de sucesso (Demo Goal)

```
# Cenario: 2 sessoes streaming ativas + 5 requests batch simultaneas
# Resultado esperado:
#   - Sessoes streaming mantêm TTFB <=300ms
#   - Batch requests enfileiradas com prioridade menor
#   - Batch requests processadas em ordem de chegada quando ha workers livres
#   - Cancelamento de batch request completa em <50ms

# Teste de priorizacao:
# 1. Iniciar 2 sessoes WebSocket streaming
# 2. Enviar 5 requests batch POST /v1/audio/transcriptions simultaneas
# 3. Verificar que streaming TTFB nao degrada
# 4. Verificar que batch requests sao enfileiradas e processadas em FIFO

# Teste de cancelamento:
# 1. Enviar batch request com audio longo (30s)
# 2. Cancelar via gRPC Cancel antes de completar
# 3. Verificar que cancel propaga em <50ms (medido por metrica)

# Teste de dynamic batching:
# 1. Enviar 4 batch requests quase simultaneas (dentro de 50ms)
# 2. Worker processa como batch (1 chamada a engine, nao 4)
# 3. Throughput >= 2x comparado com processamento serial
```

### Conexao com o Roadmap

```
PRD Fase 3 (Escala + Full-Duplex)
  M8: Scheduler Avancado   [<-- ESTE MILESTONE]
  M9: Full-Duplex

Caminho critico: M7 (completo) -> M8 -> M9
```

M8 e o primeiro milestone da Fase 3 do PRD. O Scheduler avancado e pre-requisito para M9 (Full-Duplex), que precisa de co-scheduling STT+TTS com priorizacao.

---

## 2. Pre-Requisitos (de M1-M7)

### O que ja existe e sera REUTILIZADO sem mudancas

| Componente | Pacote | Uso em M8 |
|------------|--------|-----------|
| `WorkerManager` (spawn, health, restart) | `theo.workers.manager` | Gerencia workers. `get_ready_worker()` sera chamado pelo novo scheduler. |
| `WorkerHandle` + `WorkerState` | `theo.workers.manager` | Estado de workers. M8 consulta para decidir roteamento. |
| `StreamingGRPCClient` + `StreamHandle` | `theo.scheduler.streaming` | Comunicacao gRPC streaming. M8 nao altera streaming -- apenas batch. |
| `StreamingSession` | `theo.session.streaming` | Orquestrador de streaming. Ja usa `StreamingGRPCClient` diretamente, nao via Scheduler. |
| gRPC proto (`stt_worker.proto`) | `theo.proto` | `TranscribeFile`, `TranscribeStream`, `Cancel`, `Health` ja definidos. |
| `TranscribeRequest` | `theo.server.models.requests` | Request interna batch. M8 enfileira estas requests. |
| `BatchResult` | `theo._types` | Resultado batch. Nao muda. |
| Converters proto <-> dominio | `theo.scheduler.converters` | `build_proto_request` e `proto_response_to_batch_result` reutilizados. |
| REST endpoints | `theo.server.routes` | Endpoints nao mudam -- usam `Scheduler` via DI. |
| WebSocket endpoint | `theo.server.routes.realtime` | Nao passa pelo Scheduler -- usa `StreamingGRPCClient` diretamente. |
| Error handlers e exceptions | `theo.exceptions` | Reutilizados sem mudancas. |
| `ModelRegistry` | `theo.registry.registry` | Resolucao de modelo. Nao muda. |
| Audio Preprocessing/Post-Processing | `theo.preprocessing`, `theo.postprocessing` | Pipelines. Nao mudam. |
| Metricas Prometheus existentes | `theo.session.metrics` | Metricas M5/M6. M8 adiciona metricas novas de scheduler. |

### O que sera MODIFICADO

| Componente | Pacote | O que muda |
|------------|--------|------------|
| `Scheduler` | `theo.scheduler.scheduler` | Evolucao: fila com prioridade, cancelamento, timeout tracking, interface para batching. |
| `create_app()` | `theo.server.app` | Injetar novo Scheduler com dependencias adicionais (fila, metricas). |
| `get_scheduler()` dependency | `theo.server.dependencies` | Pode precisar de ajuste se assinatura do Scheduler mudar. |

### O que sera CRIADO

| Componente | Pacote | Descricao |
|------------|--------|-----------|
| `PriorityQueue` | `theo.scheduler.queue` | Fila com 2 niveis de prioridade (REALTIME, BATCH). |
| `RequestPriority` enum | `theo.scheduler.queue` | Enum com niveis de prioridade. |
| `ScheduledRequest` | `theo.scheduler.queue` | Wrapper com metadados: request, prioridade, timestamp, cancel event. |
| `CancellationManager` | `theo.scheduler.cancel` | Gerencia propagacao de cancelamento para requests em fila e em execucao. |
| `LatencyTracker` | `theo.scheduler.latency` | Rastreia orcamento de latencia por request (TTFB, queue wait, total). |
| `BatchAccumulator` | `theo.scheduler.batching` | Acumula requests por ate 50ms para batch inference no worker. |
| Metricas de scheduler | `theo.scheduler.metrics` | `scheduler_queue_depth`, `scheduler_wait_seconds`, `cancel_latency_seconds`, etc. |

---

## 3. Visao Geral da Arquitetura M8

### 3.1 Scheduler Atual vs. M8

**Atual (M3):**

```
Request -> Scheduler.transcribe()
             |
             v
           get_ready_worker() -> se None: WorkerUnavailableError
             |
             v
           gRPC TranscribeFile -> resposta
```

Problemas: sem fila, sem prioridade, sem cancelamento, sem batching. Uma request por vez.

**M8:**

```
Request -> Scheduler.submit(request, priority)
             |
             v
           PriorityQueue
             | REALTIME (streaming-originated batch, ex: force commit)
             | BATCH (file upload)
             |
             v
           Worker disponivel?
             | SIM -> BatchAccumulator -> acumula ate 50ms
             |           |
             |           v
             |         gRPC TranscribeFile (batch de N requests)
             |           |
             |           v
             |         Distribui respostas para cada request
             |
             | NAO -> enfileira, monitora aging
             |
           CancellationManager
             | cancel(request_id) -> remove da fila OU propaga gRPC Cancel
             |
           LatencyTracker
             | registra: queue_wait, processing_time, total_latency
```

### 3.2 Decomposicao de Componentes

```
+------------------------------------------------------------+
|                      Scheduler (evoluido)                    |
|                                                              |
|  +------------------+  +-------------------+                 |
|  | PriorityQueue    |  | CancellationMgr   |                |
|  |                  |  |                    |                |
|  | REALTIME queue   |  | pending: dict      |                |
|  | BATCH queue      |  | cancel(id) ->      |                |
|  | aging logic      |  |   remove from queue |                |
|  | submit/dequeue   |  |   OR gRPC Cancel   |                |
|  +--------+---------+  +---------+----------+                |
|           |                      |                           |
|  +--------v---------+  +--------v-----------+                |
|  | LatencyTracker   |  | BatchAccumulator   |                |
|  |                  |  |                    |                |
|  | per-request:     |  | accumulate_ms: 50  |                |
|  |   enqueue_time   |  | max_batch_size: 8  |                |
|  |   dequeue_time   |  | flush() -> batch   |                |
|  |   complete_time  |  | gRPC call          |                |
|  +------------------+  +--------------------+                |
|                                                              |
+------------------------------------------------------------+
              |                          |
              v                          v
     WorkerManager               gRPC Channel Pool
     (get_ready_worker)          (reutilizado, com pool)
```

### 3.3 Data Flow: Batch Request com Priorizacao

```
Cliente
  | POST /v1/audio/transcriptions
  v
API Server (FastAPI)
  | Preprocessing -> TranscribeRequest
  v
Scheduler.submit(request, priority=BATCH)
  | 1. Enfileira na PriorityQueue
  | 2. LatencyTracker.start(request_id)
  | 3. CancellationManager.register(request_id)
  v
Worker Loop (background task)
  | Dequeue por prioridade (REALTIME primeiro)
  | Se BATCH: passa para BatchAccumulator
  v
BatchAccumulator
  | Acumula requests por ate 50ms
  | Ou flush se max_batch_size atingido
  v
gRPC TranscribeFile (batch: N requests concatenadas)
  | Worker processa batch
  v
Distribui respostas para cada request
  | LatencyTracker.complete(request_id)
  | CancellationManager.unregister(request_id)
  v
Response -> API Server -> Cliente
```

### 3.4 Data Flow: Cancelamento

```
Cenario 1: Request NA FILA (ainda nao em execucao)
  | cancel(request_id)
  v
CancellationManager
  | Remove da PriorityQueue
  | Seta cancel event
  | Retorna imediato (<1ms)
  v
Caller recebe CancelledError


Cenario 2: Request EM EXECUCAO (worker processando)
  | cancel(request_id)
  v
CancellationManager
  | Seta cancel event
  | Envia gRPC Cancel(request_id) ao worker
  | Worker interrompe inference no proximo checkpoint
  v
Caller recebe CancelledError
  | Latencia total: <50ms (gRPC propagacao + worker response)
```

### 3.5 Data Flow: Streaming (NAO passa pelo novo Scheduler)

```
WebSocket frame
  |
  v
realtime.py -> StreamingSession -> StreamingGRPCClient -> Worker
  (fluxo M5/M6 inalterado)
```

**Decisao arquitetural**: O fluxo de streaming (WebSocket) NAO passa pela PriorityQueue do Scheduler. Streaming usa `StreamingGRPCClient` diretamente (ja implementado em M5). O Scheduler avancado gerencia apenas requests batch. A priorizacao de streaming acontece indiretamente: workers dedicados para streaming nao sao compartilhados com batch.

**Perspectiva Sofia**: Essa separacao e proposital. Misturar streaming e batch na mesma fila adicionaria latencia ao streaming (enqueue/dequeue overhead) sem beneficio. A prioridade de streaming e garantida por isolamento de recurso (worker dedicado), nao por posicao na fila. O Scheduler avancado otimiza BATCH -- que e o caso onde fila, batching e cancelamento fazem sentido.

**Perspectiva Viktor**: Concordo com a separacao. Streaming tem requirements de latencia incompativeis com fila (TTFB <=300ms). Qualquer overhead de enfileiramento e inaceitavel. A prioridade de streaming e "nao passar pela fila" -- o que e a priorizacao mais eficiente possivel.

---

## 4. Epics

### Epic 1: PriorityQueue + Scheduling Loop

Implementar fila com prioridade e o loop de consumo que despacha requests para workers. O Scheduler atual (`Scheduler.transcribe()`) e sincrono e bloqueante -- M8 o torna assincrono com fila.

**Racional**: E a fundacao do scheduler avancado. Sem fila, nao ha priorizacao, aging, nem cancelamento de requests pendentes.

**Responsavel principal**: Viktor (concorrencia, loop assincrono), Sofia (design da interface, prioridades)

**Tasks**: M8-01, M8-02

### Epic 2: Cancelamento End-to-End

Implementar propagacao de cancelamento para requests em fila e em execucao, com latencia <=50ms para requests em execucao via gRPC Cancel RPC.

**Racional**: Cancelamento e requisito do PRD (RF-07, <=50ms). Sem cancelamento real, recursos de GPU sao desperdicados em requests que o cliente ja abandonou. O `Cancel` RPC ja existe no proto mas nao esta implementado no servicer.

**Responsavel principal**: Viktor (gRPC, latencia), Andre (metricas de cancelamento)

**Tasks**: M8-03, M8-04

### Epic 3: Dynamic Batching no Worker

Implementar acumulacao de requests batch por ate 50ms no scheduler, envio como batch para o worker, e distribuicao de respostas individuais. O batching e transparente para o chamador.

**Racional**: Faster-Whisper suporta `BatchedInferencePipeline` que processa multiplos audios em paralelo com 2-3x de throughput. Sem batching, cada request e uma chamada gRPC separada -- ineficiente em GPU. O batching e feito no scheduler (accumula) e no worker (batch inference).

**Responsavel principal**: Viktor (acumulacao, timing), Sofia (protocolo batch no gRPC)

**Tasks**: M8-05, M8-06

### Epic 4: Latency Tracking, Metricas e Finalizacao

Implementar rastreamento de latencia por request, metricas Prometheus para o scheduler, e finalizacao do milestone (CHANGELOG, ROADMAP, docs).

**Racional**: Sem metricas, nao temos visibilidade sobre o comportamento do scheduler em producao. Latency tracking por request permite identificar gargalos (fila vs. worker vs. post-processing).

**Responsavel principal**: Andre (metricas, observabilidade), Sofia (latency budget)

**Tasks**: M8-07, M8-08, M8-09

---

## 5. Tasks (Detalhadas)

### M8-01: PriorityQueue com Dois Niveis (REALTIME, BATCH)

**Epic**: E1 -- PriorityQueue + Scheduling Loop
**Estimativa**: M (3-5 dias)
**Dependencias**: Nenhuma (componente isolado)
**Desbloqueia**: M8-02, M8-03, M8-05, M8-07

**Contexto/Motivacao**: O Scheduler atual nao tem fila -- se o worker esta ocupado, a request falha imediatamente com `WorkerUnavailableError`. M8-01 cria a fila com dois niveis de prioridade. Requests REALTIME (originadas de streaming, ex: force commit que precisa de batch) sao processadas antes de BATCH (file upload). Dentro de cada nivel, a ordem e FIFO. A fila e `asyncio.PriorityQueue` encapsulada com tipagem e aging logic.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `RequestPriority` enum: `REALTIME = 0`, `BATCH = 1` | Mais de 2 niveis de prioridade |
| `ScheduledRequest` dataclass: request + priority + timestamp + cancel event + response future | Rate limiting de clientes (futuro) |
| `PriorityQueue`: submit, dequeue, cancel, size, depth_by_priority | Fair-share scheduling entre clientes |
| Aging: requests BATCH na fila ha mais de N segundos ganham prioridade | Preemption de requests em execucao |
| Thread-safe via `asyncio.PriorityQueue` (single event loop) | |
| `submit()` retorna `asyncio.Future[BatchResult]` para o caller aguardar | |

**Entregaveis**:
- `src/theo/scheduler/queue.py` -- `RequestPriority`, `ScheduledRequest`, `PriorityQueue`
- `tests/unit/test_scheduler_queue.py`

**DoD**:
- [ ] `RequestPriority` enum com `REALTIME = 0`, `BATCH = 1` (menor = maior prioridade)
- [ ] `ScheduledRequest` e frozen dataclass com: `request`, `priority`, `enqueued_at` (monotonic), `cancel_event` (asyncio.Event), `result_future` (asyncio.Future)
- [ ] `PriorityQueue.submit(request, priority)` retorna `asyncio.Future[BatchResult]`
- [ ] `PriorityQueue.dequeue()` retorna `ScheduledRequest` com maior prioridade (menor valor numerico)
- [ ] Dentro de mesma prioridade, FIFO (por `enqueued_at`)
- [ ] `PriorityQueue.cancel(request_id)` seta `cancel_event`, remove da fila, seta `CancelledError` no future
- [ ] `PriorityQueue.depth` retorna total de items na fila
- [ ] `PriorityQueue.depth_by_priority` retorna dict com contagem por nivel
- [ ] Aging: requests BATCH com `enqueued_at` > `aging_threshold_s` (default: 30s) sao promovidas para REALTIME
- [ ] Testes: >=15 testes (submit, dequeue por prioridade, FIFO dentro de nivel, cancel, aging, empty queue, concurrent submit/dequeue, future resolution)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: A `PriorityQueue` e uma thin wrapper sobre `asyncio.PriorityQueue`. O `ScheduledRequest` implementa `__lt__` para comparacao por `(priority.value, enqueued_at)`, garantindo ordering correto. Manter simples -- nao over-engineer com multiplos niveis ou policies plugaveis. 2 niveis sao suficientes para M8; se precisarmos de mais, estendemos.

**Perspectiva Viktor**: Aging e essencial para evitar starvation de batch requests. Se streaming e continuo (call center 24h), batch requests podem esperar indefinidamente sem aging. O threshold de 30s e generoso -- em producao, ajustar via config. O aging funciona promovendo a request para REALTIME priority, nao criando um terceiro nivel.

---

### M8-02: Scheduling Loop Assincrono no Scheduler

**Epic**: E1 -- PriorityQueue + Scheduling Loop
**Estimativa**: L (5-7 dias)
**Dependencias**: M8-01
**Desbloqueia**: M8-03, M8-05, M8-07

**Contexto/Motivacao**: O `Scheduler.transcribe()` atual e sincrono: recebe request, encontra worker, faz gRPC call, retorna. M8-02 evolui para um modelo assincrono: `submit()` enfileira e retorna future; um background loop consome a fila, acha worker, executa, e resolve o future. Isso desacopla submissao de execucao e habilita priorizacao, cancelamento e batching.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Evoluir `Scheduler` para usar `PriorityQueue` internamente | Mudanca na assinatura publica de `Scheduler.transcribe()` |
| Background task que consome a fila e despacha para workers | Multiplos workers para mesmo modelo (pool de workers) -- simplificacao: 1 worker por modelo |
| `Scheduler.transcribe()` mantem assinatura existente (compat retroativa) | Streaming scheduling (permanece fora do Scheduler) |
| Internamente: `transcribe()` chama `submit(request, BATCH)` e `await future` | |
| Pool de canais gRPC reutilizaveis (em vez de criar/fechar canal por request) | |
| Graceful shutdown: aguarda requests em execucao ao parar | |

**Entregaveis**:
- Alteracao em `src/theo/scheduler/scheduler.py` -- loop assincrono, pool de canais
- `tests/unit/test_scheduler_async.py`

**DoD**:
- [ ] `Scheduler.transcribe(request)` mantem a mesma assinatura e comportamento externo
- [ ] Internamente, `transcribe()` faz `submit(request, BATCH)` + `await future`
- [ ] Background loop (`_dispatch_loop`) roda como `asyncio.Task` iniciada em `Scheduler.start()`
- [ ] `Scheduler.start()` inicia o dispatch loop
- [ ] `Scheduler.stop()` para o loop e aguarda requests em execucao (graceful shutdown, timeout 10s)
- [ ] Dispatch loop: `dequeue()` -> verifica cancel -> `get_ready_worker()` -> gRPC call -> resolve future
- [ ] Se `get_ready_worker()` retorna None: re-enfileira a request (com backoff curto de 100ms)
- [ ] Pool de canais gRPC: `dict[str, grpc.aio.Channel]` indexado por `worker_address`, reutilizado entre requests
- [ ] Canais criados sob demanda e fechados em `Scheduler.stop()`
- [ ] Se worker crashar durante execucao: rejeita future com `WorkerCrashError`
- [ ] Se timeout gRPC: rejeita future com `WorkerTimeoutError`
- [ ] Todos os testes existentes do Scheduler continuam passando (compatibilidade retroativa)
- [ ] Testes: >=20 testes (submit+await, priorizacao, FIFO, worker unavailable -> re-enqueue, worker crash, timeout, graceful shutdown, concurrent requests, pool de canais)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: A mudanca chave e que `Scheduler.transcribe()` continua sincrono do ponto de vista do caller (await retorna `BatchResult`), mas internamente o processamento e assincrono com fila. Isso mantem compatibilidade retroativa com todos os endpoints e testes existentes. O `create_app()` deve chamar `scheduler.start()` no startup e `scheduler.stop()` no shutdown (via FastAPI lifespan events).

**Perspectiva Andre**: O pool de canais gRPC e critico para performance. Criar/fechar canal por request adiciona ~10-50ms de overhead. Com pool, a conexao e reutilizada. O pool deve ser indexado por `worker_address` (host:port). Canais sao fechados quando o worker morre ou o scheduler para.

---

### M8-03: Cancelamento de Requests na Fila

**Epic**: E2 -- Cancelamento End-to-End
**Estimativa**: M (3-5 dias)
**Dependencias**: M8-01, M8-02
**Desbloqueia**: M8-04, M8-08

**Contexto/Motivacao**: Requests na fila (ainda nao em execucao) devem ser cancelaveis instantaneamente. O caller espera o future -- ao cancelar, o future e resolvido com `CancelledError` e a request removida da fila. Sem custo de GPU, sem delay.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `Scheduler.cancel(request_id)` -- cancela request na fila ou em execucao | Cancelamento de streaming (ja implementado em M5 via `StreamHandle.cancel()`) |
| `CancellationManager` para tracking de requests cancelaveis | Cancelamento parcial (cancelar metade de um batch) |
| Dispatch loop verifica `cancel_event` antes de despachar | |
| Se request ja foi dequeued mas cancel chega antes do gRPC call: cancel sem custo | |
| Expose `cancel()` via REST endpoint (DELETE ou POST) | |

**Entregaveis**:
- `src/theo/scheduler/cancel.py` -- `CancellationManager`
- Alteracao em `src/theo/scheduler/scheduler.py` -- integracao com `CancellationManager`
- Endpoint REST `POST /v1/audio/transcriptions/{request_id}/cancel` (ou header convention)
- `tests/unit/test_scheduler_cancel.py`

**DoD**:
- [ ] `CancellationManager.register(request_id, cancel_event, future)` registra request cancelavel
- [ ] `CancellationManager.cancel(request_id)` seta cancel_event e resolve future com `asyncio.CancelledError`
- [ ] `CancellationManager.unregister(request_id)` remove apos conclusao (sucesso ou erro)
- [ ] Dispatch loop: apos `dequeue()`, verifica `scheduled_request.cancel_event.is_set()` antes de gRPC call
- [ ] Se cancel_event setado: descarta request sem enviar ao worker (zero custo de GPU)
- [ ] `Scheduler.cancel(request_id)` delega para `CancellationManager.cancel()`
- [ ] Cancel de request inexistente ou ja completada e no-op (idempotente)
- [ ] Latencia de cancelamento na fila: <1ms (seta evento e resolve future)
- [ ] Testes: >=12 testes (cancel na fila, cancel de request inexistente, cancel apos conclusao, concurrent submit+cancel, cancel antes do dequeue, idempotencia)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O `cancel_event` e um `asyncio.Event`. O dispatch loop faz `if scheduled.cancel_event.is_set(): continue` apos dequeue -- overhead minimo. O ponto critico e que o `CancellationManager` e thread-safe via event loop unico (asyncio e single-threaded por design).

---

### M8-04: Cancelamento de Requests em Execucao (gRPC Cancel)

**Epic**: E2 -- Cancelamento End-to-End
**Estimativa**: M (3-5 dias)
**Dependencias**: M8-03
**Desbloqueia**: M8-08

**Contexto/Motivacao**: Requests que ja foram despachadas ao worker estao em execucao (inference GPU). Cancelar na fila e instantaneo, mas cancelar em execucao requer propagacao via gRPC `Cancel` RPC. O `Cancel` RPC ja existe no proto (`stt_worker.proto`) mas o `STTWorkerServicer` nao o implementa (retorna `UNIMPLEMENTED`). M8-04 implementa o Cancel no servicer e a propagacao no scheduler.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `STTWorkerServicer.Cancel()` implementado no worker | Cancelamento de streaming (ja implementado via stream break) |
| Worker seta flag de cancelamento, verificado entre chunks de inference | Preemption imediata de CUDA kernel (impossivel -- cancel e cooperative) |
| `CancellationManager` detecta request em execucao e propaga gRPC Cancel | |
| Timeout de propagacao: se worker nao confirma cancel em 100ms, desiste e espera conclusao | |
| Metrica `theo_scheduler_cancel_latency_seconds` para medir latencia end-to-end | |

**Entregaveis**:
- Alteracao em `src/theo/workers/stt/servicer.py` -- implementar `Cancel()` RPC
- Alteracao em `src/theo/workers/stt/faster_whisper.py` -- flag de cancelamento entre chunks (se aplicavel)
- Alteracao em `src/theo/scheduler/cancel.py` -- propagacao gRPC para requests em execucao
- Alteracao em `src/theo/scheduler/scheduler.py` -- tracking de request_id -> worker_address para cancel
- `tests/unit/test_cancel_grpc.py`

**DoD**:
- [ ] `STTWorkerServicer.Cancel(CancelRequest)` retorna `CancelResponse(acknowledged=True)` e seta flag interno
- [ ] Worker verifica flag de cancelamento entre segmentos de inference (cooperative cancel)
- [ ] Se cancelado, worker aborta inference restante e retorna resposta parcial (ou vazia)
- [ ] `CancellationManager` rastreia `request_id -> worker_address` para requests em execucao
- [ ] `cancel(request_id)` de request em execucao: envia gRPC `Cancel(request_id=...)` ao worker
- [ ] Timeout de propagacao: 100ms. Se worker nao responde, cancel e best-effort
- [ ] Latencia end-to-end de cancel (fila + propagacao): <=50ms para P95
- [ ] Metrica `theo_scheduler_cancel_latency_seconds` (Histogram)
- [ ] Testes: >=12 testes (cancel em execucao, worker confirma, worker timeout, cancel de request ja concluida, metrica de latencia, servicer Cancel RPC, flag de cancelamento no backend)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O cancelamento de inference GPU e cooperative: nao podemos interromper um CUDA kernel no meio. Whisper processa em segmentos de ~30s -- o checkpoint de cancelamento e entre segmentos. Para audios curtos (<30s), o cancel chega mas a inference ja terminou. Para audios longos (>30s), o cancel efetivamente economiza GPU. O target de <=50ms e para a propagacao do sinal, nao para a interrupcao da inference.

**Perspectiva Sofia**: O Cancel RPC recebe `request_id`. O worker precisa mapear `request_id` para a inference em andamento. Como o worker processa 1 request por vez (M8, sem multiplos workers por modelo), basta um flag `_cancel_requested` no servicer. Para dynamic batching (M8-06), o cancel precisa identificar QUAL request no batch cancelar -- isso e mais complexo e pode ser simplificado para "cancel do batch inteiro" na primeira versao.

---

### M8-05: BatchAccumulator -- Acumulacao de Requests para Batch Inference

**Epic**: E3 -- Dynamic Batching no Worker
**Estimativa**: L (5-7 dias)
**Dependencias**: M8-02
**Desbloqueia**: M8-06, M8-08

**Contexto/Motivacao**: O `BatchedInferencePipeline` do Faster-Whisper processa multiplos audios em paralelo com throughput 2-3x maior que serial. M8-05 cria o `BatchAccumulator` que acumula requests batch por ate 50ms (ou ate max_batch_size) e as despacha como grupo ao worker. O acumulador e transparente para o caller -- cada request recebe seu `BatchResult` individualmente.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `BatchAccumulator` que acumula requests por tempo (50ms) ou count (max_batch_size) | Batching no lado do worker (M8-06) |
| Integrado no dispatch loop do Scheduler | Batching de streaming requests |
| Flush timer: se 50ms passam sem atingir max_batch_size, flush com o que tem | Batching cross-model (requests para modelos diferentes) |
| Cada request no batch recebe seu `BatchResult` individual via future | |
| Requests no batch devem ser para o MESMO modelo (mesmo worker) | |

**Entregaveis**:
- `src/theo/scheduler/batching.py` -- `BatchAccumulator`
- Alteracao em `src/theo/scheduler/scheduler.py` -- integracao no dispatch loop
- `tests/unit/test_batch_accumulator.py`

**DoD**:
- [ ] `BatchAccumulator(accumulate_ms=50, max_batch_size=8)` configuravel
- [ ] `BatchAccumulator.add(scheduled_request)` adiciona request ao batch atual
- [ ] `BatchAccumulator.flush()` retorna lista de `ScheduledRequest` acumuladas e reseta
- [ ] Flush automatico apos `accumulate_ms` via asyncio timer
- [ ] Flush automatico se `max_batch_size` atingido
- [ ] Requests no batch devem ter o mesmo `model_name` (validacao)
- [ ] Se request cancelada chega no batch: removida antes do flush
- [ ] Dispatch loop: apos dequeue BATCH request, passa para `BatchAccumulator` em vez de enviar direto
- [ ] Requests REALTIME NAO passam pelo `BatchAccumulator` (enviadas direto ao worker)
- [ ] Testes: >=15 testes (acumulacao por tempo, por count, flush com canceladas, flush manual, mixed models rejeitados, timer reset, empty flush, concurrent add)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O `BatchAccumulator` e um componente do scheduler, NAO do worker. O scheduler acumula requests e as envia como grupo. O worker pode escolher processar serialmente ou em batch (M8-06). A separacao e intencional: o scheduler decide QUANDO agrupar; o worker decide COMO processar o grupo.

**Perspectiva Viktor**: O timer de 50ms e critico. Se o acumulador espera 50ms mas so 1 request chegou, ele faz flush de 1 request (sem batching real). O ganho so aparece sob carga. Em baixa carga, o overhead do timer e negligivel (50ms adicionados ao batch latency). Em alta carga, o timer e atingido antes dos 50ms porque `max_batch_size` completa. O `accumulate_ms` deve ser configuravel para tuning em producao.

---

### M8-06: Worker Batch Inference -- Suporte a Multiplas Requests por gRPC Call

**Epic**: E3 -- Dynamic Batching no Worker
**Estimativa**: L (5-7 dias)
**Dependencias**: M8-05
**Desbloqueia**: M8-08

**Contexto/Motivacao**: O `BatchAccumulator` agrupa requests no scheduler. Agora o worker precisa aceita-las como batch. Atualmente, `TranscribeFile` aceita 1 request. M8-06 evolui o proto e o servicer para aceitar batch requests e processar via `BatchedInferencePipeline` do Faster-Whisper (ou equivalente para outras engines).

Existem duas alternativas de design para o protocolo:

**Alternativa A**: Novo RPC `TranscribeBatch(BatchRequest) returns (BatchResponse)` dedicado para batch.
**Alternativa B**: Multiplas chamadas `TranscribeFile` em paralelo no mesmo canal gRPC.

**Perspectiva Sofia**: Alternativa B e mais simples e compativel com engines que nao suportam batch nativo. O scheduler ja tem o canal gRPC reutilizado (pool). Enviar N `TranscribeFile` em paralelo no mesmo canal e trivial com `asyncio.gather()`. O worker (Faster-Whisper) pode usar `BatchedInferencePipeline` internamente para processar N chamadas simultaneas se chegar no mesmo instante. A alternativa A exige mudanca no proto, no servicer, em todos os backends, e adiciona complexidade sem ganho proporcional.

**Escopo (Alternativa B)**:

| Incluido | Fora de escopo |
|----------|---------------|
| Scheduler envia N `TranscribeFile` em paralelo via `asyncio.gather()` | Novo RPC `TranscribeBatch` no proto |
| Worker com semaphore para limitar concorrencia interna | Mudanca no proto `stt_worker.proto` |
| `FasterWhisperBackend` com `BatchedInferencePipeline` (se engine suporta) | Batch inference para WeNet (futuro -- se WeNet suportar) |
| Manifesto `capabilities.batch_inference: true` controla se worker aceita parallelismo | |
| Distribuicao: cada chamada paralela resolve seu own future | |

**Entregaveis**:
- Alteracao em `src/theo/scheduler/scheduler.py` -- dispatch de batch via `asyncio.gather()`
- Alteracao em `src/theo/workers/stt/servicer.py` -- semaphore para concorrencia controlada
- Alteracao em `src/theo/workers/stt/faster_whisper.py` -- `BatchedInferencePipeline` (se disponivel)
- `tests/unit/test_batch_dispatch.py`

**DoD**:
- [ ] Quando `BatchAccumulator` flush retorna N requests, scheduler envia N `TranscribeFile` em paralelo
- [ ] `asyncio.gather(*[stub.TranscribeFile(req) for req in batch])` no mesmo canal gRPC
- [ ] Cada response resolve o future individual da request correspondente
- [ ] Worker servicer com `asyncio.Semaphore(max_concurrent)` para limitar concorrencia
- [ ] `max_concurrent` lido do manifesto `capabilities.max_concurrent_sessions` (default: 1)
- [ ] `FasterWhisperBackend`: se `batch_inference: true` no manifesto, usa `BatchedInferencePipeline`
- [ ] Se engine nao suporta batch: requests sao processadas serialmente (semaphore=1)
- [ ] Erro em 1 request do batch nao afeta as outras (isolamento)
- [ ] Testes: >=15 testes (batch dispatch, parallel execution, isolamento de erros, semaphore limit, engine sem batch suporte, gather com cancel, timeout por request)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: A Alternativa B e a abordagem mais KISS. Nao exige mudanca de proto (que impactaria todos os workers e CI). O `asyncio.gather()` no mesmo canal gRPC e eficiente -- HTTP/2 multiplexa as chamadas. O ganho de batching vem da GPU -- `BatchedInferencePipeline` do Faster-Whisper processa N audios em batch com custo marginal baixo.

---

### M8-07: LatencyTracker -- Orcamento de Latencia por Request

**Epic**: E4 -- Latency Tracking, Metricas e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M8-02
**Desbloqueia**: M8-08

**Contexto/Motivacao**: O PRD define targets de latencia (batch <=0.5x duracao, TTFB <=300ms). Sem tracking per-request, nao sabemos onde o tempo e gasto: na fila, no preprocessing, no worker, no post-processing. M8-07 cria o `LatencyTracker` que registra timestamps por fase e calcula latencia total.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `LatencyTracker` que registra: enqueue_time, dequeue_time, grpc_start, grpc_end, complete_time | Alerting automatico (futuro, via Prometheus alertmanager) |
| Calculo de: queue_wait, grpc_time, total_time | SLA enforcement (rejeitar requests que excedam budget) |
| Integracao no Scheduler: timestamps registrados em cada fase | |
| Metricas Prometheus: histogramas por fase | |

**Entregaveis**:
- `src/theo/scheduler/latency.py` -- `LatencyTracker`
- Alteracao em `src/theo/scheduler/scheduler.py` -- instrumentacao de timestamps
- `tests/unit/test_latency_tracker.py`

**DoD**:
- [ ] `LatencyTracker.start(request_id)` registra `enqueue_time = time.monotonic()`
- [ ] `LatencyTracker.dequeued(request_id)` registra `dequeue_time`
- [ ] `LatencyTracker.grpc_started(request_id)` registra `grpc_start_time`
- [ ] `LatencyTracker.complete(request_id)` registra `complete_time` e calcula metricas
- [ ] `queue_wait = dequeue_time - enqueue_time`
- [ ] `grpc_time = complete_time - grpc_start_time`
- [ ] `total_time = complete_time - enqueue_time`
- [ ] `LatencyTracker.get_summary(request_id)` retorna dict com todas as fases
- [ ] Cleanup automatico: entries removidas apos complete ou apos TTL (60s)
- [ ] Testes: >=10 testes (start, phases, complete, summary, cleanup, missing phases)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: O `LatencyTracker` usa `time.monotonic()` (como o WAL) para consistencia com ajustes de relogio. O cleanup com TTL previne memory leak em caso de requests que nunca completam (crash do worker sem cancel). O TTL de 60s e generoso -- requests batch raramente duram mais que isso.

---

### M8-08: Metricas Prometheus do Scheduler

**Epic**: E4 -- Latency Tracking, Metricas e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M8-03, M8-04, M8-05, M8-07
**Desbloqueia**: M8-09

**Contexto/Motivacao**: O roadmap define metricas especificas para M8: `scheduler_queue_depth`, `scheduler_priority_inversions_total`, `cancel_latency_seconds`. Alem dessas, metricas de latencia por fase (queue_wait, grpc_time) sao essenciais para observabilidade em producao.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `theo_scheduler_queue_depth` (Gauge): profundidade da fila por prioridade | Dashboard Grafana (futuro) |
| `theo_scheduler_queue_wait_seconds` (Histogram): tempo na fila | |
| `theo_scheduler_grpc_duration_seconds` (Histogram): tempo de gRPC call | |
| `theo_scheduler_cancel_latency_seconds` (Histogram): latencia de cancel | |
| `theo_scheduler_batch_size` (Histogram): tamanho dos batches despachados | |
| `theo_scheduler_requests_total` (Counter): requests por prioridade e status (ok/error/cancelled) | |
| `theo_scheduler_aging_promotions_total` (Counter): requests promovidas por aging | |
| Integracao com `LatencyTracker`: observe metricas em complete/cancel | |

**Entregaveis**:
- `src/theo/scheduler/metrics.py` -- definicao de metricas Prometheus
- Alteracao em `src/theo/scheduler/scheduler.py` -- instrumentacao
- Alteracao em `src/theo/scheduler/queue.py` -- instrumentacao de queue depth
- Alteracao em `src/theo/scheduler/cancel.py` -- instrumentacao de cancel latency
- Alteracao em `src/theo/scheduler/batching.py` -- instrumentacao de batch size
- `tests/unit/test_scheduler_metrics.py`

**DoD**:
- [ ] Metricas definidas com lazy import de `prometheus_client` (pattern de `theo.session.metrics`)
- [ ] `theo_scheduler_queue_depth` (Gauge, labels: `priority`) -- atualizado em submit/dequeue
- [ ] `theo_scheduler_queue_wait_seconds` (Histogram) -- observado no dequeue (via LatencyTracker)
- [ ] `theo_scheduler_grpc_duration_seconds` (Histogram) -- observado no complete
- [ ] `theo_scheduler_cancel_latency_seconds` (Histogram) -- observado no cancel
- [ ] `theo_scheduler_batch_size` (Histogram) -- observado no batch flush
- [ ] `theo_scheduler_requests_total` (Counter, labels: `priority`, `status`) -- incrementado em complete/error/cancel
- [ ] `theo_scheduler_aging_promotions_total` (Counter) -- incrementado quando aging promove request
- [ ] Se `prometheus_client` nao instalado: metricas sao no-op (mesmo pattern de M5/M6)
- [ ] Testes: >=10 testes (cada metrica incrementa/observa corretamente, no-op sem prometheus)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: Seguir exatamente o pattern de `theo.session.metrics`: lazy import, `HAS_METRICS` flag, no-op quando prometheus nao esta instalado. Labels devem ser finitas e controladas -- nunca usar request_id como label (cardinalidade infinita). Labels por prioridade (`realtime`, `batch`) e status (`ok`, `error`, `cancelled`) sao aceitaveis.

---

### M8-09: Finalizacao -- Testes de Integracao, CHANGELOG, ROADMAP

**Epic**: E4 -- Latency Tracking, Metricas e Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M8-08
**Desbloqueia**: Nenhuma (leaf task, ultima do milestone)

**Contexto/Motivacao**: Validacao final do milestone com testes de integracao que exercitam o scheduler end-to-end: priorizacao sob contencao, cancelamento, batching. Atualizacao de documentacao de projeto.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Teste de integracao: N requests batch + streaming simultaneos | Testes de load (stress test com centenas de requests) |
| Teste de integracao: cancelamento em fila e em execucao | Deploy em k8s (M9+ scope) |
| Teste de contencao: batch requests enfileiradas enquanto streaming ocupa workers | |
| Atualizar `CHANGELOG.md` com entradas M8 | |
| Atualizar `docs/ROADMAP.md` com resultado M8 | |
| Atualizar `CLAUDE.md` com novos componentes e padroes M8 | |
| Verificar `make ci` verde com todos os testes M1-M8 | |

**Entregaveis**:
- `tests/unit/test_scheduler_integration.py` -- testes de integracao (com mocks de worker)
- `tests/unit/test_scheduler_contention.py` -- testes de contencao e priorizacao
- Atualizacoes em `CHANGELOG.md`, `docs/ROADMAP.md`, `CLAUDE.md`

**DoD**:
- [ ] Teste de priorizacao: N requests BATCH + 1 REALTIME simultaneas. REALTIME processada primeiro.
- [ ] Teste de aging: request BATCH na fila > 30s e promovida e processada.
- [ ] Teste de cancelamento na fila: cancel resolve future com CancelledError, request nao chega ao worker.
- [ ] Teste de cancelamento em execucao: cancel propaga gRPC Cancel, worker confirma.
- [ ] Teste de batching: N requests BATCH acumuladas e despachadas como grupo.
- [ ] Teste de graceful shutdown: stop() aguarda requests em execucao.
- [ ] Teste de contencao: streaming (via StreamingGRPCClient) nao e afetado por fila batch.
- [ ] `CHANGELOG.md` com entradas M8 na secao `[Unreleased]`
- [ ] `ROADMAP.md` com resultado M8 e checkpoint Fase 3 parcial
- [ ] `CLAUDE.md` atualizado com componentes M8
- [ ] `make ci` verde
- [ ] Total de testes novos M8: >=110
- [ ] Total acumulado (M1-M8): >=1327
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

## 6. Grafo de Dependencias

```
M8-01 (PriorityQueue)
  |
  +---> M8-02 (Scheduling Loop) --------+
  |          |                           |
  |          +---> M8-03 (Cancel Fila) --+---> M8-04 (Cancel gRPC)
  |          |                           |          |
  |          +---> M8-05 (BatchAccum) ---+          |
  |          |          |                |          |
  |          |          +---> M8-06 (Worker Batch)  |
  |          |                           |          |
  |          +---> M8-07 (LatencyTrack)  |          |
  |                      |               |          |
  |                      +---------------+----------+
  |                                      |
  |                                      v
  |                               M8-08 (Metricas Prometheus)
  |                                      |
  |                                      v
  |                               M8-09 (Finalizacao)
  |
  (M8-01 desbloqueia tudo via M8-02)
```

### Caminho critico

```
M8-01 -> M8-02 -> M8-05 -> M8-06 -> M8-08 -> M8-09
```

### Paralelismo maximo

- **Sprint 1**: M8-01 (PriorityQueue) -- ponto de partida isolado.
- **Sprint 2**: M8-02 (Scheduling Loop) -- depende de M8-01. Ponto de convergencia.
- **Sprint 3**: M8-03 (Cancel Fila), M8-05 (BatchAccumulator), M8-07 (LatencyTracker) em paralelo -- todos dependem de M8-02.
- **Sprint 4**: M8-04 (Cancel gRPC), M8-06 (Worker Batch) -- dependem de Sprint 3.
- **Sprint 5**: M8-08 (Metricas), M8-09 (Finalizacao) -- sequenciais no final.

---

## 7. Estrutura de Arquivos (M8)

```
src/theo/
  scheduler/
    scheduler.py                        # Scheduler (EVOLUIDO)          [M8-02, M8-03, M8-05]
    queue.py                            # PriorityQueue                 (NOVO) [M8-01]
    cancel.py                           # CancellationManager           (NOVO) [M8-03, M8-04]
    batching.py                         # BatchAccumulator              (NOVO) [M8-05]
    latency.py                          # LatencyTracker                (NOVO) [M8-07]
    metrics.py                          # Metricas Prometheus scheduler (NOVO) [M8-08]
    streaming.py                        # StreamingGRPCClient           (SEM MUDANCA)
    converters.py                       # Proto <-> dominio             (SEM MUDANCA)

  workers/
    stt/
      servicer.py                       # STTWorkerServicer             (ALTERADO) [M8-04]
      faster_whisper.py                 # FasterWhisperBackend          (ALTERADO) [M8-06]
      wenet.py                          # WeNetBackend                  (SEM MUDANCA)
      main.py                           # Worker entry point            (SEM MUDANCA)
      interface.py                      # STTBackend ABC                (SEM MUDANCA)
    manager.py                          # WorkerManager                 (SEM MUDANCA)

  server/
    app.py                              # create_app                    (ALTERADO) [M8-02]
    dependencies.py                     # get_scheduler                 (SEM MUDANCA)
    routes/
      transcriptions.py                 # POST /v1/audio/transcriptions (SEM MUDANCA)
      translations.py                   # POST /v1/audio/translations   (SEM MUDANCA)
      realtime.py                       # WS /v1/realtime               (SEM MUDANCA)
      health.py                         # GET /health                   (SEM MUDANCA)

  session/
    streaming.py                        # StreamingSession              (SEM MUDANCA)
    (todos os demais)                   # (SEM MUDANCA)

tests/
  unit/
    test_scheduler_queue.py             # [M8-01]          NOVO
    test_scheduler_async.py             # [M8-02]          NOVO
    test_scheduler_cancel.py            # [M8-03]          NOVO
    test_cancel_grpc.py                 # [M8-04]          NOVO
    test_batch_accumulator.py           # [M8-05]          NOVO
    test_batch_dispatch.py              # [M8-06]          NOVO
    test_latency_tracker.py             # [M8-07]          NOVO
    test_scheduler_metrics.py           # [M8-08]          NOVO
    test_scheduler_integration.py       # [M8-09]          NOVO
    test_scheduler_contention.py        # [M8-09]          NOVO
```

### Contagem de arquivos impactados

| Tipo | Novos | Alterados | Inalterados |
|------|-------|-----------|-------------|
| Source | 5 (`queue.py`, `cancel.py`, `batching.py`, `latency.py`, `metrics.py`) | 4 (`scheduler.py`, `servicer.py`, `faster_whisper.py`, `app.py`) | 20+ |
| Tests | 10 arquivos | 0 | -- |
| Docs | 0 | 3 (`CHANGELOG.md`, `ROADMAP.md`, `CLAUDE.md`) | -- |

---

## 8. Criterios de Saida do M8

### Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `PriorityQueue` com 2 niveis (REALTIME, BATCH) funcional | Testes unitarios |
| 2 | FIFO dentro de cada nivel de prioridade | Testes unitarios |
| 3 | Aging de requests BATCH (>30s -> promovido para REALTIME) | Testes unitarios |
| 4 | Scheduling loop assincrono consome fila e despacha para workers | Testes unitarios |
| 5 | `Scheduler.transcribe()` mantem compatibilidade retroativa | Regressao testes M1-M7 |
| 6 | Cancelamento de request na fila: <1ms, future resolve com CancelledError | Testes unitarios |
| 7 | Cancelamento de request em execucao: propagacao gRPC Cancel, <=50ms P95 | Testes unitarios + metrica |
| 8 | `STTWorkerServicer.Cancel()` implementado e funcional | Testes unitarios |
| 9 | `BatchAccumulator` acumula por tempo (50ms) e count (max_batch_size) | Testes unitarios |
| 10 | Dispatch de batch via `asyncio.gather()` no mesmo canal gRPC | Testes unitarios |
| 11 | `LatencyTracker` registra timestamps por fase (queue, grpc, total) | Testes unitarios |
| 12 | Pool de canais gRPC reutilizados entre requests | Testes unitarios |
| 13 | Graceful shutdown: aguarda requests em execucao | Testes unitarios |
| 14 | Streaming (WebSocket) NAO e afetado pelo scheduler de batch | Teste de contencao |

### Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `make check` |
| 2 | ruff check sem warnings | `make check` |
| 3 | ruff format sem diffs | `make check` |
| 4 | Todos os testes passam (M1-M8) | `make test-unit` |
| 5 | CI verde | GitHub Actions |

### Metricas Prometheus (verificaveis)

| Metrica | Tipo | Labels |
|---------|------|--------|
| `theo_scheduler_queue_depth` | Gauge | `priority` |
| `theo_scheduler_queue_wait_seconds` | Histogram | -- |
| `theo_scheduler_grpc_duration_seconds` | Histogram | -- |
| `theo_scheduler_cancel_latency_seconds` | Histogram | -- |
| `theo_scheduler_batch_size` | Histogram | -- |
| `theo_scheduler_requests_total` | Counter | `priority`, `status` |
| `theo_scheduler_aging_promotions_total` | Counter | -- |

### Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | PriorityQueue | 15 |
| Unit | Scheduling Loop | 20 |
| Unit | Cancel na fila | 12 |
| Unit | Cancel gRPC | 12 |
| Unit | BatchAccumulator | 15 |
| Unit | Batch dispatch | 15 |
| Unit | LatencyTracker | 10 |
| Unit | Metricas | 10 |
| Unit | Integracao scheduler + contencao | 15 |
| **Total novos M8** | | **>=110** |
| **Total acumulado (M1-M8)** | | **>=1327** |

---

## 9. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Priorizacao starva requests batch indefinidamente sob carga continua de streaming | Media | Medio | Aging com threshold configuravel (default 30s). Monitorar `theo_scheduler_aging_promotions_total` -- valor alto indica threshold muito baixo ou carga excessiva. |
| R2 | Dynamic batching adiciona 50ms de latencia a TODA request batch (mesmo com 1 request) | Certa | Baixo | 50ms e aceitavel para batch (PRD: <=0.5x duracao). Se 1 request isolada, timer flush apos 50ms -- overhead minimo. Configuravel via `accumulate_ms`. |
| R3 | Cancelamento gRPC em <=50ms nao e garantido se worker esta no meio de inference GPU | Alta | Baixo | Cancel e cooperative, nao preemptive. 50ms e para propagacao do sinal, nao interrupcao de CUDA kernel. Para audios curtos, cancel chega tarde. Para longos (>30s), cancel economiza. Documentar como limitacao. |
| R4 | Pool de canais gRPC pode ter canais stale (worker reiniciou, canal antigo) | Media | Medio | Detectar canal stale via erro na primeira chamada -> remover do pool e reconectar. `grpc.StatusCode.UNAVAILABLE` e o sinal de canal stale. |
| R5 | `asyncio.gather()` para batch: se 1 request falha, exception pode afetar gather | Media | Alto | Usar `asyncio.gather(*calls, return_exceptions=True)`. Cada resultado e inspecionado individualmente. Exception em 1 nao afeta as outras. |
| R6 | Scheduler assincrono com background task pode ter race conditions | Media | Alto | Single event loop do asyncio garante ordering. Usar `asyncio.Lock` apenas se necessario (ex: acesso ao pool de canais). Testes com `asyncio.gather()` para simular concorrencia. |
| R7 | Mudanca no Scheduler quebra testes existentes (M1-M7) | Media | Alto | `Scheduler.transcribe()` mantem assinatura e semantica exatas. Internamente muda, externamente identico. Rodar `make test-unit` apos cada alteracao. |

---

## 10. Out of Scope (explicitamente NAO esta em M8)

| Item | Milestone | Justificativa |
|------|-----------|---------------|
| Co-scheduling STT + TTS | M9 | M8 otimiza scheduling de STT. TTS e M9 (Full-Duplex). |
| Mute-on-speak | M9 | Requer coordenacao STT+TTS que e escopo de M9. |
| Fair-share scheduling entre clientes | Futuro | Sem multi-tenancy em M8. Todos os requests sao tratados igualmente dentro de cada prioridade. |
| Rate limiting de clientes na API | Futuro | WebSocket ja tem backpressure (M5). Rate limiting HTTP e infra (nginx/envoy). |
| Auto-scaling de workers | Futuro | M8 trabalha com workers fixos. Scaling e decision de orquestrador (K8s HPA). |
| Streaming scheduling via PriorityQueue | Design decision | Streaming usa `StreamingGRPCClient` diretamente. Priorizacao por isolamento de recurso. |
| Preemption de requests (interromper batch para atender streaming) | Design decision | Complexidade nao justificada. Isolamento de workers resolve. |
| Multiplos workers para mesmo modelo (pool) | Futuro | M8 simplifica: 1 worker por modelo. Pool e evolucao natural mas nao necessaria para validacao. |

---

## 11. Decisoes Arquiteturais de M8

### ADR-M8-01: Streaming Nao Passa pelo Scheduler de Batch

**Decisao**: Requests de streaming (WebSocket) continuam usando `StreamingGRPCClient` diretamente, sem passar pela `PriorityQueue` do Scheduler.

**Justificativa**: Streaming tem requirements de latencia incompativeis com fila (TTFB <=300ms). Qualquer overhead de enqueue/dequeue e inaceitavel. A priorizacao de streaming e garantida por isolamento de recurso (workers dedicados), que e a priorizacao mais eficiente possivel -- nao esperar em fila nenhuma.

**Alternativa rejeitada**: Unificar batch e streaming na mesma fila com prioridade REALTIME para streaming. Rejeitada porque adiciona overhead desnecessario ao streaming e acopla dois fluxos com semanticas diferentes (request-response vs. long-lived stream).

**Consequencias**: O Scheduler avancado gerencia apenas batch. Metricas de scheduling nao incluem streaming (streaming tem suas proprias metricas em `theo.session.metrics`). Isso e aceitavel porque batch e streaming sao fluxos fundamentalmente diferentes.

### ADR-M8-02: Batch via asyncio.gather (Nao Novo RPC)

**Decisao**: Dynamic batching usa multiplas chamadas `TranscribeFile` em paralelo via `asyncio.gather()`, em vez de criar um novo RPC `TranscribeBatch` no proto.

**Justificativa**: Menor mudanca no proto e nos workers. HTTP/2 multiplexa as chamadas eficientemente. O worker pode otimizar internamente com `BatchedInferencePipeline` se detectar chamadas simultaneas. Engines que nao suportam batch processam serialmente (semaphore=1) sem mudanca.

**Alternativa rejeitada**: Novo `TranscribeBatch(repeated TranscribeFileRequest) returns (repeated TranscribeFileResponse)`. Rejeitada porque exige mudanca no proto, em todos os backends, no servicer, e adiciona complexidade sem ganho proporcional. Se necessario no futuro, pode ser adicionado sem impactar o approach atual.

---

## 12. Perspectivas Cruzadas

### Onde as Perspectivas se Encontram

| Intersecao | Sofia (Arquitetura) | Viktor (Real-Time) | Andre (Platform) |
|------------|--------------------|--------------------|------------------|
| **PriorityQueue** | Design: 2 niveis, aging, future-based | Concorrencia: asyncio ordering, event-based cancel | Metricas: queue_depth por prioridade |
| **Scheduling Loop** | Compat retroativa: `transcribe()` nao muda externamente | Performance: pool de canais, minimize overhead | Observabilidade: latency tracking per phase |
| **Cancelamento** | Cancel no servicer: cooperative, flag-based | Cancel latencia: <50ms propagacao via gRPC | Cancel metricas: histogram de latencia |
| **Dynamic Batching** | gather vs novo RPC: KISS wins | Timer de 50ms: negligivel em baixa carga, benefico em alta | Batch size histogram: observar distribuicao |
| **Streaming isolation** | Design decision: streaming fora da fila | TTFB preservation: zero overhead adicional | Separacao de metricas: batch vs streaming |

### Consenso do Time

- **Sofia**: M8 evolui o Scheduler sem reescreve-lo. A interface publica (`Scheduler.transcribe()`) nao muda -- apenas o internals. Isso e KISS: maximo impacto com minima superficie de mudanca. O Scheduler de M3 foi projetado para ser substituivel (OCP); M8 valida essa decisao.

- **Viktor**: O ponto mais critico e o scheduling loop. Ele precisa ser eficiente (nao adicionar latencia) e correto (nao perder requests, nao duplicar). O `asyncio.PriorityQueue` e a base certa -- e battle-tested para single-event-loop concurrency. O timer do `BatchAccumulator` precisa de testes cuidadosos com `asyncio.sleep` mockado para garantir determinismo.

- **Andre**: Observabilidade e a entrega mais visivel de M8 para operadores. Ate M7, metricas de scheduling nao existiam. M8 adiciona 7 metricas que respondem: "a fila esta crescendo?", "requests batch estao esperando muito?", "cancelamentos estao funcionando?", "qual o tamanho medio dos batches?". Essas metricas sao pre-requisito para tuning em producao.

---

## 13. Transicao M8 -> M9

Ao completar M8, o time deve ter:

1. **Scheduler com fila e prioridade** -- requests batch sao enfileiradas e priorizadas. Streaming permanece isolado com latencia minima.

2. **Cancelamento end-to-end** -- requests podem ser canceladas na fila (<1ms) ou em execucao (<50ms via gRPC Cancel).

3. **Dynamic batching** -- requests batch agrupadas para throughput maximo em GPU.

4. **Observabilidade de scheduling** -- metricas Prometheus para profundidade de fila, latencia por fase, tamanho de batch, cancelamentos.

5. **Pontos de extensao para M9 (Full-Duplex)**:
   - **Co-scheduling STT + TTS**: Scheduler pode ser estendido com `RequestType` (STT/TTS) como dimensao adicional de prioridade.
   - **Mute-on-speak**: Requer signal entre STT session e TTS worker, coordenado pelo Scheduler.
   - **Latency budget V2V**: LatencyTracker pode ser estendido para rastrear latencia end-to-end STT -> LLM -> TTS.

**O primeiro commit de M9 sera**: adicionar `RequestType.TTS` ao Scheduler e implementar co-scheduling de workers STT e TTS para a mesma sessao.

---

*Documento gerado pelo Time de Arquitetura (ARCH) -- Sofia Castellani, Viktor Sorokin, Andre Oliveira. Sera atualizado conforme a implementacao do M8 avanca.*
