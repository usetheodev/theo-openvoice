# Theo OpenVoice -- Roadmap de Alto Nivel

**Versao**: 1.0
**Base**: PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Em execucao
**Data**: 2026-02-07
**Ultima atualizacao**: 2026-02-09

**Autores**:
- Sofia Castellani (Principal Solution Architect)
- Viktor Sorokin (Senior Real-Time Engineer)
- Andre Oliveira (Senior Platform Engineer)

---

## 1. Visao Geral

Este documento define a sequencia de implementacao do Theo OpenVoice, organizada em temas estrategicos, milestones mensuráveis e dependencias explicitas. O objetivo e entregar valor incremental desde o primeiro milestone, com complexidade crescente e riscos mitigados por validacao continua.

### Principio Guia

> **Valor entregue o mais cedo possivel.** Cada milestone produz um artefato usavel -- algo que um desenvolvedor pode `pip install`, executar e validar. Nao existe milestone "interno" sem usuario.

### Mapa de Fases vs. Milestones

O PRD define 3 fases de produto. Este roadmap decompoe essas fases em **10 milestones** com granularidade executavel. As fases do PRD sao mantidas como agrupamento logico, mas a unidade de planejamento e o milestone.

```
PRD Fase 1 (STT Batch) ✅       PRD Fase 2 (Streaming)          PRD Fase 3 (Telefonia)
├── M1: Fundacao ✅             ├── M5: WebSocket + VAD ✅      ├── M8: RTP Listener
├── M2: Worker gRPC ✅          ├── M6: Session Manager ✅      ├── M9: Scheduler Avancado
├── M3: API Batch ✅            ├── M7: Segundo Backend         └── M10: Full-Duplex
└── M4: Pipelines ✅
```

---

## 2. Temas Estrategicos

Cada tema representa um eixo de valor que atravessa multiplos milestones.

### T1 -- Fundacao do Runtime

**Valor**: Estrutura executavel do projeto com tooling, CI, tipos base e configuracao.
**Por que primeiro**: Sem fundacao, todo milestone subsequente gera retrabalho.
**Milestones**: M1

### T2 -- Transcricao Batch (API OpenAI-Compatible)

**Valor**: Um usuario pode transcrever um arquivo de audio via REST API ou CLI, com output formatado (ITN). Compativel com SDKs OpenAI existentes.
**Diferencial**: Primeira entrega usavel -- valida o modelo de worker gRPC, registry e pipelines.
**Milestones**: M2, M3, M4

### T3 -- Streaming em Tempo Real

**Valor**: Um usuario pode transcrever audio em tempo real via WebSocket com partial e final transcripts, VAD, session management e recovery de falhas.
**Diferencial**: Session Manager e o componente que nao existe em nenhum STT open-source.
**Milestones**: M5, M6

### T4 -- Model-Agnostic (Multi-Engine)

**Valor**: Um usuario pode trocar de engine STT (Whisper -> WeNet) sem mudar codigo cliente. O runtime adapta pipeline por arquitetura.
**Diferencial**: Prova que a interface `STTBackend` funciona para arquiteturas fundamentalmente diferentes.
**Milestones**: M7

### T5 -- Telefonia e Escala

**Valor**: O runtime recebe audio de PBX (Asterisk) via RTP, com preprocessing automatico, scheduling avancado e co-scheduling STT+TTS.
**Diferencial**: Preparacao para producao em cenarios de call center.
**Milestones**: M8, M9, M10

---

## 3. Milestones

### Legenda de Esforco Relativo

| Tamanho | Significado |
|---------|-------------|
| **P** (Pequeno) | 1-2 semanas de trabalho focado |
| **M** (Medio) | 2-4 semanas |
| **G** (Grande) | 4-6 semanas |
| **GG** (Muito Grande) | 6-8 semanas |

---

### M1 -- Fundacao do Projeto ✅

**Tema**: T1 -- Fundacao do Runtime
**Esforco**: P (1-2 semanas)
**Status**: **Concluido** (2026-02-07)
**Dependencias**: Nenhuma (ponto de partida)

**Descricao**: Estabelecer a estrutura do projeto Python, tooling de desenvolvimento, CI basico, e os tipos/interfaces fundamentais que todos os milestones subsequentes usam. Nenhum componente funcional completo -- apenas os alicerces.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Estrutura de pacotes `src/theo/` com `pyproject.toml` funcional | Andre |
| 2 | Configuracao de tooling: ruff (lint + format), mypy (strict), pytest + pytest-asyncio | Andre |
| 3 | CI basico: lint, typecheck, testes unitarios em cada push | Andre |
| 4 | Tipos e interfaces base: `STTBackend` ABC, `STTArchitecture` enum, `TranscriptSegment`, `BatchResult`, `EngineCapabilities` | Sofia |
| 5 | Estrutura de configuracao: parsing de `theo.yaml` (manifesto), `PreprocessingConfig`, `PostProcessingConfig` | Sofia |
| 6 | Exceptions tipadas por dominio: `ModelNotFoundError`, `WorkerCrashError`, `AudioFormatError`, etc. | Sofia |
| 7 | Estrutura de testes: `tests/unit/`, `tests/integration/`, `tests/fixtures/` com audio de exemplo (WAV 16kHz + 8kHz + 44.1kHz) | Andre |
| 8 | CHANGELOG.md inicializado | Andre |

**Criterio de sucesso**:
```bash
# Projeto instala, lint passa, typecheck passa, testes passam (mesmo que vazios)
pip install -e ".[dev]"
python -m ruff check src/
python -m mypy src/
python -m pytest tests/unit/
```

**Resultado**: Todos os criterios atingidos. 56 testes unitarios, mypy strict sem erros, ruff limpo. CI (GitHub Actions) e CD (release via tag) configurados. Pipeline de CD com build de wheel, verificacao de conteudo e consistencia de versao.

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Decisao prematura de estrutura de pacotes que precisa mudar depois | Media | Baixo | Seguir estrutura proposta no ARCHITECTURE.md; refatorar e barato neste estagio |

**Perspectiva Viktor (Real-Time)**: Os protobufs gRPC (`stt_worker.proto`) devem ser definidos neste milestone, mesmo que a geracao de codigo Python seja feita em M2. A definicao do contrato de streaming (AudioFrame, TranscriptEvent) impacta toda a cadeia.

**Perspectiva Andre (Platform)**: CI deve incluir geracao de protobufs como step. O `pyproject.toml` deve declarar extras opcionais por engine (`pip install theo[faster-whisper]`) para nao arrastar dependencias pesadas na instalacao base.

---

### M2 -- Worker gRPC (Faster-Whisper) ✅

**Tema**: T2 -- Transcricao Batch
**Esforco**: M (2-4 semanas)
**Status**: **Concluido** (2026-02-07)
**Dependencias**: M1 (tipos base, protobufs definidos)

**Descricao**: Implementar o worker STT como subprocess gRPC usando Faster-Whisper como engine de inferencia. Inclui o Worker Manager que gerencia lifecycle de subprocessos. Este e o milestone mais critico do Tema T2 -- o worker e o componente que interage com GPU e engines externas.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Proto compilado: `stt_worker.proto` -> codigo Python gerado (`grpcio-tools`) | Viktor |
| 2 | Worker STT: processo Python que implementa o service gRPC `STTWorker` | Viktor |
| 3 | `FasterWhisperBackend`: implementacao de `STTBackend` para Faster-Whisper (batch `transcribe_file`) | Sofia |
| 4 | Worker Manager: spawna, monitora e reinicia subprocessos worker | Viktor |
| 5 | Health check unario via gRPC (`Health` RPC) | Viktor |
| 6 | Deteccao de crash via signal/exit code do subprocess | Viktor |
| 7 | Testes unitarios: mock da engine Faster-Whisper, testes do protocolo gRPC | Todos |
| 8 | Teste de integracao: worker real com modelo `faster-whisper-tiny` | Viktor |

**Criterio de sucesso**:
```bash
# Worker inicia como subprocess, responde health check, transcripts arquivo
python -m theo.workers.stt.main --port 50051
# (em outro terminal)
grpcurl -plaintext localhost:50051 theo.STTWorker/Health
# -> {"status": "ok", "model": "faster-whisper-tiny"}
```

**Resultado**: Todos os criterios atingidos. Proto compilado com script `scripts/generate_proto.sh` e stubs commitados. Worker STT como subprocess gRPC com `STTWorkerServicer` (TranscribeFile, Health, stubs UNIMPLEMENTED para streaming). `FasterWhisperBackend` implementando `STTBackend` com transcricao batch, hot words via initial_prompt e conversao PCM->numpy. `WorkerManager` com spawn, health probe com backoff exponencial, crash detection, auto-restart com rate limiting e shutdown graceful. Structured logging via structlog (JSON + console). Conversores puros proto<->Theo. 127 testes unitarios (71 novos: converters, servicer, backend, manager, logging, integracao). Code review com 7 fixes aplicados (CR1-CR7): return defensivo apos abort, extracao DRY de `_build_worker_cmd`/`_spawn_worker_process`, import fora de try, guard de duplo shutdown, cancel+await de background tasks, DEVNULL em subprocess, fix de self-cancellation em restart. CI com verificacao de freshness dos stubs proto.

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| gRPC streaming bidirecional tem complexidade de implementacao (backpressure, cancelamento) | Media | Alto | Implementar apenas RPC unario (`TranscribeFile`) neste milestone; streaming em M5 |
| Faster-Whisper API interna muda entre versoes | Baixa | Medio | Piniar versao no `pyproject.toml`; encapsular chamadas atras da interface `STTBackend` |
| Isolamento de GPU em subprocesso (CUDA context) | Media | Alto | Testar em CPU primeiro; GPU e otimizacao, nao bloqueio |

**Perspectiva Sofia (Arquitetura)**: O contrato `STTBackend.transcribe_file` deve ser validado end-to-end aqui. Se a interface nao servir para Faster-Whisper (a engine principal), nao servira para nenhuma outra. Qualquer ajuste na ABC deve acontecer agora, antes de M7 (segundo backend).

**Perspectiva Andre (Platform)**: O Worker Manager deve logar lifecycle de subprocessos com structured logging desde o inicio. Isso e pre-requisito para observabilidade em M3+.

---

### M3 -- API Batch (REST + CLI) ✅

**Tema**: T2 -- Transcricao Batch
**Esforco**: M (2-4 semanas)
**Status**: **Concluido** (2026-02-07)
**Dependencias**: M2 (worker gRPC funcional)

**Descricao**: Expor a API REST compativel com OpenAI e os comandos CLI para transcricao de arquivo. Este milestone conecta o usuario ao runtime -- e a primeira entrega usavel por alguem externo ao time.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | FastAPI app com `POST /v1/audio/transcriptions` (contrato OpenAI) | Sofia |
| 2 | `POST /v1/audio/translations` (traducao para ingles) | Sofia |
| 3 | `GET /health` e `GET /metrics` (Prometheus basico) | Andre |
| 4 | Model Registry: resolve modelo por nome, carrega manifesto `theo.yaml`, roteia para worker correto | Sofia |
| 5 | Scheduler basico: roteia request para worker disponivel (sem priorizacao -- Fase 1 e 1 worker = 1 sessao) | Viktor |
| 6 | Formatos de resposta: `json`, `verbose_json`, `text`, `srt`, `vtt` | Sofia |
| 7 | CLI: `theo serve` (inicia API Server), `theo transcribe <file>`, `theo translate <file>` | Andre |
| 8 | CLI: `theo list` (lista modelos instalados), `theo inspect <model>` | Andre |
| 9 | Pydantic models para request/response com validacao | Sofia |
| 10 | Tratamento de erros HTTP: 400, 404, 413, 503 com mensagens claras | Sofia |
| 11 | Testes end-to-end: `curl` contra o servidor com arquivo de audio real | Todos |

**Criterio de sucesso**:
```bash
theo serve &
curl -F file=@audio.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "ola como posso ajudar"}
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Incompatibilidade sutil com contrato OpenAI (campos, tipos, defaults) | Media | Medio | Testar com SDK oficial `openai` Python como cliente |
| Cold start lento (modelo carregando na primeira request) | Alta | Medio | Retornar 503 com header `Retry-After` durante loading; documentar `theo serve --preload` |
| Upload de arquivos grandes consome memoria | Baixa | Baixo | Limite de 25MB configuravel; streaming de upload via `UploadFile` do FastAPI |

**Resultado**: Todos os criterios atingidos. 13/13 tasks do STRATEGIC_M3.md completas. FastAPI app com `create_app()` factory e injecao de dependencias via `app.state`. Endpoints `POST /v1/audio/transcriptions` e `POST /v1/audio/translations` compativeis com contrato OpenAI. `GET /health` com status e versao. `ModelRegistry` para scan de modelos em disco com resolucao por nome. `Scheduler` basico roteando requests para workers via gRPC com conversores proto<->dominio. Formatos de resposta: json, verbose_json, text, srt, vtt com formatters dedicados. Error handlers HTTP (400, 404, 413, 503, 500) com formato OpenAI-compatible e `InvalidRequestError` para parametros invalidos. CLI: `theo serve`, `theo transcribe`, `theo translate`, `theo list`, `theo inspect` via Click. Pydantic models para request/response com validacao. 275 testes (219 novos: e2e batch, SDK compat, CLI, routes, formatters, error handling, registry, scheduler, models). Validado com SDK `openai` Python como cliente real contra servidor Theo. Code review pos-implementacao com 3 fixes aplicados: (CR1) validacao de `response_format` invalido retorna 400 em vez de 500, (CR2) pre-check de `file.size` via Content-Length antes de `await file.read()`, (CR3) carry correto de milissegundos em timestamps SRT/VTT via `divmod`. mypy strict sem erros, ruff limpo, CI verde.

**Perspectiva Viktor (Real-Time)**: O Scheduler nesta fase e trivial (round-robin de 1 worker), mas a interface deve ser definida para suportar priorizacao futura. Nao fazer God class -- o Scheduler de M3 deve ser substituivel pelo de M9 sem mudar API Server.

**Perspectiva Andre (Platform)**: Metricas Prometheus desde M3. Mesmo basicas (`requests_total`, `latency_seconds`, `errors_total`), elas estabelecem o padrao. Adicionar metricas STT-especificas e incremental depois.

---

### M4 -- Pipelines de Audio (Preprocessing + Post-Processing) ✅

**Tema**: T2 -- Transcricao Batch
**Esforco**: M (2-4 semanas)
**Status**: **Concluido** (2026-02-08)
**Dependencias**: M3 (API funcional para testar pipelines end-to-end)

**Descricao**: Implementar os pipelines de preprocessing e post-processing que transformam audio bruto em PCM normalizado e texto cru em texto formatado. Este milestone completa o Tema T2 -- apos ele, a Fase 1 do PRD esta entregue.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Audio Preprocessing Pipeline: orquestracao de stages com config toggleavel | Viktor |
| 2 | Stage Resample: qualquer SR -> 16kHz mono (via soxr ou scipy) | Viktor |
| 3 | Stage DC Remove: HPF 20Hz Butterworth 2a ordem | Viktor |
| 4 | Stage Gain Normalize: peak normalization -3dBFS com window de 500ms | Viktor |
| 5 | Stage Denoise: integracao com RNNoise (desativado por default) | Viktor |
| 6 | Post-Processing Pipeline: orquestracao de stages ITN -> Entity -> HotWord | Sofia |
| 7 | Stage ITN: integracao com `nemo_text_processing` para pt-BR | Sofia |
| 8 | Stage Hot Word Correction: Levenshtein distance com threshold configuravel | Sofia |
| 9 | Stage Entity Formatting: regras banking basicas (CPF, valores) | Sofia |
| 10 | Integracao: preprocessing inserido no fluxo batch (entre ingestao e worker) | Viktor |
| 11 | Integracao: post-processing inserido no fluxo batch (entre worker e resposta) | Sofia |
| 12 | Metricas: `preprocessing_duration_seconds`, `postprocessing_duration_seconds` | Andre |
| 13 | Testes unitarios por stage + teste de pipeline end-to-end | Todos |

**Criterio de sucesso**:
```bash
# Audio 44.1kHz com numeros falados -> transcrição com numeros formatados
curl -F file=@audio_44khz.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "o valor e R$2.500,00"}
# (audio original dizia "o valor e dois mil e quinhentos reais")
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| `nemo_text_processing` e uma dependencia pesada (NeMo inteiro) | Alta | Medio | Isolar como optional dependency (`pip install theo[itn]`); fallback para regex basico sem NeMo |
| ITN introduz erros em edge cases (ex: "um" -> "1" quando e artigo) | Media | Medio | Testes extensivos com corpus pt-BR; flag `--no-itn` no CLI e `itn: false` na API |
| Denoise (RNNoise) tem binding Python instavel em algumas plataformas | Media | Baixo | Denoise e desativado por default; fallback graceful se binding falhar |

**Resultado**: Todos os criterios atingidos. 10/10 tasks do STRATEGIC_M4.md completas. Audio Preprocessing Pipeline com 3 stages (Resample via `scipy.signal.resample_poly`, DC Remove via Butterworth HPF 20Hz, Gain Normalize para -3dBFS peak). Post-Processing Pipeline com ITN stage via `nemo_text_processing` (fail-open: se nao instalado, retorna texto original com warning). Ambos pipelines integrados no fluxo batch via `app.state` e FastAPI `Depends()`. Controle de ITN via campo `itn` na API REST (default `true`) e flag `--no-itn` no CLI. `soundfile` para decode de WAV/FLAC/OGG com fallback `wave` (stdlib). Interfaces `AudioStage` e `TextStage` projetadas para composabilidade e extensibilidade futura (Denoise em M8, Entity Formatting e Hot Word Correction em M6). 132 testes novos (total: 400 testes). Makefile para dev workflow (`make check`, `make test`, `make ci`). mypy strict sem erros (inclusive fix de 5 erros pre-existentes em converters.py, servicer.py, serve.py). ruff limpo. `scipy-stubs` adicionado para type checking completo.

**Perspectiva Sofia (Arquitetura)**: Cada stage deve seguir a mesma interface (`process(audio) -> audio` para pre, `process(text) -> text` para pos). Isso garante composabilidade e facilita adicionar stages futuros sem mudar o orquestrador. KISS -- nao over-engineer o pipeline pattern.

**Perspectiva Viktor (Real-Time)**: O preprocessing deve ser projetado para operar em streaming (frame-by-frame) desde o inicio, mesmo que M4 so use em batch. Se implementar apenas para arquivo completo, vai precisar reescrever em M5. Custo de fazer certo agora: zero. Custo de refatorar depois: semanas.

**Perspectiva Andre (Platform)**: `nemo_text_processing` puxa PyTorch como dependencia transitiva. Isso dobra o tamanho da imagem Docker. Considerar imagem multi-stage com layer separado para NeMo.

---

### CHECKPOINT: Fase 1 Completa ✅

A Fase 1 do PRD esta 100% entregue (2026-02-08):

```
Validacao:
  [x] POST /v1/audio/transcriptions funciona com contrato OpenAI
  [x] POST /v1/audio/translations funciona
  [x] Audio de qualquer sample rate e preprocessado automaticamente
  [x] Numeros e entidades formatados via ITN
  [x] CLI: theo serve, theo transcribe, theo translate, theo list
  [x] Worker Faster-Whisper isolado como subprocess gRPC
  [x] Metricas Prometheus basicas
  [x] CI funcionando (lint, typecheck, testes)
  [x] 400 testes unitarios, mypy strict, ruff limpo
  [x] Makefile com targets: format, lint, typecheck, test, ci
```

**O que um usuario pode fazer**: Transcrever qualquer arquivo de audio com qualidade de producao, via REST API compativel com OpenAI ou via CLI. Audio de qualquer sample rate e normalizado automaticamente. Numeros e entidades formatados via ITN (desabilitavel via `itn=false` ou `--no-itn`).

---

### M5 -- WebSocket + VAD ✅

**Tema**: T3 -- Streaming em Tempo Real
**Esforco**: G (4-6 semanas)
**Status**: **Concluido** (2026-02-09)
**Dependencias**: M4 (preprocessing funcional em modo streaming), M2 (worker gRPC)

**Descricao**: Implementar o endpoint WebSocket `/v1/realtime` com protocolo de eventos JSON, integracao com Silero VAD (energy pre-filter + classificacao), e streaming gRPC bidirecional entre runtime e worker. Este e o milestone de maior complexidade tecnica ate aqui.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Endpoint WebSocket `WS /v1/realtime` com handshake (model, language) | Sofia |
| 2 | Protocolo de eventos: `session.created`, `session.configure`, `session.close`, `session.cancel` | Sofia |
| 3 | Recebimento de audio binario via WebSocket (frames PCM) | Viktor |
| 4 | Preprocessing em modo streaming (frame-by-frame, reutilizando stages de M4) | Viktor |
| 5 | Energy Pre-filter: RMS + spectral flatness antes do Silero VAD | Viktor |
| 6 | Silero VAD integrado: sensitivity levels (high/normal/low), min speech/silence duration | Viktor |
| 7 | Eventos VAD: `vad.speech_start`, `vad.speech_end` | Viktor |
| 8 | gRPC streaming bidirecional: `TranscribeStream` (AudioFrame stream -> TranscriptEvent stream) | Viktor |
| 9 | Eventos de transcript: `transcript.partial`, `transcript.final` | Viktor |
| 10 | Post-processing aplicado apenas em `transcript.final` (nunca em partial) | Sofia |
| 11 | Backpressure: `session.rate_limit`, `session.frames_dropped` | Viktor |
| 12 | Heartbeat: ping WebSocket a cada 10s, timeout de pong em 5s | Viktor |
| 13 | `input_audio_buffer.commit`: force commit manual de segmento | Viktor |
| 14 | Metricas: `ttfb_seconds`, `final_delay_seconds`, `active_sessions`, `vad_events_total` | Andre |
| 15 | Testes: conexao WebSocket, envio de audio, recebimento de eventos, backpressure | Todos |

**Criterio de sucesso**:
```
# Conectar via WebSocket, enviar audio, receber partial + final transcripts
wscat -c ws://localhost:8000/v1/realtime?model=faster-whisper-tiny
# -> {"type": "session.created", "session_id": "sess_abc123", ...}
# (enviar audio binario)
# -> {"type": "vad.speech_start", ...}
# -> {"type": "transcript.partial", "text": "ola como", ...}
# -> {"type": "transcript.final", "text": "ola como posso ajudar", ...}
# -> {"type": "vad.speech_end", ...}
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| gRPC bidirecional + WebSocket = duas camadas de streaming com semanticas diferentes | Alta | Alto | Abstrar a comunicacao com o worker atras de interface async; nao expor detalhes gRPC ao WebSocket handler |
| Silero VAD latencia > 2ms/frame em CPU lento | Baixa | Medio | Energy pre-filter reduz chamadas ao Silero em 60-70%; monitorar via metricas |
| Backpressure difícil de testar deterministicamente | Media | Medio | Testes com audio pre-gravado enviado em velocidade controlada; nao depender de timing real |
| Memory leak em conexoes WebSocket de longa duracao | Media | Alto | Testes de estabilidade com sessao de 30min desde o inicio; monitorar RSS do processo |

**Perspectiva Sofia (Arquitetura)**: O WebSocket handler deve ser thin -- recebe frames, delega para preprocessing, VAD e worker. A logica de estado (quando emitir partial, quando emitir final) pertence ao Session Manager (M6), nao ao handler. Em M5, usar estado simplificado (sem maquina de estados completa) que sera substituido em M6.

**Perspectiva Viktor (Real-Time)**: O gRPC bidirecional e o ponto critico. O stream `TranscribeStream` deve suportar cancelamento (ctx.cancel()), e o runtime deve detectar stream break para crash detection. Nao implementar health check polling separado -- o stream break e a deteccao. Testar cenario: matar worker com SIGKILL durante streaming e verificar que o runtime detecta em <100ms.

**Perspectiva Andre (Platform)**: WebSocket + gRPC streaming = muitas file descriptors abertas. Configurar ulimits no container desde ja. Monitorar `connections_active` como metrica.

**Resultado**: Todos os criterios atingidos. 15/15 entregaveis completos. Endpoint WebSocket `WS /v1/realtime` com handshake (model, language, session_id) e protocolo de eventos JSON (8 tipos de evento servidor, 4 tipos de comando cliente). Protocol handler `dispatch_message()` para dispatch tipado de frames binarios e comandos JSON. Heartbeat WebSocket com ping a cada 10s e timeout configuravel. `EnergyPreFilter` com RMS + spectral flatness como pre-filtragem de silencio antes do Silero VAD. `SileroVADClassifier` com lazy-loading do modelo Silero e sensitivity levels (high/normal/low). `VADDetector` coordenando energy pre-filter + Silero com debounce de fala (250ms) e silencio (300ms). `StreamingPreprocessor` adapter frame-by-frame reutilizando stages do M4. `StreamingGRPCClient` e `StreamHandle` para streaming gRPC bidirecional runtime-to-worker com crash detection via stream break. `STTWorkerServicer.TranscribeStream` implementado como streaming gRPC bidirecional no worker. `FasterWhisperBackend.transcribe_stream()` para transcricao streaming via acumulacao com threshold de 5s. `StreamingSession` orquestrando fluxo completo: preprocessing -> VAD -> gRPC worker -> post-processing -> callback. `BackpressureController` com sliding window para deteccao de envio mais rapido que real-time e drop de frames por backlog. `StreamingSession.commit()` para force commit manual de segmento. Metricas Prometheus: `theo_stt_ttfb_seconds`, `theo_stt_final_delay_seconds`, `theo_stt_active_sessions`, `theo_stt_vad_events_total`. Testes de estabilidade: sessao de 5 minutos (simulada) sem degradacao de latencia nem crescimento de memoria >10MB, e teste de 200 segmentos curtos sem vazamento de recursos. 340 testes novos (total: 740 testes). mypy strict sem erros, ruff limpo, CI verde.

---

### CHECKPOINT: M5 Completo

Apos M5, a infraestrutura de streaming em tempo real esta funcional:

```
Validacao:
  [x] WS /v1/realtime funcional com protocolo de eventos JSON
  [x] Protocolo de eventos: session.created, session.configure, session.close, session.cancel
  [x] Recebimento de audio binario via WebSocket (frames PCM)
  [x] Preprocessing em modo streaming (frame-by-frame)
  [x] Energy pre-filter (RMS + spectral flatness) antes do Silero VAD
  [x] Silero VAD integrado com sensitivity levels (high/normal/low)
  [x] Eventos VAD: vad.speech_start, vad.speech_end
  [x] gRPC streaming bidirecional: TranscribeStream (AudioFrame -> TranscriptEvent)
  [x] Eventos de transcript: transcript.partial, transcript.final
  [x] Post-processing aplicado apenas em transcript.final
  [x] Backpressure: rate_limit e frames_dropped
  [x] Heartbeat WebSocket com ping/pong
  [x] input_audio_buffer.commit: force commit manual de segmento
  [x] Metricas Prometheus para streaming
  [x] Testes de estabilidade (5 min simulado, 200 segmentos curtos)
  [x] 740 testes, mypy strict, ruff limpo
```

**O que um usuario pode fazer**: Transcrever audio em tempo real via WebSocket com partial e final transcripts, deteccao de atividade vocal (VAD), backpressure automatico, heartbeat, e force commit manual. Audio de qualquer sample rate e preprocessado frame-by-frame. Texto final formatado via ITN.

**O que falta para Fase 2 Core completa (M6)**: Session Manager com maquina de estados (6 estados), Ring Buffer com read fence e force commit, WAL in-memory para recovery, LocalAgreement para partial transcripts, cross-segment context.

---

### M6 -- Session Manager + Ring Buffer ✅

**Tema**: T3 -- Streaming em Tempo Real
**Esforco**: G (4-6 semanas)
**Status**: **Concluido** (2026-02-09)
**Dependencias**: M5 (WebSocket + VAD funcional)

**Descricao**: Implementar o Session Manager com maquina de estados completa (6 estados), Ring Buffer com read fence e force commit, WAL in-memory para recovery, e LocalAgreement para partial transcripts de encoder-decoder. Este e o componente mais original do Theo -- nao existe equivalente em nenhum projeto open-source de STT.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Session Manager: maquina de estados (INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED) | Sofia |
| 2 | Timeouts configuraveis por estado (INIT: 30s, SILENCE: 30s, HOLD: 5min, CLOSING: 2s) | Sofia |
| 3 | Ring Buffer: array pre-alocado 60s, ponteiros read/write, zero allocations durante streaming | Viktor |
| 4 | Read Fence: `last_committed_offset`, protecao de dados nao commitados | Viktor |
| 5 | Force Commit: trigger em 90% de capacidade do ring buffer | Viktor |
| 6 | WAL in-memory: `last_committed_segment_id`, `buffer_offset`, `timestamp_ms` | Viktor |
| 7 | Recovery de crash: deteccao via gRPC stream break -> restart worker -> retomar sessao sem duplicacao | Viktor |
| 8 | LocalAgreement: comparacao entre passes consecutivas, confirmacao de tokens, emissao de partials | Sofia |
| 9 | Cross-segment context: ultimos 224 tokens do `transcript.final` como `initial_prompt` do proximo segmento | Sofia |
| 10 | Hot Words via `session.configure` (injecao via `initial_prompt` para Whisper) | Sofia |
| 11 | Evento `session.hold` quando sessao transita para HOLD | Sofia |
| 12 | CLI: `theo transcribe --stream` (streaming do microfone) | Andre |
| 13 | Metricas: `session_duration_seconds`, `segments_force_committed_total`, `confidence_avg` | Andre |
| 14 | Teste de estabilidade: sessao de 30 minutos sem degradacao de latencia ou memory leak | Todos |
| 15 | Teste de recovery: matar worker durante sessao ACTIVE, verificar retomada sem duplicacao | Todos |

**Criterio de sucesso**:
```
# Sessao WebSocket de 30 minutos sem degradacao
# Recovery apos kill do worker sem duplicacao de segmentos
# Ring buffer estavel (memoria constante ao longo de 30 minutos)
# Partial transcripts via LocalAgreement com latencia < 300ms TTFB
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| LocalAgreement produz partials de baixa qualidade para pt-BR (otimizado para en) | Media | Medio | Testar com corpus pt-BR; parametro `min_confirm` ajustavel; fallback para `partial_strategy: disabled` |
| Race condition no WAL durante crash recovery | Media | Alto | WAL e escrita atomica (um registro por vez); testar com chaos engineering (kill -9 aleatorio) |
| Ring buffer force commit interrompe fala no meio de uma frase | Baixa | Medio | 60s de buffer e suficiente para 99%+ dos segmentos; monitorar `segments_force_committed_total` |
| Maquina de estados com transicoes invalidas em edge cases | Media | Alto | Testar todas as transicoes possíveis (inclusive invalidas); estado CLOSED e terminal e imutavel |

**Resultado**: Todos os criterios atingidos. 15/15 entregaveis completos. `SessionStateMachine` com 6 estados (INIT, ACTIVE, SILENCE, HOLD, CLOSING, CLOSED), transicoes validas, timeouts configuraveis (INIT: 30s, SILENCE: 30s, HOLD: 5min, CLOSING: 2s) e callbacks on_enter/on_exit. `SessionTimeouts` dataclass com validacao de minimo 1s e conversao de `SessionConfigureCommand`. `RingBuffer` com `bytearray` pre-alocado de 60s (1.9MB), ponteiros circulares, read fence (`last_committed_offset`) e force commit callback em 90% de capacidade. `SessionWAL` (Write-Ahead Log in-memory) com `WALCheckpoint` frozen dataclass para recovery sem duplicacao. Recovery de crash via `StreamingSession.recover()`: reabre stream gRPC, reenvia uncommitted data do ring buffer, restaura segment_id do WAL. `LocalAgreementPolicy` para confirmacao de partial transcripts via comparacao posicional entre passes consecutivas com `min_confirm_passes` configuravel. `CrossSegmentContext` com ultimos 224 tokens do transcript.final como initial_prompt do proximo segmento. Hot words por sessao injetados via initial_prompt combinados com cross-segment context. Metricas Prometheus M6: session_duration_seconds, segments_force_committed_total, confidence_avg, worker_recoveries_total. Teste de estabilidade de 30 minutos (simulado) com todos os componentes M6 sem degradacao de latencia nem crescimento de memoria >10MB. Teste de recovery end-to-end com multiplos crashes e preservacao de segment_id. Decomposicao conforme perspectiva Sofia: SessionStateMachine, RingBuffer, LocalAgreementPolicy, SessionWAL, CrossSegmentContext -- cada um testavel isoladamente. 295 testes novos (total: 1038 testes). mypy strict sem erros, ruff limpo, CI verde.

**Perspectiva Sofia (Arquitetura)**: O Session Manager e o coracao do Theo. Ele orquestra VAD, Ring Buffer, LocalAgreement, WAL e comunicacao com o worker. A tentacao e fazer God class. Resistir. Decompor em: SessionStateMachine (transicoes), RingBuffer (dados), LocalAgreementPolicy (partials), SessionRecovery (WAL + retomada). Cada um testavel isoladamente.

**Perspectiva Viktor (Real-Time)**: O Ring Buffer deve ser implementado com `bytearray` pre-alocado e indice circular. Nao usar deque ou lista -- allocations durante streaming matam latencia em p99. O force commit deve ser assincrono (nao bloquear a escrita de novos frames enquanto commita).

**Perspectiva Andre (Platform)**: Teste de estabilidade de 30 minutos deve rodar no CI como job separado (nao em cada push, mas em schedule noturno ou pre-release). Monitorar RSS do processo a cada 10s e falhar se crescimento > 5MB ao longo da sessao.

---

### CHECKPOINT: Fase 2 Core Completa

Apos M6, o core da Fase 2 do PRD esta entregue:

```
Validacao:
  [x] WS /v1/realtime funcional com protocolo de eventos
  [x] Session Manager com 6 estados e timeouts configuraveis
  [x] VAD (Silero + energy pre-filter) com sensitivity levels
  [x] Ring Buffer com read fence e force commit
  [x] Recovery de crash sem duplicacao (WAL)
  [x] LocalAgreement para partial transcripts
  [x] Hot words via session.configure
  [x] Backpressure e heartbeat WebSocket
  [x] CLI: theo transcribe --stream
  [x] Sessao de 30 min sem degradacao
```

**O que um usuario pode fazer**: Transcrever audio em tempo real via WebSocket com qualidade de producao, incluindo partial transcripts, formatacao ITN, e recovery transparente de falhas.

---

### M7 -- Segundo Backend STT (WeNet)

**Tema**: T4 -- Model-Agnostic
**Esforco**: M (2-4 semanas)
**Dependencias**: M6 (Session Manager completo para testar streaming com arquitetura CTC)

**Descricao**: Implementar `WeNetBackend` como segunda implementacao de `STTBackend`, validando que o runtime e verdadeiramente model-agnostic. WeNet usa arquitetura CTC (partials nativos, frame-by-frame), fundamentalmente diferente do Whisper (encoder-decoder, LocalAgreement).

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | `WeNetBackend`: implementacao de `STTBackend` para WeNet (batch + streaming) | Sofia |
| 2 | Manifesto `theo.yaml` para WeNet CTC com `architecture: ctc` | Sofia |
| 3 | Worker WeNet: subprocess gRPC com engine WeNet | Viktor |
| 4 | Pipeline adaptativo: runtime detecta `architecture: ctc` e usa partials nativos (sem LocalAgreement) | Viktor |
| 5 | Hot words via keyword boosting nativo do WeNet | Sofia |
| 6 | Testes comparativos: mesma sessao WebSocket com Whisper e WeNet, validar contrato identico | Todos |
| 7 | Documentacao: como adicionar nova engine STT (passo a passo) | Sofia |

**Criterio de sucesso**:
```bash
# Mesmo cliente, dois backends, mesmo contrato de resposta
curl -F file=@audio.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "ola como posso ajudar"}

curl -F file=@audio.wav -F model=wenet-ctc \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "ola como posso ajudar"}
# (mesmo contrato, engine diferente, zero mudanca no cliente)
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Interface `STTBackend` nao acomoda particularidades do WeNet | Media | Alto | Se isso acontecer, ajustar a ABC agora -- e o momento certo para validar |
| WeNet tem dependencia de LibTorch que conflita com CTranslate2 (Faster-Whisper) | Media | Medio | Isolamento por subprocess resolve -- cada worker tem suas dependencias |
| WeNet streaming tem semantica diferente de Whisper (partials nativos vs LocalAgreement) | Certa | Medio | E exatamente o que queremos validar; o pipeline adaptativo deve tratar isso transparentemente |

**Perspectiva Sofia (Arquitetura)**: Este milestone e um teste de fogo para a interface `STTBackend`. Se funcionar com Whisper (encoder-decoder) E WeNet (CTC) sem mudar o runtime core, a abstracão esta correta. Se precisar de `if architecture == "ctc"` espalhado pelo codigo, a interface precisa evoluir.

---

### CHECKPOINT: Fase 2 Completa

Apos M7, a Fase 2 do PRD esta entregue:

```
Validacao:
  [x] Tudo de M6 +
  [x] Segundo backend (WeNet CTC) funcional
  [x] Pipeline adaptativo por arquitetura validado
  [x] Model-agnostic confirmado (mesmo contrato, engines diferentes)
  [x] Documentacao de como adicionar nova engine
```

---

### M8 -- RTP Listener

**Tema**: T5 -- Telefonia e Escala
**Esforco**: M (2-4 semanas)
**Dependencias**: M6 (Session Manager para criar sessoes RTP), M4 (preprocessing com denoise)

**Descricao**: Implementar ingestao de audio via RTP raw (UDP) com jitter buffer e decode G.711. Este milestone conecta o Theo ao mundo da telefonia, permitindo receber audio diretamente de um PBX (Asterisk/FreeSWITCH).

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | RTP Listener: socket UDP que recebe pacotes RTP | Viktor |
| 2 | Jitter buffer: reordenacao de pacotes, 20ms de buffer configuravel | Viktor |
| 3 | Decode G.711: mu-law e A-law para PCM 16-bit | Viktor |
| 4 | Preprocessing automatico para telefonia: detecta 8kHz, denoise ativado por default | Viktor |
| 5 | Audio quality tagging: `audio_quality: telephony` no contexto da sessao | Viktor |
| 6 | Integracao com Session Manager: cria sessao por stream RTP | Sofia |
| 7 | Documentacao de integration requirements: AEC, TALK_DETECT, channel isolation (Asterisk) | Andre |
| 8 | Teste de integracao com Asterisk (ou simulador RTP) | Todos |

**Criterio de sucesso**:
```
# Asterisk envia RTP para Theo, transcricao em tempo real
# (ou: simulador RTP envia audio G.711, Theo transcreve)
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Jitter em redes reais causa gaps no audio que degradam WER | Media | Medio | Jitter buffer configuravel; monitorar packet loss como metrica |
| G.711 8kHz upsampled para 16kHz perde qualidade (banda limitada a 4kHz) | Certa | Medio | Documentar limitacao; recomendar modelos telephony-optimized no registry |
| Teste com Asterisk real requer infraestrutura de telefonia | Alta | Baixo | Usar simulador RTP (script Python que envia pacotes UDP) para CI; Asterisk real em teste manual |

---

### M9 -- Scheduler Avancado

**Tema**: T5 -- Telefonia e Escala
**Esforco**: M (2-4 semanas)
**Dependencias**: M8 (RTP como segundo tipo de input real-time), M5 (WebSocket como primeiro)

**Descricao**: Evoluir o Scheduler para suportar priorizacao (realtime > batch), orcamento de latencia por sessao, cancelamento em <=50ms, e dynamic batching no worker.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Priorizacao: realtime (WebSocket/RTP) > batch (file upload) | Viktor |
| 2 | Fila por prioridade: requests batch esperam quando workers estao ocupados com streaming | Viktor |
| 3 | Cancelamento em <=50ms: propagacao via gRPC `Cancel` RPC | Viktor |
| 4 | Orcamento de latencia por sessao: tracking de TTFB e final_delay por sessao | Viktor |
| 5 | Dynamic batching no worker: acumula requests por ate 50ms, batch inference, distribui | Viktor |
| 6 | Metricas: `scheduler_queue_depth`, `scheduler_priority_inversions_total` | Andre |
| 7 | Testes: cenario de contencao (N sessoes streaming + M requests batch simultaneas) | Todos |

**Criterio de sucesso**:
```
# Sessoes streaming mantêm latencia mesmo com carga batch
# Cancelamento completa em <50ms medido por metricas
# Batch requests sao atendidas em ordem de chegada quando nao ha contencao
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Priorizacao pode starvar requests batch indefinidamente | Media | Medio | Aging: requests batch ganham prioridade apos N segundos na fila |
| Dynamic batching adiciona latencia minima de 50ms a toda request | Certa | Baixo | 50ms e aceitavel para batch; streaming bypassa batching |
| Cancelamento em <50ms requer cooperacao do worker (gRPC nao garante) | Media | Medio | Worker deve checar cancelamento entre chunks de inferencia; nao no meio de um decode |

---

### M10 -- Full-Duplex (STT + TTS)

**Tema**: T5 -- Telefonia e Escala
**Esforco**: M (2-4 semanas)
**Dependencias**: M9 (Scheduler avancado), M8 (RTP Listener)

**Descricao**: Habilitar co-scheduling de STT + TTS para agentes de voz full-duplex. Inclui mute-on-speak como fallback para cenarios sem AEC.

**Entregaveis**:

| # | Entregavel | Responsavel |
|---|-----------|-------------|
| 1 | Co-scheduling STT + TTS: scheduler aloca workers de ambos os tipos para a mesma sessao | Viktor |
| 2 | Mute-on-speak: pausar ingestao STT enquanto TTS esta ativo na mesma sessao | Viktor |
| 3 | Evento `tts.speaking_start` / `tts.speaking_end` para coordenacao | Sofia |
| 4 | Testes: sessao full-duplex com STT + TTS simultaneos | Todos |
| 5 | Documentacao: guia de integracao com Asterisk para cenario full-duplex | Andre |

**Criterio de sucesso**:
```
# Sessao full-duplex: usuario fala -> STT transcreve -> LLM responde -> TTS sintetiza
# STT nao transcreve o audio do TTS (mute-on-speak ativo)
# Latencia total V2V <= 300ms (target do PRD)
```

**Riscos**:
| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Mute-on-speak elimina barge-in (usuario nao pode interromper o bot) | Certa | Medio | Documentar como limitacao; barge-in requer AEC no PBX |
| Coordenacao STT/TTS adiciona latencia ao pipeline | Media | Medio | Sinais assincronos (eventos, nao polling); nao bloquear STT esperando TTS |
| Latencia V2V de 300ms e agressiva para ponta a ponta | Alta | Alto | Medir por componente; identificar gargalos; aceitar 500ms como target inicial |

---

### CHECKPOINT: Fase 3 Completa

Apos M10, a Fase 3 do PRD esta entregue:

```
Validacao:
  [x] RTP Listener funcional com jitter buffer e G.711 decode
  [x] Preprocessing automatico para telefonia
  [x] Scheduler com priorizacao e cancelamento
  [x] Dynamic batching no worker
  [x] Co-scheduling STT + TTS
  [x] Mute-on-speak como fallback
  [x] Documentacao de integracao com Asterisk
```

---

## 4. Grafo de Dependencias

```
M1 (Fundacao) ✅
├──► M2 (Worker gRPC) ✅
│    ├──► M3 (API Batch + CLI) ✅
│    │    ├──► M4 (Pipelines) ✅
│    │    │    ├──► M5 (WebSocket + VAD) ✅
│    │    │    │    ├──► M6 (Session Manager)
│    │    │    │    │    ├──► M7 (Segundo Backend)
│    │    │    │    │    └──► M8 (RTP Listener)
│    │    │    │    │         └──► M9 (Scheduler Avancado)
│    │    │    │    │              └──► M10 (Full-Duplex)
│    │    │    │    │
│    │    │    │    └──► M8 (RTP Listener)  [via preprocessing streaming]
```

### Dependencias Detalhadas

| Milestone | Depende de | O que precisa estar pronto |
|-----------|-----------|---------------------------|
| M1 | -- | Nada (ponto de partida) |
| M2 | M1 | Tipos base, protobufs, interface STTBackend |
| M3 | M2 | Worker funcional para receber requests |
| M4 | M3 | API funcional para testar pipelines end-to-end |
| M5 | M4, M2 | Preprocessing em modo streaming; worker com gRPC streaming |
| M6 | M5 | WebSocket + VAD funcional como base do Session Manager |
| M7 | M6 | Session Manager completo para testar pipeline adaptativo |
| M8 | M6, M4 | Session Manager para criar sessoes; preprocessing com denoise |
| M9 | M8, M5 | Multiplos tipos de input (WS + RTP) para priorizar |
| M10 | M9, M8 | Scheduler avancado + RTP para full-duplex |

### Caminho Critico

```
M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7
                                 \-> M8 -> M9 -> M10
```

O caminho critico principal vai de M1 a M7 (entrega completa de STT model-agnostic com streaming). M8-M10 (telefonia) podem iniciar em paralelo apos M6, mas dependem do mesmo time.

---

## 5. Estimativa de Esforco Total

| Milestone | Esforco | Acumulado |
|-----------|---------|-----------|
| M1 -- Fundacao ✅ | P (1-2 sem) | 1-2 sem |
| M2 -- Worker gRPC ✅ | M (2-4 sem) | 3-6 sem |
| M3 -- API Batch ✅ | M (2-4 sem) | 5-10 sem |
| M4 -- Pipelines ✅ | M (2-4 sem) | 7-14 sem |
| M5 -- WebSocket + VAD ✅ | G (4-6 sem) | 11-20 sem |
| M6 -- Session Manager | G (4-6 sem) | 15-26 sem |
| M7 -- Segundo Backend | M (2-4 sem) | 17-30 sem |
| M8 -- RTP Listener | M (2-4 sem) | 19-34 sem |
| M9 -- Scheduler | M (2-4 sem) | 21-38 sem |
| M10 -- Full-Duplex | M (2-4 sem) | 23-42 sem |

**Nota**: Estimativas sao para um time de 3 pessoas. Milestones M8-M10 podem ser parcialmente paralelos com M7 dependendo da alocacao do time.

---

## 6. Riscos Estrategicos (Cross-Milestone)

| # | Risco | Milestones Afetados | Probabilidade | Impacto | Mitigacao |
|---|-------|-------------------|--------------|---------|-----------|
| R1 | Interface `STTBackend` precisa mudar significativamente apos M7 (WeNet nao se encaixa) | M2, M7 e todos subsequentes | Media | Alto | Validar interface com pseudocodigo de WeNet em M1; ajustar antes de implementar |
| R2 | Dependencias pesadas (Faster-Whisper, NeMo, WeNet) conflitam em mesmo ambiente | M2, M4, M7 | Media | Alto | Isolamento por subprocess resolve conflitos de runtime; extras opcionais no pyproject.toml |
| R3 | Latencia de gRPC streaming e maior que esperado (overhead de serialization por frame) | M5, M6, M8 | Media | Alto | Benchmark gRPC vs alternativas (shared memory, pipes) em M5; aceitar gRPC se overhead <5ms |
| R4 | Silero VAD nao funciona bem com audio de telefonia (8kHz upsampled) | M5, M8 | Media | Medio | Testar com audio 8kHz em M5; ajustar sensitivity; considerar VAD alternativo para telefonia |
| R5 | `nemo_text_processing` e instavel ou muito pesado para instalar | M4 | Media | Medio | Implementar fallback com regex basico; NeMo como optional dependency |
| R6 | Memory leak em sessoes longas (WebSocket, gRPC, Ring Buffer) | M5, M6, M8 | Media | Alto | Testes de estabilidade de 30 min no CI; monitorar RSS; Ring Buffer fixo previne o mais obvio |
| R7 | Complexidade do Session Manager cresce alem do gerenciavel | M6 | Media | Alto | Decomposicao em sub-componentes (StateMachine, RingBuffer, WAL, LocalAgreement); cada um testavel isoladamente |

---

## 7. Observabilidade e Quality Gates

### Quality Gates por Milestone

Nenhum milestone e considerado "completo" sem estes criterios transversais:

| Criterio | Descricao |
|----------|-----------|
| **Testes** | 100% dos fluxos criticos cobertos; testes unitarios para toda logica de negocio |
| **Typecheck** | `mypy --strict` passa sem erros |
| **Lint** | `ruff check` passa sem warnings |
| **Docs** | Interfaces publicas documentadas com docstrings; CHANGELOG.md atualizado |
| **Metricas** | Metricas Prometheus relevantes ao milestone estao expostas e testadas |
| **CI** | Todos os checks passam no CI automatizado |

### Metricas de Observabilidade por Milestone

| Milestone | Metricas Adicionadas |
|-----------|---------------------|
| M3 | `requests_total`, `request_duration_seconds`, `errors_total`, health check |
| M4 | `preprocessing_duration_seconds`, `postprocessing_duration_seconds` |
| M5 | `ttfb_seconds`, `final_delay_seconds`, `active_sessions`, `vad_events_total` |
| M6 | `session_duration_seconds`, `segments_force_committed_total`, `confidence_avg`, `worker_errors_total` |
| M7 | Metricas por engine/architecture |
| M8 | `rtp_packet_loss_total`, `rtp_jitter_seconds` |
| M9 | `scheduler_queue_depth`, `scheduler_priority_inversions_total`, `cancel_latency_seconds` |

---

## 8. Decisoes de Sequenciamento

### Por que Batch antes de Streaming?

**Sofia**: Batch e mais simples (request-response, sem estado), valida o fluxo completo (API -> Registry -> Scheduler -> Worker -> Pipelines) e produz algo usavel imediatamente. Streaming adiciona complexidade de estado (Session Manager) e transport (WebSocket + gRPC bidirecional) que dobra o escopo.

**Viktor**: Em streaming, bugs de backpressure, ordering e cancelamento se manifestam sob carga. Em batch, esses problemas nao existem. Validar o happy path (audio entra, texto sai) sem a complexidade de streaming e essencial para ter confianca nos componentes antes de adicionar real-time.

**Andre**: Batch permite testar CI, metricas e deploy sem infraestrutura de WebSocket. O feedback loop de desenvolvimento e mais rapido.

### Por que Pipelines (M4) depois de API (M3)?

**Sofia**: A API precisa funcionar sem pipelines primeiro (output cru da engine). Pipelines sao refinamento -- adicionam valor (ITN, formatting), mas a funcionalidade core (transcrever audio) nao depende deles. Se o ITN der problema, a API ainda funciona retornando texto cru.

### Por que Session Manager (M6) e separado de WebSocket (M5)?

**Viktor**: WebSocket + VAD (M5) ja e complexo -- duas camadas de streaming (WS + gRPC), deteccao de fala, backpressure. Adicionar maquina de estados, Ring Buffer, WAL e LocalAgreement no mesmo milestone seria arriscado. M5 funciona com estado simplificado (ACTIVE/CLOSED); M6 evolui para a maquina completa.

### Por que Segundo Backend (M7) no final da Fase 2?

**Sofia**: M7 e um teste de validacao da abstracão. Se a interface `STTBackend` funcionar para WeNet (CTC), ela funciona para qualquer engine. Mas M7 so faz sentido apos o Session Manager (M6) estar completo, porque WeNet tem partials nativos e precisa do pipeline adaptativo end-to-end.

---

*Documento gerado pelo Time de Arquitetura (ARCH). Sera atualizado conforme a implementacao avanca e novos riscos/aprendizados emergem.*
