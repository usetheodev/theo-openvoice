# M7 -- Segundo Backend STT (WeNet) -- Strategic Roadmap

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

M7 e o milestone que prova que o Theo OpenVoice e verdadeiramente **model-agnostic**. Ate M6, o runtime funciona apenas com Faster-Whisper (encoder-decoder). M7 adiciona WeNet (CTC) como segundo backend, validando que a interface `STTBackend` suporta arquiteturas fundamentalmente diferentes sem mudancas no runtime core.

### O que M7 valida

1. **Interface `STTBackend` e suficiente para CTC**: `WeNetBackend` implementa a mesma ABC que `FasterWhisperBackend`, com `architecture = STTArchitecture.CTC`.
2. **Pipeline adaptativo funciona**: O runtime detecta `architecture: ctc` no manifesto e adapta o fluxo -- partials nativos (sem LocalAgreement), frame-by-frame no streaming.
3. **Contrato identico para o cliente**: Mesmo request, mesmo response format. Trocar `model=faster-whisper-tiny` por `model=wenet-ctc` nao muda o contrato da API.
4. **Isolamento de dependencias**: WeNet (LibTorch) e Faster-Whisper (CTranslate2) coexistem via isolamento por subprocess -- cada worker carrega apenas suas dependencias.
5. **Hot words por mecanismo nativo**: Whisper usa initial_prompt; WeNet usa keyword boosting nativo. O runtime escolhe o mecanismo correto baseado na engine.

### O que M7 habilita

- **Confianca na abstracao**: Se funciona para encoder-decoder E CTC, funciona para qualquer arquitetura futura (Paraformer, Wav2Vec2).
- **Documentacao de extensibilidade**: O passo-a-passo de "como adicionar nova engine" e validado por ter sido feito duas vezes.
- **Opcoes de deploy**: Operadores podem escolher engine por cenario (Whisper para qualidade, WeNet para latencia).

### Criterio de sucesso (Demo Goal)

```bash
# Mesmo cliente, dois backends, mesmo contrato de resposta
curl -F file=@audio.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "ola como posso ajudar"}

curl -F file=@audio.wav -F model=wenet-ctc \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "ola como posso ajudar"}
# (mesmo contrato, engine diferente, zero mudanca no cliente)

# Streaming com CTC: partials nativos (sem LocalAgreement)
wscat -c "ws://localhost:8000/v1/realtime?model=wenet-ctc"
# -> {"type": "session.created", "session_id": "sess_...", ...}
# (enviar audio)
# -> {"type": "vad.speech_start", ...}
# -> {"type": "transcript.partial", "text": "ola", ...}    (nativo, <100ms TTFB)
# -> {"type": "transcript.partial", "text": "ola como", ...}
# -> {"type": "transcript.final", "text": "ola como posso ajudar", ...}
# -> {"type": "vad.speech_end", ...}
```

### Conexao com o Roadmap

```
PRD Fase 2 (Streaming)
  M5: WebSocket + VAD           [COMPLETO]
  M6: Session Manager           [COMPLETO]
  M7: Segundo Backend           [<-- ESTE MILESTONE]

Caminho critico: M5 -> M6 -> M7
                         \-> M8 (RTP Listener)
```

M7 completa a Fase 2 do PRD. Apos M7, o Theo e model-agnostic validado. M8-M10 (Fase 3) adicionam telefonia e scheduling avancado.

---

## 2. Pre-Requisitos (de M6)

### O que ja existe e sera REUTILIZADO sem mudancas

| Componente | Pacote | Uso em M7 |
|------------|--------|-----------|
| Interface `STTBackend` (ABC) | `theo.workers.stt.interface` | WeNet implementa a mesma interface. Nenhuma mudanca na ABC. |
| `STTArchitecture` enum (CTC ja definido) | `theo._types` | `STTArchitecture.CTC` ja existe desde M1. |
| gRPC proto (`stt_worker.proto`) | `theo.proto` | Mesmo protocolo -- `TranscribeFile`, `TranscribeStream`, `Health`. |
| `STTWorkerServicer` (servicer gRPC) | `theo.workers.stt.servicer` | Reutilizado sem mudancas -- ja aceita qualquer `STTBackend`. |
| Converters proto <-> dominio | `theo.workers.stt.converters` | Reutilizados sem mudancas. |
| `WorkerManager` (spawn, health, restart) | `theo.workers.manager` | Reutilizado -- spawn de worker WeNet identico ao Faster-Whisper. |
| `StreamingGRPCClient` + `StreamHandle` | `theo.scheduler.streaming` | Reutilizados -- comunicacao gRPC e identica para qualquer engine. |
| `SessionStateMachine` (6 estados) | `theo.session.state_machine` | Reutilizada -- WeNet passa pelos mesmos estados. |
| `RingBuffer` (read fence, force commit) | `theo.session.ring_buffer` | Reutilizado -- armazena audio independente da engine. |
| `SessionWAL` (recovery) | `theo.session.wal` | Reutilizado -- recovery funciona igual para qualquer engine. |
| `CrossSegmentContext` (initial_prompt) | `theo.session.cross_segment` | Reutilizado para WeNet se `supports_initial_prompt: true`. |
| Audio Preprocessing Pipeline | `theo.preprocessing` | Reutilizado -- qualquer engine recebe PCM 16kHz normalizado. |
| Post-Processing Pipeline (ITN) | `theo.postprocessing` | Reutilizado -- ITN e aplicado no texto final independente da engine. |
| `BackpressureController` | `theo.session.backpressure` | Reutilizado sem mudancas. |
| WebSocket endpoint e protocolo | `theo.server.routes.realtime` | Reutilizado -- protocolo de eventos e identico. |
| REST endpoints (transcriptions, translations) | `theo.server.routes` | Reutilizados -- contrato OpenAI-compatible e o mesmo. |
| `ModelRegistry` | `theo.registry.registry` | Reutilizado -- scan encontra manifestos WeNet automaticamente. |
| `ModelManifest` (Pydantic) | `theo.config.manifest` | Reutilizado -- `architecture: ctc` ja e parseavel. |

### O que sera MODIFICADO

| Componente | Pacote | O que muda |
|------------|--------|------------|
| `_create_backend()` (factory de engines) | `theo.workers.stt.main` | Adicionar `elif engine == "wenet"` para instanciar `WeNetBackend`. |
| `StreamingSession` | `theo.session.streaming` | Adaptar para pular LocalAgreement quando `architecture == CTC`. Ponto unico de decisao. |
| `_create_streaming_session()` | `theo.server.routes.realtime` | Ler `architecture` do manifesto e passar ao `StreamingSession` para decisao de pipeline. |
| `pyproject.toml` | raiz | Adicionar optional dependency `wenet` e extras `[wenet]`. |

### O que sera CRIADO

| Componente | Pacote | Descricao |
|------------|--------|-----------|
| `WeNetBackend` | `theo.workers.stt.wenet` | Implementacao de `STTBackend` para WeNet. |
| Manifesto WeNet | fixtures/manifestos | `theo.yaml` para modelo WeNet CTC. |
| Documentacao | `docs/ADDING_ENGINE.md` | Guia passo-a-passo de como adicionar nova engine STT. |

---

## 3. Visao Geral da Arquitetura M7

### 3.1 Decomposicao: Onde WeNet se Encaixa

```
                          +-----------------------+
                          |    API Server          |
                          |    (FastAPI)           |
                          |    /v1/audio/*         |
                          |    /v1/realtime        |
                          +-----------+-----------+
                                      |
                          +-----------+-----------+
                          |    Scheduler           |
                          |    (roteia por modelo) |
                          +-----------+-----------+
                                      |
                     +----------------+----------------+
                     |                                 |
           +---------+---------+             +---------+---------+
           | Worker Faster-    |             | Worker WeNet      |
           | Whisper           |             | (NOVO)            |
           | (subprocess gRPC) |             | (subprocess gRPC) |
           |                   |             |                   |
           | FasterWhisper-    |             | WeNetBackend      |
           | Backend           |             | (NOVO)            |
           | (encoder-decoder) |             | (CTC)             |
           +-------------------+             +-------------------+
```

### 3.2 Pipeline Adaptativo: Encoder-Decoder vs CTC

O ponto critico de M7 e que o `StreamingSession` adapte o fluxo de streaming baseado na arquitetura. A decisao e feita em um unico ponto:

```
StreamingSession._receive_worker_events():
  |
  v
  if architecture == ENCODER_DECODER:
      -> LocalAgreement compara passes
      -> Tokens confirmados -> transcript.partial
      -> VAD speech_end -> flush -> transcript.final
  |
  elif architecture == CTC:
      -> Worker retorna partials nativos (is_final=False)
      -> Emitir diretamente como transcript.partial
      -> Worker retorna final (is_final=True) -> transcript.final
```

**Perspectiva Sofia**: O `if architecture` deve estar em exatamente UM ponto do codigo. Se precisar de `if architecture == CTC` espalhado pelo runtime, a abstracão falhou. O `STTBackend.transcribe_stream()` do WeNet ja retorna `TranscriptSegment` com `is_final` correto -- o runtime so precisa decidir se aplica LocalAgreement antes de emitir.

### 3.3 Data Flow: Batch (WeNet)

```
Cliente
  | POST /v1/audio/transcriptions (model=wenet-ctc)
  v
API Server (FastAPI)
  | Resolve modelo via Registry -> engine: wenet, architecture: ctc
  v
Preprocessing Pipeline (resample, normalize)
  | PCM 16kHz mono
  v
Scheduler -> gRPC TranscribeFile -> Worker WeNet
  | WeNet inference (LibTorch CTC)
  v
BatchResult (texto, segmentos, timestamps)
  | ITN (post-processing)
  v
JSON Response (mesmo formato que Faster-Whisper)
```

### 3.4 Data Flow: Streaming (WeNet)

```
WebSocket frame (binary, PCM)
  |
  v
StreamingPreprocessor (reutilizado)
  | float32 16kHz mono
  v
VADDetector (reutilizado)
  | VADEvent (speech_start / speech_end)
  v
SessionStateMachine (reutilizado)
  | Transicoes de estado
  v
RingBuffer (reutilizado)
  | Armazena audio
  v
gRPC StreamHandle (reutilizado)
  | AudioFrame -> Worker WeNet
  v
WeNetBackend.transcribe_stream()
  | TranscriptSegment com is_final (partials nativos)
  v
StreamingSession (ADAPTADO)
  | architecture == CTC: emite segment diretamente
  | SEM LocalAgreement
  v
Post-processing (ITN) -- apenas em final
  v
WebSocket event JSON
```

---

## 4. Epics

### Epic 1: WeNet Backend

Implementar `WeNetBackend` como segunda implementacao de `STTBackend`, com batch e streaming. Criar manifesto `theo.yaml` e registrar a engine na factory do worker.

**Racional**: E a razao de existir de M7. Prova que a interface `STTBackend` e suficiente para uma arquitetura fundamentalmente diferente (CTC vs encoder-decoder).

**Responsavel principal**: Sofia (design, implementacao do backend, manifesto)

**Tasks**: M7-01, M7-02, M7-03, M7-04

### Epic 2: Pipeline Adaptativo

Adaptar o `StreamingSession` para decidir entre LocalAgreement (encoder-decoder) e passthrough direto (CTC) baseado na arquitetura do modelo. O ponto de decisao deve ser unico e limpo.

**Racional**: Sem pipeline adaptativo, o WeNet funcionaria apenas em batch. Streaming com partials nativos e o diferencial de CTC -- TTFB <100ms vs ~300ms do Whisper com LocalAgreement.

**Responsavel principal**: Viktor (integracao no streaming), Sofia (design do ponto de decisao)

**Tasks**: M7-05, M7-06

### Epic 3: Testes Comparativos, Documentacao e Finalizacao

Validar que ambos os backends produzem o mesmo contrato de resposta. Documentar como adicionar nova engine. Atualizar docs.

**Racional**: Se o contrato difere entre engines, o runtime nao e model-agnostic -- e apenas multi-engine. Testes comparativos e a prova definitiva. Documentacao garante que o terceiro backend seja adicionado por qualquer dev.

**Responsavel principal**: Andre (testes, docs, CI), Sofia (review de contrato), Viktor (testes de streaming)

**Tasks**: M7-07, M7-08, M7-09, M7-10

---

## 5. Tasks (Detalhadas)

### M7-01: WeNetBackend -- Transcricao Batch

**Epic**: E1 -- WeNet Backend
**Estimativa**: L (5-7 dias)
**Dependencias**: Nenhuma (componente isolado, usa `STTBackend` ABC existente)
**Desbloqueia**: M7-02, M7-03, M7-07

**Contexto/Motivacao**: Implementar `WeNetBackend` com foco em `transcribe_file()`. WeNet usa LibTorch como runtime de inferencia e produz output CTC (character/token-level). O backend deve encapsular toda a complexidade da API WeNet atras da interface `STTBackend`, retornando `BatchResult` identico ao do `FasterWhisperBackend`. O import de `wenet` deve ser guardado (optional dependency), seguindo o mesmo padrao de `FasterWhisperBackend` com `faster_whisper`.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Classe `WeNetBackend(STTBackend)` em `theo/workers/stt/wenet.py` | Streaming (`transcribe_stream`) -- M7-02 |
| `architecture` property retorna `STTArchitecture.CTC` | Hot words nativo -- M7-04 |
| `load()`: carrega modelo WeNet via API Python | Pipeline adaptativo no runtime -- M7-05 |
| `transcribe_file()`: transcricao batch retornando `BatchResult` | |
| `capabilities()`: retorna `EngineCapabilities` com flags corretas | |
| `unload()`: libera modelo da memoria | |
| `health()`: status do backend | |
| Import guardado: `try: import wenet except ImportError` | |
| Conversao de output WeNet -> `BatchResult` (texto, segmentos, timestamps) | |
| Funcoes auxiliares puras para conversao (como `_fw_segment_to_detail` em faster_whisper.py) | |

**Entregaveis**:
- `src/theo/workers/stt/wenet.py` -- `WeNetBackend`
- `tests/unit/test_wenet_backend.py` -- testes com mock da API WeNet

**DoD**:
- [ ] `WeNetBackend` implementa toda a interface `STTBackend`
- [ ] `architecture` retorna `STTArchitecture.CTC`
- [ ] `load()` carrega modelo WeNet (verificavel com mock ou modelo real)
- [ ] `transcribe_file()` retorna `BatchResult` com `text`, `language`, `duration`, `segments`
- [ ] `BatchResult.segments` contem `SegmentDetail` com `start`, `end`, `text`
- [ ] Import guardado: sem WeNet instalado, `load()` levanta `ModelLoadError` com mensagem clara
- [ ] `unload()` libera recursos, `health()` retorna status correto
- [ ] Funcoes de conversao sao puras (sem side effects, sem IO)
- [ ] Testes: >=15 testes com mock da API WeNet (load, transcribe, unload, health, erros, conversao)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: Seguir exatamente o padrao de `FasterWhisperBackend` como referencia: import guardado no topo, funcoes auxiliares puras no final do arquivo, conversao de tipos WeNet -> tipos Theo isolada em funcoes nomeadas. A API do WeNet retorna resultados em formato diferente do Faster-Whisper -- as funcoes de conversao sao o ponto critico para garantir que o `BatchResult` seja identico.

**Perspectiva Andre**: WeNet puxa LibTorch como dependencia. Declarar como optional no `pyproject.toml`: `wenet = ["wenet"]`. O isolamento por subprocess garante que LibTorch nao conflita com CTranslate2 no mesmo processo. Verificar se ha incompatibilidade em nivel de shared libraries no mesmo container (CUDA symbols).

---

### M7-02: WeNetBackend -- Transcricao Streaming

**Epic**: E1 -- WeNet Backend
**Estimativa**: M (3-5 dias)
**Dependencias**: M7-01
**Desbloqueia**: M7-05, M7-07

**Contexto/Motivacao**: WeNet CTC produz output frame-by-frame com partials nativos. Diferente do Faster-Whisper (que acumula 5s antes de inferir), WeNet processa cada frame e retorna tokens incrementalmente. O `transcribe_stream()` deve retornar `TranscriptSegment` com `is_final=False` para partials e `is_final=True` para finals. O runtime NAO aplica LocalAgreement -- os partials do WeNet ja sao nativos.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| `transcribe_stream()` em `WeNetBackend` | LocalAgreement (nao se aplica a CTC) |
| Async generator que yield `TranscriptSegment` (partial + final) | Pipeline adaptativo no runtime (M7-05) |
| Partials nativos: cada chunk processado retorna tokens parciais com `is_final=False` | |
| Finals: ao receber chunk vazio (fim do stream), emite segmento final com `is_final=True` | |
| Conversao de output incremental WeNet -> `TranscriptSegment` | |

**Entregaveis**:
- Alteracao em `src/theo/workers/stt/wenet.py` -- `transcribe_stream()`
- Testes adicionais em `tests/unit/test_wenet_backend.py`

**DoD**:
- [ ] `transcribe_stream()` e async generator que yield `TranscriptSegment`
- [ ] Partials nativos: frame processado -> `TranscriptSegment(is_final=False, text="parcial")`
- [ ] Finals: fim do stream -> `TranscriptSegment(is_final=True, text="completo")`
- [ ] `segment_id` incrementa corretamente entre segmentos
- [ ] `confidence` e `language` preenchidos quando disponiveis no output WeNet
- [ ] Chunk vazio (`b""`) sinaliza fim do stream (mesmo contrato que FasterWhisperBackend)
- [ ] Testes: >=10 testes com mock (streaming normal, partials, finals, stream vazio, erro durante streaming)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O WeNet retorna resultados parciais a cada frame via sua API de streaming. O `transcribe_stream()` deve fazer o yield de cada resultado parcial imediatamente -- nao acumular como o Faster-Whisper faz. Isso e o diferencial de CTC: TTFB <100ms porque o primeiro token sai no primeiro frame processado. Testar que o primeiro `TranscriptSegment` e emitido apos o PRIMEIRO chunk recebido (nao apos acumular N chunks).

---

### M7-03: Manifesto WeNet + Registro na Factory

**Epic**: E1 -- WeNet Backend
**Estimativa**: S (1-2 dias)
**Dependencias**: M7-01
**Desbloqueia**: M7-07

**Contexto/Motivacao**: O `ModelRegistry` descobre modelos via `theo.yaml` em subdiretorios. M7-03 cria o manifesto para WeNet CTC e registra a engine na factory `_create_backend()` do worker. O manifesto deve declarar `architecture: ctc` e capabilities corretas.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Manifesto `theo.yaml` para WeNet CTC (fixture de teste + exemplo em docs) | Download automatico de modelos (`theo pull`) |
| Registro de `engine: "wenet"` em `_create_backend()` de `theo.workers.stt.main` | Hot words no manifesto (M7-04) |
| Validacao: `ModelManifest.from_yaml_string()` parseia manifesto WeNet corretamente | |
| `ModelRegistry.scan()` encontra modelos WeNet ao lado de Faster-Whisper | |
| Capability `architecture: ctc` parseada como `STTArchitecture.CTC` | |

**Entregaveis**:
- Fixture `tests/fixtures/wenet-ctc/theo.yaml` -- manifesto de teste
- Alteracao em `src/theo/workers/stt/main.py` -- `_create_backend("wenet")`
- `tests/unit/test_wenet_manifest.py`

**DoD**:
- [ ] `theo.yaml` para WeNet e valido e parseavel por `ModelManifest.from_yaml_string()`
- [ ] Campos obrigatorios presentes: `name`, `version`, `engine: wenet`, `type: stt`, `architecture: ctc`
- [ ] `capabilities.streaming: true`, `capabilities.architecture: ctc`
- [ ] `capabilities.hot_words: true` (WeNet suporta keyword boosting nativo)
- [ ] `_create_backend("wenet")` retorna instancia de `WeNetBackend`
- [ ] `_create_backend("nao-existe")` continua levantando `ValueError`
- [ ] `ModelRegistry.scan()` encontra e indexa manifesto WeNet corretamente
- [ ] Testes: >=8 testes (parse do manifesto, factory, registry scan com ambos os modelos, campos invalidos)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: O manifesto e o contrato declarativo. Ele informa o runtime sobre capacidades SEM o runtime precisar instanciar a engine. O campo `architecture: ctc` e o que o `StreamingSession` usa para decidir o pipeline. Se o manifesto estiver errado, o pipeline sera errado. Testar que `architecture: ctc` no YAML resulta em `STTArchitecture.CTC` no `ModelManifest`.

---

### M7-04: Hot Words via Keyword Boosting Nativo (WeNet)

**Epic**: E1 -- WeNet Backend
**Estimativa**: M (3-5 dias)
**Dependencias**: M7-01
**Desbloqueia**: M7-07

**Contexto/Motivacao**: Whisper usa hot words injetados via `initial_prompt` (hack semantico). WeNet suporta keyword boosting nativo via parametros de decoding -- e mais eficaz e preciso. M7-04 implementa hot words no `WeNetBackend` usando o mecanismo nativo, e adapta o `StreamingSession` para escolher o mecanismo correto baseado em `EngineCapabilities.supports_hot_words`.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Hot words em `transcribe_file()` via parametros de decoding WeNet | Hot Word Correction post-processing (Levenshtein -- futuro) |
| Hot words em `transcribe_stream()` via parametros de decoding WeNet | Alteracao do mecanismo de hot words do Whisper (ja funciona via initial_prompt) |
| `EngineCapabilities.supports_hot_words = True` para WeNet | |
| `StreamingSession`: se `supports_hot_words`, enviar via `hot_words` field do AudioFrame; senao, via `initial_prompt` (comportamento existente para Whisper) | |

**Entregaveis**:
- Alteracao em `src/theo/workers/stt/wenet.py` -- hot words nos metodos de transcricao
- Alteracao em `src/theo/session/streaming.py` -- escolha de mecanismo baseada em capabilities
- `tests/unit/test_wenet_hot_words.py`

**DoD**:
- [ ] `WeNetBackend.capabilities()` retorna `supports_hot_words=True`
- [ ] `transcribe_file(hot_words=["PIX", "Selic"])` usa keyword boosting nativo do WeNet
- [ ] `transcribe_stream(hot_words=["PIX"])` usa keyword boosting nativo
- [ ] `StreamingSession` com engine CTC: hot words enviados via `AudioFrame.hot_words` (campo proto existente)
- [ ] `StreamingSession` com engine encoder-decoder: hot words enviados via `initial_prompt` (comportamento inalterado)
- [ ] A escolha de mecanismo nao usa `if architecture == CTC` -- usa `if capabilities.supports_hot_words`
- [ ] Testes: >=8 testes (hot words em batch, streaming, sem hot words, capabilities corretas, escolha de mecanismo)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: A decisao de mecanismo de hot words deve ser por capability (`supports_hot_words`), NAO por arquitetura. Uma engine encoder-decoder futura pode suportar keyword boosting nativo. Uma engine CTC pode nao suportar. A capability e a fonte de verdade, nao a arquitetura.

**Perspectiva Viktor**: O campo `hot_words` ja existe no proto `AudioFrame`. O worker WeNet recebe os hot words e os passa para a API de decoding do WeNet. O servicer (`STTWorkerServicer`) ja extrai `hot_words` do primeiro frame e passa para `backend.transcribe_stream()`. Zero mudancas no servicer ou no proto.

---

### M7-05: Pipeline Adaptativo -- StreamingSession por Arquitetura

**Epic**: E2 -- Pipeline Adaptativo
**Estimativa**: L (5-7 dias)
**Dependencias**: M7-02, M7-03
**Desbloqueia**: M7-06, M7-07, M7-08

**Contexto/Motivacao**: O `StreamingSession` de M6 assume encoder-decoder (Whisper): acumula audio, usa LocalAgreement para partials, e flush no speech_end. Para CTC (WeNet), o fluxo e diferente: o worker retorna partials nativos frame-by-frame, e o runtime apenas repassa. M7-05 adiciona a logica de adaptacao: um unico ponto de decisao que seleciona entre `_receive_worker_events_encoder_decoder()` (com LocalAgreement) e `_receive_worker_events_ctc()` (passthrough direto).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Parametro `architecture: STTArchitecture` no construtor do `StreamingSession` | Suporte a `streaming-native` (Paraformer -- futuro, M7 valida CTC apenas) |
| `_create_streaming_session()` le `architecture` do manifesto e passa ao `StreamingSession` | Mudancas na interface publica de `StreamingSession` (process_frame, commit, close) |
| `_receive_worker_events()` delega para metodo especifico por arquitetura | |
| Para `CTC`: partials nativos do worker emitidos diretamente, sem LocalAgreement | |
| Para `ENCODER_DECODER`: comportamento existente mantido (LocalAgreement) | |
| LocalAgreement NAO instanciado para CTC (economia de memoria) | |
| Cross-segment context: enviado apenas se `supports_initial_prompt` da engine | |

**Entregaveis**:
- Alteracao em `src/theo/session/streaming.py` -- parametro `architecture`, delegacao por tipo
- Alteracao em `src/theo/server/routes/realtime.py` -- passa `architecture` ao `StreamingSession`
- `tests/unit/test_streaming_session_ctc.py`

**DoD**:
- [ ] `StreamingSession(architecture=STTArchitecture.CTC)` nao instancia `LocalAgreementPolicy`
- [ ] `StreamingSession(architecture=STTArchitecture.ENCODER_DECODER)` mantem comportamento M6
- [ ] Para CTC: `TranscriptSegment(is_final=False)` do worker -> `transcript.partial` emitido imediatamente
- [ ] Para CTC: `TranscriptSegment(is_final=True)` do worker -> post-processing (ITN) -> `transcript.final`
- [ ] Para CTC: VAD speech_end ainda fecha o stream gRPC e emite `vad.speech_end`
- [ ] `_create_streaming_session()` le `manifest.capabilities.architecture` e passa ao `StreamingSession`
- [ ] O `if architecture` esta em exatamente 1 ponto de `_receive_worker_events()` (nao espalhado)
- [ ] Todos os testes de M5 e M6 (StreamingSession) continuam passando (regressao zero)
- [ ] Testes: >=15 testes (CTC partials nativos, CTC finals, CTC sem LocalAgreement, encoder-decoder mantido, transicoes de estado com CTC, force commit com CTC, recovery com CTC)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Viktor**: O ponto critico e o `_receive_worker_events()`. Para CTC, o loop e simples: `async for segment in stream: emit(segment)`. Para encoder-decoder, mantem a logica existente com LocalAgreement. A delegacao pode ser feita com um metodo strategy: `self._process_worker_segment(segment)` que faz dispatch. O VAD, state machine, ring buffer e WAL funcionam identicamente para ambas as arquiteturas -- apenas o tratamento dos eventos do worker muda.

**Perspectiva Sofia**: Resistir a tentacao de criar uma hierarquia de classes `CTC_StreamingSession` / `EncoderDecoder_StreamingSession`. Composicao, nao heranca. O `StreamingSession` e um orquestrador que delega -- a adaptacao e uma decisao interna, nao uma responsabilidade de subclasse. Um `if/elif` em um metodo e mais simples e legivel que uma hierarquia de classes para 2 casos.

---

### M7-06: Worker WeNet -- Subprocess gRPC End-to-End

**Epic**: E2 -- Pipeline Adaptativo
**Estimativa**: M (3-5 dias)
**Dependencias**: M7-01, M7-03
**Desbloqueia**: M7-07

**Contexto/Motivacao**: O worker WeNet e um subprocess gRPC identico ao worker Faster-Whisper, mas com `WeNetBackend` em vez de `FasterWhisperBackend`. O `STTWorkerServicer` ja aceita qualquer `STTBackend` -- nao precisa mudar. M7-06 valida que o worker WeNet inicia, responde health, transcreve batch e streaming via gRPC.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Worker WeNet startavel via `python -m theo.workers.stt --engine wenet` | Mudancas no `STTWorkerServicer` |
| Health check retorna `{"status": "ok", "model": "wenet-ctc", "engine": "wenet"}` | Mudancas no `WorkerManager` |
| `TranscribeFile` via gRPC funcional com `WeNetBackend` | |
| `TranscribeStream` via gRPC funcional com `WeNetBackend` | |
| `_build_worker_cmd()` suporta parametros especificos do WeNet (se houver) | |

**Entregaveis**:
- Alteracoes minimas em `src/theo/workers/stt/main.py` (factory ja alterada em M7-03)
- `tests/unit/test_wenet_worker.py` -- testes de integracao do worker gRPC com mock WeNet

**DoD**:
- [ ] `python -m theo.workers.stt --engine wenet --model-path /models/wenet-ctc --port 50052` inicia sem erro
- [ ] Health check via gRPC retorna `status: ok` com modelo e engine corretos
- [ ] `TranscribeFile` retorna `TranscribeFileResponse` com texto e segmentos
- [ ] `TranscribeStream` retorna stream de `TranscriptEvent` com partials e finals
- [ ] Worker aceita SIGTERM e encerra gracefully
- [ ] Testes: >=8 testes (spawn, health, batch, streaming, shutdown, erros)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Andre**: O `_build_worker_cmd()` pode precisar de flags diferentes para WeNet (ex: `--decoder-type ctc`). Verificar quais parametros do `engine_config` do manifesto WeNet precisam ser mapeados para flags CLI. Se WeNet nao precisa de flags extras, o `_build_worker_cmd()` existente funciona sem mudancas.

---

### M7-07: Testes Comparativos -- Contrato Identico

**Epic**: E3 -- Testes, Docs, Finalizacao
**Estimativa**: L (5-7 dias)
**Dependencias**: M7-05, M7-06
**Desbloqueia**: M7-09

**Contexto/Motivacao**: E a prova definitiva de model-agnostic: o mesmo request para ambos os backends deve retornar respostas com o mesmo contrato (mesmos campos, mesmos tipos, mesma estrutura). O texto pode diferir (engines diferentes produzem output diferente), mas o formato e identico.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Testes REST batch: mesmo audio, ambos os modelos, validar estrutura de resposta | Testes de qualidade (WER) -- nao e escopo de M7 |
| Testes WebSocket streaming: mesma sessao, ambos os modelos, validar contrato de eventos | Testes de performance (latencia comparativa) |
| Validar campos: `text` (string), `language` (string), `duration` (float), `segments` (array), `words` (array|null) | |
| Validar eventos WebSocket: `session.created`, `vad.speech_start`, `transcript.partial`, `transcript.final`, `vad.speech_end`, `session.closed` | |
| Validar que hot words funcionam em ambos (via initial_prompt para Whisper, nativo para WeNet) | |
| Validar que ITN e aplicado em `transcript.final` de ambos | |
| Validar formatos de resposta: json, verbose_json, text, srt, vtt -- todos funcionam com WeNet | |

**Entregaveis**:
- `tests/unit/test_contract_comparison.py` -- testes de contrato com mocks de ambos os backends
- `tests/integration/test_m7_e2e.py` -- testes end-to-end (se modelo WeNet disponivel)

**DoD**:
- [ ] Batch REST: `response_format=json` identico em campos e tipos para ambos
- [ ] Batch REST: `response_format=verbose_json` identico em estrutura (segments, words)
- [ ] Batch REST: `response_format=text`, `srt`, `vtt` funcionam com WeNet
- [ ] WebSocket: mesma sequencia de eventos para ambos (session.created -> vad.speech_start -> transcript.partial -> transcript.final -> vad.speech_end)
- [ ] WebSocket: `transcript.partial` com CTC tem TTFB menor (partial nativo, nao LocalAgreement)
- [ ] Hot words: ambos os backends melhoram transcricao de termos configurados
- [ ] ITN aplicado em `transcript.final` de ambos os backends
- [ ] Nenhum campo ausente ou extra entre backends
- [ ] Testes: >=20 testes (batch json, verbose_json, text, srt, vtt; streaming eventos; hot words; ITN; contrato de erro)
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

**Perspectiva Sofia**: Os testes devem ser parametrizados: `@pytest.mark.parametrize("engine", ["faster-whisper", "wenet"])`. Mesmo teste, duas engines. Se o teste falha para uma e passa para outra, o contrato esta inconsistente. E exatamente o valor de M7: forcar consistencia via testes.

**Perspectiva Viktor**: Testar especificamente que CTC partials sao nativos (emitidos frame-a-frame) enquanto encoder-decoder partials sao via LocalAgreement (emitidos apos comparacao entre passes). O tipo de evento e o mesmo (`transcript.partial`), mas a semantica de timing difere. Validar que para CTC, o primeiro partial chega rapidamente (antes de acumular 3-5s de audio).

---

### M7-08: Testes de Streaming Avancados com CTC

**Epic**: E3 -- Testes, Docs, Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M7-05
**Desbloqueia**: M7-09

**Contexto/Motivacao**: Streaming com CTC tem nuances que batch nao tem: partials nativos que podem ser frequentes (cada frame gera output), interacao com VAD, state machine, ring buffer, force commit e recovery. M7-08 testa essas interacoes especificamente para CTC.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| CTC + state machine: INIT -> ACTIVE -> SILENCE -> HOLD -> ACTIVE (funciona igual) | Testes de estabilidade 30 min (ja validado em M6, e o mesmo para CTC) |
| CTC + ring buffer: frames escritos, read fence avancado apos final | |
| CTC + force commit: buffer 90% com CTC dispara force commit | |
| CTC + recovery: crash de worker com CTC, retomada via WAL sem duplicacao | |
| CTC + backpressure: throttling funciona com CTC | |
| CTC + cross-segment context: enviado se `supports_initial_prompt: true` | |
| CTC sem LocalAgreement: verificar que `LocalAgreementPolicy` NAO e instanciada | |

**Entregaveis**:
- `tests/unit/test_streaming_session_ctc_advanced.py`

**DoD**:
- [ ] CTC + state machine: todas as transicoes funcionam identico ao encoder-decoder
- [ ] CTC + ring buffer: dados escritos e lidos corretamente com CTC
- [ ] CTC + force commit: dispara commit quando buffer atinge 90%
- [ ] CTC + recovery: crash -> WAL -> retomada sem duplicacao de segment_id
- [ ] CTC + backpressure: rate_limit e frames_dropped funcionam
- [ ] CTC sem LocalAgreement: `session._local_agreement` e `None` (ou equivalente)
- [ ] CTC + cross-segment: contexto enviado se engine suporta, ignorado se nao
- [ ] Testes: >=15 testes
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

### M7-09: Documentacao -- Como Adicionar Nova Engine STT

**Epic**: E3 -- Testes, Docs, Finalizacao
**Estimativa**: M (3-5 dias)
**Dependencias**: M7-07, M7-08
**Desbloqueia**: M7-10

**Contexto/Motivacao**: O valor de um runtime model-agnostic so se realiza se outros desenvolvedores conseguem adicionar engines sem ajuda dos autores originais. M7-09 documenta o processo passo-a-passo, validado por ter sido executado duas vezes (Faster-Whisper e WeNet).

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Documento `docs/ADDING_ENGINE.md` com passo-a-passo | Tutorial de como treinar modelos |
| Checklist de implementacao: interface, manifesto, factory, testes | Documentacao de API interna do WeNet |
| Exemplo concreto: WeNet como referencia | |
| Diagrama de pontos de extensao no runtime | |
| FAQ: perguntas comuns ao adicionar engine | |
| Atualizacao de `docs/ARCHITECTURE.md` com secao multi-engine | |

**Entregaveis**:
- `docs/ADDING_ENGINE.md` -- guia de extensibilidade
- Alteracao em `docs/ARCHITECTURE.md` -- secao atualizada

**DoD**:
- [ ] Documento cobre os 5 passos: (1) implementar STTBackend, (2) criar manifesto, (3) registrar factory, (4) declarar dependencia, (5) testes
- [ ] Cada passo tem exemplo de codigo concreto (baseado em WeNet)
- [ ] Checklist de DoD para cada passo (verificavel por qualquer dev)
- [ ] Diagrama ASCII dos pontos de extensao no runtime
- [ ] FAQ com pelo menos 5 perguntas comuns
- [ ] `ARCHITECTURE.md` atualizado com secao "Multi-Engine" e diagrama
- [ ] Revisado pelos 3 membros do time

---

### M7-10: Finalizacao -- CHANGELOG, ROADMAP, CI

**Epic**: E3 -- Testes, Docs, Finalizacao
**Estimativa**: S (1-2 dias)
**Dependencias**: M7-09
**Desbloqueia**: Nenhuma (leaf task, ultima do milestone)

**Contexto/Motivacao**: Atualizacao de documentacao de projeto e validacao final. CHANGELOG com entradas M7, ROADMAP com resultado, CLAUDE.md com novos componentes, e validacao de que CI esta verde com todos os testes.

**Escopo**:

| Incluido | Fora de escopo |
|----------|---------------|
| Atualizar `CHANGELOG.md` com entradas M7 | |
| Atualizar `docs/ROADMAP.md` com resultado M7 e checkpoint | |
| Atualizar `CLAUDE.md` com novos componentes e padroes M7 | |
| Verificar que todos os testes passam (M1-M7) | |
| Verificar que `make ci` esta verde | |
| Declarar optional dependency `[wenet]` no `pyproject.toml` | |

**Entregaveis**:
- Atualizacoes em `CHANGELOG.md`, `docs/ROADMAP.md`, `CLAUDE.md`
- Alteracao em `pyproject.toml` (optional dependency)

**DoD**:
- [ ] `CHANGELOG.md` com entradas M7 na secao `[Unreleased]`
- [ ] `ROADMAP.md` com resultado M7 e checkpoint `Fase 2 Completa`
- [ ] `CLAUDE.md` atualizado com: componentes M7, padroes de pipeline adaptativo, WeNetBackend
- [ ] `pyproject.toml` com `[project.optional-dependencies] wenet = [...]`
- [ ] `make ci` verde (format + lint + typecheck + testes)
- [ ] Total de testes novos M7: >=100
- [ ] Total acumulado (M1-M7): >=1138
- [ ] `mypy --strict` passa sem erros
- [ ] `ruff check` passa sem warnings

---

## 6. Grafo de Dependencias

```
M7-01 (WeNet Batch)
  |
  +---> M7-02 (WeNet Streaming) --------+
  |                                       |
  +---> M7-03 (Manifesto + Factory) -----+---> M7-06 (Worker gRPC E2E)
  |                                       |          |
  +---> M7-04 (Hot Words Nativo)          |          |
         |                                |          |
         +--------------------------------+----------+
                                          |
                                          v
                                   M7-05 (Pipeline Adaptativo)
                                     |
                                     +---> M7-07 (Testes Comparativos)
                                     |          |
                                     +---> M7-08 (Testes CTC Avancados)
                                               |
                                               v
                                        M7-09 (Documentacao)
                                               |
                                               v
                                        M7-10 (Finalizacao)
```

### Caminho critico

```
M7-01 -> M7-02 -> M7-05 -> M7-07 -> M7-09 -> M7-10
```

### Paralelismo maximo

- **Sprint 1**: M7-01 (WeNet batch) e unico. Ponto de partida.
- **Sprint 2**: M7-02 (streaming), M7-03 (manifesto), M7-04 (hot words) em paralelo apos M7-01.
- **Sprint 3**: M7-05 (pipeline adaptativo) e M7-06 (worker E2E) em paralelo apos Sprint 2.
- **Sprint 4**: M7-07, M7-08, M7-09, M7-10 sequenciais (testes dependem do pipeline adaptativo).

---

## 7. Sprint Plan

### Sprint 1 (Semana 1): WeNet Backend Core

**Objetivo**: `WeNetBackend` funcional com batch e streaming (isolado, com mocks).

**Demo Goal**: Unit tests passando com mock da API WeNet. `WeNetBackend` implementa toda a interface `STTBackend`.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M7-01 | WeNetBackend batch | L | Sofia |

**Checkpoint Sprint 1**:
- `WeNetBackend` implementa `STTBackend` completo
- `transcribe_file()` retorna `BatchResult` correto
- Import guardado funciona (sem WeNet instalado -> `ModelLoadError`)
- >=15 testes unitarios passando

### Sprint 2 (Semana 2): Streaming + Manifesto + Hot Words

**Objetivo**: Backend completo com streaming, manifesto registrado, hot words funcionais.

**Demo Goal**: Worker WeNet startavel. Manifesto parseavel. Hot words via keyword boosting.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M7-02 | WeNet streaming | M | Viktor |
| M7-03 | Manifesto + factory | S | Sofia |
| M7-04 | Hot words nativo | M | Sofia |

**Checkpoint Sprint 2**:
- `transcribe_stream()` yield partials nativos e finals
- Manifesto WeNet parseavel pelo `ModelManifest`
- Factory `_create_backend("wenet")` funciona
- Hot words via keyword boosting nativo
- >=35 testes novos

### Sprint 3 (Semana 3): Pipeline Adaptativo + Worker E2E

**Objetivo**: Runtime adapta fluxo por arquitetura. Worker WeNet funciona end-to-end via gRPC.

**Demo Goal**: `StreamingSession` com CTC pula LocalAgreement. Worker WeNet responde health, batch e streaming via gRPC.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M7-05 | Pipeline adaptativo | L | Viktor + Sofia |
| M7-06 | Worker WeNet E2E | M | Andre |

**Checkpoint Sprint 3**:
- `StreamingSession(architecture=CTC)` emite partials nativos sem LocalAgreement
- `StreamingSession(architecture=ENCODER_DECODER)` mantem comportamento M6
- Worker WeNet funciona via gRPC (health, batch, streaming)
- Todos os testes M1-M6 continuam passando (regressao zero)
- >=30 testes novos

### Sprint 4 (Semana 4): Testes + Docs + Finalizacao

**Objetivo**: Contrato identico validado. Documentacao de extensibilidade. Milestone completo.

**Demo Goal**: Demo completo conforme criterio de sucesso (secao 1). Dois backends, mesmo contrato, zero mudanca no cliente.

| ID | Task | Estimativa | Responsavel |
|----|------|-----------|-------------|
| M7-07 | Testes comparativos | L | Sofia + Viktor |
| M7-08 | Testes CTC avancados | M | Viktor |
| M7-09 | Documentacao | M | Andre + Sofia |
| M7-10 | Finalizacao | S | Andre |

**Checkpoint Sprint 4**:
- Contrato REST identico para ambos os backends (parametrizado)
- Contrato WebSocket identico (mesma sequencia de eventos)
- `docs/ADDING_ENGINE.md` completo e revisado
- `make ci` verde com >=1138 testes
- CHANGELOG, ROADMAP, CLAUDE.md atualizados

---

## 8. Estrutura de Arquivos (M7)

```
src/theo/
  workers/
    stt/
      wenet.py                          # WeNetBackend (NOVO)           [M7-01, M7-02, M7-04]
      main.py                           # Factory _create_backend       (ALTERADO) [M7-03]
      interface.py                      # STTBackend ABC                (SEM MUDANCA)
      faster_whisper.py                 # FasterWhisperBackend          (SEM MUDANCA)
      servicer.py                       # STTWorkerServicer             (SEM MUDANCA)
      converters.py                     # Proto <-> dominio             (SEM MUDANCA)
    manager.py                          # WorkerManager                 (SEM MUDANCA)

  session/
    streaming.py                        # StreamingSession              (ALTERADO) [M7-05, M7-04]
    local_agreement.py                  # LocalAgreementPolicy          (SEM MUDANCA)
    state_machine.py                    # SessionStateMachine           (SEM MUDANCA)
    ring_buffer.py                      # RingBuffer                    (SEM MUDANCA)
    wal.py                              # SessionWAL                    (SEM MUDANCA)
    cross_segment.py                    # CrossSegmentContext           (SEM MUDANCA)
    backpressure.py                     # BackpressureController        (SEM MUDANCA)
    metrics.py                          # Metricas                      (SEM MUDANCA)

  server/
    routes/
      realtime.py                       # Factory _create_streaming_session (ALTERADO) [M7-05]
      transcriptions.py                 # POST /v1/audio/transcriptions (SEM MUDANCA)

  config/
    manifest.py                         # ModelManifest                 (SEM MUDANCA)

  registry/
    registry.py                         # ModelRegistry                 (SEM MUDANCA)

docs/
  ADDING_ENGINE.md                      # Guia de extensibilidade       (NOVO) [M7-09]
  ARCHITECTURE.md                       # Secao multi-engine            (ALTERADO) [M7-09]
  ROADMAP.md                            # Resultado M7                  (ALTERADO) [M7-10]

tests/
  unit/
    test_wenet_backend.py               # [M7-01, M7-02]  NOVO
    test_wenet_manifest.py              # [M7-03]          NOVO
    test_wenet_hot_words.py             # [M7-04]          NOVO
    test_streaming_session_ctc.py       # [M7-05]          NOVO
    test_wenet_worker.py                # [M7-06]          NOVO
    test_contract_comparison.py         # [M7-07]          NOVO
    test_streaming_session_ctc_advanced.py # [M7-08]       NOVO
  integration/
    test_m7_e2e.py                      # [M7-07]          NOVO
  fixtures/
    wenet-ctc/
      theo.yaml                         # [M7-03]          NOVO
```

### Contagem de arquivos impactados

| Tipo | Novos | Alterados | Inalterados |
|------|-------|-----------|-------------|
| Source | 1 (`wenet.py`) | 3 (`main.py`, `streaming.py`, `realtime.py`) | 15+ |
| Docs | 1 (`ADDING_ENGINE.md`) | 3 (`ARCHITECTURE.md`, `ROADMAP.md`, `CHANGELOG.md`) | -- |
| Tests | 8 arquivos | 0 | -- |
| Config | 0 | 1 (`pyproject.toml`) | -- |

**Observacao**: A maioria esmagadora do codebase permanece inalterada. Isso e a prova arquitetural de que o design e extensivel: adicionar uma nova engine toca 4 arquivos de producao (1 novo + 3 alterados).

---

## 9. Checkpoints de Validacao

### Checkpoint 1 (apos Sprint 1): Backend Isolado

```
Validacao:
  [ ] WeNetBackend implementa STTBackend completo
  [ ] transcribe_file() retorna BatchResult correto
  [ ] Import guardado funciona
  [ ] >=15 testes passando com mock WeNet
```

### Checkpoint 2 (apos Sprint 2): Engine Completa

```
Validacao:
  [ ] transcribe_stream() yield partials nativos
  [ ] Manifesto theo.yaml parseavel
  [ ] Factory _create_backend("wenet") funciona
  [ ] Hot words via keyword boosting nativo
  [ ] Worker WeNet startavel via CLI
```

### Checkpoint 3 (apos Sprint 3): Pipeline Integrado

```
Validacao:
  [ ] StreamingSession adapta por arquitetura
  [ ] CTC: partials nativos sem LocalAgreement
  [ ] Encoder-decoder: comportamento M6 mantido
  [ ] Worker WeNet funciona E2E via gRPC
  [ ] Regressao zero em M1-M6
```

### Checkpoint 4 (apos Sprint 4): Milestone Completo

```
Validacao:
  [ ] Contrato REST identico para ambos os backends
  [ ] Contrato WebSocket identico
  [ ] Hot words funcionam em ambos (mecanismos diferentes)
  [ ] ITN aplicado em ambos
  [ ] Documentacao de extensibilidade completa
  [ ] >=1138 testes passando
  [ ] make ci verde
  [ ] CHANGELOG, ROADMAP, CLAUDE.md atualizados
```

---

## 10. Criterios de Saida do M7

### Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `WeNetBackend` implementa `STTBackend` completo (batch + streaming) | Testes unitarios |
| 2 | Manifesto `theo.yaml` para WeNet CTC parseavel pelo `ModelManifest` | Testes unitarios |
| 3 | Worker WeNet startavel como subprocess gRPC | Teste de integracao |
| 4 | `_create_backend("wenet")` retorna `WeNetBackend` | Teste unitario |
| 5 | Pipeline adaptativo: CTC usa partials nativos (sem LocalAgreement) | Testes unitarios |
| 6 | Pipeline adaptativo: encoder-decoder mantem LocalAgreement | Regressao testes M6 |
| 7 | Contrato REST identico para ambos os backends | Testes comparativos |
| 8 | Contrato WebSocket identico para ambos os backends | Testes comparativos |
| 9 | Hot words: WeNet via keyword boosting nativo | Testes unitarios |
| 10 | Hot words: Whisper via initial_prompt (inalterado) | Regressao testes M6 |
| 11 | ITN aplicado em `transcript.final` de ambos | Testes comparativos |
| 12 | Formatos de resposta (json, verbose_json, text, srt, vtt) funcionam com WeNet | Testes comparativos |
| 13 | Recovery funciona com CTC (crash -> WAL -> retomada) | Testes de streaming avancados |
| 14 | Documentacao `ADDING_ENGINE.md` completa e revisada | Review por 3 membros |
| 15 | Optional dependency `[wenet]` declarada no `pyproject.toml` | Inspecao |

### Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `make check` |
| 2 | ruff check sem warnings | `make check` |
| 3 | ruff format sem diffs | `make check` |
| 4 | Todos os testes passam (M1-M7) | `make test-unit` |
| 5 | CI verde | GitHub Actions |

### Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | WeNetBackend (batch + streaming) | 25 |
| Unit | Manifesto + factory | 8 |
| Unit | Hot words nativo | 8 |
| Unit | StreamingSession CTC (pipeline adaptativo) | 15 |
| Unit | Worker WeNet gRPC | 8 |
| Unit | Contrato comparativo | 20 |
| Unit | Streaming CTC avancado (state machine, ring buffer, recovery) | 15 |
| Integration | End-to-end (se modelo disponivel) | 5 |
| **Total novos M7** | | **>=100** |
| **Total acumulado (M1-M7)** | | **>=1138** |

---

## 11. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Interface `STTBackend` nao acomoda particularidades do WeNet (ex: decoder_type, context manager) | Media | Alto | Se isso acontecer, ajustar a ABC agora -- M7 e o momento certo. Preferir adicionar parametros opcionais ao `**kwargs` antes de mudar assinatura. |
| R2 | WeNet tem dependencia de LibTorch que conflita com CTranslate2 (CUDA symbols) | Media | Medio | Isolamento por subprocess resolve conflitos de runtime. Testar coexistencia no mesmo container Docker. Se incompativel em nivel de shared libs, documentar como limitacao (workers em containers separados). |
| R3 | WeNet streaming tem semantica diferente do esperado (partials nao sao frame-by-frame, ou precisam de chunking especifico) | Media | Medio | Estudar API WeNet antes de implementar. Se partials nao sao frame-by-frame, adaptar `transcribe_stream()` para acumular internamente (como Faster-Whisper faz). A interface `STTBackend` suporta ambos os padroes. |
| R4 | API Python do WeNet e instavel ou mal documentada | Media | Medio | Usar versao pinada do WeNet. Encapsular toda interacao com WeNet em funcoes auxiliares dentro de `wenet.py`. Se API mudar, so `wenet.py` e afetado. |
| R5 | `StreamingSession` com `if architecture` se espalha por mais de 1 ponto | Baixa | Alto | Code review rigoroso. Regra: `if architecture` em exatamente 1 metodo de `StreamingSession`. Se precisar de mais, refatorar para strategy pattern (composicao). |
| R6 | Regressao em testes M5/M6 ao modificar `StreamingSession` | Media | Alto | Rodar `make test-unit` apos cada alteracao em `streaming.py`. O parametro `architecture` e opcional com default `ENCODER_DECODER` -- comportamento existente e preservado por default. |
| R7 | WeNet nao suporta `initial_prompt` para cross-segment context | Alta | Baixo | `EngineCapabilities.supports_initial_prompt` ja existe. Se `False`, `CrossSegmentContext` e ignorado (comportamento ja implementado em M6). |

---

## 12. Out of Scope (explicitamente NAO esta em M7)

| Item | Milestone | Justificativa |
|------|-----------|---------------|
| `theo pull wenet-ctc` (download automatico de modelo) | Futuro | Registry local ainda nao tem download. Modelo instalado manualmente. |
| Paraformer (`streaming-native`) | Futuro | M7 valida CTC. Paraformer e a terceira arquitetura -- sera trivial apos M7. |
| Testes de performance comparativa (WER, latencia) | Futuro | M7 valida contrato, nao qualidade. Performance e cenario de benchmark separado. |
| Dynamic batching para WeNet | M9 | Otimizacao de throughput, fora de escopo de model-agnostic. |
| Metricas por engine/architecture | Futuro | Metricas atuais sao globais. Separar por engine e refinamento futuro. |
| RTP Listener | M8 | Telefonia e Fase 3. |
| Hot Word Correction post-processing (Levenshtein) | Futuro | Domain-specific, nao necessario para validacao model-agnostic. |
| Entity Formatting (CPF, CNPJ) | Futuro | Domain-specific. |

---

## 13. Transicao M7 -> M8

Ao completar M7, o time deve ter:

1. **Model-agnostic validado** -- duas engines fundamentalmente diferentes (encoder-decoder e CTC) funcionam com o mesmo runtime, mesmo contrato, mesmo protocolo.

2. **Pipeline adaptativo funcional** -- o runtime adapta o fluxo de streaming baseado na arquitetura declarada no manifesto. Ponto de decisao unico e limpo.

3. **Documentacao de extensibilidade** -- qualquer dev pode adicionar uma terceira engine seguindo o guia `ADDING_ENGINE.md`.

4. **Pontos de extensao para M8**:
   - **RTP Listener**: cria sessao via `StreamingSession` (mesmo contrato que WebSocket). Ring Buffer e Session Manager reutilizados.
   - **Denoise**: adicionado como stage no preprocessing pipeline. Ativado por default para audio RTP.
   - **Audio quality tagging**: `audio_quality: telephony` informado ao registry para selecionar modelos adequados.

5. **Pontos de extensao para M9**:
   - **Scheduler avancado**: priorizacao realtime > batch. Ambos os tipos de engine sao candidatos.
   - **Dynamic batching**: aplicavel a ambos os backends que suportam (`batch_inference: true` no manifesto).

**O primeiro commit de M8 sera**: implementar `RTPListener` que recebe pacotes UDP, decodifica G.711, e alimenta o preprocessing pipeline existente.

---

## 14. Perspectivas Cruzadas

### Onde as Perspectivas se Encontram

| Intersecao | Sofia (Arquitetura) | Viktor (Real-Time) | Andre (Platform) |
|------------|--------------------|--------------------|------------------|
| **Pipeline adaptativo** | Design do ponto de decisao (1 if/elif) | Implementacao no `_receive_worker_events()`, garantir que CTC nao acumula | Testes de regressao, CI |
| **Isolamento de deps** | Interface `STTBackend` garante desacoplamento | gRPC proto identico para ambos | `pyproject.toml` extras, Docker multi-stage se necessario |
| **Hot words** | Decisao por capability (nao por arquitetura) | Integracao no fluxo de streaming | Testes comparativos |
| **Contrato identico** | Validacao de tipos e estrutura | Validacao de timing (TTFB CTC vs encoder-decoder) | Testes parametrizados, CI |

### Consenso do Time

- **Sofia**: M7 e um teste de validacao da abstracão, nao um milestone de features. Se `STTBackend` funciona para CTC, funciona para qualquer coisa. A simplicidade do impacto (1 arquivo novo, 3 alterados) e a prova de que o design de M1-M6 estava correto.

- **Viktor**: O ponto critico e que CTC partials sejam realmente nativos -- emitidos frame-a-frame, sem acumulacao. Se o `WeNetBackend.transcribe_stream()` acumula internamente (como o Faster-Whisper faz), perdemos o diferencial de latencia de CTC. Testar TTFB do primeiro partial.

- **Andre**: O risco maior e dependencias conflitantes em CI. LibTorch e CTranslate2 no mesmo container pode dar problema com CUDA symbols. Testar coexistencia cedo (Sprint 2). Se incompativel, documentar e usar containers separados por engine (nao e bloqueio -- subprocess ja isola).

---

*Documento gerado pelo Time de Arquitetura (ARCH) -- Sofia Castellani, Viktor Sorokin, Andre Oliveira. Sera atualizado conforme a implementacao do M7 avanca.*
