# Theo OpenVoice -- Plano Estrategico do Milestone M3

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Aprovado para implementacao
**Data**: 2026-02-07

**Autora**: Sofia Castellani (Principal Solution Architect)

---

## 1. Contexto e Objetivo

M3 e o milestone que conecta o Theo ao usuario. Ate agora, temos tipos (M1) e um worker gRPC funcional (M2) -- mas nenhum ser humano consegue usar o sistema. M3 entrega a primeira experiencia end-to-end: um usuario executa `theo serve`, faz um `curl` com arquivo de audio, e recebe texto transcrito.

### O que ja existe (M1 + M2)

```
src/theo/
  _types.py              -> STTArchitecture, BatchResult, ResponseFormat, etc.
  exceptions.py          -> ModelNotFoundError, AudioFormatError, etc.
  logging.py             -> structlog (JSON + console)
  config/manifest.py     -> ModelManifest (parsing de theo.yaml)
  workers/manager.py     -> WorkerManager (spawn, health, restart)
  workers/stt/
    interface.py         -> STTBackend ABC
    faster_whisper.py    -> FasterWhisperBackend (transcribe_file)
    servicer.py          -> STTWorkerServicer (gRPC TranscribeFile + Health)
    main.py              -> Entry point do worker subprocess
    converters.py        -> Proto <-> Theo converters
  proto/
    stt_worker.proto     -> Contrato gRPC
    stt_worker_pb2.py    -> Stubs gerados
```

### O que M3 entrega

```
ANTES (M2):  Worker gRPC isolado, acessivel apenas via grpcurl
DEPOIS (M3): curl -F file=@audio.wav http://localhost:8000/v1/audio/transcriptions
             theo transcribe audio.wav
```

### Principio do M3

> **Vertical slice completa: do HTTP request ao texto transcrito.** Cada task deve resultar em algo executavel. A primeira task entregavel deve ser um health check respondendo 200. A ultima deve ser o SDK OpenAI Python funcionando como cliente.

---

## 2. Decisoes de Design

### 2.1 Componentes e Responsabilidades

M3 introduz 4 novos componentes. Cada um com responsabilidade clara e interface minima.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FLUXO M3 (Batch)                             │
│                                                                  │
│  curl/SDK ──> FastAPI App ──> Scheduler ──> gRPC Worker          │
│                  │                │              │                │
│                  │                │              ▼                │
│                  │                │      FasterWhisperBackend     │
│                  │                │              │                │
│                  │                ▼              │                │
│                  │          Registry             │                │
│                  │       (resolve model)          │                │
│                  │                               │                │
│                  ◄───────────────────────────────┘                │
│                  │                                                │
│                  ▼                                                │
│          Response Formatter                                      │
│    (json, verbose_json, text, srt, vtt)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Registry (`src/theo/registry/registry.py`)

**Responsabilidade**: Descobrir modelos instalados em disco e fornecer seus manifestos.

**Interface proposta**:

```python
class ModelRegistry:
    def __init__(self, models_dir: str | Path) -> None: ...
    async def scan(self) -> None: ...
    def get_manifest(self, model_name: str) -> ModelManifest: ...
    def list_models(self) -> list[ModelManifest]: ...
    def has_model(self, model_name: str) -> bool: ...
```

**O que NAO faz em M3**:
- Download de modelos (`theo pull` e M futuro)
- Load/unload de modelos em memoria (quem faz e o Scheduler/WorkerManager)
- Cache de modelos carregados
- Eviction de modelos por memoria

**Integracao com o existente**:
- Usa `ModelManifest.from_yaml_path()` (ja existe em `config/manifest.py`)
- Levanta `ModelNotFoundError` (ja existe em `exceptions.py`)

**Convencao de disco**: O Registry escaneia um diretorio `models_dir` (default: `~/.theo/models/`). Cada modelo e um subdiretorio contendo `theo.yaml`:

```
~/.theo/models/
  faster-whisper-tiny/
    theo.yaml
    model.bin (ou qualquer artefato da engine)
  faster-whisper-large-v3/
    theo.yaml
    ...
```

O Registry so le o `theo.yaml`. Nao se preocupa com os artefatos da engine -- quem sabe o que fazer com eles e o worker (via `model_path`).

#### Scheduler (`src/theo/scheduler/scheduler.py`)

**Responsabilidade**: Rotear uma request de transcricao para um worker disponivel.

**Interface proposta**:

```python
class TranscribeRequest:
    """Request interna de transcricao (nao confundir com HTTP request)."""
    request_id: str
    model_name: str
    audio_data: bytes
    language: str | None
    response_format: ResponseFormat
    temperature: float
    timestamp_granularities: list[str]
    initial_prompt: str | None
    hot_words: list[str] | None

class Scheduler:
    def __init__(self, worker_manager: WorkerManager) -> None: ...
    async def transcribe(self, request: TranscribeRequest) -> BatchResult: ...
```

**O que faz em M3** (trivial):
1. Recebe request
2. Encontra worker READY para o modelo (via `WorkerManager.get_ready_worker`)
3. Envia gRPC `TranscribeFile` ao worker
4. Retorna `BatchResult`

**O que NAO faz em M3**:
- Priorizacao (realtime > batch) -- M9
- Fila de espera -- M9
- Dynamic batching -- M9
- Cancelamento -- M5/M9

**Por que o Scheduler existe em M3 se e trivial?** Para estabelecer o contrato. O API Server nunca fala diretamente com o WorkerManager ou com gRPC. Sempre passa pelo Scheduler. Em M9, o Scheduler evolui para priorizacao e fila -- sem mudar o API Server.

**Integracao com o existente**:
- Usa `WorkerManager.get_ready_worker()` (ja existe)
- Cria `STTWorkerStub` e chama `TranscribeFile` (stubs gRPC ja gerados)
- Usa `proto_request_to_transcribe_params` e `batch_result_to_proto_response` indiretamente (a comunicacao gRPC e feita via proto)
- Levanta `WorkerUnavailableError` se nenhum worker READY (ja existe)

#### API Server (`src/theo/server/`)

**Responsabilidade**: Receber requests HTTP, validar, delegar ao Scheduler, formatar resposta.

**Estrutura**:

```
src/theo/server/
  __init__.py
  app.py                -> create_app() factory
  dependencies.py       -> FastAPI dependencies (registry, scheduler)
  routes/
    __init__.py
    transcriptions.py   -> POST /v1/audio/transcriptions
    translations.py     -> POST /v1/audio/translations
    health.py           -> GET /health, GET /metrics
  models/
    __init__.py
    requests.py         -> TranscriptionRequest, TranslationRequest (Pydantic)
    responses.py        -> TranscriptionResponse, VerboseResponse, etc.
  formatters.py         -> BatchResult -> json/text/srt/vtt
```

**Contrato OpenAI**:

O API Server deve ser compativel com o SDK oficial `openai` Python. Isso significa:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
result = client.audio.transcriptions.create(
    model="faster-whisper-tiny",
    file=open("audio.wav", "rb"),
)
print(result.text)
```

**Campos do multipart/form-data (request)**:

| Campo | Tipo | Obrigatorio | Default |
|-------|------|-------------|---------|
| `file` | UploadFile | sim | -- |
| `model` | str | sim | -- |
| `language` | str | nao | (auto-detect) |
| `prompt` | str | nao | -- |
| `response_format` | str | nao | `"json"` |
| `temperature` | float | nao | `0.0` |
| `timestamp_granularities[]` | list[str] | nao | `["segment"]` |

**Nota**: O campo `hot_words` NAO existe na API OpenAI. Em M3 nao adicionamos campos extras -- mantemos compatibilidade estrita. Hot words via API serao avaliados em milestone futuro.

**Response format**:

| Format | Content-Type | Corpo |
|--------|-------------|-------|
| `json` | application/json | `{"text": "..."}` |
| `verbose_json` | application/json | `{"task":"transcribe","language":"pt","duration":2.5,"text":"...","segments":[...],"words":[...]}` |
| `text` | text/plain | Texto cru |
| `srt` | text/plain | Legenda SRT |
| `vtt` | text/plain | Legenda WebVTT |

**Error handling HTTP**:

| Status | Quando | Exception mapeada |
|--------|--------|-------------------|
| 400 | Formato de audio invalido, campo obrigatorio faltando | `AudioFormatError`, `ValidationError` |
| 404 | Modelo nao encontrado no registry | `ModelNotFoundError` |
| 413 | Arquivo excede 25MB | `AudioTooLargeError` |
| 503 | Worker nao disponivel (modelo carregando ou crashed) | `WorkerUnavailableError` |

**Integracao com o existente**:
- Valida request com Pydantic
- Delega ao Scheduler
- Usa exceptions do `theo.exceptions` mapeadas para HTTP status codes
- Usa `ResponseFormat` enum de `theo._types`
- Structlog para logging de requests

#### CLI (`src/theo/cli/`)

**Responsabilidade**: Interface de terminal para o usuario.

**Estrutura**:

```
src/theo/cli/
  __init__.py
  main.py          -> Entry point (click ou typer)
  serve.py         -> theo serve
  transcribe.py    -> theo transcribe <file>
  translate.py     -> theo translate <file>
  models.py        -> theo list, theo inspect <model>
```

**Dependencia**: `click` (library madura, zero dependencias transitivas pesadas). Nao usar `typer` -- typer depende de click + rich + typing_extensions e adiciona complexidade desnecessaria para o que precisamos. KISS.

**Nota sobre pyproject.toml**: Adicionar `click>=8.1,<9.0` ao core dependencies (nao como extra) porque o CLI e parte fundamental do runtime. O entry point `theo` deve ser declarado em `[project.scripts]`.

**Comandos M3**:

| Comando | O que faz |
|---------|-----------|
| `theo serve` | Inicia API Server (uvicorn) + spawna workers para modelos instalados |
| `theo transcribe <file>` | Envia arquivo para `POST /v1/audio/transcriptions` (requer `theo serve` rodando) |
| `theo translate <file>` | Envia arquivo para `POST /v1/audio/translations` |
| `theo list` | Lista modelos instalados (le manifestos do disco, nao precisa de server) |
| `theo inspect <model>` | Mostra detalhes de um modelo (manifesto completo) |

**`theo serve` -- o que faz internamente**:

1. Configura logging
2. Cria `ModelRegistry`, escaneia `models_dir`
3. Para cada modelo STT encontrado:
   a. Cria `WorkerManager`
   b. Spawna worker subprocess gRPC
   c. Aguarda health check
4. Cria `Scheduler` com `WorkerManager`
5. Cria FastAPI app com `Registry` e `Scheduler`
6. Inicia uvicorn

**`theo transcribe` -- o que faz internamente**:

1. Le o arquivo
2. Faz HTTP POST para `http://localhost:8000/v1/audio/transcriptions`
3. Imprime o resultado

**O CLI nao reimplementa logica**. Ele e um thin client HTTP para o server (exceto `theo list` e `theo inspect` que leem disco diretamente).

### 2.2 O que NAO fazer (fronteiras YAGNI)

| Item | Por que nao | Quando |
|------|-------------|--------|
| Preprocessing de audio | M4. Em M3, o audio vai raw para o worker. | M4 |
| Post-processing (ITN, entity formatting) | M4. Em M3, o texto vem cru da engine. | M4 |
| WebSocket `/v1/realtime` | M5. M3 e batch-only. | M5 |
| `theo pull` (download de modelos) | Registry futuro. Em M3, modelos ja estao em disco. | Futuro |
| Dynamic batching | M9. Em M3, 1 request = 1 chamada gRPC. | M9 |
| Metricas Prometheus detalhadas | Basico em M3: `requests_total`, `request_duration_seconds`. Metricas STT detalhadas em M5+. | M5+ |
| Autenticacao/API keys | Nao objetivo do projeto. | N/A |
| Hot words via REST API | Nao existe na API OpenAI. Avaliar em milestone futuro. | Futuro |
| Multipart form parsing customizado | FastAPI/python-multipart ja resolvem. Nao reinventar. | N/A |

---

## 3. Epics e Tasks

### Grafo de Dependencias

```
E1-T1 (health endpoint)
  └──> E1-T2 (Pydantic models)
        └──> E1-T3 (transcriptions endpoint)
              └──> E1-T4 (translations endpoint)
                    └──> E1-T5 (response formatters)

E2-T1 (Registry)
  └──> E3-T1 (Scheduler)  [+ E1-T3]
        └──> E4-T1 (theo serve)  [+ E1-T3]
              └──> E4-T2 (theo transcribe/translate)
                    └──> E4-T3 (theo list/inspect)  [+ E2-T1]

E5-T1 (error handling HTTP)  [+ E1-T3]
E5-T2 (testes e2e)           [+ E4-T1]
E5-T3 (compatibilidade OpenAI SDK)  [+ E5-T2]
```

### Sequencia de implementacao recomendada

A ordem maximiza valor incremental. Apos cada task, algo novo funciona:

```
Fase A -- Vertical Slice (algo respondendo HTTP)
  1. E1-T1  Health endpoint (GET /health responde 200)
  2. E1-T2  Pydantic models (request/response validaveis)
  3. E2-T1  Model Registry (encontra modelos em disco)
  4. E3-T1  Scheduler (roteia request para worker gRPC)
  5. E1-T3  POST /v1/audio/transcriptions (fluxo completo!)

Fase B -- Completude da API
  6. E1-T4  POST /v1/audio/translations
  7. E1-T5  Response formatters (srt, vtt, text, verbose_json)
  8. E5-T1  Error handling HTTP (400, 404, 413, 503)

Fase C -- CLI
  9.  E4-T1  theo serve (inicia server + workers)
  10. E4-T2  theo transcribe / theo translate
  11. E4-T3  theo list / theo inspect

Fase D -- Validacao
  12. E5-T2  Testes end-to-end
  13. E5-T3  Compatibilidade com SDK OpenAI
```

---

### Epic 1 -- API Server (FastAPI)

#### E1-T1: Health Endpoint

**Contexto/Motivacao**: Primeiro artefato executavel de M3. Um `GET /health` respondendo 200 prova que FastAPI esta configurado, uvicorn roda, e o app factory funciona. E tambem pre-requisito para health checks em Docker e Kubernetes.

**Escopo**:

| In | Out |
|----|-----|
| `GET /health` retornando `{"status": "ok"}` | Metricas Prometheus detalhadas |
| `GET /metrics` com contadores basicos (placeholder) | Integracao com Registry/Scheduler |
| `create_app()` factory em `server/app.py` | WebSocket |
| Teste unitario do endpoint | Load balancer probes |

**Entregaveis**:
- `src/theo/server/app.py` -- `create_app()` que retorna FastAPI instance
- `src/theo/server/routes/health.py` -- `GET /health`
- Teste: `tests/unit/test_health_endpoint.py`

**Definition of Done**:
- `python -c "from theo.server.app import create_app; app = create_app()"` funciona
- Teste com `TestClient`: `GET /health` retorna 200 com `{"status": "ok"}`
- mypy e ruff passam

**Dependencias**: Nenhuma task de M3 (apenas M2 completo)

**Estimativa**: 0.5 dia

---

#### E1-T2: Pydantic Models (Request/Response)

**Contexto/Motivacao**: FastAPI usa Pydantic para validacao de request e serializacao de response. Definir os modelos antes dos endpoints garante que o contrato de API esta estavel antes de implementar logica.

**Escopo**:

| In | Out |
|----|-----|
| `TranscriptionRequest` com todos os campos do multipart | Modelos de WebSocket |
| `TranscriptionResponse` (json format) | Modelos de session/streaming |
| `VerboseTranscriptionResponse` (verbose_json format) | |
| `ErrorResponse` para erros HTTP | |
| Validacao de `response_format` contra enum `ResponseFormat` | |
| Validacao de `temperature` (0.0-1.0) | |

**Entregaveis**:
- `src/theo/server/models/requests.py`
- `src/theo/server/models/responses.py`
- Teste: `tests/unit/test_api_models.py`

**Nota sobre multipart no FastAPI**: FastAPI nao usa Pydantic models para multipart/form-data. Os campos sao declarados como parametros da funcao da rota usando `Form()` e `UploadFile`. O Pydantic model `TranscriptionRequest` e um dataclass interno de transporte (nao decorado com FastAPI) -- recebe os valores ja validados da rota e os passa ao Scheduler.

**Definition of Done**:
- Pydantic model valida campos corretamente (rejeita temperature > 1.0, response_format invalido)
- Response models geram JSON compativel com contrato OpenAI
- mypy e ruff passam

**Dependencias**: E1-T1 (app factory existe)

**Estimativa**: 1 dia

---

#### E1-T3: POST /v1/audio/transcriptions

**Contexto/Motivacao**: O endpoint principal do M3. Recebe arquivo de audio, delega transcricao, retorna texto. E a razao de existir do milestone.

**Escopo**:

| In | Out |
|----|-----|
| Endpoint `POST /v1/audio/transcriptions` | POST /v1/audio/translations (E1-T4) |
| Aceita multipart/form-data com `file` e `model` | Preprocessing de audio |
| Le bytes do upload file | Post-processing (ITN) |
| Delega ao Scheduler | Formatos srt/vtt (E1-T5) |
| Retorna JSON `{"text": "..."}` (format json) | Hot words |
| Logging de request/response | |

**Entregaveis**:
- `src/theo/server/routes/transcriptions.py`
- `src/theo/server/dependencies.py` -- injecao do Registry e Scheduler
- Teste: `tests/unit/test_transcriptions_route.py` (com mock do Scheduler)

**Fluxo interno**:

```
1. FastAPI recebe multipart (file, model, language, ...)
2. Valida campos (model obrigatorio, file obrigatorio)
3. Verifica tamanho do arquivo (< 25MB)
4. Cria TranscribeRequest com request_id (uuid4)
5. Chama scheduler.transcribe(request)
6. Scheduler:
   a. Encontra worker READY para o modelo
   b. Envia gRPC TranscribeFile ao worker
   c. Recebe TranscribeFileResponse
   d. Converte proto -> BatchResult
7. Rota formata BatchResult -> JSON response
8. Retorna 200 com response
```

**Definition of Done**:
- Teste unitario: rota com mock de Scheduler retorna JSON correto
- Campos `file` e `model` sao obrigatorios (400 sem eles)
- `request_id` e gerado e logado
- mypy e ruff passam

**Dependencias**: E1-T2 (modelos Pydantic), E2-T1 (Registry), E3-T1 (Scheduler)

**Estimativa**: 1.5 dias

---

#### E1-T4: POST /v1/audio/translations

**Contexto/Motivacao**: Endpoint de traducao para ingles. Contrato identico ao de transcricao, com `task: "translate"` passado ao worker.

**Escopo**:

| In | Out |
|----|-----|
| Endpoint `POST /v1/audio/translations` | Multiplos idiomas de destino |
| Mesmo contrato multipart que transcriptions | |
| Passa `task="translate"` ao gRPC | |

**Entregaveis**:
- `src/theo/server/routes/translations.py`
- Teste: `tests/unit/test_translations_route.py`

**Nota**: A implementacao e quase identica a transcriptions. A diferenca e que o campo `language` no `TranscribeFileRequest` indica o idioma de ENTRADA, e o output e sempre ingles. O Faster-Whisper suporta isso nativamente via `task="translate"`.

**Decisao de design**: Nao criar abstracacao prematura entre transcriptions e translations. Duplicar a rota (com a diferenca do `task`) e mais simples e claro do que criar uma generalizacao agora. Se em M4+ a duplicacao causar problemas, refatorar. Regra de 3 do DRY.

**Definition of Done**:
- Teste unitario: rota retorna traducao em ingles
- Contrato de response identico ao de transcriptions
- mypy e ruff passam

**Dependencias**: E1-T3 (transcriptions funciona)

**Estimativa**: 0.5 dia

---

#### E1-T5: Response Formatters

**Contexto/Motivacao**: A API OpenAI suporta 5 formatos de resposta. Em M3, `json` ja funciona via E1-T3. Esta task adiciona os outros 4.

**Escopo**:

| In | Out |
|----|-----|
| Formatter `verbose_json` (segments + words) | Formatos customizados |
| Formatter `text` (texto cru) | |
| Formatter `srt` (legenda SubRip) | |
| Formatter `vtt` (legenda WebVTT) | |

**Entregaveis**:
- `src/theo/server/formatters.py`
- Teste: `tests/unit/test_formatters.py`

**Conversao `BatchResult` -> formato**:

```python
def format_response(result: BatchResult, format: ResponseFormat) -> ...:
    match format:
        case ResponseFormat.JSON:
            return {"text": result.text}
        case ResponseFormat.VERBOSE_JSON:
            return {
                "task": "transcribe",
                "language": result.language,
                "duration": result.duration,
                "text": result.text,
                "segments": [...],
                "words": [...] or None,
            }
        case ResponseFormat.TEXT:
            return result.text  # plain text
        case ResponseFormat.SRT:
            return _to_srt(result.segments)
        case ResponseFormat.VTT:
            return _to_vtt(result.segments)
```

**Formato SRT**:

```
1
00:00:00,000 --> 00:00:02,500
Ola, como posso ajudar?

2
00:00:03,000 --> 00:00:05,200
Preciso de informacoes.
```

**Formato VTT**:

```
WEBVTT

00:00:00.000 --> 00:00:02.500
Ola, como posso ajudar?

00:00:03.000 --> 00:00:05.200
Preciso de informacoes.
```

**Definition of Done**:
- Testes unitarios para cada formato com `BatchResult` de exemplo
- SRT e VTT geram output valido (timestamps corretos, separadores corretos)
- `verbose_json` inclui `segments` e `words` (quando disponivel)
- mypy e ruff passam

**Dependencias**: E1-T3 (endpoint funciona com json)

**Estimativa**: 1 dia

---

### Epic 2 -- Model Registry

#### E2-T1: Model Registry

**Contexto/Motivacao**: O Registry e quem sabe quais modelos estao instalados. Sem ele, o API Server nao sabe para qual worker rotear uma request. Em M3, o Registry e um scanner de diretorio -- le `theo.yaml` de cada subdiretorio.

**Escopo**:

| In | Out |
|----|-----|
| `ModelRegistry` com scan de diretorio | Download de modelos (theo pull) |
| `get_manifest(model_name)` | Lifecycle de modelos (load/unload/evict) |
| `list_models()` | Cache em memoria de artefatos |
| `has_model(model_name)` | Watch de filesystem para hot-reload |
| Levanta `ModelNotFoundError` se modelo nao existe | |
| Testes unitarios com manifestos de fixture | |

**Entregaveis**:
- `src/theo/registry/registry.py`
- Teste: `tests/unit/test_registry.py`

**Detalhes de implementacao**:

```python
class ModelRegistry:
    """Descobre e fornece manifestos de modelos instalados.

    Escaneia um diretorio de modelos buscando theo.yaml em cada
    subdiretorio. Nao gerencia lifecycle (load/unload) -- apenas
    informa quais modelos existem e suas configuracoes.
    """

    def __init__(self, models_dir: str | Path) -> None:
        self._models_dir = Path(models_dir)
        self._manifests: dict[str, ModelManifest] = {}

    async def scan(self) -> None:
        """Escaneia models_dir e carrega todos os manifestos."""
        ...

    def get_manifest(self, model_name: str) -> ModelManifest:
        """Retorna manifesto do modelo.

        Raises:
            ModelNotFoundError: Se modelo nao encontrado.
        """
        ...

    def list_models(self) -> list[ModelManifest]:
        """Retorna lista de todos os modelos instalados."""
        ...

    def has_model(self, model_name: str) -> bool:
        """Verifica se modelo existe no registry."""
        ...

    def get_model_path(self, model_name: str) -> Path:
        """Retorna caminho do diretorio do modelo.

        Raises:
            ModelNotFoundError: Se modelo nao encontrado.
        """
        ...
```

**Convencao de diretorio**:

O `scan()` itera subdiretorios de `models_dir`. Para cada subdiretorio que contem `theo.yaml`:
1. Parseia com `ModelManifest.from_yaml_path()`
2. Indexa por `manifest.name` (nao pelo nome do diretorio)
3. Ignora subdiretorios sem `theo.yaml` (log warning)
4. Ignora manifestos invalidos (log error, nao crash)

**Definition of Done**:
- `scan()` encontra manifestos em fixtures de teste
- `get_manifest("faster-whisper-tiny")` retorna manifesto correto
- `get_manifest("inexistente")` levanta `ModelNotFoundError`
- `list_models()` retorna todos os modelos encontrados
- `get_model_path()` retorna caminho correto
- mypy e ruff passam

**Dependencias**: Nenhuma task de M3 (usa `ModelManifest` de M1)

**Estimativa**: 1 dia

---

### Epic 3 -- Scheduler

#### E3-T1: Scheduler Basico

**Contexto/Motivacao**: Orquestrador entre o API Server e os workers gRPC. Em M3, e deliberadamente trivial -- encontra worker disponivel, envia request, retorna resultado. A interface e o que importa: em M9, a implementacao evolui sem mudar o API Server.

**Escopo**:

| In | Out |
|----|-----|
| `Scheduler` com metodo `transcribe()` | Priorizacao |
| Encontra worker READY via WorkerManager | Fila de espera |
| Envia gRPC TranscribeFile | Dynamic batching |
| Retorna BatchResult | Cancelamento |
| Levanta WorkerUnavailableError se sem worker | |

**Entregaveis**:
- `src/theo/scheduler/scheduler.py`
- Teste: `tests/unit/test_scheduler.py`

**Detalhes de implementacao**:

```python
@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    """Request interna de transcricao."""
    request_id: str
    model_name: str
    audio_data: bytes
    language: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = 0.0
    timestamp_granularities: tuple[str, ...] = ("segment",)
    initial_prompt: str | None = None
    hot_words: tuple[str, ...] | None = None
    task: str = "transcribe"  # "transcribe" ou "translate"

class Scheduler:
    """Roteia requests de transcricao para workers gRPC.

    M3: implementacao trivial (1 worker = 1 request).
    M9: evolui para priorizacao, fila, batching.
    """

    def __init__(self, worker_manager: WorkerManager, registry: ModelRegistry) -> None:
        self._worker_manager = worker_manager
        self._registry = registry

    async def transcribe(self, request: TranscribeRequest) -> BatchResult:
        """Envia request ao worker e retorna resultado.

        Raises:
            ModelNotFoundError: Modelo nao existe no registry.
            WorkerUnavailableError: Nenhum worker READY para o modelo.
        """
        ...
```

**Fluxo interno do `transcribe()`**:

```
1. registry.get_manifest(request.model_name)     # valida que modelo existe
2. worker = worker_manager.get_ready_worker(request.model_name)
3. if not worker: raise WorkerUnavailableError
4. channel = grpc.aio.insecure_channel(f"localhost:{worker.port}")
5. stub = STTWorkerStub(channel)
6. proto_request = _build_proto_request(request)
7. proto_response = await stub.TranscribeFile(proto_request)
8. return _proto_response_to_batch_result(proto_response)
```

**Conversao proto -> BatchResult**:

O Scheduler precisa de funcoes que convertem `TranscribeFileResponse` protobuf de volta para `BatchResult` Theo. Essas funcoes sao o inverso dos converters ja existentes em `workers/stt/converters.py`. Criar em `scheduler/converters.py` as funcoes:

- `_build_proto_request(request: TranscribeRequest) -> TranscribeFileRequest`
- `_proto_response_to_batch_result(response: TranscribeFileResponse) -> BatchResult`

**Definition of Done**:
- Teste: mock de WorkerManager retorna worker, Scheduler constroi request gRPC corretamente
- Teste: sem worker READY levanta `WorkerUnavailableError`
- Teste: modelo inexistente levanta `ModelNotFoundError`
- mypy e ruff passam

**Dependencias**: E2-T1 (Registry)

**Estimativa**: 1.5 dias

---

### Epic 4 -- CLI

#### E4-T1: `theo serve`

**Contexto/Motivacao**: O comando que inicia tudo. `theo serve` e o ponto de entrada do usuario -- o equivalente ao `ollama serve`. Sem ele, o usuario precisa iniciar worker e uvicorn manualmente.

**Escopo**:

| In | Out |
|----|-----|
| `theo serve` inicia API Server + workers | `theo serve --preload` (otimizacao) |
| Aceita `--host`, `--port`, `--models-dir` | Hot-reload |
| Escaneia registry, spawna workers | Multi-GPU |
| Shutdown graceful (SIGTERM/SIGINT) | |

**Entregaveis**:
- `src/theo/cli/main.py` -- grupo principal de comandos (click)
- `src/theo/cli/serve.py` -- comando `theo serve`
- `pyproject.toml` -- entry point `[project.scripts] theo = "theo.cli.main:cli"`
- Teste: `tests/unit/test_cli_serve.py` (verifica que o comando existe e parseia args)

**Fluxo de `theo serve`**:

```
1. configure_logging(log_format, level)
2. registry = ModelRegistry(models_dir)
3. await registry.scan()
4. if not registry.list_models():
      log "Nenhum modelo encontrado em {models_dir}"
      exit(1)
5. worker_manager = WorkerManager()
6. port_counter = 50051
7. for manifest in registry.list_models():
      if manifest.model_type == ModelType.STT:
          await worker_manager.spawn_worker(
              model_name=manifest.name,
              port=port_counter,
              engine=manifest.engine,
              model_path=str(registry.get_model_path(manifest.name)),
              engine_config=manifest.engine_config.model_dump(),
          )
          port_counter += 1
8. scheduler = Scheduler(worker_manager, registry)
9. app = create_app(registry=registry, scheduler=scheduler)
10. uvicorn.run(app, host=host, port=port)
11. # On shutdown:
      await worker_manager.stop_all()
```

**Definition of Done**:
- `theo serve --help` mostra opcoes
- `theo serve` com modelos instalados inicia server e workers
- SIGTERM desliga gracefully
- mypy e ruff passam

**Dependencias**: E1-T3 (API funciona), E2-T1 (Registry), E3-T1 (Scheduler)

**Estimativa**: 1.5 dias

---

#### E4-T2: `theo transcribe` e `theo translate`

**Contexto/Motivacao**: Atalhos CLI para transcrever/traduzir arquivos. Sao thin clients HTTP -- fazem POST para o server rodando.

**Escopo**:

| In | Out |
|----|-----|
| `theo transcribe <file>` | Streaming do microfone (M6) |
| `theo transcribe <file> --model <name>` | |
| `theo transcribe <file> --format srt` | |
| `theo translate <file>` | |
| Output formatado no terminal | |
| Erro claro se server nao esta rodando | |

**Entregaveis**:
- `src/theo/cli/transcribe.py`
- `src/theo/cli/translate.py`
- Teste: `tests/unit/test_cli_transcribe.py` (verifica parsing de args)

**Implementacao**: Usa `httpx` (ou `urllib3`) para enviar o request. Nao adicionar dependencia pesada -- `httpx` ja e dependencia transitiva de FastAPI/uvicorn.

**Alternativa considerada**: Usar o SDK OpenAI Python como cliente. Descartado porque adicionaria uma dependencia pesada ao core. Melhor usar HTTP direto.

**Nota**: Se o server nao esta rodando, o comando deve imprimir mensagem clara: `"Erro: servidor nao disponivel em http://localhost:8000. Execute 'theo serve' primeiro."` e sair com exit code 1.

**Definition of Done**:
- `theo transcribe audio.wav --model faster-whisper-tiny` funciona com server rodando
- `theo transcribe audio.wav --format srt` gera legenda SRT no stdout
- Erro claro se server nao esta rodando
- mypy e ruff passam

**Dependencias**: E4-T1 (`theo serve` funciona)

**Estimativa**: 1 dia

---

#### E4-T3: `theo list` e `theo inspect`

**Contexto/Motivacao**: Comandos de gerenciamento de modelos. Sao uteis para o usuario saber o que esta instalado antes de fazer `theo serve`. Nao precisam de server rodando -- leem disco diretamente.

**Escopo**:

| In | Out |
|----|-----|
| `theo list` -- tabela de modelos instalados | `theo pull` (download) |
| `theo inspect <model>` -- detalhes do manifesto | `theo remove` |
| Aceita `--models-dir` | |
| Funciona sem server rodando | |

**Entregaveis**:
- `src/theo/cli/models.py`
- Teste: `tests/unit/test_cli_models.py`

**Output de `theo list`**:

```
NAME                       TYPE   ENGINE           ARCHITECTURE      SIZE
faster-whisper-tiny        stt    faster-whisper   encoder-decoder   ~150MB
faster-whisper-large-v3    stt    faster-whisper   encoder-decoder   ~3GB
```

**Output de `theo inspect`**:

```
Name:           faster-whisper-large-v3
Type:           stt
Engine:         faster-whisper
Version:        3.0.0
Architecture:   encoder-decoder
Languages:      auto, en, pt, es
Memory:         3072 MB
GPU Required:   No
GPU Recommended: Yes
Capabilities:   streaming, word_timestamps, translation, batch_inference
```

**Definition of Done**:
- `theo list` com modelos no diretorio mostra tabela formatada
- `theo list` sem modelos mostra "Nenhum modelo instalado"
- `theo inspect faster-whisper-tiny` mostra detalhes
- `theo inspect inexistente` mostra erro claro
- Nao precisa de server rodando
- mypy e ruff passam

**Dependencias**: E2-T1 (Registry para leitura de disco)

**Estimativa**: 0.5 dia

---

### Epic 5 -- Validacao e Qualidade

#### E5-T1: Error Handling HTTP

**Contexto/Motivacao**: Erros do Theo (exceptions tipadas) devem ser traduzidos em respostas HTTP claras com status codes corretos. Sem isso, qualquer erro retorna 500 Internal Server Error -- inutil para o cliente.

**Escopo**:

| In | Out |
|----|-----|
| Exception handler global no FastAPI | |
| `ModelNotFoundError` -> 404 | |
| `AudioFormatError` -> 400 | |
| `AudioTooLargeError` -> 413 | |
| `WorkerUnavailableError` -> 503 com Retry-After | |
| `ValidationError` (Pydantic) -> 400 com detalhes | |
| Erros inesperados -> 500 com mensagem generica (nao expor stack trace) | |

**Entregaveis**:
- Exception handlers registrados em `server/app.py`
- Teste: `tests/unit/test_error_handling.py`

**Formato de resposta de erro (compativel OpenAI)**:

```json
{
  "error": {
    "message": "Modelo 'whisper-inexistente' nao encontrado no registry",
    "type": "model_not_found_error",
    "code": "model_not_found"
  }
}
```

**Validacao de tamanho de arquivo**: Verificar `Content-Length` header (se presente) e tamanho real do upload. Limite default: 25MB. Configuravel via env `THEO_MAX_FILE_SIZE_MB`.

**Definition of Done**:
- Teste: upload de arquivo > 25MB retorna 413
- Teste: modelo inexistente retorna 404
- Teste: arquivo com formato invalido retorna 400
- Teste: sem worker disponivel retorna 503
- Teste: erro inesperado retorna 500 sem stack trace
- mypy e ruff passam

**Dependencias**: E1-T3 (endpoint existe)

**Estimativa**: 1 dia

---

#### E5-T2: Testes End-to-End

**Contexto/Motivacao**: Os testes unitarios validam componentes isolados com mocks. Os testes e2e validam o fluxo completo: HTTP request -> API Server -> Scheduler -> gRPC Worker -> Response. Sao a prova definitiva de que M3 funciona.

**Escopo**:

| In | Out |
|----|-----|
| Teste e2e com server real (TestClient do FastAPI) | Teste com GPU |
| Mock do worker gRPC (nao precisa de Faster-Whisper real) | Teste com modelo real (e2e de integracao) |
| Fluxo completo: upload -> transcribe -> response | |
| Todos os formatos de resposta | |
| Cenarios de erro (404, 413, 503) | |

**Entregaveis**:
- `tests/unit/test_e2e_batch.py` (e2e com mocks, roda no CI)
- `tests/integration/test_e2e_real.py` (marcado `@pytest.mark.integration`, requer modelo instalado)

**Estrategia de mock para testes e2e unitarios**:

O teste cria um mock do Scheduler que retorna `BatchResult` pre-definido. Nao precisa de worker gRPC real rodando. Isso permite testar o fluxo HTTP completo sem dependencias externas.

```python
# Pseudocodigo do teste e2e
async def test_transcribe_file_json_format():
    mock_scheduler = MockScheduler(returns=sample_batch_result)
    mock_registry = MockRegistry(models=["faster-whisper-tiny"])
    app = create_app(registry=mock_registry, scheduler=mock_scheduler)
    client = TestClient(app)

    with open("tests/fixtures/audio/sample_16khz.wav", "rb") as f:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": "faster-whisper-tiny"},
        )

    assert response.status_code == 200
    assert "text" in response.json()
```

**Definition of Done**:
- Testes e2e cobrem: json, verbose_json, text, srt, vtt
- Testes e2e cobrem: 400, 404, 413, 503
- Testes rodam no CI sem modelo instalado (mocks)
- mypy e ruff passam

**Dependencias**: E4-T1 (server completo), E1-T5 (formatters), E5-T1 (error handling)

**Estimativa**: 1.5 dias

---

#### E5-T3: Compatibilidade com SDK OpenAI

**Contexto/Motivacao**: O contrato OpenAI e um dos diferenciais do Theo. Se o SDK oficial `openai` Python funciona como cliente, qualquer aplicacao que ja usa OpenAI pode migrar mudando apenas a `base_url`.

**Escopo**:

| In | Out |
|----|-----|
| Teste com `openai.OpenAI(base_url=...)` | Compatibilidade com SDKs de outras linguagens |
| `client.audio.transcriptions.create()` funciona | Endpoints nao-audio (chat, embeddings) |
| `client.audio.translations.create()` funciona | |
| Response fields compativeis | |

**Entregaveis**:
- `tests/integration/test_openai_sdk_compat.py` (marcado `@pytest.mark.integration`)
- Documentacao: quais campos/features sao suportados vs nao-suportados

**Nota**: Este teste requer que o server esteja rodando (nao e unitario). Pode ser executado manualmente ou em CI com step separado.

**Alternativa**: Criar o teste como `@pytest.mark.integration` que sobe o server em processo via TestClient e usa `httpx` simulando o que o SDK faria. Isso evita dependencia do SDK `openai` no dev dependencies.

**Decisao**: Adicionar `openai>=1.0` como dependencia de teste (dev extra) e testar diretamente. E a validacao mais honesta.

**Definition of Done**:
- SDK `openai` Python consegue transcrever arquivo via Theo
- Response `result.text` contem o texto transcrito
- Nenhum erro de validacao no lado do SDK
- mypy e ruff passam

**Dependencias**: E5-T2 (testes e2e passam)

**Estimativa**: 0.5 dia

---

## 4. Sequencia de Implementacao Detalhada

### Timeline

```
Dia 1:
  E1-T1  Health endpoint                       [0.5d]
  E2-T1  Model Registry                        [0.5d inicio]

Dia 2:
  E2-T1  Model Registry (conclusao)            [0.5d]
  E1-T2  Pydantic models                       [1d]

Dia 3:
  E3-T1  Scheduler                             [1.5d]

Dia 4:
  E3-T1  Scheduler (conclusao)                 [0.5d]
  E1-T3  POST /v1/audio/transcriptions          [1d inicio]

Dia 5:
  E1-T3  POST /v1/audio/transcriptions (conclusao)  [0.5d]
  E1-T4  POST /v1/audio/translations                [0.5d]

  -- CHECKPOINT: curl funciona end-to-end --

Dia 6:
  E1-T5  Response formatters (srt, vtt, etc)   [1d]

Dia 7:
  E5-T1  Error handling HTTP                    [1d]

Dia 8:
  E4-T1  theo serve                             [1.5d]

Dia 9:
  E4-T1  theo serve (conclusao)                 [0.5d]
  E4-T2  theo transcribe / theo translate       [1d]

Dia 10:
  E4-T3  theo list / theo inspect               [0.5d]
  E5-T2  Testes e2e (inicio)                    [0.5d]

Dia 11:
  E5-T2  Testes e2e (conclusao)                 [1d]

Dia 12:
  E5-T3  Compatibilidade OpenAI SDK             [0.5d]
  Ajustes finais, review                        [0.5d]
```

**Total estimado: 12 dias de trabalho focado (~2.5 semanas)**

### Checkpoints de valor incremental

| Dia | O que funciona |
|-----|----------------|
| 1 | `GET /health` retorna 200 |
| 3 | Registry encontra modelos, Scheduler esta pronto |
| 5 | `curl -F file=@audio.wav -F model=... /v1/audio/transcriptions` retorna texto |
| 6 | Todos os formatos de resposta (json, text, srt, vtt, verbose_json) |
| 7 | Erros HTTP retornam status codes corretos |
| 9 | `theo serve` inicia tudo; `theo transcribe` funciona |
| 10 | `theo list` e `theo inspect` funcionam |
| 12 | Testes e2e passam, SDK OpenAI funciona |

---

## 5. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Incompatibilidade sutil com contrato OpenAI (campos, tipos, defaults) | Media | Medio | Testar com SDK oficial `openai` (E5-T3). Validar contra documentacao OpenAI. Campos extras nao causam erro no SDK -- campos faltando sim. |
| R2 | Cold start lento: modelo carregando na primeira request, usuario recebe timeout | Alta | Medio | `theo serve` pre-carrega modelos via WorkerManager. Enquanto worker esta STARTING, retornar 503 com header `Retry-After: 10`. Nao bloquear a request esperando. |
| R3 | Upload de arquivo grande consome memoria do processo FastAPI | Baixa | Baixo | `UploadFile` do FastAPI faz spooling para disco apos 1MB (comportamento default do Starlette). Limite de 25MB e enforcement adicional. |
| R4 | Worker crashou e Scheduler nao encontra worker READY | Media | Medio | WorkerManager ja tem auto-restart com backoff. Scheduler retorna 503 imediatamente. Nao tentar esperar restart -- fail fast. |
| R5 | gRPC channel entre Scheduler e Worker nao e gerenciado (leak de connections) | Media | Medio | Criar channel pool no Scheduler ou criar/fechar channel por request. Em M3, criar/fechar por request e aceitavel (batch e infrequente). Otimizar em M5+ com channel pool. |
| R6 | `theo serve` e complexo de testar (combina server + workers) | Media | Baixo | Testar `theo serve` via unit test do click command (verifica parsing de args). Teste e2e testa o fluxo completo. Nao testar o startup completo em CI (requer modelo). |
| R7 | click como dependencia CLI adiciona peso ao core | Baixa | Baixo | click tem zero dependencias transitivas pesadas. E ~100KB. Aceitavel. |

---

## 6. Mudancas no pyproject.toml

As seguintes alteracoes sao necessarias no `pyproject.toml` para M3:

```toml
# Adicionar ao core dependencies
dependencies = [
    "pydantic>=2.6,<3.0",
    "pyyaml>=6.0,<7.0",
    "structlog>=24.0,<26.0",
    "numpy>=1.26,<3.0",
    "click>=8.1,<9.0",          # NOVO: CLI framework
    "httpx>=0.27,<1.0",         # NOVO: HTTP client para CLI
]

# Adicionar entry point
[project.scripts]
theo = "theo.cli.main:cli"

# Adicionar ao dev extras
dev = [
    ...existing...,
    "openai>=1.0,<2.0",         # NOVO: teste de compatibilidade SDK
]
```

**Justificativa**:
- `click`: CLI framework (KISS -- nao usar typer). Vai no core porque o CLI e parte do produto.
- `httpx`: HTTP client para `theo transcribe`/`theo translate`. Vai no core porque esses comandos fazem parte do produto. `httpx` e leve e async-native.
- `openai`: Apenas para teste de compatibilidade. Vai no dev extras.

---

## 7. Demo Goal

### O que sera demonstrado

Ao final de M3, a seguinte sequencia funciona sem erros:

### Script de demonstracao

```bash
# 1. Verificar modelos instalados
theo list
# NAME                    TYPE   ENGINE           ARCHITECTURE
# faster-whisper-tiny     stt    faster-whisper   encoder-decoder

# 2. Inspecionar modelo
theo inspect faster-whisper-tiny
# Name:           faster-whisper-tiny
# Type:           stt
# Engine:         faster-whisper
# ...

# 3. Iniciar servidor
theo serve &
# [INFO] Escaneando modelos em ~/.theo/models/
# [INFO] Modelo encontrado: faster-whisper-tiny
# [INFO] Spawning worker faster-whisper-50051
# [INFO] Worker ready: faster-whisper-50051
# [INFO] Server iniciado em http://0.0.0.0:8000

# 4. Health check
curl http://localhost:8000/health
# {"status": "ok"}

# 5. Transcrever via curl (formato json)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-tiny
# {"text": "ola como posso ajudar"}

# 6. Transcrever via curl (formato verbose_json)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-tiny \
  -F response_format=verbose_json
# {"task":"transcribe","language":"pt","duration":2.5,"text":"...","segments":[...]}

# 7. Gerar legenda SRT
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-tiny \
  -F response_format=srt
# 1
# 00:00:00,000 --> 00:00:02,500
# ola como posso ajudar

# 8. Traduzir para ingles
curl -X POST http://localhost:8000/v1/audio/translations \
  -F file=@audio.wav \
  -F model=faster-whisper-tiny
# {"text": "hello how can I help you"}

# 9. Transcrever via CLI
theo transcribe audio.wav --model faster-whisper-tiny
# ola como posso ajudar

# 10. Transcrever via CLI com formato SRT
theo transcribe audio.wav --model faster-whisper-tiny --format srt
# 1
# 00:00:00,000 --> 00:00:02,500
# ola como posso ajudar

# 11. Traduzir via CLI
theo translate audio.wav --model faster-whisper-tiny
# hello how can I help you

# 12. Testar com SDK OpenAI Python
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='not-needed')
result = client.audio.transcriptions.create(
    model='faster-whisper-tiny',
    file=open('audio.wav', 'rb'),
)
print(result.text)
"
# ola como posso ajudar

# 13. Erro: modelo inexistente
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=inexistente
# HTTP 404
# {"error":{"message":"Modelo 'inexistente' nao encontrado","type":"model_not_found_error"}}

# 14. Shutdown graceful
kill %1
# [INFO] Shutdown iniciado
# [INFO] Workers parados
# [INFO] Server encerrado
```

---

## 8. Quality Gates

M3 esta completo quando TODOS os itens abaixo forem verdadeiros:

### 8.1 Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `theo serve` inicia server + workers | Executar manualmente |
| 2 | `GET /health` retorna 200 | `curl localhost:8000/health` |
| 3 | `POST /v1/audio/transcriptions` retorna texto | `curl -F file=... -F model=...` |
| 4 | `POST /v1/audio/translations` retorna traducao | `curl -F file=... -F model=...` |
| 5 | Todos os formatos funcionam (json, verbose_json, text, srt, vtt) | Testes e2e |
| 6 | Erros HTTP corretos (400, 404, 413, 503) | Testes e2e |
| 7 | `theo transcribe <file>` funciona | Executar manualmente |
| 8 | `theo translate <file>` funciona | Executar manualmente |
| 9 | `theo list` e `theo inspect` funcionam | Executar manualmente |
| 10 | SDK OpenAI Python funciona como cliente | Teste de integracao |
| 11 | Shutdown graceful (SIGTERM) desliga server e workers | Executar manualmente |

### 8.2 Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `.venv/bin/python -m mypy src/` |
| 2 | ruff check sem warnings | `.venv/bin/python -m ruff check src/ tests/` |
| 3 | ruff format sem diffs | `.venv/bin/python -m ruff format --check src/ tests/` |
| 4 | Todos os testes passam | `.venv/bin/python -m pytest tests/unit/ -v` |
| 5 | CI verde | GitHub Actions |

### 8.3 Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | Health endpoint | 2-3 |
| Unit | Pydantic request/response models | 10-15 |
| Unit | Registry (scan, get, list, not found) | 8-10 |
| Unit | Scheduler (transcribe, no worker, no model) | 6-8 |
| Unit | Transcriptions route (com mock) | 5-7 |
| Unit | Translations route (com mock) | 3-4 |
| Unit | Response formatters (json, verbose, text, srt, vtt) | 8-10 |
| Unit | Error handling (400, 404, 413, 503, 500) | 5-7 |
| Unit | CLI commands (parsing de args) | 6-8 |
| E2E  | Fluxo completo com mock de worker | 5-8 |
| Integration | SDK OpenAI | 2-3 |
| **Total** | | **~60-80 novos testes** |

### 8.4 Documentacao

| # | Criterio |
|---|----------|
| 1 | CHANGELOG.md atualizado com entradas de M3 |
| 2 | CLAUDE.md atualizado com comandos de M3 |
| 3 | Docstrings em todas as interfaces publicas |
| 4 | ROADMAP.md atualizado com status de M3 |

### 8.5 Checklist Final

```bash
# 1. Qualidade de codigo
.venv/bin/python -m ruff format --check src/ tests/
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m mypy src/

# 2. Testes unitarios
.venv/bin/python -m pytest tests/unit/ -v

# 3. Instalacao do CLI
pip install -e ".[dev,server,grpc]"
theo --help

# 4. Validacao de imports
python -c "
from theo.registry.registry import ModelRegistry
from theo.scheduler.scheduler import Scheduler, TranscribeRequest
from theo.server.app import create_app
from theo.server.formatters import format_response
from theo.cli.main import cli
print('Todos os modulos M3 importaveis com sucesso')
"

# 5. Demo (requer modelo instalado)
theo list
theo serve &
sleep 5  # aguardar workers
curl -f http://localhost:8000/health
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@tests/fixtures/audio/sample_16khz.wav \
  -F model=faster-whisper-tiny
kill %1
```

---

## 9. Estrutura de Arquivos Final (M3)

```
src/theo/
  server/
    __init__.py
    app.py                    # create_app() factory, exception handlers
    dependencies.py           # FastAPI Depends (registry, scheduler)
    formatters.py             # BatchResult -> json/text/srt/vtt
    routes/
      __init__.py
      health.py               # GET /health
      transcriptions.py       # POST /v1/audio/transcriptions
      translations.py         # POST /v1/audio/translations
    models/
      __init__.py
      requests.py             # Pydantic models de request (interno)
      responses.py            # Pydantic models de response
  registry/
    __init__.py
    registry.py               # ModelRegistry
  scheduler/
    __init__.py
    scheduler.py              # Scheduler, TranscribeRequest
    converters.py             # Proto <-> TranscribeRequest/BatchResult
  cli/
    __init__.py
    main.py                   # Click group principal
    serve.py                  # theo serve
    transcribe.py             # theo transcribe
    translate.py              # theo translate
    models.py                 # theo list, theo inspect

tests/
  unit/
    test_health_endpoint.py
    test_api_models.py
    test_registry.py
    test_scheduler.py
    test_transcriptions_route.py
    test_translations_route.py
    test_formatters.py
    test_error_handling.py
    test_cli_serve.py
    test_cli_transcribe.py
    test_cli_models.py
    test_e2e_batch.py
  integration/
    test_e2e_real.py
    test_openai_sdk_compat.py
```

---

## 10. Transicao M3 -> M4

Ao completar M3, o time deve ter:

1. **API funcional** -- `POST /v1/audio/transcriptions` retorna texto cru da engine. Nao formatado (sem ITN).

2. **CLI funcional** -- `theo serve`, `theo transcribe`, `theo list` funcionam.

3. **Contrato estavel** -- O formato da API e testado contra o SDK OpenAI. Nao muda em M4.

4. **Pontos de extensao claros** para M4:
   - **Preprocessing**: Inserir entre o upload do arquivo e o envio ao worker. O Scheduler recebe `audio_data: bytes` -- em M4, esses bytes passam pelo preprocessing pipeline antes de ir ao worker.
   - **Post-processing**: Inserir entre o `BatchResult` do worker e o response formatter. Em M4, `BatchResult.text` passa pelo ITN antes de ser formatado.

5. **Scheduler substituivel** -- A interface `Scheduler.transcribe()` nao muda em M9. A implementacao interna evolui.

O primeiro commit de M4 sera: inserir um preprocessing pipeline (identity function) no fluxo batch e adicionar o stage de resample.

---

*Documento gerado por Sofia Castellani (Principal Solution Architect, ARCH). Sera atualizado conforme a implementacao do M3 avanca.*
