# Theo OpenVoice -- Roadmap Estrategico do Milestone M2

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0, STRATEGIC_M1.md v1.0
**Status**: Aprovado pelo Time de Arquitetura (ARCH)
**Data**: 2026-02-07

**Autores**:
- Sofia Castellani (Principal Solution Architect) -- FasterWhisperBackend, conversao gRPC<->tipos, design do Worker Manager
- Viktor Sorokin (Senior Real-Time Engineer) -- Compilacao de protobufs, gRPC servicer, subprocess lifecycle
- Andre Oliveira (Senior Platform Engineer) -- Structured logging, CI para integracao, pyproject.toml extras

---

## 1. Objetivos Estrategicos do M2

O M2 e o primeiro milestone onde **codigo executavel aparece**. Pela primeira vez, audio entra no sistema e texto sai. E a transicao de "definicoes e contratos" (M1) para "processo rodando que faz algo util". A qualidade do M2 determina a confianca do time em todo o pipeline gRPC que sera usado ate M10.

### 1.1 Por que cada objetivo importa

| Objetivo | O que e | Valor Estrategico |
|----------|---------|-------------------|
| **Proto compilado** | `stt_worker.proto` -> `stt_worker_pb2.py` + `stt_worker_pb2_grpc.py` | Transforma o contrato definido em M1 em codigo usavel. Sem isso, nenhuma comunicacao runtime-worker existe. Todo milestone de M2 a M10 depende destes modulos gerados. |
| **Worker gRPC** | Processo Python que implementa o service `STTWorker` | E o processo que roda a engine de inferencia. O worker e o componente que interage com GPU, consome memoria, e pode crashar. Isolamento via subprocess e a decisao arquitetural mais importante do Theo. |
| **FasterWhisperBackend** | Implementacao concreta de `STTBackend` | Primeira validacao real da interface definida em M1. Se `STTBackend.transcribe_file` nao servir para Faster-Whisper, nao servira para nenhuma engine. Ajustes na ABC devem acontecer AGORA, antes de M7 (segundo backend). |
| **Worker Manager** | Spawna, monitora e reinicia subprocessos | E o "supervisor" dos workers. Sem ele, workers sao processos avulsos que ninguem gerencia. O Manager garante que workers estao vivos, detecta crashes, e prepara o terreno para o Scheduler (M3/M9). |
| **Health check gRPC** | RPC unario que reporta status do worker | Primeira forma de saber se um worker esta operacional. Em M5+, o stream break substitui para deteccao de crash, mas o health check unario e necessario para tooling (CLI `theo ps`, monitoring). |
| **Deteccao de crash** | Monitorar exit code do subprocess | Workers sao subprocessos que podem crashar (SIGSEGV do CUDA, OOM, bug na engine). O Manager precisa detectar isso em milissegundos e reagir. Sem deteccao, sessoes ficam penduradas indefinidamente. |
| **Structured logging** | Logging com campos estruturados desde M2 | Workers sao subprocessos separados. Sem logging estruturado com correlation IDs, debugar problemas cross-processo e impossivel. Estabelecer o padrao agora evita retrofitting doloroso em M5+. |

### 1.2 Principio do M2

> **Fazer com que `python -m theo.workers.stt.main --port 50051` inicie um worker que responde Health e transcreve um arquivo via gRPC.** Se esse worker funciona como subprocess gerenciado, todo milestone subsequente tem um worker confiavel para construir sobre.

---

## 2. Decisoes Tecnicas Chave

### 2.1 Compilacao de Protobufs

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Ferramenta** | `grpcio-tools` (via `python -m grpc_tools.protoc`) | Ja declarado como dev dependency em M1. Alternativa (`buf`, `protoc` standalone) adicionaria dependencia externa sem ganho. |
| **Onde gerar** | `src/theo/proto/stt_worker_pb2.py` e `stt_worker_pb2_grpc.py` | Junto ao `.proto` fonte. O `__init__.py` do pacote `proto/` re-exporta os modulos gerados. |
| **Versionamento** | Arquivos gerados **commitados** no repositorio | Evita exigir `grpcio-tools` em ambiente de producao. CI valida que os gerados estao em sync com o `.proto`. |
| **Script de geracao** | `scripts/generate_proto.sh` | Um unico comando reproduzivel. CI roda o script e `git diff --exit-code` para validar sincronia. |
| **mypy override** | `theo.proto.*` com `ignore_errors = true` | Ja configurado em M1. Codigo gerado por `grpcio-tools` nao e tipado -- aceitar e isolar. |

**Sofia**: Commitar os arquivos gerados e a decisao pragmatica. A alternativa (gerar no build) adiciona complexidade ao `pyproject.toml` e quebra installs em ambientes sem `grpcio-tools`. O trade-off e manter os gerados em sync -- o CI resolve isso.

**Viktor**: O comando de geracao deve usar `--proto_path` apontando para `src/theo/proto/` e output no mesmo diretorio. Testar que `from theo.proto.stt_worker_pb2 import TranscribeFileRequest` funciona apos instalacao.

**Andre**: O script `scripts/generate_proto.sh` deve ser idempotente e verificavel. CI step: `bash scripts/generate_proto.sh && git diff --exit-code src/theo/proto/` -- se houver diff, o proto foi alterado sem regenerar.

### 2.2 Estrutura do Worker como Subprocess

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Mecanismo** | `subprocess.Popen` (nao `multiprocessing`) | Workers sao processos independentes com seu proprio Python interpreter. `subprocess` da controle total sobre stdin/stdout/stderr, environment vars e signals. `multiprocessing` compartilha estado e complica isolamento de GPU. |
| **Entrypoint** | `python -m theo.workers.stt.main` | Modulo executavel (`__main__` pattern). Aceita `--port`, `--model-path`, `--engine-config` como argumentos CLI. |
| **Comunicacao** | Exclusivamente via gRPC (TCP localhost) | Nenhuma comunicacao via stdin/stdout/pipe entre runtime e worker. gRPC e o unico canal. stdout/stderr do worker sao capturados para logging. |
| **Lifecycle** | Start -> Ready (health ok) -> Running -> Stop/Crash | O Manager spawna o processo, espera health ok (polling com backoff), e entao marca como disponivel. |
| **Shutdown** | SIGTERM com grace period de 5s, depois SIGKILL | Worker recebe SIGTERM, faz cleanup (unload modelo), e encerra. Se nao encerrar em 5s, SIGKILL. |

**Sofia**: `subprocess.Popen` sobre `multiprocessing.Process` e a decisao correta para isolamento. Em M5+ com gRPC bidirecional, o stream gRPC vai ser o heartbeat implicito. Mas em M2, onde so temos RPC unario, o health check polling e necessario.

**Viktor**: O worker deve configurar signal handlers para SIGTERM (graceful shutdown) e SIGINT (mesma coisa). SIGKILL nao pode ser interceptado -- e o ultimo recurso do Manager. O stdout/stderr do worker deve ser capturado via `subprocess.PIPE` e logado pelo Manager com prefixo `[worker-{id}]`.

**Andre**: O entrypoint `python -m theo.workers.stt.main` deve funcionar standalone (para desenvolvimento e debugging) e como subprocess gerenciado (pelo Worker Manager). O mesmo binario, dois modos de uso.

### 2.3 FasterWhisperBackend Design

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Classe** | `FasterWhisperBackend(STTBackend)` | Implementacao concreta da ABC definida em M1. Fica em `src/theo/workers/stt/faster_whisper.py`. |
| **Escopo M2** | Apenas `transcribe_file` (batch) | `transcribe_stream` e implementado em M5. Em M2, o metodo existe mas levanta `NotImplementedError` com mensagem clara. |
| **Wrapping** | Instancia `faster_whisper.WhisperModel` internamente | `load()` cria o `WhisperModel`. `transcribe_file()` chama `model.transcribe()`. `unload()` deleta a referencia e forca GC. |
| **Conversao** | Faster-Whisper types -> Theo types | O backend converte `faster_whisper.TranscriptionSegment` para `theo._types.SegmentDetail` e `BatchResult`. Nenhum tipo do Faster-Whisper vaza para fora do backend. |
| **Audio input** | `bytes` (PCM 16-bit 16kHz mono) | O backend recebe bytes e converte para `numpy.ndarray` float32 normalizado (que e o que Faster-Whisper espera). |
| **Thread safety** | Nao thread-safe | Cada worker tem um unico `FasterWhisperBackend`. Concorrencia e gerenciada pelo Scheduler (uma request por vez por worker em M2). |

**Sofia**: O ponto critico e a conversao de tipos. Faster-Whisper retorna `namedtuple`-like objects com campos como `start`, `end`, `text`, `avg_logprob`. O backend deve converter para `SegmentDetail` e `BatchResult` definidos em `_types.py`. Se os tipos do M1 nao acomodarem o output do Faster-Whisper, e hora de ajustar -- melhor agora do que em M7.

**Viktor**: O `audio_data: bytes` recebido via gRPC precisa ser convertido para `numpy.ndarray` float32 normalizado [-1.0, 1.0] antes de passar para `WhisperModel.transcribe()`. Essa conversao e responsabilidade do backend, nao do gRPC servicer.

### 2.4 Worker Manager Design

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Classe** | `WorkerManager` | Fica em `src/theo/workers/manager.py`. Gerencia lifecycle de um ou mais subprocessos worker. |
| **Escopo M2** | Gerenciar 1 worker STT | M2 suporta apenas 1 worker por modelo. Multi-worker vem em M9 (Scheduler Avancado). |
| **Spawn** | `spawn_worker(model_name, port, engine_config) -> WorkerHandle` | Retorna um handle com `process`, `port`, `worker_id`, `status`. |
| **Health probe** | Polling gRPC `Health` com exponential backoff | Apos spawn, o Manager faz health check a cada 500ms (backoff ate 5s). Quando `status: "ok"`, marca como `READY`. |
| **Crash detection** | `process.poll()` + exit code | Loop assincrono que monitora `returncode`. Exit code != None -> crash detectado -> emit `WorkerCrashError`. |
| **Restart** | Automatico com backoff | Crash -> espera 1s -> restart. Segundo crash -> espera 2s. Terceiro -> espera 4s. Max 3 restarts em 60s, depois desiste. |
| **Stop** | `stop_worker(worker_id)` | Envia SIGTERM, espera 5s, SIGKILL se necessario. |

**Sofia**: O `WorkerHandle` deve ser uma dataclass com estado explicito: `STARTING`, `READY`, `BUSY`, `STOPPING`, `CRASHED`. Isso prepara o terreno para o Scheduler (M3) que precisa saber quais workers estao disponiveis.

**Viktor**: O loop de monitoramento deve ser async (`asyncio.create_task`) para nao bloquear o event loop do runtime. `process.poll()` e nao-bloqueante, mas o loop precisa de um `asyncio.sleep` entre checks.

**Andre**: O backoff de restart deve ser configuravel via environment variable ou config. Em producao, pode-se querer desabilitar auto-restart e deixar o orquestrador (Kubernetes) lidar com restarts.

### 2.5 gRPC Servicer Pattern

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Padrao** | `STTWorkerServicer` herda do gerado `stt_worker_pb2_grpc.STTWorkerServicer` | Padrao oficial do gRPC Python. O servicer implementa os RPCs definidos no proto. |
| **Escopo M2** | `TranscribeFile` e `Health` implementados. `TranscribeStream` e `Cancel` levantam `UNIMPLEMENTED`. | M2 e batch-only. Streaming em M5. |
| **Server** | `grpc.aio.server()` | Servidor async para compatibilidade com o event loop do FastAPI em M3+. Usar `grpc.aio` desde o inicio evita reescrita de sync para async. |
| **Interceptors** | Logging interceptor basico | Loga request_id, metodo, duracao, status. Prepara o padrao para metricas em M3. |
| **Reflection** | `grpc_reflection` habilitado em dev | Permite uso de `grpcurl` e `grpcui` para debug. Desabilitado em producao. |

**Viktor**: `grpc.aio.server()` e essencial. O Faster-Whisper `model.transcribe()` e CPU/GPU-bound e bloqueante. O servicer deve rodar a inferencia em `asyncio.get_event_loop().run_in_executor(None, ...)` para nao bloquear o event loop gRPC.

**Sofia**: O servicer nao deve conter logica de negocio. Ele recebe `TranscribeFileRequest` (protobuf), converte para chamada no `FasterWhisperBackend`, e converte o `BatchResult` de volta para `TranscribeFileResponse` (protobuf). O servicer e um adapter -- nada mais.

### 2.6 Structured Logging

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Biblioteca** | `logging` stdlib com `structlog` como processor | `structlog` adiciona structured logging sem substituir a stdlib. Compativel com ferramentas que esperam `logging`. |
| **Formato** | JSON em producao, console colorido em dev | `structlog.dev.ConsoleRenderer` para dev, `structlog.processors.JSONRenderer` para producao. Toggle via env `THEO_LOG_FORMAT=json|console`. |
| **Campos obrigatorios** | `timestamp`, `level`, `event`, `component` | Todo log tem contexto minimo para filtragem. `component` identifica a fonte (ex: `worker_manager`, `stt_servicer`, `faster_whisper`). |
| **Correlation** | `request_id` propagado via gRPC metadata | O runtime gera UUID, envia como metadata gRPC, worker loga com o mesmo ID. Em M5+, o `session_id` tambem e propagado. |
| **Worker stdout** | Capturado e re-logado pelo Manager | stdout/stderr do subprocess sao capturados e logados com prefixo `[worker-{id}]` e level inferido (stderr -> WARNING). |

**Andre**: `structlog` e uma dependencia leve (~50KB) que nao traz nada transitivo. Adicionar ao core dependencies do `pyproject.toml` (nao como optional) porque logging e transversal a todo o sistema. Alternativa seria apenas `logging.config.dictConfig` com formatacao JSON -- mais simples, menos features. Recomendo `structlog` pelo bind de contexto (`log.bind(request_id=rid)`), que evita repetir campos em cada chamada.

**Sofia**: O logging deve ser configurado em um unico lugar (`src/theo/logging.py`) e importado por todos os modulos. Nao criar loggers ad-hoc com `logging.getLogger(__name__)` em cada arquivo -- usar o structlog configurado centralmente.

---

## 3. Dependencias Novas no M2

### 3.1 Atualizacao do pyproject.toml

O M2 ativa os extras `[grpc]` e `[faster-whisper]` que ja estao declarados no `pyproject.toml` desde M1. Nova dependencia core:

| Pacote | Tipo | Versao | Uso |
|--------|------|--------|-----|
| `grpcio` | extra `[grpc]` | `>=1.68,<2.0` | Runtime gRPC (client e server) |
| `protobuf` | extra `[grpc]` | `>=5.29,<6.0` | Serializacao de mensagens protobuf |
| `faster-whisper` | extra `[faster-whisper]` | `>=1.1,<2.0` | Engine de inferencia STT |
| `structlog` | core | `>=24.0,<26.0` | Structured logging |
| `numpy` | transitiva (via faster-whisper) | -- | Conversao audio bytes -> array |

**Andre**: `structlog` entra como dependencia core porque todo componente precisa de logging. `numpy` nao precisa ser declarada explicitamente -- vem transitivamente via `faster-whisper`. Mas se quisermos usar numpy no runtime (para conversao de audio no preprocessing), devemos declara-la. Em M2, a conversao bytes->ndarray acontece dentro do `FasterWhisperBackend` que so existe quando `faster-whisper` esta instalado, entao nao precisa.

---

## 4. Iniciativas

O M2 e decomposto em 6 iniciativas + 1 auxiliar, ordenadas por dependencia. Cada iniciativa e independente o suficiente para ser um PR atomico.

### 4.1 Grafo de Dependencias entre Iniciativas

```
I1 (Proto compilado + script)
├──> I2 (gRPC Worker Server / Servicer)
│    ├──> I3 (FasterWhisperBackend)
│    │    └──> I5 (Testes unitarios)
│    └──> I4 (Worker Manager)
│         └──> I5 (Testes unitarios)
└──> I5 (Testes unitarios)

I-AUX (Structured Logging) ──> [usado por I2, I3, I4]

I6 (Teste de integracao) ──> [depende de I2, I3, I4, I5]
```

**Legenda**:
- I1 e pre-requisito de tudo (sem proto compilado, nenhum gRPC funciona)
- I-AUX (logging) pode ser feito em paralelo com I1 e e usado por I2, I3, I4
- I2 depende de I1 (servicer precisa dos stubs gerados)
- I3 depende de I2 (backend e chamado pelo servicer, precisa dos tipos proto para conversao)
- I4 depende de I2 (Manager spawna o worker que contem o servicer)
- I5 depende de I2, I3, I4 (testa todos os componentes)
- I6 depende de tudo (teste end-to-end com modelo real)

**Paralelismo possivel**:
- I-AUX e I1 podem ser feitas em paralelo
- Apos I1: I2 e o proximo passo
- Apos I2: I3 e I4 podem ser feitas em paralelo
- I5 e incremental (testes adicionados junto com cada componente, consolidados ao final)

---

### Iniciativa I1 -- Compilacao de Protobuf e Script de Geracao

**Objetivo**: Gerar codigo Python a partir do `stt_worker.proto` definido em M1, com script reproduzivel e validacao no CI.

**Responsavel**: Viktor Sorokin

**Dependencia**: Nenhuma (usa proto definido em M1)

**Contexto**: O `stt_worker.proto` foi definido e commitado em M1, mas nunca compilado. Este e o primeiro passo para que qualquer comunicacao gRPC exista.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| Script `scripts/generate_proto.sh` | Modificar o `.proto` (ja definido em M1) |
| Arquivos gerados `stt_worker_pb2.py`, `stt_worker_pb2_grpc.py` | gRPC reflection (vem em I2) |
| `__init__.py` do pacote `proto/` com re-exports | Validacao semantica do proto |
| Step no CI para validar sincronia proto -> gerado | Geracao automatica no build (pip install) |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `scripts/generate_proto.sh` | Script idempotente que gera os stubs Python |
| `src/theo/proto/stt_worker_pb2.py` | Modulo gerado com classes de mensagens protobuf |
| `src/theo/proto/stt_worker_pb2_grpc.py` | Modulo gerado com stubs de client e servicer |
| `src/theo/proto/__init__.py` | Re-exports dos modulos gerados para import limpo |

**Script `scripts/generate_proto.sh`**:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROTO_DIR="src/theo/proto"
PROTO_FILE="${PROTO_DIR}/stt_worker.proto"

python -m grpc_tools.protoc \
  --proto_path="${PROTO_DIR}" \
  --python_out="${PROTO_DIR}" \
  --grpc_python_out="${PROTO_DIR}" \
  "${PROTO_FILE}"

echo "Proto compiled: ${PROTO_DIR}/stt_worker_pb2.py, stt_worker_pb2_grpc.py"
```

**CI step adicionado ao `ci.yml`**:

```yaml
- name: Validate proto sync
  run: |
    bash scripts/generate_proto.sh
    git diff --exit-code src/theo/proto/
```

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Script gera os arquivos sem erros | `bash scripts/generate_proto.sh` (exit 0) |
| 2 | Modulos importaveis | `python -c "from theo.proto.stt_worker_pb2 import TranscribeFileRequest; print('ok')"` |
| 3 | CI valida sincronia | Step `Validate proto sync` passa (git diff vazio) |
| 4 | mypy nao falha nos gerados | `python -m mypy src/` passa (override `ignore_errors` ja configurado) |

**Estimativa**: S (0.5 dia)

---

### Iniciativa I-AUX -- Structured Logging Foundation

**Objetivo**: Estabelecer o padrao de logging estruturado que todo componente do Theo usa, antes de implementar os componentes que precisam logar.

**Responsavel**: Andre Oliveira

**Dependencia**: Nenhuma

**Contexto**: Workers sao subprocessos separados. Sem logging estruturado com campos de contexto (`request_id`, `worker_id`, `component`), debugar problemas cross-processo e impossivel. Estabelecer o padrao agora evita retrofitting em M5+ quando a complexidade explode.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| `src/theo/logging.py` com configuracao structlog | Log aggregation (ELK, Loki) |
| Funcao `configure_logging(format, level)` | Metricas Prometheus (M3) |
| Formato JSON para producao, console para dev | Tracing distribuido (OpenTelemetry) |
| Dependencia `structlog` no `pyproject.toml` | Rotation de log files |
| Testes basicos de configuracao | |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/logging.py` | Modulo de configuracao de logging |
| Atualizacao `pyproject.toml` | `structlog>=24.0,<26.0` nas dependencies core |
| `tests/unit/test_logging.py` | Testes de configuracao e formatacao |

**Interface publica de `theo.logging`**:

```python
# Uso em qualquer modulo:
from theo.logging import get_logger

log = get_logger("worker_manager")
log.info("worker_spawned", worker_id="w-001", port=50051, model="faster-whisper-tiny")
# -> {"timestamp": "...", "level": "info", "event": "worker_spawned",
#     "component": "worker_manager", "worker_id": "w-001", "port": 50051, ...}
```

**Decisoes de design**:

| Decisao | Alternativa | Justificativa |
|---------|-------------|---------------|
| `structlog` sobre stdlib pura | `logging.config.dictConfig` com JSON formatter | `structlog.bind()` permite agregar contexto incrementalmente (ex: `log = log.bind(session_id=sid)`) sem repetir campos. Stdlib pura requer `extra={}` em cada chamada. |
| `structlog` sobre `loguru` | `loguru` | `loguru` substitui a stdlib inteiramente, o que causa conflitos com bibliotecas que usam `logging` (como gRPC). `structlog` decora a stdlib sem substituir. |
| Dependencia core (nao optional) | Optional com fallback para stdlib | Logging e transversal. Todo componente precisa. Tornar optional geraria `if structlog_available` em todo lugar. |

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `get_logger` retorna logger funcional | `python -c "from theo.logging import get_logger; log = get_logger('test'); log.info('ok')"` |
| 2 | Formato JSON quando `THEO_LOG_FORMAT=json` | Saida e JSON valido parseavel |
| 3 | Formato console quando `THEO_LOG_FORMAT=console` (default dev) | Saida e legivel com cores |
| 4 | Campos obrigatorios presentes | Todo log contem `timestamp`, `level`, `event`, `component` |
| 5 | Testes passam | `.venv/bin/python -m pytest tests/unit/test_logging.py` |
| 6 | mypy passa | `.venv/bin/python -m mypy src/theo/logging.py` |

**Estimativa**: S (0.5 dia)

---

### Iniciativa I2 -- gRPC Worker Server (Servicer)

**Objetivo**: Implementar o processo worker que roda um servidor gRPC implementando os RPCs `TranscribeFile` e `Health`. Este e o "esqueleto" do worker que sera preenchido com a engine real em I3.

**Responsavel**: Viktor Sorokin

**Dependencia**: I1 (proto compilado), I-AUX (logging)

**Contexto**: O worker e um processo Python standalone que:
1. Inicia um servidor gRPC na porta especificada
2. Carrega o backend STT (recebido como dependencia)
3. Implementa os RPCs definidos no proto
4. Responde health checks
5. Faz graceful shutdown via signal handlers

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| `STTWorkerServicer` implementando `TranscribeFile` e `Health` | `TranscribeStream` (M5) |
| `__main__.py` como entrypoint executavel | `Cancel` (M5) |
| Signal handlers (SIGTERM, SIGINT) | gRPC reflection (nice-to-have, nao bloqueante) |
| Conversao proto messages <-> tipos Theo | Metricas Prometheus no worker |
| `grpc.aio.server()` (async) | Autenticacao gRPC (mutual TLS) |
| Logging estruturado via `theo.logging` | Load balancing entre workers |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/workers/stt/servicer.py` | `STTWorkerServicer` -- implementa os RPCs |
| `src/theo/workers/stt/main.py` | Entrypoint: parse args, configura logger, inicia gRPC server |
| `src/theo/workers/stt/__main__.py` | Permite `python -m theo.workers.stt` |
| `src/theo/workers/stt/converters.py` | Funcoes de conversao proto <-> Theo types |

**Arquitetura do Worker (processo standalone)**:

```
┌────────────────────────────────────────────────────┐
│            WORKER PROCESS (subprocess)              │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │          gRPC Server (grpc.aio)               │  │
│  │                                               │  │
│  │  ┌──────────────────────────────────────┐    │  │
│  │  │      STTWorkerServicer               │    │  │
│  │  │                                      │    │  │
│  │  │  TranscribeFile(req) -> resp         │    │  │
│  │  │    1. Converte proto -> Theo types   │    │  │
│  │  │    2. Chama backend.transcribe_file  │    │  │
│  │  │    3. Converte BatchResult -> proto  │    │  │
│  │  │                                      │    │  │
│  │  │  Health(req) -> resp                 │    │  │
│  │  │    1. Chama backend.health()         │    │  │
│  │  │    2. Retorna status                 │    │  │
│  │  └──────────────────────────────────────┘    │  │
│  │                    |                          │  │
│  │                    v                          │  │
│  │  ┌──────────────────────────────────────┐    │  │
│  │  │      STTBackend (injetado)           │    │  │
│  │  │      (ex: FasterWhisperBackend)      │    │  │
│  │  └──────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  Signal Handlers: SIGTERM -> graceful shutdown       │
│  Logging: structlog com component="stt_worker"       │
└────────────────────────────────────────────────────┘
```

**Entrypoint `main.py` -- interface CLI do worker**:

```
python -m theo.workers.stt.main \
  --port 50051 \
  --engine faster-whisper \
  --model-path /models/faster-whisper-large-v3 \
  --compute-type float16 \
  --device auto
```

**Conversao proto <-> Theo types (`converters.py`)**:

O servicer nao deve manipular protobuf diretamente alem da fronteira de conversao. As funcoes de conversao sao:

```
proto_request_to_transcribe_params(request: TranscribeFileRequest) -> dict
    Extrai language, temperature, hot_words, etc. do request protobuf.

batch_result_to_proto_response(result: BatchResult) -> TranscribeFileResponse
    Converte BatchResult (Theo type) para TranscribeFileResponse (protobuf).

health_dict_to_proto_response(health: dict[str, str]) -> HealthResponse
    Converte dict de health para HealthResponse (protobuf).
```

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Worker inicia e escuta na porta | `python -m theo.workers.stt.main --port 50051 --engine mock` (nao crashar) |
| 2 | Health responde via gRPC | `grpcurl -plaintext localhost:50051 theo.stt.STTWorker/Health` retorna JSON com `status` |
| 3 | TranscribeFile responde (com mock backend) | Teste unitario com mock backend que retorna `BatchResult` fixo |
| 4 | TranscribeStream retorna UNIMPLEMENTED | Chamada ao RPC retorna status code `UNIMPLEMENTED` |
| 5 | Graceful shutdown funciona | Enviar SIGTERM -> worker encerra em <5s |
| 6 | Logs estruturados presentes | Logs contem `component`, `event`, campos de contexto |
| 7 | mypy passa | `.venv/bin/python -m mypy src/theo/workers/stt/` |

**Nota sobre "mock backend"**: Para I2, o servicer deve funcionar com qualquer `STTBackend`. Nos testes, usaremos um `MockSTTBackend` que retorna resultados fixos. O `FasterWhisperBackend` real vem em I3. Isso garante que o servicer e testavel sem instalar Faster-Whisper.

**Estimativa**: M (2-3 dias)

---

### Iniciativa I3 -- FasterWhisperBackend

**Objetivo**: Implementar a classe `FasterWhisperBackend` que faz bridge entre a interface `STTBackend` (Theo) e a biblioteca `faster-whisper`.

**Responsavel**: Sofia Castellani

**Dependencia**: I2 (servicer que chama o backend), I-AUX (logging)

**Contexto**: Este e o componente que finalmente interage com uma engine de inferencia real. A qualidade desta implementacao determina se a interface `STTBackend` definida em M1 funciona na pratica.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| `FasterWhisperBackend` implementando `STTBackend` | `transcribe_stream` (M5) |
| `load()`: instancia `WhisperModel` com configs | `BatchedInferencePipeline` (M4) |
| `transcribe_file()`: transcreve bytes -> `BatchResult` | Hot words via `initial_prompt` (funcional mas nao otimizado) |
| `capabilities()`: reporta o que a engine suporta | Preprocessamento de audio (M4) |
| `unload()`: libera recursos | Post-processamento de texto (M4) |
| `health()`: reporta status | Multi-GPU |
| Conversao audio bytes -> numpy float32 | |
| Conversao Faster-Whisper types -> Theo types | |
| Testes unitarios com mock de `WhisperModel` | |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/workers/stt/faster_whisper.py` | `FasterWhisperBackend` -- implementacao concreta |

**Mapeamento Faster-Whisper -> Theo types**:

| Faster-Whisper | Theo | Notas |
|----------------|------|-------|
| `Segment.text` | `SegmentDetail.text` | Direto |
| `Segment.start` | `SegmentDetail.start` | Segundos (float) |
| `Segment.end` | `SegmentDetail.end` | Segundos (float) |
| `Segment.avg_logprob` | `SegmentDetail.avg_logprob` | Direto |
| `Segment.no_speech_prob` | `SegmentDetail.no_speech_prob` | Direto |
| `Word.word` | `WordTimestamp.word` | Direto |
| `Word.start` | `WordTimestamp.start` | Segundos (float) |
| `Word.end` | `WordTimestamp.end` | Segundos (float) |
| `TranscriptionInfo.language` | `BatchResult.language` | Direto |
| `TranscriptionInfo.duration` | `BatchResult.duration` | Segundos (float) |

**Conversao de audio (bytes -> numpy)**:

```
Input:  bytes (PCM 16-bit, 16kHz, mono) -- recebido via gRPC
  |
  v
numpy.frombuffer(audio_data, dtype=numpy.int16)
  |
  v
array.astype(numpy.float32) / 32768.0   -- normaliza para [-1.0, 1.0]
  |
  v
WhisperModel.transcribe(audio_array, ...)
```

**Tratamento de erros**:

| Erro Faster-Whisper | Theo Exception | Quando |
|--------------------|----------------|--------|
| Arquivo de modelo nao encontrado | `ModelLoadError` | `load()` |
| CUDA out of memory | `ModelLoadError` | `load()` |
| Audio vazio ou corrompido | `AudioFormatError` | `transcribe_file()` |
| Timeout de inferencia | `WorkerTimeoutError` | `transcribe_file()` |

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `FasterWhisperBackend` implementa todos os metodos abstratos de `STTBackend` | `mypy` passa sem erros |
| 2 | `architecture` retorna `ENCODER_DECODER` | Teste unitario |
| 3 | `load()` cria `WhisperModel` com config do manifesto | Teste com mock de `WhisperModel` |
| 4 | `transcribe_file()` retorna `BatchResult` valido | Teste com mock que retorna segments conhecidos |
| 5 | `transcribe_stream()` levanta `NotImplementedError` | Teste unitario |
| 6 | `unload()` libera referencia ao modelo | Teste: `health()` retorna `error` apos `unload()` |
| 7 | Conversao bytes->numpy esta correta | Teste: bytes conhecidos -> array esperado |
| 8 | Nenhum tipo do Faster-Whisper vaza para fora | Todos os retornos sao tipos Theo (`BatchResult`, `EngineCapabilities`) |

**Estimativa**: M (2-3 dias)

---

### Iniciativa I4 -- Worker Manager (Subprocess Lifecycle)

**Objetivo**: Implementar o `WorkerManager` que spawna, monitora e reinicia subprocessos worker.

**Responsavel**: Viktor Sorokin

**Dependencia**: I2 (worker que pode ser spawnado), I-AUX (logging)

**Contexto**: O Worker Manager e o componente do runtime (processo principal) que gerencia o lifecycle dos workers (subprocessos). Em M2, gerencia 1 worker. Em M3+, o Scheduler usa o Manager para rotear requests.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| `WorkerManager` com `spawn`, `stop`, `get_worker` | Multi-worker por modelo (M9) |
| `WorkerHandle` dataclass com estado | Alocacao round-robin (M3) |
| Health probe com exponential backoff | gRPC client pool |
| Crash detection via `process.poll()` | Dynamic batching (M9) |
| Auto-restart com backoff (max 3 em 60s) | |
| Graceful shutdown (SIGTERM -> SIGKILL) | |
| gRPC health client (chama `Health` RPC) | |
| Testes unitarios com subprocess mockado | |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/workers/manager.py` | `WorkerManager` e `WorkerHandle` |

**Estados do WorkerHandle**:

```
STARTING ──> READY ──> BUSY ──> READY (ciclo)
    |           |        |
    v           v        v
 CRASHED    STOPPING  CRASHED
    |           |        |
    v           v        v
 STARTING   STOPPED   STARTING (auto-restart)
(restart)              (restart)
```

| Estado | Significado | Transicoes possiveis |
|--------|------------|---------------------|
| `STARTING` | Processo spawnado, aguardando health ok | `READY` (health ok), `CRASHED` (exit code) |
| `READY` | Worker disponivel para receber requests | `BUSY` (request em andamento), `STOPPING` (shutdown), `CRASHED` |
| `BUSY` | Worker processando uma request | `READY` (request concluida), `CRASHED` (crash durante request) |
| `STOPPING` | SIGTERM enviado, aguardando encerramento | `STOPPED` (encerrou), `CRASHED` (nao encerrou, SIGKILL) |
| `STOPPED` | Worker encerrado normalmente | Terminal |
| `CRASHED` | Worker encerrou com erro | `STARTING` (auto-restart) ou terminal (max restarts) |

**Ciclo de vida**:

```
Manager.spawn_worker(model_name, port, engine_config)
  |
  v
subprocess.Popen(["python", "-m", "theo.workers.stt.main", "--port", str(port), ...])
  |
  v
WorkerHandle(status=STARTING, process=proc, port=port)
  |
  v
[async loop] Health probe com backoff (500ms, 1s, 2s, 4s, timeout 30s)
  |
  v
Health response OK -> status = READY
  |
  v
[async loop] Monitor process.poll() a cada 1s
  |
  v
process.returncode != None -> CRASHED -> auto_restart(backoff)
```

**gRPC Health Client**:

O Manager precisa de um client gRPC para chamar `Health` no worker. Criar um client minimo:

```python
async def check_worker_health(port: int, timeout: float = 2.0) -> dict[str, str]:
    """Chama Health RPC no worker e retorna o resultado."""
```

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | `spawn_worker` cria subprocess e retorna `WorkerHandle` | Teste: handle retornado com status `STARTING` |
| 2 | Health probe transita para `READY` | Teste: mock de gRPC health retorna ok -> status muda para `READY` |
| 3 | Crash detectado via `process.poll()` | Teste: processo que faz `sys.exit(1)` -> status muda para `CRASHED` |
| 4 | Auto-restart com backoff | Teste: apos crash, novo processo spawnado apos delay |
| 5 | Max restarts respeitado | Teste: apos 3 crashes em 60s, nao tenta mais restart |
| 6 | `stop_worker` envia SIGTERM | Teste: processo recebe SIGTERM e encerra |
| 7 | SIGKILL apos grace period | Teste: processo que ignora SIGTERM recebe SIGKILL apos 5s |
| 8 | Logs estruturados em todas as transicoes | Logs contem `worker_id`, `status`, `event` |
| 9 | mypy passa | `.venv/bin/python -m mypy src/theo/workers/manager.py` |

**Nota sobre testes**: Testes do Worker Manager NAO devem spawnar processos reais do worker. Usar `subprocess.Popen` mockado que simula exit codes e stdout. Testes com processo real sao de integracao (I6).

**Estimativa**: M (2-3 dias)

---

### Iniciativa I5 -- Testes Unitarios

**Objetivo**: Cobrir todos os componentes do M2 com testes unitarios usando mocks para dependencias externas (Faster-Whisper, gRPC, subprocess).

**Responsavel**: Sofia Castellani (testes de backend e converters), Viktor Sorokin (testes de servicer e manager), Andre Oliveira (testes de logging)

**Dependencia**: I1, I2, I3, I4, I-AUX (testa os componentes implementados)

**Contexto**: Os testes do M2 devem funcionar sem instalar Faster-Whisper e sem iniciar processos gRPC reais. Tudo e mockado. Testes de integracao (com modelo real) ficam em I6.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| Testes do `FasterWhisperBackend` com mock de `WhisperModel` | Testes com modelo Faster-Whisper real (I6) |
| Testes do `STTWorkerServicer` com mock backend | Testes com gRPC server real (I6) |
| Testes do `WorkerManager` com subprocess mockado | Testes com subprocess real (I6) |
| Testes dos conversores proto <-> types | |
| Testes de configuracao de logging | |
| Fixtures compartilhadas (`conftest.py`) | |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `tests/unit/test_faster_whisper_backend.py` | Testes do backend com mock da engine |
| `tests/unit/test_stt_servicer.py` | Testes do servicer gRPC com mock backend |
| `tests/unit/test_worker_manager.py` | Testes do manager com subprocess mockado |
| `tests/unit/test_proto_converters.py` | Testes das funcoes de conversao |
| `tests/unit/test_logging.py` | Testes de structured logging |
| Atualizacao `tests/conftest.py` | Novas fixtures para M2 |

**Fixtures novas no conftest.py**:

```python
@pytest.fixture
def sample_batch_result() -> BatchResult:
    """BatchResult de exemplo para testes."""

@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Audio PCM 16-bit 16kHz mono de exemplo (tom 440Hz, 1s)."""

@pytest.fixture
def mock_stt_backend() -> STTBackend:
    """Backend STT mockado que retorna resultados fixos."""
```

**Cenarios de teste chave**:

| Componente | Cenario | Tipo |
|-----------|---------|------|
| FasterWhisperBackend | `load()` cria WhisperModel com config correta | Unit (mock WhisperModel) |
| FasterWhisperBackend | `transcribe_file()` converte types corretamente | Unit (mock WhisperModel.transcribe) |
| FasterWhisperBackend | `transcribe_file()` com audio vazio levanta `AudioFormatError` | Unit |
| FasterWhisperBackend | `transcribe_stream()` levanta `NotImplementedError` | Unit |
| FasterWhisperBackend | `unload()` libera referencia | Unit |
| FasterWhisperBackend | `health()` retorna status correto (antes e apos load) | Unit |
| FasterWhisperBackend | Conversao bytes -> numpy float32 correta | Unit |
| STTWorkerServicer | `Health` retorna status do backend | Unit (mock backend) |
| STTWorkerServicer | `TranscribeFile` chama backend e converte resposta | Unit (mock backend) |
| STTWorkerServicer | `TranscribeStream` retorna UNIMPLEMENTED | Unit |
| STTWorkerServicer | Request com audio vazio retorna erro gRPC | Unit |
| Converters | proto request -> dict de params | Unit |
| Converters | BatchResult -> proto response | Unit |
| Converters | Segmentos com word timestamps | Unit |
| Converters | Segmentos sem word timestamps (None) | Unit |
| WorkerManager | `spawn_worker` retorna handle STARTING | Unit (mock Popen) |
| WorkerManager | Health probe transita para READY | Unit (mock gRPC client) |
| WorkerManager | Crash detection (exit code != 0) | Unit (mock Popen.poll) |
| WorkerManager | Auto-restart apos crash | Unit (mock) |
| WorkerManager | Max restarts excedido -> nao tenta mais | Unit (mock) |
| WorkerManager | `stop_worker` envia SIGTERM | Unit (mock Popen.terminate) |
| Logging | `get_logger` retorna logger funcional | Unit |
| Logging | Formato JSON contem campos obrigatorios | Unit |
| Logging | `bind()` adiciona contexto | Unit |

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Todos os testes passam | `.venv/bin/python -m pytest tests/unit/ -v` |
| 2 | Testes nao dependem de Faster-Whisper instalado | Testes passam com `pip install -e ".[dev]"` (sem `[faster-whisper]`) |
| 3 | Testes nao iniciam subprocessos reais | Nenhum `subprocess.Popen` real e chamado |
| 4 | Testes nao iniciam servidor gRPC real | Nenhum server gRPC e iniciado |
| 5 | Cobertura dos fluxos criticos | Cada componente tem testes de happy path e erro |
| 6 | Testes sao independentes entre si | Ordem de execucao nao importa |

**Estimativa**: M (2-3 dias, distribuido entre a equipe)

---

### Iniciativa I6 -- Teste de Integracao

**Objetivo**: Validar que o worker real funciona end-to-end: subprocess gRPC com Faster-Whisper tiny transcrevendo audio de verdade.

**Responsavel**: Viktor Sorokin

**Dependencia**: I1, I2, I3, I4, I5 (todos os componentes implementados e testados unitariamente)

**Contexto**: Este e o "teste de fogo" do M2. Um worker real inicia como subprocess, carrega `faster-whisper-tiny`, recebe audio via gRPC, e retorna transcricao. Se isso funcionar, o M2 esta entregue.

**Escopo**:

| Incluido | FORA do escopo |
|----------|---------------|
| Worker real como subprocess | Testes em GPU (rodar em CPU) |
| Faster-Whisper tiny em CPU | Testes de performance/latencia |
| Audio real com fala (fixture WAV com fala em ingles ou portugues) | Testes de longa duracao (30 min) |
| gRPC client chamando TranscribeFile e Health | Testes de streaming |
| Worker Manager spawnando e parando worker real | Testes de restart automatico com modelo real |
| Marcador `@pytest.mark.integration` | |

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `tests/integration/test_worker_e2e.py` | Testes end-to-end do worker |
| `tests/fixtures/audio/speech_en_short.wav` | Audio WAV com fala em ingles (~3-5s) |
| Atualizacao `.github/workflows/ci.yml` | Job separado para integracao (optional, manual trigger) |

**Pre-requisitos para rodar os testes de integracao**:

```bash
# Instalar com extras necessarios
.venv/bin/pip install -e ".[dev,grpc,faster-whisper]"

# Modelo sera baixado automaticamente pelo faster-whisper na primeira execucao
# (faster-whisper-tiny ~= 75MB download)
```

**Cenarios de teste**:

| Cenario | O que valida |
|---------|-------------|
| Worker inicia e responde Health | Subprocess + gRPC server + backend load |
| Health retorna model_name e engine | Metadados propagados corretamente |
| TranscribeFile com audio real retorna texto | Pipeline completo: audio -> engine -> texto |
| TranscribeFile com audio de silencio retorna texto vazio | Engine lida com no-speech |
| Worker Manager spawna e detecta worker READY | Manager + health probe + real subprocess |
| Worker Manager para worker gracefully | SIGTERM -> cleanup -> encerramento |

**Fixture de audio com fala**:

Para os testes de integracao, precisamos de um WAV curto com fala real (nao tom senoidal). Opcoes:
1. Gerar via TTS (ex: `pyttsx3` ou qualquer TTS offline) -- a frase "Hello, how can I help you?" em ingles
2. Gravar manualmente um WAV de ~3s
3. Usar audio de dominio publico (LibriSpeech)

Recomendacao: Gerar via TTS para determinismo. O audio gerado e commitado no repositorio.

**Definition of Done**:

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Worker inicia, responde Health, transcreve arquivo | `.venv/bin/python -m pytest tests/integration/ -v -m integration` |
| 2 | Transcricao retorna texto nao-vazio para audio com fala | Assert `len(result.text) > 0` |
| 3 | Health retorna `{"status": "ok", "model_name": "...", "engine": "faster-whisper"}` | Assert nos campos |
| 4 | Worker encerra limpo apos stop | Processo nao existe mais apos stop |
| 5 | Testes marcados com `@pytest.mark.integration` | Nao rodam em `pytest tests/unit/` |
| 6 | CI tem job separado para integracao (manual trigger) | `workflow_dispatch` no CI |

**Estimativa**: M (1-2 dias)

**Nota**: Os testes de integracao NAO rodam no CI em cada push (requerem download de modelo). Rodam em `workflow_dispatch` ou em schedule noturno. O criterio de sucesso do M2 exige que passem pelo menos uma vez antes de fechar o milestone.

---

## 5. Sequencia de Implementacao

### 5.1 Ordem de Execucao

```
Semana 1 (dias 1-2):
  ┌─────────────────────────────────────────────────┐
  │ I1: Compilacao de protobuf + script              │
  │     (Viktor) -- 0.5 dia                         │
  ├─────────────────────────────────────────────────┤
  │ I-AUX: Structured logging foundation             │  <- paralelo com I1
  │     (Andre) -- 0.5 dia                          │
  ├─────────────────────────────────────────────────┤
  │ I2: gRPC Worker Server (servicer + main)         │  <- depende de I1
  │     (Viktor) -- 2-3 dias                        │
  └─────────────────────────────────────────────────┘

Semana 2 (dias 3-5):
  ┌─────────────────────────────────────────────────┐
  │ I3: FasterWhisperBackend                         │  <- paralelo com I4
  │     (Sofia) -- 2-3 dias                         │
  ├─────────────────────────────────────────────────┤
  │ I4: Worker Manager                               │  <- paralelo com I3
  │     (Viktor) -- 2-3 dias                        │
  └─────────────────────────────────────────────────┘

Semana 3 (dias 6-8):
  ┌─────────────────────────────────────────────────┐
  │ I5: Testes unitarios (consolidacao)              │
  │     (Todos) -- 2-3 dias                         │
  ├─────────────────────────────────────────────────┤
  │ I6: Teste de integracao                          │
  │     (Viktor) -- 1-2 dias                        │
  ├─────────────────────────────────────────────────┤
  │ Review cruzado, ajustes finais, CHANGELOG        │
  │     (Todos) -- 1 dia                            │
  └─────────────────────────────────────────────────┘
```

### 5.2 Justificativa da Sequencia

| Ordem | Iniciativa | Por que nesta posicao |
|-------|------------|----------------------|
| 1 | I1 (proto) | Tudo depende do proto compilado. Sem `stt_worker_pb2.py`, nenhum gRPC funciona. |
| 1' | I-AUX (logging) | Paralelo com I1. Logging e necessario antes de implementar componentes que logam. |
| 2 | I2 (servicer) | Depende de I1. O servicer e o esqueleto do worker -- precisa existir antes de ter um backend real (I3) ou um manager que spawne (I4). |
| 3 | I3 (backend) | Depende de I2 (precisa dos tipos proto para conversao e do servicer para integrar). Pode ser paralelo com I4. |
| 3' | I4 (manager) | Depende de I2 (spawna o worker implementado em I2). Pode ser paralelo com I3. |
| 4 | I5 (testes unit) | Testes sao escritos junto com cada componente, mas consolidados e revisados ao final. |
| 5 | I6 (integracao) | Ultimo. Depende de tudo. E a validacao final. |

### 5.3 Paralelismo

Com 2 pessoas (Sofia + Viktor) e Andre pontual:

| Dia | Sofia | Viktor | Andre |
|-----|-------|--------|-------|
| 1 | -- | I1 (proto) | I-AUX (logging) |
| 2 | Review I1 | I2 (servicer) | Suporte I2 (CI proto sync) |
| 3 | I3 (backend) | I2 (servicer cont.) | -- |
| 4 | I3 (backend cont.) | I4 (manager) | -- |
| 5 | I5 (testes backend + converters) | I4 (manager cont.) | -- |
| 6 | I5 (testes servicer) | I5 (testes manager) | -- |
| 7 | Review cruzado | I6 (integracao) | Review CI |
| 8 | Ajustes finais | I6 (integracao cont.) | CHANGELOG |

**Andre** participa pontualmente em I-AUX (logging), CI (proto sync step), e CHANGELOG.

---

## 6. Estrutura de Arquivos Apos M2

```
src/theo/
├── __init__.py
├── py.typed
├── _types.py                          # [M1] Tipos base
├── exceptions.py                      # [M1] Exceptions tipadas
├── logging.py                         # [M2-NEW] Structured logging
│
├── config/                            # [M1] Configuracao
│   ├── __init__.py
│   ├── manifest.py
│   ├── preprocessing.py
│   └── postprocessing.py
│
├── workers/                           # [M2-NEW] Worker management
│   ├── __init__.py
│   ├── manager.py                     # [M2-NEW] WorkerManager + WorkerHandle
│   └── stt/
│       ├── __init__.py
│       ├── interface.py               # [M1] STTBackend ABC
│       ├── faster_whisper.py          # [M2-NEW] FasterWhisperBackend
│       ├── servicer.py                # [M2-NEW] STTWorkerServicer (gRPC)
│       ├── converters.py             # [M2-NEW] Proto <-> Theo type converters
│       ├── main.py                    # [M2-NEW] Worker entrypoint
│       └── __main__.py                # [M2-NEW] python -m theo.workers.stt
│
├── proto/                             # [M1 def, M2 compiled]
│   ├── __init__.py                    # [M2-UPDATED] Re-exports
│   ├── stt_worker.proto               # [M1] Definicao
│   ├── stt_worker_pb2.py             # [M2-NEW] Gerado
│   └── stt_worker_pb2_grpc.py        # [M2-NEW] Gerado
│
├── server/                            # [vazio] M3
├── scheduler/                         # [vazio] M3
├── registry/                          # [vazio] M3
├── preprocessing/                     # [vazio] M4
├── postprocessing/                    # [vazio] M4
├── session/                           # [vazio] M6
└── cli/                               # [vazio] M3

tests/
├── unit/
│   ├── test_types.py                  # [M1]
│   ├── test_exceptions.py            # [M1]
│   ├── test_manifest.py              # [M1]
│   ├── test_config.py                # [M1]
│   ├── test_stt_interface.py         # [M1]
│   ├── test_faster_whisper_backend.py # [M2-NEW]
│   ├── test_stt_servicer.py          # [M2-NEW]
│   ├── test_worker_manager.py        # [M2-NEW]
│   ├── test_proto_converters.py      # [M2-NEW]
│   └── test_logging.py               # [M2-NEW]
├── integration/
│   └── test_worker_e2e.py            # [M2-NEW]
└── fixtures/
    ├── audio/
    │   ├── sample_16khz.wav           # [M1] Tom senoidal
    │   ├── sample_8khz.wav            # [M1] Tom senoidal
    │   ├── sample_44khz.wav           # [M1] Tom senoidal
    │   └── speech_en_short.wav        # [M2-NEW] Fala real (~3-5s)
    └── manifests/                     # [M1]

scripts/
└── generate_proto.sh                  # [M2-NEW]
```

---

## 7. Definition of Done do M2

O M2 esta completo quando TODOS os itens abaixo forem verdadeiros:

### 7.1 Funcional

| # | Criterio | Comando de Verificacao |
|---|----------|----------------------|
| 1 | Proto compilado e importavel | `python -c "from theo.proto.stt_worker_pb2 import TranscribeFileRequest; print('ok')"` |
| 2 | Worker inicia standalone | `python -m theo.workers.stt.main --port 50051 --engine mock` (nao crashar, logar startup) |
| 3 | Health check via gRPC | `grpcurl -plaintext localhost:50051 theo.stt.STTWorker/Health` -> JSON com `status` |
| 4 | TranscribeFile via gRPC (com mock) | Teste unitario: request com audio -> resposta com texto |
| 5 | FasterWhisperBackend implementa STTBackend | `mypy` valida subclass |
| 6 | Worker Manager spawna e para worker | Teste unitario com mock subprocess |
| 7 | Crash detection funciona | Teste unitario: mock process exit -> CRASHED |
| 8 | Structured logging funcional | Logs contem campos estruturados (timestamp, component, event) |

### 7.2 Integracao (executar pelo menos uma vez)

| # | Criterio | Comando de Verificacao |
|---|----------|----------------------|
| 1 | Worker real com faster-whisper-tiny transcreve audio | `.venv/bin/python -m pytest tests/integration/ -v -m integration` |
| 2 | Transcricao retorna texto nao-vazio | Assert no teste |
| 3 | Worker Manager spawna worker real e detecta READY | Teste de integracao |

### 7.3 Artefatos Existentes

| # | Artefato | Localizacao |
|---|----------|-------------|
| 1 | Proto compilado (pb2 + pb2_grpc) | `src/theo/proto/stt_worker_pb2.py`, `stt_worker_pb2_grpc.py` |
| 2 | Script de geracao | `scripts/generate_proto.sh` |
| 3 | Worker servicer | `src/theo/workers/stt/servicer.py` |
| 4 | Worker entrypoint | `src/theo/workers/stt/main.py` |
| 5 | FasterWhisperBackend | `src/theo/workers/stt/faster_whisper.py` |
| 6 | Conversores proto <-> types | `src/theo/workers/stt/converters.py` |
| 7 | Worker Manager | `src/theo/workers/manager.py` |
| 8 | Structured logging | `src/theo/logging.py` |
| 9 | Fixture audio com fala | `tests/fixtures/audio/speech_en_short.wav` |
| 10 | 5 novos arquivos de teste | `tests/unit/test_faster_whisper_backend.py`, etc. |

### 7.4 Qualidade

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | mypy strict passa | `.venv/bin/python -m mypy src/` |
| 2 | ruff check passa | `.venv/bin/python -m ruff check src/ tests/` |
| 3 | ruff format passa | `.venv/bin/python -m ruff format --check src/ tests/` |
| 4 | Testes unitarios passam | `.venv/bin/python -m pytest tests/unit/ -v` |
| 5 | Proto em sync com gerado | `bash scripts/generate_proto.sh && git diff --exit-code src/theo/proto/` |
| 6 | Nenhum tipo do Faster-Whisper vaza para fora do backend | Review: todos os retornos sao tipos Theo |
| 7 | Testes unitarios nao requerem Faster-Whisper instalado | Passam com `pip install -e ".[dev,grpc]"` (sem `[faster-whisper]`) |
| 8 | Todo componente novo tem docstring | Review manual |

### 7.5 Documentacao

| # | Criterio |
|---|----------|
| 1 | CHANGELOG.md atualizado com entradas do M2 |
| 2 | CLAUDE.md atualizado (se padroes de logging mudaram) |
| 3 | Docstrings em todas as interfaces publicas |

### 7.6 Checklist Final (executar em sequencia)

```bash
# 1. Instalacao com extras necessarios
.venv/bin/pip install -e ".[dev,grpc]"

# 2. Proto em sync
bash scripts/generate_proto.sh && git diff --exit-code src/theo/proto/

# 3. Qualidade de codigo
.venv/bin/python -m ruff format --check src/ tests/
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m mypy src/

# 4. Testes unitarios (nao requer faster-whisper)
.venv/bin/python -m pytest tests/unit/ -v

# 5. Imports dos novos modulos
.venv/bin/python -c "
from theo.proto.stt_worker_pb2 import TranscribeFileRequest, HealthRequest
from theo.workers.stt.servicer import STTWorkerServicer
from theo.workers.stt.converters import batch_result_to_proto_response
from theo.workers.manager import WorkerManager, WorkerHandle
from theo.logging import get_logger
print('Todos os modulos M2 importaveis com sucesso')
"

# 6. Testes de integracao (requer faster-whisper)
.venv/bin/pip install -e ".[dev,grpc,faster-whisper]"
.venv/bin/python -m pytest tests/integration/ -v -m integration
```

Se todos os comandos acima passam sem erros, M2 esta completo.

---

## 8. O Que NAO Esta no Escopo do M2

Aplicando YAGNI rigorosamente:

| Item | Por que nao | Quando entra |
|------|-------------|-------------|
| gRPC streaming bidirecional (`TranscribeStream`) | Complexidade de backpressure e cancelamento. M2 e batch-only. | M5 |
| Cancelamento via gRPC (`Cancel` RPC) | So faz sentido com streaming ou requests longas. Em batch, o overhead nao justifica. | M5 |
| gRPC reflection | Nice-to-have para debug. Nao bloqueante. | M3 (se necessario) |
| Metricas Prometheus no worker | Sem servidor HTTP no worker para expor metricas. Metricas vem em M3 via API Server. | M3 |
| API REST (FastAPI) | Nenhum endpoint HTTP. O worker e acessivel apenas via gRPC. | M3 |
| Model Registry | O worker recebe config via args de linha de comando. Registry resolve modelo por nome em M3. | M3 |
| Preprocessing de audio | O worker recebe audio ja em PCM 16-bit 16kHz. Preprocessing e responsabilidade do runtime (M4). | M4 |
| Post-processing de texto | O worker retorna texto cru. ITN e formatting sao do runtime (M4). | M4 |
| Multi-worker por modelo | M2 e 1 worker = 1 modelo. Multi-worker e Scheduler Avancado. | M9 |
| CLI (`theo serve`, `theo transcribe`) | Sem API Server, sem CLI. Worker e acessivel apenas via gRPC direto. | M3 |
| Docker / Dockerfile | Zero valor sem API Server. | M3 |
| GPU optimization | M2 roda em CPU. GPU e otimizacao, nao funcionalidade. | Qualquer M apos M2 |
| Hot words otimizados | `initial_prompt` basico funciona. Otimizacao de hot words e M6. | M6 |
| Mutual TLS no gRPC | Worker roda em localhost. Autenticacao entre runtime e worker e desnecessaria. | Nunca (escopo atual) |

---

## 9. Riscos Especificos do M2

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Interface `STTBackend.transcribe_file` nao acomoda output do Faster-Whisper | Media | Alto | Validar mapeamento de tipos ANTES de implementar. Se `SegmentDetail` ou `BatchResult` precisar de campos extras, ajustar os tipos agora (custo baixo, zero implementacoes alem de M1 tests). |
| R2 | `grpc.aio.server()` tem bugs sutis em combinacao com `asyncio.run()` | Baixa | Medio | Usar pattern testado: `asyncio.run(serve())` com `server.wait_for_termination()`. Testar shutdown graceful extensivamente. |
| R3 | Faster-Whisper `model.transcribe()` bloqueia o event loop async | Alta | Alto | Rodar em `run_in_executor(None, model.transcribe, ...)`. Validar que o gRPC server continua respondendo Health enquanto inference roda. |
| R4 | `subprocess.Popen` tem comportamento diferente em Linux vs macOS | Baixa | Baixo | CI roda em Linux (ubuntu-latest). Desenvolvedores em macOS podem encontrar diferencas em signal handling. Documentar. |
| R5 | Download do modelo `faster-whisper-tiny` no CI e lento (~75MB) | Media | Baixo | Cache do modelo no CI via `actions/cache`. Testes de integracao em job separado com trigger manual. |
| R6 | `structlog` pode conflitar com logging de bibliotecas que usam stdlib | Baixa | Baixo | `structlog` foi projetado para coexistir com stdlib. Configurar via `structlog.configure()` sem substituir `logging.root`. |
| R7 | Arquivos proto gerados ficam desatualizados (dev esquece de regenerar) | Media | Medio | CI step `bash scripts/generate_proto.sh && git diff --exit-code` falha se desatualizado. Documentar no CONTRIBUTING. |

---

## 10. Transicao M2 -> M3

Ao completar M2, o time deve ter:

1. **Worker funcional como subprocess** -- carrega modelo, responde gRPC, transcreve audio batch. O worker e o "motor" que todo milestone futuro usa.

2. **Confianca na interface STTBackend** -- validada com Faster-Whisper REAL (nao mock). Se `transcribe_file` precisou de ajustes, eles foram feitos.

3. **Worker Manager confiavel** -- spawn, health probe, crash detection, restart. O Scheduler (M3) constroi sobre o Manager.

4. **Padrao de logging estabelecido** -- todo componente novo ja nasce com structured logging. Nao ha retrofitting em M5+.

5. **Proto compilado e em sync** -- CI garante que o proto e os gerados estao alinhados.

### O que M3 constroi sobre M2:

```
M2 (Worker gRPC)                    M3 (API Batch + CLI)
├── Worker funciona standalone       ├── FastAPI conecta ao Worker via gRPC client
├── Manager spawna/para worker       ├── Scheduler usa Manager para rotear requests
├── Health check via gRPC            ├── /health agrega status de workers
├── TranscribeFile funciona          ├── POST /transcriptions chama TranscribeFile
└── Logging estruturado              └── CLI (theo serve/transcribe) orquestra tudo
```

O primeiro commit de M3 sera: criar o FastAPI app com o endpoint `POST /v1/audio/transcriptions` que recebe audio, delega ao Worker via gRPC client, e retorna a resposta formatada. O Worker Manager (M2) spawna o worker. O Scheduler (M3) decide qual worker atende.

A transicao e limpa porque M2 expos tudo via gRPC -- M3 precisa apenas de um gRPC client para chamar `TranscribeFile`.

---

*Documento gerado pelo Time de Arquitetura (ARCH). Sera atualizado conforme a implementacao do M2 avanca.*
