# Changelog

Todas as mudancas relevantes deste projeto serao documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/),
e este projeto adere ao [Versionamento Semantico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
- Estrutura de projeto com `pyproject.toml`, src layout e extras opcionais (#M1-I1)
- Tooling: ruff (lint+format), mypy (strict), pytest com pytest-asyncio (#M1-I2)
- CI basico via GitHub Actions com matrix Python 3.11/3.12 (#M1-I3)
- Tipos fundamentais: `STTArchitecture`, `ModelType`, `SessionState`, `VADSensitivity`, `ResponseFormat`, `TranscriptSegment`, `BatchResult`, `EngineCapabilities` (#M1-I4)
- Hierarquia de exceptions tipadas por dominio: `TheoError`, `ConfigError`, `ModelError`, `WorkerError`, `AudioError`, `SessionError` (#M1-I5)
- Parsing e validacao de manifestos `theo.yaml` via `ModelManifest` com Pydantic v2 (#M1-I6)
- Interface abstrata `STTBackend` ABC para engines STT plugaveis (#M1-I7)
- Protobuf gRPC `stt_worker.proto` com servico `STTWorker` (#M1-I7)
- Configuracoes de preprocessing e postprocessing com defaults (#M1-I6)
- Fixtures de teste: audio WAV (16kHz, 8kHz, 44.1kHz) e manifestos YAML (#M1-AUX)
- 56 testes unitarios cobrindo tipos, exceptions, manifesto, configs e contrato STTBackend (#M1-AUX)
- Pipeline de CD: workflow de release via tag `v*` com build de wheel e GitHub Release (#CD-01)
- Build validation no CI: verificacao de conteudo do wheel e consistencia de versao (#CD-02)
- Documentacao de CD pipeline com ADR-011 e estrategia de versionamento (#CD-03)
- Script `scripts/generate_proto.sh` para compilacao de protobuf com fix de import paths (#M2-I1)
- Stubs protobuf gerados e commitados: `stt_worker_pb2.py`, `stt_worker_pb2_grpc.py` (#M2-I1)
- Re-exports em `theo.proto.__init__` para imports limpos de mensagens e servicos gRPC (#M2-I1)
- Structured logging via structlog com formatos JSON e console: `configure_logging()`, `get_logger()` (#M2-AUX)
- Conversores puros proto<->Theo em `theo.workers.stt.converters` (#M2-I2)
- gRPC servicer `STTWorkerServicer` com TranscribeFile, Health e stubs UNIMPLEMENTED (#M2-I2)
- Entry point de worker STT como subprocess: `python -m theo.workers.stt` com argparse (#M2-I2)
- `FasterWhisperBackend` implementando `STTBackend` com transcricao batch, hot words via initial_prompt e conversao PCM->numpy (#M2-I3)
- `WorkerManager` para lifecycle de workers como subprocessos: spawn, health probe com backoff, crash detection, auto-restart com rate limiting e shutdown graceful (#M2-I4)
- 71 novos testes unitarios: converters (12), servicer (11), faster-whisper backend (21), worker manager (16), logging (6), integracao (5) (#M2-I5)
- Verificacao de freshness dos stubs proto no CI (#M2-I1)

### Changed
- Campos `compression_ratio` e `probability` adicionados aos tipos `SegmentDetail` e `WordTimestamp` (#M2-I0)
- Proto `Segment` e `Word` estendidos com campos `compression_ratio` e `probability` (#M2-I0)
- Dependencias core: adicionados `structlog>=24.0` e `numpy>=1.26` (#M2-I0)
- Dependencias dev: adicionado `types-grpcio>=1.0` para type stubs gRPC (#M2-I0)
- CI instala extras `dev,grpc` para suportar compilacao e testes de proto (#M2-I1)
- Ruff config: excluidos arquivos gerados `stt_worker_pb2*.py` do linting (#M2-I0)
- Mypy config: override para `faster_whisper.*` (import opcional) (#M2-I0)
- README: tabela de compatibilidade com endpoints OpenAI API (#README-01)
- README: secao Quick Start com exemplo minimo de uso (#README-02)
- README: diagrama de arquitetura convertido para Mermaid (#README-03)
- README: licenca atualizada de "A definir" para MIT (consistente com pyproject.toml) (#README-04)

### Fixed
- `STTWorkerServicer.TranscribeFile` agora tem return explicito apos `context.abort` evitando `UnboundLocalError` se abort nao levantar excecao (#M2-CR1)
- Logica duplicada de construcao de comando CLI no `WorkerManager` extraida para `_build_worker_cmd` e `_spawn_worker_process` (#M2-CR2)
- `_check_worker_health` moveu import de `STTWorkerStub` para fora do bloco try, evitando potencial leak de channel gRPC (#M2-CR3)
- Signal handler no worker STT agora protege contra duplo shutdown com flag `shutting_down` e funcao nomeada em vez de lambda (#M2-CR4)
- Background tasks do `WorkerManager` agora sao awaited apos cancel em `stop_worker` e removidas do dict via `_cancel_background_tasks` (#M2-CR5)
- Subprocess do worker usa `DEVNULL` em vez de `PIPE` para stdout/stderr, eliminando risco de deadlock quando worker gera output (#M2-CR6)
- `_attempt_restart` nao cancela mais as proprias background tasks (self-cancellation bug), substitui diretamente no dict (#M2-CR7)
