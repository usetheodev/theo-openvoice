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
