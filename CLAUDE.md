# Theo OpenVoice

Runtime unificado de voz (STT + TTS) construido do zero em Python. Orquestra engines de inferencia (Faster-Whisper, Silero VAD, Kokoro, Piper) com API OpenAI-compatible.

**Status: Fase 2 Completa do PRD entregue.** M1 (Foundation), M2 (Worker gRPC), M3 (API Batch), M4 (Pipelines), M5 (WebSocket + VAD), M6 (Session Manager), M7 (Segundo Backend - WeNet) completos. 1217 testes passando. Proximo: M8 (Scheduler Avancado).

## Commands

```bash
# Development (always use make targets — they use .venv/bin/ automatically)
make check              # format + lint + typecheck
make test               # all tests
make test-unit          # unit tests only (prefer during development)
make test-integration   # integration tests only
make test-fast          # all tests except @pytest.mark.slow
make ci                 # full pipeline: format + lint + typecheck + test
make proto              # generate protobuf stubs

# Individual test
.venv/bin/python -m pytest tests/unit/test_foo.py::test_bar -q
```

## Architecture

```
src/theo/
├── server/           # FastAPI — endpoints REST + WebSocket
│   └── routes/       # transcriptions, translations, health, realtime
├── scheduler/        # Request routing and prioritization
├── registry/         # Model Registry (theo.yaml, lifecycle)
├── workers/          # Subprocess gRPC management
│   └── stt/          # STTBackend interface + FasterWhisperBackend + WeNetBackend
├── preprocessing/    # Audio pipeline (resample, DC remove, gain normalize)
├── postprocessing/   # Text pipeline (ITN via NeMo, fail-open)
├── vad/              # Voice Activity Detection (energy pre-filter + Silero)
├── session/          # Session Manager (state machine, ring buffer, WAL, recovery, backpressure, metrics)
├── cli/              # CLI commands (click)
└── proto/            # gRPC protobuf definitions
```

Full details: @docs/ARCHITECTURE.md
Roadmap: @docs/ROADMAP.md
PRD: @docs/PRD.md

## Key Design Decisions

- **Um binario, um processo, dois tipos de worker.** STT e TTS compartilham API Server, Registry, Scheduler, CLI. Workers sao subprocessos gRPC isolados.
- **Model-agnostic via campo `architecture`** no manifesto: `encoder-decoder` (Whisper), `ctc` (WeNet), `streaming-native` (Paraformer). O runtime adapta o pipeline automaticamente.
- **VAD no runtime, nao na engine.** Silero VAD como biblioteca, com energy pre-filter proprio. Garante comportamento consistente entre engines.
- **Preprocessing e post-processing sao responsabilidade do runtime**, nao da engine. Engines recebem PCM 16kHz normalizado e retornam texto cru.
- **Workers sao subprocessos gRPC** — crash do worker nao derruba o runtime. Recovery via WAL in-memory.

- **Pipeline adaptativo por `architecture`**: `StreamingSession` adapta comportamento automaticamente — encoder-decoder usa LocalAgreement + cross-segment context; CTC usa partials nativos, sem LocalAgreement, sem cross-segment context.

ADRs completos: @docs/PRD.md (secao "Architecture Decision Records")
Como adicionar nova engine: @docs/ADDING_ENGINE.md

## Code Style

- Python 3.12 (via `uv`), tipagem estrita com mypy
- Async-first: todas as interfaces publicas sao `async`
- Formatacao: ruff (format + lint)
- Imports: absolutos a partir de `theo.` (ex: `from theo.registry import Registry`)
- Nomenclatura: snake_case para funcoes/variaveis, PascalCase para classes
- Docstrings: apenas em interfaces publicas (ABC) e funcoes nao-obvias
- Sem comentarios obvios — o codigo deve ser auto-explicativo
- Erros: exceptions tipadas por dominio, nunca `Exception` generico
- Commits seguem conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Testing

- Framework: pytest + pytest-asyncio com `asyncio_mode = "auto"` (sem `@pytest.mark.asyncio`)
- Async FastAPI tests: `httpx.AsyncClient` com `ASGITransport`
- Para testar error handlers (500): `ASGITransport(raise_app_exceptions=False)`
- Fixtures em `tests/conftest.py` (audio WAV sine tones 440Hz gerados automaticamente)
- Mocks: usar `unittest.mock` para engines de inferencia nos testes unitarios
- Testes de integracao marcados com `@pytest.mark.integration`
- **IMPORTANTE**: Sempre rodar `make test-unit` durante desenvolvimento, nao a suite inteira

## Things That Will Bite You

- **gRPC streams sao o heartbeat.** Nao implemente health check polling separado para detectar crash de worker — o stream break do gRPC bidirecional e a detecao.
- **Ring buffer tem read fence.** Nunca sobrescrever dados apos `last_committed_offset` — sao necessarios para recovery.
- **ITN so em `transcript.final`.** Nunca aplicar ITN em `transcript.partial` — partials sao instaveis e ITN geraria output confuso.
- **Preprocessing vem ANTES do VAD.** O audio deve estar normalizado antes de chegar no Silero VAD, senao os thresholds nao funcionam.
- **`vad_filter: false` no manifesto.** O VAD e do runtime. Nao habilitar o VAD interno da engine (ex: Faster-Whisper) — duplicaria o trabalho.
- **Session Manager e so STT.** TTS e stateless por request. Nao tentar reusar Session Manager para TTS.
- **LocalAgreement e para encoder-decoder apenas.** CTC e streaming-native tem partials nativos — nao aplicar LocalAgreement neles. O `StreamingSession` faz isso automaticamente via `_architecture` field.
- **CTC nao usa cross-segment context.** CTC nao suporta `initial_prompt` — `_build_initial_prompt()` retorna `None` para CTC (exceto hot words sem suporte nativo).
- **Manifest `type` field** e normalizado para `model_type` no `ModelManifest` (evita conflito com built-in Python).
- **Force commit e assincrono.** O RingBuffer callback `on_force_commit` e sincrono (chamado de `write()`), mas seta um flag `_force_commit_pending` que `process_frame()` (async) verifica. Nunca bloquear a escrita de novos frames dentro do callback.
- **SessionStateMachine e frozen.** Transicoes invalidas levantam `InvalidTransitionError`. Estado CLOSED e terminal — nenhuma transicao permitida depois.
- **WAL checkpoint e monotonic.** Usa `time.monotonic()` para timestamps, nunca `time.time()`. Garante consistencia mesmo com ajustes de relogio.
- **Recovery reenvia uncommitted data.** Apos crash do worker, `recover()` reenvia `ring_buffer.get_uncommitted()` ao novo stream. Nao duplica dados ja commitados gracas ao read fence.

## Workflow

- Ler o PRD (@docs/PRD.md) antes de implementar qualquer componente
- Consultar a arquitetura (@docs/ARCHITECTURE.md) para entender onde cada peca se encaixa
- Ao adicionar nova engine STT: seguir guia em @docs/ADDING_ENGINE.md (5 passos: implementar `STTBackend` ABC, criar manifesto `theo.yaml`, registrar na factory, declarar dependencia, escrever testes). Zero mudancas no runtime core.
- Ao adicionar novo stage de preprocessing/postprocessing: seguir o padrao pipeline existente (cada stage e toggleavel via config)

## Environment

- **Python 3.12** via `uv` (sistema tem 3.10, projeto requer >=3.11)
- Venv: `.venv/` criado com `uv venv --python 3.12`
- Todos os comandos devem usar `.venv/bin/` ou `make` targets (que ja usam `.venv/bin/`)
- CUDA opcional (fallback para CPU transparente)
- Dependencias pesadas (Faster-Whisper, WeNet, nemo_text_processing) sao opcionais por engine
- gRPC tools necessarios para gerar protobufs: `grpcio-tools`

## API Contracts

- REST: compativel com OpenAI Audio API (`/v1/audio/transcriptions`, `/v1/audio/translations`)
- WebSocket: protocolo de eventos JSON original (`/v1/realtime`) — inspirado na OpenAI Realtime API mas simplificado para STT-only [M5]
- gRPC: protocolo interno runtime <-> worker (nao exposto a clientes)

Contratos detalhados: @docs/PRD.md (secoes 9-13)