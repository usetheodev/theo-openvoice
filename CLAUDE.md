# Theo OpenVoice

Runtime unificado de voz (STT + TTS) construido do zero em Python. Orquestra engines de inferencia (Faster-Whisper, Silero VAD, Kokoro, Piper) com API OpenAI-compatible.

**Status: Pre-implementacao.** PRD completo, zero codigo Python ainda.

## Commands

```bash
# Runtime (quando implementado)
theo serve                              # Inicia API Server
theo pull faster-whisper-large-v3       # Baixa modelo STT
theo list                               # Lista modelos instalados
theo transcribe <file>                  # Transcreve arquivo
theo transcribe --stream                # Streaming do microfone

# Desenvolvimento
python -m pytest tests/                 # Roda todos os testes
python -m pytest tests/unit/            # Apenas testes unitarios
python -m pytest tests/unit/test_foo.py::test_bar  # Teste individual
python -m mypy src/                # Typecheck
python -m ruff check src/               # Lint
python -m ruff format src/              # Format
```

## Architecture

```
src/
├── server/           # FastAPI — endpoints REST + WebSocket
├── scheduler/        # Roteamento e priorizacao de requests
├── registry/         # Model Registry (theo.yaml, lifecycle)
├── workers/          # Subprocess gRPC management
│   └── stt/          # STTBackend interface + implementations
├── preprocessing/    # Audio pipeline (resample, normalize, denoise)
├── postprocessing/   # Text pipeline (ITN, entity formatting, hot words)
├── session/          # Session Manager (estados, ring buffer, VAD, recovery)
├── cli/              # CLI commands (typer/click)
└── proto/            # gRPC protobuf definitions
```

Detalhes completos: @docs/ARCHITECTURE.md

## Key Design Decisions

- **Um binario, um processo, dois tipos de worker.** STT e TTS compartilham API Server, Registry, Scheduler, CLI. Workers sao subprocessos gRPC isolados.
- **Model-agnostic via campo `architecture`** no manifesto: `encoder-decoder` (Whisper), `ctc` (WeNet), `streaming-native` (Paraformer). O runtime adapta o pipeline automaticamente.
- **VAD no runtime, nao na engine.** Silero VAD como biblioteca, com energy pre-filter proprio. Garante comportamento consistente entre engines.
- **Preprocessing e post-processing sao responsabilidade do runtime**, nao da engine. Engines recebem PCM 16kHz normalizado e retornam texto cru.
- **Workers sao subprocessos gRPC** — crash do worker nao derruba o runtime. Recovery via WAL in-memory.

ADRs completos: @docs/PRD.md (secao "Architecture Decision Records")

## Code Style

- Python 3.11+, tipagem estrita com mypy
- Async-first: todas as interfaces publicas sao `async`
- Formatacao: ruff (format + lint)
- Imports: absolutos a partir de `theo.` (ex: `from theo.registry import Registry`)
- Nomenclatura: snake_case para funcoes/variaveis, PascalCase para classes
- Docstrings: apenas em interfaces publicas (ABC) e funcoes nao-obvias
- Sem comentarios obvios — o codigo deve ser auto-explicativo
- Erros: exceptions tipadas por dominio, nunca `Exception` generico

## Testing

- Framework: pytest + pytest-asyncio
- Fixtures em `tests/fixtures/` (arquivos de audio de exemplo)
- Mocks: usar `unittest.mock` para engines de inferencia nos testes unitarios
- Testes de integracao requerem modelo instalado (`theo pull`)
- **IMPORTANTE**: Sempre rodar teste individual durante desenvolvimento, nao a suite inteira

## Things That Will Bite You

- **gRPC streams sao o heartbeat.** Nao implemente health check polling separado para detectar crash de worker — o stream break do gRPC bidirecional e a detecao.
- **Ring buffer tem read fence.** Nunca sobrescrever dados apos `last_committed_offset` — sao necessarios para recovery.
- **ITN so em `transcript.final`.** Nunca aplicar ITN em `transcript.partial` — partials sao instaveis e ITN geraria output confuso.
- **Preprocessing vem ANTES do VAD.** O audio deve estar normalizado antes de chegar no Silero VAD, senao os thresholds nao funcionam.
- **`vad_filter: false` no manifesto.** O VAD e do runtime. Nao habilitar o VAD interno da engine (ex: Faster-Whisper) — duplicaria o trabalho.
- **Session Manager e so STT.** TTS e stateless por request. Nao tentar reusar Session Manager para TTS.
- **LocalAgreement e para encoder-decoder apenas.** CTC e streaming-native tem partials nativos — nao aplicar LocalAgreement neles.

## Workflow

- Ler o PRD (@docs/PRD.md) antes de implementar qualquer componente
- Consultar a arquitetura (@docs/ARCHITECTURE.md) para entender onde cada peca se encaixa
- Ao adicionar nova engine STT: implementar `STTBackend` ABC, criar manifesto `theo.yaml`, registrar no Registry. Zero mudancas no runtime core.
- Ao adicionar novo stage de preprocessing/postprocessing: seguir o padrao pipeline existente (cada stage e toggleavel via config)
- Commits seguem conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Environment

- Python 3.11+
- CUDA opcional (fallback para CPU transparente)
- Dependencias pesadas (Faster-Whisper, nemo_text_processing) sao opcionais por engine
- gRPC tools necessarios para gerar protobufs: `grpcio-tools`

## API Contracts

- REST: compativel com OpenAI Audio API (`/v1/audio/transcriptions`, `/v1/audio/translations`)
- WebSocket: protocolo de eventos JSON original (`/v1/realtime`) — inspirado na OpenAI Realtime API mas simplificado para STT-only
- gRPC: protocolo interno runtime <-> worker (nao exposto a clientes)

Contratos detalhados: @docs/PRD.md (secoes 9-13)

## Available Skills

- `/architecture-review` — Revisao de arquitetura (layering, interfaces, patterns)
- `/streaming-audit` — Auditoria de latencia, backpressure, cancelamento
- `/latency-budget` — Analise de latencia voice-to-voice por estagio
- `/voice-test-scenarios` — Geracao de cenarios de teste realistas
- `/dx-audit` — Auditoria de developer experience
