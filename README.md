# Theo OpenVoice

Runtime unificado de voz (STT + TTS) com API compativel com OpenAI, construido do zero com bibliotecas maduras de inferencia como componentes substituiveis.

## Status

**Em fase de design.** Nenhum codigo implementado ainda.

O PRD completo (v2.1) esta disponivel em [`docs/PRD.md`](docs/PRD.md).

## Visao

Um unico binario que orquestra engines de inferencia (Faster-Whisper, Silero VAD, Kokoro, Piper) com:

- **Session Manager** com estados explicitos e recovery
- **Model Registry** com manifesto declarativo (inspirado no Ollama)
- **Audio Preprocessing/Post-Processing Pipelines**
- **CLI unificado**: `theo pull`, `theo serve`, `theo transcribe`
- **Streaming real** via WebSocket com partial/final transcripts
- **Ingestao RTP** para telefonia

## Arquitetura Alvo

```
CLI / API Server (FastAPI)
        |
    Scheduler --> Model Registry (theo.yaml)
        |
   +---------+-----------+
   | STT Workers (gRPC)  | TTS Workers (gRPC)
   | Faster-Whisper,WeNet | Kokoro, Piper
   +---------+-----------+
        |
  Preprocessing Pipeline --> VAD (Silero) --> Session Manager
```

## Roadmap

- **Fase 1** — STT Batch + Preprocessing (em planejamento)
- **Fase 2** — Streaming Real + Session Manager
- **Fase 3** — Telefonia + Scheduler Avancado

Detalhes completos no [PRD](docs/PRD.md).

## Inspiracoes

- **Ollama** — UX de CLI e modelo de registry local
- **Speaches** — Validacao de API compativel com OpenAI para STT
- **whisper-streaming** — Conceito de LocalAgreement para partial transcripts

## Licenca

A definir.
