# Plano de Implementacao — Theo OpenVoice: Da Arquitetura a Producao

## Contexto

Todas as 9 milestones (M1-M9) estao completas: tipos, interfaces, workers gRPC, API REST, WebSocket, VAD, Session Manager, pipelines, scheduler avancado e full-duplex. 1600 testes passando com mypy strict e ruff clean.

**Problema**: tudo funciona com mocks. O sistema NAO esta pronto para uso real. Faltam:

1. **`theo pull`** — nao existe como baixar modelos
2. **`theo remove`** — nao existe como remover modelos
3. **`theo ps`** — nao existe como ver modelos carregados
4. **`theo transcribe --stream`** — streaming do microfone nao implementado
5. **`--hot-words`** flag no CLI transcribe
6. **Integracao real** — nenhum teste com engines reais (Faster-Whisper, Kokoro)
7. **Docker** — nao existe Dockerfile nem docker-compose
8. **Codigo M9 nao commitado** — tudo no working tree

## Milestones

### M10 — Model Download (`theo pull`)

**Objetivo**: Permitir que usuarios baixem modelos do HuggingFace Hub via CLI.

**Arquivos novos**:
- `src/theo/registry/catalog.py` — `ModelCatalog` class que le o catalogo de modelos disponiveis
- `src/theo/registry/catalog.yaml` — catalogo declarativo com modelos oficiais (nome, repo HF, arquivos, engine, tipo)
- `src/theo/registry/downloader.py` — `ModelDownloader` class que usa `huggingface_hub` para download
- `src/theo/cli/pull.py` — comando `theo pull <model>`
- `tests/unit/test_catalog.py` — testes do catalogo
- `tests/unit/test_downloader.py` — testes do downloader (com mocks de HF Hub)
- `tests/unit/test_pull_cli.py` — testes do comando pull

**Arquivos editados**:
- `src/theo/cli/__init__.py` — registrar comando pull
- `pyproject.toml` — adicionar `huggingface_hub` como dependencia

**Design do catalogo** (`catalog.yaml`):
```yaml
models:
  faster-whisper-large-v3:
    repo: "Systran/faster-whisper-large-v3"
    engine: faster-whisper
    type: stt
    architecture: encoder-decoder
    files:
      - model.bin
      - config.json
      - tokenizer.json
      - vocabulary.json
    manifest_template: manifests/faster-whisper-large-v3.yaml

  faster-whisper-tiny:
    repo: "Systran/faster-whisper-tiny"
    engine: faster-whisper
    type: stt
    architecture: encoder-decoder
    files:
      - model.bin
      - config.json
      - tokenizer.json
      - vocabulary.json
    manifest_template: manifests/faster-whisper-tiny.yaml

  kokoro-v1:
    repo: "hexgrad/Kokoro-82M"
    engine: kokoro
    type: tts
    files:
      - kokoro-v0_19.pth
      - config.json
    manifest_template: manifests/kokoro-v1.yaml
```

**Fluxo do `theo pull <model>`**:
1. Ler catalogo → validar que modelo existe
2. Criar diretorio `~/.theo/models/<model>/`
3. Baixar arquivos do HuggingFace Hub com progress bar (via `huggingface_hub.hf_hub_download`)
4. Copiar manifesto `theo.yaml` para o diretorio do modelo
5. Validar manifesto apos copia

**Dependencia**: `huggingface_hub>=0.20,<1.0` (leve, sem PyTorch)

**Criterio de sucesso**:
```bash
theo pull faster-whisper-tiny
# -> Progress bar, download completo
theo list
# -> faster-whisper-tiny aparece na lista
```

---

### M11 — CLI Completo (`remove`, `ps`, `--stream`, `--hot-words`)

**Objetivo**: Implementar todos os comandos CLI que faltam do PRD.

**Arquivos novos**:
- `src/theo/cli/remove.py` — comando `theo remove <model>`
- `src/theo/cli/ps.py` — comando `theo ps`
- `tests/unit/test_remove_cli.py`
- `tests/unit/test_ps_cli.py`
- `tests/unit/test_transcribe_stream.py`

**Arquivos editados**:
- `src/theo/cli/__init__.py` — registrar remove e ps
- `src/theo/cli/transcribe.py` — adicionar `--stream` e `--hot-words`
- `src/theo/server/routes/health.py` — adicionar `GET /v1/models` (para `theo ps`)

**`theo remove <model>`**:
1. Verificar que modelo existe em `~/.theo/models/<model>/`
2. Confirmar com usuario (click.confirm)
3. Remover diretorio completo (`shutil.rmtree`)
4. Exibir confirmacao

**`theo ps`**:
1. Fazer GET request para `/v1/models` (novo endpoint)
2. Endpoint retorna lista de modelos carregados com status (loaded/idle)
3. Formatar como tabela no terminal (click.echo)

**`theo transcribe --stream`**:
1. Conectar via WebSocket a `ws://host:port/v1/realtime?model=<model>`
2. Capturar audio do microfone via `sounddevice` (dependencia opcional)
3. Enviar frames PCM como mensagens binarias
4. Receber e exibir eventos `transcript.partial` e `transcript.final` em tempo real
5. Ctrl+C para encerrar (envia `session.close`)

**`--hot-words`** flag:
- Adicionar ao `theo transcribe` e `theo translate`
- Para batch: enviar como campo no form-data
- Para stream: enviar via `session.configure` apos conectar

**Dependencia opcional**: `sounddevice` (para `--stream`)

**Criterio de sucesso**:
```bash
theo remove faster-whisper-tiny    # Remove modelo
theo ps                             # Lista modelos carregados
theo transcribe audio.wav --hot-words "PIX,TED"  # Hot words
theo transcribe --stream --model faster-whisper-tiny  # Microfone
```

---

### M12 — Docker & Deployment

**Objetivo**: Criar infraestrutura de containerizacao para deploy do Theo.

**Arquivos novos**:
- `Dockerfile` — multi-stage, CPU-only
- `Dockerfile.gpu` — com CUDA runtime
- `docker-compose.yml` — orquestracao local (server + volume para modelos)
- `.dockerignore` — excluir .venv, .git, tests, docs, __pycache__

**`Dockerfile`** (multi-stage):
```dockerfile
# Stage 1: build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && uv pip install --system ".[server,grpc]"
COPY src/ src/

# Stage 2: runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
EXPOSE 8000
ENTRYPOINT ["theo", "serve", "--host", "0.0.0.0"]
```

**`docker-compose.yml`**:
```yaml
services:
  theo:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - theo-models:/root/.theo/models
    environment:
      - THEO_LOG_FORMAT=json
volumes:
  theo-models:
```

**Criterio de sucesso**:
```bash
docker build -t theo .
docker run -p 8000:8000 -v theo-models:/root/.theo/models theo
# -> Server inicia, responde em /health
```

---

### M13 — Integracao Real & Commit

**Objetivo**: Validar o sistema end-to-end com engines reais e commitar todo o codigo.

**Parte 1 — Testes de integracao com engines reais**:

**Arquivos novos**:
- `tests/integration/test_faster_whisper_e2e.py` — teste com modelo real faster-whisper-tiny
- `tests/integration/test_tts_kokoro_e2e.py` — teste com KokoroBackend real (se instalado)
- `tests/integration/test_pull_e2e.py` — teste de download real de modelo
- `tests/fixtures/audio/` — audio de teste real (WAV curto com fala clara)

**Testes de integracao** (marcados `@pytest.mark.integration`):
1. `theo pull faster-whisper-tiny` → modelo baixado com sucesso
2. `theo serve` → server inicia com modelo real carregado
3. `POST /v1/audio/transcriptions` com audio real → texto correto
4. `WS /v1/realtime` com audio real → partial + final events

**Parte 2 — Commit de todo o codigo M9+**:
- Criar branch `feat/production-ready`
- Commit M9 (full-duplex) que esta no working tree
- Commits incrementais por milestone (M10, M11, M12, M13)
- PR para main

**Criterio de sucesso**:
```bash
# Fluxo completo end-to-end:
theo pull faster-whisper-tiny
theo serve &
curl -F file=@audio.wav -F model=faster-whisper-tiny \
  http://localhost:8000/v1/audio/transcriptions
# -> {"text": "transcricao correta do audio"}
```

---

## Ordem de Execucao

```
M10 (theo pull)  →  M11 (CLI completo)  →  M12 (Docker)  →  M13 (Integracao + Commit)
```

**Dependencias**:
- M11 depende de M10 (remove precisa de modelos instalados para testar)
- M12 pode ser feito em paralelo com M11
- M13 depende de M10+M11 (integracao precisa de pull + serve funcionais)

## Estimativa

| Milestone | Esforco | Arquivos Novos | Arquivos Editados |
|-----------|---------|----------------|-------------------|
| M10 | P (1-2 dias) | 6 | 2 |
| M11 | M (2-3 dias) | 5 | 3 |
| M12 | P (1 dia) | 4 | 0 |
| M13 | M (2-3 dias) | 4 | 0 |

**Total**: ~7-9 dias de trabalho focado.

## O Que NAO Esta no Escopo

- Treinamento de modelos
- UI grafica
- SIP/RTP signaling
- Speaker diarization
- Barge-in (requer AEC externo)
- Auto-scaling / Kubernetes manifests
- CI/CD pipeline (GitHub Actions ja existe basico)
