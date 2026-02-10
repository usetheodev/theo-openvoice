# Plano: Preparacao para Open Source Release

## Contexto

Theo OpenVoice esta com todos os 9 milestones completos (M1-M9), 1686 testes passando, mypy strict e ruff limpos. O projeto precisa ser preparado para lancamento open-source com documentacao profissional, arquivos de comunidade e metadata correta. O usuario escolheu **licenca Apache 2.0** e **Docusaurus** para documentacao (estilo Claude/LangChain).

### Lacunas identificadas

| Lacuna | Impacto |
|--------|---------|
| Sem arquivo LICENSE na raiz | Projeto sem licenca valida |
| pyproject.toml diz MIT, nao Apache 2.0 | Inconsistencia legal |
| pyproject.toml sem URLs, keywords, classifiers completos | PyPI metadata incompleta |
| README.md em portugues, desatualizado (diz M8 "Proximo") | Primeira impressao ruim |
| Sem CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md | Falta de governanca |
| Sem GitHub issue/PR templates | Contribuicoes desorganizadas |
| CHANGELOG.md so tem [Unreleased] | Sem versao oficial |
| Sem site de documentacao | Docs so em markdown no repo |

---

## Objetivo

Entregar um pacote de release open-source pronto para uso e contribuicao, com:
- Licenca e metadata consistentes (Apache 2.0)
- README profissional em ingles
- Arquivos de comunidade (contribuicao, conduta, seguranca)
- Templates de GitHub (issues e PR)
- Changelog com versao inicial
- Site Docusaurus navegavel e pronto para deploy

## Definicao de pronto (DoD)

- `LICENSE` Apache 2.0 na raiz
- `pyproject.toml` com license Apache-2.0, urls, keywords, classifiers completos
- `README.md` em ingles, com quick start funcional e links corretos
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` presentes e validos
- Templates GitHub em `.github/`
- `CHANGELOG.md` com `[Unreleased]` e `[0.1.0] - 2026-02-10`
- Site Docusaurus compila (`npm run build`)
- `make test-unit` e `make check` passam

## Passo 1: Criar LICENSE (Apache 2.0)

**Arquivo novo**: `LICENSE`

Texto padrao da Apache License 2.0 com `Copyright 2026 Theo Team`.

### Tarefas

- Baixar o texto oficial da Apache 2.0
- Substituir o copyright
- Validar que o arquivo esta na raiz

---

## Passo 2: Atualizar pyproject.toml

**Arquivo**: `pyproject.toml`

Mudancas:
- `license = "MIT"` -> `license = "Apache-2.0"`
- `description` em ingles: `"Unified voice runtime (STT + TTS) with OpenAI-compatible API"`
- Adicionar `keywords = ["stt", "tts", "speech-to-text", "text-to-speech", "whisper", "voice", "openai", "realtime", "streaming", "vad"]`
- Adicionar secao `[project.urls]`:
  ```
  Homepage = "https://github.com/usetheo/theo-openvoice"
  Documentation = "https://usetheo.github.io/theo-openvoice"
  Repository = "https://github.com/usetheo/theo-openvoice"
  "Bug Tracker" = "https://github.com/usetheo/theo-openvoice/issues"
  Changelog = "https://github.com/usetheo/theo-openvoice/blob/main/CHANGELOG.md"
  ```
- Expandir classifiers: License (Apache), Intended Audience (Developers), Framework (FastAPI), Topic (Speech)

### Tarefas

- Atualizar campos do `[project]`
- Conferir se nao ha duplicidade de metadata
- Validar formatacao do arquivo

---

## Passo 3: Reescrever README.md

**Arquivo**: `README.md`

Reescrever completamente em ingles, profissional, com:

1. **Header**: Logo + titulo + tagline + badges (license, Python, tests, PyPI)
2. **What is Theo**: 3-4 frases explicando o projeto
3. **Features**: lista com checkmarks (STT streaming, TTS, full-duplex, VAD, multi-engine, session manager, etc.)
4. **Quick Start**: instalacao + primeiro uso em <30 segundos
5. **Architecture Overview**: diagrama ASCII simplificado
6. **Supported Models**: tabela (Faster-Whisper, WeNet, Kokoro)
7. **API Compatibility**: OpenAI Audio API
8. **WebSocket Protocol**: exemplo basico de streaming
9. **CLI**: exemplos dos comandos principais
10. **Documentation**: link para Docusaurus site
11. **Contributing**: link para CONTRIBUTING.md
12. **License**: Apache 2.0

### Tarefas

- Reescrever em ingles com tom profissional
- Incluir exemplos reais com comandos de uso
- Garantir que todos os links apontam para arquivos corretos

---

## Passo 4: Criar arquivos de comunidade

### 4.1 CONTRIBUTING.md (novo)

Guia de contribuicao em ingles:
- Prerequisites (Python 3.11+, uv, make)
- Development setup (`uv venv`, `pip install -e ".[dev]"`, `make check`)
- Code style (ruff, mypy strict, `from __future__ import annotations`)
- Testing guidelines (pytest, `make test-unit`)
- PR process (branch naming, commit format, CI checks)
- Adding new engines (link para docs/ADDING_ENGINE.md)

### 4.2 CODE_OF_CONDUCT.md (novo)

Contributor Covenant v2.1 (padrao da industria).

### 4.3 SECURITY.md (novo)

Security policy:
- Supported versions
- How to report vulnerabilities (email, not public issue)
- Expected response time

### Tarefas

- Criar os tres arquivos com texto padrao revisado
- Garantir links e emails corretos
- Revisar consistencia com README

---

## Passo 5: GitHub templates

### 5.1 `.github/ISSUE_TEMPLATE/bug_report.yml`

Formulario YAML para bug reports com campos:
- Description, steps to reproduce, expected vs actual behavior
- Environment (OS, Python version, model)

### 5.2 `.github/ISSUE_TEMPLATE/feature_request.yml`

Formulario YAML para feature requests com campos:
- Problem description, proposed solution, alternatives considered

### 5.3 `.github/ISSUE_TEMPLATE/config.yml`

Desabilitar issues em branco, link para discussions.

### 5.4 `.github/PULL_REQUEST_TEMPLATE.md`

Template de PR com checklist:
- Description of changes
- Type of change (bug fix, feature, docs)
- Checklist: tests added, `make check` passes, CHANGELOG updated

### Tarefas

- Criar estrutura `.github/ISSUE_TEMPLATE/`
- Validar YAML com schema simples
- Garantir que templates referenciam a documentacao

---

## Passo 6: Atualizar CHANGELOG.md

**Arquivo**: `CHANGELOG.md`

- Mover conteudo de `[Unreleased]` para `[0.1.0] - 2026-02-10`
- Manter secao `[Unreleased]` vazia no topo
- Traduzir header para ingles

### Tarefas

- Ajustar cabecalho para `Changelog`
- Criar entry `[0.1.0] - 2026-02-10` e mover itens
- Manter `[Unreleased]` vazio

---

## Passo 7: Criar site Docusaurus

**Diretorio novo**: `website/`

### 7.1 Scaffold Docusaurus

Inicializar projeto Docusaurus em `website/` com:
- `package.json` com deps (docusaurus 3.x)
- `docusaurus.config.ts` com tema, navbar, footer, search
- `sidebars.ts` com estrutura de navegacao

### 7.2 Estrutura de docs

```
website/docs/
  intro.md                    # Welcome + What is Theo
  getting-started/
    installation.md           # pip install, uv, Docker
    quickstart.md             # First transcription in 60s
    configuration.md          # theo.yaml, preprocessing, etc.
  guides/
    batch-transcription.md    # REST API usage
    streaming-stt.md          # WebSocket streaming guide
    full-duplex.md            # STT + TTS guide (from docs/FULL_DUPLEX.md)
    adding-engine.md          # From docs/ADDING_ENGINE.md
    cli.md                    # CLI reference
  api-reference/
    rest-api.md               # POST /v1/audio/transcriptions, /speech, etc.
    websocket-protocol.md     # Full event reference
    grpc-internal.md          # Internal proto docs
  architecture/
    overview.md               # From docs/ARCHITECTURE.md (simplified)
    session-manager.md        # Deep dive on Session Manager
    vad-pipeline.md           # VAD + preprocessing
    scheduling.md             # Scheduler architecture
  community/
    contributing.md           # Link to CONTRIBUTING.md
    changelog.md              # Link to CHANGELOG.md
    roadmap.md                # From docs/ROADMAP.md (simplified)
```

### 7.3 Homepage (`website/src/pages/index.tsx`)

Landing page com:
- Hero section (logo, tagline, CTA buttons)
- Feature cards (STT, TTS, Full-Duplex, Multi-Engine, Session Manager)
- Quick start code snippet
- Architecture diagram

### 7.4 Configuracao

- Theme: `@docusaurus/preset-classic`
- Search: algolia ou local search plugin
- GitHub link no navbar
- Deploy via GitHub Pages (`usetheo.github.io/theo-openvoice`)
- Dark mode habilitado

### Tarefas

- Scaffolding do Docusaurus em `website/`
- Copiar/transformar docs existentes para o formato do Docusaurus
- Criar pagina inicial com CTA e snippet de quick start
- Validar navegacao e links internos

---

## Verificacao

1. `LICENSE` existe na raiz com texto Apache 2.0
2. `pyproject.toml` tem license Apache-2.0, URLs, keywords, classifiers
3. `README.md` em ingles com badges, features, quick start
4. `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` existem
5. `.github/ISSUE_TEMPLATE/` tem 3 templates + config.yml
6. `.github/PULL_REQUEST_TEMPLATE.md` existe
7. `CHANGELOG.md` tem versao `[0.1.0]` com data de hoje
8. `website/` inicializado com Docusaurus:
   - `cd website && npm install` sem erros
   - `npm run build` sem erros
   - `npm start` abre site local com docs navegaveis
9. `make test-unit` continua passando (nenhuma mudanca em codigo Python)
10. `make check` continua passando

---

## Sequencia executavel (ordem recomendada)

1. LICENSE e pyproject.toml (base legal e metadata)
2. CHANGELOG.md (versao inicial)
3. README.md (primeira impressao)
4. Arquivos de comunidade
5. Templates GitHub
6. Docusaurus (docs e website)
7. Verificacoes finais

## Riscos e mitigacoes

- Links incorretos no README ou Docusaurus: validar com build local
- Inconsistencia de licenca: garantir Apache-2.0 em `LICENSE` e `pyproject.toml`
- Docs divergentes do runtime real: checar com `docs/ARCHITECTURE.md` e `docs/PRD.md`

## Checklist de execucao

- [ ] LICENSE criado na raiz
- [ ] `pyproject.toml` atualizado (license, urls, keywords, classifiers)
- [ ] README reescrito em ingles
- [ ] CONTRIBUTING.md criado
- [ ] CODE_OF_CONDUCT.md criado
- [ ] SECURITY.md criado
- [ ] Templates GitHub adicionados
- [ ] CHANGELOG com `[0.1.0] - 2026-02-10`
- [ ] Docusaurus scaffold e docs
- [ ] `npm run build` no `website/`
- [ ] `make test-unit`
- [ ] `make check`
