# Theo OpenVoice -- Pipeline de CD (Continuous Delivery)

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, STRATEGIC_M1.md v1.0
**Status**: Aprovado pelo Time de Arquitetura (ARCH)
**Data**: 2026-02-07

**Autores**:
- Sofia Castellani (Principal Solution Architect) -- Estrategia de versionamento, decisoes de escopo
- Viktor Sorokin (Senior Real-Time Engineer) -- Validacao de build, integridade de artefatos
- Andre Oliveira (Senior Platform Engineer) -- Workflows, CI/CD, infraestrutura

---

## 1. Contexto e Motivacao

O projeto esta em **pre-alpha** (v0.1.0). O M1 (Fundacao) esta completo: tipos, interfaces, configs, exceptions, protobuf definido, 56 testes unitarios, CI basico.

**O que existe hoje**:
- CI: lint (ruff), typecheck (mypy), testes (pytest) em push/PR para main
- Build system: hatchling via pyproject.toml
- Zero codigo executavel (sem servidor, sem CLI funcional, sem Docker)

**O que precisamos**:
- Processo de release reproduzivel desde o dia 1
- Garantia de que o wheel Python esta correto
- Registro de cada release com artefatos rastreáveis
- Base extensivel para quando Docker e deploy fizerem sentido (M3+)

---

## 2. Principio: YAGNI Rigoroso

> **Construir apenas o que agrega valor nos proximos 2-3 milestones (M2-M4).**
> O resto e preparacao que pode ser adicionada incrementalmente quando necessario.

### O Que Faz Sentido Agora

| Componente | Valor | Milestone |
|------------|-------|-----------|
| Release workflow (tag -> build -> GH Release) | Reproducibilidade, artefato verificavel | M1+ |
| Build validation no CI | Detectar problemas de empacotamento antes do release | M1+ |
| Changelog enforcement no release | Impedir release sem documentacao | M1+ |

### O Que NAO Faz Sentido Agora

| Componente | Por que NAO | Quando fara sentido |
|------------|-------------|---------------------|
| PyPI publish automatico | Pre-alpha, sem audiencia externa, ninguem instala de PyPI | M5+ (quando houver usuarios externos) |
| Docker build/push | Sem servidor (FastAPI so entra em M3), sem Dockerfile | M3 (quando `theo serve` existir) |
| Deploy automatico (staging/prod) | Sem ambiente de execucao, sem servidor | M3+ |
| Semantic-release automatico | 1 contribuidor, releases infrequentes, overengineering | Quando houver time + cadencia regular |
| Matrix de plataformas (macOS/Windows) | Linux-only e suficiente; engines STT sao Linux-first | M5+ (se houver demanda) |
| Assinatura de artefatos (sigstore) | Zero usuarios externos, pre-alpha | Quando public release |
| Environments separados (staging/prod) | Nada para deployar ainda | M3+ |

---

## 3. Estrategia de Versionamento

### 3.1 SemVer 2.0

O projeto segue [Semantic Versioning](https://semver.org/). Estamos em `0.x.y` -- breaking changes sao esperados e permitidos entre minor versions.

| Versao | Significado | Quando |
|--------|-------------|--------|
| 0.1.0 | M1 -- Fundacao | Atual |
| 0.2.0 | M2 -- Worker gRPC | Proximo release |
| 0.3.0 | M3 -- API Batch + CLI | Primeiro servidor |
| 0.4.0 | M4 -- Pipelines | Fase 1 completa |
| 0.x.y (patch) | Bugfix sem nova funcionalidade | Quando necessario |
| 1.0.0 | Contrato de API estavel, pronto para producao | Apos Fase 2+ |

### 3.2 Fonte Unica de Versao

A versao e definida em **dois lugares** que devem estar sincronizados:

1. `pyproject.toml` -- `project.version` (usado pelo build system)
2. `src/theo/__init__.py` -- `__version__` (acessivel em runtime)

**Por que nao usar dynamic versioning (hatch-vcs, setuptools-scm)?**

Avaliamos e descartamos. Dynamic versioning adiciona complexidade (dependencia de tags git, configuracao de plugins) sem beneficio proporcional. O projeto tem cadencia de release baixa (1 release por milestone, ~mensalmente). Manter manual e mais simples e mais previsivel.

### 3.3 Processo de Release

```
1. Atualizar versao em pyproject.toml e src/theo/__init__.py
2. Mover entradas de [Unreleased] para [X.Y.Z] no CHANGELOG.md
3. Commit: "release: vX.Y.Z"
4. Criar tag: git tag vX.Y.Z
5. Push: git push origin main --tags
6. GitHub Actions: build wheel -> criar GitHub Release -> anexar artefato
```

O passo 6 e automatizado pelo workflow `release.yml`.

---

## 4. Workflows

### 4.1 CI (existente, com adição de build validation)

O CI existente (`ci.yml`) e estendido com um step de build para garantir que o wheel e construido corretamente em cada push. Isso detecta problemas de empacotamento (arquivos faltando, imports quebrados) antes que cheguem a um release.

**Arquivo**: `.github/workflows/ci.yml`

### 4.2 Release (novo)

Triggered por tags `v*`. Executa:

1. Checkout do codigo na tag
2. Validacao completa de qualidade (lint, typecheck, testes) -- sim, redundante com CI, mas garante que a tag especifica esta saudavel
3. Build do wheel
4. Verificacao de conteudo do wheel (arquivos esperados estao presentes)
5. Validacao de que a versao na tag corresponde a versao no pyproject.toml
6. Verificacao de que o CHANGELOG tem entrada para esta versao
7. Criacao de GitHub Release com wheel anexado e release notes extraidas do CHANGELOG

**Arquivo**: `.github/workflows/release.yml`

### 4.3 Diagrama de Fluxo

```
Developer
    |
    | git push (branch/PR)
    v
CI (ci.yml)
    |-- lint (ruff check + format)
    |-- typecheck (mypy)
    |-- testes (pytest)
    |-- build validation (hatch build + verify)
    |
    v
    [verde? merge para main]

Developer
    |
    | git tag v0.2.0 && git push --tags
    v
Release (release.yml)
    |-- checkout tag
    |-- quality gate (lint + type + tests) <-- safety net
    |-- build wheel
    |-- verify wheel contents
    |-- verify version match (tag vs pyproject.toml)
    |-- verify changelog has entry
    |-- create GitHub Release
    |   |-- upload wheel
    |   |-- release notes from CHANGELOG
    v
    [GitHub Release publicado]
```

---

## 5. Artefatos Produzidos

### 5.1 Agora (M1-M4)

| Artefato | Formato | Onde fica | Para que serve |
|----------|---------|-----------|----------------|
| Python wheel | `.whl` | GitHub Release | Instalacao reproduzivel: `pip install theo_openvoice-X.Y.Z-py3-none-any.whl` |
| Source distribution | `.tar.gz` | GitHub Release | Backup, build em ambientes sem wheel |
| Release notes | Markdown | GitHub Release | Comunicacao de mudancas |

### 5.2 Futuro (extensoes planejadas, NAO implementadas agora)

| Artefato | Quando | Trigger |
|----------|--------|---------|
| Docker image | M3 (quando `theo serve` existir) | Tag `v*` no release.yml (step adicional) |
| PyPI package | M5+ (quando houver usuarios externos) | Tag `v*` no release.yml (step adicional) |

Essas extensoes serao adicionadas como steps ao `release.yml` existente. O workflow foi projetado para ser extensivel sem reestruturacao.

---

## 6. Validacoes de Release

O release workflow inclui validacoes que impedem releases quebrados:

### 6.1 Version Match

```
Tag: v0.2.0
pyproject.toml: version = "0.2.0"
__init__.py: __version__ = "0.2.0"

Se qualquer um divergir -> release falha.
```

### 6.2 Changelog Entry

O workflow verifica que `CHANGELOG.md` contem uma secao `## [X.Y.Z]` correspondente a versao sendo released. Release sem changelog = release sem documentacao = proibido.

### 6.3 Quality Gate

Mesmo que o CI tenha passado no commit, o release re-executa lint + typecheck + testes na tag exata. Isso cobre o cenario de: CI passou no commit, alguem criou a tag num commit diferente (errado), e o release pega a tag certa.

### 6.4 Wheel Contents

Apos o build, o workflow lista o conteudo do wheel e verifica que arquivos criticos estao presentes:
- `theo/__init__.py`
- `theo/py.typed`
- `theo/_types.py`
- `theo/proto/stt_worker.proto`

Se algum arquivo critico estiver faltando, o release falha.

---

## 7. ADR-011: Pipeline de CD Minimal para Pre-Alpha

### Status
Aceito

### Contexto
O projeto completou M1 (Fundacao) e precisa de um processo de release. O estado atual e pre-alpha: zero codigo executavel, sem servidor, sem Docker, sem usuarios externos. Os proximos milestones (M2-M4) adicionam worker gRPC, API REST e pipelines, mas o primeiro "deployable" so existe em M3.

### Drivers
- Reproducibilidade de artefatos desde o dia 1
- Disciplina de versionamento por milestone
- YAGNI -- nao construir infraestrutura que so sera util em M5+
- Facilidade de extensao quando Docker/PyPI fizerem sentido

### Alternativas Consideradas

#### Alternativa A: Minimal (tag -> build -> GitHub Release)
- 1 workflow file, ~80 linhas
- Produz wheel + source dist
- Anexa ao GitHub Release com notes do CHANGELOG
- Validacoes de integridade (version match, changelog, wheel contents)

**Pros**: Simples, zero dependencias externas (sem tokens, sem registries), cobre 100% do necessario agora
**Contras**: Nao publica em PyPI nem Docker registry

#### Alternativa B: Full (tag -> build -> PyPI + Docker + multi-platform)
- 2-3 workflow files, ~300 linhas
- Publica em PyPI (requer API token)
- Build Docker multi-stage (requer Dockerfile que nao existe)
- Matrix de plataformas

**Pros**: Tudo pronto quando precisar
**Contras**: Overengineering, requer secrets, Docker sem servidor, PyPI sem audiencia

### Trade-off Matrix

| Criterio | Alt. A (Minimal) | Alt. B (Full) |
|----------|-----------------|--------------|
| Valor imediato | Alto | Baixo |
| Complexidade | Baixa | Alta |
| Manutencao | Minima | Alta |
| Tempo de setup | 30 min | 2-3 horas |
| Extensibilidade | Facil (adicionar steps) | Ja pronto |
| Dependencias externas | Nenhuma | PyPI token, Docker registry |

### Decisao
Alternativa A. O workflow minimal entrega todo o valor necessario ate M4. Docker sera adicionado como step em M3. PyPI sera avaliado em M5+.

### Consequencias

**Positivas**:
- Release process automatizado desde o dia 1
- Zero dependencias externas (sem secrets para gerenciar)
- Extensivel -- adicionar Docker e PyPI e adicionar steps, nao reescrever
- CHANGELOG enforcement previne releases sem documentacao

**Negativas**:
- Nao distribui automaticamente em PyPI (aceitavel: ninguem instala de PyPI em pre-alpha)
- Nao produz Docker image (aceitavel: sem servidor ate M3)

### Riscos Aceitos
- Se alguem precisar instalar de PyPI antes de M5, pode publicar manualmente com `hatch publish`
- Se Docker for necessario antes de M3, o Dockerfile sera criado ad-hoc

---

## 8. Evolucao Planejada por Milestone

| Milestone | Mudanca na Pipeline de CD |
|-----------|--------------------------|
| M2 (Worker gRPC) | Nenhuma. Wheel ja inclui proto. Adicionar verificacao de proto compilado no wheel. |
| M3 (API Batch) | Adicionar Dockerfile. Adicionar step de Docker build ao release.yml. Avaliar deploy staging. |
| M4 (Pipelines) | Nenhuma. Verificar que optional dependencies (nemo) nao quebram o build. |
| M5 (WebSocket) | Avaliar PyPI publish. Avaliar nightly builds para testes de estabilidade. |
| M8+ (Telefonia) | Avaliar multi-arch Docker (amd64 + arm64). |

Cada evolucao e um incremento ao `release.yml` existente, nao uma reescrita.

---

## 9. Checklist de Release (Manual)

Antes de criar uma tag de release, verificar:

```
[ ] Versao atualizada em pyproject.toml
[ ] Versao atualizada em src/theo/__init__.py
[ ] Ambas as versoes sao identicas
[ ] CHANGELOG.md tem secao [X.Y.Z] com data
[ ] Secao [Unreleased] esta vazia (tudo migrou para [X.Y.Z])
[ ] CI esta verde no commit que sera taggeado
[ ] git tag vX.Y.Z no commit correto
[ ] git push origin main --tags
```

O release workflow valida os itens 1-5 automaticamente. Os itens 6-8 sao responsabilidade do desenvolvedor.

---

*Documento gerado pelo Time de Arquitetura (ARCH). Sera atualizado conforme milestones avancem e a pipeline evolua.*
