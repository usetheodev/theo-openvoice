# Theo OpenVoice -- Roadmap Estrategico do Milestone M1

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: Aprovado pelo Time de Arquitetura (ARCH)
**Data**: 2026-02-07

**Autores**:
- Sofia Castellani (Principal Solution Architect) -- Decomposicao, interfaces, contratos, padroes de codigo
- Viktor Sorokin (Senior Real-Time Engineer) -- Protobufs gRPC, tipos de streaming, interfaces async-first
- Andre Oliveira (Senior Platform Engineer) -- pyproject.toml, CI/CD, tooling, estrutura de testes

---

## 1. Objetivos Estrategicos do M1

O M1 nao e "apenas setup de projeto". E a fundacao tecnica que determina a velocidade e qualidade de TODOS os milestones subsequentes. Cada decisao tomada aqui reverbera por 10+ meses de desenvolvimento.

### 1.1 Por que cada objetivo importa

| Objetivo | O que e | Valor Estrategico |
|----------|---------|-------------------|
| **Estrutura de pacotes** | `src/theo/` com `pyproject.toml` | Define o modelo mental do projeto. Um desenvolvedor novo deve olhar a arvore de diretorios e entender a arquitetura sem ler documentacao. Errar aqui gera refatoracoes dolorosas em M3+. |
| **Tooling** | ruff, mypy, pytest | Elimina debates de estilo para sempre. Tipagem estrita desde o dia 1 previne classes inteiras de bugs em runtime async. Sem isso, M5 (WebSocket) e M6 (Session Manager) serao campos minados. |
| **CI basico** | lint + typecheck + testes em cada push | Feedback loop automatizado. Cada commit e validado. Sem CI, regressoes se acumulam silenciosamente e explodem em milestones complexos. |
| **Tipos e interfaces base** | STTBackend ABC, enums, dataclasses | O contrato entre todos os componentes do sistema. A interface `STTBackend` e usada por M2, M3, M5, M6, M7. Se estiver errada, TUDO precisa mudar. |
| **Configuracao** | Parsing de `theo.yaml` | O manifesto e o DNA de cada modelo. Registry (M3), Worker Manager (M2), e Pipeline Adaptativo (M5-M7) dependem dele. |
| **Exceptions tipadas** | Hierarquia de erros por dominio | Fail-fast com mensagens claras. Sem isso, erros genericos tornam debugging impossivel em sistemas distribuidos (runtime + worker subprocess). |
| **Estrutura de testes** | Diretorios, fixtures, audio samples | A piramide de testes comeca aqui. Audio de exemplo (WAV 16kHz, 8kHz, 44.1kHz) sera usado em TODOS os milestones -- de M2 a M10. |
| **Protobufs gRPC** | `stt_worker.proto` definido | O contrato runtime-worker. Definir agora (mesmo sem compilar) garante que M2 comece implementando, nao discutindo protocolo. |
| **CHANGELOG.md** | Registro de mudancas | Contrato de comunicacao. Sem ele, ninguem sabe o que mudou entre versoes. |

### 1.2 Principio do M1

> **Fazer com que `pip install -e ".[dev]" && ruff check src/ && mypy src/ && pytest tests/` passe desde o primeiro commit de codigo.** Se esse loop de feedback funciona, todo milestone subsequente tem uma base solida para construir.

---

## 2. Decisoes Tecnicas Chave

Decisoes tomadas no M1 que impactam todo o ciclo de vida do projeto. Cada decisao foi avaliada pelas tres perspectivas do time.

### 2.1 Python e Versionamento

| Decisao | Valor | Justificativa |
|---------|-------|---------------|
| **Python minimo** | 3.11 | Requirido por: `TaskGroup` (asyncio), type unions nativas (`X \| Y`), `ExceptionGroup`, `tomllib` builtin. Faster-Whisper suporta 3.8+, entao 3.11 nao e restricao. |
| **Python recomendado** | 3.12 | Performance melhorada, mensagens de erro mais claras. Nao exigir 3.13 porque nem todas as dependencias (grpcio, numpy) tem wheels estáveis. |
| **Versioning** | SemVer 2.0 | Primeiro release sera `0.1.0`. Pre-1.0, breaking changes sao esperados. |

**Sofia**: Python 3.11 como minimo e a decisao correta. `type X | Y` elimina `Optional[X]` e `Union[X, Y]`, tornando type hints mais legveis. `TaskGroup` e essencial para o Session Manager (M6) onde multiplas tasks async precisam de lifecycle gerenciado.

**Viktor**: `asyncio.TaskGroup` (3.11+) e non-negotiable para streaming. Sem ele, gerenciar cancelamento de tasks concorrentes (VAD + inference + WebSocket write) seria manual e propenso a leaks.

**Andre**: Python 3.12 oferece ~5% de performance via specializing adaptive interpreter. Mas 3.11 como minimo garante compatibilidade com mais ambientes de producao.

### 2.2 Dependencias e Versoes

#### Dependencias Core (instalacao base)

| Pacote | Versao | Uso | Nota |
|--------|--------|-----|------|
| `pydantic` | `>=2.6,<3.0` | Validacao de configs, request/response models | v2 obrigatorio (performance, model_validator) |
| `pyyaml` | `>=6.0,<7.0` | Parsing de `theo.yaml` | Estavel, sem breaking changes previstos |

**Por que nao incluir FastAPI, gRPC, etc. no core?** Porque M1 nao precisa de servidor nem de workers. Essas dependencias entram em M2/M3 como extras. Manter o core leve permite que `pip install theo` funcione sem arrastar frameworks.

#### Dependencias de Desenvolvimento

| Pacote | Versao | Uso |
|--------|--------|-----|
| `ruff` | `>=0.9,<1.0` | Lint + format (substitui black, isort, flake8) |
| `mypy` | `>=1.14,<2.0` | Typecheck estrito |
| `pytest` | `>=8.0,<9.0` | Framework de testes |
| `pytest-asyncio` | `>=0.25,<1.0` | Suporte a testes async |
| `grpcio-tools` | `>=1.68,<2.0` | Compilacao de protobufs (dev-only) |

#### Extras Opcionais (definidos no M1, instalados em milestones posteriores)

| Extra | Pacotes | Quando |
|-------|---------|--------|
| `[server]` | `fastapi`, `uvicorn`, `python-multipart` | M3 |
| `[grpc]` | `grpcio`, `protobuf` | M2 |
| `[faster-whisper]` | `faster-whisper` | M2 |
| `[itn]` | `nemo_text_processing` | M4 |
| `[denoise]` | `rnnoise-python` ou binding equivalente | M4 |
| `[dev]` | ruff, mypy, pytest, pytest-asyncio, grpcio-tools | M1 |
| `[all]` | Todos os extras acima | Conveniencia |

**Andre**: A estrategia de extras e critica. `pip install theo` instala apenas o core (tipos, configs, exceptions). `pip install theo[server,faster-whisper]` instala o necessario para servir STT. Isso evita que um usuario de CLI precise instalar FastAPI, e que um usuario de API precise instalar WeNet.

**Sofia**: Os extras devem ser definidos no `pyproject.toml` do M1, mesmo que os pacotes so sejam usados em milestones posteriores. Isso evita reestruturar o pyproject.toml a cada milestone.

### 2.3 Estrutura Definitiva de Pacotes

```
theo-openvoice/
├── src/
│   └── theo/
│       ├── __init__.py                  # Versao do pacote (__version__)
│       ├── py.typed                     # Marker PEP 561 (pacote tipado)
│       │
│       ├── _types.py                    # Enums e dataclasses fundamentais
│       │                                # STTArchitecture, TranscriptSegment,
│       │                                # BatchResult, EngineCapabilities
│       │
│       ├── exceptions.py               # Hierarquia de exceptions tipadas
│       │
│       ├── config/                      # Configuracao e manifestos
│       │   ├── __init__.py
│       │   ├── manifest.py             # Parsing de theo.yaml (ModelManifest)
│       │   ├── preprocessing.py        # PreprocessingConfig
│       │   └── postprocessing.py       # PostProcessingConfig
│       │
│       ├── workers/                     # Worker management e interfaces
│       │   ├── __init__.py
│       │   └── stt/
│       │       ├── __init__.py
│       │       └── interface.py        # STTBackend ABC
│       │
│       ├── server/                      # [vazio em M1] API Server FastAPI
│       │   └── __init__.py
│       │
│       ├── scheduler/                   # [vazio em M1] Request scheduling
│       │   └── __init__.py
│       │
│       ├── registry/                    # [vazio em M1] Model Registry
│       │   └── __init__.py
│       │
│       ├── preprocessing/               # [vazio em M1] Audio pipeline
│       │   └── __init__.py
│       │
│       ├── postprocessing/              # [vazio em M1] Text pipeline
│       │   └── __init__.py
│       │
│       ├── session/                     # [vazio em M1] Session Manager
│       │   └── __init__.py
│       │
│       ├── cli/                         # [vazio em M1] CLI commands
│       │   └── __init__.py
│       │
│       └── proto/                       # gRPC protobuf definitions
│           ├── __init__.py
│           └── stt_worker.proto         # Definicao do contrato gRPC
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Fixtures compartilhadas
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_types.py               # Testes dos tipos base
│   │   ├── test_exceptions.py          # Testes das exceptions
│   │   ├── test_manifest.py            # Testes do parsing de theo.yaml
│   │   └── test_config.py              # Testes de PreprocessingConfig, etc.
│   ├── integration/
│   │   └── __init__.py
│   └── fixtures/
│       ├── audio/
│       │   ├── sample_16khz.wav         # PCM 16-bit, 16kHz, mono, ~3s
│       │   ├── sample_8khz.wav          # PCM 16-bit, 8kHz, mono, ~3s
│       │   └── sample_44khz.wav         # PCM 16-bit, 44.1kHz, mono, ~3s
│       └── manifests/
│           ├── valid_stt.yaml           # Manifesto STT valido completo
│           ├── valid_tts.yaml           # Manifesto TTS valido (para testar campo type)
│           ├── minimal.yaml             # Manifesto com campos minimos
│           └── invalid_missing.yaml     # Manifesto com campos obrigatorios faltando
│
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   └── STRATEGIC_M1.md                 # Este documento
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions: lint, typecheck, testes
│
├── pyproject.toml                       # Configuracao unica do projeto
├── CHANGELOG.md                         # Registro de mudancas
├── README.md
├── CLAUDE.md
└── .gitignore
```

**Decisoes de estrutura justificadas:**

| Decisao | Alternativa Descartada | Justificativa |
|---------|----------------------|---------------|
| `src/theo/` (src layout) | Flat layout (`theo/`) | src layout previne import acidental do pacote local vs instalado. Padrao recomendado pelo PyPA. |
| `_types.py` na raiz do pacote | `types/` como diretorio | Arquivo unico porque sao poucos tipos (~5 dataclasses + 1 enum). Diretorio seria over-engineering. Underscore prefix porque `types` e nome de modulo builtin. |
| `config/` como sub-pacote | Tudo em `manifest.py` | Separar manifest (theo.yaml) de configs de runtime (preprocessing, postprocessing) porque sao conceitos distintos com lifecycles diferentes. |
| `workers/stt/interface.py` | `interfaces/stt.py` | Manter interface junto ao dominio (workers/stt/) porque em M2 a implementacao (`faster_whisper.py`) fica no mesmo diretorio. Coesao por dominio. |
| `__init__.py` em pacotes vazios | Nao criar diretorios ainda | Criar agora garante que a estrutura esta visivel e que imports como `from theo.server import ...` sao detectaveis pelo mypy mesmo antes de haver codigo. |
| `py.typed` marker | Omitir | PEP 561 exige este arquivo para que mypy reconheca o pacote como tipado quando instalado. Sem ele, consumidores do pacote nao teriam type checking. |

**Viktor**: O `proto/stt_worker.proto` fica na arvore de fontes (`src/theo/proto/`) e nao em um diretorio separado `proto/` na raiz. Isso garante que o proto e parte do pacote distribuido e que as ferramentas de build encontram facilmente.

**Andre**: Os `__init__.py` nos pacotes vazios (`server/`, `scheduler/`, etc.) podem ser apenas arquivos vazios. O objetivo e reservar o namespace e permitir que o mypy valide imports futuros sem erros de "module not found".

### 2.4 Padroes de Codigo (Definitivos)

#### Imports

```python
# Absolutos a partir de theo.
from theo._types import STTArchitecture, TranscriptSegment
from theo.config.manifest import ModelManifest
from theo.exceptions import ModelNotFoundError

# NUNCA imports relativos
# from ._types import STTArchitecture  # PROIBIDO
# from .manifest import ModelManifest  # PROIBIDO
```

**Justificativa**: Imports absolutos sao mais claros, refactoring-safe, e funcionam identicamente em testes e codigo de producao.

#### Type Hints

```python
# Usar syntax nativa do Python 3.11+
def process(audio: bytes, language: str | None = None) -> TranscriptSegment:
    ...

# NUNCA usar Optional ou Union
# from typing import Optional, Union  # DESNECESSARIO em 3.11+

# Para collections, usar tipos builtin
def get_segments() -> list[TranscriptSegment]:
    ...

# Para dicionarios tipados, usar TypedDict quando a estrutura e conhecida
class WordTimestamp(TypedDict):
    word: str
    start: float
    end: float
```

#### Async

```python
# Todas as interfaces publicas sao async
class STTBackend(ABC):
    @abstractmethod
    async def transcribe_file(self, ...) -> BatchResult:
        ...

# Implementacoes internas que nao fazem I/O podem ser sync
def _calculate_rms(frame: bytes) -> float:
    ...
```

#### Exceptions

```python
# Hierarquia com exception base do dominio
class TheoError(Exception):
    """Base para todas as exceptions do Theo."""

class ModelNotFoundError(TheoError):
    """Modelo nao encontrado no registry."""
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Modelo '{model_name}' nao encontrado no registry")

# NUNCA raise Exception("mensagem generica")
# NUNCA except Exception: pass
```

#### Dataclasses e Enums

```python
from dataclasses import dataclass
from enum import Enum

# Enums para valores finitos e conhecidos
class STTArchitecture(Enum):
    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"

# Dataclasses para dados estruturados
@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    is_final: bool
    segment_id: int
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    confidence: float | None = None
    words: list[dict[str, float | str]] | None = None
```

**Sofia**: `frozen=True` e `slots=True` em dataclasses de dados imutaveis (como `TranscriptSegment`). `frozen` garante imutabilidade (thread-safe), `slots` reduz uso de memoria e melhora performance de acesso a atributos. Isso e relevante para M5+ onde milhares de segments trafegam por segundo.

### 2.5 Protobuf gRPC

O contrato gRPC e definido no M1 para garantir estabilidade antes da implementacao em M2.

```protobuf
// stt_worker.proto
syntax = "proto3";

package theo.stt;

option python_package = "theo.proto";

service STTWorker {
  // Transcricao batch (arquivo completo)
  rpc TranscribeFile (TranscribeFileRequest) returns (TranscribeFileResponse);

  // Transcricao streaming (bidirecional)
  rpc TranscribeStream (stream AudioFrame) returns (stream TranscriptEvent);

  // Cancelamento de request/sessao
  rpc Cancel (CancelRequest) returns (CancelResponse);

  // Health check (unario)
  rpc Health (HealthRequest) returns (HealthResponse);
}

// --- Mensagens de TranscribeFile ---

message TranscribeFileRequest {
  string request_id = 1;
  bytes audio_data = 2;
  string language = 3;           // ISO 639-1 ou "auto" ou "mixed"
  string response_format = 4;    // json, verbose_json, text, srt, vtt
  float temperature = 5;
  repeated string timestamp_granularities = 6;  // "segment", "word"
  string initial_prompt = 7;
  repeated string hot_words = 8;
}

message TranscribeFileResponse {
  string text = 1;
  string language = 2;
  float duration = 3;
  repeated Segment segments = 4;
  repeated Word words = 5;
}

// --- Mensagens de TranscribeStream ---

message AudioFrame {
  string session_id = 1;
  bytes data = 2;               // PCM 16-bit 16kHz mono (ja preprocessado)
  bool is_last = 3;             // Sinaliza fim do stream
  string initial_prompt = 4;    // Contexto do segmento anterior
  repeated string hot_words = 5;
}

message TranscriptEvent {
  string session_id = 1;
  string event_type = 2;        // "partial" ou "final"
  string text = 3;
  int32 segment_id = 4;
  int64 start_ms = 5;
  int64 end_ms = 6;
  string language = 7;
  float confidence = 8;
  repeated Word words = 9;
}

// --- Mensagens compartilhadas ---

message Segment {
  int32 id = 1;
  float start = 2;
  float end = 3;
  string text = 4;
  float avg_logprob = 5;
  float no_speech_prob = 6;
}

message Word {
  string word = 1;
  float start = 2;
  float end = 3;
}

// --- Cancelamento ---

message CancelRequest {
  string request_id = 1;
  string session_id = 2;        // Para streaming
}

message CancelResponse {
  bool acknowledged = 1;
}

// --- Health ---

message HealthRequest {}

message HealthResponse {
  string status = 1;            // "ok", "loading", "error"
  string model_name = 2;
  string engine = 3;
  map<string, string> metadata = 4;
}
```

**Viktor**: O campo `event_type` em `TranscriptEvent` usa string em vez de enum protobuf para facilitar extensibilidade futura (adicionar novos tipos sem recompilar). O trade-off e perda de type safety no proto, compensada por validacao no codigo Python.

**Sofia**: O `request_id` em `TranscribeFileRequest` e o `session_id` em `AudioFrame` sao strings (UUIDs gerados pelo runtime). Isso permite rastreabilidade end-to-end de uma request desde o API Server ate o worker.

### 2.6 Configuracao de Tooling

#### ruff

```toml
# Em pyproject.toml
[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "RUF",    # ruff-specific rules
]
ignore = [
    "E501",   # Line too long (gerenciado por formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["theo"]
```

**Andre**: `line-length = 99` e um compromisso pragmatico. 79 e muito restritivo para type hints complexos. 120 dificulta side-by-side. 99 funciona bem em monitores modernos e com type hints verbosos.

#### mypy

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true

[[tool.mypy.overrides]]
module = "theo.proto.*"
ignore_errors = true       # Codigo gerado por grpcio-tools nao e tipado
```

**Sofia**: `strict = true` desde o dia 1. Adicionar mypy strict depois (com codigo existente) e doloroso -- centenas de erros para corrigir retroativamente. Fazer desde o inicio custa zero.

#### pytest

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "error",
    "ignore::DeprecationWarning:google.protobuf.*",
]
markers = [
    "integration: testes que requerem modelo instalado",
    "slow: testes que demoram mais de 10s",
]
```

**Andre**: `asyncio_mode = "auto"` permite que funcoes `async def test_...` sejam detectadas automaticamente como testes async, sem precisar do decorator `@pytest.mark.asyncio` em cada teste. Reduz boilerplate significativamente.

---

## 3. Iniciativas

O M1 e decomposto em 7 iniciativas ordenadas. Cada uma e independente o suficiente para ser implementada como um commit (ou conjunto de commits) atomico.

### 3.1 Grafo de Dependencias entre Iniciativas

```
I1 (pyproject.toml + estrutura)
├──> I2 (Tooling: ruff, mypy, pytest)
│    └──> I3 (CI: GitHub Actions)
├──> I4 (Tipos e Enums)
│    ├──> I5 (Exceptions tipadas)
│    ├──> I6 (Configuracao e Manifesto)
│    └──> I7 (Interface STTBackend + Protobufs)
```

**Legenda**:
- I1 e pre-requisito de TUDO (sem pyproject.toml, nada instala)
- I2 depende de I1 (tooling precisa de pyproject.toml para configurar)
- I3 depende de I2 (CI executa as ferramentas configuradas em I2)
- I4 depende de I1 (precisa do pacote theo/ existindo)
- I5, I6, I7 dependem de I4 (usam os tipos definidos)
- I3 e independente de I4-I7 (CI valida qualquer codigo que exista)

**Paralelismo possivel**:
- Apos I1: I2 e I4 podem ser feitas em paralelo
- Apos I4: I5, I6 e I7 podem ser feitas em paralelo
- I3 pode ser feita a qualquer momento apos I2

---

### Iniciativa I1 -- Estrutura do Projeto e pyproject.toml

**Objetivo**: Criar a estrutura de diretorios e o `pyproject.toml` que permite `pip install -e ".[dev]"` funcionar.

**Responsavel**: Andre Oliveira

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `pyproject.toml` | Build system, metadata, dependencias, extras, config de tools |
| `src/theo/__init__.py` | `__version__ = "0.1.0"` |
| `src/theo/py.typed` | Marker PEP 561 (arquivo vazio) |
| `src/theo/{server,scheduler,registry,workers,workers/stt,preprocessing,postprocessing,session,cli,config,proto}/__init__.py` | Pacotes vazios (reserva de namespace) |
| `tests/__init__.py` | Pacote de testes |
| `tests/conftest.py` | Configuracao compartilhada do pytest (vazio inicialmente) |
| `tests/unit/__init__.py` | Pacote de testes unitarios |
| `tests/integration/__init__.py` | Pacote de testes de integracao |
| `tests/fixtures/audio/` | Diretorio para audio samples |
| `tests/fixtures/manifests/` | Diretorio para manifestos de teste |
| `CHANGELOG.md` | Inicializado com secao `[Unreleased]` |

**pyproject.toml definitivo**:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "theo-openvoice"
version = "0.1.0"
description = "Runtime unificado de voz (STT + TTS) com API OpenAI-compatible"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Theo Team" },
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Typing :: Typed",
]

dependencies = [
    "pydantic>=2.6,<3.0",
    "pyyaml>=6.0,<7.0",
]

[project.optional-dependencies]
server = [
    "fastapi>=0.115,<1.0",
    "uvicorn[standard]>=0.34,<1.0",
    "python-multipart>=0.0.18",
]
grpc = [
    "grpcio>=1.68,<2.0",
    "protobuf>=5.29,<6.0",
]
faster-whisper = [
    "faster-whisper>=1.1,<2.0",
]
itn = [
    "nemo_text_processing>=1.1,<2.0",
]
dev = [
    "ruff>=0.9,<1.0",
    "mypy>=1.14,<2.0",
    "pytest>=8.0,<9.0",
    "pytest-asyncio>=0.25,<1.0",
    "grpcio-tools>=1.68,<2.0",
    "types-PyYAML>=6.0",
    "types-protobuf>=5.29",
]
all = [
    "theo-openvoice[server,grpc,faster-whisper,itn,dev]",
]

[tool.hatch.build.targets.wheel]
packages = ["src/theo"]

[tool.ruff]
target-version = "py311"
line-length = 99
src = ["src"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "RUF",    # ruff-specific rules
]
ignore = [
    "E501",   # Gerenciado pelo formatter
]

[tool.ruff.lint.isort]
known-first-party = ["theo"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true
mypy_path = "src"

[[tool.mypy.overrides]]
module = "theo.proto.*"
ignore_errors = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "error",
    "ignore::DeprecationWarning:google.protobuf.*",
]
markers = [
    "integration: testes que requerem modelo instalado",
    "slow: testes que demoram mais de 10s",
]
```

**CHANGELOG.md**:

```markdown
# Changelog

Todas as mudancas relevantes deste projeto serao documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/),
e este projeto adere ao [Versionamento Semantico](https://semver.org/lang/pt-BR/).

## [Unreleased]
```

**Criterio de done**:
```bash
cd /home/paulo/Projetos/usetheo/voz/theo-openvoice
pip install -e ".[dev]"
python -c "import theo; print(theo.__version__)"
# -> 0.1.0
```

**Decisoes registradas**:

| Decisao | Alternativa | Por que esta |
|---------|-------------|--------------|
| `hatchling` como build backend | `setuptools`, `flit`, `pdm-backend` | Hatchling e rapido, moderno, e suporta src layout nativamente sem config extra. setuptools requer `setup.cfg` adicional. flit nao suporta extras complexos. |
| Uma unica `pyproject.toml` | `setup.cfg` + `setup.py` | pyproject.toml e o padrao PEP 621. Um arquivo para tudo (build, deps, tools). |
| `src/` layout | Flat layout | Previne import acidental do pacote local. `python -c "import theo"` so funciona apos `pip install`. |

---

### Iniciativa I2 -- Configuracao de Tooling

**Objetivo**: Garantir que `ruff check`, `ruff format`, `mypy` e `pytest` funcionam corretamente no projeto vazio.

**Responsavel**: Andre Oliveira

**Dependencia**: I1

**Entregaveis**:

| Verificacao | Comando | Resultado esperado |
|-------------|---------|-------------------|
| Lint | `python -m ruff check src/` | 0 erros |
| Format | `python -m ruff format --check src/` | 0 arquivos a formatar |
| Typecheck | `python -m mypy src/` | Success: no issues found |
| Testes | `python -m pytest tests/unit/` | 0 testes coletados, 0 erros |

**Arquivo `.github/workflows/ci.yml`** sera criado em I3, mas as ferramentas devem funcionar localmente primeiro.

**Criterio de done**:
```bash
python -m ruff check src/ && \
python -m ruff format --check src/ && \
python -m mypy src/ && \
python -m pytest tests/unit/
# Todos devem passar com 0 erros
```

---

### Iniciativa I3 -- CI Basico (GitHub Actions)

**Objetivo**: Automatizar lint, typecheck e testes em cada push e pull request.

**Responsavel**: Andre Oliveira

**Dependencia**: I2

**Entregavel**:

Arquivo `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint (ruff check)
        run: python -m ruff check src/ tests/

      - name: Format check (ruff format)
        run: python -m ruff format --check src/ tests/

      - name: Type check (mypy)
        run: python -m mypy src/

      - name: Unit tests
        run: python -m pytest tests/unit/ -v
```

**Decisoes**:

| Decisao | Justificativa |
|---------|---------------|
| Matrix com 3.11 e 3.12 | Valida o minimo suportado e a versao recomendada. 3.13 pode ser adicionado quando estabilizar. |
| Nao incluir testes de integracao no CI | Testes de integracao requerem modelo instalado (Faster-Whisper). Serao adicionados como job separado em M2. |
| ubuntu-latest como runner | Suficiente para M1. GPU runners serao necessarios a partir de M2 (integracao). |

**Criterio de done**: CI passa no GitHub com status verde para ambas as versoes de Python.

---

### Iniciativa I4 -- Tipos e Enums Base

**Objetivo**: Definir os tipos fundamentais que todos os componentes do Theo usam. Estes tipos sao o vocabulario compartilhado do sistema.

**Responsavel**: Sofia Castellani

**Dependencia**: I1

**Entregavel**: `src/theo/_types.py`

```python
"""Tipos fundamentais do Theo OpenVoice.

Este modulo define enums, dataclasses e type aliases que sao usados
por todos os componentes do runtime. Alteracoes aqui impactam o sistema inteiro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class STTArchitecture(Enum):
    """Arquitetura de modelo STT.

    Determina como o runtime adapta o pipeline de streaming:
    - ENCODER_DECODER: acumula windows, LocalAgreement para partials (ex: Whisper)
    - CTC: frame-by-frame, partials nativos (ex: WeNet CTC)
    - STREAMING_NATIVE: streaming verdadeiro, engine gerencia estado (ex: Paraformer)
    """

    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"


class ModelType(Enum):
    """Tipo de modelo no registry."""

    STT = "stt"
    TTS = "tts"


class SessionState(Enum):
    """Estado de uma sessao de streaming STT.

    Transicoes validas:
        INIT -> ACTIVE (primeiro audio com fala)
        INIT -> CLOSED (timeout 30s sem audio)
        ACTIVE -> SILENCE (VAD detecta silencio)
        SILENCE -> ACTIVE (VAD detecta fala)
        SILENCE -> HOLD (timeout 30s sem fala)
        HOLD -> ACTIVE (VAD detecta fala)
        HOLD -> CLOSING (timeout 5min)
        CLOSING -> CLOSED (flush completo ou timeout 2s)
        Qualquer -> CLOSED (erro irrecuperavel)
    """

    INIT = "init"
    ACTIVE = "active"
    SILENCE = "silence"
    HOLD = "hold"
    CLOSING = "closing"
    CLOSED = "closed"


class VADSensitivity(Enum):
    """Nivel de sensibilidade do VAD.

    Ajusta threshold do Silero VAD e energy pre-filter conjuntamente.
    """

    HIGH = "high"       # threshold=0.3, energy=-50dBFS (sussurro, banking)
    NORMAL = "normal"   # threshold=0.5, energy=-40dBFS (conversacao normal)
    LOW = "low"         # threshold=0.7, energy=-30dBFS (ambiente ruidoso)


class ResponseFormat(Enum):
    """Formato de resposta da API de transcricao."""

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Timestamp de uma palavra individual."""

    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Segmento de transcricao (partial ou final).

    Emitido pelo worker via gRPC e propagado ao cliente via WebSocket.
    """

    text: str
    is_final: bool
    segment_id: int
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    confidence: float | None = None
    words: tuple[WordTimestamp, ...] | None = None


@dataclass(frozen=True, slots=True)
class SegmentDetail:
    """Detalhes de um segmento no formato verbose_json."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Resultado de transcricao batch (arquivo completo)."""

    text: str
    language: str
    duration: float
    segments: tuple[SegmentDetail, ...]
    words: tuple[WordTimestamp, ...] | None = None


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Capabilities reportadas pela engine STT em runtime.

    Pode diferir do manifesto (theo.yaml) se a engine descobrir
    capabilities adicionais apos load.
    """

    supports_hot_words: bool = False
    supports_initial_prompt: bool = False
    supports_batch: bool = False
    supports_word_timestamps: bool = False
    max_concurrent_sessions: int = 1
```

**Decisoes de design**:

| Decisao | Alternativa | Justificativa |
|---------|-------------|---------------|
| `tuple` em vez de `list` para collections em dataclasses frozen | `list[WordTimestamp]` | Tuples sao imutaveis e hashable, consistente com `frozen=True`. Lists dentro de frozen dataclasses sao mutaveis (furo na imutabilidade). |
| `WordTimestamp` como dataclass separada | `dict[str, float \| str]` | Dataclass tipada e mais segura que dict. mypy valida campos. Acesso por atributo e mais claro que acesso por chave. |
| Enums para todos os valores finitos | Strings cruas | Enums previnem typos, sao validaveis em runtime e autocomplete funciona. `SessionState.ACTIVE` e mais seguro que `"active"`. |
| `SessionState` definido em `_types.py` | Em `session/` | E um tipo fundamental usado por multiplos componentes (Session Manager, WebSocket handler, metricas). Pertence ao vocabulario compartilhado. |

**Testes** (`tests/unit/test_types.py`):

```python
"""Testes dos tipos fundamentais do Theo."""

from theo._types import (
    BatchResult,
    EngineCapabilities,
    ModelType,
    ResponseFormat,
    SegmentDetail,
    SessionState,
    STTArchitecture,
    TranscriptSegment,
    VADSensitivity,
    WordTimestamp,
)


class TestSTTArchitecture:
    def test_values(self) -> None:
        assert STTArchitecture.ENCODER_DECODER.value == "encoder-decoder"
        assert STTArchitecture.CTC.value == "ctc"
        assert STTArchitecture.STREAMING_NATIVE.value == "streaming-native"

    def test_from_string(self) -> None:
        assert STTArchitecture("encoder-decoder") == STTArchitecture.ENCODER_DECODER


class TestModelType:
    def test_values(self) -> None:
        assert ModelType.STT.value == "stt"
        assert ModelType.TTS.value == "tts"


class TestSessionState:
    def test_all_states_exist(self) -> None:
        states = {s.value for s in SessionState}
        assert states == {"init", "active", "silence", "hold", "closing", "closed"}


class TestTranscriptSegment:
    def test_minimal(self) -> None:
        seg = TranscriptSegment(text="ola", is_final=True, segment_id=0)
        assert seg.text == "ola"
        assert seg.is_final is True
        assert seg.start_ms is None

    def test_with_words(self) -> None:
        words = (WordTimestamp(word="ola", start=0.0, end=0.5),)
        seg = TranscriptSegment(
            text="ola",
            is_final=True,
            segment_id=0,
            words=words,
        )
        assert seg.words is not None
        assert seg.words[0].word == "ola"

    def test_frozen(self) -> None:
        seg = TranscriptSegment(text="ola", is_final=True, segment_id=0)
        try:
            seg.text = "outro"  # type: ignore[misc]
            assert False, "Deveria ter lancado FrozenInstanceError"
        except AttributeError:
            pass


class TestBatchResult:
    def test_creation(self) -> None:
        result = BatchResult(
            text="ola mundo",
            language="pt",
            duration=1.5,
            segments=(
                SegmentDetail(id=0, start=0.0, end=1.5, text="ola mundo"),
            ),
        )
        assert result.text == "ola mundo"
        assert result.language == "pt"
        assert len(result.segments) == 1


class TestEngineCapabilities:
    def test_defaults(self) -> None:
        caps = EngineCapabilities()
        assert caps.supports_hot_words is False
        assert caps.max_concurrent_sessions == 1
```

**Criterio de done**:
- `python -m mypy src/theo/_types.py` passa sem erros
- `python -m pytest tests/unit/test_types.py` passa (todos os testes)
- `python -m ruff check src/theo/_types.py` sem warnings

---

### Iniciativa I5 -- Exceptions Tipadas por Dominio

**Objetivo**: Definir a hierarquia de exceptions que todo o sistema usa. Erros claros, tipados, com contexto suficiente para diagnostico.

**Responsavel**: Sofia Castellani

**Dependencia**: I4 (usa `STTArchitecture` em algumas exceptions)

**Entregavel**: `src/theo/exceptions.py`

```python
"""Exceptions tipadas do Theo OpenVoice.

Hierarquia:
    TheoError (base)
    ├── ConfigError
    │   ├── ManifestParseError
    │   └── ManifestValidationError
    ├── ModelError
    │   ├── ModelNotFoundError
    │   └── ModelLoadError
    ├── WorkerError
    │   ├── WorkerCrashError
    │   ├── WorkerTimeoutError
    │   └── WorkerUnavailableError
    ├── AudioError
    │   ├── AudioFormatError
    │   └── AudioTooLargeError
    └── SessionError
        ├── SessionNotFoundError
        └── SessionClosedError
"""


class TheoError(Exception):
    """Base para todas as exceptions do Theo OpenVoice."""


# --- Configuracao ---


class ConfigError(TheoError):
    """Erro de configuracao do runtime."""


class ManifestParseError(ConfigError):
    """Falha ao parsear arquivo theo.yaml."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Falha ao parsear manifesto '{path}': {reason}")


class ManifestValidationError(ConfigError):
    """Manifesto theo.yaml invalido (campos obrigatorios faltando, tipos errados)."""

    def __init__(self, path: str, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        detail = "; ".join(errors)
        super().__init__(f"Manifesto '{path}' invalido: {detail}")


# --- Modelo ---


class ModelError(TheoError):
    """Erro relacionado a modelos."""


class ModelNotFoundError(ModelError):
    """Modelo nao encontrado no registry."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Modelo '{model_name}' nao encontrado no registry")


class ModelLoadError(ModelError):
    """Falha ao carregar modelo em memoria."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Falha ao carregar modelo '{model_name}': {reason}")


# --- Worker ---


class WorkerError(TheoError):
    """Erro relacionado a workers (subprocessos gRPC)."""


class WorkerCrashError(WorkerError):
    """Worker crashou durante operacao."""

    def __init__(self, worker_id: str, exit_code: int | None = None) -> None:
        self.worker_id = worker_id
        self.exit_code = exit_code
        msg = f"Worker '{worker_id}' crashou"
        if exit_code is not None:
            msg += f" (exit code: {exit_code})"
        super().__init__(msg)


class WorkerTimeoutError(WorkerError):
    """Worker nao respondeu dentro do timeout."""

    def __init__(self, worker_id: str, timeout_seconds: float) -> None:
        self.worker_id = worker_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Worker '{worker_id}' nao respondeu em {timeout_seconds}s"
        )


class WorkerUnavailableError(WorkerError):
    """Nenhum worker disponivel para atender a request."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Nenhum worker disponivel para modelo '{model_name}'"
        )


# --- Audio ---


class AudioError(TheoError):
    """Erro relacionado a processamento de audio."""


class AudioFormatError(AudioError):
    """Formato de audio nao suportado ou invalido."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Formato de audio invalido: {detail}")


class AudioTooLargeError(AudioError):
    """Arquivo de audio excede o limite permitido."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"Arquivo de audio ({size_mb:.1f}MB) excede limite de {max_mb:.1f}MB"
        )


# --- Sessao ---


class SessionError(TheoError):
    """Erro relacionado a sessoes de streaming."""


class SessionNotFoundError(SessionError):
    """Sessao nao encontrada."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Sessao '{session_id}' nao encontrada")


class SessionClosedError(SessionError):
    """Operacao tentada em sessao ja fechada."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Sessao '{session_id}' ja esta fechada")
```

**Testes** (`tests/unit/test_exceptions.py`):

```python
"""Testes das exceptions tipadas do Theo."""

import pytest

from theo.exceptions import (
    AudioFormatError,
    AudioTooLargeError,
    ConfigError,
    ManifestParseError,
    ManifestValidationError,
    ModelLoadError,
    ModelNotFoundError,
    SessionClosedError,
    SessionNotFoundError,
    TheoError,
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)


class TestHierarchy:
    def test_all_exceptions_inherit_from_theo_error(self) -> None:
        exceptions = [
            ManifestParseError("f", "r"),
            ManifestValidationError("f", ["e"]),
            ModelNotFoundError("m"),
            ModelLoadError("m", "r"),
            WorkerCrashError("w"),
            WorkerTimeoutError("w", 5.0),
            WorkerUnavailableError("m"),
            AudioFormatError("d"),
            AudioTooLargeError(100, 50),
            SessionNotFoundError("s"),
            SessionClosedError("s"),
        ]
        for exc in exceptions:
            assert isinstance(exc, TheoError)

    def test_config_errors_are_config_error(self) -> None:
        assert isinstance(ManifestParseError("f", "r"), ConfigError)
        assert isinstance(ManifestValidationError("f", ["e"]), ConfigError)


class TestExceptionMessages:
    def test_model_not_found_has_model_name(self) -> None:
        exc = ModelNotFoundError("faster-whisper-large-v3")
        assert "faster-whisper-large-v3" in str(exc)
        assert exc.model_name == "faster-whisper-large-v3"

    def test_worker_crash_with_exit_code(self) -> None:
        exc = WorkerCrashError("worker-1", exit_code=137)
        assert "137" in str(exc)
        assert exc.exit_code == 137

    def test_worker_crash_without_exit_code(self) -> None:
        exc = WorkerCrashError("worker-1")
        assert exc.exit_code is None

    def test_audio_too_large_shows_mb(self) -> None:
        exc = AudioTooLargeError(
            size_bytes=30 * 1024 * 1024,
            max_bytes=25 * 1024 * 1024,
        )
        assert "30.0MB" in str(exc)
        assert "25.0MB" in str(exc)

    def test_manifest_validation_joins_errors(self) -> None:
        exc = ManifestValidationError("theo.yaml", ["campo 'name' faltando", "campo 'type' invalido"])
        assert "campo 'name' faltando" in str(exc)
        assert "campo 'type' invalido" in str(exc)

    def test_exceptions_are_catchable_by_base(self) -> None:
        with pytest.raises(TheoError):
            raise ModelNotFoundError("test")
```

**Criterio de done**:
- `python -m mypy src/theo/exceptions.py` passa
- `python -m pytest tests/unit/test_exceptions.py` passa
- Toda exception tem atributos tipados acessiveis (nao so mensagem string)

---

### Iniciativa I6 -- Configuracao e Parsing de Manifesto

**Objetivo**: Implementar o parsing e validacao de `theo.yaml` (manifesto de modelo) e as configuracoes de runtime (preprocessing, postprocessing).

**Responsavel**: Sofia Castellani

**Dependencia**: I4 (usa `STTArchitecture`, `ModelType`), I5 (usa `ManifestParseError`, `ManifestValidationError`)

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/config/manifest.py` | `ModelManifest` -- parsing e validacao de theo.yaml |
| `src/theo/config/preprocessing.py` | `PreprocessingConfig` -- configuracao do pipeline de audio |
| `src/theo/config/postprocessing.py` | `PostProcessingConfig` -- configuracao do pipeline de texto |
| `tests/unit/test_manifest.py` | Testes de parsing (valid, minimal, invalid) |
| `tests/unit/test_config.py` | Testes de configs de pipeline |
| `tests/fixtures/manifests/*.yaml` | Manifestos de teste |

**Modelo Pydantic para manifesto (`src/theo/config/manifest.py`)**:

```python
"""Parsing e validacao de manifestos theo.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from theo._types import ModelType, STTArchitecture
from theo.exceptions import ManifestParseError, ManifestValidationError


class ModelCapabilities(BaseModel):
    """Capabilities declaradas no manifesto."""

    streaming: bool = False
    architecture: STTArchitecture | None = None
    languages: list[str] = []
    word_timestamps: bool = False
    translation: bool = False
    partial_transcripts: bool = False
    hot_words: bool = False
    batch_inference: bool = False
    language_detection: bool = False
    initial_prompt: bool = False


class ModelResources(BaseModel):
    """Recursos necessarios para o modelo."""

    memory_mb: int
    gpu_required: bool = False
    gpu_recommended: bool = False
    load_time_seconds: int = 10


class EngineConfig(BaseModel, extra="allow"):
    """Configuracao especifica da engine.

    Aceita campos extras porque cada engine tem parametros proprios.
    """

    model_size: str | None = None
    compute_type: str = "float16"
    device: str = "auto"
    beam_size: int = 5
    vad_filter: bool = False


class ModelManifest(BaseModel):
    """Manifesto de modelo Theo (theo.yaml).

    Descreve capabilities, recursos e configuracao de um modelo
    instalado no registry local.
    """

    name: str
    version: str
    engine: str
    model_type: ModelType
    description: str = ""
    capabilities: ModelCapabilities = ModelCapabilities()
    resources: ModelResources
    engine_config: EngineConfig = EngineConfig()

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            msg = f"Nome de modelo invalido: '{v}'. Use apenas alfanumericos, hifens e underscores."
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> ModelManifest:
        """Carrega manifesto a partir de arquivo YAML."""
        path = Path(path)
        if not path.exists():
            raise ManifestParseError(str(path), "Arquivo nao encontrado")

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ManifestParseError(str(path), f"Erro ao ler arquivo: {e}") from e

        return cls.from_yaml_string(raw, source_path=str(path))

    @classmethod
    def from_yaml_string(
        cls, raw: str, source_path: str = "<string>"
    ) -> ModelManifest:
        """Carrega manifesto a partir de string YAML."""
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ManifestParseError(source_path, f"YAML invalido: {e}") from e

        if not isinstance(data, dict):
            raise ManifestParseError(source_path, "Conteudo YAML deve ser um mapeamento")

        # Normalizar campo 'type' -> 'model_type' (theo.yaml usa 'type')
        if "type" in data and "model_type" not in data:
            data["model_type"] = data.pop("type")

        try:
            return cls.model_validate(data)
        except Exception as e:
            errors = [str(e)]
            raise ManifestValidationError(source_path, errors) from e
```

**Configs de pipeline (`src/theo/config/preprocessing.py`)**:

```python
"""Configuracao do Audio Preprocessing Pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    """Configuracao do pipeline de preprocessamento de audio.

    Cada stage e toggleavel independentemente.
    """

    resample: bool = True
    target_sample_rate: int = 16000
    dc_remove: bool = True
    dc_remove_cutoff_hz: int = 20
    gain_normalize: bool = True
    target_dbfs: float = -3.0
    normalize_window_ms: int = 500
    denoise: bool = False
    denoise_engine: str = "rnnoise"
```

**Config de postprocessing (`src/theo/config/postprocessing.py`)**:

```python
"""Configuracao do Post-Processing Pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class ITNConfig(BaseModel):
    """Configuracao de Inverse Text Normalization."""

    enabled: bool = True
    language: str = "pt"


class EntityFormattingConfig(BaseModel):
    """Configuracao de formatacao de entidades."""

    enabled: bool = False
    domain: str = "generic"


class HotWordCorrectionConfig(BaseModel):
    """Configuracao de correcao de hot words."""

    enabled: bool = False
    max_edit_distance: int = 2


class PostProcessingConfig(BaseModel):
    """Configuracao do pipeline de pos-processamento de texto."""

    itn: ITNConfig = ITNConfig()
    entity_formatting: EntityFormattingConfig = EntityFormattingConfig()
    hot_word_correction: HotWordCorrectionConfig = HotWordCorrectionConfig()
```

**Fixtures de manifesto (`tests/fixtures/manifests/valid_stt.yaml`)**:

```yaml
name: faster-whisper-large-v3
version: 3.0.0
engine: faster-whisper
type: stt
description: "Faster Whisper Large V3 - encoder-decoder STT"

capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["auto", "en", "pt", "es"]
  word_timestamps: true
  translation: true
  partial_transcripts: true
  hot_words: false
  batch_inference: true
  language_detection: true
  initial_prompt: true

resources:
  memory_mb: 3072
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 8

engine_config:
  model_size: "large-v3"
  compute_type: "float16"
  device: "auto"
  beam_size: 5
  vad_filter: false
```

**Testes** (`tests/unit/test_manifest.py`) -- cenarios chave:

- Parse de manifesto valido completo
- Parse de manifesto com campos minimos obrigatorios
- Erro em manifesto com campo `name` faltando
- Erro em manifesto com YAML sintaticamente invalido
- Normalizacao de `type` para `model_type`
- Validacao de nome de modelo (caracteres invalidos)
- Arquivo inexistente gera `ManifestParseError`
- Campo `architecture` parseia corretamente para enum

**Criterio de done**:
- `python -m mypy src/theo/config/` passa
- `python -m pytest tests/unit/test_manifest.py tests/unit/test_config.py` passa
- Manifesto valido do PRD (Faster-Whisper) parseia sem erros

---

### Iniciativa I7 -- Interface STTBackend e Protobuf

**Objetivo**: Definir a interface abstrata que toda engine STT deve implementar e o contrato gRPC de comunicacao runtime-worker.

**Responsavel**: Sofia Castellani (interface), Viktor Sorokin (protobuf)

**Dependencia**: I4 (usa todos os tipos)

**Entregaveis**:

| Arquivo | Descricao |
|---------|-----------|
| `src/theo/workers/stt/interface.py` | `STTBackend` ABC |
| `src/theo/proto/stt_worker.proto` | Contrato gRPC (definicao, sem compilacao) |
| `tests/unit/test_stt_interface.py` | Testes da interface (contrato, nao implementacao) |

**Interface STTBackend (`src/theo/workers/stt/interface.py`)**:

```python
"""Interface abstrata para backends STT.

Todo backend STT (Faster-Whisper, WeNet, Paraformer) deve implementar
esta interface para ser plugavel no runtime Theo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from theo._types import (
    BatchResult,
    EngineCapabilities,
    STTArchitecture,
    TranscriptSegment,
)


class STTBackend(ABC):
    """Contrato que toda engine STT deve implementar.

    O runtime interage com engines exclusivamente atraves desta interface.
    Adicionar uma nova engine requer:
    1. Implementar STTBackend
    2. Criar manifesto theo.yaml
    3. Registrar no Model Registry
    Zero mudancas no runtime core.
    """

    @property
    @abstractmethod
    def architecture(self) -> STTArchitecture:
        """Arquitetura do modelo (encoder-decoder, ctc, streaming-native).

        Determina como o runtime adapta o pipeline de streaming.
        """
        ...

    @abstractmethod
    async def load(self, model_path: str, config: dict[str, object]) -> None:
        """Carrega o modelo em memoria.

        Args:
            model_path: Caminho para os arquivos do modelo.
            config: engine_config do manifesto theo.yaml.

        Raises:
            ModelLoadError: Se o modelo nao puder ser carregado.
        """
        ...

    @abstractmethod
    async def capabilities(self) -> EngineCapabilities:
        """Capabilities da engine em runtime.

        Pode diferir do manifesto se a engine descobrir capabilities
        adicionais apos load.
        """
        ...

    @abstractmethod
    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        """Transcreve audio completo (batch).

        Args:
            audio_data: Audio PCM 16-bit, 16kHz, mono (ja preprocessado).
            language: Codigo ISO 639-1, "auto", ou "mixed".
            initial_prompt: Contexto para guiar transcricao.
            hot_words: Palavras para keyword boosting.
            temperature: Temperatura de sampling (0.0-1.0).
            word_timestamps: Se True, retorna timestamps por palavra.

        Returns:
            BatchResult com texto, idioma, duracao e segmentos.
        """
        ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcreve audio em streaming.

        Recebe chunks de audio PCM 16-bit, 16kHz, mono (ja preprocessado
        pelo Audio Preprocessing Pipeline do runtime).

        Para encoder-decoder: runtime usa LocalAgreement para partials.
        Para CTC: engine produz partials nativos frame-by-frame.
        Para streaming-native: engine gerencia estado interno.

        Args:
            audio_chunks: Iterator assincrono de chunks PCM.
            language: Codigo ISO 639-1, "auto", ou "mixed".
            initial_prompt: Contexto do segmento anterior.
            hot_words: Palavras para keyword boosting.

        Yields:
            TranscriptSegment (partial e final).
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Descarrega o modelo da memoria.

        Libera recursos (GPU memory, buffers). Apos unload, load()
        deve ser chamado novamente antes de transcribe.
        """
        ...

    @abstractmethod
    async def health(self) -> dict[str, str]:
        """Status do backend.

        Returns:
            Dict com pelo menos {"status": "ok"|"loading"|"error"}.
        """
        ...
```

**Protobuf** (`src/theo/proto/stt_worker.proto`): Conteudo definido na secao 2.5 deste documento.

**Testes da interface** (`tests/unit/test_stt_interface.py`):

```python
"""Testes do contrato STTBackend.

Verifica que a interface esta corretamente definida e que
implementacoes concretas sao forcadas a implementar todos os metodos.
"""

from collections.abc import AsyncIterator

import pytest

from theo._types import (
    BatchResult,
    EngineCapabilities,
    STTArchitecture,
    TranscriptSegment,
)
from theo.workers.stt.interface import STTBackend


class IncompleteBackend(STTBackend):
    """Backend que nao implementa nenhum metodo abstrato."""

    pass


class MinimalBackend(STTBackend):
    """Backend com implementacao minima para validar o contrato."""

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        pass

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

    async def transcribe_file(
        self,
        audio_data: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
    ) -> BatchResult:
        return BatchResult(
            text="", language="pt", duration=0.0, segments=()
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        return
        yield  # noqa: RUF027 - necessario para async generator

    async def unload(self) -> None:
        pass

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


class TestSTTBackendContract:
    def test_cannot_instantiate_incomplete(self) -> None:
        with pytest.raises(TypeError):
            IncompleteBackend()  # type: ignore[abstract]

    def test_can_instantiate_complete(self) -> None:
        backend = MinimalBackend()
        assert backend.architecture == STTArchitecture.ENCODER_DECODER

    async def test_health_returns_dict(self) -> None:
        backend = MinimalBackend()
        result = await backend.health()
        assert "status" in result

    async def test_capabilities_returns_engine_capabilities(self) -> None:
        backend = MinimalBackend()
        caps = await backend.capabilities()
        assert isinstance(caps, EngineCapabilities)

    async def test_transcribe_file_returns_batch_result(self) -> None:
        backend = MinimalBackend()
        result = await backend.transcribe_file(audio_data=b"fake_audio")
        assert isinstance(result, BatchResult)
```

**Criterio de done**:
- `python -m mypy src/theo/workers/stt/interface.py` passa
- `python -m pytest tests/unit/test_stt_interface.py` passa
- Protobuf `.proto` esta sintaticamente valido (verificavel com `protoc --lint_out=.` ou validacao manual)
- A interface `STTBackend` pode ser subclassada e mypy valida a implementacao

---

### Iniciativa I-AUX -- Fixtures de Audio

**Objetivo**: Criar audio samples para uso em testes de todos os milestones.

**Responsavel**: Andre Oliveira

**Dependencia**: I1 (diretorio de fixtures existe)

**Entregaveis**:

| Arquivo | Especificacao |
|---------|---------------|
| `tests/fixtures/audio/sample_16khz.wav` | PCM 16-bit, 16kHz, mono, ~3 segundos, tom senoidal 440Hz |
| `tests/fixtures/audio/sample_8khz.wav` | PCM 16-bit, 8kHz, mono, ~3 segundos, tom senoidal 440Hz |
| `tests/fixtures/audio/sample_44khz.wav` | PCM 16-bit, 44.1kHz, mono, ~3 segundos, tom senoidal 440Hz |

**Nota**: Estes nao sao audios com fala (nao precisamos de fala para testes unitarios de M1). Sao tons sintetizados que servem para:
- M1: Testar que fixtures existem e sao carregaveis
- M2: Validar que o worker recebe e processa bytes corretamente
- M4: Testar resample (44.1kHz -> 16kHz), DC remove, gain normalize

Audio com fala real sera adicionado em M2 (teste de integracao com Faster-Whisper).

**Geracao**: Usar script Python com `struct` + `math` (zero dependencias externas) ou `scipy` se disponivel.

**Criterio de done**: Arquivos existem, sao WAV validos, e `conftest.py` tem fixture para carrega-los.

**`tests/conftest.py`**:

```python
"""Fixtures compartilhadas para todos os testes."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
MANIFESTS_DIR = FIXTURES_DIR / "manifests"


@pytest.fixture
def audio_16khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 16kHz, mono."""
    path = AUDIO_DIR / "sample_16khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_8khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 8kHz, mono."""
    path = AUDIO_DIR / "sample_8khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def audio_44khz() -> Path:
    """Caminho para audio sample PCM 16-bit, 44.1kHz, mono."""
    path = AUDIO_DIR / "sample_44khz.wav"
    assert path.exists(), f"Fixture de audio nao encontrada: {path}"
    return path


@pytest.fixture
def valid_stt_manifest_path() -> Path:
    """Caminho para manifesto STT valido."""
    path = MANIFESTS_DIR / "valid_stt.yaml"
    assert path.exists(), f"Fixture de manifesto nao encontrada: {path}"
    return path
```

---

## 4. Sequencia de Implementacao

### 4.1 Ordem de Execucao

```
Semana 1 (dias 1-3):
  ┌─────────────────────────────────────────────────┐
  │ I1: pyproject.toml + estrutura de diretorios     │
  │     (Andre) -- 0.5 dia                          │
  ├─────────────────────────────────────────────────┤
  │ I2: Configuracao de tooling (ruff, mypy, pytest) │
  │     (Andre) -- 0.5 dia                          │
  ├─────────────────────────────────────────────────┤
  │ I4: Tipos e Enums base (_types.py)               │  <- pode ser paralelo com I2
  │     (Sofia) -- 1 dia                            │
  ├─────────────────────────────────────────────────┤
  │ I-AUX: Fixtures de audio + manifestos            │  <- paralelo com I4
  │     (Andre) -- 0.5 dia                          │
  └─────────────────────────────────────────────────┘

Semana 1 (dias 3-5):
  ┌─────────────────────────────────────────────────┐
  │ I5: Exceptions tipadas                           │  <- depende de I4
  │     (Sofia) -- 0.5 dia                          │
  ├─────────────────────────────────────────────────┤
  │ I6: Configuracao e Manifesto                     │  <- depende de I4, I5
  │     (Sofia) -- 1 dia                            │
  ├─────────────────────────────────────────────────┤
  │ I7: Interface STTBackend + Protobuf              │  <- depende de I4
  │     (Sofia + Viktor) -- 1 dia                   │
  ├─────────────────────────────────────────────────┤
  │ I3: CI (GitHub Actions)                          │  <- depende de I2
  │     (Andre) -- 0.5 dia                          │
  └─────────────────────────────────────────────────┘
```

### 4.2 Justificativa da Sequencia

| Ordem | Iniciativa | Por que nesta posicao |
|-------|------------|----------------------|
| 1 | I1 (estrutura) | Tudo depende do pyproject.toml. Sem ele, nada instala. |
| 2 | I2 (tooling) | Configurar ferramentas antes de escrever codigo garante que TODO codigo ja nasce validado. |
| 2' | I4 (tipos) | Pode comecar em paralelo com I2 porque nao depende de tooling estar rodando -- mas sera validado por ele. |
| 2'' | I-AUX (fixtures) | Paralelo com I4. Gerar audios e manifestos e independente. |
| 3 | I5 (exceptions) | Depende de I4 (usa enums). Rapido -- meio dia. |
| 3' | I7 (interface + proto) | Depende de I4 (usa tipos). Pode ser paralelo com I5. |
| 4 | I6 (config/manifesto) | Depende de I4 e I5 (usa tipos e exceptions). |
| 5 | I3 (CI) | Pode ser feito a qualquer momento apos I2, mas faz mais sentido apos existir codigo para validar (I4-I7). |

### 4.3 Paralelismo

Com 2 pessoas (Sofia + Andre), o paralelismo maximo e:

| Dia | Sofia | Andre |
|-----|-------|-------|
| 1 | -- | I1 + I2 |
| 2 | I4 (tipos) | I-AUX (fixtures) |
| 3 | I5 (exceptions) + I7 (interface, com Viktor) | I3 (CI) |
| 4 | I6 (config/manifesto) | Review de I4-I7, ajustes CI |
| 5 | Review cruzado, ajustes finais | Validacao end-to-end |

**Viktor** participa pontualmente em I7 (definicao do protobuf e review da interface async).

---

## 5. Definition of Done do M1

O M1 esta completo quando TODOS os itens abaixo forem verdadeiros:

### 5.1 Funcional

| # | Criterio | Comando de Verificacao |
|---|----------|----------------------|
| 1 | Projeto instala sem erros | `pip install -e ".[dev]"` |
| 2 | Import do pacote funciona | `python -c "import theo; print(theo.__version__)"` -> `0.1.0` |
| 3 | Lint passa sem erros | `python -m ruff check src/ tests/` |
| 4 | Format esta correto | `python -m ruff format --check src/ tests/` |
| 5 | Typecheck passa sem erros | `python -m mypy src/` |
| 6 | Testes unitarios passam | `python -m pytest tests/unit/ -v` |
| 7 | CI passa no GitHub | Status verde para Python 3.11 e 3.12 |

### 5.2 Artefatos Existentes

| # | Artefato | Localizacao |
|---|----------|-------------|
| 1 | `pyproject.toml` com todas as dependencias e extras | Raiz do projeto |
| 2 | Tipos base (`_types.py`) | `src/theo/_types.py` |
| 3 | Exceptions tipadas | `src/theo/exceptions.py` |
| 4 | Manifesto parser (`ModelManifest`) | `src/theo/config/manifest.py` |
| 5 | Configs de pipeline | `src/theo/config/preprocessing.py`, `postprocessing.py` |
| 6 | Interface `STTBackend` | `src/theo/workers/stt/interface.py` |
| 7 | Protobuf definido | `src/theo/proto/stt_worker.proto` |
| 8 | CI configurado | `.github/workflows/ci.yml` |
| 9 | CHANGELOG.md | Raiz do projeto |
| 10 | Fixtures de audio (3 samples) | `tests/fixtures/audio/` |
| 11 | Fixtures de manifesto (4 arquivos) | `tests/fixtures/manifests/` |
| 12 | Marker `py.typed` | `src/theo/py.typed` |

### 5.3 Qualidade

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Zero `Any` nos type hints (exceto proto gerado) | `mypy --strict` passa |
| 2 | Toda interface publica (ABC) tem docstring | Review manual |
| 3 | Todo teste segue padrao AAA (Arrange-Act-Assert) | Review manual |
| 4 | Nomes de testes descrevem comportamento | Review manual |
| 5 | Imports sao absolutos (`from theo.xxx`) | `ruff check` com regra I valida |
| 6 | Exceptions tem atributos tipados | Testes em test_exceptions.py |
| 7 | Dataclasses imutaveis usam `frozen=True, slots=True` | Review manual |

### 5.4 Documentacao

| # | Criterio |
|---|----------|
| 1 | CHANGELOG.md tem secao `[Unreleased]` com entradas do M1 |
| 2 | CLAUDE.md atualizado se algum padrao mudar |
| 3 | README.md atualizado com instrucoes de instalacao |

### 5.5 Checklist Final (executar em sequencia)

```bash
# 1. Instalacao limpa
pip install -e ".[dev]"

# 2. Verificacao de versao
python -c "import theo; print(theo.__version__)"

# 3. Qualidade de codigo
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy src/

# 4. Testes
python -m pytest tests/unit/ -v

# 5. Validacao de tipos importaveis
python -c "
from theo._types import STTArchitecture, TranscriptSegment, BatchResult
from theo.exceptions import TheoError, ModelNotFoundError
from theo.config.manifest import ModelManifest
from theo.config.preprocessing import PreprocessingConfig
from theo.config.postprocessing import PostProcessingConfig
from theo.workers.stt.interface import STTBackend
print('Todos os tipos importaveis com sucesso')
"

# 6. Validacao de manifesto
python -c "
from theo.config.manifest import ModelManifest
m = ModelManifest.from_yaml_path('tests/fixtures/manifests/valid_stt.yaml')
print(f'Manifesto: {m.name} ({m.model_type.value})')
print(f'Arquitetura: {m.capabilities.architecture}')
"
```

Se todos os comandos acima passam sem erros, M1 esta completo.

---

## 6. O Que NAO Esta no Escopo do M1

Aplicando YAGNI rigorosamente, os seguintes itens foram explicitamente excluidos do M1:

| Item | Por que nao | Quando entra |
|------|-------------|-------------|
| Compilacao de protobuf (`grpcio-tools`) | Definir o proto e suficiente. Compilar so em M2 quando o worker for implementado. | M2 |
| FastAPI app | Nenhum endpoint existe em M1. Declarar como extra e suficiente. | M3 |
| CLI (typer/click) | Nenhum comando funcional em M1. | M3 |
| Model Registry | Parsing de manifesto (M1) != gerenciamento de lifecycle (M3). | M3 |
| Structured logging | Util em M2+ quando workers existem. Overkill para tipos e configs. | M2 |
| Docker / Dockerfile | Zero valor sem codigo executavel. | M3 |
| Metricas Prometheus | Nenhum servidor para expor metricas. | M3 |
| TTSBackend interface | Foco e STT. TTS vira quando houver demand real. YAGNI. | Futuro |
| Scripts de geracao de audio | Audio fixtures podem ser gerados manualmente com qualquer ferramenta. Um script seria over-engineering. | N/A |

---

## 7. Riscos Especificos do M1

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | Interface `STTBackend` precisa mudar em M2 quando confrontada com Faster-Whisper real | Media | Baixo | Custo de mudar a interface e baixo em M1 (zero implementacoes existentes). Validar com pseudocodigo de Faster-Whisper antes de finalizar. |
| R2 | Protobuf gRPC precisa de campos adicionais em M2 | Baixa | Baixo | Protobuf e forward-compatible (campos novos nao quebram clients antigos). Adicionar campos e trivial. |
| R3 | `pydantic v2` tem breaking changes sutis vs v1 | Baixa | Medio | Usar apenas features estaveis de v2 (BaseModel, field_validator). Testar manifesto parser extensivamente. |
| R4 | Estrutura de pacotes precisa mudar | Baixa | Baixo | Em M1 nao ha codigo suficiente para tornar refatoracao dolorosa. Mudar no inicio e barato. |
| R5 | `pytest-asyncio` com `asyncio_mode = "auto"` tem edge cases | Baixa | Baixo | Testar que testes async rodam corretamente. Fallback: usar `@pytest.mark.asyncio` explicitamente. |

---

## 8. Transicao M1 -> M2

Ao completar M1, o time deve ter:

1. **Confianca na interface `STTBackend`** -- validada com `MinimalBackend` nos testes e revisada contra a API real do Faster-Whisper.

2. **Protobuf definido e revisado** -- pronto para compilacao em M2 com `grpcio-tools`.

3. **Loop de feedback funcionando** -- qualquer novo codigo passa por lint + typecheck + testes automaticamente (local e CI).

4. **Vocabulario compartilhado** -- tipos, enums e exceptions que toda conversa sobre o sistema pode referenciar sem ambiguidade.

O primeiro commit de M2 sera: compilar o protobuf e criar o scaffolding do worker subprocess. A interface `STTBackend` e o contrato gRPC ja estao definidos -- M2 e implementacao, nao design.

---

*Documento gerado pelo Time de Arquitetura (ARCH). Sera atualizado conforme a implementacao do M1 avanca.*
