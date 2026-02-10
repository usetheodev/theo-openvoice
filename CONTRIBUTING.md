# Contributing to Theo OpenVoice

Thank you for your interest in contributing to Theo OpenVoice! This guide will help you get started.

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** for virtual environment and dependency management
- **make** for development workflow
- **Git** with conventional commit knowledge

## Development Setup

```bash
# Clone the repository
git clone https://github.com/usetheo/theo-openvoice.git
cd theo-openvoice

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Install all dependencies (including dev tools)
uv sync --all-extras

# Verify everything works
make ci
```

## Development Workflow

### Running checks

```bash
make check       # format (ruff) + lint (ruff) + typecheck (mypy strict)
make test-unit   # unit tests only (preferred during development)
make test        # all tests (1600+)
make ci          # full pipeline: format + lint + typecheck + test
```

### Running individual tests

```bash
.venv/bin/python -m pytest tests/unit/test_foo.py::test_bar -q
```

### Generating protobuf stubs

```bash
make proto
```

## Code Style

### Formatting and linting

- **ruff** handles both formatting and linting
- **mypy** runs in strict mode — all code must pass `mypy --strict`
- Run `make check` before committing

### Python conventions

- `from __future__ import annotations` is required in **all** source files
- Use `TYPE_CHECKING` blocks for imports used only in type hints (ruff TCH rules)
- Dataclasses use `frozen=True, slots=True` for immutable value objects
- Domain-specific exceptions in `src/theo/exceptions.py` — never raise bare `Exception`
- Async-first: all public interfaces are `async`
- Imports are absolute from `theo.` (e.g., `from theo.registry import Registry`)

### Naming

- `snake_case` for functions and variables
- `PascalCase` for classes
- Descriptive names over short names (`user_metadata` not `metadata`)

## Testing Guidelines

- Framework: **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed)
- Test names describe behavior: `test_transcription_fails_when_model_not_found`
- Follow **Arrange-Act-Assert** (AAA) pattern
- Each test tests **one thing**
- Tests must be independent — no shared mutable state between tests
- Use `unittest.mock` for inference engines in unit tests
- Mark integration tests with `@pytest.mark.integration`
- Never commit real audio files — generate test audio in fixtures

### Test structure

```
tests/
  unit/           # Mirrors src/theo/ structure
  integration/    # Tests requiring external resources
  fixtures/       # Shared test fixtures (audio, manifests)
  conftest.py     # Shared fixtures
```

## Pull Request Process

### Branch naming

Use descriptive branch names:
- `feat/add-paraformer-backend`
- `fix/ring-buffer-force-commit-race`
- `docs/update-websocket-protocol`

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Paraformer STT backend
fix: resolve race condition in ring buffer force commit
refactor: extract common audio decoding logic
test: add streaming stability test for 30-minute session
docs: update WebSocket protocol reference
```

### PR checklist

Before submitting your PR, ensure:

- [ ] `make check` passes (format + lint + typecheck)
- [ ] `make test-unit` passes with no new failures
- [ ] New code has unit tests covering business logic
- [ ] CHANGELOG.md is updated under `[Unreleased]`
- [ ] No secrets or credentials in committed files
- [ ] PR description explains the "why", not just the "what"

### CI checks

All PRs must pass:
1. **ruff** — formatting and linting
2. **mypy --strict** — type checking
3. **pytest** — unit and integration tests
4. **Build verification** — wheel builds correctly

## Adding a New STT Engine

If you're adding a new STT engine (e.g., Paraformer, Wav2Vec2), follow the step-by-step guide in [docs/ADDING_ENGINE.md](docs/ADDING_ENGINE.md). It requires changes to exactly 4 files plus tests — zero changes to the runtime core.

## Architecture

Before making significant changes, familiarize yourself with:
- [Architecture Document](docs/ARCHITECTURE.md) — system design and component interactions
- [PRD](docs/PRD.md) — product requirements and design decisions (ADRs)

## Getting Help

- Open a [GitHub Issue](https://github.com/usetheo/theo-openvoice/issues) for bug reports or feature requests
- Use [GitHub Discussions](https://github.com/usetheo/theo-openvoice/discussions) for questions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
