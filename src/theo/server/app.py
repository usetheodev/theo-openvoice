"""FastAPI application factory para o Theo OpenVoice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

import theo
from theo.server.error_handlers import register_error_handlers
from theo.server.routes import health, transcriptions, translations

if TYPE_CHECKING:
    from theo.registry.registry import ModelRegistry
    from theo.scheduler.scheduler import Scheduler


def create_app(
    registry: ModelRegistry | None = None,
    scheduler: Scheduler | None = None,
) -> FastAPI:
    """Cria a aplicacao FastAPI.

    Args:
        registry: Model Registry (opcional, None apenas para testes do health endpoint).
        scheduler: Scheduler (opcional, None apenas para testes do health endpoint).

    Returns:
        FastAPI application configurada.
    """
    app = FastAPI(
        title="Theo OpenVoice",
        version=theo.__version__,
        description="Runtime unificado de voz (STT + TTS) com API OpenAI-compatible",
    )

    app.state.registry = registry
    app.state.scheduler = scheduler

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(transcriptions.router)
    app.include_router(translations.router)

    return app
