"""FastAPI dependencies para injecao do Registry e Scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002

if TYPE_CHECKING:
    from theo.registry.registry import ModelRegistry
    from theo.scheduler.scheduler import Scheduler


def get_registry(request: Request) -> ModelRegistry:
    """Retorna o ModelRegistry do app state.

    Raises:
        RuntimeError: Se registry nao foi configurado em create_app().
    """
    registry = request.app.state.registry
    if registry is None:
        raise RuntimeError("Registry nao configurado. Passe registry= em create_app().")
    return registry  # type: ignore[no-any-return]


def get_scheduler(request: Request) -> Scheduler:
    """Retorna o Scheduler do app state.

    Raises:
        RuntimeError: Se scheduler nao foi configurado em create_app().
    """
    scheduler = request.app.state.scheduler
    if scheduler is None:
        raise RuntimeError("Scheduler nao configurado. Passe scheduler= em create_app().")
    return scheduler  # type: ignore[no-any-return]
