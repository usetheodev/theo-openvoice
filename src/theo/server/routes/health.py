"""Health check endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

import theo

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Health check do runtime.

    Retorna status basico (liveness) e informacao de readiness
    baseada no estado do registry e scheduler.
    """
    response: dict[str, Any] = {
        "status": "ok",
        "version": theo.__version__,
    }

    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        models = registry.list_models()
        response["models_loaded"] = len(models)

    return response
