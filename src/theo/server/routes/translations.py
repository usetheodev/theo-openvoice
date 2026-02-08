"""POST /v1/audio/translations — traducao de audio para ingles."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, UploadFile

from theo.scheduler.scheduler import Scheduler  # noqa: TC001
from theo.server.dependencies import get_scheduler
from theo.server.routes._common import handle_audio_request

router = APIRouter()


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile,
    model: str = Form(),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    scheduler: Scheduler = Depends(get_scheduler),  # noqa: B008
) -> Any:
    """Traduz audio para ingles.

    Compativel com OpenAI Audio API POST /v1/audio/translations.
    """
    return await handle_audio_request(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        task="translate",
        scheduler=scheduler,
    )
