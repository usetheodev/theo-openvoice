"""Comandos `theo transcribe` e `theo translate` — thin clients HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from theo.cli.main import cli

DEFAULT_SERVER_URL = "http://localhost:8000"


def _post_audio(
    server_url: str,
    endpoint: str,
    file_path: Path,
    model: str,
    response_format: str,
    language: str | None,
) -> None:
    """Envia audio para o server via HTTP e imprime resultado."""
    import httpx

    url = f"{server_url}{endpoint}"

    if not file_path.exists():
        click.echo(f"Erro: arquivo nao encontrado: {file_path}", err=True)
        sys.exit(1)

    data: dict[str, str] = {"model": model, "response_format": response_format}
    if language:
        data["language"] = language

    try:
        with file_path.open("rb") as f:
            response = httpx.post(
                url,
                files={"file": (file_path.name, f, "audio/wav")},
                data=data,
                timeout=120.0,
            )
    except httpx.ConnectError:
        click.echo(
            f"Erro: servidor nao disponivel em {server_url}. Execute 'theo serve' primeiro.",
            err=True,
        )
        sys.exit(1)

    if response.status_code != 200:
        try:
            error = response.json()
            msg = error.get("error", {}).get("message", response.text)
        except Exception:
            msg = response.text
        click.echo(f"Erro ({response.status_code}): {msg}", err=True)
        sys.exit(1)

    # Output depende do formato
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = response.json()
        if "text" in body:
            click.echo(body["text"])
        else:
            import json

            click.echo(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        click.echo(response.text)


@cli.command()
@click.argument("file", type=click.Path(exists=False))
@click.option("--model", "-m", required=True, help="Nome do modelo STT.")
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["json", "verbose_json", "text", "srt", "vtt"]),
    default="json",
    show_default=True,
    help="Formato de resposta.",
)
@click.option("--language", "-l", default=None, help="Codigo ISO 639-1 do idioma.")
@click.option(
    "--server",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="URL do servidor Theo.",
)
def transcribe(
    file: str,
    model: str,
    response_format: str,
    language: str | None,
    server: str,
) -> None:
    """Transcreve um arquivo de audio."""
    _post_audio(
        server_url=server,
        endpoint="/v1/audio/transcriptions",
        file_path=Path(file),
        model=model,
        response_format=response_format,
        language=language,
    )


@cli.command()
@click.argument("file", type=click.Path(exists=False))
@click.option("--model", "-m", required=True, help="Nome do modelo STT.")
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["json", "verbose_json", "text", "srt", "vtt"]),
    default="json",
    show_default=True,
    help="Formato de resposta.",
)
@click.option(
    "--server",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="URL do servidor Theo.",
)
def translate(
    file: str,
    model: str,
    response_format: str,
    server: str,
) -> None:
    """Traduz um arquivo de audio para ingles."""
    _post_audio(
        server_url=server,
        endpoint="/v1/audio/translations",
        file_path=Path(file),
        model=model,
        response_format=response_format,
        language=None,
    )
