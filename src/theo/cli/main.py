"""Grupo principal de comandos CLI do Theo OpenVoice."""

from __future__ import annotations

import click

import theo


@click.group()
@click.version_option(version=theo.__version__, prog_name="theo")
def cli() -> None:
    """Theo OpenVoice — Runtime unificado de voz (STT + TTS)."""
