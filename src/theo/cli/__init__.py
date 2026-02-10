"""CLI do Theo OpenVoice.

Registra todos os comandos no grupo principal.
"""

from theo.cli.main import cli
from theo.cli.models import inspect, list_models
from theo.cli.ps import ps
from theo.cli.pull import pull
from theo.cli.remove import remove
from theo.cli.serve import serve
from theo.cli.transcribe import transcribe, translate

__all__ = [
    "cli",
    "inspect",
    "list_models",
    "ps",
    "pull",
    "remove",
    "serve",
    "transcribe",
    "translate",
]
