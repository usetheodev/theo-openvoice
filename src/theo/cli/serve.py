"""Comando `theo serve` — inicia API Server + workers."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import click

from theo.cli.main import cli
from theo.logging import configure_logging, get_logger

logger = get_logger("cli.serve")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODELS_DIR = "~/.theo/models"
DEFAULT_WORKER_BASE_PORT = 50051


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Host para o API Server.")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Porta HTTP.")
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    show_default=True,
    help="Diretorio com modelos instalados.",
)
@click.option(
    "--log-format",
    type=click.Choice(["console", "json"]),
    default="console",
    show_default=True,
    help="Formato de log.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    show_default=True,
    help="Nivel de log.",
)
def serve(
    host: str,
    port: int,
    models_dir: str,
    log_format: str,
    log_level: str,
) -> None:
    """Inicia o Theo API Server com workers para modelos instalados."""
    configure_logging(log_format=log_format, level=log_level)
    asyncio.run(_serve(host, port, models_dir))


async def _serve(host: str, port: int, models_dir: str) -> None:
    """Fluxo async principal do serve."""
    import uvicorn

    from theo._types import ModelType
    from theo.registry.registry import ModelRegistry
    from theo.scheduler.scheduler import Scheduler
    from theo.server.app import create_app
    from theo.workers.manager import WorkerManager

    models_path = Path(models_dir).expanduser()

    # 1. Scan registry
    registry = ModelRegistry(models_path)
    await registry.scan()

    models = registry.list_models()
    if not models:
        logger.error("no_models_found", models_dir=str(models_path))
        click.echo(f"Erro: nenhum modelo encontrado em {models_path}", err=True)
        click.echo("Execute 'theo pull <model>' para instalar um modelo.", err=True)
        sys.exit(1)

    # 2. Spawn workers para modelos STT
    worker_manager = WorkerManager()
    port_counter = DEFAULT_WORKER_BASE_PORT

    stt_models = [m for m in models if m.model_type == ModelType.STT]
    for manifest in stt_models:
        model_path = str(registry.get_model_path(manifest.name))
        await worker_manager.spawn_worker(
            model_name=manifest.name,
            port=port_counter,
            engine=manifest.engine,
            model_path=model_path,
            engine_config=manifest.engine_config.model_dump(),
        )
        logger.info(
            "worker_spawned",
            model=manifest.name,
            engine=manifest.engine,
            port=port_counter,
        )
        port_counter += 1

    logger.info(
        "server_starting",
        host=host,
        port=port,
        models_count=len(models),
        stt_workers=len(stt_models),
    )

    # 3. Create app
    scheduler = Scheduler(worker_manager, registry)
    app = create_app(registry=registry, scheduler=scheduler)

    # 4. Setup shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: (
                logger.info("shutdown_signal", signal=signal.Signals(s).name),
                shutdown_event.set(),
            ),
        )

    # 5. Run uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    # Wait for shutdown signal or server to stop
    _done, _ = await asyncio.wait(
        [server_task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 6. Graceful shutdown
    if not server_task.done():
        server.should_exit = True
        await server_task

    logger.info("stopping_workers")
    await worker_manager.stop_all()
    logger.info("server_stopped")
