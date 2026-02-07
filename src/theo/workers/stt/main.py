"""Entry point do worker STT como subprocess gRPC.

Uso:
    python -m theo.workers.stt --port 50051 --engine faster-whisper \
        --model-path /models/large-v3 --model-size large-v3
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import TYPE_CHECKING

import grpc.aio

from theo.logging import configure_logging, get_logger
from theo.proto import add_STTWorkerServicer_to_server
from theo.workers.stt.servicer import STTWorkerServicer

if TYPE_CHECKING:
    from theo.workers.stt.interface import STTBackend

logger = get_logger("worker.stt.main")

STOP_GRACE_PERIOD = 5.0


def _create_backend(engine: str) -> STTBackend:
    """Cria a instancia do STTBackend baseado no nome da engine.

    Raises:
        ValueError: Se a engine nao e suportada.
    """
    if engine == "faster-whisper":
        from theo.workers.stt.faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend()

    msg = f"Engine STT nao suportada: {engine}"
    raise ValueError(msg)


async def serve(
    port: int,
    engine: str,
    model_path: str,
    engine_config: dict[str, object],
) -> None:
    """Inicia o servidor gRPC do worker STT.

    Args:
        port: Porta para escutar.
        engine: Nome da engine (ex: "faster-whisper").
        model_path: Caminho para arquivos do modelo.
        engine_config: Configuracoes da engine (compute_type, device, etc).
    """
    backend = _create_backend(engine)

    logger.info("loading_model", engine=engine, model_path=model_path)
    await backend.load(model_path, engine_config)
    logger.info("model_loaded", engine=engine)

    model_name = str(engine_config.get("model_size", "unknown"))
    servicer = STTWorkerServicer(
        backend=backend,
        model_name=model_name,
        engine=engine,
    )

    server = grpc.aio.server()
    add_STTWorkerServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    loop = asyncio.get_running_loop()

    async def _shutdown() -> None:
        logger.info("shutdown_start", grace_period=STOP_GRACE_PERIOD)
        await server.stop(STOP_GRACE_PERIOD)
        await backend.unload()
        logger.info("shutdown_complete")

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown()))

    await server.start()
    logger.info("worker_started", port=port, engine=engine)

    await server.wait_for_termination()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Theo STT Worker (gRPC)")
    parser.add_argument("--port", type=int, default=50051, help="Porta gRPC (default: 50051)")
    parser.add_argument(
        "--engine", type=str, default="faster-whisper", help="Engine STT (default: faster-whisper)"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Caminho do modelo")
    parser.add_argument(
        "--compute-type", type=str, default="float16", help="Tipo de compute (default: float16)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda (default: auto)"
    )
    parser.add_argument(
        "--model-size", type=str, default="large-v3", help="Tamanho do modelo (default: large-v3)"
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size para decoding (default: 5)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point principal do worker STT."""
    configure_logging()
    args = parse_args(argv)

    engine_config: dict[str, object] = {
        "model_size": args.model_size,
        "compute_type": args.compute_type,
        "device": args.device,
        "beam_size": args.beam_size,
    }

    try:
        asyncio.run(
            serve(
                port=args.port,
                engine=args.engine,
                model_path=args.model_path,
                engine_config=engine_config,
            )
        )
    except KeyboardInterrupt:
        logger.info("worker_interrupted")
        sys.exit(0)
