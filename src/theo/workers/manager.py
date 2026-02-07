"""Worker Manager — gerencia lifecycle de workers STT/TTS como subprocessos.

Responsabilidades:
- Spawn de workers como subprocessos gRPC
- Health probing com backoff exponencial
- Monitoramento de processo (crash detection)
- Auto-restart com rate limiting
- Shutdown graceful
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

from theo.logging import get_logger

logger = get_logger("worker.manager")

# Constants
MAX_CRASHES_IN_WINDOW = 3
CRASH_WINDOW_SECONDS = 60.0
HEALTH_PROBE_INITIAL_DELAY = 0.5
HEALTH_PROBE_MAX_DELAY = 5.0
HEALTH_PROBE_TIMEOUT = 30.0
STOP_GRACE_PERIOD = 5.0
MONITOR_INTERVAL = 1.0


class WorkerState(Enum):
    """Estado do worker no lifecycle."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"


@dataclass
class WorkerHandle:
    """Handle para um worker em execucao."""

    worker_id: str
    port: int
    model_name: str
    engine: str
    process: subprocess.Popen[bytes] | None = None
    state: WorkerState = WorkerState.STARTING
    crash_count: int = 0
    crash_timestamps: list[float] = field(default_factory=list)
    last_started_at: float = field(default_factory=time.monotonic)
    model_path: str = ""
    engine_config: dict[str, object] = field(default_factory=dict)


class WorkerManager:
    """Gerencia lifecycle de workers gRPC como subprocessos.

    Cada worker e um processo separado rodando o servicer STT gRPC.
    O manager cuida de spawn, health check, crash detection e restart.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerHandle] = {}
        self._tasks: dict[str, list[asyncio.Task[None]]] = {}

    async def spawn_worker(
        self,
        model_name: str,
        port: int,
        engine: str,
        model_path: str,
        engine_config: dict[str, object],
    ) -> WorkerHandle:
        """Inicia um novo worker como subprocess.

        Args:
            model_name: Nome do modelo (para identificacao).
            port: Porta gRPC para o worker escutar.
            engine: Nome da engine (ex: "faster-whisper").
            model_path: Caminho para os arquivos do modelo.
            engine_config: Configuracoes da engine.

        Returns:
            WorkerHandle com informacoes do worker.
        """
        worker_id = f"{engine}-{port}"

        cmd = [
            sys.executable,
            "-m",
            "theo.workers.stt",
            "--port",
            str(port),
            "--engine",
            engine,
            "--model-path",
            model_path,
        ]

        # Add engine config args
        if "compute_type" in engine_config:
            cmd.extend(["--compute-type", str(engine_config["compute_type"])])
        if "device" in engine_config:
            cmd.extend(["--device", str(engine_config["device"])])
        if "model_size" in engine_config:
            cmd.extend(["--model-size", str(engine_config["model_size"])])
        if "beam_size" in engine_config:
            cmd.extend(["--beam-size", str(engine_config["beam_size"])])

        logger.info("spawning_worker", worker_id=worker_id, port=port, engine=engine)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        handle = WorkerHandle(
            worker_id=worker_id,
            port=port,
            model_name=model_name,
            engine=engine,
            process=process,
            state=WorkerState.STARTING,
            model_path=model_path,
            engine_config=engine_config,
        )

        self._workers[worker_id] = handle

        # Start background tasks for health probe and monitoring
        health_task = asyncio.create_task(self._health_probe(worker_id))
        monitor_task = asyncio.create_task(self._monitor_worker(worker_id))
        self._tasks[worker_id] = [health_task, monitor_task]

        return handle

    async def stop_worker(self, worker_id: str) -> None:
        """Para um worker gracefully (SIGTERM, espera, SIGKILL se necessario).

        Args:
            worker_id: ID do worker.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        handle.state = WorkerState.STOPPING
        logger.info("stopping_worker", worker_id=worker_id)

        # Cancel background tasks
        for task in self._tasks.get(worker_id, []):
            task.cancel()

        process = handle.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, process.wait),
                    timeout=STOP_GRACE_PERIOD,
                )
            except TimeoutError:
                logger.warning("worker_force_kill", worker_id=worker_id)
                process.kill()
                await asyncio.get_running_loop().run_in_executor(None, process.wait)

        handle.state = WorkerState.STOPPED
        logger.info("worker_stopped", worker_id=worker_id)

    async def stop_all(self) -> None:
        """Para todos os workers em paralelo."""
        worker_ids = list(self._workers.keys())
        if worker_ids:
            await asyncio.gather(*(self.stop_worker(wid) for wid in worker_ids))

    def get_worker(self, worker_id: str) -> WorkerHandle | None:
        """Retorna handle do worker pelo ID."""
        return self._workers.get(worker_id)

    def get_ready_worker(self, model_name: str) -> WorkerHandle | None:
        """Retorna primeiro worker READY para um dado modelo."""
        for handle in self._workers.values():
            if handle.model_name == model_name and handle.state == WorkerState.READY:
                return handle
        return None

    async def _health_probe(self, worker_id: str) -> None:
        """Verifica saude do worker com backoff exponencial apos spawn.

        Transita de STARTING -> READY quando health retorna ok.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        delay = HEALTH_PROBE_INITIAL_DELAY
        start = time.monotonic()

        while handle.state == WorkerState.STARTING:
            elapsed = time.monotonic() - start
            if elapsed > HEALTH_PROBE_TIMEOUT:
                logger.error("health_probe_timeout", worker_id=worker_id)
                handle.state = WorkerState.CRASHED
                return

            try:
                await asyncio.sleep(delay)
                result = await _check_worker_health(handle.port, timeout=2.0)
                if result.get("status") == "ok":
                    handle.state = WorkerState.READY
                    logger.info("worker_ready", worker_id=worker_id, elapsed_s=round(elapsed, 2))
                    return
            except asyncio.CancelledError:
                return
            except Exception:
                pass

            delay = min(delay * 2, HEALTH_PROBE_MAX_DELAY)

    async def _monitor_worker(self, worker_id: str) -> None:
        """Monitora processo do worker, detecta crashes."""
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        try:
            while handle.state not in (WorkerState.STOPPING, WorkerState.STOPPED):
                await asyncio.sleep(MONITOR_INTERVAL)

                process = handle.process
                if process is None:
                    continue

                exit_code = process.poll()
                if exit_code is not None and handle.state not in (
                    WorkerState.STOPPING,
                    WorkerState.STOPPED,
                ):
                    logger.error(
                        "worker_crashed",
                        worker_id=worker_id,
                        exit_code=exit_code,
                    )
                    handle.state = WorkerState.CRASHED
                    await self._attempt_restart(worker_id)
                    return
        except asyncio.CancelledError:
            return

    async def _attempt_restart(self, worker_id: str) -> None:
        """Tenta reiniciar worker com rate limiting.

        Nao reinicia se exceder MAX_CRASHES_IN_WINDOW dentro de CRASH_WINDOW_SECONDS.
        """
        handle = self._workers.get(worker_id)
        if handle is None:
            return

        now = time.monotonic()
        handle.crash_count += 1
        handle.crash_timestamps.append(now)

        # Limpar timestamps fora da janela
        handle.crash_timestamps = [
            ts for ts in handle.crash_timestamps if now - ts < CRASH_WINDOW_SECONDS
        ]

        if len(handle.crash_timestamps) >= MAX_CRASHES_IN_WINDOW:
            logger.error(
                "worker_max_crashes_exceeded",
                worker_id=worker_id,
                crashes=handle.crash_count,
                window_seconds=CRASH_WINDOW_SECONDS,
            )
            handle.state = WorkerState.CRASHED
            return

        # Backoff baseado no numero de crashes recentes
        backoff = HEALTH_PROBE_INITIAL_DELAY * (2 ** len(handle.crash_timestamps))
        logger.info(
            "worker_restarting",
            worker_id=worker_id,
            backoff_s=backoff,
            crash_count=handle.crash_count,
        )

        await asyncio.sleep(backoff)

        # Re-spawn
        cmd = [
            sys.executable,
            "-m",
            "theo.workers.stt",
            "--port",
            str(handle.port),
            "--engine",
            handle.engine,
            "--model-path",
            handle.model_path,
        ]

        engine_config = handle.engine_config
        if "compute_type" in engine_config:
            cmd.extend(["--compute-type", str(engine_config["compute_type"])])
        if "device" in engine_config:
            cmd.extend(["--device", str(engine_config["device"])])
        if "model_size" in engine_config:
            cmd.extend(["--model-size", str(engine_config["model_size"])])
        if "beam_size" in engine_config:
            cmd.extend(["--beam-size", str(engine_config["beam_size"])])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        handle.process = process
        handle.state = WorkerState.STARTING
        handle.last_started_at = time.monotonic()

        # Restart background tasks
        for task in self._tasks.get(worker_id, []):
            task.cancel()

        health_task = asyncio.create_task(self._health_probe(worker_id))
        monitor_task = asyncio.create_task(self._monitor_worker(worker_id))
        self._tasks[worker_id] = [health_task, monitor_task]


async def _check_worker_health(port: int, timeout: float = 2.0) -> dict[str, str]:
    """Verifica health de um worker via gRPC Health RPC.

    Args:
        port: Porta gRPC do worker.
        timeout: Timeout em segundos.

    Returns:
        Dict com status do worker.
    """
    import grpc.aio

    from theo.proto import HealthRequest

    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        from theo.proto.stt_worker_pb2_grpc import STTWorkerStub

        stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
        response = await asyncio.wait_for(
            stub.Health(HealthRequest()),
            timeout=timeout,
        )
        return {
            "status": response.status,
            "model_name": response.model_name,
            "engine": response.engine,
        }
    finally:
        await channel.close()
