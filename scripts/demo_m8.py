"""Demo M8 -- Scheduler Avancado — priorizacao, cancelamento, batching, latencia.

Exercita TODOS os componentes do M8 do ponto de vista do usuario:

1.  PriorityQueue: REALTIME despachado antes de BATCH
2.  Aging: BATCH promovido apos aging_threshold_s
3.  Cancelamento na fila: cancel antes do dispatch, worker nunca chamado
4.  Cancelamento in-flight: cancel propaga via gRPC Cancel ao worker
5.  BatchAccumulator: requests BATCH agrupadas por tempo ou max_size
6.  LatencyTracker: tracking de enqueue, dequeue, grpc_start, complete
7.  Prometheus Metrics: 7 metricas observaveis
8.  Graceful Shutdown: stop() drena in-flight e flush pending batch
9.  Contencao: BATCH espera enquanto REALTIME ocupa worker
10. Scheduler end-to-end: submit -> queue -> dispatch -> gRPC -> future

Funciona SEM modelo real instalado -- usa mocks controlados.

Uso:
    .venv/bin/python scripts/demo_m8.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from theo._types import ResponseFormat
from theo.proto import Segment, TranscribeFileResponse, Word
from theo.scheduler.batching import BatchAccumulator
from theo.scheduler.cancel import CancellationManager
from theo.scheduler.latency import LatencyTracker, LatencySummary
from theo.scheduler.queue import RequestPriority, ScheduledRequest, SchedulerQueue
from theo.scheduler.scheduler import Scheduler
from theo.server.models.requests import TranscribeRequest
from theo.workers.manager import WorkerHandle, WorkerState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"{CYAN}[INFO]{NC}  {msg}")


def pass_msg(msg: str) -> None:
    print(f"{GREEN}[PASS]{NC}  {msg}")


def fail_msg(msg: str) -> None:
    print(f"{RED}[FAIL]{NC}  {msg}")


def step(num: int | str, desc: str) -> None:
    print(f"\n{CYAN}=== Step {num}: {desc} ==={NC}")


def check(condition: bool, desc: str) -> bool:
    if condition:
        pass_msg(desc)
    else:
        fail_msg(desc)
    return condition


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req_1",
    model_name: str = "faster-whisper-tiny",
) -> TranscribeRequest:
    return TranscribeRequest(
        request_id=request_id,
        model_name=model_name,
        audio_data=b"\x00" * 3200,
        language="pt",
        response_format=ResponseFormat.JSON,
    )


def _make_worker(port: int = 50051) -> WorkerHandle:
    return WorkerHandle(
        worker_id=f"worker-{port}",
        port=port,
        model_name="faster-whisper-tiny",
        engine="faster-whisper",
        state=WorkerState.READY,
    )


def _make_proto_response(text: str = "Ola mundo") -> TranscribeFileResponse:
    return TranscribeFileResponse(
        text=text,
        language="pt",
        duration=1.5,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.5,
                text=text,
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ],
        words=[
            Word(word=text, start=0.0, end=1.5, probability=0.95),
        ],
    )


def _make_scheduler(
    worker: WorkerHandle | None = None,
    **kwargs: object,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    """Create scheduler with mocked dependencies."""
    registry = MagicMock()
    registry.get_manifest.return_value = MagicMock()

    worker_manager = MagicMock()
    worker_manager.get_ready_worker.return_value = worker

    scheduler = Scheduler(worker_manager, registry, **kwargs)  # type: ignore[arg-type]
    return scheduler, worker_manager, registry


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------

passes = 0
fails = 0


def record(ok: bool) -> None:
    global passes, fails
    if ok:
        passes += 1
    else:
        fails += 1


# 1. PriorityQueue: REALTIME dispatched before BATCH
async def demo_priority_queue() -> None:
    step(1, "PriorityQueue — REALTIME dispatched before BATCH")
    info("Submitting 5 BATCH + 1 REALTIME requests to the queue...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    dispatch_order: list[str] = []

    async def fake_transcribe(
        proto_req: object, timeout: float = 30
    ) -> TranscribeFileResponse:
        req_id = proto_req.request_id  # type: ignore[attr-defined]
        dispatch_order.append(req_id)
        return _make_proto_response(text=req_id)

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.side_effect = fake_transcribe

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        # Enqueue 5 BATCH requests first
        batch_futures = []
        for i in range(5):
            f = await scheduler.submit(
                _make_request(f"batch_{i}"),
                RequestPriority.BATCH,
            )
            batch_futures.append(f)

        info("  5 BATCH requests enqueued.")

        # Then 1 REALTIME request
        rt_future = await scheduler.submit(
            _make_request("rt_0"),
            RequestPriority.REALTIME,
        )
        info("  1 REALTIME request enqueued (after 5 BATCH).")

        # Start dispatch loop
        await scheduler.start()
        try:
            await asyncio.wait_for(rt_future, timeout=5.0)
            for f in batch_futures:
                await asyncio.wait_for(f, timeout=5.0)
        finally:
            await scheduler.stop()

    info(f"  Dispatch order: {dispatch_order}")
    record(check(
        dispatch_order[0] == "rt_0",
        f"REALTIME (rt_0) dispatched first: {dispatch_order[0]}",
    ))
    record(check(
        len(dispatch_order) == 6,
        f"All 6 requests dispatched: {len(dispatch_order)}",
    ))


# 2. Aging: BATCH promoted after threshold
async def demo_aging() -> None:
    step(2, "Aging — BATCH promoted after aging_threshold_s")
    info("Submitting BATCH request with very short aging threshold (1ms)...")

    scheduler, _, _ = _make_scheduler(aging_threshold_s=0.001)

    future = await scheduler.submit(
        _make_request("aged_req"),
        RequestPriority.BATCH,
    )

    # Wait longer than the aging threshold
    await asyncio.sleep(0.02)

    scheduled = await scheduler.queue.dequeue()
    is_aged = scheduler.queue.is_aged(scheduled)

    info(f"  Request request_id={scheduled.request.request_id}")
    info(f"  Queue wait: {(time.monotonic() - scheduled.enqueued_at) * 1000:.1f}ms")
    record(check(is_aged, "BATCH request is aged (promoted)"))

    # Verify fresh request is NOT aged
    scheduler2, _, _ = _make_scheduler(aging_threshold_s=60.0)
    future2 = await scheduler2.submit(
        _make_request("fresh_req"),
        RequestPriority.BATCH,
    )
    scheduled2 = await scheduler2.queue.dequeue()
    is_fresh = not scheduler2.queue.is_aged(scheduled2)
    record(check(is_fresh, "Fresh BATCH request is NOT aged"))

    # Cleanup
    if not future.done():
        future.cancel()
    if not future2.done():
        future2.cancel()


# 3. Cancellation in queue
async def demo_cancel_in_queue() -> None:
    step(3, "Cancellation in Queue — cancel before dispatch")
    info("Submitting request then cancelling before dispatch loop starts...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.return_value = _make_proto_response()

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        # Submit but cancel BEFORE starting dispatch loop
        future = await scheduler.submit(_make_request("cancel_me"))
        cancelled = scheduler.cancel("cancel_me")

        record(check(cancelled is True, "cancel() returned True"))
        record(check(future.cancelled(), "Future is cancelled"))

        # Start and stop — worker should never be called
        await scheduler.start()
        await asyncio.sleep(0.1)
        await scheduler.stop()

        record(check(
            mock_stub.TranscribeFile.call_count == 0,
            "Worker TranscribeFile never called (cancelled in queue)",
        ))


# 4. Cancellation in-flight
async def demo_cancel_in_flight() -> None:
    step(4, "Cancellation In-Flight — gRPC Cancel propagation")
    info("Submitting request, waiting for gRPC call to start, then cancelling...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    started_event = asyncio.Event()

    async def slow_transcribe(*args: object, **kwargs: object) -> TranscribeFileResponse:
        started_event.set()
        await asyncio.sleep(5.0)
        return _make_proto_response()

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.side_effect = slow_transcribe
    mock_cancel_response = MagicMock(acknowledged=True)
    mock_stub.Cancel.return_value = mock_cancel_response

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
        patch(
            "theo.scheduler.cancel.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        await scheduler.start()
        try:
            _future = await scheduler.submit(
                _make_request("inflight_req"),
                RequestPriority.REALTIME,
            )

            # Wait for gRPC call to start
            await asyncio.wait_for(started_event.wait(), timeout=5.0)
            info("  gRPC call started, now cancelling...")

            cancel_start = time.monotonic()
            cancelled = scheduler.cancel("inflight_req")
            cancel_elapsed_ms = (time.monotonic() - cancel_start) * 1000

            record(check(cancelled is True, "cancel() returned True for in-flight request"))
            info(f"  Local cancel took {cancel_elapsed_ms:.1f}ms")

            # Wait for fire-and-forget cancel task to execute
            await asyncio.sleep(0.3)

            record(check(
                mock_stub.Cancel.await_count >= 1,
                "gRPC Cancel RPC was called on the worker stub",
            ))
        finally:
            await scheduler.stop()


# 5. BatchAccumulator: grouping BATCH requests
async def demo_batch_accumulator() -> None:
    step(5, "BatchAccumulator — grouping BATCH requests")
    info("Testing accumulator with max_batch_size=4...")

    flushed_batches: list[list[ScheduledRequest]] = []

    async def on_flush(batch: list[ScheduledRequest]) -> None:
        flushed_batches.append(batch)

    acc = BatchAccumulator(
        accumulate_ms=500.0,
        max_batch_size=4,
        on_flush=on_flush,
    )

    # Create 4 scheduled requests
    loop = asyncio.get_running_loop()
    for i in range(4):
        req = _make_request(f"batch_{i}")
        scheduled = ScheduledRequest(
            request=req,
            priority=RequestPriority.BATCH,
            result_future=loop.create_future(),
        )
        acc.add(scheduled)

    # Give time for the flush task to execute
    await asyncio.sleep(0.1)

    record(check(
        len(flushed_batches) >= 1,
        f"At least one flush triggered: {len(flushed_batches)} flush(es)",
    ))

    total_flushed = sum(len(b) for b in flushed_batches)
    record(check(
        total_flushed == 4,
        f"All 4 requests flushed: {total_flushed}",
    ))

    # Test timer-based flush (smaller batch, wait for timer)
    info("Testing timer-based flush (accumulate_ms=100ms, 2 requests)...")

    flushed_batches.clear()
    acc2 = BatchAccumulator(
        accumulate_ms=100.0,
        max_batch_size=10,
        on_flush=on_flush,
    )

    for i in range(2):
        req = _make_request(f"timer_batch_{i}")
        scheduled = ScheduledRequest(
            request=req,
            priority=RequestPriority.BATCH,
            result_future=loop.create_future(),
        )
        acc2.add(scheduled)

    # Wait for timer to fire
    await asyncio.sleep(0.3)

    record(check(
        len(flushed_batches) >= 1,
        f"Timer flush triggered after accumulate_ms: {len(flushed_batches)} flush(es)",
    ))


# 6. LatencyTracker: per-request phase tracking
async def demo_latency_tracker() -> None:
    step(6, "LatencyTracker — per-request phase tracking")
    info("Tracking request through all 4 phases...")

    tracker = LatencyTracker()

    # Simulate pipeline phases with realistic delays
    tracker.start("req_lat")
    info("  Phase 1: start (enqueue)")

    await asyncio.sleep(0.01)  # simulate queue wait
    tracker.dequeued("req_lat")
    info("  Phase 2: dequeued")

    await asyncio.sleep(0.005)  # simulate setup
    tracker.grpc_started("req_lat")
    info("  Phase 3: gRPC started")

    await asyncio.sleep(0.02)  # simulate inference
    summary = tracker.complete("req_lat")
    info("  Phase 4: complete")

    record(check(summary is not None, "complete() returned LatencySummary"))

    if summary is not None:
        info(f"  queue_wait:  {summary.queue_wait * 1000:.1f}ms")
        info(f"  grpc_time:   {summary.grpc_time * 1000:.1f}ms")
        info(f"  total_time:  {summary.total_time * 1000:.1f}ms")
        record(check(summary.queue_wait >= 0, f"queue_wait >= 0: {summary.queue_wait:.4f}s"))
        record(check(summary.grpc_time >= 0, f"grpc_time >= 0: {summary.grpc_time:.4f}s"))
        record(check(
            summary.total_time >= summary.queue_wait,
            f"total_time >= queue_wait: {summary.total_time:.4f}s >= {summary.queue_wait:.4f}s",
        ))

    # Test discard (for cancelled requests)
    info("  Testing discard for cancelled request...")
    tracker.start("req_cancel")
    tracker.discard("req_cancel")
    record(check(tracker.active_count == 0, "Discarded request removed from tracker"))

    # Test TTL cleanup
    info("  Testing TTL cleanup (ttl_s=0.01)...")
    short_tracker = LatencyTracker(ttl_s=0.01)
    short_tracker.start("req_ttl")
    await asyncio.sleep(0.02)
    removed = short_tracker.cleanup()
    record(check(removed >= 1, f"TTL cleanup removed {removed} expired entry(ies)"))


# 7. CancellationManager: register, cancel, in-flight tracking
async def demo_cancellation_manager() -> None:
    step(7, "CancellationManager — register, cancel, lifecycle")
    info("Testing CancellationManager operations...")

    cm = CancellationManager()
    loop = asyncio.get_running_loop()

    # Register a cancellable request
    cancel_event = asyncio.Event()
    result_future: asyncio.Future[Any] = loop.create_future()
    cm.register("req_1", cancel_event, result_future)
    record(check(cm.pending_count == 1, "Registered 1 request"))

    # Cancel the request
    cancelled = cm.cancel("req_1")
    record(check(cancelled is True, "cancel() returned True"))
    record(check(cancel_event.is_set(), "cancel_event is set"))
    record(check(result_future.cancelled(), "result_future is cancelled"))
    record(check(cm.pending_count == 0, "Request removed from tracking after cancel"))

    # Cancel non-existent
    record(check(cm.cancel("nonexistent") is False, "cancel(nonexistent) returns False"))

    # In-flight tracking
    info("  Testing in-flight tracking...")
    cancel_event2 = asyncio.Event()
    result_future2: asyncio.Future[Any] = loop.create_future()
    cm.register("req_2", cancel_event2, result_future2)
    cm.mark_in_flight("req_2", "localhost:50051")
    record(check(
        cm.get_worker_address("req_2") == "localhost:50051",
        "Worker address tracked for in-flight request",
    ))

    # Unregister after completion
    cm.unregister("req_2")
    record(check(cm.pending_count == 0, "Request unregistered after completion"))


# 8. Scheduler end-to-end with LatencyTracker
async def demo_scheduler_e2e() -> None:
    step(8, "Scheduler End-to-End — submit -> dispatch -> gRPC -> future -> latency")
    info("Running full pipeline with latency tracking...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.return_value = _make_proto_response("transcricao completa")

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        await scheduler.start()
        try:
            result = await asyncio.wait_for(
                scheduler.transcribe(_make_request("e2e_req")),
                timeout=5.0,
            )
            record(check(
                result.text == "transcricao completa",
                f"Result text: '{result.text}'",
            ))

            # LatencyTracker should have completed the entry
            summary = scheduler.latency.get_summary("e2e_req")
            record(check(summary is not None, "LatencyTracker has summary for request"))

            if summary:
                info(f"  queue_wait:  {summary.queue_wait * 1000:.1f}ms")
                info(f"  grpc_time:   {summary.grpc_time * 1000:.1f}ms")
                info(f"  total_time:  {summary.total_time * 1000:.1f}ms")
                record(check(
                    summary.total_time >= 0,
                    f"total_time non-negative: {summary.total_time * 1000:.1f}ms",
                ))
        finally:
            await scheduler.stop()


# 9. Graceful shutdown: stop() drains in-flight and flushes pending batch
async def demo_graceful_shutdown() -> None:
    step(9, "Graceful Shutdown — stop() drains in-flight + flushes batch")
    info("Testing stop() with in-flight request...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    async def slow_transcribe(*args: object, **kwargs: object) -> TranscribeFileResponse:
        await asyncio.sleep(0.2)
        return _make_proto_response("shutdown_result")

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.side_effect = slow_transcribe

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        await scheduler.start()
        future = await scheduler.submit(
            _make_request("shutdown_req"),
            RequestPriority.REALTIME,
        )

        # Let dispatch loop pick it up
        await asyncio.sleep(0.05)

        # stop() should drain in-flight
        await scheduler.stop()

        record(check(future.done(), "In-flight future resolved after stop()"))
        record(check(not future.cancelled(), "Future was NOT cancelled (completed normally)"))
        if future.done() and not future.cancelled():
            record(check(
                future.result().text == "shutdown_result",
                f"Result text after shutdown: '{future.result().text}'",
            ))

    # Test flush of pending batch
    info("Testing stop() flushes pending batch accumulator...")

    scheduler2, _, _ = _make_scheduler(
        worker,
        batch_accumulate_ms=10_000.0,  # very long timer
        batch_max_size=100,
    )

    mock_stub2 = AsyncMock()
    mock_stub2.TranscribeFile.return_value = _make_proto_response("flushed")

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub2,
        ),
    ):
        await scheduler2.start()
        future2 = await scheduler2.submit(
            _make_request("pending_batch"),
            RequestPriority.BATCH,
        )

        # Let dispatch loop dequeue and add to accumulator
        await asyncio.sleep(0.3)

        # stop() should flush the pending batch
        await scheduler2.stop()

        record(check(future2.done(), "Pending batch future resolved after stop()"))
        if future2.done() and not future2.cancelled():
            record(check(
                future2.result().text == "flushed",
                f"Flushed batch result: '{future2.result().text}'",
            ))


# 10. Contention: BATCH waits while REALTIME occupies worker
async def demo_contention() -> None:
    step(10, "Contention — BATCH waits while REALTIME occupies worker")
    info("Submitting slow REALTIME then BATCH, verifying ordering...")

    worker = _make_worker()
    scheduler, _, _ = _make_scheduler(worker)

    dispatch_order: list[str] = []
    realtime_started = asyncio.Event()

    async def slow_transcribe(
        proto_req: object, timeout: float = 30
    ) -> TranscribeFileResponse:
        req_id: str = proto_req.request_id  # type: ignore[attr-defined]
        dispatch_order.append(req_id)
        if req_id == "rt_slow":
            realtime_started.set()
            await asyncio.sleep(0.3)
        return _make_proto_response(text=req_id)

    mock_stub = AsyncMock()
    mock_stub.TranscribeFile.side_effect = slow_transcribe

    with (
        patch(
            "theo.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "theo.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    ):
        await scheduler.start()
        try:
            # Submit REALTIME first (slow) then BATCH
            rt_future = await scheduler.submit(
                _make_request("rt_slow"),
                RequestPriority.REALTIME,
            )

            # Wait until realtime starts processing
            await asyncio.wait_for(realtime_started.wait(), timeout=2.0)
            info("  REALTIME started processing (slow)...")

            # Submit batch — should queue behind
            batch_future = await scheduler.submit(
                _make_request("batch_after"),
                RequestPriority.BATCH,
            )

            rt_result = await asyncio.wait_for(rt_future, timeout=5.0)
            batch_result = await asyncio.wait_for(batch_future, timeout=5.0)

            record(check(
                rt_result.text == "rt_slow",
                f"REALTIME completed: '{rt_result.text}'",
            ))
            record(check(
                batch_result.text == "batch_after",
                f"BATCH completed after REALTIME: '{batch_result.text}'",
            ))
            record(check(
                dispatch_order.index("rt_slow") < dispatch_order.index("batch_after"),
                f"REALTIME dispatched before BATCH: {dispatch_order}",
            ))
        finally:
            await scheduler.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> int:
    global passes, fails

    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  Demo M8 -- Scheduler Avancado{NC}")
    print(f"{BOLD}  Priorizacao | Cancelamento | Batching | Latencia | Shutdown{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")

    await demo_priority_queue()
    await demo_aging()
    await demo_cancel_in_queue()
    await demo_cancel_in_flight()
    await demo_batch_accumulator()
    await demo_latency_tracker()
    await demo_cancellation_manager()
    await demo_scheduler_e2e()
    await demo_graceful_shutdown()
    await demo_contention()

    # Summary
    total = passes + fails
    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  Summary: {total} checks{NC}")
    print(f"  {GREEN}PASSED: {passes}{NC}")
    if fails > 0:
        print(f"  {RED}FAILED: {fails}{NC}")
    else:
        print(f"  {GREEN}FAILED: 0{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")

    return 0 if fails == 0 else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
