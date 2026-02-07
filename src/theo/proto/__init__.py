"""Protobuf gRPC definitions para comunicacao runtime <-> worker."""

from theo.proto.stt_worker_pb2 import (
    AudioFrame,
    CancelRequest,
    CancelResponse,
    HealthRequest,
    HealthResponse,
    Segment,
    TranscribeFileRequest,
    TranscribeFileResponse,
    TranscriptEvent,
    Word,
)
from theo.proto.stt_worker_pb2_grpc import (
    STTWorkerServicer,
    STTWorkerStub,
    add_STTWorkerServicer_to_server,
)

__all__ = [
    "AudioFrame",
    "CancelRequest",
    "CancelResponse",
    "HealthRequest",
    "HealthResponse",
    "STTWorkerServicer",
    "STTWorkerStub",
    "Segment",
    "TranscribeFileRequest",
    "TranscribeFileResponse",
    "TranscriptEvent",
    "Word",
    "add_STTWorkerServicer_to_server",
]
