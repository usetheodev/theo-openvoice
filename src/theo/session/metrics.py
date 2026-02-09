"""Metricas Prometheus para streaming STT.

Metricas sao opcionais: se prometheus_client nao estiver instalado,
o modulo exporta None para cada metrica e o codigo consumidor deve
verificar antes de usar.

Metricas definidas:
- theo_stt_ttfb_seconds: Time to First Byte (primeiro partial/final apos speech start)
- theo_stt_final_delay_seconds: Delay do final transcript apos fim de fala
- theo_stt_active_sessions: Gauge de sessoes WebSocket ativas
- theo_stt_vad_events_total: Counter de eventos VAD por tipo (speech_start, speech_end)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram

    stt_ttfb_seconds: Histogram | None = _Histogram(
        "theo_stt_ttfb_seconds",
        "Time to first byte (first partial transcript after speech start)",
        buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    )

    stt_final_delay_seconds: Histogram | None = _Histogram(
        "theo_stt_final_delay_seconds",
        "Delay of final transcript after end of speech",
        buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    )

    stt_active_sessions: Gauge | None = _Gauge(
        "theo_stt_active_sessions",
        "Number of active WebSocket streaming sessions",
    )

    stt_vad_events_total: Counter | None = _Counter(
        "theo_stt_vad_events_total",
        "Total VAD events by type",
        ["event_type"],
    )

    HAS_METRICS = True

except ImportError:
    stt_ttfb_seconds = None
    stt_final_delay_seconds = None
    stt_active_sessions = None
    stt_vad_events_total = None

    HAS_METRICS = False
