"""Prometheus metrics for the pipeline (the design's Monitoring layer).

Each agent increments counters / observes histograms here; a Prometheus server
scrapes the HTTP endpoint exposed by start_metrics_server(), and Grafana charts it.

Import is side-effect-free: if prometheus_client isn't installed, no-op stand-ins
keep the agents running (same graceful-degradation rule as Redis/Ollama).
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    # type: ignore on the REAL import, not the stubs: pyright takes the _Noop
    # factories below as the declared type for these names and then reports the
    # genuine classes as incompatible with them. The stubs are the fallback, so
    # the real ones are what the annotation should follow.
    from prometheus_client import Counter, Histogram, Gauge, start_http_server  # type: ignore
    _ENABLED = True
except Exception:  # prometheus_client missing
    _ENABLED = False

    class _Noop:
        def __init__(self, *a, **k): ...
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): ...
        def observe(self, *a, **k): ...
        def set(self, *a, **k): ...

    def Counter(*a, **k): return _Noop()      # type: ignore
    def Histogram(*a, **k): return _Noop()    # type: ignore
    def Gauge(*a, **k): return _Noop()        # type: ignore
    def start_http_server(*a, **k): ...       # type: ignore


# ── Metric definitions ───────────────────────────────────────────────────────

DOCTORS_PROCESSED = Counter(
    "doctors_processed_total", "Doctors run through the pipeline", ["final_status"]
)
EMAILS_SENT = Counter("emails_sent_total", "Verification emails sent")
CALLS_MADE = Counter("calls_made_total", "Voice calls placed", ["resolved"])
CALLS_FAILED = Counter("calls_failed_total", "Voice calls that errored")
LLM_FALLBACKS = Counter(
    "llm_fallbacks_total", "Times Qwen was used instead of heuristics", ["agent"]
)

PIPELINE_DURATION = Histogram(
    "pipeline_duration_seconds", "Per-doctor pipeline wall time",
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
)
CALL_DURATION = Histogram(
    "call_duration_seconds", "Voice call duration",
    buckets=(10, 30, 60, 120, 180, 300),
)
CONFIDENCE = Histogram(
    "doctor_confidence_percent", "Final confidence scores",
    buckets=(0, 25, 50, 75, 90, 100),
)

QUEUE_DEPTH = Gauge("queue_depth", "Doctors waiting in the Celery queue")


@contextmanager
def timed(histogram):
    """`with timed(PIPELINE_DURATION): ...` to record elapsed seconds."""
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)


def start_metrics_server(port: int = 8000) -> bool:
    """Expose /metrics for Prometheus to scrape. Returns False if disabled."""
    if not _ENABLED:
        return False
    start_http_server(port)
    return True


def enabled() -> bool:
    return _ENABLED
