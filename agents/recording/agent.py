"""Recording agent — summarizes and persists a completed call."""
from __future__ import annotations

from core.models import CallRecord
from agents.recording.summarize import summarize
from agents.recording.store import _to_record, save


def record_call(
    snap: dict,
    call_id: str,
    audio_path: str | None = None,
    duration_seconds: int = 0,
    use_llm: bool = True,
    persist: bool = True,
    cost_usd: float | None = None,
) -> tuple[CallRecord, str]:
    """Summarize + persist a call. Returns (CallRecord, backend_name)."""
    summary = summarize(snap, use_llm=use_llm)
    record = _to_record(snap, call_id, audio_path, duration_seconds, summary, cost_usd=cost_usd)
    backend = save(record) if persist else "not saved"
    return record, backend
