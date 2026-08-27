"""Persist a CallRecord to PostgreSQL (primary) or JSON file (fallback)."""
from __future__ import annotations

import json
from pathlib import Path

from core.models import CallRecord, TranscriptTurn


def _to_record(snap: dict, call_id: str, audio_path: str | None,
               duration_seconds: int, summary: str,
               cost_usd: float | None = None) -> CallRecord:
    turns = [
        TranscriptTurn(**t) if isinstance(t, dict) else t
        for t in snap.get("transcript", [])
    ]
    return CallRecord(
        call_id=call_id,
        doctor_name=snap.get("doctor", ""),
        hospital_name=snap.get("hospital"),
        branch=snap.get("branch"),
        # A SECOND DEFINITION of success lived here — "a branch was recorded" —
        # independent of the one save_branch asserted and of anything a template
        # declares. It agreed with the old behaviour by coincidence and would
        # have disagreed with the first template that collects two fields.
        # The verdict is now derived once, by the call's objective, and written
        # to memory after every tool call; this reads it. The branch fallback is
        # for a snapshot that predates that write, not a rule of its own.
        resolved=bool(snap.get("resolved", bool(snap.get("branch")))),
        duration_seconds=duration_seconds,
        cost_usd=cost_usd,
        transcript=turns,
        summary=summary,
        audio_path=audio_path,
        # NOT passed explicitly. CallRecord's own default_factory already uses
        # datetime.now(timezone.utc) — this used to override it with
        # datetime.utcnow(), which returns a NAIVE datetime and silently
        # un-fixes the exact bug models.py's comment says it was fixing, for
        # every record that goes through here.
    )


def save(record: CallRecord) -> str:
    """Try PostgreSQL first, fall back to JSON. Returns backend name."""
    try:
        return _save_postgres(record)
    except Exception:
        return _save_json(record)


def _save_postgres(record: CallRecord) -> str:
    # NOT A TYPING PROBLEM. core.db exposes get_backend(), never get_connection,
    # so this import raises ImportError on EVERY call and save() has always
    # fallen through to _save_json. Postgres persistence from this path has
    # never run.
    #
    # Left as-is on purpose: get_backend() is a different API (this function
    # wants a raw DB-API connection for its own CREATE TABLE and %s-parameterised
    # INSERT), so rewiring it would start writing to Postgres for the first
    # time. That is a data decision, not a fix for a type error.
    from core.db import get_connection  # type: ignore[attr-defined]
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            call_id TEXT PRIMARY KEY,
            doctor_name TEXT,
            hospital_name TEXT,
            branch TEXT,
            resolved BOOLEAN,
            duration_seconds INTEGER,
            cost_usd REAL,
            summary TEXT,
            audio_path TEXT,
            transcript JSONB,
            recorded_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Add cost_usd column if it doesn't exist yet (for existing tables)
    cur.execute("""
        ALTER TABLE calls ADD COLUMN IF NOT EXISTS cost_usd REAL
    """)
    cur.execute("""
        INSERT INTO calls
            (call_id, doctor_name, hospital_name, branch, resolved,
             duration_seconds, cost_usd, summary, audio_path, transcript, recorded_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (call_id) DO UPDATE SET
            branch=EXCLUDED.branch, resolved=EXCLUDED.resolved,
            cost_usd=EXCLUDED.cost_usd,
            summary=EXCLUDED.summary, audio_path=EXCLUDED.audio_path,
            transcript=EXCLUDED.transcript
    """, (
        record.call_id, record.doctor_name, record.hospital_name,
        record.branch, record.resolved, record.duration_seconds,
        record.cost_usd,
        record.summary, record.audio_path,
        json.dumps([t.model_dump() for t in record.transcript]),
        record.recorded_at,
    ))
    conn.commit()
    cur.close()
    conn.close()
    return "PostgreSQL"


def _save_json(record: CallRecord) -> str:
    folder = Path("data") / "3 cases jsons"
    folder.mkdir(parents=True, exist_ok=True)

    # ── Per-call JSON ─────────────────────────────────────────────────────────
    data = record.model_dump()
    data["recorded_at"] = data["recorded_at"].isoformat() if data.get("recorded_at") else None
    per_call = folder / f"{record.call_id}.json"
    per_call.write_text(json.dumps(data, indent=2, default=str))

    # ── Master JSON (one entry per call, append-only) ─────────────────────────
    master_path = folder / "master.json"
    master = json.loads(master_path.read_text()) if master_path.exists() else []
    master.append({
        "call_id":          record.call_id,
        "time":             data["recorded_at"],
        "doctor":           record.doctor_name,
        "hospital":         record.hospital_name,
        "branch":           record.branch,
        "resolved":         record.resolved,
        "duration_seconds": record.duration_seconds,
        "cost_usd":         record.cost_usd,
        "summary":          record.summary,
        "audio_path":       record.audio_path,
        "json_path":        str(per_call),
    })
    master_path.write_text(json.dumps(master, indent=2, default=str))

    return str(per_call)
