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
    """Write the call to disk. Returns the path of the per-call JSON.

    THERE WAS A POSTGRES BRANCH HERE AND IT NEVER RAN, not once. It opened with
    `from core.db import get_connection`, and core.db exposed get_backend() --
    never get_connection -- so the import raised ImportError on every call and
    `except Exception` fell through to JSON every time. The module it imported
    from was itself unreachable: nothing in the repo called get_backend() or
    slug(), and PostgresBackend loaded a schema from agents/database/schema.sql,
    a path that does not exist.

    A previous pass left it standing on the grounds that rewiring it to
    get_backend() would start writing to Postgres for the first time, which is a
    data decision rather than a fix. That reasoning still holds -- and it is an
    argument for not REWIRING it, not for keeping fifty lines of CREATE TABLE
    and INSERT that have never touched a database. Deleted with core/db.py.
    Bringing Postgres back is a deliberate piece of work, and it should start
    from a schema that exists.

    No behaviour changes: every call already took the JSON path. The summary
    line said "Returns backend name" and never did -- _save_json returns the
    per-call file path, which is what the one caller has always received.
    """
    return _save_json(record)


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
