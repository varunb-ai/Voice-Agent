"""Sequential mockup runner — one dataset in, one JSONL of outcomes out.

WHAT THIS IS FOR. The prospective-patient prompt is 5,821 static tokens. Every
call that ever RESOLVED on this system ran at <=4,641, and the prompt ceiling in
test_realtime_protocol.py records why that is evidence rather than proof: the
twelve-day regression changed several things at once, so it says no long prompt
has worked, not that a long one cannot. This runner exists to settle that with
measurements instead of argument — extraction quality, latency and cost, per
call, written down.

WHY IT IS SEQUENTIAL, AND NOT CONFIGURABLY OTHERWISE.
run_twilio.py places one call and then blocks in uvicorn forever, so "13 calls"
has always meant 13 process invocations. This starts the server ONCE and places
calls through it one at a time. That is not a throttling workaround: two calls
in flight would race the read-modify-write of master.json (session.save()
documents the lost update), so concurrency here would silently drop call
records. One at a time is the correctness requirement, and the pacing delay
rides on top of it.

    python run_mockups.py --dataset lini.csv --delay 90
    python run_mockups.py --dataset lini.csv --dry-run     # parse only, no calls

THE BRANCH COLUMN IS GROUND TRUTH, NOT INPUT. If the dataset carries a known
branch it is recorded as `expected_branch` and compared against what the call
extracted. It is never put on the Doctor: the branch is the thing the call is
placed to discover, and seeding it would contaminate both the prompt's context
item and the doctors.json row the call writes.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import core.bootstrap  # noqa: F401

from core.config import settings
from core.models import Doctor
import agents.voice.twilio_worker as worker
from agents.voice.session import json_dir
from agents.voice.templates import get_template
from run_twilio import _check_ngrok_url_is_current, _place_call, _warmup


# ── Dataset ──────────────────────────────────────────────────────────────────
# ALIASES, NOT A FIXED SCHEMA. The dataset is not in the repo and its exact
# headers are not known here, so the mapping is by alias with an explicit
# --map override and a --dry-run that prints what it resolved. The alternative
# — guessing one spelling and failing at call time — spends a phone call to
# discover a typo.
_ALIASES: dict[str, tuple[str, ...]] = {
    "doctor":   ("doctor", "doctor_name", "name", "physician", "physician_name",
                 "provider", "provider_name", "dr", "full_name"),
    "hospital": ("hospital", "hospital_name", "practice", "practice_name",
                 "facility", "facility_name", "clinic", "clinic_name",
                 "organisation", "organization", "org"),
    "phone":    ("phone", "to", "number", "phone_number", "to_number",
                 "contact", "tel", "telephone", "telephone_number",
                 "contact_number", "main_number"),
    "specialty": ("specialty", "specialization", "speciality",
                  "specialisation", "department", "field"),
    "expected_branch": ("branch", "expected_branch", "branch_name", "location",
                        "site", "campus", "known_branch"),
}

# Resolution order. Fixed, because a column is claimed by the first field that
# wants it and "first" must not depend on dict iteration luck.
_FIELD_ORDER = ("doctor", "hospital", "phone", "specialty", "expected_branch")

# Aliases too generic to match on a TOKEN of a longer header. "name" is a
# doctor alias, and matching it token-wise would bind `doctor` to a column
# called "Hospital Name" — the exact silent mis-mapping that would then dial
# the right number and ask about the wrong person. These stay exact-match only.
_WEAK = frozenset({"name", "to", "dr", "number", "contact", "field", "org",
                   "site", "location", "branch", "phone", "clinic"})

# Twilio wants E.164. A number that is merely digits is ambiguous about country
# and is rejected here rather than at call time, where the failure costs a
# round trip and reads like an account problem.
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")


@dataclass
class Row:
    index: int
    doctor: str
    hospital: str
    phone: str
    specialty: Optional[str] = None
    expected_branch: Optional[str] = None
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems


def _load_records(path: Path) -> list[dict]:
    """CSV, JSON array, or JSONL — decided by content, not only by suffix."""
    text = io.open(path, encoding="utf-8-sig").read().strip()
    if not text:
        raise SystemExit(f"  dataset is empty: {path}")
    if path.suffix.lower() in (".json", ".jsonl") or text[:1] in "[{":
        if text[:1] == "[":
            rows = json.loads(text)
        else:
            rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if not isinstance(rows, list):
            raise SystemExit(f"  expected a list of records in {path}")
        return [r for r in rows if isinstance(r, dict)]
    return list(csv.DictReader(io.StringIO(text)))


def _resolve_columns(records: list[dict],
                     overrides: dict[str, str]) -> dict[str, str]:
    """{canonical field: actual column}. Overrides win; aliases fill the rest."""
    present = {_norm_header(k): k for k in (records[0].keys() if records else [])}
    chosen: dict[str, str] = {}
    claimed: set[str] = set()

    # Overrides first, and they claim their column before anything else can.
    for canon in _FIELD_ORDER:
        if canon not in overrides:
            continue
        want = _norm_header(overrides[canon])
        if want not in present:
            raise SystemExit(
                f"  --map {canon}={overrides[canon]!r}: no such column. "
                f"Columns are {sorted(present.values())}")
        chosen[canon], _ = present[want], claimed.add(want)

    # Exact alias match.
    for canon in _FIELD_ORDER:
        if canon in chosen:
            continue
        for a in _ALIASES[canon]:
            if a in present and a not in claimed:
                chosen[canon] = present[a]
                claimed.add(a)
                break

    # Token match, strong aliases only — "Provider Name" -> doctor. Runs after
    # every exact match, so a header that matches one field exactly can never
    # be stolen by another field matching it loosely.
    for canon in _FIELD_ORDER:
        if canon in chosen:
            continue
        strong = [a for a in _ALIASES[canon] if a not in _WEAK]
        for norm, original in present.items():
            if norm in claimed:
                continue
            tokens = set(norm.split("_"))
            if any(a in tokens or a in norm for a in strong):
                chosen[canon] = original
                claimed.add(norm)
                break
    return chosen


def load_dataset(path: Path, overrides: dict[str, str]) -> tuple[list[Row], dict]:
    records = _load_records(path)
    if not records:
        raise SystemExit(f"  no records in {path}")
    cols = _resolve_columns(records, overrides)
    for req in ("doctor", "phone"):
        if req not in cols:
            raise SystemExit(
                f"  could not find a {req!r} column in {path}.\n"
                f"  Columns present : {sorted(records[0].keys())}\n"
                f"  Tried aliases   : {', '.join(_ALIASES[req])}\n"
                f"  Override it with --map {req}=<column>")

    rows: list[Row] = []
    for i, rec in enumerate(records, 1):
        def get(k: str) -> str:
            col = cols.get(k)
            return str(rec.get(col, "") or "").strip() if col else ""

        r = Row(index=i, doctor=get("doctor"), hospital=get("hospital"),
                phone=get("phone"), specialty=get("specialty") or None,
                expected_branch=get("expected_branch") or None)
        if not r.doctor:
            r.problems.append("no doctor name")
        if not r.phone:
            r.problems.append("no phone number")
        elif not _E164.match(r.phone):
            # Not silently reformatted. Inferring a country code is a guess,
            # and a guess here dials a real stranger.
            r.problems.append(
                f"phone {r.phone!r} is not E.164 (must look like +14155550123)")
        if not r.hospital:
            # Not fatal: build_context prints "unknown" for it. But the opener
            # and the identity question both lean on it, so it is worth saying.
            r.problems.append("no hospital name")
        rows.append(r)
    return rows, cols


# ── One call ─────────────────────────────────────────────────────────────────

def _wait_for_answer(sid: str, timeout: float) -> Optional[str]:
    """The call_id, once Twilio has fetched /answer. None if it never rang through.

    `_call_id_by_sid` is filled in by the /answer handler, so its appearance IS
    the callee picking up — there is no separate signal to poll and no need to
    ask Twilio.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cid = worker._call_id_by_sid.get(sid)
        if cid:
            return cid
        time.sleep(0.5)
    return None


def _wait_for_artifact(call_id: str, timeout: float) -> Optional[Path]:
    """The call's JSON, written by session.save() in handle_realtime's finally."""
    path = json_dir() / f"{call_id}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            # The writer is write_text, not an atomic rename, so a zero-length
            # or truncated read is possible for an instant. Parse before
            # believing it.
            try:
                json.loads(path.read_text(encoding="utf-8"))
                return path
            except Exception:
                pass
        time.sleep(1.0)
    return None


def _branch_match(got: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Did the call find the branch the dataset says is right?

    None when the dataset offers no expectation — which is NOT the same as a
    miss, and conflating them would report every unlabelled row as a failure.
    Loose comparison: case, punctuation and the generic nouns a front desk
    adds ("the Midtown office" vs "Midtown") are not differences worth failing.
    """
    if not expected:
        return None
    if not got:
        return False

    def norm(s: str) -> set:
        words = re.findall(r"[a-z0-9]+", s.lower())
        return {w for w in words
                if w not in {"the", "a", "an", "of", "at", "office", "branch",
                             "clinic", "center", "centre", "campus", "site",
                             "location", "hospital", "medical", "health"}}
    g, e = norm(got), norm(expected)
    return bool(g and e and (g & e))


def summarise(artifact: dict, row: Row) -> dict:
    """The fields worth comparing across a batch, pulled out of the record."""
    notes = artifact.get("notes") or {}
    branch = artifact.get("branch")
    usage = artifact.get("usage") or {}
    return {
        "resolved":     artifact.get("resolved"),
        "outcome":      artifact.get("outcome"),
        "collected":    artifact.get("collected"),
        "missing":      artifact.get("missing"),
        # The telemetry label this script exists to produce. note_info writes
        # note_<key>; session.notes() strips the prefix back off.
        "call_outcome": notes.get("call_outcome"),
        "waitlist":     notes.get("waitlist"),
        "new_hospital": notes.get("new_hospital"),
        "branch":       branch,
        "expected_branch": row.expected_branch,
        "branch_matches_expected": _branch_match(branch, row.expected_branch),
        "fields":       artifact.get("fields"),
        "grounding":    artifact.get("grounding"),
        "branch_rejections": artifact.get("branch_rejections"),
        "save_refusals": artifact.get("save_refusals"),
        "duration_seconds": artifact.get("duration_seconds"),
        "cost_usd":     artifact.get("cost_usd"),
        "template":     artifact.get("template"),
        "usage":        usage,
    }


def run_one(row: Row, *, answer_timeout: float, call_timeout: float) -> dict:
    """Place one call and wait it out. Always returns a record, never raises."""
    doctor = Doctor(doctor_name=row.doctor,
                    hospital_name=row.hospital or None,
                    specialization=row.specialty)
    out: dict[str, Any] = {
        "row":        row.index,
        "placed_at":  datetime.now(timezone.utc).isoformat(),
        "doctor":     row.doctor,
        "hospital":   row.hospital,
        "to":         row.phone,
        "specialty":  row.specialty,
    }
    try:
        sid = _place_call(row.phone, doctor)
    except SystemExit as e:
        # _place_call turns a Twilio error into an explanation and exits. In a
        # batch that must not take the other twelve rows down with it.
        out |= {"status": "place_failed", "error": str(e)}
        return out
    except Exception as e:                     # noqa: BLE001
        out |= {"status": "place_failed", "error": f"{type(e).__name__}: {e}"}
        return out

    out["call_sid"] = sid
    call_id = _wait_for_answer(sid, answer_timeout)
    if not call_id:
        out |= {"status": "no_answer"}
        return out
    out["call_id"] = call_id

    path = _wait_for_artifact(call_id, call_timeout)
    if path is None:
        # The call connected and then outran the timeout, or ended in a way
        # that never reached session.save(). Recorded as its own state: a row
        # with no artifact is not a row that failed to extract anything, and
        # merging the two would flatter or damn the prompt for the wrong
        # reason.
        out |= {"status": "timeout", "artifact": None}
        return out

    artifact = json.loads(path.read_text(encoding="utf-8"))
    out |= {"status": "completed", "artifact": str(path)}
    out |= summarise(artifact, row)
    return out


# ── Batch ────────────────────────────────────────────────────────────────────

def _serve(port: int) -> None:
    import uvicorn
    config = uvicorn.Config(worker.app, host="0.0.0.0", port=port,
                            log_level="warning")
    uvicorn.Server(config).run()


def _sleep_with_abort(seconds: float) -> None:
    """The manual abort window. Ctrl-C here stops the batch, not the process.

    Slept in short slices so the interrupt lands promptly instead of at the end
    of a 90-second block.
    """
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(0.5, max(0.0, end - time.monotonic())))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Place one call per dataset row, sequentially, and record "
                    "what each one extracted.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--delay", type=float, default=60.0,
                    help="Seconds between calls. Also the window to Ctrl-C "
                         "the batch if the first results look wrong. "
                         "Default 60.")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL destination. Default data/mockups/<stamp>.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only the first N usable rows. 0 = all.")
    ap.add_argument("--start", type=int, default=1,
                    help="Resume from this 1-based row.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse, map and validate the dataset. No server, no "
                         "calls, no cost.")
    ap.add_argument("--map", action="append", default=[], metavar="FIELD=COLUMN",
                    help="Override a column mapping, e.g. --map phone=Contact")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--answer-timeout", type=float, default=90.0,
                    help="Seconds to wait for the callee to pick up.")
    ap.add_argument("--call-timeout", type=float, default=420.0,
                    help="Seconds to wait for a connected call to finish.")
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for m in args.map:
        if "=" not in m:
            raise SystemExit(f"  --map wants FIELD=COLUMN, got {m!r}")
        k, v = m.split("=", 1)
        if k.strip() not in _ALIASES:
            raise SystemExit(f"  --map: unknown field {k!r}. "
                             f"Known: {', '.join(sorted(_ALIASES))}")
        overrides[k.strip()] = v.strip()

    rows, cols = load_dataset(args.dataset, overrides)

    template = get_template(settings.call_template)
    print(f"\n  Dataset  : {args.dataset}  ({len(rows)} rows)")
    print(f"  Mapping  : " + ", ".join(f"{k} <- {v!r}" for k, v in cols.items()))
    unmapped = [k for k in _ALIASES if k not in cols]
    if unmapped:
        print(f"  Unmapped : {', '.join(unmapped)}")
    print(f"  Template : {template.name}  "
          f"({len(template.instructions):,} chars static prompt)")
    for w in template.config_warnings(agent_language=settings.agent_language):
        print(f"  ⚠  {w}")

    bad = [r for r in rows if not r.usable]
    if bad:
        print(f"\n  {len(bad)} row(s) will be SKIPPED:")
        for r in bad:
            print(f"    row {r.index}: {r.doctor or '(no name)'} — "
                  f"{'; '.join(r.problems)}")

    todo = [r for r in rows if r.usable and r.index >= args.start]
    if args.limit:
        todo = todo[:args.limit]
    print(f"\n  Will call: {len(todo)} row(s), {args.delay:g}s apart "
          f"(~{(len(todo) * (args.delay + 90)) / 60:.0f} min including calls)")

    if args.dry_run:
        print("\n  DRY RUN — nothing placed. Resolved rows:\n")
        for r in todo:
            print(f"    {r.index:>3}. {r.doctor}  |  {r.hospital or '(none)'}  "
                  f"|  {r.phone}  |  {r.specialty or '(no specialty)'}"
                  + (f"  |  expect branch: {r.expected_branch}"
                     if r.expected_branch else ""))
        print()
        return

    if not todo:
        raise SystemExit("  nothing to call")

    out_path = args.out or (Path("data") / "mockups" /
                            f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _check_ngrok_url_is_current()
    _warmup()

    threading.Thread(target=_serve, args=(args.port,), daemon=True).start()
    time.sleep(2.0)          # same bind grace run_twilio.py allows
    print(f"  Server   : listening on :{args.port}")
    print(f"  Writing  : {out_path}\n")

    results: list[dict] = []
    try:
        for n, row in enumerate(todo, 1):
            print("═" * 60)
            print(f"  [{n}/{len(todo)}]  row {row.index}  {row.doctor}"
                  f"  —  {row.hospital}")
            print("═" * 60)
            rec = run_one(row, answer_timeout=args.answer_timeout,
                          call_timeout=args.call_timeout)
            results.append(rec)
            # Appended per call, not at the end. A batch interrupted at row 9
            # keeps the first eight; buffering would throw away the calls that
            # were actually placed and paid for.
            with io.open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

            if rec["status"] == "completed":
                print(f"\n  -> {rec['outcome']}  resolved={rec['resolved']}  "
                      f"call_outcome={rec.get('call_outcome')}  "
                      f"branch={rec.get('branch')!r}  "
                      f"{rec.get('duration_seconds')}s  "
                      f"${rec.get('cost_usd')}")
            else:
                print(f"\n  -> {rec['status']}"
                      + (f": {rec['error']}" if rec.get("error") else ""))

            if n < len(todo):
                print(f"\n  pacing {args.delay:g}s — Ctrl-C now to stop the "
                      f"batch\n")
                _sleep_with_abort(args.delay)
    except KeyboardInterrupt:
        print("\n\n  interrupted — stopping after "
              f"{len(results)} call(s). Results kept in {out_path}\n")

    # ── Summary ──────────────────────────────────────────────────────────
    done = [r for r in results if r["status"] == "completed"]
    print("\n" + "═" * 60)
    print(f"  BATCH DONE — {len(done)}/{len(results)} calls produced an artifact")
    print("─" * 60)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    for k, v in sorted(by_status.items()):
        print(f"    {k:<14} {v}")
    if done:
        by_outcome: dict[str, int] = {}
        by_label: dict[str, int] = {}
        for r in done:
            by_outcome[str(r["outcome"])] = by_outcome.get(str(r["outcome"]), 0) + 1
            lab = str(r.get("call_outcome"))
            by_label[lab] = by_label.get(lab, 0) + 1
        print("─" * 60)
        print(f"    resolved       {sum(1 for r in done if r['resolved'])}/{len(done)}")
        for k, v in sorted(by_outcome.items()):
            print(f"    outcome={k:<7} {v}")
        for k, v in sorted(by_label.items()):
            print(f"    call_outcome={k:<12} {v}")
        checked = [r for r in done if r["branch_matches_expected"] is not None]
        if checked:
            hit = sum(1 for r in checked if r["branch_matches_expected"])
            print(f"    branch matches expected  {hit}/{len(checked)}")
        durs = [r["duration_seconds"] for r in done
                if isinstance(r.get("duration_seconds"), (int, float))]
        if durs:
            print(f"    duration  mean {sum(durs)/len(durs):.0f}s  "
                  f"min {min(durs):.0f}s  max {max(durs):.0f}s")
        costs = [r["cost_usd"] for r in done
                 if isinstance(r.get("cost_usd"), (int, float))]
        if costs:
            print(f"    cost      ${sum(costs):.2f} total, "
                  f"${sum(costs)/len(costs):.3f} mean")
    print("═" * 60)
    print(f"  {out_path}\n")


if __name__ == "__main__":
    main()
