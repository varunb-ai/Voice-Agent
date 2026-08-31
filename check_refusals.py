"""Find the phrasings the guards could not read, from calls that already happened.

WHY THIS EXISTS. Seven probe gaps have been found on this project, and every
one of them the same way: a person read a console log and noticed a correct
answer being refused. "Right now no.", "It's depend upon situation.", "is this
THE OFFICE FOR Dr. Jennifer", "taking THE new patients" — each cost a re-ask or
a lost field, and each sat in an artifact nobody was scanning.

The alternative that was considered and rejected was classifying intent with a
second LLM on the live path. The arithmetic killed it: 5 fields x 2 evaluations
x 6 caller turns is 60 classifications on a median 81-second call, which is 44
requests per minute from ONE call against a 10 RPM ceiling — and it would put a
network round trip back on a reply path measured at 1.52s median. The deeper
objection was that a language model checking a language model's reading of the
same words is not a guardrail; the guards are worth having precisely because
they cannot be talked into agreeing.

So the judging stays mechanical and the FINDING is automated instead. This runs
offline, over data the system already writes, and costs the live call nothing.

THE SIGNATURE. A guard refusing is not evidence of anything by itself — most
refusals are the guard working. What indicts a refusal is what the call went on
to do:

  COST     the field was refused and never landed. The guard did not delay the
           answer, it destroyed it.
  PREMATURE the field was refused and landed LATER. The caller was made to say
           it again, in words the probe finally recognised. The refusal was
           wrong and the transcript proves it.

Both are set intersections over the artifact. Neither needs a model.

Usage:  python check_refusals.py [--dir DIR] [--since YYYYMMDD]
Exit code is 1 when anything is flagged, so it can gate a batch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

# Which objective field each save tool writes. A refusal names a TOOL; the
# outcome names a FIELD, and the whole scan is the join between them.
TOOL_TO_FIELD = {
    "save_branch": "branch",
    "save_doctor_identity": "identity",
    "save_new_patient_status": "accepting",
    "save_scheduling_status": "scheduling",
    "save_referral_requirement": "referral",
}

# The argument each save tool carries its VALUE in. Distinct from
# TOOL_TO_FIELD, which names where that value would have LANDED — the join
# needs both, and one map cannot answer two questions. Without this the scan
# could see that a refusal happened but never what was refused, which is the
# half both fixes below turn on.
TOOL_TO_ARG = {
    "save_branch": "branch",
    "save_doctor_identity": "identity",
    "save_new_patient_status": "status",
    "save_scheduling_status": "status",
    "save_referral_requirement": "requirement",
}

# Values that assert there is NO answer.
#
# A refusal is only interesting to this scan when a guard destroyed something a
# PERSON said — that is what COST means in the docstring above, and what makes
# the `heard` string worth taking to a probe. The model writing "unknown" into
# save_branch is not that. It is the model reaching for a placeholder because
# the caller had none to give, and the guard blocking it is the system working.
#
# NOT `_INVALID_BRANCH_WORDS` from tools.py, though it contains every word
# here. That set answers a different question — "is this a usable branch?" —
# and it is much wider: "campus", "office", "clinic", "location" are in it too.
# Those are VAGUE ANSWERS, not assertions of absence, and a caller who says
# "she's at the clinic" has given us something the probe arguably should have
# read. Importing the wider set to save a dozen strings would silence a whole
# class of finding this scan exists to surface.
#
# So it is deliberately narrow, and it is about the shape of the value rather
# than the field: a choice tool cannot reach here with "unknown" (its own valid
# set would have rejected it first), so applying it to every tool costs
# nothing and means a new PLACE field is covered the day it is added.
SENTINELS = {
    "unknown", "none", "n/a", "na", "nothing", "null", "nil",
    "nobody", "no one", "noone", "nowhere",
    "not provided", "not given", "not available", "unavailable",
    "not specified", "unspecified", "not stated", "not known",
    "no branch", "no location", "no answer", "unclear",
}


def _attempted(tool: str, args) -> str:
    """The value the tool was CALLED with, lowercased. "" when unrecoverable."""
    return str((args or {}).get(TOOL_TO_ARG.get(tool, ""), "") or "").strip().lower()


def _refusals(call: dict) -> list[dict]:
    """Every refusal in one artifact, from all three places they are recorded.

    THREE SOURCES, because a guard can refuse at three different moments and
    the artifact records each differently. Reading only one is how a scan
    reports "nothing to see" on a call that lost a field:

      save_refusals      refused on the spot, with the caller turn that caused it
      deferred_saves     held for the transcript, then contradicted by it
      branch_rejections  the branch guard's own record, which predates the first

    THE THIRD SOURCE OVERLAPS THE FIRST, which the line above says out loud and
    this function did not act on. A branch the guard blocks is written to
    branch_rejections by _ungrounded_terms AND to save_refusals by the tool
    handler, same call, same value, same second — so every branch refusal was
    counted twice. call-20260831-1209 is the FIRST artifact in 127 to populate
    both, so the double-count sat here from the day the scan was written with
    nothing in the corpus able to fire it. Exactly the shape this project keeps
    finding: a guard written past the real bug ships silently broken.

    KEYED ON (tool, value, at), not on `heard`, because the two records carry
    DIFFERENT `heard` strings for the one event — save_refusals stores the
    caller turn that caused it ("No, no, no one there."), branch_rejections
    stores the rejected value ("unknown"). What they agree on is which tool,
    which value, and which second.

    THE INDEX FALLBACK DOES NOT DEDUPE, and that is the point. deferred_saves
    carries no `at` (it records `waited_s` instead), so without a timestamp
    there is nothing to prove two records describe one event — and merging two
    genuine refusals is a lost finding, which is the expensive direction here.
    The prefix makes that explicit rather than accidental: bare indices would
    collide across sources at position 0 and silently merge. The deferred path
    cannot collide with the on-the-spot one anyway — a save either refuses
    immediately or is held, never both.
    """
    out: list[dict] = []
    seen: set = set()

    def _add(entry: dict, key: tuple) -> None:
        if key in seen:
            return
        seen.add(key)
        out.append(entry)

    for i, r in enumerate(call.get("save_refusals") or []):
        _tool = r.get("tool", "")
        _val = _attempted(_tool, r.get("args"))
        _add({"tool": _tool, "why": r.get("why", ""),
              "heard": r.get("heard", ""), "value": _val,
              "when": "on the spot"},
             (_tool, _val, r.get("at") or f"sr{i}"))
    for i, d in enumerate(call.get("deferred_saves") or []):
        if d.get("outcome") != "contradicted":
            continue
        _tool = d.get("tool", "")
        _val = _attempted(_tool, d.get("args"))
        _add({"tool": _tool, "why": d.get("why", ""),
              "heard": (d.get("args") or {}).get("heard", ""), "value": _val,
              "when": "after the words landed"},
             (_tool, _val, d.get("at") or f"ds{i}"))
    for i, b in enumerate(call.get("branch_rejections") or []):
        _val = str(b.get("value", "") or "").strip().lower()
        _add({"tool": "save_branch", "why": b.get("why", ""),
              "heard": b.get("value", ""), "value": _val,
              "when": "on the spot"},
             ("save_branch", _val, b.get("at") or f"br{i}"))
    return out


def audit(path: pathlib.Path) -> list[dict]:
    """Refusals this call went on to contradict. Empty means the guards held."""
    return audit_dict(json.loads(path.read_text(encoding="utf-8")))


def audit_dict(call: dict) -> list[dict]:
    """The judgement, on an artifact already parsed.

    Split from `audit` so the suite can put a hand-built call in front of it.
    A scan whose only entry point needs a file on disk is a scan whose verdict
    logic gets tested by writing temp files, or not at all.
    """
    if not (call.get("conversation") or {}).get("agent_turns"):
        return []                       # never got off the ground; nothing to judge
    missing = set(call.get("missing") or [])
    collected = set(call.get("collected") or [])
    flagged = []
    for r in _refusals(call):
        field = TOOL_TO_FIELD.get(r["tool"])
        if field is None:
            continue
        # A REFUSED SENTINEL IS NOT A LOST ANSWER. COST is defined above as
        # "the guard did not delay the answer, it destroyed it", and the
        # deliverable is a phrasing to take to a probe. Neither survives here:
        # on call-20260831-1209 the model called save_branch("unknown") after
        # the caller said "No, no, no one there." — there was no branch to
        # lose, the guard was right, and the scan reported it as a probe gap
        # with the literal string "unknown" as the phrasing to go and fix.
        #
        # Dropped BEFORE the verdict rather than only from COST. A sentinel
        # that is refused and then the field lands anyway is the model
        # correcting itself, which is the guard working too — reporting it as
        # PREMATURE would say the caller was made to repeat something they
        # never said.
        if r.get("value") in SENTINELS:
            continue
        if field in missing:
            verdict = "COST"
        elif field in collected:
            verdict = "PREMATURE"
        else:
            continue                    # field was never required on this call
        flagged.append({**r, "field": field, "verdict": verdict,
                        "call_id": call.get("call_id", "?")})
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/3 cases jsons")
    ap.add_argument("--since", default="", help="only calls on/after YYYYMMDD")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(args.dir).glob("call-*.json"))
    if args.since:
        paths = [p for p in paths
                 if (m := re.search(r"call-(\d{8})", p.name))
                 and m.group(1) >= args.since]
    findings, scanned = [], 0
    for p in paths:
        try:
            findings += audit(p)
            scanned += 1
        except Exception as e:                      # a half-written artifact
            print(f"  ! could not read {p.name}: {e}", file=sys.stderr)

    print(f"scanned {scanned} calls in {args.dir}"
          + (f" since {args.since}" if args.since else ""))
    if not findings:
        print("no refusal was contradicted by its own call — the guards held.")
        return 0

    by = Counter(f["verdict"] for f in findings)
    _cost, _prem = by["COST"], by["PREMATURE"]
    print(f"\n{len(findings)} refusal(s) the call itself contradicted "
          f"({_cost} COST, {_prem} PREMATURE)")
    for f in sorted(findings, key=lambda x: (x["verdict"], x["call_id"])):
        print()
        print(f"  {f['verdict']:9} {f['call_id']}   field={f['field']}"
              f"   refused {f['when']}")
        # THE PHRASING IS THE DELIVERABLE. Everything else is context; this is
        # the string to take to _CHOICE_PATTERNS, and then to the suite.
        print(f"      caller said : {f['heard']!r}")
        print(f"      guard said  : {f['why'][:110]}")
    print("\nEach 'caller said' is a phrasing a probe could not read. Fix the "
          "pattern, pin the phrase in test_realtime_protocol.py, and this line "
          "stops appearing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
