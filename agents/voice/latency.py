"""Per-turn latency instrumentation. Measure-only.

Split from realtime_worker 2026-08-26, verbatim.

- Takes a dict of monotonic marks; returns a dict and a string. No session, no
  websocket, no module state. That is why it was the safe first extraction.
- Measure-only is TESTED: a mutation that makes any guard read a stage timing
  fails the suite. Nothing here may grow a production caller.
- conversation_metrics is not here. See metrics.py.
"""
from __future__ import annotations


def _stage_row(st: dict, felt: float) -> dict:
    """Turn the six marks of one turn into the five intervals they bound.

    A MISSING INTERVAL IS None, NEVER 0.0, and that is the whole discipline of
    this function. A turn with no tool call has no t2/t3/t4 at all; reporting
    those as zero would put it in the same bucket as a tool turn whose deferral
    happened to be instant, which is precisely the conflation the record exists
    to break. The median over these fields must be taken across turns that
    HAVE them, so absence has to survive into the artifact as absence.
    """
    g = st.get

    def _d(a: str, b: str):
        x, y = g(a), g(b)
        return round(y - x, 3) if (x is not None and y is not None) else None

    return {
        "tool":        g("tool"),
        # Every tool the turn carried, in order, each with its verdict. `tool`
        # above stays the FIRST one because the intervals are measured from it;
        # this is the full list, and it is what makes a turn auditable when the
        # model calls two tools in one response.
        "tools":       g("tools"),
        "detector_s":  g("detector_s"),
        "felt_s":      round(felt, 3),
        # caller stopped -> OpenAI opened a response. Contains our detector
        # window and the uplink; it is the one stage we partly own.
        "vad_to_resp": _d("t0", "t1"),
        # response opened -> tool call emitted. THE STAGE THAT CARRIED THE
        # VARIANCE on call-1134: 4.99s of spread against 0.55s everywhere else.
        "inference_1": _d("t1", "t2"),
        # guards, grounding, the tool itself, any transcript wait. Ours.
        "our_work":    _d("t2", "t3"),
        # answered -> the tool response closed. Nothing is asked of OpenAI in
        # this window; it is the cost of deferring response.create.
        "deferral":    _d("t3", "t4"),
        # spoken response -> first audio delta.
        "inference_2": _d("t4", "t5"),
        # A turn with no tool call has one inference, not two. Recorded under
        # its own name so it never averages together with either half.
        "no_tool_s":   _d("t1", "t5") if g("t2") is None else None,
        # response opened -> the caller heard SOMETHING. Recorded on every
        # turn, including tool turns, because on a turn where the model speaks
        # BEFORE calling its tool this is the only interval that says when the
        # line stopped being silent — and `inference_1` above then spans the
        # padding sentence rather than measuring an inference.
        "to_first_audio": _d("t1", "t5"),
        # DID THE MODEL SPEAK BEFORE IT CALLED ITS TOOL? Decided at t5 and
        # frozen there, because after a late tool stamps t2 the ordering is no
        # longer recoverable from the marks alone. This is the discriminator
        # between the two shapes of tool turn: tool-first, where the caller
        # waits in silence for the whole round trip, and speech-first, where
        # they hear a contentless sentence quickly and wait for the real one
        # afterwards. They cost the same and only one of them looks fast.
        "spoke_first": g("spoke_first"),
    }


def _restage(stage: dict, rows: list) -> None:
    """Re-render a row that was already appended, after later marks arrived.

    WHY A ROW IS WRITTEN TWICE. The record used to be closed and DISCARDED at
    the first audio delta, which is correct only if nothing interesting happens
    afterwards. On a turn where the model speaks before calling its tool,
    everything interesting happens afterwards: call-20260904-1734 made three
    save_branch calls and its artifact reported `tool: null` on all eight
    turns, median 2.2s, because every t2/t3 stamp is guarded on the record
    still existing and it no longer did.

    So the row is appended at t5 — the printed line still arrives while the
    call is in front of someone — and rewritten in place by whichever late mark
    lands. `row` and `felt` are carried in the stage dict itself so this stays
    session-free, which is the property that made this module safe to split
    out and is asserted by the suite.
    """
    i = stage.get("row")
    if i is None or not (0 <= i < len(rows)):
        return
    rows[i] = _stage_row(stage, stage.get("felt") or 0.0)


def _fmt_stages(r: dict) -> str:
    """One line, only the stages this turn actually had."""
    out = []
    for key, label in (("vad_to_resp", "vad->resp"), ("inference_1", "infer1"),
                       ("our_work", "ours"), ("deferral", "defer"),
                       ("inference_2", "infer2"), ("no_tool_s", "infer")):
        v = r.get(key)
        if v is not None:
            out.append(f"{label} {v:.2f}s")
    return "  ".join(out) + f"   [{r.get('tool') or 'no tool'}]"


# The re-exported surface, declared. These are called from realtime_worker and
# from audio.py, never from inside this module, so without this the checker
# reports the module's whole reason for existing as unused. Same purpose as the
# list in evidence.py: it says what the module is FOR, and it keeps a hint storm
# from burying a real warning.
__all__ = [
    "_fmt_stages",
    "_restage",
    "_stage_row",
]
