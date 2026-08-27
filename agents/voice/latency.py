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
    }


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
    "_stage_row",
]
