"""One tool call, start to finish. The router and nothing else.

_handle_tool_call was 869 lines and mutated a dozen locals across four
separate dispatches on the tool name before acting on them at the bottom.
It is twenty statements now, in the order they happen: parse, guard,
record, report, decide, answer, close.

THE PHASES ARE NOT INTERCHANGEABLE. The objective delta is measured
between the guard and the report; `ts` is stamped once so the report and
the teardown agree to the second; and the function_call_output goes back
AFTER the close decision, because escalate's own refusal path answers the
call itself and returns early.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # A binding for the string annotations below. TYPE_CHECKING is False
    # at run time, so no import happens and the one-way rule holds: the
    # worker imports this package, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.evidence import (
    _transcript_pending,
)
from agents.voice.tools import run_tool
from agents.voice.grounding.vocabulary import (
    _CHOICE_SAVE_TOOLS,
)
from agents.voice.grounding.handlers import (
    _guard_choice_save,
    _guard_escalate,
    _guard_save_branch,
)
from agents.voice.grounding.telemetry import (
    _collected_pairs,
    _record_progress,
    _report_tool_result,
)
from agents.voice.grounding.teardown import (
    _close_or_continue,
    _decide_close,
)

class _ToolOutcome(NamedTuple):
    """What a tool call changed in the event loop's own state.

    None means "not touched", which is NOT the same as False: the loop must
    not clobber _closing_sent with False just because a tool call that had
    nothing to say about it happened to run. None is safe as the sentinel
    because "" and False are the meaningful values here and both are distinct
    from it.

    Typed concretely rather than as `object`. The first version used an
    `object()` sentinel, which widened _agent_text_buf to `object` all the way
    back into the loop and broke the call into _handle_agent_transcript(...,
    _agent_text_buf: str). Pyright caught that the moment the split brought the
    function back under its analysis ceiling — a type error that had been
    sitting there invisible.
    """
    agent_text_buf: Optional[str]
    closing_sent: Optional[bool]
    pending_response_create: Optional[bool]
    stop: bool



async def _handle_tool_call(msg: dict, sess: "RealtimeSession", oai_ws,
                            _pending_tools: dict,
                            _response_had_audio: bool) -> _ToolOutcome:
    """Run one tool call and its guards. Extracted from _oai_to_twilio.

    Pyright refused to analyse that function at all —

        Code is too complex to analyze; reduce complexity by refactoring
        into subroutines or reducing conditional code paths

    — and when it gives up it can no longer prove any local inside is read, so
    the editor greyed out ~60 names as unused and stopped seeing the calls the
    function makes. Raising maxCodeComplexity does NOT help; the ceiling is
    not the binding constraint. The only fix is the one the message names.

    That mattered beyond the noise. Every recurring bug this week lived in
    that unanalysed function: the barge-in pre-audio race, the six
    response.create sites, the five-clause dead-air condition, the audio_rms
    overwrite, and a dead assignment. Most bugs, least tooling.

    This handler is the largest self-contained piece — 290 lines, 34 branch
    points, a quarter of the function's total — and its coupling to the loop
    is three flags and one `continue`, which is why it goes first.
    """
    _agent_text_buf: Optional[str] = None
    _closing_sent: Optional[bool] = None
    _pending_response_create: Optional[bool] = None
    call_id  = msg.get("call_id", "")
    name     = msg.get("name", "")
    args_str = msg.get("arguments") or _pending_tools.get(call_id, {}).get("args", "{}")
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        # A TRUNCATED TOOL CALL IS AN EVENT, NOT AN EMPTY ONE. This used to set
        # args = {} and say nothing, which turns "the model's tool call was cut
        # off mid-arguments" into "the model called this tool with no
        # arguments" -- two different facts, and only the first has a cause you
        # can act on. The cause is a barge-in: OpenAI cancels the response
        # while function_call_arguments is still streaming, and .done fires on
        # the partial string anyway.
        #
        # Recorded here and refused downstream. run_tool now reads the impl's
        # signature and returns a rejection for a call missing its required
        # arguments, so this no longer has to guess at a repair -- it only has
        # to stop pretending the call was well-formed.
        args = {}
        sess.malformed_tool_calls.append(
            {"tool": name, "raw": (args_str or "")[:160],
             "why": "arguments did not parse - the call was cut off"})
        print(f"[Realtime] TOOL CALL TRUNCATED - {name}: arguments did not "
              f"parse ({(args_str or '')[:60]!r}); refusing it", flush=True)

    # t2 — the tool call is here.
    if sess._stage is not None and "t2" not in sess._stage:
        sess._stage["t2"] = time.monotonic()
        sess._stage["tool"] = name

    # THERE WAS A BLOCKING WAIT HERE, and it is gone. Every guard below asks
    # what the caller said, and the model reached this line from audio the
    # transcript has not necessarily caught up with — so this used to hold the
    # whole handler up to 1.5s for the words. Measured over 119 artifacts it
    # never once returned early (14 waits, 12 timeouts, 0 landed) and cost
    # 1.5s a time; the deferral below does the same job on the transcript event
    # itself, which is where the evidence actually appears. See the comment
    # above _transcript_pending for the distribution that settled it.

    # What the call had collected BEFORE this tool ran, so the no-progress
    # ceiling can be reset by progress rather than by a guess about which tool
    # constitutes progress. save_branch is not the only way a field arrives —
    # a template may point a field at a note_* key — and hard-coding the tool
    # name here is how the success condition ended up inside save_branch in the
    # first place.
    #
    # (name, VALUE) pairs, not names — see _collected_pairs. A field that is
    # overwritten with a different state is progress this set has to be able to
    # see, and until 2026-08-31 it could not.
    _collected_before = _collected_pairs(sess)

    if name == "save_branch":
        result = _guard_save_branch(name, args, sess)
    elif name in _CHOICE_SAVE_TOOLS:
        result = _guard_choice_save(name, args, sess)
    elif name == "escalate":
        result, _stop = await _guard_escalate(
            name, args, sess, oai_ws, call_id, _pending_tools)
        if _stop:
            # It answered the tool call itself: the refusal went back to the
            # model with the nudge saying what to do instead, and everything
            # below this point reports an escalation that is not happening.
            # Clearing the buffer is what the original did on this path.
            _agent_text_buf = ""
            return _ToolOutcome(_agent_text_buf, _closing_sent,
                                _pending_response_create, True)
    else:
        result = run_tool(name, sess.memory, args, sess.objective)

    _record_progress(sess, _collected_before)

    # Report what the tool ACTUALLY did. This used to print
    # "✅ BRANCH SAVED" unconditionally, without looking at the
    # result — so a live call logged
    #     🚫 HALLUCINATED BRANCH BLOCKED: {'branch': 'Downtown'}
    #     ✅ BRANCH SAVED : {'branch': 'Downtown'}
    # one line apart. The guard had worked and nothing was saved,
    # but the log said otherwise. A safeguard that reports itself as
    # having failed is worse than no log at all: it sends you
    # hunting a bug that isn't there and hides the one that is.
    ts = datetime.now().strftime("%H:%M:%S")
    ok = bool(result.get("ok"))
    await _report_tool_result(name, args, result, ok, ts, sess,
                              oai_ws)

    # _response_had_audio IS PART OF THE CLOSE DECISION, not only of what to
    # say once it is made. See the "spoken" branch in _decide_close: a
    # completing tool that spoke nothing, after a turn of ours that did, means
    # any goodbye asked for now stacks on our own finished utterance.
    _close_deferred = _decide_close(name, result, sess, ts,
                                    _response_had_audio)

    await oai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type":    "function_call_output",
            "call_id": call_id,
            "output":  json.dumps(result),
        },
    }))
    # t3 — answered. Everything after this point is OpenAI's, and on the
    # deferred path (sess.done False) nothing is even ASKED of it until
    # response.done arrives. That gap is t4-t3 and it is the cost of the
    # deferral, isolated.
    if sess._stage is not None and "t3" not in sess._stage:
        sess._stage["t3"] = time.monotonic()

    # EVERY TOOL IN THE TURN, WITH ITS VERDICT - not just the first.
    #
    # t2/t3 above deliberately mark only the FIRST tool, because that is what
    # the inference_1 interval measures. But a response may carry several tool
    # calls, and recording only the first made call-20260826-1656 unreadable:
    # identity is saved `confirmed`, the only save_doctor_identity in the stage
    # data sits on a turn whose transcript the guard REJECTS, and the stored
    # quote appears in two different caller turns. Which turn grounded identity
    # could not be determined from the artifact at all.
    #
    # Appended here because `result` is final at this line - every accept,
    # reject and hold path has converged by the time the output goes out.
    #
    # Written with sess._stage[...] rather than .setdefault so it stays inside
    # the shapes the measure-only test allows: this list is written and never
    # read by anything that decides behaviour.
    if sess._stage is not None:
        if "tools" not in sess._stage:
            sess._stage["tools"] = []
        sess._stage["tools"].append(
            {"tool": name, "ok": bool(result.get("ok"))})

    _closing_sent, _pending_response_create = await _close_or_continue(
        sess, oai_ws, _close_deferred, _response_had_audio)
    return _ToolOutcome(_agent_text_buf, _closing_sent,
                        _pending_response_create, False)


__all__ = [
    "_ToolOutcome",
    "_handle_tool_call",
]
