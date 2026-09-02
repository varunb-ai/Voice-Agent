"""What the call says about itself once the handlers have decided.

AFTER the guards, which is why this layer sits above them rather than
beside them: there is nothing to report until something has been decided,
and the objective delta is measured against what the call held BEFORE the
tool ran. The order is load-bearing and is the reason the guard and the
report were not merged into one per-tool function when _handle_tool_call
was broken up - an operator reads the objective moving and then reads what
moved it.

REPORTING THAT ANSWERS BACK IS STILL REPORTING. Three branches of
_report_tool_result do more than print: a refused save nudges the model
toward the words it needs, a false save claim corrects a model that has
just told the caller something untrue. Those travel with the report, not
with the teardown.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # A binding for the string annotations below. TYPE_CHECKING is False
    # at run time, so no import happens and the one-way rule holds: the
    # worker imports this package, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.objectives import (
    CallObjective,
    default_objective,
    describe as _describe_objective,
)
from agents.voice.grounding.vocabulary import (
    _CHOICE_SAVE_TOOLS,
    _MAX_SAVE_REJECTIONS,
    _claims_saved,
)
from agents.voice.grounding.handlers import (
    _candidate_location,
)

def _collected_pairs(sess: "RealtimeSession") -> set:
    """Every collected field as (name, value), not just its name.

    A SET OF NAMES CANNOT SEE A STATE FLIP, and on call-20260831-1048 that hid
    the exact moment the call was decided. `identity` was already collected
    when it was overwritten `confirmed` -> `unsure`, so the difference of the
    two name-sets was empty, the 🎯 line did not print, and the objective went
    PARTIAL -> COMPLETE with nothing whatsoever in the log to say so. The one
    line that would have told an operator why the call ended three seconds
    later was suppressed by the very write that ended it — the same shape as
    the metrics that tidied away the repeat before it could be counted.

    A FLIP IS ALSO PROGRESS, which is the other half of what `_gained` feeds.
    confirmed -> not_here is the caller putting us right, and a no-progress
    counter that cannot see it would keep ticking through the most informative
    turn of the call. The regression to `unsure` that motivated this is refused
    in _save_state now and never reaches here.

    Values are read through the field's own memory_key and lowercased, so this
    compares the same strings `present()` accepted rather than a second opinion
    about them.
    """
    obj = _objective_of(sess)
    return {(f.name, str(sess.memory.get(f.memory_key) or "").strip().lower())
            for f in obj.fields if f.present(sess.memory)}



def _objective_of(sess: "RealtimeSession") -> CallObjective:
    """The objective this call is working to.

    getattr, because the guards in this module are routinely handed a namespace
    carrying only the four attributes they read — see `double()` in the test
    suite — and a guard that raises on a test double is a guard that stops being
    tested.
    """
    obj = getattr(sess, "objective", None)
    return obj if isinstance(obj, CallObjective) else default_objective()



def _record_progress(sess: "RealtimeSession",
                     collected_before: set) -> None:
    """
    Did this tool call move the objective on, and say so if it did.

    Between the guards and the reporting, and it has to stay there. The
    delta is measured against what the call held BEFORE the tool ran, so
    it cannot be computed later; and the console line it prints belongs
    ABOVE the per-tool report, because an operator reads the objective
    moving and then reads what moved it.
    """
    # Something new was collected: the no-progress ceiling starts over, whether
    # the call is finished or has another field (or another doctor) to go. This
    # is the reset that makes one ceiling work for a multi-field, multi-doctor
    # call without the counter having to know either number.
    _gained = _collected_pairs(sess) - collected_before
    if _gained:
        # Named with the value it landed on, because a flip and a first
        # collection are now both in here and "collected identity" alone no
        # longer says which happened.
        _what = ", ".join(f"{n}={v}" for n, v in sorted(_gained))
        sess.reset_ask_budget("collected " + _what)
        print(f"[Realtime] 🎯 {_describe_objective(_objective_of(sess), sess.memory)}",
              flush=True)



async def _report_tool_result(name: str, args: dict, result: dict,
                              ok: bool, ts: str,
                              sess: "RealtimeSession", oai_ws) -> None:
    """
    Say what the tool ACTUALLY did - to the console, to the artifact, and
    where a refusal needs answering, to the model.

    THIS USED TO PRINT SUCCESS UNCONDITIONALLY. A live call logged

        BLOCKED : {'branch': 'Downtown'}
        SAVED   : {'branch': 'Downtown'}

    one line apart: the guard had worked and nothing was saved, but the
    log said otherwise. A safeguard that reports itself as having failed
    is worse than no log at all - it sends you hunting a bug that is not
    there and hides the one that is.

    ASYNC BECAUSE REPORTING SOMETIMES HAS TO ANSWER BACK. Three of these
    branches do more than print: a refused save nudges the model toward
    the words it needs, a false save claim corrects a model that has just
    told the caller something untrue. Those are part of reporting, not of
    the lifecycle, which is why they travel with it rather than with the
    teardown below.

    `ts` is passed rather than taken here: the teardown stamps its own
    lines with the same second, and two clocks a few milliseconds apart
    read as two events in the log.
    """
    # ── EVERY REFUSAL, ONE RECORD, WITH THE WORDS THAT CAUSED IT ───────────
    # The counter below this only ever covered save_branch, and a choice-field
    # refusal reached the artifact through nothing at all: on
    # call-20260827-1428 the identity save was refused at 14:29:35 and the
    # finished JSON contains no "BLOCKED", no "REJECTED", no "NOT SAVED".
    # Seven probe gaps have been found on this project by a person reading a
    # console log, and this is why — the evidence was never written down.
    #
    # `heard` IS THE POINT. A refusal without the caller turn that caused it
    # says a guard fired; with it, it says which phrasing the probe could not
    # read, which is the one thing that turns an audit into a fix. The deferred
    # path is not double-counted: a hold returns ok=True and records itself in
    # deferred_saves, and its own refusal lands there as "contradicted".
    if not ok and name.startswith("save_"):
        sess.save_refusals.append({
            "tool": name,
            "args": args,
            "why": str(result.get("error", ""))[:200],
            "heard": next((t.text for t in reversed(sess.turns)
                           if t.role == "caller" and t.text
                           and t.text.strip() != "[...]"), ""),
            "at": ts,
        })
    if name == "save_branch":
        if ok:
            print(f"\n[{ts}] ✅ BRANCH SAVED   : {args}", flush=True)
        else:
            print(f"\n[{ts}] ⛔ BRANCH REJECTED: {args}", flush=True)
            print(f"          reason: {result.get('error', '')}", flush=True)
            # ── NOTHING BOUNDED THIS ────────────────────────────────────────
            # Every correction here is one-shot, and there was no counter at
            # all, so a model that cannot produce an acceptable value simply
            # keeps trying. call-20260820-1321: three attempts, each with a
            # closing line attached — "I'll note that and wrap up", "I'll note
            # it and let you go", "take care" — twenty seconds of a caller
            # being thanked for a branch that had not been recorded. The second
            # rejection got no correction at all, because _false_save_nudged
            # was already spent on the first.
            #
            # That call ended only because the third attempt slipped through
            # the spelled-number bypass. Closing that bypass removes the
            # accidental exit and leaves the loop unbounded, so the bound has
            # to be explicit — a fix that makes a guard stricter has to carry
            # the liveness that the leak was accidentally providing.
            #
            # Guessing is not the way out of this. The caller's own words are
            # already on the transcript and _candidate_location can quote them,
            # so at the limit the model is handed the answer verbatim rather
            # than asked to try again. If it still cannot save, escalating with
            # a true reason beats a call that never ends.
            sess._save_rejections += 1
            if sess._save_rejections >= _MAX_SAVE_REJECTIONS and not sess.done:
                _cand = _candidate_location(sess)
                print(f"[{ts}] 🧱 {sess._save_rejections} save attempts "
                      f"rejected — handing the agent the caller's own words",
                      flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 # Opens with plain lowercase words, like every
                                 # other directive here: the suite finds them by
                                 # reading the source, and an f-string starting
                                 # with a placeholder is invisible to it.
                                 f"(system: nothing has been recorded and "
                                 f"{sess._save_rejections} save attempts have "
                                 f"been rejected. Stop rephrasing it. "
                                 + (f"The caller's own words were: {_cand}. "
                                    f"Call save_branch with exactly that "
                                    f"wording, copying any number digit for "
                                    f"digit. " if _cand else "")
                                 + "If that is rejected too, call escalate "
                                   "with reason 'could not obtain the "
                                   "location'. Do not tell them it is saved "
                                   "and do not say goodbye again until one of "
                                   "those succeeds.)")}]},
                }))
            # The agent may already have TOLD them it was saved.
            # On call-20260818-1613:
            #   "Thanks for checking — I'll save that and then
            #    we'll be all set."          <- spoken
            #   ⛔ BRANCH REJECTED                <- 0.0s later
            # The caller was told the job was done. It was not.
            # That call recovered because the next turn happened to
            # ask a follow-up; the same shape on a rejection that
            # does not recover leaves a receptionist hanging up
            # believing a location was recorded when nothing was
            # written.
            #
            # Same class as the lying console log fixed in 0c28baa:
            # a success message emitted before the operation that
            # decides success. That was fixed in the print; the
            # model does it on the wire.
            #
            # Not fixable by prompt — the model cannot know the
            # result before the tool returns, so no rule makes it
            # reliable. The prompt already carries "Never claim to
            # have noted, saved, or recorded a location you were
            # not given" and it did not hold. But the PROCESS knows
            # both halves: what was said, and that it was rejected.
            _said = next((t.text for t in reversed(sess.turns)
                          if t.role == "agent"), "")
            # FIRES ON EVERY FALSE CLAIM, not once per call. It was
            # one-shot because nothing bounded the retry loop and a guard
            # that can nag forever is its own failure. _MAX_SAVE_REJECTIONS
            # bounds it now, so this can cost at most that many nudges —
            # and each one answers a separate thing the caller was actually
            # told. On call-20260820-1321 the second claim, "Thanks for
            # that branch name — I'll note it and let you go", got no
            # correction at all: the flag was spent on the first, so the
            # caller was left believing a branch had been recorded that
            # had not. Leaving a false statement standing to avoid
            # repeating yourself is the wrong trade.
            # SIGNED OFF WITHOUT ENDING ANYTHING moved to the agent-transcript
            # handler in turns.py. It stood HERE, inside the save_branch-
            # REJECTED branch, so it could only ever see a goodbye that shared
            # a turn with a refused branch save — not a goodbye on a turn with
            # no tool at all, and not one after a save that SUCCEEDED. Those
            # are the two shapes the calls of 2026-09-02 actually produced,
            # and farewell_without_close was null on both.
            #
            # The correction it asks for is unchanged and still the right one:
            # make the TOOL fire, never hang up at the farewell. escalate is
            # what writes the reason, and on 1516 it was the only record of why
            # the call produced nothing.
            if _claims_saved(_said) and not sess.done:
                sess._false_save_claims += 1
                print(f"[{ts}] ⚠️  FALSE SAVE CLAIM — they were told "
                      f"it was saved; correcting", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you just told them the "
                                 "location was saved, or that you "
                                 "were finished. Neither is true — "
                                 "nothing has been recorded. Do not "
                                 "imply it has been. Do not thank "
                                 "them as though the call is over. "
                                 "Say you need one more detail, and "
                                 "ask for it.)")}]},
                }))
    elif name == "escalate":
        label = "⚠️  ESCALATED     " if ok else "⛔ ESCALATE FAILED"
        print(f"\n[{ts}] {label}: {args}", flush=True)
    elif name == "note_info":
        print(f"[{ts}] {'📝 NOTE           ' if ok else '⛔ NOTE REJECTED  '}: {args}",
              flush=True)
    elif name in _CHOICE_SAVE_TOOLS:
        _short = name.replace("save_", "").replace("_status", "")
        if ok:
            print(f"\n[{ts}] ✅ {_short.upper():<14}: {args}", flush=True)
        else:
            print(f"\n[{ts}] ⛔ {_short.upper():<14}: REJECTED {args}", flush=True)
            print(f"          reason: {result.get('error', '')}", flush=True)
    else:
        print(f"[{ts}] 🔧 TOOL           : {name}({args}) → {result}", flush=True)


__all__ = [
    "_collected_pairs",
    "_objective_of",
    "_record_progress",
    "_report_tool_result",
]
