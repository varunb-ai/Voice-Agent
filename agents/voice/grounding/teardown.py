"""The call's lifecycle: when to speak again, when to stop, when to wait.

_create_response is the ONE place response.create is sent, and its
per-site policy is load-bearing rather than cautious - six call sites, and
two of them shipped without checking whether a response was already in
flight and produced dead air on live calls. Read its docstring before
adding a seventh.

The close has three outcomes, not two. Done, deferred, and neither: an
objective can finish inside the same response that just asked the caller a
question, and hanging up there is call-20260831-1048 - a goodbye created
with allow_when_done, which skips the playback guard, whose audio began
1.43s before the question had finished playing.
"""
from __future__ import annotations

import logging
import json
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # A binding for the string annotations below. TYPE_CHECKING is False
    # at run time, so no import happens and the one-way rule holds: the
    # worker imports this package, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.evidence import (
    _ungrounded_terms,
)
from agents.voice.objectives import (
    Outcome,
    describe as _describe_objective,
)
from agents.voice.tools import run_tool
from agents.voice.grounding.vocabulary import (
    _CHOICE_SAVE_TOOLS,
)
from agents.voice.grounding.telemetry import (
    _objective_of,
)

log = logging.getLogger(__name__)


async def _create_response(oai_ws, sess: "RealtimeSession", *, why: str,
                           allow_when_done: bool = False,
                           allow_when_active: bool = False,
                           allow_when_vad_pending: bool = False) -> bool:
    """The one place `response.create` is sent. Returns True if it was.

    There are six call sites and each carried its own guard conditions. Two
    shipped without checking `_response_active` and both produced dead air on
    live calls: 97ff46d fixed the silence watchdog, and the empty-response
    re-request was fixed on 2026-08-11 after a rejected response was read as
    dead air, prompting another that collided and failed in turn. That is one
    missing abstraction, not two bugs — guard logic duplicated per call site
    cannot be made correct by review, and the seventh site would have had the
    same coin-flip.

    THE SITES DO NOT SHARE ONE POLICY. A helper that simply refused when
    `sess.done` would silently kill the goodbye and the goodbye retry, which
    fire *because* the call is done — reintroducing the exact silent no-op this
    exists to prevent. So the policy is declared per site rather than assumed:

      default                  in-flight? refuse.  call over? refuse.
      allow_when_done=True     the closing goodbye and its retry
      allow_when_active=True   the goodbye, which is sent from inside the
                               tool-call handler while that response is still
                               open (see its call site — this one is load-
                               bearing, not caution)
      allow_when_vad_pending=True
                               the three RECOVERY sites — silence watchdog,
                               owed substance, empty-response re-request. They
                               exist because the expected response did NOT
                               arrive, so refusing them on the grounds that one
                               is expected is the exact inversion of their job.
                               Adding this parameter was not optional: without
                               it the watchdog went silent in the suite, which
                               is the dead air it was written to end.

    `why` is logged on refusal. A guard that silently does nothing looks
    exactly like a guard that works, and this module has been bitten by that
    three times.
    """
    if sess._response_active and not allow_when_active:
        log.info("[Realtime] response.create skipped (%s): one already in flight", why)
        return False
    if sess.done and not allow_when_done:
        log.info("[Realtime] response.create skipped (%s): call is closing", why)
        return False
    # STILL PLAYING is not the same as STILL GENERATING, and _response_active
    # only knows the second. OpenAI produces a reply far faster than realtime —
    # a 6.25s turn arrives in about a second — and we forward every delta to
    # Twilio immediately, so the rest sits in Twilio's queue long after OpenAI
    # calls the response done.
    #
    # Creating the next one then does not talk over the caller; it APPENDS.
    # They hear one unbroken monologue with no gap to speak into. On
    # call-20260819-2006 that surfaced as three identical questions inside a
    # single 50-word turn, and the callee hung up.
    #
    # The closing sites are exempt: a goodbye that waits for the queue to drain
    # is a goodbye that arrives after the line is already being torn down.
    _left = sess._playback_ends_at - time.monotonic()
    if _left > 0 and not allow_when_done:
        log.info("[Realtime] response.create skipped (%s): %.1fs of audio is "
                 "still playing out to the caller", why, _left)
        return False
    # THE CALLER HAS STOPPED AND OPENAI IS ABOUT TO ANSWER THEM. Our create
    # would be the second one in that conversation and the server refuses the
    # loser. Skipping is right on the merits as well as the mechanics: the
    # response we would be asking for is the one the VAD is already opening.
    _vad_due = getattr(sess, "_vad_response_due_until", 0.0)
    if time.monotonic() < _vad_due and not (allow_when_active
                                            or allow_when_vad_pending):
        log.info("[Realtime] response.create skipped (%s): OpenAI's VAD is "
                 "already opening one for the turn that just ended", why)
        return False
    await oai_ws.send(json.dumps({"type": "response.create"}))
    # OPTIMISTIC, AND THAT IS THE POINT. _response_active was set only when
    # `response.created` came back, leaving a whole round trip in which a
    # second call site could pass this same guard. Marking it here closes the
    # window against ourselves; the VAD window above closes it against OpenAI.
    sess._response_active = True
    return True



def _decide_close(name: str, result: dict,
                  sess: "RealtimeSession", ts: str) -> bool:
    """
    Is the call over, and if it is, may we hang up yet?

    Sets sess.done, and returns whether the close was DEFERRED instead -
    which the teardown needs, because a deferred close must not fall into
    the ordinary post-tool path either. Two outcomes, not one flag: done
    means hang up after the goodbye, deferred means wait for the person.

    ASKED OF THE OBJECTIVE, NOT OF THE TOOL. See the block below - a
    successful save_branch used to end a call by definition, which was
    right only while the branch was the only thing any template collected.
    """
    # WHEN THE CALL IS OVER, asked of the objective rather than of the tool.
    #
    # This was `name in ("save_branch", "escalate")`, which made a successful
    # save_branch the end of the call by definition — correct only for as long
    # as the branch was the only thing any call collected. On a template that
    # also collects the new-patient status it would hang up the moment the
    # branch landed, before the second question was ever asked, and the artifact
    # would record a PARTIAL call with no sign that we cut it short ourselves.
    #
    # COMPLETE, deliberately, not `is_success`. `success_at` says what counts as
    # a reportable success when the call is over; it must not decide when to
    # stop asking. A template that accepts a partial as success still wants the
    # rest of what it came for.
    #
    # ASKED OF NO TOOL NAME AT ALL, which is the second half of the same
    # correction. The name test that stood here — `save_branch or in
    # _CHOICE_SAVE_TOOLS` — was the outcome test's twin: it assumed the tool
    # that completes an objective is always one that SAVES a field. On
    # patient_discovery it never is. Three of that template's six fields
    # (waitlist_available, new_location_known, call_outcome) are written by
    # note_info, and its own prompt makes the outcome label "THE LAST THING
    # YOU DO" — so the tool that finishes the call was the one tool forbidden
    # from ending it. call-20260902-1544 went outcome=complete at 15:46:06 and
    # ran 19 more seconds and four more turns until the CALLER said "bye";
    # 3 of 5 calls on that template ended that way, against 1 of 45 on
    # provider_verification and 0 of 69 on forage_data_collection.
    #
    # The deferred path beside this one (see _resolve_deferred_save) already
    # asks only the objective, and got that fix after call-20260827-1010 — the
    # identical symptom, "another 24 seconds and four agent turns". This is
    # that same fix, arriving at the synchronous path.
    #
    # `result["ok"]` still gates it: a REFUSED tool must not close a call, and
    # a tool that changed nothing cannot have completed anything the previous
    # turn had not already.
    #
    # ESCALATE IS NOT DEFERRABLE and the branch below does not touch it. It is
    # the model saying it has given up; holding that open for an answer is how
    # a call that has already failed stays on the line. Only the objective path
    # can be deferred, because only it can finish WITHOUT anyone deciding to.
    _close_deferred = False
    if name == "escalate" and result.get("ok"):
        sess.done = True
    elif (result.get("ok")
            and _objective_of(sess).outcome(sess.memory) is Outcome.COMPLETE):
        # THE OBJECTIVE FINISHED ON A QUESTION WE HAVE NOT HEARD BACK ON.
        #
        # call-20260831-1048, and it is the second half of the same defect the
        # `sounded_like_a_goodbye` test below was written for. That test asks
        # the right question — an utterance ending in "?" is not a farewell —
        # and then has nowhere to put the answer: its only two branches are
        # "let the model's line stand as the goodbye" and "ask for a goodbye
        # anyway". Neither is "do not hang up yet". So the agent asked "would
        # scheduling be the best group to ask about where she sees patients?",
        # the objective flipped COMPLETE inside the same response, a goodbye
        # was requested with allow_when_done (which bypasses the playback
        # guard), and its audio began 1.43s BEFORE the question had finished
        # playing out. The caller was talked over and hung up on, mid-question.
        #
        # The third branch, then. The objective really is complete and nothing
        # here disputes that — the close is deferred, not cancelled, and it
        # re-arms the moment they answer (see _close_when_answered, consumed in
        # _handle_caller_transcript). If they never answer, the silence
        # watchdog still ends the call on its own budget, so this cannot hold a
        # line open indefinitely.
        # UNANSWERED, WHICH IS NOT THE SAME AS "the last agent turn ends in ?".
        # The happy path ends every call on a question the caller then answered
        # — "which location is Dr. Okafor practising at?", "She's at the
        # Northgate campus." — and the save that completes the objective is
        # grounded in that very answer. Reading only the last AGENT turn would
        # defer the close on every well-run call in the suite. So walk back
        # from the end: a real caller turn in between means the question was
        # answered and nothing is owed. A "[...]" placeholder does not — that
        # is their answer still in flight, which is a reason to wait, not to
        # hang up.
        # THEIRS COUNTS TOO, and only this direction was ever checked. The
        # walk below asks whether OUR last question went unanswered; it says
        # nothing about a question THEY asked that we have not answered yet.
        # call-20260902-1842 ended on "Would you like me to add you there?" --
        # a direct offer, on the table, with the objective completing in the
        # same breath. Hanging up on that is the same discourtesy the walk
        # below exists to prevent, arriving from the other side.
        #
        # NEWEST REAL TURN ONLY. If any agent turn has followed their question
        # we have said something back and this does not fire; a "[...]"
        # placeholder is a transcript still in flight, not a reply.
        _their_question = ""
        for _t in reversed(sess.turns):
            _txt = (_t.text or "").strip()
            if _txt == "[...]":
                continue
            if _t.role == "agent":
                break                   # we have spoken since; nothing owed
            if _t.role == "caller":
                if _txt.endswith("?"):
                    _their_question = _txt
                break

        _unanswered = ""
        for _t in reversed(sess.turns):
            if _t.role == "caller":
                if (_t.text or "").strip() != "[...]":
                    break
                continue
            if _t.role == "agent":
                if (_t.text or "").rstrip().endswith("?"):
                    _unanswered = _t.text.rstrip()
                break
        if _their_question and not _unanswered:
            sess._close_when_answered = True
            _close_deferred = True
            print(f"\n[{ts}] CLOSE DEFERRED  : objective complete, but "
                  f"they have just asked us something and we have not "
                  f"answered it\n          they asked: "
                  f"{_their_question[-70:]!r}",
                  flush=True)
            return _close_deferred
        if _unanswered:
            sess._close_when_answered = True
            _close_deferred = True
            print(f"\n[{ts}] ⏸️  CLOSE DEFERRED  : objective complete, but "
                  f"the turn just spoken is a question they have not "
                  f"answered — waiting for them\n"
                  f"          asked: {_unanswered[-70:]!r}", flush=True)
        else:
            sess.done = True
    return _close_deferred



async def _close_or_continue(sess: "RealtimeSession", oai_ws,
                             _close_deferred: bool,
                             _response_had_audio: bool
                             ) -> tuple[Optional[bool], Optional[bool]]:
    """
    What happens to the line once the tool has been answered. Three ways out.

      done              the objective is met and the question outstanding
                        was answered - say goodbye and hang up, either by
                        letting the turn just spoken stand as one or by
                        asking for one.
      close deferred    the objective is met but the agent has just asked
                        them something. Say NOTHING. The only correct next
                        sound on the line is theirs.
      neither           an ordinary tool call mid-conversation; the model
                        gets a response to speak the result into.

    The parameter names keep their leading underscores because the
    comments below argue about them by name, and a history that no longer
    matches the code it describes is worse than no comment.

    Returns (closing_sent, pending_response_create) for the event loop.
    """
    # None, not False, on both. The event loop reads None as 'this tool
    # call had no opinion' and keeps whatever it already held - see
    # _ToolOutcome. Returning False here would clobber a closing flag set
    # by an earlier response, which is the bug that shape exists to stop.
    _closing_sent: Optional[bool] = None
    _pending_response_create: Optional[bool] = None
    if sess.done:
        # "_response_had_audio" was being read as "the agent said
        # goodbye", so the call hung up on whatever it happened to
        # be saying. On a live call it asked "which office is Dr.
        # Okafor working out of?", called save_branch in the same
        # response, and hung up — leaving the caller answering a
        # question to a dead line.
        #
        # An utterance ending in a question mark is not a farewell.
        last_agent = next((t.text for t in reversed(sess.turns)
                           if t.role == "agent"), "")
        sounded_like_a_goodbye = bool(last_agent) and not last_agent.rstrip().endswith("?")

        if _response_had_audio and sounded_like_a_goodbye:
            # Model already said goodbye in its audio — don't inject another line
            # The current response.done will trigger the close
            _closing_sent = False
        else:
            # Tool fired with no spoken goodbye. Ask for one via a
            # conversation item rather than a per-response
            # `instructions` override — an override swaps out the
            # session instructions and lands this response on a
            # different, uncacheable prefix.
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "(say a brief warm goodbye now, then stop)",
                    }],
                },
            }))
            # BOTH overrides, and both are load-bearing:
            #  - done: sess.done was set 40 lines up, by the very
            #    tool call this goodbye belongs to.
            #  - active: we are inside the tool-call handler, so
            #    the response carrying that tool call has not
            #    emitted response.done yet. Before the barge-in fix
            #    _response_active was set on the first AUDIO delta,
            #    and a tool-only response emits none — so this read
            #    False by accident. Setting it on response.created
            #    made it correctly True, which would have made a
            #    naive helper eat the goodbye. The call is left
            #    unguarded exactly as it was, deliberately.
            await _create_response(oai_ws, sess, why="closing goodbye",
                                   allow_when_done=True,
                                   allow_when_active=True)
            _closing_sent = True  # skip tool-call response.done, close on closing's
    elif _close_deferred:
        # NO CONVERSATION ITEM AND NO response.create. The response carrying
        # this tool call has already put a question to them; the only correct
        # next sound on this line is theirs.
        #
        # Which is why this is its own branch and not a fall-through to the
        # `else`. `_pending_response_create` fires a create at the next
        # response.done — that is right after an ordinary tool call, where the
        # model has a result to speak to, and wrong here, where speaking again
        # is the whole thing being avoided. It would talk over the question by
        # the same 1.4s the injected goodbye did, minus the goodbye.
        #
        # Left as None in the outcome below rather than set False, so the event
        # loop keeps whatever it already had — see _ToolOutcome on why None is
        # not False.
        pass
    else:
        _pending_response_create = True
    return _closing_sent, _pending_response_create


async def _resolve_deferred_save(sess: "RealtimeSession", oai_ws) -> None:
    """Judge a held save now that the caller's words have actually arrived.

    THE SAME GUARD, ON THE SAME ARGUMENTS, AGAINST REAL EVIDENCE. Nothing here
    is more permissive than the path that deferred: `_guard` is re-read from
    `_CHOICE_SAVE_TOOLS` rather than carried along, so a save can only pass by
    satisfying exactly the check that objected. The deferral buys the guard its
    evidence; it does not lower the bar, and the model's `heard` string is
    never consulted — selection runs over the transcript as it always did.

    Three outcomes, all recorded:
      applied      the words bear it out; the tool runs for real, late
      contradicted the words arrived and refuse it; the model is told to ask
                   again — the re-ask this whole mechanism avoids is CORRECT
                   here, because now there is evidence for it
      (unresolved) the transcript never came; nothing is written and the row
                   is closed out in save() as dropped

    WHY THE CORRECTION IS INJECTED RATHER THAN RETURNED. The tool call this
    belongs to was answered a turn ago, with ok=True. There is no result left
    to fail. A conversation item is the only channel that still reaches the
    model, and it is the same one the false-save and silence directives use.
    """
    held = sess._deferred_save
    if held is None:
        return
    sess._deferred_save = None
    name = held["name"]
    # TWO FAMILIES, ONE RESOLVER. The choice fields carry their guard in
    # _CHOICE_SAVE_TOOLS; save_branch has its own guard and no BLOCKED
    # memory key, so _gkey is None for it and the contradiction path below
    # skips the memory write rather than inventing a key.
    if name == "save_branch":
        _guard = _ungrounded_terms
        _need = "their own words, any number in digits"
        _gkey = None
    else:
        spec = _CHOICE_SAVE_TOOLS.get(name)
        if spec is None:
            return
        # The value argument is not needed here — the guard reads it off
        # `args` itself. Named out rather than indexed so this stays in step
        # with _CHOICE_SAVE_TOOLS if its shape changes.
        _, _guard, _need, _gkey = spec
    args = held["args"]
    waited = round(time.monotonic() - held["at"], 2)
    still = _guard(args, sess)
    ts = datetime.now().strftime("%H:%M:%S")

    if not still:
        result = run_tool(name, sess.memory, args, sess.objective)
        ok = bool(result.get("ok"))
        sess.deferred_saves.append(
            {"tool": name, "args": args, "waited_s": waited,
             "outcome": "applied" if ok else "refused_by_tool",
             "held_because": held["why"],
             "error": None if ok else str(result.get("error"))[:160]})
        if ok:
            sess.reset_ask_budget(f"deferred save landed: {name}")
            print(f"\n[{ts}] ✅ {name.upper()} (held {waited:.2f}s for the "
                  f"transcript): {args}", flush=True)
            print(f"[Realtime] 🎯 "
                  f"{_describe_objective(_objective_of(sess), sess.memory)}",
                  flush=True)
            # THE DEFERRED PATH COULD NOT END A CALL. The synchronous tool
            # handler sets sess.done when a successful save completes the
            # objective — see "WHEN THE CALL IS OVER" above. This path runs the
            # same tool, for real, a turn later, and had no such check. On
            # call-20260827-1010 the new-patient status landed here,
            # `outcome=complete` printed, and the call ran another 24 seconds
            # and four agent turns, ending on a "Take care." the model had
            # already said once.
            #
            # A FLAG, NOT THE GOODBYE, and both halves of that are forced:
            #  - `_closing_sent` is a local of the event loop. We are inside
            #    _handle_caller_transcript, which returns None and "shares NO
            #    mutable state with the event loop" by design. Injecting the
            #    closing here would leave `_closing_sent` False, so the
            #    in-flight response's own response.done would read "done,
            #    nothing pending" and hang up ON the goodbye we just asked for.
            #  - the sounded_like_a_goodbye test cannot run yet. The response
            #    in flight has not produced its transcript, so there is no last
            #    agent turn to inspect — and hanging up on a question is the
            #    live defect that test exists for.
            # Both are answerable one event later, which is where it is done.
            if (not sess.done
                    and _objective_of(sess).outcome(sess.memory)
                        is Outcome.COMPLETE):
                sess._close_after_response = True
                print(f"[Realtime] 🏁 objective complete on the deferred save "
                      f"— closing after the response already in flight",
                      flush=True)
        else:
            print(f"\n[{ts}] ⛔ {name.upper()} REJECTED after the wait: "
                  f"{result.get('error', '')}", flush=True)
        return

    # THE WORDS ARRIVED AND THEY DO NOT SUPPORT IT. Refuse exactly as the
    # undeferred path would have, and say so out loud — a held save that
    # quietly evaporates is the invisible-guard failure this project keeps
    # paying for.
    # save_branch has no BLOCKED key of its own; only the choice fields do.
    if _gkey:
        sess.memory.update(**{_gkey: f"BLOCKED — {still}"})
    sess.deferred_saves.append(
        {"tool": name, "args": args, "waited_s": waited,
         "outcome": "contradicted", "held_because": held["why"], "why": still})
    print(f"\n[{ts}] 🚫 HELD ANSWER REFUSED — the transcript arrived and does "
          f"not bear it out: {name}({args})", flush=True)
    print(f"          {still}", flush=True)
    await oai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": (
                     f"(system: the answer you recorded for {name} was not "
                     f"borne out by what they actually said. It has NOT been "
                     f"saved. Ask them again, plainly, and wait for their "
                     f"reply. NEED: {_need})")}]},
    }))


__all__ = [
    "_close_or_continue",
    "_create_response",
    "_decide_close",
    "_resolve_deferred_save",
    "log",
]
