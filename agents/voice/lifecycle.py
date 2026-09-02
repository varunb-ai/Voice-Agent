"""One event: `response.done`, and everything that hangs off it.

EXTRACTED FROM `_oai_to_twilio` 2026-08-27, and the reason is written down in
pyrightconfig.json rather than here: on 2026-08-18 pyright refused to analyse
that function at all — "Code is too complex to analyze" — and once it gives up
it can no longer prove any local in the function is read, so ~60 names greyed
out as false positives. Raising maxCodeComplexity does not fix it (4096, 16384,
65536 and 262144 were all tried); splitting does, and the config says so:

    "If it ever comes back, split the function again rather than adding a
     ceiling here."

The loop had grown back from 599 lines to 815, and `response.done` alone was
397 of them — half the pump, holding the usage accounting, the playback clock,
the echo cooldown, the empty-response re-request, the deferred close and the
hang-up. One event, one handler, the same shape as `_handle_agent_transcript`
and `_handle_caller_transcript`.

NO BEHAVIOUR CHANGES HERE. The body is the block verbatim; what is new is the
state carried in and out, and the two control-flow exits — `continue` and
`break` cannot cross a function boundary, so they return a `flow` the caller
acts on. Getting that wrong is the one way this refactor can break the call,
which is why it is a named field and not a bool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple, Optional

from core.config import settings
from agents.voice.audio import (
    _agent_wire_sample_rate,
    _drop_held_items,
    _wire_bytes_per_ms,
)
from agents.voice.grounding import _create_response, _spoken_farewell

if TYPE_CHECKING:                                    # pragma: no cover
    from agents.voice.session import RealtimeSession

log = logging.getLogger(__name__)


class _ResponseDone(NamedTuple):
    """The event-loop locals this handler reads and writes, round-tripped.

    Same device as `_AudioDelta` in audio.py and for the same reason: these are
    locals of `_oai_to_twilio`, the loop keeps reading them on later
    iterations, and passing them as a named tuple is what lets the analyser
    follow them across the call.

    `flow` is the loop's next move — "" to carry on, "continue" for the goodbye
    retry, "break" to leave the loop and hang up.
    """
    samples_this_response: int
    first_delta_sent_at: Optional[float]
    current_response_start: Optional[float]
    spoken_item_id: Optional[str]
    response_had_audio: bool
    current_item_id: Optional[str]
    closing_sent: bool
    closing_retries: int
    empty_responses: int
    pending_response_create: bool
    barge_in_pending: bool
    echo_cooldown: float
    flow: str = ""


async def _handle_response_done(
    msg: dict,
    sess: "RealtimeSession",
    oai_ws,
    twilio_ws,
    done_event,
    current_response_pcm: list,
    counted_responses: set,
    state: _ResponseDone,
) -> _ResponseDone:
    """Close out one response: account for it, drain it, and decide what next.

    `current_response_pcm` and `counted_responses` are mutated in place — they
    are a buffer and a set, and threading them through the return value would
    say they are decisions when they are storage.
    """
    _samples_this_response = state.samples_this_response
    _first_delta_sent_at = state.first_delta_sent_at
    _current_response_start = state.current_response_start
    _spoken_item_id = state.spoken_item_id
    _response_had_audio = state.response_had_audio
    _current_item_id = state.current_item_id
    _closing_sent = state.closing_sent
    _closing_retries = state.closing_retries
    _empty_responses = state.empty_responses
    _pending_response_create = state.pending_response_create
    _barge_in_pending = state.barge_in_pending
    _echo_cooldown = state.echo_cooldown
    # The two collections keep the names the extracted body already uses.
    _current_response_pcm = current_response_pcm
    _counted_responses = counted_responses

    def _out(flow: str = "") -> _ResponseDone:
        return _ResponseDone(
            _samples_this_response, _first_delta_sent_at,
            _current_response_start, _spoken_item_id, _response_had_audio,
            _current_item_id, _closing_sent, _closing_retries,
            _empty_responses, _pending_response_create, _barge_in_pending,
            _echo_cooldown, flow)

    sess._response_active = False
    # t4 — the tool-carrying response closed, which is the event
    # the deferred response.create waits for. Guarded on t3 so the
    # SPOKEN response's own done (which arrives long after t5)
    # cannot claim this mark.
    if (sess._stage is not None and "t3" in sess._stage
            and "t4" not in sess._stage):
        sess._stage["t4"] = time.monotonic()
    # `_response_spoke = _response_had_audio` stood here, assigned
    # and never read. It came in with c443356 (the 8.2s dead-air
    # fix) and was orphaned when that check moved to the model's
    # own `_out_audio_tokens` from the usage block, which is the
    # honest measure — our delta flag cannot see a response whose
    # audio we gated. Removed 2026-08-18.
    _response_had_audio = False   # reset for next response
    sess._responses    += 1
    # "completed" | "cancelled" | "incomplete" | "failed". This was
    # never read, so a closing response the caller talked over was
    # indistinguishable from one that actually played, and the call
    # hung up on a goodbye nobody heard.
    _resp_status = ((msg.get("response") or {}).get("status")
                    or "completed")
    # WHY it failed, which was being thrown away.
    #
    # call-20260819-2216 had SEVEN `[failed]` responses with
    # in_text=0, and four stretches of 8-11 seconds where nobody on
    # the call made a sound — the failures and the dead air line up
    # one for one. Twilio's own recording showed every agent block
    # reaching the line within 0.4s of generation, so the transport
    # was never the problem, and two rounds of diagnosis went into
    # guessing at a reason the event carried all along.
    #
    # `status_details` holds {type, reason} and, for failures, an
    # {error: {type, code, message}}. Printed, not logged, so it
    # lands in the call log next to the response it explains.
    _sd = ((msg.get("response") or {}).get("status_details") or {})
    if _resp_status in ("failed", "incomplete") and _sd:
        _sd_err = _sd.get("error") or {}
        _why_failed = (_sd_err.get("message")
                       or _sd_err.get("code")
                       or _sd.get("reason") or "no reason given")
        print(f"[Realtime] ⚠️  response {_resp_status}: "
              f"{_why_failed}", flush=True)
        sess.response_failures.append(
            {"status": _resp_status,
             "reason": str(_why_failed)[:200]})
    # The model's own count of audio it produced. Zero on a
    # completed response means it said nothing at all, which on a
    # phone line is indistinguishable from the call having dropped.
    # Read from usage rather than from our local audio-delta flag so
    # that a response carrying a tool call, or one whose deltas we
    # gated, is judged by what the model actually emitted.
    _out_audio_tokens = (((msg.get("response") or {}).get("usage") or {})
                         .get("output_token_details", {})
                         .get("audio_tokens", 0))
    # Input tokens this response consumed. A response that was
    # REJECTED before it ran — conversation_already_has_active_response
    # is the one that matters — comes back failed having read
    # nothing, so both of these are zero. A response that genuinely
    # ran and simply produced no audio has read the conversation and
    # reports input tokens. That difference is the only way to tell
    # "say something, the line is dead" apart from "you already have
    # a response in flight", and re-requesting on the latter is what
    # produced the 25s of dead air on call-20260811-1640.
    _resp_in = (((msg.get("response") or {}).get("usage") or {})
                .get("input_token_details", {}))
    _in_tokens = ((_resp_in.get("text_tokens")  or 0)
                  + (_resp_in.get("audio_tokens") or 0))
    # A response can be cancelled by US (the barge-in handler above,
    # which sets _barge_in_pending) or by OPENAI, whose server VAD
    # interrupts on caller speech on its own. Until now the second
    # kind was completely silent: status came back "cancelled",
    # nothing had logged a barge-in, and no `clear` was ever sent to
    # Twilio, so any audio already buffered there kept playing after
    # generation had stopped.
    #
    # Closing the response.created race above should make this rare
    # — our handler now fires first in the common case. It is kept
    # because "rare" is not "never": the server can still win the
    # race on a slow link, and an interruption path that only works
    # when we win a race is the thing that has been invisible for
    # eight sessions. Logged distinctly so the two are told apart in
    # the transcript rather than inferred.
    # A response that completed and made a sound means the agent has
    # since been heard, so any earlier truncation is no longer the
    # thing to read the next caller turn against. _REPAIR_WINDOW_S
    # bounds this by time; this bounds it by events, which is the
    # tighter of the two and the one that is actually the reason.
    if _resp_status == "completed" and _out_audio_tokens > 0:
        sess._truncated_at = None
    if _resp_status == "cancelled" and not _barge_in_pending:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"✋ BARGE-IN  : cancelled by OpenAI's VAD "
              f"(audio_out={_out_audio_tokens} tok)", flush=True)
        if sess.stream_sid:
            try:
                await twilio_ws.send_text(json.dumps({
                    "event": "clear", "streamSid": sess.stream_sid,
                }))
            except Exception:
                pass
        sess.agent_speaking = False
        _drop_held_items(sess, "OpenAI's VAD cancelled the response")
    # A cancelled response may never emit transcript.done. Clearing
    # the flag only there meant it leaked into the NEXT response and
    # silently swallowed a real transcript line.
    #
    # THE HELD AUDIO IS THE SAME HAZARD, and the same sentence is
    # the reason: an item that never gets a transcript.done is
    # never judged, so without this sweep its PCM would sit in the
    # buffer and be played by whatever released an item NEXT
    # response — audio from a turn that was cancelled, arriving
    # inside a later one. Everything legitimate has already been
    # flushed or popped by now: transcript.done precedes this.
    _barge_in_pending   = False
    _drop_held_items(sess, "the response ended without judging it")
    # Flush buffered agent audio as one contiguous block.
    # Placing it all at _current_response_start means the PCM runs at the correct
    # sample rate (24 kHz) from that point — no overlap, no gaps.
    if _current_response_pcm and _current_response_start is not None:
        sess._agent_pcm.append((_current_response_start, b"".join(_current_response_pcm)))
        print(f"[Realtime] Flushed agent response: {len(_current_response_pcm)} chunks, "
              f"start={_current_response_start:.2f}s, "
              f"dur={_samples_this_response/_agent_wire_sample_rate():.2f}s", flush=True)
    _current_response_pcm.clear()
    _current_response_start = None
    # Dynamic echo cooldown: wait until audio finishes playing on the phone +
    # echo travel time.  response.done fires when the SERVER finishes generating
    # (fast), but the audio is still playing on the handset.  Using a fixed 0.5s
    # caused the agent to hear its own echo and generate a duplicate response.
    # Formula: playback_duration + 0.65s echo margin (min 0.5s for very short clips).
    # Wait until the audio has finished PLAYING on the handset,
    # then a small margin — and no longer, because caller audio is
    # dropped for this whole window.
    #
    # The old formula measured the wait from response.done, which
    # fires when the SERVER finishes generating. Generation runs
    # faster than realtime, so response.done lands well before
    # playback ends, and adding the full clip duration on top of it
    # over-waited by roughly the generation time — about 2s of
    # deafness added to every single turn, directly inflating the
    # measured 2.5-4s response latency.
    #
    # Playback ends at (first chunk sent) + (audio duration), since
    # Twilio plays what we send at realtime speed.
    _audio_seconds = _samples_this_response / _agent_wire_sample_rate()
    if _first_delta_sent_at is not None:
        _playback_ends_at = _first_delta_sent_at + _audio_seconds
        # Kept on the session so _create_response can see it. We
        # hand Twilio audio as fast as OpenAI produces it, and
        # OpenAI produces far faster than realtime — a 6.25s reply
        # arrives in about a second. Everything after that sits in
        # Twilio's queue. Creating another response before the
        # queue drains does not talk OVER the caller; it appends,
        # so they hear one unbroken monologue with no gap to speak
        # into. On call-20260819-2006 that came out as three
        # identical questions in a single 50-word turn, and she
        # hung up.
        sess._playback_ends_at = _playback_ends_at
        _echo_cooldown = max(0.3, _playback_ends_at + 0.25 - time.monotonic())
        # How much of this clip the callee has STILL not heard. The
        # echo gate already reasons in these terms; the silence
        # watchdog did not, and that was the bug — see the comment
        # where _agent_quiet_since is set below.
        _playback_remaining = max(0.0, _playback_ends_at - time.monotonic())
    else:
        _echo_cooldown = max(0.3, _audio_seconds + 0.25)
        # No delta was ever sent, so nothing is playing out.
        _playback_remaining = 0.0
    _first_delta_sent_at = None
    _current_item_id = None
    _spoken_item_id = None
    _samples_this_response = 0
    asyncio.create_task(_end_speaking_gate(sess, _echo_cooldown))
    # Account each response's tokens ONCE. A live call logged the
    # same usage line twice, identical to the token
    # (in_text=4572 cached=4416 in_audio=372 out_audio=108), and
    # counted 6 responses against 4 audio blocks. Every duplicate
    # inflates the cost figure — the one number this project has
    # been trying to get honest.
    _resp_id = msg.get("response", {}).get("id")
    if _resp_id and _resp_id in _counted_responses:
        log.debug("[Realtime] duplicate response.done for %s — "
                  "usage already counted", _resp_id)
        usage = {}
    else:
        if _resp_id:
            _counted_responses.add(_resp_id)
        usage = msg.get("response", {}).get("usage", {})
    if usage:
        details_in  = usage.get("input_token_details",  {})
        details_out = usage.get("output_token_details", {})
        sess._input_audio_tokens  += details_in.get("audio_tokens",  0)
        sess._input_text_tokens   += details_in.get("text_tokens",   0)
        sess._output_audio_tokens += details_out.get("audio_tokens", 0)
        sess._output_text_tokens  += details_out.get("text_tokens",  0)
        # Cached tokens — the only direct evidence that the prompt
        # cache is engaging. Shape varies by API version: a flat
        # `cached_tokens` plus an optional per-modality breakdown.
        cached = details_in.get("cached_tokens_details") or {}
        c_audio = cached.get("audio_tokens", 0)
        c_text  = cached.get("text_tokens",  0)
        if not (c_audio or c_text):
            # No breakdown available — attribute the flat total to
            # text, which is where the static prompt prefix lives.
            c_text = details_in.get("cached_tokens", 0)
        sess._input_audio_cached_tokens += c_audio
        sess._input_text_cached_tokens  += c_text
        # out_text is printed alongside out_audio because the token
        # CAP counts both, and only out_audio was ever shown. When
        # call-20260820-1230 came back "incomplete:
        # max_output_tokens" the line read out_audio=151 against a
        # cap of 400, which looks like it had plenty of room and
        # made the truncation unexplainable from the log alone.
        # The missing half was the text.
        _ot_audio = details_out.get("audio_tokens", 0)
        _ot_text  = details_out.get("text_tokens", 0)
        print(f"[Realtime] usage: in_text={details_in.get('text_tokens', 0)} "
              f"(cached {c_text})  in_audio={details_in.get('audio_tokens', 0)} "
              f"(cached {c_audio})  out_audio={_ot_audio}  out_text={_ot_text}"
              f"  (cap {settings.realtime_max_response_tokens})"
              f"  [{_resp_status}]",
              flush=True)
    # The agent has stopped talking; the ball is with the callee. If
    # they never speak, no VAD event fires and nothing else in this
    # loop will ever run again.
    #
    # The clock starts when the callee STOPS HEARING us, not when
    # response.done arrives. response.done fires when the server
    # finishes generating, and generation runs faster than realtime,
    # so this used to start counting while the agent was still
    # talking — the agent's own voice was counted as the callee's
    # silence. Measured on call-20260811-1649: the watchdog reported
    # 3.5s before "Are you still with me?" when the real gap was
    # 1.41s, and 7.0s before the goodbye when the real gap was 2.45s.
    # The error scales with clip length, so the longest turns were
    # cut off hardest — the call was hung up 2.45s after a handover
    # line, while the callee was still drawing breath.
    #
    # Pointing this at a moment in the FUTURE is intentional: the
    # watchdog compares time.time() - quiet_since, which simply goes
    # negative until playback ends.
    sess._agent_quiet_since = time.time() + _playback_remaining
    # Enable caller audio forwarding after first response (greeting) finishes
    if not sess.listen_enabled.is_set():
        # Everything buffered up to here was never sent to OpenAI,
        # so its ms timestamps count from THIS point, not from
        # stream start. Record where that is before any caller turn
        # can exist — every utterance slice is measured from it.
        sess._listen_start_bytes = sum(len(c) for c in sess._caller_oai_pcm)
        _lead_s = sess._listen_start_bytes / max(_wire_bytes_per_ms(), 1e-9) / 1000
        print(f"[Realtime] Greeting done — now listening to caller "
              f"(OpenAI's audio clock starts {_lead_s:.2f}s into ours)",
              flush=True)
        sess.listen_enabled.set()
    # Deferred response.create from a tool result — safe now that the
    # previous response has completed.
    if _pending_response_create and not sess.done:
        _pending_response_create = False
        await _create_response(oai_ws, sess, why="deferred tool result")
    elif (not sess.done and _resp_status != "cancelled"
          and _out_audio_tokens == 0 and _empty_responses < 2
          and not (_resp_status == "failed" and _in_tokens == 0)
          and not sess._response_active):
        # A response that COMPLETED without producing any audio is
        # dead air: nothing is queued behind it, so the line stays
        # silent until the caller gives up and speaks. On a live
        # call this ran 8.2 seconds and the caller asked "are you
        # there?" — exactly what a person says to a dropped line.
        # Only 'cancelled' is excluded — those are barge-ins, where
        # silence is correct because the caller is talking. This
        # used to require status == 'completed', so an 'incomplete'
        # or 'failed' response producing no audio slipped through
        # and became 10s of dead air on a live call. The status was
        # not logged either, so there was no way to tell which.
        #
        # Widening it to 'failed' then caused the opposite failure.
        # This is the sixth response.create call site and the second
        # to be written without checking _response_active — the same
        # bug 97ff46d fixed in the watchdog. A rejected response
        # comes back failed, this handler read that as dead air and
        # created another, which collided and failed in turn. Two
        # guards, because the two causes are different: skip when a
        # response is already in flight, and skip a failure that
        # never consumed input, which is what a rejection looks like.
        _empty_responses += 1
        print(f"[Realtime] Response produced no audio — "
              f"re-requesting to avoid dead air "
              f"({_empty_responses}/2)", flush=True)
        await _create_response(oai_ws, sess, why="empty response",
                              allow_when_vad_pending=True)
    # ── THE OBJECTIVE FINISHED ON A DEFERRED SAVE ───────────
    # _resolve_deferred_save set the flag and deliberately did not
    # act on it: it runs inside the caller-transcript handler,
    # where `_closing_sent` does not exist and the in-flight
    # response has not spoken yet. Here both are available, so this
    # is where the same two decisions the tool handler makes get
    # made — is this already a goodbye, and does the loop owe
    # itself one more response.done before hanging up.
    #
    # DEFERRED AGAIN WHILE A RESPONSE IS ACTIVE. _pending_response_
    # create may have just started one a few lines above; asking
    # for the goodbye into that is the collision this module has
    # been bitten by twice. The flag survives to the next
    # response.done, which is the correct place to try again.
    if sess._close_after_response and not sess.done:
        if sess._response_active:
            print("[Realtime] 🏁 close deferred — a response is "
                  "already in flight", flush=True)
        else:
            sess._close_after_response = False
            sess.done = True
            _last_agent = next((t.text for t in reversed(sess.turns)
                                if t.role == "agent"), "")
            # ── DID IT ACTUALLY SAY GOODBYE ─────────────────────────────
            # This was `not _last_agent.endswith("?")` -- every declarative
            # sentence in English counted as a farewell. On call-20260902-1842
            # the caller asked "Would you like me to add you there?", the agent
            # said "Okay, thanks for explaining that list - let me think about
            # it for a moment", and the close took that as its goodbye and cut
            # the line 2.2s later. The caller was mid-exchange and was hung up
            # on without a word of parting.
            #
            # The heuristic was standing in for a detector that did not exist
            # when it was written. _spoken_farewell exists now and is the same
            # predicate the sign-off guard uses, so both halves of "was a
            # goodbye said" are answered by one definition rather than two --
            # and the else-branch below, which ASKS the model for a goodbye,
            # finally becomes reachable on the calls that need it.
            #
            # A QUESTION IS STILL NOT A FAREWELL. That was the whole content of
            # the old test and it survives: _SPOKEN_FAREWELL cannot match a
            # turn that only asks something.
            _sounded_like_a_goodbye = _spoken_farewell(_last_agent)
            if _out_audio_tokens > 0 and _sounded_like_a_goodbye:
                # It already said something that can stand as a
                # farewell. Fall through: this response.done is the
                # closing one and the branch below drains the audio.
                print("[Realtime] 🏁 objective complete — the turn "
                      "just spoken stands as the goodbye",
                      flush=True)
            else:
                print(f"[Realtime] 🏁 objective complete — asking "
                      f"for a goodbye (last turn "
                      f"{_last_agent[:40]!r})", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": ("(say a brief warm goodbye "
                                     "now, then stop)"),
                        }],
                    },
                }))
                await _create_response(oai_ws, sess,
                                       why="closing goodbye",
                                       allow_when_done=True)
                # Consumed by THIS response.done immediately below,
                # so the goodbye's own response.done is the one that
                # reaches the hang-up branch.
                _closing_sent = True

    if sess.done:
        if _closing_sent:
            # This is the tool-call response.done — closing response is being generated, wait for it
            _closing_sent = False
        elif _resp_status != "completed" and _closing_retries < 1:
            # The goodbye was cancelled — the caller was still
            # talking, so barge-in killed it. Hanging up here is
            # what drops the line in silence. The goodbye item is
            # still in the conversation; ask for it once more, after
            # a beat so we are not talking over them again.
            _closing_retries += 1
            print(f"[Realtime] Closing response was {_resp_status} — "
                  f"caller talked over it. Retrying the goodbye once.",
                  flush=True)
            # Hand the retry to the watchdog instead of sleeping
            # here. This block runs INSIDE the event loop, so an
            # `await asyncio.sleep(0.8)` stops us reading the
            # socket for 0.8s — and OpenAI's server VAD creates its
            # own response the moment the caller speaks. On
            # call-20260818-1338 the caller said "Mercy Medical
            # Center" during that sleep, `response.created` sat
            # unread so `_response_active` was still False, and the
            # retry went out against stale state:
            #     conversation_already_has_active_response
            # Sleeping inside an event handler means acting on a
            # snapshot of the world taken before the nap.
            #
            # The watchdog is a separate task, so events keep being
            # processed while it waits and `_response_active` is
            # true by the time it fires.
            sess._goodbye_retry_at = time.time() + 0.8
            return _out("continue")
        else:
            # This is the closing response.done.
            # Wait for the FULL audio to finish playing on the caller's phone before hanging up.
            # _echo_cooldown = audio_duration + 0.65s, computed just above from _samples_this_response.
            # Sleeping only 1s was cutting off the goodbye mid-sentence.
            hangup_wait = max(_echo_cooldown, 1.5)
            print(f"[Realtime] Closing done — waiting {hangup_wait:.1f}s for audio to finish playing", flush=True)
            await asyncio.sleep(hangup_wait)
            print("[Realtime] Hanging up now", flush=True)
            done_event.set()
            try:
                await twilio_ws.close()
            except Exception:
                pass
            return _out("break")

    return _out()


async def _end_speaking_gate(sess: "RealtimeSession", delay: float) -> None:
    """Clear agent_speaking once the audio we sent has finished playing out.

    Was a closure redefined inside the event loop on every response, with its
    arguments smuggled in as default values (`s=sess, delay=_echo_cooldown`).
    Pyright could not resolve its type at all — "refers to itself" — which is
    the last thing that stayed unanalysed after the loop was split. Rebuilding
    a coroutine function per response was also pure waste.

    Module level, arguments passed explicitly. Same behaviour, and now typed.

    Moved here with the block that is its only caller.
    """
    await asyncio.sleep(delay)
    sess.agent_speaking = False
    # Under REALTIME_ECHO_GATE=pass this window gates nothing — frames flow
    # throughout — so announcing it as "now listening" was misleading output,
    # implying the caller had been unheard for 6.91s when they had not.
    if settings.realtime_echo_gate != "pass":
        print(f"[Realtime] Echo cooldown done ({delay:.2f}s) — "
              f"listening for caller", flush=True)


# The re-exported surface, declared. Every name here is called from
# realtime_worker and never from inside this module, so without this the
# checker reports the module's whole reason for existing as unused — Pylance
# greys `_handle_response_done` at its own definition, which is how this was
# noticed. Same purpose and same wording as the lists in audio.py and
# evidence.py: it says what the module is FOR, and it keeps a hint storm from
# burying a real warning.
__all__ = [
    "_ResponseDone",
    "_end_speaking_gate",
    "_handle_response_done",
]
