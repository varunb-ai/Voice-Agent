"""Offline protocol test for the Realtime voice agent.

Drives handle_realtime() through complete calls against fake Twilio and OpenAI
sockets, then asserts on the actual wire traffic. No network, no API key, no
phone call, no cost.

What it guards:
  * exactly one session.update, carrying STATIC instructions
  * no per-call data in `instructions` (the prompt-cache prefix stays stable)
  * no `instructions` override on any response.create (same reason)
  * voice / turn detection / transcription / token cap are configured
  * per-call facts arrive as a conversation item
  * the greeting discloses both automation and recording
  * tool results are returned to the model
  * a refusal ends the call via escalate() rather than another ask

Run:
    python test_realtime_protocol.py
Exit code 0 = all assertions passed.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
from datetime import datetime as _dt
import pathlib
import re
import sys
import tempfile
import time
import types
from typing import Any
from unittest import mock

import numpy as np

import core.bootstrap  # noqa: F401  (UTF-8 stdout on Windows)

# soundfile is only needed to write the WAV; stub it so this runs anywhere.
if "soundfile" not in sys.modules:
    try:
        import soundfile  # noqa: F401
    except ImportError:
        _sf = types.ModuleType("soundfile")
        setattr(_sf, "write", lambda *a, **k: None)
        sys.modules["soundfile"] = _sf

from core.models import Doctor
import agents.voice.realtime_worker as rw
NL = chr(10)
import agents.voice.audio as _rwaudio
import agents.voice.grounding as _rwground
import agents.voice.turns as _rwturns
import agents.voice.session as _rwsession

# THE VOICE PACKAGE AS ONE TEXT. Most source-anchored checks here assert that
# THIS CODEBASE does something — the reset lives in the response.created
# handler, the watchdog routes through _create_response, the drop records a
# phantom. None of those claims was ever about a FILE, and after the 2026-08-26
# split each one would otherwise have to name the module its subject happens to
# live in today, and be re-pointed again at the next extraction. Worse, a check
# that reads one module after its subject moves does not fail — it finds
# nothing and passes, which is how four checks went quiet and were only caught
# by diffing check names against a baseline.
_PKG_SRC = "\n".join(
    _p.read_text(encoding="utf-8")
    for _p in sorted(pathlib.Path("agents/voice").glob("*.py")))


# ── Fakes ─────────────────────────────────────────────────────────────────────


def _fake(obj) -> Any:
    """Hand a purpose-built test double to a parameter annotated for the real
    thing. The hand-rolled fake CLASSES below (_S, _Sess, _FakeSess, FakeTwilio)
    carry only the attributes the function under test reads; this states that
    once per call site instead of leaving twenty type errors standing."""
    return obj


def double(**kw) -> Any:
    """A duck-typed stand-in for a RealtimeSession (or a Doctor, or a turn).

    The guards take `sess: "RealtimeSession"` but read four or five attributes
    off it, so the unit checks below hand them a namespace carrying exactly
    those. That is deliberate — building a real session per case would drag in
    a websocket, a Doctor record and a template for no added coverage.

    Returning `Any` is the point. Without it every one of these call sites is a
    reportArgumentType error, forty-odd of them, and real type errors in this
    file are invisible in the noise. This states the duck-typing once instead
    of suppressing it at each site.
    """
    return types.SimpleNamespace(**kw)


class FakeOAI:
    """Scripted OpenAI Realtime socket. Records everything sent to it."""

    def __init__(self, script, sent):
        self._script = list(script)
        self.sent = sent

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return json.dumps(self._script.pop(0))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._script:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        return json.dumps(self._script.pop(0))


class FakeConn:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *a):
        return False


class FakeTwilio:
    def __init__(self, msgs):
        self._msgs = list(msgs)
        self.sent = []
        self.closed = False

    async def iter_text(self):
        for m in self._msgs:
            yield json.dumps(m)
            await asyncio.sleep(0)
        while not self.closed:
            await asyncio.sleep(0.01)

    async def send_text(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True


def _b64_silence(ms: int = 750) -> str:
    """A base64 μ-law payload of the given duration, for driving audio deltas.

    Its CONTENT is irrelevant — the handler only needs a delta to arrive so
    that _first_delta_sent_at is stamped and a truncation has something to
    measure against. 750ms because that is what call-20260818-1338 truncated
    to, and the whole point is reproducing that call.
    """
    return base64.b64encode(b"\xff" * int(8 * ms)).decode()


def usage(text_in=2000, cached=1887, audio_in=400, audio_out=800):
    return {"usage": {
        "input_token_details": {
            "text_tokens": text_in, "audio_tokens": audio_in,
            "cached_tokens": cached,
            "cached_tokens_details": {"text_tokens": cached, "audio_tokens": 0},
        },
        "output_token_details": {"text_tokens": 20, "audio_tokens": audio_out},
    }}


_ARTEFACTS = pathlib.Path(tempfile.gettempdir()) / "realtime-protocol-test"


async def run_call(script, out=None, connect_failures=0, answered_at=None):
    """Run one scripted call, return (messages_sent_to_oai, session_memory).

    ``out`` is an optional dict that receives the FakeTwilio under key
    "twilio". Half of the barge-in contract is what goes to Twilio — the
    `clear` that drains buffered audio — and that direction was unobservable
    from the two-tuple, so nothing had ever asserted on it.

    Fully offline. Two things are stubbed that otherwise reach the outside
    world on every run:

      * the Twilio REST client — the handler starts a call recording, which
        for a fake CallSid produced a 20404 against the LIVE api.twilio.com
        using real credentials, and printed a wall of traceback per scenario.
      * the artefact directories — save() wrote real WAVs and JSON into
        data/, mixing test runs in with genuine call records.
    """
    sent = []
    twilio = FakeTwilio([{"event": "start", "start": {"streamSid": "MZtest"}}])
    if out is not None:
        out["twilio"] = twilio
    captured = {}

    real_init = rw.RealtimeSession.__init__

    def spy_init(self, call_sid, doctor):
        real_init(self, call_sid, doctor)
        captured["session"] = self

    _ARTEFACTS.mkdir(parents=True, exist_ok=True)

    class _NoTwilio:
        """Stands in for twilio.rest.Client — never touches the network."""
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, _): return self
        def create(self, *a, **k):
            raise RuntimeError("Twilio disabled in tests")

    # `connect_failures` makes the first N handshakes raise, standing in for the
    # transient stall that killed call CAd1a20b on 2026-08-18 — the callee
    # answered and heard seventeen seconds of silence because one bad TCP setup
    # ended the call.
    _attempts = {"n": 0}

    def _connect(*a, **k):
        _attempts["n"] += 1
        if _attempts["n"] <= connect_failures:
            raise TimeoutError("timed out during opening handshake")
        return FakeConn(FakeOAI(script, sent))

    if out is not None:
        out["attempts"] = _attempts

    with mock.patch.object(rw.websockets, "connect", _connect), \
         mock.patch.object(rw, "audio_dir", lambda: _ARTEFACTS), \
         mock.patch.object(rw, "json_dir", lambda: _ARTEFACTS), \
         mock.patch.object(_rwsession, "audio_dir", lambda: _ARTEFACTS), \
         mock.patch.object(_rwsession, "json_dir", lambda: _ARTEFACTS), \
         mock.patch.dict("sys.modules",
                         {"twilio.rest": double(Client=_NoTwilio)}), \
         mock.patch.object(rw.RealtimeSession, "__init__", spy_init):
        doctor = Doctor(doctor_name="Dr. Jane Okafor",
                        hospital_name="Northside Medical Group",
                        specialization="Cardiology")
        await asyncio.wait_for(
            rw.handle_realtime(_fake(twilio), "CA000000000000000000000000testsid",
                               doctor, answered_at=answered_at),
            timeout=30,
        )
    session = captured.get("session")
    # Not Optional. Every caller reads .memory or .turns off this, and pyright
    # flagged forty of those as accesses on None. Asserting here turns a
    # would-be AttributeError three hundred lines away into one clear failure,
    # and lets the type checker see the rest of the file.
    assert session is not None, "handle_realtime never constructed a session"
    return sent, session


# ── Scenarios ─────────────────────────────────────────────────────────────────

HANDSHAKE = [{"type": "session.created"}, {"type": "session.updated"}]


def script_happy_path():
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done",
         "transcript": "Hi, good afternoon - this is an automated assistant calling from Forage AI..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Yes, this is Northside."},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Great - which location is Dr. Okafor practicing at?"},
        {"type": "response.done", "response": usage()},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "She's at the Northgate campus."},
        {"type": "response.function_call_arguments.done", "call_id": "c1",
         "name": "save_branch", "arguments": json.dumps({"branch": "Northgate Campus"})},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Perfect, thank you so much. Have a great day!"},
        {"type": "response.done", "response": usage()},
    ]


def script_silent_response():
    """A response that COMPLETES having produced no audio at all.

    This happened on a live call: the caller said the doctor had retired, the
    model returned a response with out_audio=0, and the line went quiet for 8.2
    seconds until the caller asked "are you there?" — which is what a person says
    to a call they think has dropped. Nothing is queued behind an empty response,
    so the silence lasts until the caller gives up and speaks.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is Sarah..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Actually, he is not working now. He's retired."},
        # No transcript, no audio, status completed.
        {"type": "response.done", "response": usage(1900, 1800, audio_out=0)},
    ]


def script_barge_in_before_first_audio():
    """The caller talks over a response that has not made a sound yet.

    _response_active was set on the FIRST AUDIO DELTA, never on
    response.created, which was not handled at all. Between the two there is
    real latency — 1.19s measured on call-20260818-1112 — and for that window
    the barge-in handler saw no active response and skipped: no
    response.cancel, no Twilio `clear`, no truncate, no BARGE-IN line.

    The response then died anyway, cancelled by OpenAI's own VAD, which is why
    that call logged two `[cancelled]` responses with out_audio=0 and not one
    barge-in. The visible cost was a LOST TURN: the agent was asked a question,
    its answer was killed before it made a sound, and the dead-air guards fired
    afterwards trying to account for the silence.
    """
    return HANDSHAKE + [
        {"type": "response.created"},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Hi, this is David, calling on behalf of..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "May I know why you are calling? Is it an emergency?"},
        # A response is created to answer that — and the caller keeps talking
        # before a single audio delta arrives.
        {"type": "response.created"},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.done",
         "response": {**usage(1900, 1800, audio_out=0), "status": "cancelled"}},
    ]


def script_cut_off_then_hello():
    """call-20260818-1338, verbatim: truncated to 750ms, caller says "Hello."

    The interruption path fired correctly and made the call worse. The caller
    heard three-quarters of a second, lost the thread, and said "Hello." The
    agent classified that as filler and asked its question again — burning an
    ask on a turn that had never been delivered.

    "Hello" after a cut is a REPAIR SIGNAL, not filler. The same word arriving
    cold means something else, and the distinguishing fact — that we truncated,
    and to how many milliseconds — is not in the transcript at all. It is in
    the process, so the recognition belongs in the process.
    """
    return HANDSHAKE + [
        {"type": "response.created"},
        {"type": "response.output_audio.delta", "delta": _b64_silence(),
         "item_id": "item_greet"},
        # Caller talks over it almost immediately -> our barge-in truncates.
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.done",
         "response": {**usage(1900, 0, audio_out=8), "status": "cancelled"}},
        # ...and what they say next is them checking the line is alive.
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Hello."},
        {"type": "response.done", "response": usage()},
    ]


def script_announce_then_rejected():
    """call-20260818-1613: told the caller it was saved, then it wasn't.

        16:14:19  "Thanks for checking — I'll save that and then we'll be
                   all set."                       <- spoken to a real person
        16:14:19  ⛔ BRANCH REJECTED: possibly the city restated

    A success message emitted before the operation that decides success —
    the same class as the lying console log fixed in 0c28baa, except this one
    goes down the phone line. That call recovered by accident; the same shape
    on a rejection that does not recover leaves a receptionist hanging up
    believing a location was recorded when nothing was written.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done",
         "transcript": "Hi, this is David — which branch is Dr. Okafor working out of?"},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "He is working at downtown branch Los Angeles."},
        # The agent announces the save FIRST...
        {"type": "response.output_audio_transcript.done",
         "transcript": "Thanks for checking — I’ll save that and then we’ll be all set."},
        # ...and only then does the tool run, and reject: a bare city restated.
        {"type": "response.function_call_arguments.done", "call_id": "c9",
         "name": "save_branch",
         "arguments": json.dumps({"branch": "Los Angeles", "city": "Los Angeles"})},
        {"type": "response.done", "response": usage()},
    ]


def script_server_side_cancel():
    """OpenAI cancels the response itself; we never sent response.cancel.

    interrupt_response defaults to true, so the server interrupts on caller
    speech independently of this module. When it wins the race there is no
    speech_started for us to act on before the fact, and until now nothing sent
    Twilio a `clear` — so audio already buffered there kept playing after
    generation had stopped.
    """
    return HANDSHAKE + [
        {"type": "response.created"},
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "response.created"},
        # No speech_started reaches us — the server acted on its own.
        {"type": "response.done",
         "response": {**usage(1900, 1800, audio_out=0), "status": "cancelled"}},
    ]


def script_rejected_response():
    """A response.create the API REJECTED, which must NOT be retried.

    conversation_already_has_active_response comes back as response.done with
    status=failed having consumed no input at all — zero text tokens and zero
    audio tokens, because it never ran. The empty-response recovery read that
    as dead air and created another response, which collided and failed in
    turn. On call-20260811-1640 that cascade held the line silent for 25
    seconds while the caller said "Hello?", "What do you want?" and "How was
    your day today?" into a call that never answered.

    The distinction that makes this recoverable: a response that genuinely ran
    and produced no audio HAS read the conversation and reports input tokens. A
    rejected one reports none.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is Sarah..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "response.done",
         "response": {**usage(0, 0, audio_in=0, audio_out=0), "status": "failed"}},
    ]


def script_repeats_itself():
    """One response emitting the same sentence twice, word for word.

    From call-20260813-1409: "could I just get the exact branch name or address
    so I don't save the wrong place?" arrived as two transcript items inside a
    single 10.65s response. Because it was one response, the re-ask gap guard
    measured 0.0s and had no following turn to correct — the duplication has to
    be caught where the transcript lands, not on the next turn.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done",
         "transcript": "Hi, this is Sarah..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Could I get the exact branch name or address?"},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Could I get the exact branch name or address?"},
        {"type": "response.done", "response": usage(1900, 1800)},
    ]


def script_hold_then_escalate():
    """The give-up directive fires, then the caller offers to go and check.

    From a live call, in this order:
        budget hit 4 asks -> "stop asking and escalate" injected
        CALLER: "can you please give me a minute? I just need to check"
        escalate(caller engaged but never provided a location)
    The most cooperative thing said on the call, answered by hanging up.

    Clearing the internal flag is not enough — the directive is already in the
    model's context and it will act on it. The escalation has to be blocked at
    the tool call.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Oh yeah, can you please give me a minute? I just need to check"},
        {"type": "response.function_call_arguments.done", "call_id": "c9",
         "name": "escalate",
         "arguments": json.dumps({"reason": "caller engaged but never provided a location"})},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Sure, no rush."},
        {"type": "response.done", "response": usage()},
    ]


def script_refusal():
    """Test line 5: 'I'm not allowed to give out that information.'"""
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, good afternoon..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "I'm not allowed to give out that information."},
        {"type": "response.function_call_arguments.done", "call_id": "c2",
         "name": "escalate",
         "arguments": json.dumps({"reason": "declined - hospital policy"})},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Completely understand - thanks for your time. Have a good day!"},
        {"type": "response.done", "response": usage()},
    ]


def script_identity_reask():
    """Receptionist asks who's calling MID-call, after the greeting already said it.

    This path was broken by a rule added for a good reason: "never reuse a
    sentence you have already said" plus "if you have already explained who you
    are, do not re-explain it" told the agent to brush off a second request for
    its identity — in the one template whose purpose is honest identification.

    The call must continue normally: no escalate, no save_branch, no hangup.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done",
         "transcript": "Hi, good afternoon - this is an automated assistant calling from Forage AI..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Yes, this is Northside."},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Great - which location is Dr. Okafor practicing at?"},
        {"type": "response.done", "response": usage()},
        # mid-call re-ask — the case that regressed
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Sorry, which company was that again?"},
        {"type": "response.output_audio_transcript.done",
         "transcript": "Of course - this is Forage AI, and I'm an automated assistant. "
                       "We collect and validate publicly available information about doctors."},
        {"type": "response.done", "response": usage()},
        # and again, plus how to reach us
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "And what's your callback number?"},
        {"type": "response.output_audio_transcript.done",
         "transcript": "I don't have a direct phone line, but you can reach us at "
                       "directory@forageai.com."},
        {"type": "response.done", "response": usage()},
    ]


def script_tool_fires_mid_question():
    """save_branch succeeds while the agent's last utterance was a QUESTION.

    A live call did exactly this: the agent asked "which office is Dr. Okafor
    working out of?", called save_branch in the same response, and the bridge
    hung up — the caller was answering a question into a dead line. The code
    treated "this response contained audio" as "the agent said goodbye".

    The call must request a proper closing instead of hanging up on a question.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, good afternoon..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Yes, she's at the Northgate campus."},
        # agent's turn ends in a question, yet the tool fires in the same response
        {"type": "response.output_audio_transcript.done",
         "transcript": "Which branch is Dr. Okafor working out of?"},
        {"type": "response.function_call_arguments.done", "call_id": "c9",
         "name": "save_branch", "arguments": json.dumps({"branch": "Northgate Campus"})},
        {"type": "response.done", "response": usage()},
    ]


def script_barge_in():
    """Caller interrupts while the agent is mid-sentence.

    This path was unreachable for the whole project: caller audio was dropped
    while agent_speaking was true, OpenAI's VAD only fires on audio it
    receives, so input_audio_buffer.speech_started never arrived and the
    barge-in handler was dead code. The agent talked over anyone who tried.

    With the gate open it fires, and cancelling must be followed by
    conversation.item.truncate — otherwise OpenAI's context keeps the whole
    generated response while the caller heard only the opening words, and the
    model later refers back to things nobody heard.
    """
    import base64
    chunk = base64.b64encode(b"\xff" * 800).decode()
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, good afternoon..."},
        {"type": "response.done", "response": usage(1900, 0)},
        # agent starts a long turn
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_abc"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_abc"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_abc"},
        # caller cuts in partway through
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Sorry, she's at the Northgate campus."},
        {"type": "response.done", "response": usage()},
    ]


def script_double_spoken_item(second_text: str = "Sure, no rush."):
    """ONE response that speaks the same short line twice — call-20260819-2044.

    The callee asked for a minute to look something up and heard "Sure, no
    rush. Sure, no rush." One response.done, one usage line, 2.85s of audio,
    and two `response.output_audio_transcript.done` events with identical text.
    Nothing in the codebase asked twice: the hold branch only stands the
    watchdog down, and a second response.create would have produced a second
    response.done. The model emitted two assistant items and both were spoken.

    The item_id is the only thing that separates them while the audio can still
    be stopped — the transcript arrives after the deltas are already on the
    wire.
    """
    chunk = _b64_silence(600)
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Okay, alright, give me a minute, let me pull that up."},
        {"type": "response.created"},
        # First spoken item — this one the caller is meant to hear.
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_one"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_one"},
        {"type": "response.output_audio_transcript.done", "item_id": "item_one",
         "transcript": "Sure, no rush."},
        # Second spoken item, same response, same words.
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_two"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_two"},
        {"type": "response.output_audio_transcript.done", "item_id": "item_two",
         "transcript": second_text},
        {"type": "response.done", "response": usage()},
    ]


def script_held_item_then_barge_in(second_text: str = "Of course, take your time."):
    """A second item is HELD, and then the caller talks over it.

    The dangerous half of hold-and-decide. Holding the audio buys the verdict
    its evidence, but it also means unplayed audio is sitting in a buffer when
    the caller starts speaking — and playing a turn somebody has already
    interrupted is worse than muting it ever was. The barge-in must throw it
    away, and no transcript arriving afterwards may resurrect it.
    """
    chunk = _b64_silence(600)
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "response.created"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_one"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_one"},
        {"type": "response.output_audio_transcript.done", "item_id": "item_one",
         "transcript": "Sure, no rush."},
        # Second item starts arriving — held, not sent.
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_two"},
        {"type": "response.output_audio.delta", "delta": chunk, "item_id": "item_two"},
        # ...and the caller talks over it before its transcript lands.
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.output_audio_transcript.done", "item_id": "item_two",
         "transcript": second_text},
        {"type": "response.done",
         "response": {**usage(1900, 0), "status": "cancelled"}},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Sorry, go on."},
    ]


def script_repeat_across_responses():
    """The same short line spoken in two SEPARATE responses.

    Distinct from script_double_spoken_item: both of these really are heard by
    the caller, so both are turns and both must survive into the artifact. The
    transcript cleanup in save() collapsed them — "Sure, no rush." is three
    words, under the <=4-word fragment threshold, so the merge replaced the
    pair with the second one and the duplicate-drop below would have removed it
    anyway. Both rules were written for a barge-in logging double-fire and
    cannot tell that from the model actually saying it twice.
    """
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Hang on, let me pull that up."},
        {"type": "response.output_audio_transcript.done", "item_id": "r1",
         "transcript": "Sure, no rush."},
        {"type": "response.done", "response": usage()},
        {"type": "response.output_audio_transcript.done", "item_id": "r2",
         "transcript": "Sure, no rush."},
        {"type": "response.done", "response": usage()},
    ]


def script_vetting_then_invitation():
    """call-20260819-2121, replayed. Four questions, no refusal, hung up on.

    The caller screened the call the way a front desk is supposed to — who is
    calling, is this about a patient, is it patient related — and then said
    "How can I help you?". The ask budget counted all four exchanges as asks
    that went unanswered, hit its limit, and told the agent to escalate. The
    agent closed the call on the most willing thing anyone said on it.

    Two separate defects, both asserted below: the budget must not spend on a
    caller who is questioning back, and `escalate` must be REFUSED outright
    when their last turn was an invitation — the directive is already in the
    model's context by then and clearing a flag cannot unsay it.
    """
    def vet(said):
        return [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": said},
        ]
    ask = "do you know which branch Dr. Okafor works out of?"
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done",
         "transcript": f"Hi, this is David, calling on behalf of Definitive "
                       f"Healthcare about a doctor listing, {ask}"},
        {"type": "response.done", "response": usage(1900, 0)},
    ] + vet("This is Northside Medical Group and I'm Varun. Sorry, who's calling again?") + [
        {"type": "response.output_audio_transcript.done",
         "transcript": f"Yeah, this is David, calling on behalf of Definitive Healthcare — {ask}"},
        {"type": "response.done", "response": usage()},
    ] + vet("Um, is this about a patient or something urgent?") + [
        {"type": "response.output_audio_transcript.done",
         "transcript": f"No, nothing urgent — I'm just trying to find out {ask}"},
        {"type": "response.done", "response": usage()},
    ] + vet("Is this about patient related?") + [
        {"type": "response.output_audio_transcript.done",
         "transcript": f"No, there's no patient involved — {ask}"},
        {"type": "response.done", "response": usage()},
    ] + vet("How can I help you?") + [
        # The model escalates anyway — which is what happened, because the
        # give-up directive was already in its context.
        {"type": "response.function_call_arguments.done", "call_id": "e1",
         "name": "escalate", "arguments": json.dumps(
             {"reason": "caller engaged but never provided a location"})},
        {"type": "response.done", "response": usage()},
    ]


def script_failed_response():
    """A response that FAILS, carrying its reason in status_details.

    call-20260819-2216 had seven of these and four stretches of 8-11 seconds
    where nobody on the call made a sound; the failures and the dead air line
    up one for one. The reason was in every event and nothing read it, so the
    cause was guessed at twice — first blamed on the tunnel, which Twilio's own
    recording then disproved by showing every agent block reaching the line
    within 0.4s of generation.
    """
    fail: dict = dict(usage())
    fail["status"] = "failed"
    fail["status_details"] = {
        "type": "failed",
        "error": {"type": "invalid_request_error",
                  "code": "conversation_already_has_active_response",
                  "message": "Conversation already has an active response"},
    }
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, this is David..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "Sorry, who is this?"},
        {"type": "response.done", "response": fail},
    ]


def script_invalid_branch():
    """A bare city must be rejected by save_branch and the call must continue."""
    return HANDSHAKE + [
        {"type": "response.output_audio_transcript.done", "transcript": "Hi, good afternoon..."},
        {"type": "response.done", "response": usage(1900, 0)},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "She works at the branch."},
        {"type": "response.function_call_arguments.done", "call_id": "c3",
         "name": "save_branch", "arguments": json.dumps({"branch": "branch"})},
        {"type": "response.done", "response": usage()},
    ]


# ── Assertions ────────────────────────────────────────────────────────────────

FAILURES = 0


def check(ok, label, detail=""):
    global FAILURES
    if not ok:
        FAILURES += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))


async def main():
    from agents.voice.templates import get_template
    from core.config import settings

    # THE SUITE TESTS THE CODE, NOT THE OPERATOR'S .env.
    #
    # `tpl` here and `objectives.default_objective()` both read
    # settings.call_template, so switching the live campaign to a different
    # template silently changed what ~20 checks were asserting — and they then
    # failed on content that was perfectly correct for the template they
    # happened to land on. A test whose meaning depends on an env var is a test
    # that will one day pass or fail for a reason nobody can see in the diff.
    #
    # Pinned to the branch script, which is what the template-specific
    # assertions below were written against. Every template is still covered:
    # the loops over TEMPLATES check identity, ceilings, greetings and
    # objectives for all of them, whichever one is configured to run.
    settings.call_template = "forage_data_collection"
    tpl = get_template(settings.call_template)

    print("\n" + "=" * 66)
    print("  SCENARIO 0 — a response that says nothing must not become dead air")
    print("=" * 66)
    _s_sent, _ = await run_call(script_silent_response())
    _s_creates = [m for m in _s_sent if m.get("type") == "response.create"]
    # One for the greeting, one to recover the empty response.
    check(len(_s_creates) >= 2,
          "empty response triggers a re-request instead of silence",
          f"{len(_s_creates)} response.create sent")
    print(f"  response.create sent: {len(_s_creates)} (greeting + recovery)")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a — a callee who never speaks must not stall the call")
    print("=" * 66)
    # Both greetings now end on a statement, which hands the turn over properly.
    # But a callee who simply waits produces no speech, so server VAD never
    # fires, no response is created, and neither pump runs again — the failure
    # is the ABSENCE of events, which nothing else in this file can see.
    class _WS:
        def __init__(self): self.sent = []
        async def send(self, raw): self.sent.append(json.loads(raw))

    def _wd_sess(**over):
        """A stand-in RealtimeSession for the watchdog, built in ONE place.

        These were four hand-rolled SimpleNamespaces. Adding _goodbye_retry_at
        to the real session broke each of them in turn with an AttributeError —
        the fixtures doing their job, four times, one round-trip each. A fake
        that mirrors a real object has to be built where the mirroring can be
        maintained. Resist "fixing" this class of break with getattr() in the
        watchdog instead: a missing attribute there is a real bug and should
        fail loudly rather than quietly read as None.
        """
        base = dict(done=False, _agent_quiet_since=0.0,
                    _silence_prompts_opening=0, _silence_prompts_midcall=0,
                    _response_active=False, turns=[], _goodbye_retry_at=None,
                    # backchannel state — the caller is not speaking by default
                    _hold_until=0.0, _claimed_done_at=0.0,
                    _playback_ends_at=0.0,
                    _owed_substance="", _owed_recovered=0,
                    _owed_directive_sent=False,
                    # The caps that let the recovery give up. Without them the
                    # recovery re-scheduled itself off its own muted second
                    # item — call-20260825-1435, which never terminated.
                    _owed_attempts={}, _owed_tried=0, owed_abandoned=[],
                    _claimed_done_nudged=False, memory={},
                    _caller_speaking_since=None, agent_speaking=False,
                    _backchannel_done_this_utterance=False,
                    _last_backchannel_at=0.0, _last_backchannel_clip=None,
                    _backchannels_sent=0, stream_sid="MZtest",
                    listen_enabled=asyncio.Event())
        base.update(over)
        s = double(**base)
        s.listen_enabled.set()
        return s

    _ws, _done = _WS(), asyncio.Event()
    # Mirrors the real RealtimeSession fields the watchdog reads. A hand-rolled
    # stand-in drifts from the object it stands in for — adding
    # _goodbye_retry_at to the session broke this fixture with an
    # AttributeError, which is the fixture doing its job. Resist "fixing" that
    # with getattr() in the watchdog: a missing attribute there is a real bug
    # and should fail loudly rather than read as None.
    _sess = _wd_sess()
    async def _keep_quiet():
        """Stand in for the agent speaking and getting nothing back, forever.

        The watchdog clears _agent_quiet_since after prompting so it cannot
        re-fire while the agent talks; response.done sets it again. Replaying
        that is what exercises the CAP — without it the timer never rearms and
        only one prompt ever fires, which would leave the runaway case untested.
        """
        for _ in range(6):
            await asyncio.sleep(0.4)
            if _sess._agent_quiet_since is None:
                _sess._agent_quiet_since = 0.0

    with mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _wd = asyncio.create_task(rw._silence_watchdog(_ws, _fake(_sess), _done))
        _rearm = asyncio.create_task(_keep_quiet())
        await asyncio.sleep(3.2)
        _done.set()
        await asyncio.wait_for(_wd, timeout=2)
        await _rearm
    _creates = [m for m in _ws.sent if m.get("type") == "response.create"]
    _nudges = [c["text"] for m in _ws.sent
               if m.get("type") == "conversation.item.create"
               for c in m["item"].get("content", []) if c.get("type") == "input_text"]
    check(len(_creates) >= 1, "silence triggers a response", f"{len(_creates)} sent")
    check(any("not said anything" in n for n in _nudges),
          "the model is told the callee has gone quiet")
    # Capped, or a silent line becomes the agent talking to itself forever.
    # turns=[] means the callee has never spoken, so this is the OPENING phase.
    check(_sess._silence_prompts_opening <= rw._MAX_SILENCE_PROMPTS,
          "opening silence prompts are capped", _sess._silence_prompts_opening)
    check(_sess._silence_prompts_midcall == 0,
          "opening silence does not spend the mid-call budget",
          _sess._silence_prompts_midcall)
    # ── The muted half must be SAID, not just logged ─────────────────────────
    # The one-item guard keeps the first item to produce audio and mutes the
    # rest, and it has to: the first is already on the wire when the second
    # appears. But the model does not reliably put the substance first, and
    # when it does not the guard deletes the answer and keeps the filler.
    #
    # call-20260820-1421, caller: "can you repeat that question please?"
    #     spoken : "Sure, I'll repeat it clearly."
    #     muted  : "I'm trying to find out which branch Dr. Okafor works out of."
    # They asked for the question and got a promise to give it. Seven seconds
    # of silence, the watchdog asked "Are you still with me?" — the wrong
    # sentence, the line was never the problem — they asked twice more and hung
    # up at 88s. The artifact said "2nd items muted 4" and nothing else.
    for _sp, _dr, _want in [
        # the call above, both halves
        ("Sure, I'll repeat it clearly.",
         "I'm trying to find out which branch Dr. Okafor works out of.", True),
        ("I'm calling on behalf of Definitive Healthcare, and this is an "
         "automated call.",
         "I'm calling on behalf of Definitive Healthcare, and this is an "
         "automated call. Could you tell me which branch Dr. Okafor sees "
         "patients at?", True),
        # ...and the cases that owe NOTHING. Saying these anyway is repetition,
        # which this project treats as what makes people hang up.
        ("Sure, no rush.", "Sure, no rush.", False),
        ("Got it, I just need the specific branch or site name in San Francisco.",
         "Got it, which branch in San Francisco is it?", False),
        # A REPHRASING of an ask they already heard owes nothing, and word
        # overlap cannot see that on its own — these two share almost no
        # vocabulary and are the same request. If the spoken half asked,
        # they have the question.
        ("Got it, which branch is she at?",
         "Got it, do you know which branch she works out of these days?",
         False),
        # ...but a watchdog line is not an ask, so the muted ask behind it
        # IS owed. Same call, 14:23:00.
        ("Are you still with me?",
         "Which branch does Dr. Okafor see patients at?", True),
    ]:
        check(rw._drop_lost_substance(_sp, _dr) == _want,
              f"owed after muting: {_want!s:5} for {_dr[:40]!r}")

    # And it has to actually be spoken. Owned by the watchdog for the same
    # reason the goodbye retry is — the drop is detected inside the event pump
    # while a response is still settling.
    _sess_ow = _wd_sess(
        _owed_substance="I'm trying to find out which branch Dr. Okafor works "
                        "out of.",
        _agent_quiet_since=None)          # not silence: the owed path must fire
    _ws_ow, _done_ow = _WS(), asyncio.Event()
    _wd_ow = asyncio.create_task(rw._silence_watchdog(_ws_ow, _sess_ow, _done_ow))
    await asyncio.sleep(1.4)
    _done_ow.set()
    await asyncio.wait_for(_wd_ow, timeout=2)
    _ow_nudges = [c["text"] for m in _ws_ow.sent
                  if m.get("type") == "conversation.item.create"
                  for c in m["item"].get("content", []) if c.get("type") == "input_text"]
    check(any("only the first half" in n for n in _ow_nudges),
          "the muted half is recovered on the next turn")
    check(any("which branch Dr. Okafor works out of" in n for n in _ow_nudges),
          "and it is quoted back verbatim, not paraphrased")
    check(any(m.get("type") == "response.create" for m in _ws_ow.sent),
          "and a response is created to speak it")
    check(_sess_ow._owed_tried == 1
          and sum(_sess_ow._owed_attempts.values()) == 1,
          f"the attempt is counted where it is SPENT ({_sess_ow._owed_tried} "
          f"this call)",
          "counting on the recovery landing instead would never count the "
          "case that matters — a recovery muted exactly like the turn that "
          "created the debt, which is the livelock on call-20260825-1435")
    check(_sess_ow._owed_substance == "" and _sess_ow._owed_recovered == 1,
          "cleared once said, so it cannot fire twice",
          f"{_sess_ow._owed_recovered} recovered")

    # ── ...and it must not claim success when the create is REFUSED ──────────
    # The first cut cleared the text, counted a recovery and printed "saying it
    # now" all BEFORE _create_response, which can decline while audio is still
    # playing out. It declined on the very next live call: call-20260820-1440
    # detected the owed text at t=45.0s with the previous reply running to
    # t=45.86s, so nothing was created and the owed half was dropped — while
    # the log said it had been said. The false-save shape exactly: success
    # reported before the operation that decides it.
    #
    # Driven with audio still queued, which is the condition that broke it.
    _sess_rf = _wd_sess(
        _owed_substance="I'm trying to find out which branch she works out of.",
        _agent_quiet_since=None,
        _playback_ends_at=time.monotonic() + 30)     # queue will not drain
    _ws_rf, _done_rf = _WS(), asyncio.Event()
    _wd_rf = asyncio.create_task(rw._silence_watchdog(_ws_rf, _sess_rf, _done_rf))
    await asyncio.sleep(1.4)
    _done_rf.set()
    await asyncio.wait_for(_wd_rf, timeout=2)
    check(not any(m.get("type") == "response.create" for m in _ws_rf.sent),
          "no response is created while the queue is still playing out")
    check(_sess_rf._owed_substance != "" and _sess_rf._owed_recovered == 0,
          "and the owed text is KEPT, not counted as recovered",
          f"owed={_sess_rf._owed_substance[:24]!r} recovered={_sess_rf._owed_recovered}")
    # Retrying must not re-inject the directive on every tick — the model would
    # be told the same thing several times over.
    _injected = [c["text"] for m in _ws_rf.sent
                 if m.get("type") == "conversation.item.create"
                 for c in m["item"].get("content", []) if c.get("type") == "input_text"]
    check(len([n for n in _injected if "only the first half" in n]) == 1,
          "the directive is injected once across many refused retries",
          f"{len(_injected)} injected over ~{int(1.4 / 0.25)} ticks")

    # And it must stand down the moment they speak.
    _sess2 = _wd_sess(_agent_quiet_since=None)
    _ws2, _done2 = _WS(), asyncio.Event()
    with mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _wd2 = asyncio.create_task(rw._silence_watchdog(_ws2, _sess2, _done2))
        await asyncio.sleep(1.2)
        _done2.set()
        await asyncio.wait_for(_wd2, timeout=2)
    check(not _ws2.sent, "no prompt while the caller has the turn",
          f"{len(_ws2.sent)} sent")

    # A response the VAD started in the same tick is already on its way. Sending
    # a second response.create raises conversation_already_has_active_response,
    # which the error handler logs and swallows — invisible. _response_active was
    # a local in _oai_to_twilio and the watchdog runs in a different task, so it
    # could not see it at all.
    _ws3, _done3 = _WS(), asyncio.Event()
    _sess3 = _wd_sess(_response_active=True)
    with mock.patch.object(rw, "_SILENCE_PROMPT_FIRST", 0.0), \
         mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _wd3 = asyncio.create_task(rw._silence_watchdog(_ws3, _sess3, _done3))
        await asyncio.sleep(1.2)
        _done3.set()
        await asyncio.wait_for(_wd3, timeout=2)
    check(not _ws3.sent,
          "no response.create while one is already generating",
          f"{len(_ws3.sent)} sent")

    # Two thresholds. Mid-conversation a pause is someone thinking, and seven
    # seconds of thinking room is right. Straight after the opening line there is
    # nothing to think about, and seven seconds of dead air on a cold call is
    # when people hang up.
    check(rw._SILENCE_PROMPT_FIRST < rw._SILENCE_PROMPT_AFTER,
          "first silence is given less rope than a mid-call one",
          f"{rw._SILENCE_PROMPT_FIRST}s vs {rw._SILENCE_PROMPT_AFTER}s")
    # The cap is for the CALL. Resetting it whenever the caller spoke meant a
    # callee who says "hello?" and nothing else could be prompted forever.
    _src = _PKG_SRC
    # Absence of one exact spelling is the weaker claim, and it passes by
    # finding nothing: reformat the line, drop the spaces, reset by another
    # route, and the check still reports success. The claim actually being made
    # is about a CATEGORY — no assignment anywhere puts this counter back to
    # zero except the one in __init__ — so enumerate every assignment and judge
    # them, rather than searching for the one spelling of the mistake we
    # happened to make once.
    # \w* so the family is covered, not one member: the counter was split into
    # _silence_prompts_opening and _silence_prompts_midcall, and a pattern
    # naming only the old field would have found zero assignments and reported
    # a population of nothing. The >= 2 guard below is what makes that loud
    # instead of a silent pass.
    #
    # IF THIS TEST FAILS, CONSIDER THAT THE ASSERTION MAY BE THE THING THAT IS
    # WRONG. It encodes a judgement — "zeroing belongs only in __init__" — that
    # is true today and need not stay true. An earlier version asserted
    # `len(zeroing) == 1`, and splitting the counter by phase gave __init__ two
    # perfectly legitimate declarations and turned the test red. The count was
    # never the claim; where the zeroing happens is. Written as a count, the
    # next person reads a red test and "fixes" correct code.
    _sp_lines = [l for l in _src.splitlines()
                 if re.search(r"_silence_prompts\w*\s*(?::[^=\n]+)?\s*(?:\+=|-=|=)"
                              r"\s*[^\n=]", l) and not l.strip().startswith("#")]
    check(len(_sp_lines) >= 2,
          "found the silence-budget assignments to judge",
          f"{len(_sp_lines)} assignments")
    _sp_zero = [l.strip() for l in _sp_lines
                if re.search(r"=\s*0\s*$", l)]
    # Declarations look like `self._x: int = 0` in __init__. A mid-call reset
    # goes through the session object as `sess._x = 0`. That distinction, not a
    # count, is the actual claim — and it survives adding a third counter.
    check(all(l.startswith("self.") for l in _sp_zero),
          "only __init__ declarations zero the silence budget, never a reset",
          f"{_sp_zero}")
    check(not any(re.search(r"\bsess\._silence_prompts\w*\s*=\s*0", l)
                  for l in _sp_lines),
          "no mid-call reset of the silence budget through the session")
    # check() reports and continues, so a line that does not parse must not be
    # dereferenced anyway — that is a crash mid-suite, not a failed assertion.
    _sp_m = [re.search(r"=\s*([^\n=]+)$", l) for l in _sp_lines]
    check(all(m is not None for m in _sp_m),
          "every silence-budget assignment parsed",
          f"{sum(m is None for m in _sp_m)} unparsed")
    _sp_vals = [m.group(1).strip().rstrip(",)") for m in _sp_m if m]
    check(all(v in ("0", "1") for v in _sp_vals),
          "silence budget is only ever initialised or incremented", f"{_sp_vals}")

    # The silence clock must start when the callee STOPS HEARING us, not when
    # response.done arrives. Generation runs faster than realtime, so the old
    # code counted the agent's own voice as the callee's silence. Measured on
    # call-20260811-1649: the watchdog reported 3.5s before "Are you still with
    # me?" when the real gap was 1.41s, and 7.0s before the goodbye when the
    # real gap was 2.45s — the call hung up on someone mid-breath. The error
    # scales with clip length, so the longest turns were cut off hardest.
    #
    # A quiet_since in the FUTURE is how "audio is still playing" is
    # represented, so the watchdog must not prompt while that is true.
    _ws4, _done4 = _WS(), asyncio.Event()
    _sess4 = _wd_sess(_agent_quiet_since=time.time() + 1.5)
    with mock.patch.object(rw, "_SILENCE_PROMPT_FIRST", 0.0), \
         mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _wd4 = asyncio.create_task(rw._silence_watchdog(_ws4, _sess4, _done4))
        await asyncio.sleep(0.9)
        _done4.set()
        await asyncio.wait_for(_wd4, timeout=2)
    check(not _ws4.sent,
          "no prompt while the agent's audio is still playing out",
          f"{len(_ws4.sent)} sent")

    # Guard the fix itself: the bare assignment is what caused it.
    check("_agent_quiet_since = time.time()\n" not in _src,
          "silence clock is never set to bare response.done time")
    check("_agent_quiet_since = time.time() + _playback_remaining" in _src,
          "silence clock is offset by the audio still to play")

    # The API-error branch prints rather than logs, for consistency with the
    # rest of the module and for flush=True.
    #
    # CORRECTED: this used to say the errors were invisible because "nothing
    # configures logging for the uvicorn process". That was wrong. With no
    # config, Python's lastResort handler prints WARNING and above to stderr,
    # so log.error was always reaching the terminal — which is also why
    # twilio_worker's log.warning about a missing signature header WAS visible
    # in live output. INFO and DEBUG are the levels that vanish, and that is
    # what actually hid the call outcome in the /status handler. The evidence
    # for the 25s of dead air is the [failed] responses reporting in_text=0
    # in_audio=0, not a missing error line.
    #
    # The assertion below is still worth keeping, and still a CATEGORY rather
    # than one message: absence of an exact log.error string proves nothing,
    # since rewording it would pass. Four log.error calls elsewhere in the
    # module are legitimate, so a module-wide count would be aimed wrong; scope
    # it to the branch.
    _err_lines = _src.splitlines()
    _err_start = next((i for i, l in enumerate(_err_lines)
                       if 'event_type == "error"' in l), None)
    check(_err_start is not None, "found the API-error branch in the module")
    # check() reports and CONTINUES, so this has to guard for itself — the
    # slice below on a None index is a crash, not a failed assertion.
    assert _err_start is not None
    _err_indent = len(_err_lines[_err_start]) - len(_err_lines[_err_start].lstrip())
    _err_body = []
    for _l in _err_lines[_err_start + 1:]:
        if _l.strip() and (len(_l) - len(_l.lstrip())) <= _err_indent:
            break
        _err_body.append(_l)
    check(len(_err_body) >= 5,
          "API-error branch body extracted, not empty",
          f"{len(_err_body)} lines")
    check(not any(re.search(r"\blog\.\w+\(", _l) for _l in _err_body),
          "nothing on the API-error branch goes to the unconfigured logger")
    check(any("print(" in _l for _l in _err_body),
          "the API-error branch prints, so errors reach the terminal")

    # Identity questions must be recognised, and a branch question must not be
    # mistaken for one. On call-20260811-1649 "Hello, may I ask who is
    # speaking?" was answered with "Sorry, I didn't catch that — could you say
    # the branch name again?", which dodged the question and spent an ask from
    # the budget doing it.
    for _q in ("Hello, may I ask who is speaking?", "who are you",
               "Who is this?", "where are you calling from",
               "are you a robot", "who gave you this number"):
        check(bool(rw._IDENTITY_ASK.search(_q)),
              f"identity question recognised: {_q[:34]!r}")
    for _q in ("which branch is he at", "what location does she use",
               "who does he see on Tuesdays"):
        check(not rw._IDENTITY_ASK.search(_q),
              f"not mistaken for an identity question: {_q[:34]!r}")
    # THE CONTRACTION. The pattern was `who (is|are|'s) …` with a literal
    # space, so it needed "who 's" and never matched "who's" — the way the
    # question is actually asked. Every probe above avoided the contraction, so
    # the gap survived a suite that looked like it covered this.
    #
    # call-20260820-1440: "Sorry, who's calling again?" did not match, the
    # identity nudge never went out, and _is_reintroduction then flagged the
    # correct answer as a re-introduction. The detector that should have fired
    # did not; the one that should not have, did.
    for _q in ("Sorry, who's calling again?", "who's this?", "Who's speaking?",
               "who's calling"):
        check(bool(rw._IDENTITY_ASK.search(_q)),
              f"contraction recognised: {_q[:34]!r}")

    # ── Answering a direct WHO is not a re-introduction ──────────────────────
    # _is_reintroduction fires on self-name + org, which is exactly the correct
    # answer to "who's calling?" — and the prompt's own EXCEPTION requires it:
    # identity facts get repeated every time they are asked. Flagging it told
    # the model to stop doing the one thing it had just got right.
    #
    # The detector's docstring argued it could not key off "did they ask who I
    # am", because the case it was built for had a mis-transcription the model
    # read as an identity question. That held while _IDENTITY_ASK could not see
    # the commonest phrasing. It does not hold now.
    _reintro_src = _PKG_SRC
    _rb = _reintro_src[_reintro_src.find("# Re-introduction: the greeting delivered"):][:2200]
    check("_IDENTITY_ASK.search(_prev_caller)" in _rb,
          "the re-introduction guard consults the identity detector")
    check("not _answered_who" in _rb,
          "and stands down when the caller just asked who is calling")
    # Only the turn IMMEDIATELY before counts — an identity question four turns
    # back does not license re-delivering the greeting now, which is the
    # failure the guard exists for.
    check("next((t.text for t in reversed(sess.turns)" in _rb,
          "and only the most recent caller turn licenses it")
    # The detector itself is unchanged: still a positive test on self-name+org.
    check(rw._is_reintroduction(
              "Oh, sorry Varun — I'm David, calling on behalf of Definitive "
              "Healthcare.", "David", "Definitive Healthcare"),
          "the detector still recognises the greeting formula itself")

    # The re-ask gap has to be long enough to catch the observed case: two asks
    # 3s apart, the second landing 0.14s after the agent's own audio ended.
    check(rw._MIN_REASK_GAP_S > 3.0,
          "re-ask gap covers the 3s badgering seen on call-20260811-1649",
          f"{rw._MIN_REASK_GAP_S}s")

    # Both new guards nudge at most once. A second copy of a directive the
    # model already ignored is just context it pays for twice.
    for _flag in ("_reask_nudged", "_identity_nudged"):
        check(f"sess.{_flag} = True" in _src, f"{_flag} is one-shot")

    # A transcription hint is a PROMPT: whatever is in it can come back out as
    # transcript. The hint used to list complete caller responses under "Likely
    # phrases", and "Yes, speaking" — the second item on that list — was
    # transcribed four times in one call, on audio measured at 0.45-0.69 peak,
    # i.e. a clean line. A hint may supply proper nouns the model would mangle;
    # it must not supply whole utterances.
    from agents.voice.templates import _US_TRANSCRIBE_HINT as _hint
    for _phrase in ("yes, speaking", "hold on", "one moment", "let me check",
                    "let me transfer you", "this is", "not available",
                    "we can't share that", "he practices at"):
        check(_phrase not in _hint.lower(),
              f"hint does not supply the caller's line: {_phrase!r}")
    check("likely phrases" not in _hint.lower(),
          "hint has no complete-utterance section at all")
    # Positive control: the vocabulary the hint legitimately exists for is
    # still there, so gutting it fails rather than passing this section.
    #
    # NARROWED 2026-08-20 to location words only. It used to require health
    # systems here too — "Kaiser Permanente", "Cleveland Clinic" — and those
    # are now deliberately absent. A controlled A/B on identical audio (Arm A)
    # showed the list was the SOURCE of the fabrications: 0.7s of near-silence
    # returned "Hello, this is the Methodist Hospital. How may I assist you?"
    # with the list present, and single non-English tokens without it. Arm C
    # then showed removing it costs nothing measurable on real branch audio —
    # branch names 7/11 -> 9/11, digits 8/11 -> 9/11, over identical bytes.
    # RETIRED 2026-08-26, and the argument is the one recorded directly above
    # for the health-system list, applied to what was left. That narrowing was
    # justified by a controlled A/B showing the list was the SOURCE of the
    # fabrications and cost nothing to remove. The location and scheduling
    # words then did the same thing on 2026-08-26: call-1633 collected NOTHING
    # in 35s because the caller's first turn came back as this vocabulary, and
    # call-1625 got identity only in 94s.
    #
    # THE ASSERTION FLIPS BUT THE DETECTORS MUST NOT. What primes the
    # transcriber is now nothing; what the guards recognise is unchanged,
    # because the text moved to _RETIRED_HINT_TEXT and both _strip_hint_run and
    # _hint_vocabulary read it. Checking the retired text still carries these
    # words is what stops a future tidy-up deleting the detectors' evidence
    # along with the dead constant.
    check(_hint == "", "the template primes the transcriber with nothing",
          f"{_hint[:60]!r}")
    for _noun in ("campus", "medical center", "boulevard", "street", "clinic",
                  "waitlist", "referral"):
        check(_noun.lower() in rw._RETIRED_VOCAB_TEXT.lower(),
              f"the retired vocabulary still carries {_noun!r} for the detectors",
              "deleting it here disarms the guard that catches it coming back")
    # And the priming vocabulary is GONE from what we send. Asserted as an
    # absence only because the corresponding presence is asserted on the
    # DETECTOR below — the pair is what makes this meaningful rather than a
    # check that passes by finding nothing.
    for _primer in ("Mercy", "Baptist", "Mayo", "Northwell", "receptionist"):
        check(_primer.lower() not in _hint.lower(),
              f"hint no longer primes: {_primer!r}")
    check(len(_hint) < 200,
          "hint is location vocabulary only", f"{len(_hint)} chars")

    # ── ...and the DETECTOR keeps that vocabulary, independently ─────────────
    # Deleting the list from the hint disarmed the fabrication detector with
    # it: _hint_proper_nouns read the live hint, so it went from 21 names to
    # zero and stopped recognising every fabrication on record. The two jobs
    # were never the same job — the hint is what we SEND, the vocabulary is
    # what we RECOGNISE — so they are now separate constants.
    for _name in ("mercy", "mayo", "northwell", "baptist", "methodist",
                  "sutter", "providence", "cleveland", "kaiser"):
        check(_name in rw._FABRICATION_VOCAB,
              f"detector still knows the fabrication vocabulary: {_name!r}")
    check(len(rw._FABRICATION_VOCAB) >= 20,
          "detector vocabulary survived the hint being minimised",
          f"{len(rw._FABRICATION_VOCAB)} names")
    # Every observed fabrication must still be recognised with the NEW hint in
    # force — this is the regression the decoupling exists to prevent.
    for _fab in ("Mercy Hospital",
                 "Hello, I need to schedule an appointment at the Mayo",
                 "across from the Northwell campus",
                 "Hi, this is Mercy Hospital. How may I help you?",
                 "Okay, I'm looking at her profile now. Baptist"):
        check(rw._reads_as_hint_vocabulary(_fab, _hint),
              f"fabrication still caught under the minimal hint: {_fab[:38]!r}")
    # ...and real caller answers still are not.
    for _real in ("It's the Mission Bay clinic, 1825 Fourth Street",
                  "She works at the Abadan branch", "Northgate", "Yes."):
        check(not rw._reads_as_hint_vocabulary(_real, _hint),
              f"real answer not mistaken for fabrication: {_real[:38]!r}")

    # Re-introduction: the greeting delivered a second time. The prompt has had
    # a rule against this since templates.py:296 and it lost on turn TWO of
    # call-20260813-1409. Detection keys off the greeting FORMULA — self-naming
    # plus the org — not a bare mention of the org, because naming the org is
    # the correct answer when someone genuinely asks who is calling.
    for _t, _exp, _why in (
        ("Sure, let me explain who I am and why I'm calling. I'm David, "
         "calling on behalf of Definitive Healthcare.", True,
         "the observed turn-2 failure"),
        ("Yes — I'm an automated system from Definitive Healthcare.", False,
         "employment claim is a different defect, not a re-introduction"),
        ("I just need to know which branch Dr. Okafor works at.", False,
         "the correct answer to WHY"),
        ("We keep the Definitive Healthcare directory updated.", False,
         "org named without self-naming"),
        ("I'm David.", False, "name without org"),
    ):
        check(rw._is_reintroduction(_t, "David", "Definitive Healthcare") == _exp,
              f"re-introduction detector: {_why}")

    # The employment claim. The agent calls ON BEHALF OF the client and is not
    # employed by them; "from {org}" is a false statement about who is on the
    # phone, made to a medical office. It was removed from every greeting and
    # asserted against per-template — and came back out of the model mid-call
    # on call-20260813-1409 at 14:11:33, because those assertions check
    # build_greeting() and nothing watched what the model actually said. Same
    # three forms the greeting test uses, so the runtime and artifact checks
    # cannot disagree about what the claim is.
    for _t, _exp, _why in (
        ("Yes — I'm an automated system from Definitive Healthcare.", True,
         "the observed mid-call claim"),
        ("Hi, this is David, calling on behalf of Definitive Healthcare.", False,
         "the greeting itself must not trip it"),
        ("I work with Definitive Healthcare on their listings.", True, "'with'"),
        ("I am at Definitive Healthcare right now.", True, "'at'"),
        ("We update the Definitive Healthcare directory.", False,
         "org named without an employment claim"),
    ):
        check(rw._claims_employment(_t, "Definitive Healthcare") == _exp,
              f"employment claim: {_why}")
    # The greeting for EVERY template must survive its own runtime detector —
    # the artifact check and the behaviour check agreeing on one example.
    from agents.voice.templates import TEMPLATES as _EMP_ALL
    _emp_probe = Doctor(doctor_name="Dr. Jane Okafor",
                        hospital_name="Northside Medical Group")
    for _name, _t2 in _EMP_ALL.items():
        check(not rw._claims_employment(
                  _t2.build_greeting(_emp_probe, org=settings.org_name),
                  settings.org_name),
              f"{_name}: greeting does not trip the employment detector")
    # And the INSTRUCTIONS must not mandate the claim either. templates.py:411
    # used to tell the model to say "you're an automated system from your
    # organisation" — so the false claim on call-20260813-1409 was the model
    # obeying the prompt, not departing from it. A runtime detector fighting
    # its own instructions would have nudged against the script mid-call.
    for _name, _t2 in _EMP_ALL.items():
        _flat = " ".join(_t2.instructions.split())
        # Narrow to the MANDATE, not the substring. "from your organisation"
        # alone also matches the section header "# Identity — you present as a
        # person from your organisation", which instructs nothing. An absence
        # check aimed one word too wide fails on correct code, which is the
        # same defect as one aimed too narrow — it just fails loudly instead of
        # quietly.
        check("system from your organisation" not in _flat,
              f"{_name}: instructions do not mandate an employment claim")
        # And the positive: the rule telling it which form to use is present.
        # Case-insensitive — the prompt shouts it in one template and not the
        # other, and the claim is that the rule exists, not how it is cased.
        check("on behalf of" in _flat.lower(),
              f"{_name}: instructions specify the on-behalf-of form")

    # Hint-echo. The grounding check trusts caller turns, but a caller turn is a
    # model's guess at the caller and the hint is a prompt to that model. Two
    # independent signals must BOTH fail before a term is discounted — a bare
    # one-word answer on strong audio is legitimate, and so is a quiet turn that
    # carries surrounding words.
    from core.models import TranscriptTurn as _TT
    for _turns, _args, _exp, _why in (
        ([_TT(role="caller", text="Mercy", audio_rms=0.002)],
         {"branch": "Mercy"}, True, "bare hint word on dead air is rejected"),
        ([_TT(role="caller", text="Northgate", audio_rms=0.05)],
         {"branch": "Northgate"}, False,
         "bare one-word answer on strong audio still grounds"),
        ([_TT(role="caller", text="she's at the Mercy campus", audio_rms=0.002)],
         {"branch": "Mercy"}, False,
         "quiet turn with surrounding words still grounds"),
        ([_TT(role="caller", text="Mercy", audio_rms=None)],
         {"branch": "Mercy"}, False,
         "unmeasured audio gets the benefit of the doubt"),
        ([_TT(role="caller", text="hello", audio_rms=0.05)],
         {"branch": "Northgate"}, True,
         "a term never said at all is still rejected"),
    ):
        _got = bool(rw._ungrounded_terms(_args, double(turns=_turns)))
        check(_got == _exp, f"hint-echo: {_why}")

    # ── Repeats are clauses, not only sentences ──────────────────────────────
    # call-20260818-1613 contained a 45-character EXACT repeat and scored
    # repeated_sentences: 0. Neither turn had an internal sentence break, so
    # each counted as one "sentence", the two differed, and nothing was
    # counted. The repeated part is the clause after the em-dash — and that is
    # not an accident of this call: the prompt's own turn shape is "React, THEN
    # say the thing", which produces `reaction — ask`, so the ask is the unit
    # that repeats and it never sits at a sentence boundary.
    #
    # This metric is one of the figures used to compare calls, so a clean
    # number on a dirty call weakened every comparison drawn from it.
    _mkr = lambda r, x: double(role=r, text=x)
    _r1 = ("Hi, this is David, calling on behalf of Definitive Healthcare "
           "about a doctor listing — which branch is Dr. Okafor working out of?")
    _r3 = "I can hear you now — which branch is Dr. Okafor working out of?"
    _rm = rw.conversation_metrics([_mkr("agent", _r1), _mkr("caller", "Hello?"),
                                   _mkr("agent", _r3)])
    check(_rm["repeated_sentences"] == 1,
          "a repeated clause inside differing turns is caught",
          f"{_rm['repeated_sentences']}")
    # Punctuation is not the thing being measured: "...out of?" and
    # "...out of." are the same thing said twice.
    _rp = rw.conversation_metrics([
        _mkr("agent", "So — which branch is she working out of?"),
        _mkr("caller", "hm"),
        _mkr("agent", "Right — which branch is she working out of.")])
    check(_rp["repeated_sentences"] == 1,
          "trailing punctuation does not hide a repeat",
          f"{_rp['repeated_sentences']}")
    # KNOWN LIMIT, deliberate: splitting is on dashes/semicolons/colons, which
    # is the shape the prompt's "React, THEN say the thing" actually produces.
    # A comma-led restatement is NOT caught. Splitting on commas would
    # fragment almost every turn and manufacture false repeats, so the gap is
    # accepted rather than closed. Asserted so it stays a decision instead of
    # quietly becoming a surprise.
    _rk = rw.conversation_metrics([
        _mkr("agent", "So — which branch is she working out of?"),
        _mkr("caller", "hm"),
        _mkr("agent", "Right, which branch is she working out of.")])
    check(_rk["repeated_sentences"] == 0,
          "comma-led restatements are a known, accepted blind spot",
          f"{_rk['repeated_sentences']}")
    # A whole sentence said twice is ONE repetition, not one for the sentence
    # plus one for each clause inside it. Counting clauses INSTEAD of sentences
    # was the first attempt and it silently dropped a real repeat when a short
    # sentence split into sub-threshold clauses.
    _dbl = "Thanks for that — I'll get it noted down now."
    _rd = rw.conversation_metrics([_mkr("agent", _dbl), _mkr("caller", "ok"),
                                   _mkr("agent", _dbl)])
    check(_rd["repeated_sentences"] == 1,
          "a doubled sentence counts once, not once per clause",
          f"{_rd['repeated_sentences']}")
    # And it must not fire on ordinary distinct turns.
    _rc = rw.conversation_metrics([
        _mkr("agent", "Got it — which branch is she at?"),
        _mkr("caller", "the north one"),
        _mkr("agent", "Perfect, thanks — I'll note the north site down.")])
    check(_rc["repeated_sentences"] == 0,
          "distinct turns are not flagged as repeats",
          f"{_rc['repeated_sentences']}")

    # ── The measurement the guards rest on ───────────────────────────────────
    # audio_rms is the sole input to _is_hint_echo, and it was being OVERWRITTEN
    # rather than accumulated: set at every speech_stopped, consumed only when
    # the transcript arrives, and transcription lags the VAD. A second segment
    # on trailing silence between the real speech and its transcript replaced
    # the real measurement with the silence.
    #
    # call-20260818-1613: the caller's "Yes, yes." recorded audio_rms=0.0025
    # while Twilio's own caller channel shows that utterance at ~0.13 peak — a
    # 50x under-report, erring toward calling real speech silence, which is the
    # direction that throws away genuine answers. Two conclusions drawn from
    # that number this week were wrong.
    _ms = rw.RealtimeSession("CA000000000000000000000000rmsacc",
                             Doctor(doctor_name="Dr. R"))
    check(_ms.take_utterance_rms() == (None, 0),
          "no segments yet -> nothing to report")
    _ms.note_utterance_rms(0.1300)      # the caller actually speaking
    _ms.note_utterance_rms(0.0025)      # trailing silence, segmented separately
    _got, _segs = _ms.take_utterance_rms()
    check(_got == 0.1300 and _segs == 2,
          "the loudest segment survives a trailing silent one", f"{_got}, {_segs} segs")
    check(_ms.take_utterance_rms() == (None, 0),
          "and consuming it clears the accumulator")
    # Order must not matter — silence can precede the speech just as easily.
    _ms.note_utterance_rms(0.0025); _ms.note_utterance_rms(0.1300)
    check(_ms.take_utterance_rms()[0] == 0.1300,
          "a leading silent segment does not win either")
    # Zero/None are not measurements and must not create a segment, or a
    # genuinely unmeasured turn would look measured and lose its benefit of the
    # doubt in _is_hint_echo.
    _ms.note_utterance_rms(0.0); _ms.note_utterance_rms(None)
    check(_ms.take_utterance_rms() == (None, 0),
          "an unmeasurable segment stays unmeasured, not zero")

    # ── The worked examples must not teach the bare demand ───────────────────
    # On call-20260819-1716 the agent's second turn was
    #   "Hi Priya, this is David calling on behalf of Definitive Healthcare
    #    — which branch is Dr. Okafor working out of?"
    # which is an order with an introduction in front of it. The model was not
    # disobeying: Shape Of A Turn carried
    #       Right: "Got it — which branch is she at?"
    # — the same structure, marked Right — while the prose twenty lines earlier
    # says "You are asking a favour of someone at work: 'do you know...'".
    # Prose against a worked example is the ONE MOVE PER TURN contradiction
    # again, and the example wins every time.
    #
    # The old annotation also mis-diagnosed it: "no reaction — an
    # interrogation" blames the missing opener, when the problem is the ask.
    # A reaction in front of a demand is still a demand.
    from agents.voice.templates import TEMPLATES as _ALL6
    for _n6, _t6 in _ALL6.items():
        _f6 = " ".join(_t6.instructions.split())
        check('Right: "Got it — do you know which branch she\'s at?"' in _f6,
              f"{_n6}: the Right example softens the ask, not just the opener")
        check('Wrong: "Got it — which branch is she at?"' in _f6,
              f"{_n6}: and the cushioned demand is explicitly marked Wrong")
        # The prose rule it has to agree with.
        check("asking a favour of someone at work" in _f6,
              f"{_n6}: the prose rule it now matches is still present")

    import pathlib as _plb
    # ── The utterance must be cut by OpenAI's clock, not ours ────────────────
    # ROOT CAUSE, found 2026-08-19. The utterance was sliced from
    # len(_caller_pcm) at the moment speech_started ARRIVED. That event is
    # generated in the US and reaches India up to a second late, by which time
    # a short utterance is already fully buffered — so the slice held only the
    # silence after it.
    #
    # The signature is exact and appeared twice: audio_rms 0.000244140625,
    # which is what a buffer of mu-law 0xFF decodes to. On call-20260819-2006
    # that was recorded for a turn whose Twilio caller channel measures 0.2425.
    # Two "fixes" before this one patched around it.
    _bpms = rw._wire_bytes_per_ms()
    _silence = b"\xff" * int(4000 * _bpms)        # 4s of mu-law silence
    _loud = bytes([0x00, 0x80] * int(2000 * _bpms // 2))   # 2s of loud audio
    class _S:  # minimal session: the MIRROR and the listen offset are read
        def __init__(self, chunks, listen_start_bytes=0):
            # _caller_oai_pcm, not _caller_pcm: the slicer indexes what
            # OpenAI received. Keeping both names bound to the same list
            # here would hide the very divergence this split exists for,
            # so the recording buffer is deliberately left empty.
            self._caller_oai_pcm = chunks
            self._caller_pcm = []
            self._listen_start_bytes = listen_start_bytes
    # Speech at 1000-3000ms, then silence. The old code, marking the position
    # late, would have taken the tail.
    _sess_sl: Any = _S([_silence[:int(1000*_bpms)], _loud, _silence])
    _cut = rw._utterance_slice(_sess_sl, 1000, 3000, fallback_chunk_pos=2)
    _rms_cut = rw._loudest_window_rms(rw._wire_to_pcm16(_cut))
    _tail = b"".join(_sess_sl._caller_oai_pcm[2:])
    _rms_tail = rw._loudest_window_rms(rw._wire_to_pcm16(_tail))
    check(_rms_cut > 10 * max(_rms_tail, 1e-9),
          "OpenAI's timestamps cut the SPEECH, not the silence after it",
          f"timestamped {_rms_cut:.4f} vs arrival-time {_rms_tail:.6f}")
    # Missing or nonsensical timestamps fall back rather than measuring nothing.
    check(rw._utterance_slice(_sess_sl, None, None, 0) == b"".join(_sess_sl._caller_oai_pcm),
          "no timestamps -> fall back to the chunk position")
    check(rw._utterance_slice(_sess_sl, 99_000, 99_500, 0) == b"".join(_sess_sl._caller_oai_pcm),
          "out-of-range timestamps fall back rather than slicing nothing")

    # ── ...and OpenAI's clock does not start when ours does ──────────────────
    # SECOND ROOT CAUSE, found 2026-08-20 on call-20260820-1154, and the reason
    # the fix above did not land. _caller_pcm is appended for every inbound
    # frame from stream start; frames are only FORWARDED to OpenAI once
    # listen_enabled is set, after the greeting finishes. So OpenAI's
    # audio_start_ms counts from "greeting done" and ours from "stream start",
    # and every slice read that far too early.
    #
    # Solving for the offset that reproduces all six recorded audio_rms values
    # against the Twilio caller channel gave 9.6s, against a greeting that
    # ended at 9.50s. Offset 0 predicts 0.13-0.19 for every turn and matches
    # none. The damage: four turns of audible speech recorded 0.000244140625 —
    # the SAME signature the previous fix was written to remove, arriving for a
    # new reason — and one turn of pure silence recorded 0.1230, the loudest on
    # the call, because 9.6s earlier the caller was mid-sentence. That turn was
    # a fabricated transcript, and the quarantine waved it through as the
    # clearest speech on the call.
    _lead = _silence[:int(9_600 * _bpms)]          # greeting: never sent to OpenAI
    _sess_off: Any = _S([_lead, _silence[:int(1000*_bpms)], _loud, _silence],
                        listen_start_bytes=len(_lead))
    # OpenAI reports speech at 1000-3000ms of ITS buffer = 10600-12600ms of ours.
    _cut_off = rw._utterance_slice(_sess_off, 1000, 3000, fallback_chunk_pos=0)
    _rms_off = rw._loudest_window_rms(rw._wire_to_pcm16(_cut_off))
    check(_rms_off > 0.01,
          "OpenAI's ms are offset by where ITS buffer starts, not ours",
          f"rms {_rms_off:.4f} (speech), not the pre-greeting silence")
    # The exact failure that shipped: without the offset the slice lands in the
    # greeting, which is mu-law silence, and returns the 0.000244140625 that
    # two previous fixes were each written to eliminate.
    _sess_bug: Any = _S([_lead, _silence[:int(1000*_bpms)], _loud, _silence],
                        listen_start_bytes=0)
    _rms_bug = rw._loudest_window_rms(rw._wire_to_pcm16(
        rw._utterance_slice(_sess_bug, 1000, 3000, fallback_chunk_pos=0)))
    check(_rms_bug < 0.001 and _rms_off > 100 * _rms_bug,
          "and ignoring the offset collapses the measurement to the floor",
          f"offset {_rms_off:.4f} vs no-offset {_rms_bug:.9f}")
    # The offset must be captured BEFORE any caller turn can exist, at the one
    # place listening is enabled — not recomputed later, when _caller_pcm has
    # grown past it.
    _src_all = _PKG_SRC
    check("sess._listen_start_bytes = sum(len(c) for c in sess._caller_oai_pcm)" in _src_all,
          "the offset is recorded where listening is enabled")
    _assigns = re.findall(r"_listen_start_bytes(?:\s*:\s*int)?\s*=", _src_all)
    check(len(_assigns) == 2,
          "and set in exactly two places: the initialiser and that one site",
          f"{len(_assigns)} assignments")

    # ── The quarantine needs TWO signals, never audio alone ──────────────────
    # A guard that DISCARDS a caller's words must not rest on one number that
    # has been wrong twice. The words must also name a health system straight
    # out of our own hint.
    _hint2 = tpl.transcribe_hint
    check(len(rw._hint_proper_nouns(_hint2)) >= 15,
          "the hint's proper nouns are derived from its capitalisation",
          f"{len(rw._hint_proper_nouns(_hint2))} found")
    for _want, _t9 in [
        # both fabrications from call-20260819-2006 / -1323
        (True,  "Hello, I need to schedule an appointment at the Mayo"),
        (True,  "Mercy Hospital"),
        (True,  "across from the Northwell campus"),
        # real answers, including quiet ones, must never qualify
        (False, "It's the Mission Bay clinic, 1825 Fourth Street"),
        (False, "Northside Medical Group, this is Varun"),
        (False, "She works at the Abadan branch"),
        (False, "Okay, she is in San Francisco."),
        (False, "Yes."),
    ]:
        check(rw._reads_as_hint_vocabulary(_t9, _hint2) == _want,
              f"hint-vocabulary signal: {_want!s:5} for {_t9[:40]!r}")
    # The narrow form is load-bearing: requiring EVERY content word to come
    # from the hint missed the Mayo case, where only one word did.
    check(rw._reads_as_hint_vocabulary(
              "Hello, I need to schedule an appointment at the Mayo", _hint2),
          "a fabrication wrapped in ordinary English is still caught")
    # And the drop site must consult both, so a bad rms alone cannot discard.
    _ct_src = re.search(r"async def _handle_caller_transcript.*?(?=\nasync def |\ndef )",
                        _PKG_SRC, re.S)
    check(_ct_src and "_audio_carried_nothing" in _ct_src.group(0)
          and "_reads_as_hint_vocabulary" in _ct_src.group(0),
          "the drop requires both signals together")

    # ── ...and the hole that rule leaves: a fabrication in ordinary English ──
    # Three confirmed fabrications, all adjudicated from the Twilio caller
    # channel rather than from our own numbers. NONE of them quotes the hint,
    # so _reads_as_hint_vocabulary is False for all three and the two-signal
    # rule cannot fire:
    #
    #   call-20260819-2006  "...schedule an appointment at the Mayo"
    #   call-20260820-1154  "...appointment for my annual check-up"   13s of 0.0003
    #   call-20260820-1230  "Hello,"                                  72-80s of 0.0003
    #
    # The claim that such a turn "corrupts nothing because grounding still
    # applies" held for the SAVE and failed for the CALL: on -1230 the phantom
    # drew a reply, the reply stacked on audio still playing, and the callee
    # spent 7.35s saying "Hello?", "campus", "Hello," into a line with no gap.
    for _t10 in ("Hi, I need to schedule an appointment for my annual check-up.",
                 "Hello,"):
        check(not rw._reads_as_hint_vocabulary(_t10, _hint2),
              f"vocabulary test cannot see this fabrication: {_t10[:34]!r}")

    # Silence is decided absolutely, never as a fraction of the caller's own
    # level: the median it would be compared against is computed from turns
    # this predicate exists to exclude.
    check(rw._audio_was_silent(0.000244140625),
          "mu-law digital silence is silence")
    check(rw._audio_was_silent(0.0003),
          "and so is the 0.0003 the Twilio channel shows under all three")
    # Real turns must NEVER read as silent. These are the nine recorded on
    # call-20260820-1230 (the first call after the measurement was fixed) plus
    # the quietest genuine turn seen across all 48 dual-channel recordings.
    for _r10 in (0.1583, 0.1197, 0.1882, 0.0969, 0.1584, 0.1304, 0.1642,
                 0.1239, 0.1227, 0.0793, 0.030):
        check(not rw._audio_was_silent(_r10),
              f"real caller speech is not silence: {_r10}")
    check(rw._audio_was_silent(None) is False,
          "unmeasured is not silent — absence of measurement proves nothing")

    # The margin is the whole reason this may act on audio alone. If someone
    # later tunes _SILENT_AUDIO_RMS up toward the faint threshold, it stops
    # being a different question and starts being able to discard real speech.
    check(rw._SILENT_AUDIO_RMS < rw._LOW_AUDIO_RMS / 5,
          "the silence floor stays far below the faint threshold",
          f"{rw._SILENT_AUDIO_RMS} vs {rw._LOW_AUDIO_RMS}")
    check(0.030 / rw._SILENT_AUDIO_RMS >= 10,
          "and at least 10x below the quietest genuine turn on record",
          f"{0.030 / rw._SILENT_AUDIO_RMS:.0f}x")

    # The drop site must reach the silence branch WITHOUT the vocabulary test,
    # and must not have quietly merged the two thresholds into one.
    check(_ct_src and "_audio_was_silent" in _ct_src.group(0),
          "the drop site acts on silence independently of vocabulary")
    check(_ct_src and "fabricated_turns" in _ct_src.group(0),
          "and records it, so the artifact says the model was told a phantom")
    # The faint-line warning is once per call; this nudge is not. There can be
    # more than one phantom, and suppressing the second leaves exactly the
    # failure the branch exists for.
    # Prove the slice exists before asserting an absence inside it. Unguarded,
    # a regex miss makes _ct_src None and this raises; guarded with `and`, it
    # would silently assert nothing — which is the worse of the two, and the
    # shape this file has been caught by. So check the anchor was found first.
    _ct_body = _ct_src.group(0) if _ct_src else ""
    check("if _silent:" in _ct_body,
          "found the silence branch to inspect")
    _sil_branch = _ct_body[_ct_body.find("if _silent:"):][:1600]
    check("_low_audio_warned" not in _sil_branch,
          "the silence nudge is not rationed like the faint-line warning")

    # END TO END, because every check above is source-level and source-level
    # checks cannot see a branch being disabled. Setting `_silent = False`
    # leaves _audio_was_silent and fabricated_turns both still present in the
    # file, so all of them keep passing while the guard does nothing — the
    # exact false-negative shape this suite has been caught by before.
    # _handle_caller_transcript mutates only `sess`, so it can be driven here.
    class _SilWS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(json.loads(s))

    async def _drive(rms, text):
        _s = rw.RealtimeSession("CA00000000000000000000silence",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s._pending_utterance_rms = rms
        _w = _SilWS()
        await rw._handle_caller_transcript({"transcript": text}, _s, _w)
        _said = [t.text for t in _s.turns if t.role == "caller"]
        _nudges = [m for m in _w.sent
                   if m.get("item", {}).get("role") == "user"]
        return _s, _said, _nudges

    _s, _said, _nudges = await _drive(0.0003, "Hello,")
    check("Hello," not in _said,
          "a transcript over a silent line never becomes a caller turn")
    check(_s.fabricated_turns == ["Hello,"],
          "it is recorded as fabricated", f"{_s.fabricated_turns}")
    check(len(_nudges) == 1 and "was silent" in json.dumps(_nudges),
          "and the model is told not to answer it", f"{len(_nudges)} nudges")

    # The direction that costs a real answer. A genuine turn at the quietest
    # level ever recorded must pass straight through, untouched and unflagged.
    _s2, _said2, _nudges2 = await _drive(
        0.030, "She's at the Mission Bay clinic, 1825 Fourth Street.")
    check(_said2 == ["She's at the Mission Bay clinic, 1825 Fourth Street."],
          "the quietest genuine turn on record is kept in full")
    check(not _s2.fabricated_turns and not _nudges2,
          "not flagged, and no nudge sent")
    # Unmeasured audio must behave like the quiet turn, not like the silent one.
    _s3, _said3, _nudges3 = await _drive(None, "Northgate")
    check(_said3 == ["Northgate"] and not _s3.fabricated_turns and not _nudges3,
          "an unmeasured turn is kept — absence of measurement is not evidence")

    # ── REJECTING A TRANSCRIPT ≠ PREVENTING A REPLY TO IT ────────────────────
    # `create_response` is not set in build_audio_config, so it runs on the API
    # default of true: OpenAI's server VAD creates the response at
    # speech_stopped, strictly BEFORE transcription exists. Every guard here is
    # therefore downstream of a decision already taken.
    #
    #   speech_stopped -> [VAD creates response] -> response.created
    #       -> input_audio_transcription.completed -> guard rejects
    #       -> response.output_audio.delta ... the agent answers it anyway
    #
    # call-20260820-1611: "Hi, I'm looking to schedule an appointment at Mercy
    # Hospital" was dropped as unevidenced and the agent replied "Okay, I'll
    # hold." to it. The drop printed BEFORE the first audio delta, so the reply
    # was suppressible and nothing tried.
    #
    # Three states, three different right answers — and the third must not be
    # dressed up as the second.
    async def _reject_with(active, audio_started):
        _s = rw.RealtimeSession("CA00000000000000000000cancel",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s._pending_utterance_rms = 0.0003            # digital silence
        _s._response_active = active
        _s._response_audio_started = audio_started
        _s._caller_stopped_at = time.monotonic() - 0.4
        _s._response_created_at = time.monotonic() - 0.3
        _w = _SilWS()
        await rw._handle_caller_transcript({"transcript": "Hello,"}, _s, _w)
        _cancels = [m for m in _w.sent if m.get("type") == "response.cancel"]
        return _s, _cancels

    _sc, _cx = await _reject_with(active=True, audio_started=False)
    check(len(_cx) == 1,
          "reply in flight with no audio yet -> response.cancel is sent")
    check(_sc._suppressed_response is True,
          "and the cancelled response's own transcript is marked to be skipped")
    check(_sc.rejection_cancels[0]["outcome"] == "cancelled before any audio",
          "recorded as cancelled", _sc.rejection_cancels[0]["outcome"])
    check(isinstance(_sc.rejection_cancels[0]["since_speech_stopped_s"], float),
          "with the margin, so it can be measured across calls",
          f"{_sc.rejection_cancels[0]['since_speech_stopped_s']}s")

    # Already audible: cancelling cannot unsay it. Report it, do not pretend.
    _sl, _cx2 = await _reject_with(active=True, audio_started=True)
    check(not _cx2,
          "audio already reaching the caller -> no cancel is attempted")
    check(_sl._suppressed_response is False,
          "and its transcript is NOT skipped — the caller did hear it")
    check("TOO LATE" in _sl.rejection_cancels[0]["outcome"],
          "recorded as too late, not as a prevention",
          _sl.rejection_cancels[0]["outcome"])

    # Nothing in flight at all — no protocol traffic for the sake of it.
    _sn, _cx3 = await _reject_with(active=False, audio_started=False)
    check(not _cx3 and _sn.rejection_cancels[0]["outcome"] == "no reply in flight",
          "nothing in flight -> nothing cancelled")

    # THE WIRING, which the three checks above cannot see. They set the state
    # by hand, so deleting the lines that MAINTAIN it leaves them all passing —
    # mutation-proven blind before these were added.
    #
    # A cancelled response still emits its transcript. Nothing was heard, so
    # letting it become a turn would put words in the record the caller never
    # got, and hand them to the guards as evidence.
    _sup_sess = rw.RealtimeSession("CA0000000000000000000supskip",
                                   Doctor(doctor_name="Dr. Jane Okafor"))
    _sup_sess._suppressed_response = True
    _buf, _bip = await rw._handle_agent_transcript(
        {"transcript": "Okay, I'll hold."}, _sup_sess, _SilWS(), "", False)
    check(not [t for t in _sup_sess.turns if t.role == "agent"],
          "a suppressed response's transcript never becomes an agent turn")
    check(_sup_sess._suppressed_response is False,
          "and the flag is consumed, so the next response is unaffected")
    # ...and an ordinary response still does become one.
    _ok_sess = rw.RealtimeSession("CA00000000000000000000supok",
                                  Doctor(doctor_name="Dr. Jane Okafor"))
    await rw._handle_agent_transcript(
        {"transcript": "Okay, I'll hold."}, _ok_sess, _SilWS(), "", False)
    check([t.text for t in _ok_sess.turns if t.role == "agent"] == ["Okay, I'll hold."],
          "an ordinary response is still recorded")

    # _response_audio_started must be maintained in exactly two places: reset
    # per response, and set on the first delta. Losing either silently breaks
    # the state distinction the whole decision rests on — the "too late" branch
    # would stop firing, or would fire always.
    # WORKER PLUS AUDIO, because the three assignments no longer share a file:
    # the reset sits in the response.created handler here, the set sits in the
    # delta handler that moved to agents/voice/audio.py. The invariant the
    # comment below states — two Falses and one True, WHEREVER THEY LIVE — is
    # unchanged; only the population being counted had to follow the code.
    # Counting one module after an extraction is how a check silently shrinks.
    _rw_txt = _PKG_SRC
    _aud = re.findall(r"_response_audio_started\s*(?::\s*bool\s*)?=\s*(\w+)",
                      _rw_txt)
    # COUNTED, NOT ORDERED. This asserted the literal sequence
    # ["False","False","True"], which encoded where in the FILE each assignment
    # happened to sit — so extracting the audio-delta block into its own
    # coroutine (defined above _oai_to_twilio, as pyright's complexity bail
    # required) flipped the order and failed a test about state maintenance for
    # a reason that had nothing to do with state maintenance.
    #
    # The invariant was never positional. It is: initialised once, reset once
    # per response, set once when audio arrives — two Falses and one True,
    # wherever they live.
    check(sorted(_aud) == ["False", "False", "True"],
          "audio-started is initialised, reset per response, and set on audio",
          f"{_aud}")
    # BOUNDED BY THE HANDLER, NOT BY A CHARACTER COUNT. This read [:1800] and
    # broke the day per-turn stage marks were added to the same handler — the
    # reset had not moved an inch, it had simply been pushed past an arbitrary
    # window. That is the identical failure the comment above describes for the
    # ["False","False","True"] ordering, repeated in a check written to replace
    # it. The claim is "the reset is inside this handler"; the handler ends at
    # the next elif, so that is where the slice ends.
    _rc_start = _rw_txt.find('event_type == "response.created"')
    _rc_end = _rw_txt.find("elif event_type ==", _rc_start + 10)
    _resp_created = _rw_txt[_rc_start:_rc_end]
    check("_response_audio_started = False" in _resp_created,
          "the reset lives in the response.created handler, before any audio",
          f"handler spans {len(_resp_created)} chars")
    # ANCHORED ON THE FUNCTION, not on a character window after a string. The
    # window version broke the moment the body moved; asking the interpreter
    # which source belongs to _handle_audio_delta cannot drift.
    _delta_src = inspect.getsource(rw._handle_audio_delta)
    check("_response_audio_started = True" in _delta_src,
          "and the set lives in the audio-delta handler")
    # THE HANDLER IS STILL WIRED IN. The check above passes just as well if
    # _handle_audio_delta is never called — which is exactly what a bad
    # extraction looks like, and the extraction that introduced this check
    # shipped with an unbound local that would have raised on the first delta
    # of every call.
    check("_handle_audio_delta(" in
          inspect.getsource(rw._oai_to_twilio),
          "and _oai_to_twilio actually dispatches to it")

    # ── A silent drop must leave a trace ─────────────────────────────────────
    # Two turns were dropped on call-20260819-2006 and the artifact recorded
    # nothing — the only evidence was a terminal that happened to still be open.
    # ANCHORED ON THE RECORD LINE, not on how many characters away from
    # `grounding` it happens to sit. The first cut measured a 600-character
    # window and broke the day a comment was added above it — a check that
    # fails on prose is a check nobody trusts, and the claim was never about
    # distance. Every guard that DELETES something the call produced is listed
    # here, because a guard that fires invisibly cannot be reviewed afterwards
    # and each of these was retrofitted after a call where it did.
    _rec = _PKG_SRC
    for _key, _what in [
        ("suppressed_echoes", "turns the quarantine discarded"),
        ("dropped_second_items", "audio muted before it reached the caller"),
        ("owed_abandoned", "substance the recovery gave up on"),
        ("name_mismatches", "a surname that was not our doctor's"),
        ("grounding_at_save", "a grounding verdict the transcript revised"),
    ]:
        check(f'"{_key}":' in _rec,
              f"{_what} is written into the call artifact",
              f"{_key} is missing from the record — the guard would fire "
              f"invisibly, which is the failure every one of these was "
              f"retrofitted to stop")

    # ── PER-TURN STAGE CLOCKS ────────────────────────────────────────────────
    # reply_latency gives one number per turn. On call-20260826-1134 that number
    # ran 1.69s to 6.64s across five structurally identical tool turns and
    # nothing could say which stage moved. These six marks split it.
    print("\n" + "-" * 66)
    print("  Where a turn's seconds actually went")
    print("-" * 66)

    # A MISSING STAGE IS None, NEVER 0.0. This is the whole discipline: a turn
    # with no tool call has no t2/t3/t4, and reporting them as zero would put it
    # in the same bucket as a tool turn whose deferral was instant — the exact
    # conflation the record exists to break, and it would drag every median
    # toward zero while looking perfectly healthy.
    _no_tool = rw._stage_row({"t0": 100.0, "t1": 100.4, "t5": 101.9,
                              "detector_s": 0.45}, 1.9)
    check(_no_tool["no_tool_s"] == 1.5,
          "a turn with no tool records ONE inference under its own name",
          f"{_no_tool['no_tool_s']}")
    check(all(_no_tool[k] is None for k in
              ("inference_1", "our_work", "deferral", "inference_2")),
          "and the tool-only stages are absent, not zero",
          f"{[(k, _no_tool[k]) for k in ('inference_1','our_work','deferral','inference_2')]}")
    check(_no_tool["tool"] is None, "and it is not attributed to a tool")

    # The referral turn's shape, with the real numbers it has to be able to
    # express: 6.64s of think time that the console log could only bound.
    _tool = rw._stage_row({"t0": 200.0, "t1": 200.45, "t2": 205.95,
                           "t3": 205.96, "t4": 206.6, "t5": 207.09,
                           "detector_s": 0.45, "tool": "save_referral_requirement"}, 7.09)
    check(_tool["vad_to_resp"] == 0.45 and _tool["inference_1"] == 5.5,
          "a tool turn separates the pre-response wait from inference 1",
          f"vad->resp {_tool['vad_to_resp']}  infer1 {_tool['inference_1']}")
    check(_tool["our_work"] == 0.01 and _tool["deferral"] == 0.64
          and _tool["inference_2"] == 0.49,
          "and splits our work, the deferral, and inference 2",
          f"ours {_tool['our_work']}  defer {_tool['deferral']}  "
          f"infer2 {_tool['inference_2']}")
    check(_tool["no_tool_s"] is None,
          "a tool turn does not also report a no-tool inference",
          "both present would double-count the same seconds")
    check(abs(sum(_tool[k] for k in ("vad_to_resp", "inference_1", "our_work",
                                     "deferral", "inference_2")) - 7.09) < 0.02,
          "the five stages add up to the gap the caller actually felt",
          f"{sum(_tool[k] for k in ('vad_to_resp','inference_1','our_work','deferral','inference_2')):.2f}s "
          f"vs felt {_tool['felt_s']}s")

    # ── ARE THE MARKS ACTUALLY SET? ─────────────────────────────────────────
    # Every check above drives _stage_row directly, so all of them keep passing
    # if no handler ever writes a mark — which is what a broken wiring looks
    # like, and this file has been bitten twice (the unwired _turn_asserts, the
    # conditioner the delta handler never called). Anchored on the FUNCTION the
    # mark must live in, so moving code cannot silently disarm it.
    for _fn, _mark, _why in [
        (rw._handle_tool_call, '"t2"', "the tool call arriving"),
        (rw._handle_tool_call, '"t3"', "the result going back"),
        (rw._stage_row,        '"t4"', "the deferral ending"),
    ]:
        check(_mark in inspect.getsource(_fn),
              f"{_mark} is stamped where it belongs — {_why}")
    # ANCHORED ON THE FUNCTION THAT OWNS THE EVENT, which is the whole point of
    # the comment above — and it earned its keep on 2026-08-27, when the
    # response.done block moved to lifecycle.py and this check went red rather
    # than quietly passing on a file that no longer held the code.
    for _mark, _fn, _why in [('"t1"', rw._oai_to_twilio,        "response.created"),
                             ('"t4"', rw._handle_response_done, "response.done")]:
        check(_mark in inspect.getsource(_fn),
              f"{_mark} is wired into the handler that owns it — {_why}")
    # t5 and the close live in the AUDIO DELTA handler, not the event loop —
    # same place reply_latency is computed, because they answer the same
    # question and must not be able to disagree about when the agent spoke.
    check("_stage_row(" in inspect.getsource(rw._handle_audio_delta),
          "the row is closed where reply_latency is measured, on first audio")
    check('sess._stage = {"t0"' in _rec,
          "t0 opens the record at speech_stopped",
          "opening it at response.created cannot measure the stage BEFORE the "
          "response exists, which is the one that carried the variance")
    check('"stages": self.turn_stages' in _rec,
          "and the rows reach the call artifact")

    # AND THE TOOL MARKS LAND AT RUNTIME, not just in the source. The existing
    # scenarios all take the no-tool branch (four stage rows, every one of them
    # "[no tool]"), so t2/t3 were covered by source anchors alone — which is
    # exactly the coverage that let an unwired conditioner and an unwired
    # _turn_asserts through. Drive the real handler and read the marks back.
    class _ToolWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    _tsess = rw.RealtimeSession("CA00000000000000000000stage1",
                                Doctor(doctor_name="Dr. Jane Okafor"))
    _tsess._stage = {"t0": time.monotonic(), "detector_s": 0.4}
    _tsess._stage["t1"] = time.monotonic()
    _tws = _ToolWS()
    await rw._handle_tool_call(
        {"call_id": "c1", "name": "save_branch",
         "arguments": json.dumps({"branch": "Riverside Campus"})},
        _tsess, _tws, {}, False)
    check("t2" in _tsess._stage and "t3" in _tsess._stage,
          "the tool handler stamps t2 and t3 on a real call",
          f"marks now: {sorted(k for k in _tsess._stage if k.startswith('t'))}")
    check(_tsess._stage.get("tool") == "save_branch",
          "and records which tool the turn was spent on",
          f"{_tsess._stage.get('tool')}")
    check(_tsess._stage["t3"] >= _tsess._stage["t2"],
          "with the result submitted no earlier than the call arrived")
    check(any(m.get("item", {}).get("type") == "function_call_output"
              for m in _tws.sent),
          "and the tool output really was sent — the marks bracket real work",
          str([m.get("type") for m in _tws.sent]))

    print("\n" + "-" * 66)
    print("  Every tool in a turn, not just the first")
    print("-" * 66)
    # call-20260826-1656 could not be audited: identity is saved `confirmed`,
    # the only save_doctor_identity in the stage data sits on a turn whose
    # transcript the guard REJECTS, and the stored quote appears in two caller
    # turns. t2/t3 mark the FIRST tool because that is what inference_1
    # measures, so a second tool in the same response left no trace at all.
    _row2 = rw._stage_row({"t0": 1.0, "t1": 1.4, "t2": 2.0, "t3": 2.1,
                           "t4": 2.3, "t5": 3.0, "detector_s": 0.4,
                           "tool": "save_branch",
                           "tools": [{"tool": "save_branch", "ok": False},
                                     {"tool": "save_doctor_identity", "ok": True}]}, 2.0)
    check(_row2["tool"] == "save_branch",
          "the first tool still names the turn — the intervals are measured from it")
    check(_row2["tools"] == [{"tool": "save_branch", "ok": False},
                             {"tool": "save_doctor_identity", "ok": True}],
          "and the full list carries every tool WITH its verdict",
          "without the verdict a rejected save and an accepted one look identical")
    check(rw._stage_row({"t0": 1.0, "t1": 1.4, "t5": 2.0, "detector_s": 0.4},
                        1.0)["tools"] is None,
          "a turn with no tool reports tools as absent, not as an empty list",
          "absent and empty mean different things when a median is taken")

    # ── AND IT IS ACTUALLY APPENDED, on every call, at runtime ─────────────
    # The checks above drive _stage_row directly and keep passing if nothing
    # ever writes the list — the unwired-extraction shape this file has been
    # bitten by three times.
    class _MultiWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    _ms = rw.RealtimeSession("CA00000000000000000multi",
                             Doctor(doctor_name="Dr. Jane Okafor"))
    _ms._stage = {"t0": time.monotonic(), "detector_s": 0.4}
    _ms._stage["t1"] = time.monotonic()
    _mw = _MultiWS()
    await rw._handle_tool_call(
        {"call_id": "m1", "name": "save_branch",
         "arguments": json.dumps({"branch": "Riverside Campus"})}, _ms, _mw, {}, False)
    await rw._handle_tool_call(
        {"call_id": "m2", "name": "note_info",
         "arguments": json.dumps({"key": "address", "value": "1467 River Street"})},
        _ms, _mw, {}, False)
    _tools = _ms._stage.get("tools") or []
    check(len(_tools) == 2,
          "two tool calls in one turn produce two entries",
          f"{_tools}")
    check([t["tool"] for t in _tools] == ["save_branch", "note_info"],
          "in the order they arrived", f"{[t['tool'] for t in _tools]}")
    check(all("ok" in t for t in _tools),
          "each carrying whether it was accepted")
    check(_ms._stage.get("tool") == "save_branch" and "t2" in _ms._stage,
          "and the FIRST tool still owns t2, so inference_1 is unchanged",
          "recording more must not move the interval boundaries")

    # MEASURE-ONLY, and this is the check that keeps it that way. The moment a
    # guard reads a stage timing, the instrument starts changing the thing it
    # measures — and every other counter in this file earned its place by NOT
    # doing that.
    # STATED AS THE INVARIANT, NOT AS AN ALLOWLIST - and the difference is
    # not academic. This was an allowlist of permitted line shapes until
    # 2026-08-26, when recording every tool per turn needed a plain
    # `if sess._stage is not None:` guard and that shape was added to it.
    # A mutation immediately walked through the hole: 
    #
    #     if sess._stage is not None and sess._stage.get("t2", 0.0) > 5.0:
    #         sess.done = True
    #
    # passed, because the line CONTAINS an allowed substring. The check went
    # green on a guard reading a stage timing to end calls - the exact thing
    # it exists to forbid.
    #
    # The claim was never "these line shapes are permitted". It is: NO STAGE
    # TIMING KEY IS EVER READ. Timing keys are t0..t5; `tool`, `tools` and
    # `detector_s` are labels the row carries, not durations. So the test is
    # a search for a read of t<digit> that is not an assignment, and no
    # amount of surrounding syntax can satisfy it.
    _TIMING_READ = re.compile(r'_stage(?:\.get\(\s*"t\d"|\["t\d"\](?!\s*=))')
    _illegal = [ln.strip() for ln in _rec.splitlines()
                if not ln.strip().startswith("#")
                and _TIMING_READ.search(ln)]
    check(not _illegal,
          "nothing reads a stage timing to decide anything",
          f"{_illegal} — a stage DURATION must never decide behaviour")

    # ── Pre-warming the OpenAI session while the phone rings ─────────────────
    # call-20260819-1915: 6.4 SECONDS between the callee pressing answer and
    # hearing a word, all of it before the media stream opened. ~2.2s of that
    # was the OpenAI handshake and session.update — work that needs NOTHING
    # call-specific and was only being started once someone had already picked
    # up, while the phone had been ringing for seconds doing nothing.
    rw._PREWARMED.clear()
    check(rw.take_prewarmed("CA_never_warmed") is None,
          "claiming a session that was never warmed returns nothing")

    class _FakeConn:
        def __init__(self): self.closed = False
        async def __aexit__(self, *a): self.closed = True

    _c1 = _FakeConn()
    rw._PREWARMED["CA_ready"] = (_c1, "ws-handle", time.time())
    _claim = rw.take_prewarmed("CA_ready")
    check(_claim is not None and _claim[0] is _c1 and _claim[1] == "ws-handle",
          "a warmed session is handed to the call that placed it")
    check(rw.take_prewarmed("CA_ready") is None,
          "and it is claimed exactly once — never handed to a second call")

    # A call nobody answered must not leave its socket for a LATER call: that
    # call would get a connection idle for minutes.
    _c2 = _FakeConn()
    rw._PREWARMED["CA_stale"] = (_c2, "ws", time.time() - rw._PREWARM_TTL_S - 10)
    check(rw.take_prewarmed("CA_stale") is None,
          "a stale session is refused rather than reused",
          f"TTL {rw._PREWARM_TTL_S}s")
    await asyncio.sleep(0)   # let the close task run
    check(_c2.closed, "and the stale socket is closed, not leaked")
    rw._PREWARMED.clear()

    # FAILING IS FREE. If OpenAI is unreachable when the call is placed, the
    # call must still work — it just pays the handshake on answer, exactly as
    # before. prewarm_realtime therefore cannot raise.
    with mock.patch.object(rw, "_open_realtime_session",
                           side_effect=RuntimeError("openai down")):
        await rw.prewarm_realtime("CA_fails")   # must not raise
    check("CA_fails" not in rw._PREWARMED,
          "a failed pre-warm leaves nothing behind for the call to claim")
    check(rw.take_prewarmed("CA_fails") is None,
          "so handle_realtime falls back to connecting on answer")

    # The extraction must not have produced a second copy of the session
    # config — two places to get the cached-prefix rule subtly wrong.
    import pathlib as _plb
    _rw_src2 = _PKG_SRC
    check(_rw_src2.count('"type": "session.update"') == 1,
          "session.update is built in exactly one place",
          f"{_rw_src2.count(chr(34) + 'type' + chr(34) + ': ' + chr(34) + 'session.update' + chr(34))} found")
    # And the placing side has to actually trigger it, or the whole thing is
    # dead code that quietly never runs.
    import agents.voice.twilio_worker as _tw2
    _tw_src2 = _plb.Path(_tw2.__file__).read_text(encoding="utf-8")
    check("prewarm_realtime" in _tw_src2 and "run_coroutine_threadsafe" in _tw_src2,
          "register_call schedules the pre-warm onto the server's loop")

    # ── "Hang on" is not always a hold ───────────────────────────────────────
    # call-20260819-1915: the caller said "Hang on, are you a real person or is
    # this a recording?" and is_hold_request returned True — the console
    # printed "Caller is going to check" while she was challenging the agent.
    #
    # Harmless until _HOLD_GRACE_S existed. Not harmless after: a hold silences
    # the watchdog for 45s, so "hang on, who is this?" followed by waiting
    # would be met with 45 seconds of nothing. A regression the hold fix
    # introduced, surfacing on the very next call.
    #
    # The discriminator is WHO is being asked to act. A hold says the CALLER
    # will ("let me check"); a challenge asks the AGENT ("are you real?").
    # A second-person test alone is NOT enough — "can you please give me a
    # minute? I just need to check" is a question, says "you", and is the most
    # cooperative sentence on the call that scenario 0b was built from.
    for _want, _t7 in [
        (False, "Hang on, are you a real person or is this a recording?"),
        (False, "Hang on — who did you say you were?"),
        (False, "Hold on, what do you want exactly?"),
        (True,  "can you please give me a minute? I just need to check"),
        (True,  "Ok, alright. Give me a minute, let me pull that up."),
        (True,  "Hang on a second, let me check."),
        (True,  "Can you hold on a moment?"),
        (True,  "Let me just look that up for you."),
        (True,  "One moment."),
        (True,  "Bear with me."),
    ]:
        check(rw.is_hold_request(_t7) == _want,
              f"hold vs challenge: {_want!s:5} for {_t7[:44]!r}")

    # ── "Is this about a patient?" must be answered ──────────────────────────
    # Asked on call-20260819-1847 and again on -1915, half-answered both times:
    # "No, nothing urgent — it's just a listing check" addresses only the
    # second half. At a medical office that question decides whether they pull
    # a record or route to clinical staff; leaving it inferred is the ambiguity
    # a front desk is trained not to accept.
    #
    # The prompt already says "Answer EVERY one of them" and it did not hold
    # twice running — so the process asks, the same way it does for the
    # identity question.
    for _want, _t8 in [
        (True,  "Is it about a patient or something urgent?"),
        (True,  "Is this about a patient, or something urgent?"),
        (True,  "Is this about one of our patients?"),
        (False, "Which branch do you need?"),
        (False, "She is not seeing patients today."),
        (False, "Are you a real person?"),
    ]:
        check(rw._asks_about_patient(_t8) == _want,
              f"patient question: {_want!s:5} for {_t8[:40]!r}")
    check(rw.RealtimeSession("CA00000000000000000000000pat01",
                             Doctor(doctor_name="Dr. P"))._patient_nudged is False,
          "the patient nudge starts unfired, and is one-shot")

    # ── The caller gave more than we recorded ────────────────────────────────
    # call-20260819-1847: she said "it's the Mission Bay clinic, 1825 Fourth
    # Street" and the agent saved just "Mission Bay Clinic". Nothing blocked
    # the fuller value — grounding accepts it — the model simply left it out,
    # despite the prompt saying "Several: pass them all, comma-separated".
    #
    # Mirror image of the SAME morning's failure, where it invented a street
    # number: one question asked in opposite directions. _ungrounded_terms asks
    # whether we recorded too MUCH; this asks whether we recorded too LITTLE.
    _mk_c = lambda t: double(role="caller", text=t, audio_rms=0.18)
    _addr1847 = double(turns=[
        _mk_c("Okay, she's in San Francisco."),
        _mk_c("Right, it's the Mission Bay clinic, 1825 Fourth Street, I think so."),
    ])
    check(rw._address_offered(_addr1847) == "1825 Fourth Street",
          "a street address in the transcript is recognised",
          repr(rw._address_offered(_addr1847)))
    check(bool(rw._address_dropped(
              {"branch": "Mission Bay Clinic", "city": "San Francisco"}, _addr1847)),
          "saving only the site name, when a street address was given, is caught")
    check(not rw._address_dropped(
              {"branch": "Mission Bay Clinic, 1825 Fourth Street"}, _addr1847),
          "and saving both is accepted")
    # The house NUMBER is the key, not the street words: "Fourth Street" may
    # legitimately be absent from a value that names the site, but the number
    # is either recorded or it is lost.
    check(not rw._address_dropped({"branch": "Clinic at 1825"}, _addr1847),
          "the number alone is enough to count as recorded")
    # Must not fire when no address was ever offered — that is 32 of the 36
    # calls in the history.
    _no_addr = double(turns=[
        _mk_c("She's at the Northgate campus."),
        _mk_c("Just the main office, I think.")])
    check(rw._address_offered(_no_addr) is None,
          "no address offered -> the guard cannot fire")
    check(not rw._address_dropped({"branch": "Northgate Campus"}, _no_addr),
          "and a save with no address in play is untouched")
    # A bare number is not an address. "give me a minute", "one of two sites",
    # a year — all contain digits and none is a street.
    for _t in ["Give me a minute, I need to check.",
               "We have 3 sites in the area.",
               "She joined us in 2019."]:
        check(rw._address_offered(double(turns=[_mk_c(_t)])) is None,
              f"not an address: {_t[:34]!r}")
    # ONE-SHOT is load-bearing: the value being saved is CORRECT, just thinner
    # than what they said, so this must never be able to stop a call finishing.
    # A true-but-thin record beats no record.
    _as = rw.RealtimeSession("CA000000000000000000000000addr1",
                             Doctor(doctor_name="Dr. A"))
    check(_as._address_nudged is False, "the address nudge starts unfired")
    # And the rejection stays unspeakable, like every other one.
    _ar = (f"NOT SAVED — a street address was given and this value omits it "
           f"| THEY SAID: '1825 Fourth Street' | RETRY: save_branch with both, "
           f"comma-separated | ALREADY SUPPLIED, nothing further needed from them")
    for _ph in ("ask them", "could you", "tell me", "please provide", "you should"):
        check(_ph not in _ar.lower(),
              f"address rejection has no speakable imperative ({_ph!r})")

    # ── Digits must match exactly ────────────────────────────────────────────
    # The word rule in _ungrounded_terms is deliberately lenient — one content
    # word matching is enough, because transcription is imperfect and a real
    # answer beats a blocked one. Right for words. Exactly wrong for numbers.
    #
    # call-20260819-1716: the caller said "1825 4th Street". The agent saved
    # "Mission Bay Clinic, 1855 Fourth Street" and grounding PASSED it, because
    # "bay" appeared and one word was enough. A four-digit house number nobody
    # said was written into the client directory, and the record was marked
    # "verified against caller transcript".
    #
    # Worst failure category available: not empty, not obviously wrong, but
    # PLAUSIBLE. No reviewer catches it and someone sent to 1855 Fourth Street
    # finds the wrong building.
    _hf = lambda t: double(role="caller", text=t, audio_rms=0.15)
    _addr_sess = double(turns=[
        _hf("Okay, she is in San Francisco."),
        _hf("Examination Bay Clinic, 1825 4th Street"),
    ])
    for _want, _label, _args in [
        (True,  "the real call: 1825 heard, 1855 saved",
         {"branch": "Mission Bay Clinic, 1855 Fourth Street", "city": "San Francisco"}),
        (True,  "a digit dropped",  {"branch": "Bay Clinic, 182 Fourth Street"}),
        (True,  "a digit appended", {"branch": "Bay Clinic, 18250 Fourth Street"}),
        (False, "the number as spoken survives",
         {"branch": "Mission Bay Clinic, 1825 Fourth Street", "city": "San Francisco"}),
        (False, "a value with no digits is unaffected",
         {"branch": "Mission Bay Clinic", "city": "San Francisco"}),
    ]:
        check(bool(rw._ungrounded_terms(_args, _addr_sess)) == _want,
              f"digit grounding: {'blocked' if _want else 'allowed'} — {_label}")
    # The word tolerance must survive: this is a narrowing for numbers only.
    # "mission" was never said either, and that is still forgiven, because a
    # misheard street NAME is recoverable in a way a street NUMBER is not.
    check(not rw._ungrounded_terms({"branch": "Mission Bay Clinic"}, _addr_sess),
          "a misheard word is still forgiven — only digits are strict")
    # And the rejection has to stay unspeakable, like every other one.
    _dv = rw._ungrounded_terms(
        {"branch": "Mission Bay Clinic, 1855 Fourth Street"}, _addr_sess)
    for _ph in ("ask them", "could you", "tell me", "please provide"):
        check(_ph not in _dv.lower(),
              f"digit rejection has no speakable imperative ({_ph!r})")

    # ── ...and the same tolerance for a number SPELLED OUT ───────────────────
    # The digit rule only inspects digit runs, so a value carrying no digits
    # skipped it entirely and passed vacuously. Spelling the number in words
    # was a complete bypass of the strictest guard in the file.
    #
    # call-20260820-1321 found it, and the guard drove the model there. Caller
    # said "It's Mission Bay Clinic, 1844th Street":
    #   1st  'Mission Bay Clinic, 18 4th Street'              REJECTED, rightly
    #   2nd  'Mission Bay Clinic, 18 4th Street'              REJECTED, rightly
    #   3rd  'mission bay clinic, eighteen forty fourth street'  SAVED
    # into doctors.json as partially_verified, grounding "verified against
    # caller transcript". Nothing verified it — there were no digits to check.
    _bypass = double(turns=[
        _hf("It's Mission Bay Clinic, 1844th Street."),
        _hf("Okay, she lives San Francisco."),
    ])
    for _want, _label, _args in [
        (True,  "the exact bypass that shipped a bad record",
         {"branch": "mission bay clinic, eighteen forty fourth street",
          "city": "San Francisco"}),
        (True,  "and the same trick with a different invented number",
         {"branch": "Mission Bay Clinic, eighteen twenty fifth street"}),
        (False, "the caller's own digits are still accepted",
         {"branch": "Mission Bay Clinic, 1844th Street", "city": "San Francisco"}),
    ]:
        check(bool(rw._ungrounded_terms(_args, _bypass)) == _want,
              f"spelled-number grounding: "
              f"{'blocked' if _want else 'allowed'} — {_label}")

    # ── Closing a hole must not open a liveness one ─────────────────────────
    # Every correction at the save-rejection site is one-shot and nothing
    # counted the rejections, so a model that cannot produce an acceptable
    # value retried indefinitely. call-20260820-1321 attached a closing line to
    # each attempt — "I'll note that and wrap up", "I'll note it and let you
    # go", "take care" — twenty seconds of thanking a caller for a branch that
    # was never recorded, and the SECOND rejection got no correction at all
    # because _false_save_nudged was spent on the first.
    #
    # That call terminated only because the third attempt slipped through the
    # spelled-number bypass. Closing that bypass removes the accidental exit,
    # so the bound has to be explicit: a guard made stricter must carry the
    # liveness the leak was accidentally providing.
    check(rw._MAX_SAVE_REJECTIONS >= 2,
          "a save budget exists and allows a normal correction cycle",
          f"{rw._MAX_SAVE_REJECTIONS}")
    # The tool-call rejection path moved to agents/voice/grounding.py with
    # _handle_tool_call. Same assertions, read from where the code now lives.
    _rsrc = _plb.Path(_rwground.__file__).read_text(encoding="utf-8")
    _rej = _rsrc[_rsrc.find("sess._save_rejections += 1"):][:1800]
    check(_rej, "found the rejection-counting site")
    # Guessing is not the exit. The caller's words are already on the
    # transcript, so the directive must QUOTE them rather than ask for another
    # attempt — and must offer escalate as the way out if that fails too.
    check("_candidate_location(sess)" in _rej,
          "at the limit the agent is handed the caller's verbatim words")
    check("escalate" in _rej,
          "and a truthful escalation is offered as the exit")
    check("do not say goodbye again" in _rej,
          "and told to stop closing the call until something succeeds")
    # END TO END. Every check above is source-level, and source-level checks
    # cannot see the branch being disabled: `if False:` leaves
    # _candidate_location, "escalate" and "do not say goodbye" all present in
    # the slice, so all of them keep passing while the budget never fires.
    # Mutation-proven blind before this was added. _handle_tool_call takes five
    # plain arguments, so it can be driven directly.
    class _TcWS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(json.loads(s))

    async def _reject_n(n):
        _s = rw.RealtimeSession("CA0000000000000000000rejbud",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.turns = [rw.TranscriptTurn(role="caller", text=t, timestamp="00:00:00",
                                      audio_rms=0.15)
                    for t in ("Okay, she lives San Francisco.",
                              "It's Mission Bay Clinic, 1844th Street.")]
        _w = _TcWS()
        for i in range(n):
            await rw._handle_tool_call(
                {"name": "save_branch", "call_id": f"c{i}",
                 "arguments": json.dumps(
                     {"branch": "mission bay clinic, eighteen forty fourth street",
                      "city": "San Francisco"})},
                _s, _w, {}, True)
        _d = [m for m in _w.sent
              if "nothing has been recorded" in json.dumps(m)]
        return _s, _d

    _s_b, _d_b = await _reject_n(rw._MAX_SAVE_REJECTIONS - 1)
    check(_s_b._save_rejections == rw._MAX_SAVE_REJECTIONS - 1 and not _d_b,
          "under the budget the agent is left to correct itself",
          f"{_s_b._save_rejections} rejections, {len(_d_b)} directives")
    _s_b, _d_b = await _reject_n(rw._MAX_SAVE_REJECTIONS)
    check(len(_d_b) == 1,
          "at the budget the directive fires exactly once",
          f"{len(_d_b)} directives")
    check("1844th Street" in json.dumps(_d_b),
          "and it quotes the caller's number, digit for digit")
    # The counter must be a plain increment, never reset mid-call — the same
    # shape the silence budget is asserted on, and for the same reason: a reset
    # makes a budget unreachable while every test of the budget still passes.
    # WORKER PLUS GROUNDING: the "= 0" initialiser lives on RealtimeSession in
    # the worker, the "+= 1" moved with _handle_tool_call. Reading one module
    # sees half the invariant and calls it satisfied.
    _assigns = re.findall(
        r"_save_rejections\s*(?::\s*int\s*)?([+]?=)\s*(\S+)", _PKG_SRC)
    # SORTED, not positional. The initialiser is on RealtimeSession in the
    # worker and the increment moved to grounding, so their order now depends on
    # which filename sorts first — which is not a property this check ever meant
    # to assert. The claim is the SET of assignments: one init, one increment,
    # no reset. Concatenating _rsrc as well counted the increment twice.
    check(sorted(_assigns) == [("+=", "1"), ("=", "0")],
          "the save counter is only initialised or incremented, never reset",
          f"{_assigns}")

    # ── A QUESTION IS NOT AN ANSWER ──────────────────────────────────────────
    # Grounding compared a saved value against one blob of every caller turn,
    # so a value the caller ASKED about grounded exactly like one they stated.
    #
    # call-20260820-1703: "She's in San Francisco, right?" — never confirmed
    # afterwards — put city="San Francisco" into the directory stamped
    # "verified against caller transcript". They were asking US, and we had
    # nothing to confirm it with: the record holds an organisation, not a city.
    #
    # The distinction already existed for the ask budget (_caller_is_vetting,
    # built after the agent hung up on "How can I help you?"). Grounding never
    # consulted it.
    _ta = lambda t: rw._turn_asserts(t, double(
        turns=[], doctor=double(hospital_name="Northside Medical Group",
                                doctor_name="Dr. Jane Okafor"),
        org_name="Definitive Healthcare", agent_name="David"))
    for _kind, _txt, _want in [
        # ── direct assertions: every one must stay usable ──────────────────
        ("assertion", "It's the Mission Bay Clinic.", True),
        ("assertion", "She's in San Francisco.", True),
        ("assertion", "Northgate.", True),
        # THE EXPENSIVE DIRECTION. _caller_is_vetting fires on _VETTING_OPENER
        # alone, so without the "?" conjunct this bare answer is discarded —
        # the case the digit/word rule below defends by name.
        ("assertion", "Sorry, Northgate.", True),
        ("assertion", "Um, Northgate.", True),
        ("assertion", "Yes, San Francisco.", True),
        ("assertion", "She works out of the Abadan branch.", True),
        ("assertion", "Where she works is the Mission Bay clinic.", True),
        ("assertion", "Northgate and Riverside.", True),
        ("assertion", "Can confirm it's the Abadan branch.", True),
        # ── confirmation-seeking: the defect this closes ────────────────────
        ("confirm-seeking", "She's in San Francisco, right?", False),
        ("confirm-seeking", "Is she in San Francisco?", False),
        ("confirm-seeking", "Maybe San Francisco? I'm not sure.", False),
        # ── screening questions ─────────────────────────────────────────────
        ("screening", "Sorry, who's calling again?", False),
        ("screening", "Is this about a patient?", False),
        ("screening", "How can I help you?", False),
        # ── hedged is still telling ─────────────────────────────────────────
        ("hedged", "I think it's the Mission Bay clinic.", True),
        # ── MIXED: an answer wearing a question's shape. All must survive. ──
        ("mixed", "Which one — the Mission Bay clinic?", True),
        ("mixed", "It's Mission Bay Clinic, right?", True),
        ("mixed", "It's the Mission Bay clinic. Is that what you needed?", True),
        ("mixed", "Who's calling? Oh — she's at the Northgate campus.", True),
        ("mixed", "Mission Bay Clinic. Anything else?", True),
        # ── OUT OF SCOPE, asserted so the gap is not mistaken for a
        # regression. Negation is a different axis and is tracked separately:
        # this turn still grounds "Northside Medical Group" today and after.
        ("negation (untouched)", "We're not Northside Medical Group.", True),
    ]:
        check(_ta(_txt) == _want,
              f"turn asserts ({_kind}): {_want!s:5} for {_txt[:44]!r}")

    # ── A "?" THE CALLER DID NOT PUT THERE ──────────────────────────────────
    # The fixtures above are all PLACE answers, and every one of them is
    # rescued (when it should be) by a proper noun sitting beside a location
    # anchor. A CHOICE field has no proper noun, so for those four fields the
    # "?" conjunct stood alone and decided the question by itself.
    #
    # call-20260825-1847: agent "is this Dr. Carol, Neurosurgery, at New York
    # Presbyterian?", caller "Yes?" — a rising confirmation, punctuated by the
    # transcriber's ear. _turn_asserts returned False, the save was refused as
    # "they have only asked back", and the agent asked the IDENTICAL question
    # nine seconds later. classify_identity("Yes?") is `confirmed` and was
    # never consulted.
    from agents.voice.objectives import (classify_choice, classify_identity,
                                         classify_referral)
    _cs_sess = double(turns=[], doctor=double(hospital_name="Northside Medical Group",
                                         doctor_name="Dr. Jane Okafor"),
                 org_name="Definitive Healthcare", agent_name="David")
    for _txt, _cls, _state, _want, _why in [
        # THE LIVE CASE and its neighbours. Bare, classifies, rescued.
        ("Yes?",       classify_identity, "confirmed", True,  "the 1847 turn"),
        ("Yeah?",      classify_identity, "confirmed", True,  "bare, informal"),
        ("Correct?",   classify_identity, "confirmed", True,  "bare, formal"),
        ("Speaking?",  classify_identity, "confirmed", True,  "the phone idiom"),
        ("Yes, yes?",  classify_identity, "confirmed", True,  "repeated bare"),
        # Unchanged: no "?" at all, so the predicate never reaches the rescue.
        ("Yes.",       classify_identity, "confirmed", True,  "no mark at all"),
        # ── THE CONJUNCTS, each mutation-checked on its own ────────────────
        # Drop _ONLY_AFFIRM and this turn starts grounding a confirmation.
        ("Yeah, hi David, how are you?", classify_identity, "confirmed", False,
         "affirmative + a real question"),
        # Same, with our own words read back at us for confirmation.
        ("Yes, that's right?", classify_identity, "confirmed", False,
         "echoing us back, not telling us"),
        ("That's her?",  classify_identity, "confirmed", False, "asking us"),
        # Drop the classifier conjunct and ANY bare affirmative grounds ANY
        # state — "No?" would confirm the doctor.
        ("No?",        classify_identity, "confirmed", False, "wrong state"),
        # Screening questions must stay out however the field is configured.
        ("Sorry, who's calling again?", classify_identity, "confirmed", False,
         "screening"),
        # THE FIELD'S OWN VOCABULARY, not a shared one. A bare "No?" IS an
        # answer to "do you need a referral?" and is not an identity state.
        ("No?",        classify_referral, "no",        True,  "referral says no"),
        ("No?",        classify_choice,   "no",        True,  "accepting says no"),
    ]:
        check(rw._turn_asserts(_txt, _cs_sess, classifier=_cls, state=_state) == _want,
              f"turn asserts CHOICE ({_why}): {_want!s:5} for {_txt[:34]!r}")

    # THE DEFAULT ARGUMENTS ARE THE PLACE PATH, and it must not have moved.
    # Asserted as an IDENTITY, not as an absence: every fixture above is
    # re-run through the no-kwargs call and must give the same verdict it gave
    # before this parameter existed.
    check(all(rw._turn_asserts(_t, _cs_sess) == _w for _t, _w in [
              ("It's the Mission Bay Clinic.", True), ("Northgate.", True),
              ("Sorry, Northgate.", True), ("Yes, San Francisco.", True),
              ("She's in San Francisco, right?", False),
              ("Is she in San Francisco?", False),
              ("Sorry, who's calling again?", False)]),
          "PLACE callers pass no classifier and are byte-identical")
    # ...and "Yes?" WITHOUT the opt-in is still discarded, which is what makes
    # the kwargs load-bearing rather than decorative.
    check(rw._turn_asserts("Yes?", _cs_sess) is False,
          "a bare 'Yes?' still fails the PLACE path — the rescue is opt-in")

    # ── END TO END, THROUGH THE REAL GUARD ──────────────────────────────────
    # The unit checks above call _turn_asserts directly, so they would all keep
    # passing if the kwargs were dropped at the _ungrounded_choice call site
    # and the live path silently reverted. This is the check that cannot: it
    # reproduces call-20260825-1847's tool call and asks the shipped guard.
    _1847 = rw.RealtimeSession("CA0000000000000000000000id1847",
                               Doctor(doctor_name="Dr. Carol",
                                      specialization="Neurosurgery",
                                      hospital_name="New York Presbyterian"))
    _1847.note_utterance_rms(0.06)      # the line was audible, as measured
    _1847.add_turn("agent", "Thanks - is this Dr. Carol, Neurosurgery, "
                            "at New York Presbyterian?")
    _1847.add_turn("caller", "Yes?")
    _args1847 = {"identity": "confirmed", "heard": "Yes.",
                 "detail": "Dr. Carol, Neurosurgery at New York Presbyterian"}
    check(rw._ungrounded_identity(_args1847, _1847) == "",
          "call-1847: 'Yes?' now grounds the identity it always meant")
    # SELECTION STILL RUNS. The model wrote "Yes." and the caller said "Yes?";
    # the record must carry the caller's words, not the model's tidied copy.
    check(_args1847["heard"] == "Yes?",
          "and `heard` is the caller's real turn, not the model's version",
          _args1847["heard"])
    # THE GUARD IS NOT DISARMED. Same session shape, a state nobody uttered.
    _wrong = rw.RealtimeSession("CA000000000000000000000idwrong",
                                Doctor(doctor_name="Dr. Carol",
                                       hospital_name="New York Presbyterian"))
    _wrong.note_utterance_rms(0.06)
    _wrong.add_turn("agent", "Thanks - is this Dr. Carol at New York Presbyterian?")
    _wrong.add_turn("caller", "Sorry, who's calling again?")
    check(bool(rw._ungrounded_identity({"identity": "confirmed",
                                        "heard": "Yes."}, _wrong)),
          "a screening question still grounds nothing")

    # ── FIX 2: THE GUARD THE OTHER THREE LEAVE A HOLE FOR ───────────────────
    # _field_vocabulary is keyed on the field's own declared `states`. Bind it
    # wrong and the check either never fires or fires on another field's
    # answer — the failure objectives.states_in_its_own_right already paid for.
    from agents.voice.objectives import (CHOICE_STATES, IDENTITY_STATES,
                                         REFERRAL_STATES)
    for _states, _want, _name in [(IDENTITY_STATES, classify_identity, "identity"),
                                  (REFERRAL_STATES, classify_referral, "referral"),
                                  (CHOICE_STATES, classify_choice, "choice"),
                                  (frozenset(), None, "PLACE (no states)")]:
        check(rw._field_vocabulary(double(states=_states)) is _want,
              f"_field_vocabulary binds {_name} to its own classifier")

    _idf = double(name="identity", states=IDENTITY_STATES, label="the doctor")
    _reff = double(name="referral", states=REFERRAL_STATES, label="a referral")
    _fa = lambda turns, field, since: rw._field_already_answered(
        double(turns=[double(role=r, text=t) for r, t in turns]), field, since)
    check(_fa([("agent", "is this Dr. Carol?"), ("caller", "Yes?")], _idf, 0)
          == "Yes?", "an answered identity question is found")
    check(_fa([("agent", "is this Dr. Carol?"), ("caller", "Hello?")], _idf, 0)
          == "", "an unanswered one is not")
    check(_fa([("agent", "is this Dr. Carol?"), ("caller", "[...]")], _idf, 0)
          == "", "an untranscribed turn is not an answer")
    # THE INDEX IS LOAD-BEARING. Scan from 0 instead of the last ask and the
    # guard fires on the FIRST re-ask of a question answered long before.
    check(_fa([("caller", "Yes?"), ("agent", "is this Dr. Carol?")], _idf, 1)
          == "", "turns BEFORE the last ask do not count")
    # Cross-field: "Yes?" is an identity answer and not a referral answer.
    check(_fa([("agent", "do you need a referral?"), ("caller", "Yes?")],
              _reff, 0) == "", "identity words do not answer the referral field")

    # ── AN ANSWER TO ANOTHER QUESTION IS NOT AN ANSWER TO THIS ONE ────────
    # call-20260827-1010. The window opens at OUR ask and runs to now, and on
    # a multi-field call other questions get asked inside it. Vocabulary alone
    # cannot tell those answers apart — every CHOICE field's classifier reads
    # a bare "No." — so the scan tracks whose question is outstanding.
    #
    # WHY IT MATTERS MORE THAN A MISCOUNT: the nudge this feeds does not just
    # note the re-ask, it tells the model `they said X, take that as their
    # answer, record it`. On 1010 X was "No, I don't have it." — the answer to
    # the street address — offered as proof they had already given a new-patient
    # status they had never been asked for.
    _pvobj = get_template("provider_verification").objective
    _accf = next(f for f in _pvobj.fields if f.name == "accepting")
    _floor = lambda turns, since: rw._field_already_answered(
        double(turns=[double(role=r, text=t) for r, t in turns],
               objective=_pvobj), _accf, since)
    _addr_between = [
        ("agent",  "Is Dr. Jennifer currently taking new patients?"),
        ("caller", "Let me have a look."),
        ("agent",  "Sure - could you share the street address for the campus?"),
        ("caller", "No, I don't have it."),
        ("agent",  "No problem - so is she taking new patients at the moment?"),
    ]
    check(_floor(_addr_between, 0) == "",
          "call-1010: 'No, I don't have it.' answered the ADDRESS, not this",
          "the branch ask between the two took the floor")
    # The positive control, and the mutation that matters: strip the floor
    # tracking and the check above passes for the wrong reason on a call where
    # they really did answer.
    _really_answered = [
        ("agent",  "Is Dr. Jennifer currently taking new patients?"),
        ("caller", "No, we're not at the moment."),
        ("agent",  "Understood - and which branch is she at?"),
        ("caller", "Riverside."),
        ("agent",  "Thanks - is she taking new patients?"),
    ]
    check(_floor(_really_answered, 0) == "No, we're not at the moment.",
          "an answer given while THIS question was on the table still counts")
    # A later ask that names ours takes the floor back, so an answer after it
    # counts even though a different field held it in between.
    check(_floor(_really_answered + [("caller", "Actually no, she isn't.")], 0)
          == "No, we're not at the moment.",
          "and the re-ask itself reopens the floor for what follows it")

    # ── THEY ANSWERED SOMETHING NOBODY ASKED ──────────────────────────────
    # A front desk volunteers, and the ordinary path only ever looks at the
    # field on the table — so the second answer sits in the transcript and the
    # agent asks for it again four turns later. That is the "robotic loop"
    # complaint in its most concrete form, and it is a GUARD rather than a
    # prompt rule for the reason call-20260827-1130 settled: the recovery
    # directive already said "say just that, in one short sentence, do not
    # apologise" and the model disobeyed it four times inside one call.
    def _vsess(asked=(), mem=None):
        _s = rw.RealtimeSession("CA0000000000000000000000000vol1",
                                Doctor(doctor_name="Dr. Jennifer",
                                       hospital_name="New York Baptist"))
        _s.objective = _pvobj
        if mem:
            _s.memory.update(**mem)
        for _a in asked:
            _s._field_ask_at[_a] = 0
        return _s
    _pv_accf = next(f for f in _pvobj.fields if f.name == "accepting")
    _vf = lambda text, **kw: [(f.name, v) for f, v
                              in rw._volunteered_fields(_vsess(**kw), text)]
    check(_vf("She's at Riverside, and she's not taking new patients right now.")
          == [("accepting", "no")],
          "an answer volunteered inside another one is caught")
    check(_vf("A referral is always required for new patients.")
          == [("referral", "always")],
          "and so is one for a field several questions away")
    # THE MUTATION THAT MATTERS, and the whole safety of the guard. Every
    # CHOICE field shares classify_choice, so vocabulary ALONE would read a
    # bare "Yes." answering the branch question as "they volunteered that the
    # doctor is accepting new patients" — and the directive tells the model to
    # RECORD it. That is _field_already_answered's old defect arriving from the
    # other end, where it would put a value in the record nobody gave.
    for _bare in ["Yes.", "No.", "Yes, that is correct.", "Okay.",
                  "We only see patients on Tuesdays."]:
        check(_vf(_bare) == [],
              f"a turn that does not NAME the field volunteers nothing: "
              f"{_bare!r}",
              "the caller's own words have to be about it, exactly as "
              "_ungrounded_status requires when there is no ask to anchor to")
    # ONCE IT HAS BEEN ASKED, the ordinary path owns it — the ask budget, the
    # re-ask guard and the save gate all key off that ask.
    check(_vf("She is accepting new patients.", asked=("accepting",)) == [],
          "a field already asked for is not 'volunteered'")
    check(_vf("She is accepting new patients.",
              mem={_pv_accf.memory_key: "yes"}) == [],
          "and neither is one already collected")

    # END TO END, through the shipped handler.
    class _VolWS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(json.loads(s))

    _vs = _vsess(); _vw = _VolWS()
    _vs.add_turn("agent", "Which branch does Dr. Jennifer work out of?")
    _vs.add_turn("caller", "[...]")
    await rw._handle_caller_transcript(
        {"transcript": "She's at Riverside, and she's not taking new "
                       "patients right now."}, _vs, _vw)
    _said = [m["item"]["content"][0]["text"] for m in _vw.sent
             if m.get("type") == "conversation.item.create"]
    check(any("without being asked" in d for d in _said),
          f"the directive goes out on the turn that carried it ({_said})")
    check([r["field"] for r in _vs.volunteered_answers] == ["accepting"],
          f"and it is recorded, so a guard that acted leaves a trace "
          f"({_vs.volunteered_answers})")
    # ONE PER FIELD PER CALL. A second copy of a directive the model ignored is
    # context spent for nothing — the rule every other nudge here uses.
    _vw2 = _VolWS()
    _vs.add_turn("caller", "[...]")
    await rw._handle_caller_transcript(
        {"transcript": "She's really not taking new patients."}, _vs, _vw2)
    check(not any("without being asked" in
                  m.get("item", {}).get("content", [{}])[0].get("text", "")
                  for m in _vw2.sent),
          "the directive is one-shot per field")

    # END TO END: call-20260825-1847 replayed through the shipped handler.
    class _AskWS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(json.loads(s))

    async def _replay_1847(second_ask: str, answer: str = "Yes?"):
        _s = rw.RealtimeSession("CA000000000000000000000reask1",
                                Doctor(doctor_name="Dr. Carol",
                                       specialization="Neurosurgery",
                                       hospital_name="New York Presbyterian"))
        # PINNED HERE, EXPLICITLY. This suite fixes settings.call_template to
        # the single-field branch script, and under that objective there IS no
        # identity field — so the guard has nothing to recognise and every
        # check below passes vacuously. That is the trap the pin at the top of
        # main() was added to close, arriving from the other direction: a test
        # whose meaning depends on an env var, and one whose meaning depends on
        # the pin, fail the same way for the same invisible reason.
        #
        # Fix 2 is ABOUT multi-field objectives, so it must be tested against
        # one. Set on the session rather than on settings: _objective_of reads
        # sess.objective first, and mutating global config mid-suite is how the
        # ~20 checks that comment describes went wrong.
        _s.objective = get_template("provider_verification").objective
        _w = _AskWS()
        _q = "is this Dr. Carol, Neurosurgery, at New York Presbyterian?"
        await rw._handle_agent_transcript({"transcript": "Thanks - " + _q},
                                          _s, _w, "", False)
        if answer:
            _s.add_turn("caller", answer)
        await rw._handle_agent_transcript(
            {"transcript": "Got it, thanks for confirming - let me just clarify "
                           "one more detail about where she sees patients."},
            _s, _w, "", False)
        _s.add_turn("caller", "Okay.")
        await rw._handle_agent_transcript({"transcript": second_ask},
                                          _s, _w, "", False)
        return _s, [m["item"]["content"][0]["text"] for m in _w.sent
                    if m.get("type") == "conversation.item.create"]

    _s1847, _said = await _replay_1847(
        "Got it - is this Dr. Carol, Neurosurgery, at New York Presbyterian?")
    check(any("already answered" in d for d in _said),
          "call-1847: re-asking an ANSWERED question is now caught")
    # The identity clause reaching _ask_phrasings is what the one-line half of
    # the fix buys. Under `_is_location_ask` this set is empty and the verbatim
    # guard is blind to three of the four fields.
    check(any("carol" in p for p in _s1847._ask_phrasings),
          "the identity clause is recorded as an asked phrasing",
          str(sorted(_s1847._ask_phrasings)))
    check(any("exact words" in d for d in _said),
          "and the verbatim-phrasing guard fires on the repeat")
    check(_s1847._field_ask_at.get("identity") is not None,
          "the per-field ask index is maintained")

    # NOT ON A FIRST ASK, and not when they never answered. Both are the
    # mutations that make the guard fire on healthy calls.
    _s_first, _first_said = await _replay_1847(
        "And do you know which branch Dr. Carol works out of?")
    check(not any("already answered" in d for d in _first_said),
          "a DIFFERENT question after an answer is not a re-ask")
    _s_mute, _mute_said = await _replay_1847(
        "Got it - is this Dr. Carol, Neurosurgery, at New York Presbyterian?",
        answer="")
    check(not any("already answered" in d for d in _mute_said),
          "re-asking a question they never answered is still allowed")
    # ONCE PER CALL. A second copy of a directive the model already ignored is
    # context spent for nothing — same rule the other nudges in that block use.
    check(_s1847._answered_reask_nudged is True,
          "the nudge marks itself sent so it cannot repeat")

    # MULTI-TURN, the real call. Reproduced exactly: the city was only ever
    # asked about, the branch was stated. One must ground and the other must
    # not — which the blob could not express, because it had no per-turn view.
    _1703 = double(turns=[_hf("Okay, that's fine. I'm just trying to, then okay, "
                              "I'm sorry. She's in San Francisco, right?"),
                          _hf("Let me check that again."),
                          _hf("It's the Mission Bay Clinic")],
                   doctor=double(hospital_name="Northside Medical Group",
                                 doctor_name="Dr. Jane Okafor"),
                   org_name="Definitive Healthcare", agent_name="David")
    check(not rw._ungrounded_terms({"branch": "Mission Bay Clinic"}, _1703),
          "call-1703: the branch they STATED still grounds")
    check(bool(rw._ungrounded_terms(
              {"branch": "Mission Bay Clinic", "city": "San Francisco"}, _1703)),
          "call-1703: the city they ASKED about no longer does")
    # And the whole-call safety valve: if every turn is a question there is no
    # assertion to judge against, which must not block — same conservative
    # direction as an untranscribed call.
    _allq = double(turns=[_hf("Who's calling?"), _hf("Is this about a patient?")],
                   doctor=double(hospital_name="Northside Medical Group",
                                 doctor_name="Dr. Jane Okafor"),
                   org_name="Definitive Healthcare", agent_name="David")
    check(not rw._ungrounded_terms({"branch": "Northgate"}, _allq),
          "a call of nothing but questions does not block — it cannot judge")

    # RENDERING IS NOT SUBSTITUTION, and the first cut of this rule could not
    # tell them apart — it blocked "1825 Fourth Street" against a caller who
    # said "1825 4th Street", which throws away a correct address. That is the
    # expensive direction. A number-word grounds if the caller said the WORD or
    # said the DIGIT it stands for.
    check(not rw._ungrounded_terms(
              {"branch": "Mission Bay Clinic, 1825 Fourth Street"}, _addr_sess),
          "'4th' heard, 'Fourth' saved — rendering still passes")
    for _said, _val in [("She's at the Seven Hills clinic.", "Seven Hills Clinic"),
                        ("It's the Fourth Avenue site.",     "Fourth Avenue"),
                        ("One Medical on Broadway.",         "One Medical")]:
        check(not rw._ungrounded_terms({"branch": _val},
                                       double(turns=[_hf(_said)])),
              f"a real name whose number-word the caller said: {_val!r}")

    # THE REASON HAS TO REACH THE MODEL. _ungrounded_terms always computed a
    # specific one and the tool site discarded it for a generic line — the
    # same shape as 5aed263, where the failure reason was in every event and
    # was thrown away. Two rejections saying only "NEED: wording the caller
    # used out loud" are why the model reached for words: "out loud" reads as
    # "as spoken". It must no longer say that, and must pass the detail on.
    _gsrc = _plb.Path(_rwground.__file__).read_text(encoding="utf-8")
    check("REJECTED — {ungrounded} " in _gsrc,
          "the grounding rejection carries the specific reason")
    # Asserted POSITIVELY, on the clause that must be there. The obvious
    # version — "wording the caller used out loud" not in source — passes by
    # finding nothing, and here it would have failed on the comment explaining
    # the fix rather than on the message itself.
    check("| NEED: their own words, any number in digits" in _gsrc,
          "and the NEED clause asks for digits, not for spoken wording")
    # Still machinery, not speech — one of these was read aloud to a caller.
    _sv = rw._ungrounded_terms(
        {"branch": "mission bay clinic, eighteen forty fourth street"}, _bypass)
    for _ph in ("ask them", "could you", "tell me", "please provide", "you should"):
        check(_ph not in _sv.lower(),
              f"spelled-number rejection stays unspeakable ({_ph!r})")

    # ── The watchdog must not break a hold ───────────────────────────────────
    # call-20260819-1619. The caller said "give me a minute I just need to
    # check", the agent correctly answered "No rush." — and 7s later the
    # silence watchdog fired and made it ask again. Twice in one call, while
    # the caller was still looking:
    #
    #   16:20:08 AGENT  "is there a specific site name or street address?"
    #   16:20:18 AGENT  "could you share the exact site name, or the street
    #                    address for that location?"        <- nobody spoke
    #   16:20:26 caller "Yeah, give me a minute."           <- still looking
    #
    # The prompt already says "THE HOLD LASTS UNTIL THEY COME BACK WITH AN
    # ANSWER. Not one turn — the whole time." The model obeyed it. The
    # watchdog, which had no idea a hold was in progress, overrode it — a rule
    # the code enforces beating a rule the prompt requests, in the wrong
    # direction.
    import pathlib as _plb
    check(rw._HOLD_GRACE_S >= 30,
          "a hold stands the watchdog down long enough to actually look "
          "something up", f"{rw._HOLD_GRACE_S}s")
    check(rw._HOLD_GRACE_S < 120,
          "but is bounded, so a caller who never returns still gets checked on")
    _wd_body = re.search(r"async def _silence_watchdog.*?(?=\nasync def |\ndef )",
                         _PKG_SRC, re.S)
    # BEHAVIOURAL, not a source grep. Asserting that "_hold_until" appears in
    # the watchdog passed even with the check disabled as
    # `if False and time.time() < sess._hold_until` — the string was still
    # there. A guard aimed at the text instead of the behaviour, which is the
    # failure this suite keeps rediscovering. So: run the watchdog with a hold
    # in progress and require that it says nothing.
    _hs = _wd_sess(_hold_until=time.time() + 30)
    _hws, _hdone = _WS(), asyncio.Event()
    with mock.patch.object(rw, "_SILENCE_PROMPT_FIRST", 0.0),          mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _hwd = asyncio.create_task(rw._silence_watchdog(_hws, _hs, _hdone))
        await asyncio.sleep(1.4)
        _hdone.set()
        await asyncio.wait_for(_hwd, timeout=2)
    check(not _hws.sent,
          "the watchdog stays silent while the caller is on hold",
          f"{len(_hws.sent)} messages sent during a hold")
    # And it must resume once the hold expires, or a caller who never comes
    # back would leave the line dead forever.
    _es = _wd_sess(_hold_until=time.time() - 1)
    _ews, _edone = _WS(), asyncio.Event()
    with mock.patch.object(rw, "_SILENCE_PROMPT_FIRST", 0.0),          mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _ewd = asyncio.create_task(rw._silence_watchdog(_ews, _es, _edone))
        await asyncio.sleep(1.4)
        _edone.set()
        await asyncio.wait_for(_ewd, timeout=2)
    check(bool(_ews.sent),
          "and resumes once the hold has expired", f"{len(_ews.sent)} messages")
    # The utterances from that call must be recognised as holds at all.
    for _t in ["Okay, give me a minute I just need to check and I'll tell you.",
               "Yeah, give me a minute.",
               "Can you wait for a minute? I need to check."]:
        check(rw.is_hold_request(_t), f"hold recognised: {_t[:38]!r}")
    check(not rw.is_hold_request("He's working at the Northgate campus."),
          "an answer is not a hold")

    # ── Claimed done, saved nothing ──────────────────────────────────────────
    # Same call, and the expensive half. The caller gave "It's actually at 100
    # Main Street" — a valid location — and the agent said "Thanks for the
    # address, that's all I needed" and stopped. save_branch was never called;
    # resolved=False, branch=None. A resolvable call, answered, thrown away.
    #
    # The false-save guard cannot see this: it fires when save_branch is
    # REJECTED, and here it was never invoked. Both look identical to the
    # caller — told the job is done — but only one reached a guard.
    check(rw._claims_saved("Thanks for the address — that's all I needed."),
          "the completion claim from that call is detected")
    for _t, _want in [("I have everything I need, thanks.", True),
                      ("That's all I need — thanks for your time.", True),
                      ("Thanks — which branch is that?", False),
                      ("Sorry, I need one more detail.", False)]:
        check(rw._claims_saved(_t) == _want,
              f"completion claim: {_want!s:5} for {_t[:38]!r}")
    # It is DEFERRED, not fired on the transcript: the tool call for that same
    # response has not landed yet, so "never called" is not knowable there —
    # and firing early also fires on the rejected save, which is a different
    # failure with its own correction.
    _ht_body = re.search(r"async def _handle_agent_transcript.*?(?=\nasync def |\ndef )",
                         _PKG_SRC, re.S)
    check(_ht_body and "_claimed_done_at = time.time()" in _ht_body.group(0),
          "the transcript handler only records the claim, it does not correct it")
    check(_wd_body and "_claimed_done_nudged" in _wd_body.group(0),
          "the watchdog makes the correction, once the state has settled")
    check("sess._claimed_done_at = 0.0" in _PKG_SRC,
          "a rejected save cancels it — one correction per call, not two")

    # ── Backchannels ─────────────────────────────────────────────────────────
    # A human listener is not silent while you talk. The agent was, and on this
    # rig the callee then waits 1.9-3.1s with no evidence anyone is there.
    #
    # The audio is pre-rendered and pushed straight into the Twilio stream. It
    # deliberately does NOT go through the model: a response.create mid-
    # utterance collides with turn detection, gets cancelled by the caller's
    # own speech, and costs a response — exactly what barge-in exists to stop.
    from agents.voice import backchannel as _bc
    _v = settings.realtime_voice
    check(_bc.pick("no-such-voice-exists") is None,
          "no clips for a voice -> feature is simply off, not an error")
    if _bc.available(_v):
        _a = _bc.pick(_v)
        _b = _bc.pick(_v, exclude=_a)
        check(isinstance(_a, str) and len(_a) > 100,
              f"a clip for {_v!r} is returned base64-encoded for Twilio")
        check(_a != _b or _bc.available(_v) == 1,
              "the same clip is not repeated back to back",
              f"{_bc.available(_v)} clips")
        import base64 as _b64
        _raw = _b64.b64decode(_a or "")
        check(800 <= len(_raw) <= 9600,
              "a backchannel is 0.1-1.2s of 8kHz mu-law — longer is a turn, "
              "and talking over someone for that long is worse than silence",
              f"{len(_raw)/8000:.2f}s")
    else:
        check(True, f"no backchannel clips installed for {_v!r} — feature idle")
    # It must never become a turn. The whole point is that it is a noise: if it
    # entered the transcript it would inflate agent_turns, trip the repetition
    # detector, and be visible to the grounding guards.
    import pathlib as _plb
    _bc_src = (_plb.Path(rw.__file__).parent / "backchannel.py").read_text(encoding="utf-8")
    check("add_turn" not in _bc_src and "sess.turns" not in _bc_src,
          "the backchannel module cannot record a turn")
    _wd_src = re.search(r"async def _silence_watchdog.*?(?=\nasync def |\ndef )",
                        _PKG_SRC, re.S)
    check(_wd_src and "add_turn" not in _wd_src.group(0),
          "and the watchdog that sends it does not record one either")
    # Guard rails on when it may fire.
    check(rw._BACKCHANNEL_AFTER_S >= 2.0,
          "it waits until the caller is genuinely mid-utterance",
          f"{rw._BACKCHANNEL_AFTER_S}s")
    check(rw._BACKCHANNEL_COOLDOWN_S > rw._BACKCHANNEL_AFTER_S,
          "and cannot fire twice in quick succession — that is a tic",
          f"cooldown {rw._BACKCHANNEL_COOLDOWN_S}s")
    # ON now, and gated in the watchdog rather than merely by absent clips.
    # The speakerphone echo this used to be held back for is guarded at the
    # audio instead of watched for in the transcript — it could never have
    # been seen there, because our clips and a real caller 'Okay.' produce
    # the same string. See the BACKCHANNEL ECHO section below.
    check(isinstance(settings.realtime_backchannels, bool),
          f"backchannels flag is explicit: {settings.realtime_backchannels}")
    check(_wd_src and "settings.realtime_backchannels" in _wd_src.group(0),
          "and the watchdog checks the flag, so clips alone cannot enable it")

    # ── Hint regurgitation, quarantined at ingestion ─────────────────────────
    # The transcription hint is sent to the transcriber as `prompt`, so anything
    # in it can come back as transcript. Proven beyond argument on
    # call-20260819-1324, where the ENTIRE hint arrived as a caller turn.
    #
    # The architectural bug was not the hint. Every guard in realtime_worker
    # reads sess.turns as ground truth, and _is_hint_echo was only consulted
    # inside save_branch grounding — so a fabricated turn that triggered no
    # save entered the transcript unexamined. On 1324 that fed a 'Northwell'
    # the caller never said to _discarded_location, which blocked a LEGITIMATE
    # escalation and left the agent unable to end the call. Checking at one
    # consumer could never hold; it has to be quarantined at ingestion.
    _hint = tpl.transcribe_hint

    # (a) Verbatim run -> truncate. Not drop: on 1324 a caller genuinely said
    #     "We are having only one branch..." and the hint was APPENDED to it.
    _echo_turn = (
        "We are having only one branch, that is the downtown branch in Los Angeles. "
        "Phone call with a hospital or medical office receptionist. Health systems: "
        "Mercy, Ascension, CommonSpirit, Providence, Sutter, Kaiser Permanente, HCA, "
        "Tenet, Baptist, Methodist, Presbyterian, Mount Sinai, Cleveland Clinic, "
        "Mayo Clinic, Johns Hopkins, Banner, Advocate, Trinity Health, Northwell, "
        "NewYork-Presbyterian, Cedars-Sinai. Location words: campus, clinic, medical "
        "center, satellite office, north, south, east, west, downtown, midtown, "
        "uptown, suite, boulevard, avenue, parkway, drive, street.")
    _kept = rw._strip_hint_run(_echo_turn, _hint)
    check(_kept == "We are having only one branch, that is the downtown branch in Los Angeles.",
          "the recited hint is truncated and the caller's real words survive",
          _kept[:60])
    check("Ascension" not in _kept and "Cedars" not in _kept,
          "no fragment of the recited list is left behind")
    check(rw._strip_hint_run("She's at the Northgate campus.", _hint)
          == "She's at the Northgate campus.",
          "ordinary speech that merely uses hint vocabulary is untouched")

    # ── THE HINT IS RETIRED, AND THE DETECTOR IS NOT ────────────────────────
    # 2026-08-26: two calls destroyed in eight minutes by the transcriber
    # reciting our own prompt as the caller. 1633 collected NOTHING in 35s
    # because the caller's first turn came back as hint vocabulary; 1625 got
    # identity only in 94s. Removing the hint removes the source. Removing the
    # GUARDS instead would have been the opposite fix — see the grounding check
    # at the end of this block, which is what that would actually have cost.
    for _tname in ("forage_data_collection", "forage_ai_disclosed",
                   "provider_verification"):
        check(get_template(_tname).transcribe_hint == "",
              f"{_tname} sends no transcription prompt",
              "a hint is a sentence the transcriber can put in the caller's "
              "mouth; this one put two calls in the bin")

    # NOTHING ON THE REQUEST AT ALL, not an empty string. Whether an empty
    # prompt is still a prompt is a question worth not having.
    _no_hint = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check("prompt" not in _no_hint["input"]["transcription"],
          "an empty hint is omitted from the request, not sent as ''",
          f"{_no_hint['input']['transcription']}")
    _with_hint = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="something",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check(_with_hint["input"]["transcription"].get("prompt") == "something",
          "and a hint that IS set still reaches the transcriber",
          "the omission must be driven by emptiness, not hardcoded")

    # ── THE TRANSCRIPTION LANGUAGE IS A SEAM NOW, NOT A LITERAL ───────────
    # call-20260826-1656 returned three garbled caller turns, one of them a
    # line of Urdu on 3.30s of real audio with the agent channel silent. That
    # is not a fabrication - somebody spoke - but whether it was the caller in
    # Urdu or accented English forced through an "en" decode is unknown, and
    # cannot be settled without changing the value on one call.
    #
    # THE DEFAULT DOES NOT MOVE. "en" is what has always been sent and it is
    # what these assert; the seam only makes the A/B one setting instead of an
    # edit.
    check(settings.realtime_transcribe_language == "en",
          "the shipped transcription language is still en",
          f"{settings.realtime_transcribe_language!r}")
    _lang_default = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check(_lang_default["input"]["transcription"].get("language") == "en",
          "and omitting the argument reproduces it exactly",
          "a seam that changes behaviour by existing is not a seam")
    _lang_off = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="", language="",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check("language" not in _lang_off["input"]["transcription"],
          "an empty language is OMITTED, not sent as an empty string",
          str(_lang_off["input"]["transcription"]))
    _lang_ur = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="", language="ur",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check(_lang_ur["input"]["transcription"].get("language") == "ur",
          "and a value that IS set reaches the transcriber")
    check("settings.realtime_transcribe_language" in _PKG_SRC,
          "the live session config reads the setting, not a literal",
          "otherwise the A/B needs a code edit and will not be run")

    # THE DETECTOR SURVIVES THE RETIREMENT. _strip_hint_run searches the live
    # hint PLUS _RETIRED_HINT_TEXT, so a transcriber still primed from an
    # earlier session — or a future hint regression — is caught with the live
    # hint empty. These two strings are the verbatim `raw` values out of
    # call-1625 and call-1633's suppressed_echoes.
    _raw_1625 = ("Context: ###" + chr(10) + "Location words: campus, clinic, medical "
                 "center, satellite office, north, south, east, west, downtown, "
                 "midtown, uptown, suite, boulevard, avenue, parkway, drive, street. "
                 "Scheduling words: waitlist, waiting list, referral, new patients, "
                 "accepting, scheduling, insurance." + chr(10) + "###")
    _raw_1633 = ("Location words: campus, clinic, medical center, satellite office, "
                 "north, south, east, west, downtown, midtown, uptown, suite, "
                 "boulevard, avenue, parkway, drive, street. Scheduling words: "
                 "waitlist, waiting list, referral, new patients, accepting, "
                 "scheduling, insurance.")
    check(rw._strip_hint_run(_raw_1633, "") == "",
          "call-1633's recitation is still stripped with the hint retired",
          f"{rw._strip_hint_run(_raw_1633, '')!r}")
    check("waitlist" not in rw._strip_hint_run(_raw_1625, "").lower()
          and "campus" not in rw._strip_hint_run(_raw_1625, "").lower(),
          "and call-1625's, leaving no scheduling or location fragment",
          f"{rw._strip_hint_run(_raw_1625, '')!r}")

    # AND IT DOES NOT REACH FURTHER THAN IT DID. Every one of these uses the
    # retired vocabulary the way a receptionist actually would. Truncating any
    # of them would be the expensive direction — discarding a real answer to
    # avoid a fabricated one.
    for _real in ("Yes, she is at the downtown clinic on Oak Street.",
                  "We have a waitlist for new patients right now.",
                  "She works out of the Riverside campus, 1476 8th Street.",
                  "A referral is required if your insurance needs one.",
                  "The satellite office on Parkway Drive handles scheduling."):
        check(rw._strip_hint_run(_real, "") == _real,
              f"real speech survives: {_real[:44]!r}",
              "the retired text is matched as a RUN of six, never as vocabulary")

    # THE LOWERCASE WORDS DID NOT LEAK INTO THE MEMBERSHIP TEST. _FABRICATION_VOCAB
    # is built from CAPITALISED words in the retired text and is used with a
    # single-word membership check; admitting "clinic" or "waitlist" would start
    # discarding turns on one ordinary noun.
    for _common_word in ("clinic", "campus", "street", "waitlist", "referral",
                         "scheduling", "insurance", "downtown"):
        check(_common_word not in rw._FABRICATION_VOCAB,
              f"'{_common_word}' is not a fabrication marker",
              "a word callers say constantly cannot condemn a turn on its own")

    # ── WHAT REMOVING THE GUARDS WOULD HAVE COST ────────────────────────────
    # This is the check that answers "can we just delete the hint guards?".
    # Un-quarantined, 1633's fabricated sentence becomes a caller turn — and
    # then the grounding guard PASSES a branch of "downtown clinic", because
    # those words really do appear in what it believes the caller said. A call
    # that collected nothing would instead have written a fabricated address to
    # doctors.json marked verified.
    _fab = ("waitlist referral The downtown clinic is accepting new patients "
            "and scheduling appointments for the satellite office")
    _fs = rw.RealtimeSession("CA00000000000000000000fabri",
                             Doctor(doctor_name="Dr. Jennifer",
                                    hospital_name="New York Baptist Hospital"))
    _fs.add_turn("agent", "Which branch does Dr. Jennifer work out of?")
    _fs.turns.append(rw.TranscriptTurn(role="caller", text=_fab,
                                       timestamp="00:00:00", audio_rms=0.11))
    check(rw._ungrounded_terms({"branch": "downtown clinic"}, _fs) == "",
          "un-quarantined, the fabrication GROUNDS a branch it invented",
          "this is why the guards stay and the hint goes — deleting the guards "
          "turns a loud failure into a silent wrong row")

    # (b) Words on silence did not come from the caller. This catches the
    #     fabrication whose wording is too ordinary for a vocabulary test:
    #     1324's "Sure, our clinic is located on 123 Main Street, across from
    #     the Northwell campus" arrived at audio_rms 0.000259.
    _lvl2 = 0.135
    for _want, _label, _rms, _lv in [
        (True,  "1323 'Mercy Hospital' (rms 0.0114, no level yet)", 0.011417, None),
        (True,  "1324 silent 'Northwell campus'",                   0.000259, _lvl2),
        (True,  "1324 'Hello, how can'",                            0.005624, _lvl2),
        (False, "real: downtown branch",                            0.134868, _lvl2),
        (False, "real: quiet but audible",                          0.095400, _lvl2),
        (False, "unmeasured gets the benefit of the doubt",         None,     _lvl2),
    ]:
        check(rw._audio_carried_nothing(_rms, _lv) == _want,
              f"unevidenced-turn rule: {_want!s:5} for {_label}")

    # ── The adaptive quiet threshold ─────────────────────────────────────────
    # _LOW_AUDIO_RMS is an ABSOLUTE threshold on a quantity with no absolute
    # meaning: line gain, handset and distance all move it, so one constant
    # cannot be right for two calls. The threshold is therefore
    # max(absolute floor, this caller's own median * _QUIET_FRACTION).
    #
    # HISTORY, because it governs how far to trust this: the fraction was first
    # set to 0.35 to catch a "Mercy Medical Center" turn believed to be a
    # transcription fabrication. That accusation was RETRACTED — the audio is
    # real. It had been "measured" by slicing the LOCAL mix WAV at offsets
    # derived from transcript timestamps, and the local mix and the Twilio
    # recording are on different timelines (75.0s vs 68.2s for one call), so
    # empty windows were read as silence. This guard therefore has NO confirmed
    # positive case.
    #
    # Re-derived 2026-08-18 against 30 dual-channel recordings: for each, the N
    # loudest caller-channel bursts where N is the number of transcribed caller
    # turns, then min/median across them — how quiet a GENUINE turn gets
    # relative to that caller's own level.
    #     lowest 0.291   p10 0.458   median 0.766
    #     genuine turn below median*0.35 : 2/30   <- 0.35 was unsafe
    #     genuine turn below median*0.20 : 0/30
    _sess_lv = double(turns=[
        _TT(role="caller", text=t, audio_rms=r) for t, r in [
            ("Hello, who is this?", 0.1223),
            ("But why are you collecting this?", 0.0954),
            ("He's working at the Northgate campus.", 0.1532),
            ("It is Los Angeles only.", 0.0465),
        ]])
    _lvl = rw._caller_speech_level(_sess_lv)
    check(_lvl is not None,
          "a caller speech level is derived once there are enough turns")
    check(rw._QUIET_FRACTION <= 0.20,
          "the fraction sits where no genuine turn in 30 calls fell below it",
          f"{rw._QUIET_FRACTION}")
    # The half that matters most: a real turn must never be discounted. A
    # rejected genuine answer is the expensive failure for a directory.
    for _txt, _rms in [("Hello, who is this?", 0.1223),
                       ("He's working at the Northgate campus.", 0.1532),
                       ("It is Los Angeles only.", 0.0465)]:
        _t = _TT(role="caller", text=_txt, audio_rms=_rms)
        _w = [w.strip(".,!?") for w in _txt.lower().split()
              if w.strip(".,!?") not in rw._UNGROUNDED_STOPWORDS][:1]
        check(not rw._is_hint_echo(_t, _w, _lvl),
              f"real turn not discounted: {_txt[:34]!r}")
    # Near-silence still is, which is what the absolute floor was always for.
    check(rw._is_hint_echo(_TT(role="caller", text="Mercy", audio_rms=0.0008),
                           ["mercy"], _lvl),
          "a bare term on near-silence is still discounted")
    # Too few turns to have a median: fall back to the absolute rule rather
    # than computing a fraction of one sample, which is that sample and can
    # never be below itself — a check that silently never fires.
    _thin = double(turns=[
        _TT(role="caller", text="Mercy", audio_rms=0.05)])
    check(rw._caller_speech_level(_thin) is None,
          "no adaptive level until there are enough measured turns",
          f"{rw._MIN_TURNS_FOR_ADAPTIVE} needed")
    # The absolute stays a FLOOR: on a uniformly quiet call a fraction of a
    # small number must not drive the threshold toward zero.
    _quiet = double(turns=[
        _TT(role="caller", text=f"turn {i}", audio_rms=0.004) for i in range(3)])
    check(rw._is_hint_echo(_TT(role="caller", text="Mercy", audio_rms=0.002),
                           ["mercy"], rw._caller_speech_level(_quiet)),
          "the absolute threshold still applies as a floor on a quiet line")
    print("\n" + "=" * 66)
    print("  SCENARIO 0a3 — the agent must not repeat itself verbatim")
    print("=" * 66)
    _rp_sent, _rp_sess = await run_call(script_repeats_itself())
    _rp_nudges = [c["text"] for m in _rp_sent
                  if m.get("type") == "conversation.item.create"
                  for c in m["item"].get("content", [])
                  if c.get("type") == "input_text"]
    check(any("said the same" in n for n in _rp_nudges),
          "verbatim self-repeat is detected and the model is told",
          f"{len(_rp_nudges)} directives sent")
    # One-shot: a second copy of a directive the model already ignored is
    # context it pays for twice.
    check(sum("said the same" in n for n in _rp_nudges) == 1,
          "the self-repeat nudge is sent at most once per call")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a3 — barge-in inside the pre-audio window")
    print("=" * 66)
    _bi_out: dict = {}
    _bi_sent, _bi_sess = await run_call(script_barge_in_before_first_audio(), out=_bi_out)
    _bi_tw = _bi_out["twilio"].sent
    # Both halves of the contract. The cancel goes to OpenAI; the `clear` goes
    # to Twilio and is what actually stops audio reaching the caller's ear.
    # Only the first was ever reachable from run_call's return value, so the
    # half that matters on the phone had no assertion at all.
    check(any(m.get("type") == "response.cancel" for m in _bi_sent),
          "caller talking over a not-yet-audible response cancels it",
          f"{sorted({m.get('type') for m in _bi_sent})}")
    check(any(m.get("event") == "clear" for m in _bi_tw),
          "and Twilio is told to drop the audio it has buffered",
          f"{[m.get('event') for m in _bi_tw]}")
    # The flag that made this possible. Set on response.created, it answers
    # "is a response in flight"; the audio-delta point answered "is the agent
    # audible", which is what sess.agent_speaking is for. The silence watchdog
    # and the empty-response guard both read this and both wanted the former.
    check(rw.RealtimeSession("CA0000000000000000000000000abc",
                             Doctor(doctor_name="Dr. X"))._response_active is False,
          "_response_active starts False")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a6 — told them it was saved, then it wasn't")
    print("=" * 66)
    _fs_sent, _fs_sess = await run_call(script_announce_then_rejected())
    _fs_msgs = [c["text"] for m in _fs_sent
                if m.get("type") == "conversation.item.create"
                for c in (m.get("item", {}).get("content") or [])
                if c.get("type") == "input_text"]
    _fs = [t for t in _fs_msgs if "nothing has been recorded" in t]
    check(len(_fs) == 1,
          "a save-claim followed by a rejection is corrected exactly once",
          f"{len(_fs)} of {len(_fs_msgs)} injected")
    check(_fs and "as though the call is over" in _fs[0],
          "and the correction covers the 'we're all set' half, not just the save")
    check(_fs_sess is not None and not _fs_sess.memory.get("resolved"),
          "the call is still unresolved — nothing was written")

    # The detector, on the real utterances. Two families: a claim about the
    # TOOL ("I'll save that") and a claim about the CALL ("we'll be all set").
    # The model produced both in one sentence, and the second is worse because
    # it invites them to hang up.
    for _want, _txt in [
        (True,  "Thanks for checking — I’ll save that and then we’ll be all set."),
        (True,  "Perfect, thanks for confirming — I’ll save that and we’re done."),
        (True,  "Got it noted down."),
        (True,  "All set — thanks for your time."),
        (False, "Got it — which branch is she at?"),
        (False, "Sure, take your time."),
        (False, "I'm trying to find out which branch Dr. Okafor works at."),
        (False, "Is that the only location they have in Los Angeles?"),
    ]:
        check(rw._claims_saved(_txt) == _want,
              f"save-claim detector: {_want!s:5} for {_txt[:40]!r}")
    # Typographic apostrophes, since that is what the model actually emits and
    # what defeated _REPORTS_FAILURE until it was normalised.
    check(rw._claims_saved("I’ll save that") and rw._claims_saved("I'll save that"),
          "the detector sees both apostrophe spellings")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a5 — repair after an interruption")
    print("=" * 66)
    # Barge-in working is not the same as barge-in helping. On
    # call-20260818-1338 it fired correctly and made the call worse: truncated
    # to 750ms, the caller heard almost nothing, said "Hello.", and the agent
    # read that as filler and re-asked. A human cut off mid-sentence restates;
    # it does not ask again.
    #
    # This is deliberately NOT a Conversation Flow rule. The fact that
    # distinguishes a repair signal from filler — that we truncated, and to how
    # many ms — never appears in the transcript, so the model cannot condition
    # on it however the prose is worded. The process has it exactly.
    _rp_sent, _rp_sess = await run_call(script_cut_off_then_hello())
    _rp_msgs = [c["text"] for m in _rp_sent
                if m.get("type") == "conversation.item.create"
                for c in (m.get("item", {}).get("content") or [])
                if c.get("type") == "input_text"]
    _repair = [t for t in _rp_msgs if "cut off" in t]
    check(len(_repair) == 1,
          "a cut-off caller turn triggers exactly one repair directive",
          f"{len(_repair)} of {len(_rp_msgs)} injected messages")
    check(_repair and "not ask anything new" in _repair[0].lower(),
          "the directive says restate, not re-ask")
    check(_repair and "checking the line" in _repair[0],
          "and names what the caller's turn actually was")
    check(_rp_sess is not None and _rp_sess._repair_nudged,
          "the repair is one-shot, like every other injected directive")
    # The thresholds are the claim, so they get asserted rather than assumed.
    # A long truncation means they heard most of it and may have interrupted on
    # purpose — a normal move that needs no repair.
    check(rw._CUT_SHORT_MS <= 2000,
          "only SHORT truncations count as 'they heard nothing'",
          f"{rw._CUT_SHORT_MS}ms")
    check(rw._REPAIR_WINDOW_S <= 20,
          "and the window is bounded, so an early cut cannot colour a later turn",
          f"{rw._REPAIR_WINDOW_S}s")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a4 — OpenAI cancels the response, not us")
    print("=" * 66)
    _sc_out: dict = {}
    _sc_sent, _ = await run_call(script_server_side_cancel(), out=_sc_out)
    check(not any(m.get("type") == "response.cancel" for m in _sc_sent),
          "we did not cancel it — the server did")
    check(any(m.get("event") == "clear" for m in _sc_out["twilio"].sent),
          "a server-side cancel still drains Twilio's buffer",
          f"{[m.get('event') for m in _sc_out['twilio'].sent]}")

    # interrupt_response was never sent, so this ran on the API default and
    # nobody had chosen it. Declared now — asserted so it stays declared, since
    # an inherited default is exactly what made the interruption path
    # unobservable for eight sessions.
    _ic = rw.build_audio_config(
        transcribe_model="whisper-1", transcribe_hint="", audio_format="pcmu",
        noise_reduction="near_field", turn_detection="server_vad",
        eagerness="medium", voice="cedar")
    check("interrupt_response" in _ic["input"]["turn_detection"],
          "turn detection declares interrupt_response rather than inheriting it",
          f"{sorted(_ic['input']['turn_detection'])}")

    print("\n" + "=" * 66)
    print("  SCENARIO 0a2 — a REJECTED response must not be retried")
    print("=" * 66)
    _r_sent, _ = await run_call(script_rejected_response())
    _r_creates = [m for m in _r_sent if m.get("type") == "response.create"]
    check(len(_r_creates) == 1,
          "rejected response is not mistaken for dead air",
          f"{len(_r_creates)} response.create sent (greeting only expected)")

    print("\n" + "=" * 66)
    print("  SCENARIO 0b — must not hang up on someone going to check")
    print("=" * 66)
    _h_sent, _h_sess = await run_call(script_hold_then_escalate())
    _h_out = [json.loads(m["item"]["output"]) for m in _h_sent
              if m.get("type") == "conversation.item.create"
              and m["item"].get("type") == "function_call_output"]
    check(any(o.get("ok") is False for o in _h_out),
          "escalation refused while the caller is checking",
          str(_h_out[:1]))
    check(not _h_sess.memory.get("escalated"),
          "call is NOT marked escalated")
    check(not _h_sess.done, "call stays open so they can come back with it")
    # And the model is told the earlier directive no longer applies, otherwise
    # it will simply try to escalate again on the next turn.
    _texts = " ".join(
        c["text"] for m in _h_sent
        if m.get("type") == "conversation.item.create"
        and m["item"].get("type") == "message"
        for c in m["item"].get("content", []) if c.get("type") == "input_text")
    check("disregard the earlier instruction" in _texts,
          "the stop-and-escalate directive is explicitly withdrawn")

    print("\n" + "=" * 66)
    print("  SCENARIO 1 — happy path (branch obtained)")
    print("=" * 66)
    sent, sess = await run_call(script_happy_path())

    updates = [m for m in sent if m.get("type") == "session.update"]
    creates = [m for m in sent if m.get("type") == "response.create"]
    items   = [m for m in sent if m.get("type") == "conversation.item.create"]

    check(len(updates) == 1, "exactly one session.update", f"got {len(updates)}")
    session = updates[0]["session"]
    check(session["instructions"] == tpl.instructions,
          "instructions are the template's static text")
    check("Okafor" not in session["instructions"] and "Northside" not in session["instructions"],
          "no per-call data in instructions (cache prefix stable)")
    check(all("instructions" not in (c.get("response") or {}) for c in creates),
          "no response.create carries an instructions override",
          f"{len(creates)} response.create sent")
    check(session["audio"]["output"]["voice"] == settings.realtime_voice, "voice configured")
    check(session["audio"]["input"]["turn_detection"]["type"]
          == settings.realtime_turn_detection,
          f"turn detection configured ({settings.realtime_turn_detection})")
    # THE TWO LEGS NO LONGER AGREE, DELIBERATELY, and this check used to say
    # they must. Its stated reason — "the session speaks one codec and Twilio
    # hears another, silence on a connected call" — was never quite the risk:
    # Twilio's leg is 8kHz mu-law whatever OpenAI sends, because THIS PROCESS
    # sits between them and converts. What the old assertion actually pinned
    # was that we do no conversion, which is what foreclosed conditioning the
    # outbound audio at all.
    #
    # So the invariant is restated as what it was protecting: whatever OpenAI
    # sends must be a format this process knows how to put on the wire.
    _in_fmt = session["audio"]["input"]["format"]
    _out_fmt = session["audio"]["output"]["format"]
    check(_in_fmt["type"] == "audio/pcmu",
          "inbound is the codec Twilio already speaks — no conversion in")
    check(_out_fmt["type"] in ("audio/pcmu", "audio/pcm"),
          "outbound is a format this process can convert for Twilio", _out_fmt)
    check((_out_fmt["type"] == "audio/pcm") == rw._outbound_conditioned(),
          "the negotiated outbound format matches what the code will do with it")
    expected_fmt = ("audio/pcmu" if settings.realtime_audio_format == "pcmu"
                    else "audio/pcm")
    check(session["audio"]["input"]["format"]["type"] == expected_fmt,
          f"audio format is {expected_fmt}")
    check(session["audio"]["input"]["transcription"]["language"] == "en",
          "transcription pinned to en")
    check(session.get("max_output_tokens") == settings.realtime_max_response_tokens,
          "response token cap set")
    # And the cap has to clear the longest turn the script can legitimately
    # ask for. It counts AUDIO tokens (~20/s of speech) as well as text, which
    # is what made 400 look generous and truncate a live disclosure on
    # call-20260820-1230 at out_audio=151. The voicemail message is the long
    # pole: organisation, doctor, purpose, and an email read out character by
    # character — ~25s of speech, ~500 audio tokens plus its transcript.
    #
    # Asserted as a floor, not an equality: raising it further is fine, and
    # this must not become a test that has to be edited to tune a knob. What
    # it catches is the cap drifting back DOWN to where it silently cuts the
    # agent off mid-sentence.
    check(settings.realtime_max_response_tokens >= 1000,
          "token cap clears the longest legitimate turn (voicemail ~650 tok)",
          f"{settings.realtime_max_response_tokens} tok")
    # The log must show BOTH halves of what the cap counts. Showing only
    # out_audio is why the truncation was unexplainable from the log.
    _rw_src = _PKG_SRC
    check("out_text=" in _rw_src and "out_audio=" in _rw_src,
          "usage log shows out_text too — the cap counts it")

    ctx = items[0]["item"]["content"][0]["text"]
    check("CALL CONTEXT" in ctx, "per-call facts sent as a conversation item")
    check("Dr. Jane Okafor" in ctx, "context names the doctor")

    # ── BRANCH vs NEW EMPLOYER lives PER CALL, not in the cached prompt ──────
    # call-20260821-1304: record "Northside Medical Group", caller "She works
    # at a Methodist hospital in San Francisco", and branch "Methodist
    # Hospital" was written to the Northside listing stamped "verified against
    # caller transcript". Every save gate passed truthfully — grounding,
    # address, wrong-organisation — because none of them asks whether a branch
    # is a site OF THE RECORDED ORGANISATION.
    #
    # It is per-call because the rule cannot be stated without naming that
    # organisation. That also keeps it out of the 4,800-token ceiling, so no
    # already-proven rule had to be evicted to make room.
    check("BRANCH vs NEW EMPLOYER" in ctx,
          "the branch-vs-employer rule is in the per-call context")
    check("BRANCH vs NEW EMPLOYER" not in tpl.instructions,
          "and NOT in the cached instructions — the prefix stays byte-identical")
    check("A branch is a site OF Northside Medical Group" in ctx,
          "stated against THIS call's organisation, by name")
    for _clause in ("LEFT, MOVED, JOINED, TRANSFERRED", "NEW EMPLOYER",
                    "note_info", "Never save_branch it",
                    "NEVER INVENT AN AFFILIATION"):
        check(_clause in ctx, f"rule carries its load-bearing clause: {_clause!r}")
    # A bare site name must stay a branch candidate — this is the half that
    # protects the existing Methodist/Baptist/Northgate/Riverside saves.
    check("They simply name a site, with nothing to say it is a different "
          "employer -> branch candidate, normal flow." in ctx,
          "a plain site name is still a branch candidate")
    # AND IT MUST NOT ASSERT A FACT THE TRANSCRIPT CANNOT CARRY. Naming a real
    # health system as "a different organisation" would be the same fabrication
    # in the opposite direction — the examples are placeholders on purpose.
    for _org in ("Methodist", "Baptist", "Mercy", "Mayo"):
        check(_org not in ctx,
              f"context claims no organisation is unaffiliated: {_org!r}")
    # The rule needs an organisation to be about. No record, no rule.
    _nohosp = tpl.build_context(Doctor(doctor_name="Dr. Jane Okafor"),
                                callback_number="+15706532193",
                                callback_email="a@b.com",
                                org=settings.org_name, agent_name="Alex")
    check("BRANCH vs NEW EMPLOYER" not in _nohosp,
          "omitted when there is no organisation on record to compare against")
    # Template 1 is truthful about WHO and WHY — it names the organisation and
    # uses no pretext. It does NOT announce itself as automated; that is the
    # forage_ai_disclosed variant, asserted separately below.
    # The organisation is a per-call runtime value now, not a template constant.
    # It must reach the model through the CONTEXT item, never the instructions —
    # it used to sit 14 tokens into a ~4,000-token prompt, so changing clients
    # invalidated 99% of the cache prefix.
    check(settings.org_name in ctx, "context names the organisation")
    # "on behalf of", never "with"/"from". Sarah is not an employee of the client
    # — the calling entity is a different company — so "Sarah with <client>" is a
    # false claim about who the receptionist spoke to, and it does not survive
    # them checking later. Same category of invented identity as the fake
    # "Forage AI Healthcare" org this replaced; it just reads more naturally,
    # which is exactly why it would slip back in.
    _greet = tpl.build_greeting(
        Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside Medical Group"),
        org=settings.org_name)
    # EVERY template, not just the configured one. Asserting on `tpl` alone let
    # forage_ai_disclosed keep "I'm an automated assistant FROM {org}" — the
    # exact employment claim that had just been removed from the human greeting.
    # Same shape as the pronunciation line: fixed in one place, silently kept in
    # the other. A per-template loop is the only thing that catches it.
    from agents.voice.templates import TEMPLATES as _ALL
    _probe = Doctor(doctor_name="Dr. Jane Okafor",
                    hospital_name="Northside Medical Group")
    for _name, _t in _ALL.items():
        _g = _t.build_greeting(_probe, org=settings.org_name)
        check("on behalf of " + settings.org_name in _g,
              f"{_name}: says 'on behalf of', not claiming employment", _g[:52])
        for _claim in (f"with {settings.org_name}", f"from {settings.org_name}",
                       f"at {settings.org_name}"):
            check(_claim not in _g,
                  f"{_name}: no employment claim {_claim!r}")
        # IF THIS FAILS, THE ASSERTION MAY BE WHAT IS WRONG — it encodes a
        # judgement about openers that has now changed once.
        #
        # It used to read `"?" not in _g`, banning every question because the
        # hospital-confirmation question had been ignored by 10 of 11 callees.
        # That over-generalised from one dead question to all questions, and
        # what replaced it was a full stop, which hands over no turn at all. On
        # call-20260813-1409 the callee did not know what was wanted, answered
        # with noise, and the opener spent the next forty seconds being
        # recovered by the silence watchdog.
        #
        # The real distinction is what the question is FOR. A confirmation
        # question asks for something the callee gains nothing by answering. The
        # actual ask gives them something concrete to respond to and advances
        # the call in the same breath. So: no hospital confirmation, and end on
        # a question.
        # Not `(_probe.hospital_name or "")` — an empty needle is in every
        # string, so a Doctor with no hospital would make this pass for the
        # wrong reason. The name has to be there for the check to mean
        # anything, so say so.
        check(bool(_probe.hospital_name), f"{_name}: probe has a hospital name")
        check(str(_probe.hospital_name) not in (_g or ""),
              f"{_name}: opener does not spend itself confirming the hospital",
              (_g or "")[:52])
        check((_g or "").rstrip().endswith("?"),
              f"{_name}: opener ends on the ask, handing over the turn",
              (_g or "")[-40:])
        check(settings.org_name not in _t.instructions,
              f"{_name}: organisation stays out of the cached instructions")

    # The persona name must match the voice. A cedar (male) call introduced
    # itself as Sarah and the caller spent three of six turns on it — "why is
    # your name Sarah? I think you're a boy" — and never gave the branch. Voice
    # and name were independent settings with nothing checking they agreed.
    from core.config import persona_for_voice as _persona, VOICE_PERSONA as _VP, \
        REALTIME_VOICES as _VOICES
    check(set(_VP) == _VOICES,
          "every valid voice has a persona name",
          f"missing: {sorted(_VOICES - set(_VP))}")
    check(_persona("marin") == "Sarah", "female voice keeps a female name")
    check(_persona("cedar") == "David", "male voice gets a male name")
    check(_persona("") != "" and _persona("nonsense") != "",
          "unknown voice still yields a usable name")
    for _v in sorted(_VOICES):
        _g = tpl.build_greeting(_probe, org=settings.org_name,
                                agent_name=_persona(_v))
        check(_persona(_v) in _g, f"greeting uses the {_v} persona name")
    # Derived per call, so it must not be baked into the cached prefix — the
    # marin/cedar A/B otherwise costs a cold cache each time it switches.
    for _n, _t2 in _ALL.items():
        for _name_val in set(_VP.values()):
            check(_name_val not in _t2.instructions,
                  f"{_n}: persona {_name_val!r} stays out of the instructions")
    check(settings.org_name not in tpl.instructions,
          "organisation absent from the cached instructions")
    for _other in ("Definitive Healthcare", "Forage AI Healthcare", "Acme Health"):
        check(_other not in tpl.instructions,
              f"no hardcoded client in instructions: {_other!r}")
    # Purpose, not a company description. "we keep a directory of doctors up to
    # date" is how a brochure explains an employer; "about a doctor listing" is
    # how a person says why they rang.
    check("doctor listing" in ctx or "directory of doctors" in ctx,
          "greeting states the purpose")
    # Not "opens with a softener" — that assertion was written for a greeting
    # that turned out to be British-sounding ("oh hi, sorry to bother you —
    # is that...?"). What actually matters is US phone convention and one
    # breath: name, company, why, then the confirmation question.
    greeting = tpl.build_greeting(
        Doctor(doctor_name="Dr. Jane Okafor",
               hospital_name="Northside Medical Group"))
    check(greeting.count(".") + greeting.count("?") <= 2,
          "greeting is one or two sentences, not a paragraph")
    check(len(greeting.split()) <= 24,
          f"greeting stays short ({len(greeting.split())} words)")
    # The opener ASKS, it does not instruct. A bare wh-question ("which branch
    # is Dr. X working out of?") presupposes an answer and offers no way out,
    # and it arrives before the callee has said anything — on
    # call-20260819-1619 they replied "Hello David, good evening. How can I
    # help you?", resetting the exchange back to a normal opening, which is
    # what people do when someone skips one.
    #
    # This also contradicted the prompt's own rule — "You are asking a favour
    # of someone at work: 'do you know...'" — with a worked Right example using
    # exactly that form. Every other ask in the file is softened; the one
    # sentence the callee hears first was not.
    for _n5, _t5 in _ALL.items():
        _g5 = _t5.build_greeting(
            Doctor(doctor_name="Dr. Jane Okafor",
                   hospital_name="Northside Medical Group"))
        _low5 = _g5.lower()
        # A PERMISSION QUESTION QUALIFIES, and qualifies harder than the rest.
        # The rule exists so the opener is a request rather than an
        # instruction; "is now a good time?" is not merely softened, it offers
        # a way out, which is what the softening was standing in for.
        check(any(s in _low5 for s in ("do you know", "any chance",
                                       "would you know", "i'm hoping",
                                       "could you tell", "good time")),
              f"{_n5}: the opener softens its ask rather than instructing",
              _g5[-46:])
    # Length is the reason the softener had to be paid for elsewhere, not a
    # nicety: the greeting is already 6.5-7.5s of speech on live calls before
    # the callee can speak at all. If this budget ever needs raising, shorten
    # something rather than extending the opener.
    check(len(greeting.split()) <= 24,
          "and it is paid for in the same breath, not by running longer",
          f"{len(greeting.split())} words")
    # "this is <name>" is the US convention and stays. The "<name> with <company>"
    # half was dropped deliberately: it is the natural phrasing, and it is a
    # false employment claim. Accuracy wins over idiom on the identity line.
    check("this is" in greeting.lower(),
          "uses US phone convention: 'this is <name>'")
    check(any(i["item"].get("type") == "function_call_output" for i in items),
          "tool result returned to the model")

    # A truthful script must never recite an unreachable callback number.
    from agents.voice.templates import is_usable_callback_number
    if not is_usable_callback_number(settings.callback_number):
        check(settings.callback_number not in ctx,
              "unusable CALLBACK_NUMBER withheld from the call context")
        check("NONE AVAILABLE" in ctx,
              "agent told explicitly that no callback number exists")
    # Match against whitespace-normalised text throughout: the instructions are
    # hard-wrapped, so an assertion must not depend on where a line happens to
    # break. Two of these failed on the consolidation purely from rewrapping.
    # One whitespace-normalised copy of the instructions. Assertions match
    # against this so they never depend on where a hard-wrapped line breaks.
    flat = " ".join(tpl.instructions.split())
    check("NEVER invent, guess, or approximate a phone number" in flat,
          "instructions forbid inventing a phone number")
    check("repeat it plainly and in full" in flat,
          "mid-call identity re-ask is handled")
    # ...and answering WHO must END there. The rule used to say "give your
    # name and the organisation, however many times they ask", which is what
    # _is_reintroduction flags as re-delivering the greeting — a genuine
    # prompt/guard contradiction, and the prompt wins because it is in
    # context from turn one while the guard arrives after the fact.
    # call-20260820-1440: "Sorry, who's calling again?" -> "Oh, sorry Varun —
    # I'm David, calling on behalf of Definitive Healthcare." Correct answer,
    # flagged as a fault. Asserted positively: an absence check on the old
    # wording would pass by finding nothing.
    check("do not re-run the opening line" in flat,
          "answering WHO does not re-deliver the greeting")
    check("do not put the branch question on the end" in flat,
          "and does not staple the branch question onto the identity answer")
    check("EXCEPTION: identity and contact facts" in flat,
          "identity facts exempt from the no-repetition rule")

    # The brevity rules once combined to make 5 of 6 agent turns bare
    # questions, including answering a caller's direct question with another
    # question. Guard the counterweight so tightening brevity again cannot
    # silently reintroduce it.
    check("answer it before asking anything of your own" in flat,
          "caller questions get answered before the agent asks its own")
    check("never reply to a question with only a question" in flat,
          "never answers a question with only a question")
    # A live call welded the same branch question onto four consecutive
    # answers. The prose rule against it was already there and was ignored
    # every time, so it is now a hard constraint on the shape of the output.
    # NOT "never ask while answering" — that rule shipped, worked (staple_rate
    # 100% -> 50%), and produced 13 seconds of dead air while a confused caller
    # asked "hello, are you there?". Ending a turn with nothing to respond to
    # is worse than asking. The failure was always repetition, not the question.
    check("asking in the same breath is how a person hands the" in flat,
          "answering and asking in one turn is permitted")
    check("Repetition is what makes people hang up" in flat,
          "repetition named as the actual failure")
    # A live call answered "what's the reason for calling?" by repeating its
    # name, org and job — all three already said in the greeting 15 seconds
    # earlier — and never mentioned what it wanted. The caller replied "What
    # should I do?". WHO and WHY are different questions.
    check("A job description is not a reason for calling" in flat,
          "why-are-you-calling is answered with the actual ask")
    check("This covers WHO you are. It does NOT cover WHY you are calling" in flat,
          "identity-repetition exemption does not extend to purpose")
    check("never ask for the branch twice in the same wording" in flat,
          "cannot repeat the branch question verbatim")
    # This used to assert the literal 'say ONLY "Of course, take your time."',
    # which is the instruction that produced the verbatim repeat. The assertion
    # was pinning the wording rather than the behaviour, so it locked the bug in
    # place and would have failed the fix. Assert what must be true instead:
    # acknowledge, stop, ask nothing.
    check("acknowledge in ONE short line, then STOP" in flat,
          "a hold request gets a short acknowledgement and nothing else")
    check("Do not re-ask, do not rephrase the question" in flat,
          "a hold request gets no follow-up question")
    check("THE HOLD LASTS UNTIL THEY COME BACK WITH AN ANSWER" in flat,
          "the hold persists across turns, not just one")
    # Normalise whitespace — the instructions are hard-wrapped, so asserting on
    # a phrase that spans a line break must not depend on where it wraps.
    # (flat defined above)
    check('Wrong: "Which branch is she at?"' in flat,
          "a bare question with no reaction is shown as wrong")
    # Hospitals have branches, not offices — and the field we store into is
    # literally called `branch`, so asking "which office" is inconsistent with
    # both the domain and the schema.
    check("say BRANCH, not office" in flat,
          "agent asks for a branch, not an office")
    check('Never "which office"' in flat,
          "'which office' explicitly ruled out as the agent's wording")
    check("Wrong: \"Which branch is she at?\"" in tpl.instructions,
          "bare-question turn shown as a counter-example")

    # A caller who said only "Bye." was told "Thanks for checking" — thanked
    # for help never given, as the last thing they heard.
    check("THANK THEM FOR WHAT THEY ACTUALLY DID" in tpl.instructions,
          "closing is tied to the actual outcome")
    # Three turns from call-20260818-1112 that the prompt had no rule for.
    # Asserted on both templates: this file already has a bug class from fixing
    # one template and leaving the other, and a caller asking "is it an
    # emergency?" gets the same wrong answer whichever identity block is loaded.
    for _n3, _t3 in _ALL.items():
        _f3 = " ".join(_t3.instructions.split())
        # "May I know why are you calling? Is it an emergency call?" — the agent
        # answered WHY and dropped the yes/no entirely.
        _em = _f3[_f3.find("Is this an EMERGENCY"):][:300]
        check("EMERGENCY" in _f3 and "say NO" in _em,
              f"{_n3}: prompt answers the is-this-an-emergency question")
        # It used to require the literal phrase "nothing urgent", which is
        # the sentence the prompt handed over as a quoted example — and the
        # model reproduced it verbatim in all 11 turns across 60 calls that
        # mentioned urgency, including 3 where the caller had asked only
        # about a patient. The test was pinning the defect in place. Assert
        # that the question is ANSWERED, never in which words.
        # "Is it urgent?" and "is it about a patient?" are two questions with
        # two answers, and merging them costs the second one. The 2026-08-20
        # deletion pass folded the patient question into the EMERGENCY branch
        # to save lines, which handed it the emergency ANSWER — and on
        # call-20260820-1321 the agent said "No, nothing urgent — it's a
        # listing check" to "is it about a patient?", leaving the actual
        # question hanging. _asks_about_patient exists precisely to stop that
        # ("answering only the 'urgent' half leaves them guessing") and lost:
        # its nudge is injected when the transcript lands, which is after
        # OpenAI's VAD has already begun the reply, so the prompt wins the
        # race whenever generation starts first. A guard that races cannot be
        # the only thing saying this.
        check("about a PATIENT" in _f3,
              f"{_n3}: the patient question has its own branch")
        _pat0 = _f3[_f3.find("about a PATIENT"):][:300]
        check("say NO" in _pat0 or "Say NO" in _pat0,
              f"{_n3}: and its own answer, in its own terms")
        # THE NEW INVARIANT. Neither branch may hand over a ready-made
        # sentence. A quoted example is the one thing in a prompt that gets
        # copied rather than varied, whatever line 93 says about patterns
        # to vary from, and copying it is how one answer reached the other
        # question. Describe the answer; never write it out.
        # Quoting the caller's QUESTIONS is fine and useful — that is how the
        # branch is recognised. Quoting a STATEMENT is not: a declarative in
        # quotes is a script, and a script gets copied rather than varied.
        # Every quoted span in these two branches must end in a question mark.
        _both = _em + " " + _pat0
        _quoted = re.findall(r'"([^"]{4,})"', _both)
        _scripts = [q for q in _quoted if not q.rstrip().endswith("?")]
        check(not _scripts,
              f"{_n3}: neither branch quotes a sentence for the agent to say",
              "; ".join(_scripts)[:110] or f"{len(_quoted)} quoted questions, 0 scripts")
        # AND THE GUARD MUST NOT PRIME WHAT IT CORRECTS. _asks_about_patient
        # injects a directive when the caller asks about a patient, and that
        # text used to contain the word it exists to suppress ("answering only
        # the 'urgent' half"). It lands in context immediately before the
        # model speaks. On call-20260821-1952 the caller asked only about a
        # patient and got both answers stapled together — "No, nothing urgent
        # — it's just about the listing. No, no patient is involved here."
        # Two nos, two answers, one question. A directive says what to answer,
        # never what not to.
        _dsrc = _PKG_SRC
        _k = _dsrc.find("they asked whether this is about a ")
        _dir = _dsrc[_k:_dsrc.find(")\")}]}", _k)] if _k > 0 else ""
        check(_dir, "found the patient-question directive")
        check("urgent" not in _dir.lower(),
              "the patient directive never says the word it is correcting",
              _dir[:100])
        _pat = _f3[_f3.find("about a PATIENT"):][:240]
        check("DIFFERENT question" in _pat,
              f"{_n3}: marked as distinct from the urgency question")
        # It opened that same turn with "It's just me, calling on behalf of..."
        # — a phrase that identifies nobody, from a stranger on their phone.
        check("it's just me" in _f3.lower(),
              f"{_n3}: prompt bans the identifies-nobody non-answer")
        # "Need anything?" — an open offer, answered with nothing.
        # MOVED OUT OF THE PROMPT 2026-08-27. _invites_continuation existed and
        # had two callers, both defensive: _caller_is_vetting and the escalate
        # blocker. Neither fires in the ordinary mid-call case, so the rule was
        # carried as 68 tokens of prose the model had to recall 4,000 tokens
        # later at the one moment it mattered. The assertion follows the
        # enforcement — see the injection check below.
        check("They OFFER to help" not in _f3,
              f"{_n3}: the offer rule is no longer carried as prose",
              "asserted positively at the guard, so this absence is not the "
              "only thing standing between the rule and nothing")
        # "Can you share those details with me?" — answered by reciting what was
        # on the record, to someone unverified.
        check("collect this information, not" in _f3,
              f"{_n3}: prompt refuses to read the record back to the caller")
        # And the rejection-handling rule that lost the branch — MOVED into the
        # rejection itself 2026-08-27. save_branch had carried `RE-READ` in its
        # error since call-20260820-1321; the four choice fields got the reason
        # and nothing about where to look, and the prompt covered the gap for
        # all five. A tool result is not the cached prefix, so unifying it cost
        # nothing and let 65 tokens go.
        check("RE-READ WHAT THEY ACTUALLY SAID" not in _f3,
              f"{_n3}: the re-read rule is no longer carried as prose",
              "asserted positively against both rejection messages below")
    # ── THE THREE RULES THAT MOVED FROM THE PROMPT INTO THE PROCESS ───────
    # 2026-08-27. Each of these was prose the model had to recall; each is now
    # a directive that arrives at the moment it applies. The absence checks
    # above are worth nothing on their own — an absence assertion passes by
    # finding nothing the day a wording changes — so every one of them is
    # paired here with a positive check on the enforcement that replaced it.
    #
    # 1. RE-READ now rides on the rejection, for ALL FIVE fields.
    _gsrc3 = _PKG_SRC
    check(_gsrc3.count("RE-READ: ") >= 2,
          "both the branch and the choice rejections carry RE-READ",
          "save_branch had it since 1321; the four choice fields did not, and "
          "the prompt was covering the difference for all five")
    _choice_err = _gsrc3[_gsrc3.find('f"NOT SAVED — {ungrounded_choice} '):][:300]
    check("RE-READ" in _choice_err and "NEED:" in _choice_err,
          "the choice-field rejection says WHERE to look, not just what failed",
          _choice_err[:90])
    # 2. The silence watchdog ends the call instead of falling silent.
    check("no_response" in rw.GIVE_UP_REASONS
          and "no_response" in rw.GIVE_UP_MARKERS,
          "the watchdog's exit is a real give-up trigger with its own marker")
    _wd = _PKG_SRC[_PKG_SRC.find("if _used >= _MAX_SILENCE_PROMPTS:"):][:900]
    check("give_up_directive" in _wd and "no_response" in _wd,
          "spending the silence budget closes the call through the same "
          "directive the ask budget uses",
          "it used to `continue` forever, holding a line nobody was on")
    check(rw.GIVE_UP_MARKERS["no_response"]
          in rw.give_up_directive(
              double(_unanswered_asks=0, _asks_without_progress=0),
              "no_response"),
          "and the no_response directive really contains its marker",
          "a trigger whose marker does not appear in its own text makes every "
          "absence assertion downstream vouch for nothing")
    # 3. An offer of help is spent at the moment it is made.
    _off = _PKG_SRC[_PKG_SRC.find("if (_invites_continuation(text) and not sess.done"):][:900]
    check(_off and "_offer_nudged" in _off,
          "an offer of help now injects mid-call, one-shot",
          "_invites_continuation had two callers and both were defensive")
    check("just offered to help" in _off,
          "and the directive tells the agent to spend it, not return it")

    # The prose rule "Never claim to have noted, saved, or recorded a location
    # you were not given" was DELETED from the prompt on 2026-08-20. It was
    # observed failing on call-20260818-1613 (told the caller it was saved
    # 0.0s before the save was rejected) and again on call-20260819-1619, and
    # the guard written for it — _claims_saved, feeding both the tool-site
    # nudge and the claimed-done watchdog — is what actually holds. The guard's
    # own comment says so: "The prompt already carries [it] and it did not
    # hold." Carrying both left the model arbitrating a rule that never won.
    #
    # Asserting the prose is now ABSENT would be the wrong invariant: it passes
    # by finding nothing, so it would pass just as well on the day someone
    # deletes the guard as well. Assert what must be true — the rule is still
    # enforced, somewhere — which passes only by finding something.
    check(rw._claims_saved("Thanks — I've got that saved, that's all I needed."),
          "the deleted false-save rule is still enforced, in code")
    # Directives are injected as role:"user" input_text items — the standard
    # workaround, but it means a fake caller utterance is one bug away from
    # landing in the saved transcript and quietly polluting the dataset.
    # They cannot reach add_turn (which only fires on audio-derived events),
    # and this asserts it rather than trusting it.
    for t in sess.turns:
        check("(system:" not in t.text,
              f"no injected directive leaked into the transcript as a turn")
        break
    check(not any("(system:" in t.text for t in sess.turns),
          "transcript contains no injected system directives")
    check(not any(t.role == "caller" and "goodbye now" in t.text for t in sess.turns),
          "transcript contains no injected closing prompt")
    # This was written when there were TWO injection paths. There are now five —
    # ask budget, closing, silence watchdog, hold-cancels-escalation, and the
    # caller-repeated nudge — and a scenario that never triggers one gives only
    # incidental coverage. So assert against every directive the module can send,
    # found by reading the source rather than by remembering to add them here.
    # ACROSS THE PACKAGE, not one file. Four directives moved to grounding.py
    # in the 2026-08-26 split and this scan lost them silently: it still found
    # the rest, still passed, and simply stopped asserting anything about the
    # four it no longer looked at. The count went 1,845 -> 1,841 and nothing
    # went red — the third time this particular check has under-counted, after
    # the '[a-z]{4,}' pattern and the de-duplicating set. The population is the
    # DIRECTIVES THIS PACKAGE INJECTS, and that was never a property of a file.
    _worker_src = NL.join(
        _p.read_text(encoding="utf-8")
        for _p in sorted(pathlib.Path("agents/voice").glob("*.py")))
    # The first word may be short. '[a-z]{4,}' silently skipped every directive
    # opening with a three-letter word, which was both of the ones beginning
    # "you ..." — including the ask-budget directive that ENDS the call. The
    # test reported 5 paths while the module had 7, so the point of deriving
    # them from source (catch a new path the day it lands) was defeated by the
    # pattern used to derive them.
    # Counted as a LIST, not a set. Two directives can legitimately open with
    # the same words — the hold block and the invitation block both begin
    # "disregard the earlier instruction to stop and escalate", because they
    # are retracting the same directive for different reasons. De-duplicating
    # first made the count read 16 against 17 declared and looked exactly like
    # a directive the pattern could not see.
    _found = re.findall(r'"\(system: ([a-z][a-z ]{9,})', _worker_src)
    _injected = set(_found)
    _declared = _worker_src.count('"(system: ')
    check(len(_found) == _declared,
          "every injected directive is found, none skipped by the pattern",
          f"{len(_found)} found vs {_declared} in source")
    check(len(_injected) >= 7,
          "found the module's injected directives to check against",
          f"{len(_injected)} paths")
    for _phrase in _injected:
        check(not any(_phrase in t.text for t in sess.turns),
              f"injected directive stays out of the transcript: {_phrase[:34]!r}")

    check(sess.memory.get("branch") == "Northgate Campus", "branch saved to memory")
    check(bool(sess.memory.get("resolved")), "call marked resolved")

    print("\n" + "=" * 66)
    print("  SCENARIO 2 — refusal ('I'm not allowed to give out that information')")
    print("=" * 66)
    sent2, sess2 = await run_call(script_refusal())
    creates2 = [m for m in sent2 if m.get("type") == "response.create"]
    check(sess2.memory.get("escalated") is True, "refusal escalated")
    check(not sess2.memory.get("resolved"), "refusal not marked resolved")
    check("policy" in (sess2.memory.get("escalate_reason") or "").lower(),
          "escalate reason records the policy refusal",
          repr(sess2.memory.get("escalate_reason")))
    check(len(creates2) <= 2, "call did not keep re-asking after refusal",
          f"{len(creates2)} response.create sent")

    print("\n" + "=" * 66)
    print("  SCENARIO 3 — identity re-asked mid-call")
    print("=" * 66)
    sent_id, sess_id = await run_call(script_identity_reask())
    check(not sess_id.memory.get("escalated"),
          "identity re-ask did not end the call")
    check(not sess_id.memory.get("resolved"),
          "identity re-ask did not falsely resolve the call")
    check(not [m for m in sent_id if m.get("type") == "conversation.item.create"
               and m["item"].get("type") == "function_call_output"],
          "no tool fired on an identity question")
    agent_turns = [t.text for t in sess_id.turns if t.role == "agent"]
    check(len(agent_turns) >= 3, "agent kept answering across re-asks",
          f"{len(agent_turns)} agent turns")
    check(sum("Forage AI" in t for t in agent_turns) >= 2,
          "org named more than once — no-repetition rule did not suppress it")

    # These guard the prompt wording itself. They catch deletion, not
    # misbehaviour — only a live call tests whether the model obeys.
    check("PRECEDENCE" in tpl.instructions,
          "identity rules declare precedence over brevity/pacing/closing rules")
    # Both identity blocks forbid deferring a disclosure, in their own wording
    # ("never deferred to a later turn" / "Never defer either to a later
    # turn"). Assert the rule, not one phrasing.
    check("defer" in flat and "later turn" in flat,
          "disclosures cannot be postponed to a later turn")

    print("\n" + "=" * 66)
    print("  TEMPLATE INVARIANTS")
    print("=" * 66)
    from agents.voice.templates import TEMPLATES
    for name, t in TEMPLATES.items():
        flat_t = " ".join(t.instructions.split())
        # Whatever the persona, two things must survive in every template:
        # the call is genuinely recorded, and a point-blank question about
        # what it is gets a straight answer. Presenting as a person is a style
        # choice; denying it when asked outright is regulated in several US
        # states, and these calls go to US numbers.
        check("If anyone asks whether you are a real person" in flat_t
              or "IF ASKED DIRECTLY whether you are a real person" in flat_t,
              f"{name}: answers a point-blank are-you-real question")
        check("recorded" in flat_t,
              f"{name}: acknowledges the call is recorded")
        check("Never claim to be a nurse" in flat_t,
              f"{name}: never impersonates hospital staff or a patient")
        check("{{" not in t.instructions,
              f"{name}: no unsubstituted template placeholders")

    # ── The prompt has a ceiling, and it is a hard one ───────────────────────
    # Every prompt edit before 2026-08-20 was additive. A live call produced a
    # failure, a rule was written for it, and the rule stayed — so the prompt
    # went 5,434 -> 6,125 in-text tokens over two sessions while a deletion
    # pass was deferred four times. Each deferral came with a test ("if the
    # next call adds no Conversation Flow rules, the pass has no reason to
    # wait"), the test was met twice, and the section grew both times. A
    # deferral that never fires is not a deferral.
    #
    # A ceiling is the only thing that makes the trade explicit: adding a rule
    # now costs evicting one, at the moment of adding, rather than costing a
    # cleanup nobody schedules. The number is deliberately close to where the
    # prompt sits after the pass — a ceiling with 1,500 tokens of headroom is
    # the same thing as no ceiling.
    #
    # HISTORY: 5,828 static tokens before the 2026-08-20 pass (the high-water
    # mark, 37% above the prompt it replaced for bloat); 4,733 after. Every
    # call that ever RESOLVED ran at <=4,641 in-text tokens. That is
    # confounded — the twelve-day regression changed several things at once —
    # so it is not evidence that a short prompt resolves calls, only that no
    # long one ever has. Raising this ceiling is allowed; doing it silently,
    # as a side effect of adding a rule, is what it exists to stop.
    # PER TEMPLATE, as of 2026-08-24, because one number stopped being able to
    # mean one thing. A template that collects two fields legitimately needs
    # more instruction than one that collects one — but "legitimately needs
    # more" is exactly the argument every additive edit made before the ceiling
    # existed, so it is not accepted as a reason to raise a shared number and
    # let everything drift up behind it. Each template is pinned separately,
    # just above where it actually sits, and the branch scripts do not move.
    #
    # RAISED DELIBERATELY FOR provider_verification, WITH THE CAVEAT ATTACHED:
    # every call that ever RESOLVED ran at <=4,641 in-text tokens, and this
    # prompt is 5,285. That is confounded — the twelve-day regression changed
    # several things at once, so it is evidence that no long prompt has ever
    # worked, not that a long prompt cannot. It is still the strongest reason
    # on record to distrust this template before it has been on a live call,
    # and the first thing to suspect if it underperforms the branch scripts.
    # RAISED 2026-08-24, second time, and the trade was made before the number
    # moved. call-20260824-1604 ended on "let me quickly pin down what that
    # means for scheduling" and hung up — a promise followed by a goodbye. The
    # rule against that went into the SHARED closing block, because the failure
    # is not specific to one script, which cost every template ~20 tokens.
    # About 60 tokens of pure enumeration were compressed out first (banned
    # closing phrases, the mishearing bullets, two example re-ask phrasings);
    # past that point shaving was costing real prose to hit a round number, so
    # the pins moved instead. That is the trade the ceiling exists to force
    # someone to make on purpose, and this is it being made on purpose.
    _PROMPT_CEILINGS = {
        "forage_data_collection": (4_850, 20_400),
        "forage_ai_disclosed":    (4_850, 20_400),
        # FIVE fields now. Identity confirmation went in 2026-08-25 and cost
        # ~480 tokens gross; eviction paid 102 of that — "# The Doctor"
        # compressed to the one rule identity does not supersede, and the two
        # Conversation Flow exits ("doctor left", "wrong number") which are
        # now STATES of a recorded field rather than escalate reasons.
        #
        # THE TRADE WAS ONLY PARTLY PAID, and saying so is the point. The
        # eviction that was principled did not cover the addition; the rest is
        # a raise. 5,878 tok / 24,700 chars when it landed.
        "provider_verification":  (5_900, 24_900),
    }
    try:
        import tiktoken as _tk
        _enc = _tk.get_encoding("o200k_base")
    except Exception:
        _enc = None
    # A template with no ceiling is a template that grows unmeasured, which is
    # the whole failure this section exists to stop — and adding a template is
    # precisely when it would be forgotten.
    check(set(_PROMPT_CEILINGS) == set(TEMPLATES),
          "every template has a declared prompt ceiling",
          f"missing {sorted(set(TEMPLATES) - set(_PROMPT_CEILINGS))}, "
          f"stale {sorted(set(_PROMPT_CEILINGS) - set(TEMPLATES))}")
    for name, t in TEMPLATES.items():
        _tok_ceiling, _char_ceiling = _PROMPT_CEILINGS.get(name, (4_800, 20_400))
        # The char ceiling runs ALWAYS, not only when tiktoken is missing. A
        # guard that reports nothing when its input is unavailable is the
        # false-negative shape this suite has been bitten by before: it would
        # pass loudly in CI while measuring nothing at all.
        check(len(t.instructions) <= _char_ceiling,
              f"{name}: prompt within the character ceiling",
              f"{len(t.instructions):,} / {_char_ceiling:,} chars")
        if _enc is None:
            check(False, f"{name}: token ceiling NOT measured — tiktoken "
                         f"unavailable, only the char proxy ran")
            continue
        _n = len(_enc.encode(t.instructions))
        check(_n <= _tok_ceiling,
              f"{name}: prompt within the token ceiling — to add a rule, "
              f"evict one",
              f"{_n:,} / {_tok_ceiling:,} tok")

    probe = Doctor(doctor_name="Dr. Jane Okafor",
                   hospital_name="Northside Medical Group")
    disclosed = TEMPLATES["forage_ai_disclosed"]
    # The claim is that automation is DISCLOSED, not that one particular phrase
    # is used. Asserting on "automated assistant" pinned the wording, and the
    # wording is now banned outright — the agent must not describe itself in
    # assistant register anywhere. Check the disclosure, not the phrasing.
    _disc_greet = disclosed.build_greeting(probe)
    check("automated" in _disc_greet.lower(),
          "forage_ai_disclosed announces automation upfront", _disc_greet[:56])
    check("automated assistant" not in _disc_greet.lower(),
          "forage_ai_disclosed discloses without the assistant register",
          _disc_greet[:56])
    check("automated" not in tpl.build_greeting(probe).lower(),
          "Template 1 does not announce automation upfront")

    print("\n" + "=" * 66)
    print("  SCENARIO 5 — tool fires while the agent is mid-question")
    print("=" * 66)
    sent5, sess5 = await run_call(script_tool_fires_mid_question())
    items5 = [m for m in sent5 if m.get("type") == "conversation.item.create"]
    texts5 = [i["item"].get("content", [{}])[0].get("text", "")
              for i in items5 if i["item"].get("type") == "message"]
    check(any("goodbye" in t for t in texts5),
          "asks for a closing instead of hanging up on a question")
    check(sess5.memory.get("branch") == "Northgate Campus",
          "the grounded branch is still saved")

    print("\n" + "=" * 66)
    print("  SCENARIO 6 — caller interrupts mid-sentence")
    print("=" * 66)
    check(settings.realtime_echo_gate != "drop",
          "echo gate lets caller audio through while the agent speaks",
          "REALTIME_ECHO_GATE=drop makes the agent uninterruptible — OpenAI's "
          "VAD never sees the audio, so barge-in cannot fire")
    sent6, _ = await run_call(script_barge_in())
    cancels = [m for m in sent6 if m.get("type") == "response.cancel"]
    truncs = [m for m in sent6 if m.get("type") == "conversation.item.truncate"]
    check(bool(cancels), "barge-in cancels the in-flight response")
    check(bool(truncs), "barge-in also TRUNCATES the item to what was heard")
    if truncs:
        t = truncs[0]
        check(t.get("item_id") == "item_abc",
              "truncate targets the item that was being spoken")
        ms = t.get("audio_end_ms")
        check(isinstance(ms, int) and ms >= 0,
              f"audio_end_ms is a sane offset ({ms}ms)")

    # ── One spoken item per response ─────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SCENARIO 6b — the model speaks twice inside ONE response")
    print("=" * 66)
    _d_out = {}
    _d_sent, _d_sess = await run_call(script_double_spoken_item(), out=_d_out)
    _d_media = [m for m in _d_out["twilio"].sent if m.get("event") == "media"]
    # Four deltas arrived, two per item. Only the first item's may be forwarded.
    check(len(_d_media) == 2,
          f"only the first item's audio reached the caller "
          f"({len(_d_media)} of 4 deltas forwarded)",
          "every delta forwarded means the callee hears it twice, which is "
          "what call-20260819-2044 sounded like")
    _d_agent = [t.text for t in _d_sess.turns if t.role == "agent"]
    check(_d_agent.count("Sure, no rush.") == 1,
          f"the muted item is not recorded as a turn ({_d_agent.count('Sure, no rush.')}x)",
          "a turn nobody heard must not reach the guards or the metrics")
    # Reply latency: measured from the caller stopping to the agent's first
    # sound. Only the greeting was ever timed, so "the agent takes a while to
    # answer" had no number on it for every turn after the first. Four audio
    # deltas arrive here but they are ONE reply — the measurement is taken at
    # the first and the clock is cleared, so it must not record four.
    check(len(_d_sess.reply_latencies) == 1,
          f"one reply latency recorded for one reply "
          f"({len(_d_sess.reply_latencies)})",
          "not cleared means every delta of every turn records a sample and "
          "the median is meaningless")
    # Whatever the detector itself waits is part of the gap the caller felt,
    # so it is counted in — but it is charged to the detector that actually
    # waits. server_vad holds silence_ms; semantic_vad holds nothing, and
    # billing it 0.7s inflated every reported gap on call-20260821-1856.
    check(all(x >= 0.0 for x in _d_sess.reply_latencies),
          "the detector's own wait is inside the gap, not added to it",
          "the caller waits through it too — it starts when they stop "
          "talking, not when OpenAI notices")
    # A turn that never came is not a slow reply, it is the silence watchdog's
    # failure, and letting it into this list would drag the median somewhere
    # no caller experienced.
    _before = len(_d_sess.reply_latencies)
    _d_sess.note_reply_latency(3600.0)
    _d_sess.note_reply_latency(-1.0)
    _d_sess.note_reply_latency(2.4)
    check(len(_d_sess.reply_latencies) == _before + 1
          and _d_sess.reply_latencies[-1] == 2.4,
          "absurd gaps are rejected, a real one is kept",
          f"list grew by {len(_d_sess.reply_latencies) - _before}, expected 1")
    # WITH THE VERDICT, not just the text. On call-20260825-1428 the model
    # emitted the GREETING twice and the mute stopped the caller hearing it
    # twice — the guard earning its keep. This entry and the one
    # call-20260825-1435 produced are the same shape and opposite meanings:
    # here the model said the same three words twice and the mute is the guard
    # earning its keep; there it was the caller's answer being deleted. Nobody
    # reading the artifact could tell them apart, so the verdict is decided
    # where the spoken half is still in hand and recorded with the text.
    check(_d_sess.dropped_second_items ==
          [{"text": "Sure, no rush.", "verdict": "duplicate"}],
          f"what was suppressed is kept for review, and marked as costing "
          f"nothing ({_d_sess.dropped_second_items})",
          "a guard that fires invisibly cannot be checked after the call")
    check(not _d_sess.owed_abandoned and not _d_sess._owed_substance,
          "a duplicate owes the caller nothing",
          f"nothing was lost, so nothing may be scheduled to be re-said "
          f"({_d_sess._owed_substance!r})")
    # AND A DUPLICATE IS STILL SILENCED. The hold changed what happens to a
    # second item carrying NEW words; it must change nothing about the case the
    # guard was built for, or the greeting goes out twice.
    check(len([m for m in _d_out["twilio"].sent if m.get("event") == "media"]) == 2,
          "the repeated line is still never played twice",
          "held, judged a duplicate, discarded — the caller hears it once")
    check(not _d_sess.released_second_items and _d_sess._held_item_pcm == {},
          "nothing is released and nothing is left held",
          f"{_d_sess.released_second_items}")
    # The rule is about the SECOND ITEM, not about the words. "Of course. Of
    # course, take your time." is the same defect with different text, and a
    # text-equality guard would let it straight through — which is why this
    # fires on item_id and never reads the transcript.
    _v_out = {}
    _v_sent, _v_sess = await run_call(
        script_double_spoken_item(second_text="Of course, take your time."),
        out=_v_out)
    # HELD, THEN RELEASED - the behaviour that changed on call-20260827-1130.
    # The mute still fires on item_id, because item_id is the only handle that
    # early. What changed is that the audio is HELD instead of deleted, so when
    # the transcript arrives and shows the second item carried something the
    # spoken half did not, the caller hears what the model actually said.
    #
    # The old contract was 2 media frames and an OWED entry: the words were
    # thrown away and the model was asked to say them again. On 1130 it was
    # asked four times, split the retry the same way every time, and the caller
    # heard four filler intros and never the question. Deleting the evidence
    # and asking the model to reproduce it is what this replaces.
    check(len([m for m in _v_out["twilio"].sent if m.get("event") == "media"]) == 4,
          "a second item carrying NEW substance now reaches the caller",
          "item_one's two frames plus item_two's two, released after the "
          "transcript said it was not a repeat")
    check(_v_sess.released_second_items ==
          [{"text": "Of course, take your time."}],
          f"and it is recorded as RELEASED, not suppressed "
          f"({_v_sess.released_second_items})")
    check(_v_sess.dropped_second_items == [],
          f"it is NOT also filed as dropped - the caller heard it "
          f"({_v_sess.dropped_second_items})",
          "an artifact that says both is an artifact nobody can read")
    check(not _v_sess._owed_substance and not _v_sess.owed_abandoned,
          f"and nothing is owed, because nothing was lost "
          f"({_v_sess._owed_substance!r})",
          "the owed-substance recovery is what the livelock ran on; released "
          "audio never enters it")
    # A REAL TURN. They heard it, so the turn guards must see it — filing it as
    # suppressed would hide a sentence the caller is about to answer.
    check(any(t.role == "agent" and t.text == "Of course, take your time."
              for t in _v_sess.turns),
          "the released item becomes an agent turn like any other",
          f"{[t.text for t in _v_sess.turns if t.role == 'agent']}")
    # NOTHING IS LEFT HOLDING. A buffer that survives its response would be
    # played inside a later one.
    check(_v_sess._held_item_pcm == {} and _v_sess._release_item == "",
          "and the hold buffer is empty once the response is done",
          f"{list(_v_sess._held_item_pcm)}")

    # -- AND THE DANGEROUS HALF: HELD AUDIO THE CALLER TALKED OVER ----------
    # Holding buys the verdict its evidence, but it also means unplayed audio
    # is sitting in a buffer when the caller starts speaking. Playing a turn
    # somebody has already interrupted is worse than muting it ever was, and a
    # transcript arriving after the barge-in must not resurrect it.
    _bh_out = {}
    _bh_sent, _bh_sess = await run_call(script_held_item_then_barge_in(),
                                        out=_bh_out)
    check(len([m for m in _bh_out["twilio"].sent
               if m.get("event") == "media"]) == 2,
          "held audio is NOT played once the caller has barged in",
          "item_one's two frames reached them; item_two's never do")
    check(not _bh_sess.released_second_items,
          f"nothing is recorded as released ({_bh_sess.released_second_items})")
    check(_bh_sess._held_item_pcm == {} and _bh_sess._release_item == "",
          f"and the buffer is emptied rather than left for the next response "
          f"({list(_bh_sess._held_item_pcm)})",
          "a buffer that survives its response is audio from a cancelled turn "
          "arriving inside a later one")

    # -- THE UNIT PIECES ---------------------------------------------------
    _hs = double(_held_item_pcm={"x": ["aaaa"]}, _release_item="x")
    rw._drop_held_items(_hs, "test")
    check(_hs._held_item_pcm == {} and _hs._release_item == "",
          "_drop_held_items clears the buffer and the pending release")
    # A CEILING, because this buffers a stream rather than a message.
    check(rw._MAX_HELD_ITEM_CHUNKS > 0 and rw._MAX_HELD_ITEM_CHUNKS <= 1000,
          f"the hold is bounded ({rw._MAX_HELD_ITEM_CHUNKS} chunks)",
          "past it an item degrades to the old behaviour rather than growing "
          "without bound on a model that will not stop talking")
    _fs = double(_held_item_pcm={}, stream_sid=None, _release_item="")
    check(await rw._flush_held_item(_fs, None, "nothing-here", []) == 0,
          "flushing an item with nothing held plays nothing and returns 0",
          "the capped-out and barged-over cases both arrive here")

    # ── The recovery has to be able to give up ───────────────────────────────
    # call-20260825-1435. The mute on a second spoken item is unconditional and
    # has to be — by the time the second item exists the first is on the wire.
    # The recovery this schedules is itself a response, the model produced TWO
    # items for that one too, and the second was muted carrying the same
    # substance, which owed it again. Nothing counted attempts, so the fourth
    # pass through the loop was indistinguishable from the first and the
    # caller's question was never answered on any of them.
    _owed_txt = "Which branch does Dr. Okafor work out of?"
    check(rw._owed_refusal(_v_sess, _owed_txt) == "",
          "the first attempt at owed substance is allowed",
          "refusing at zero deletes the recovery this whole path exists for")
    _v_sess._owed_attempts[rw._owed_key(_owed_txt)] = rw._MAX_OWED_PER_TEXT
    check(rw._owed_refusal(_v_sess, _owed_txt) != "",
          f"the same sentence is refused after {rw._MAX_OWED_PER_TEXT} "
          f"attempts that were muted",
          "unbounded is the livelock: every retry looked like the first")
    # And the shape a per-text counter can never see: the model REGENERATES the
    # owed half slightly differently each time, so every key is a new one.
    _v_sess._owed_attempts.clear()
    _v_sess._owed_tried = rw._MAX_OWED_PER_CALL
    check(rw._owed_refusal(_v_sess, "a sentence never attempted before") != "",
          f"a sentence never tried is still refused once the call has spent "
          f"{rw._MAX_OWED_PER_CALL} recoveries",
          "per-text counting alone never matches a regenerated sentence, so "
          "the loop survives it")
    _v_sess._owed_tried = 0

    # ── Repeats must survive into the artifact ───────────────────────────────
    # ── Pickup to greeting ───────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SCENARIO 6e — how long from picking up to hearing a voice")
    print("=" * 66)
    # "First audio 1.08s after response.create" starts its clock at OUR
    # request — after /answer, after the media WebSocket, after Twilio's
    # stream-start handshake. It can read 1.08s on a call that felt like ten
    # seconds of silence, because it is not measuring the same thing.
    # A script that actually emits audio deltas: the measurement is taken at
    # the first sound, so a transcript-only script never reaches it. The first
    # version of this test used script_happy_path, got None, and would have
    # read as a broken feature rather than a test that never exercised it.
    _pu_out = {}
    _pu_sent, _pu_sess = await run_call(script_double_spoken_item(), out=_pu_out,
                                        answered_at=time.monotonic() - 2.5)
    check(_pu_sess.pickup_to_greeting_s is not None,
          f"pickup-to-greeting is measured ({_pu_sess.pickup_to_greeting_s})",
          "the one figure the 'why is it slow to say hello' question is about")
    check((_pu_sess.pickup_to_greeting_s or 0) >= 2.5,
          f"and it counts the setup BEFORE response.create "
          f"({_pu_sess.pickup_to_greeting_s}s from a 2.5s-old pickup)",
          "measuring only from our own request is what hid this")
    # Optional by design: the value must be absent, never invented, when the
    # pickup could not be timed.
    _np_sent, _np_sess = await run_call(script_double_spoken_item())
    check(_np_sess.pickup_to_greeting_s is None,
          f"absent, not guessed, when the pickup was never timed "
          f"({_np_sess.pickup_to_greeting_s!r})")

    # ── A failure must say why ───────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SCENARIO 6d — a failed response records its reason")
    print("=" * 66)
    _fr_sent, _fr_sess = await run_call(script_failed_response())
    check(len(_fr_sess.response_failures) == 1,
          f"the failure is recorded ({len(_fr_sess.response_failures)})",
          "seven of these produced four stretches of 8-11s dead air and the "
          "reason was in the event the whole time")
    if _fr_sess.response_failures:
        _fr = _fr_sess.response_failures[0]
        check(_fr["status"] == "failed", f"status kept ({_fr['status']!r})")
        check("active response" in _fr["reason"],
              f"the REASON is kept, not just the status ({_fr['reason'][:44]!r})",
              "'a response failed' is not diagnosable; 'the conversation "
              "already had one' is")
    # A completed response must not be recorded as a failure — otherwise the
    # count is noise and the end-of-call line cries wolf on every call.
    _ok_sent, _ok_sess = await run_call(script_happy_path())
    check(not _ok_sess.response_failures,
          f"a clean call records no failures ({_ok_sess.response_failures})")

    # ── A question back is not a refusal ─────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SCENARIO 6c — the caller screens the call, and gets hung up on")
    print("=" * 66)
    _vet_sess = double(
        doctor=double(hospital_name="Northside Medical Group",
                                     doctor_name="Dr. Jane Okafor"),
        org_name="Definitive Healthcare", agent_name="David", turns=[])
    for _t, _want in [
        # Every caller turn from call-20260819-2121. Not one is a refusal.
        ("This is Northside Medical Group and I'm Varun. Sorry, who's calling again?", True),
        ("Um, is this about a patient or something urgent?", True),
        ("Is this about patient related?", True),
        ("How can I help you?", True),
        ("What can I do for you?", True),
        ("What's this regarding?", True),
        ("Are you a real person or is this a recording?", True),
        # Answers, however they are phrased. The second one opens with an
        # interrogative and is still an answer, which is why the shape test
        # alone is not enough.
        ("She's at the Mission Bay Clinic.", False),
        ("Which one — the Mission Bay clinic?", False),
        ("It's 1825 4th Street.", False),
        ("She works at Northgate campus.", False),
        # Refusals and holds are not vetting either — they have their own
        # handling and must not be swallowed by this.
        ("No, I can't give that out.", False),
        ("We don't share that information.", False),
        ("Sure, let me check our schedule.", False),
    ]:
        check(rw._caller_is_vetting(_t, _vet_sess) == _want,
              f"vetting={_want!s:5} {_t[:52]!r}")
    for _t, _want in [
        ("How can I help you?", True), ("What can I do for you?", True),
        ("What do you need?", True), ("Go ahead.", True),
        # Screening questions are NOT invitations. They stop the budget but
        # they are not an open door, and conflating the two would block
        # escalation on any question at all.
        ("Who's calling?", False), ("Is this about a patient?", False),
        ("She's at Mission Bay.", False),
    ]:
        check(rw._invites_continuation(_t) == _want,
              f"invitation={_want!s:5} {_t[:40]!r}")

    _vt_sent, _vt_sess = await run_call(script_vetting_then_invitation())
    check(not _vt_sess.memory.get("escalated"),
          f"the call is NOT escalated after 'How can I help you?' "
          f"({_vt_sess.memory.get('escalated')!r})",
          "they offered to help and the agent closed the call on it")
    _vt_out = [json.loads(m["item"]["output"])
               for m in _vt_sent
               if m.get("item", {}).get("type") == "function_call_output"]
    check(any(o.get("ok") is False and "NOT ESCALATED" in str(o.get("error"))
              for o in _vt_out),
          "escalate is REFUSED at the tool, not just un-flagged",
          "the give-up directive is already in the model's context by then — "
          "clearing a flag cannot unsay it")
    # The same clause on the end of every reply. Nothing caught this before:
    # _MIN_REASK_GAP_S measures speed and the gaps were eleven seconds, the
    # budget counts asks and not their wording, and repeated_sentences is
    # computed once the call is already over.
    _vt_directives = [m["item"]["content"][0]["text"] for m in _vt_sent
                      if m.get("type") == "conversation.item.create"
                      and m.get("item", {}).get("role") == "user"]
    check(any("those exact words" in d for d in _vt_directives),
          "asking in the identical clause again is caught while the call runs",
          f"{len(_vt_directives)} directives sent, none about the wording")
    check(sum("those exact words" in d for d in _vt_directives) == 1,
          "and it is said once, not on every subsequent ask",
          "a nudge repeated every turn is the same noise it is complaining "
          "about")
    # Asserted on the DIRECTIVE, not on sess._location_asks. The escalation
    # block resets the counter, so reading it after the call returns 0 whether
    # the budget burned or not — the first version of this check passed under
    # a mutation that disabled the exemption entirely, which is the whole
    # failure mode it was written to catch. The give-up directive is on the
    # wire and nothing rewrites it.
    # ASSERTED AGAINST THE TEXT THAT ACTUALLY GOES OUT. This used to hold a
    # hand-copied literal, "you have now asked for the location", and it is an
    # ABSENCE assertion — so the day the directive is reworded it passes by
    # finding nothing, which is exactly the failure mode it exists to catch.
    # give_up_directive() is now a function for this reason: the check reads the
    # real wording, and both triggers are covered rather than whichever one the
    # literal happened to be copied from.
    _give_up_marks = list(rw.GIVE_UP_MARKERS.values())
    check(all(m and m in rw.give_up_directive(_vt_sess, _t)
              for _t, m in rw.GIVE_UP_MARKERS.items()),
          "every give-up marker really appears in the directive it stands for",
          "otherwise the absence check below asserts nothing")
    # THREE since 2026-08-27: the silence watchdog used to spend its prompts
    # and then `continue` forever, carrying its exit only as prompt prose
    # ("Silence -> ... If it continues, escalate"). It now ends the call
    # through this same directive, so it needs a marker like the other two.
    check(len(_give_up_marks) == len(rw.GIVE_UP_REASONS) == 3,
          "every trigger is covered — one marker cannot vouch for another",
          f"{sorted(rw.GIVE_UP_REASONS)}")
    check(not any(m in d for m in _give_up_marks for d in _vt_directives),
          "the budget did not burn on four screening questions",
          "a front desk asking who you are is doing its job, not refusing")
    # Bounded, or a caller who only ever asks questions keeps the call alive
    # for as long as they feel like it.
    check(rw._MAX_VETTING_REASKS > 0,
          f"vetting exemption is bounded ({rw._MAX_VETTING_REASKS})")
    _bound_sess = double(
        doctor=_vet_sess.doctor, org_name="Definitive Healthcare",
        agent_name="David",
        turns=[rw.TranscriptTurn(role="caller", text="Who's calling?",
                                 timestamp="21:21:24")])
    check(rw._caller_vetted_since(_bound_sess, 0) is True,
          "a lone vetting turn is recognised by the since-helper")
    _bound_sess.turns.append(
        rw.TranscriptTurn(role="caller", text="She's at Northgate campus.",
                          timestamp="21:21:40"))
    check(rw._caller_vetted_since(_bound_sess, 0) is False,
          "one real answer in the run ends the exemption",
          "otherwise an answered call would never advance the budget")

    print("\n" + "=" * 66)
    print("  METRICS — a verbatim repeat must not be tidied away")
    print("=" * 66)
    _rep = [
        rw.TranscriptTurn(role="agent", text="Hi, this is David.", timestamp="20:45:29"),
        rw.TranscriptTurn(role="caller", text="Give me a minute.", timestamp="20:45:30"),
        rw.TranscriptTurn(role="agent", text="Sure, no rush.", timestamp="20:45:31"),
        rw.TranscriptTurn(role="agent", text="Sure, no rush.", timestamp="20:45:31"),
    ]
    _m = rw.conversation_metrics(_rep)
    check(_m["back_to_back_repeats"] == 1,
          f"three-word verbatim repeat is counted ({_m['back_to_back_repeats']})",
          "repeated_sentences has a >=4-word floor, so 'Sure, no rush.' scored "
          "zero on the call where the live detector had already flagged it")
    # "Consecutive" means consecutive AGENT turns — a caller turn in between
    # does not excuse it. This deliberately matches the live 🔁 detector, which
    # compares against the last agent turn and ignores what the caller said in
    # between. The metric disagreeing with the console marker is how the
    # call-20260819-2044 repeat came to be flagged live and scored zero after.
    _apart = [
        rw.TranscriptTurn(role="agent", text="Got it.", timestamp="20:45:01"),
        rw.TranscriptTurn(role="caller", text="One second.", timestamp="20:45:05"),
        rw.TranscriptTurn(role="agent", text="Got it.", timestamp="20:45:20"),
    ]
    check(rw.conversation_metrics(_apart)["back_to_back_repeats"] == 1,
          "a caller turn in between does not excuse a verbatim repeat",
          "the live detector fires here, so the metric must agree with it")
    # But a different agent turn in between DOES break the run: coming back to
    # the same phrasing later in a call is ordinary speech.
    _broken = [
        rw.TranscriptTurn(role="agent", text="Got it.", timestamp="20:45:01"),
        rw.TranscriptTurn(role="agent", text="Which branch is that?", timestamp="20:45:05"),
        rw.TranscriptTurn(role="agent", text="Got it.", timestamp="20:45:20"),
    ]
    check(rw.conversation_metrics(_broken)["back_to_back_repeats"] == 0,
          "the same phrase with another turn between is NOT counted",
          "otherwise every 'Got it.' in a call reads as a defect")
    # Punctuation and case must not decide it — the same complaint as
    # _norm_clause was written for.
    _punct = [
        rw.TranscriptTurn(role="agent", text="Sure, no rush.", timestamp="20:45:31"),
        rw.TranscriptTurn(role="agent", text="sure, no rush", timestamp="20:45:31"),
    ]
    check(rw.conversation_metrics(_punct)["back_to_back_repeats"] == 1,
          "case and trailing punctuation do not hide a repeat")

    # End to end: the repeat must reach the SAVED artifact. The metric above is
    # computed on the cleaned-up transcript, so a cleanup that deletes the pair
    # makes a correct metric read zero — which is precisely what happened.
    # Only call-*.json — the same directory also holds the rolled-up master
    # index, which is a LIST, and picking it up made this blow up on a
    # TypeError instead of asserting anything.
    for _f in _ARTEFACTS.glob("call-*.json"):
        _f.unlink()
    _rr_sent, _rr_sess = await run_call(script_repeat_across_responses())
    _saved = sorted(_ARTEFACTS.glob("call-*.json"))
    check(bool(_saved), "the call wrote an artifact to read back")
    if _saved:
        _rec = json.loads(_saved[-1].read_text(encoding="utf-8"))
        _texts = [t["text"] for t in _rec["transcript"] if t["role"] == "agent"]
        check(_texts.count("Sure, no rush.") == 2,
              f"both spoken repeats survive the transcript cleanup "
              f"({_texts.count('Sure, no rush.')} of 2)",
              "the <=4-word fragment merge collapsed them, so the artifact "
              "showed one turn on a call where the console printed 🔁")
        check(_rec["conversation"]["back_to_back_repeats"] == 1,
              f"and the saved metric counts it "
              f"({_rec['conversation']['back_to_back_repeats']})",
              "an instrument that reads zero on the fault it exists to count "
              "is worse than no instrument, because it is believed")

    print("\n" + "=" * 66)
    print("  DETECTORS — guard against silently matching nothing")
    print("=" * 66)
    # A patch script once wrote literal backspace bytes (0x08) into this regex,
    # turning it into '\x08(which|...' so it matched nothing at all. The code
    # ran, raised nothing, and the feature simply did not exist. Caught only
    # because a manual check returned False on an obvious positive. Code that
    # silently does nothing looks exactly like code that works.
    for text, expected in [
        ("Which branch is Dr. Okafor working out of?", True),
        ("Yes, it is recorded. Which branch is she working out of?", True),
        ("And where does she practise?", True),
        ("What location is that?", True),
        ("Of course, take your time.", False),
        ("Got it, thanks — have a good day.", False),
        ("Perfect, I have that — thanks a lot.", False),
        # Transcription routinely drops question marks, so a missing '?' cannot
        # be what decides this. It is an ask.
        ("Which branch is she at.", True),
        # Real wordings from a live call that the old whitelist scored as 0 asks
        # while the agent asked four times.
        ("I'm trying to confirm which branch Dr. Okafor works at.", True),
        ("Thanks for checking; I'm ready for the branch details when you are.", True),
        ("Yes, please share the branch name or address where she sees patients.", True),
        # Reading a value back is not asking for one.
        ("Thanks, I have that as the Riverside branch. Have a good day.", False),
    ]:
        check(rw._is_location_ask(text) == expected,
              f"location-ask detector: {expected!s:5} for {text[:44]!r}")
    # Every compiled pattern in the module, not just one. A patch script has now
    # written 0x08 into a regex twice; the first time only the location-ask
    # pattern was guarded, and the second landed in a pattern the guard did not
    # cover. (That pattern, _LOCATION_ASK, was itself deleted on 2026-08-18 —
    # dead since _is_location_ask was inverted away from a phrasing whitelist.
    # Named here as history only; this loop finds patterns by type, so it never
    # needed the name and does not age when one is removed.)
    import re as _re
    import agents.voice.evidence as _ev
    import agents.voice.objectives as _obj_mod
    # ACROSS EVERY MODULE THAT HOLDS ONE, and COUNTED. Scoped to `vars(rw)`,
    # this loop shrank silently the moment the guards moved to evidence.py:
    # _LOCATION_NOUN stopped being scanned, the suite reported one check fewer
    # and still said ALL PASSED. A population check that quietly covers less is
    # the exact "passes by finding nothing" shape this file keeps warning about,
    # and the floor below is what makes the next extraction loud instead.
    _pat_mods = [rw, _ev, _obj_mod]
    _pats = {(_m.__name__, _n): _v for _m in _pat_mods
             for _n, _v in vars(_m).items() if isinstance(_v, _re.Pattern)}
    check(len(_pats) >= 40,
          f"the pattern population is still being found ({len(_pats)})",
          "scoped to one module it falls to whatever survived the last move, and reports success for everything it no longer looks at")
    for (_mod, _name), _val in _pats.items():
        check(not any(ord(c) < 32 and c not in "\t\n" for c in _val.pattern),
              f"no control characters in {_name}")
    # Plain string constants too, not just regexes. A 0x01 sentinel landed in
    # _ABBREV_MARK and the regex-only guard could not see it — the third
    # control byte to reach this file. Read renders them invisibly, so nothing
    # catches these by eye.
    _strs = {(_m.__name__, _n): _v for _m in _pat_mods
             for _n, _v in vars(_m).items()
             if isinstance(_v, str) and not _n.startswith("__")}
    check(len(_strs) >= 12,
          f"the string-constant population is still being found ({len(_strs)})",
          "same reason as the patterns above")
    for (_mod, _name), _val in _strs.items():
        check(not any(ord(c) < 32 and c not in "\t\n" for c in _val),
              f"no control characters in string {_name}")

    # "Dr." is not the end of a sentence. Without protecting it, "Which branch
    # is Dr. Okafor at?" splits into "Which branch is Dr." + "Okafor at?" — a
    # statement-request followed by a question — so double_ask fired on nearly
    # every turn, because nearly every turn names the doctor. The original tests
    # missed it: none of the negative cases contained "Dr.", which is the single
    # most common token in the agent's speech.
    check(rw._sentences("Which office is Dr. Okafor working out of?") ==
          ["Which office is Dr. Okafor working out of?"],
          "'Dr.' does not split a sentence")
    for _t, _want in [
        ("Which office is Dr. Okafor working out of?", False),
        ("Right, thanks for checking — which branch does Dr. Okafor practice at?", False),
        ("Got it, thanks for coming back — which branch does Dr. Okafor work at?", False),
        ("Yes, which office is that?", False),
        ("I need the branch name or street address where Dr. Okafor sees "
         "patients. Which one is it?", True),
    ]:
        check(rw._double_ask(_t) == _want,
              f"double-ask with a title: {_want!s:5} for {_t[:38]!r}")

    # Tool rejections must not be speakable. On a live call the agent read this
    # module's rejection text to a receptionist, lightly paraphrased:
    #   "I need the specific site name or street address. If that's the only
    #    site, tell me that and I'll take it."
    # The old messages were fluent English imperatives, so relaying one produced
    # a grammatical sentence. They are terse fragments now, and a rejection that
    # drifts back toward speakable prose should fail here rather than on a call.
    from agents.voice.tools import save_branch as _save
    from agents.experiment.memory import CallMemory as _Mem
    _SPEAKABLE = ("ask whether", "ask them", "get the site", "tell me",
                  "i'll take it", "please provide", "could you", "you should")
    for _bad in ("California Branch", "Cardiology", "and", "the", "x"):
        _r = _save(_Mem("t"), _bad)
        if _r.get("ok"):
            continue
        _err = _r["error"]
        check(_err.startswith(("REJECTED", "NOT SAVED")),
              f"rejection is machine-shaped: {_bad!r}", _err[:48])
        for _phrase in _SPEAKABLE:
            check(_phrase not in _err.lower(),
                  f"rejection has no speakable imperative ({_phrase!r}): {_bad!r}")

    check("TOOL RESULTS ARE INTERNAL" in tpl.instructions,
          "prompt forbids reading tool results aloud")

    # The loop above enumerates ONE source of rejections: tools.py. That is not
    # the population. realtime_worker builds four more inline at the tool-call
    # site — the grounding block, the wrong-organisation block, and both
    # escalation blocks — and no assertion had ever looked at them. So they
    # stayed fluent English imperatives while tools.py was tightened around
    # them, and on call-20260818-1112 the agent read one to a caller, lightly
    # paraphrased: "Sorry, I can't use that unless you've actually said the
    # place name" — to someone who had just said a branch name. The guard
    # existed, worked, and was aimed at half the code.
    #
    # So: find every rejection, prove the search found them, judge each one.
    # Parsed rather than text-searched — a dict value is a dict value however
    # the line wraps and whether or not it is an f-string.
    import ast as _ast, pathlib as _pl
    _inline_errs: list[str] = []
    # WORKER PLUS GROUNDING. The four inline rejections moved with
    # _handle_tool_call; parsing one module after that split would find zero and
    # the judging loop below would pass over an empty list — the exact
    # vacuous-pass this check's own lower bound was written to catch.
    _rej_src = (_pl.Path(rw.__file__).read_text(encoding="utf-8") + chr(10)
                + _pl.Path(_rwground.__file__).read_text(encoding="utf-8"))
    for _node in _ast.walk(_ast.parse(_rej_src)):
        if not isinstance(_node, _ast.Dict):
            continue
        for _k, _v in zip(_node.keys, _node.values):
            if not (isinstance(_k, _ast.Constant) and _k.value == "error"):
                continue
            if isinstance(_v, _ast.Constant):
                _inline_errs.append(str(_v.value))
            elif isinstance(_v, _ast.JoinedStr):
                # Keep the authored literal segments. The interpolated parts are
                # runtime values (a mismatch description, a reason) — not a
                # register this file chose, and not something it can assert on.
                _inline_errs.append("".join(
                    str(_p.value) for _p in _v.values
                    if isinstance(_p, _ast.Constant)))
    # Lower bound, so it ages without edits when a fifth guard is added, but
    # fails loudly if the walk stops finding them — the key renamed, the dicts
    # moved behind a helper. A judging loop over an empty list passes.
    check(len(_inline_errs) >= 4,
          "found realtime_worker's inline rejections before judging them",
          f"{len(_inline_errs)} found")
    for _err in _inline_errs:
        check(_err.startswith(("REJECTED", "NOT SAVED", "NOT ESCALATED")),
              "inline rejection is machine-shaped", _err[:44])
        for _phrase in _SPEAKABLE:
            check(_phrase not in _err.lower(),
                  f"inline rejection has no speakable imperative ({_phrase!r})",
                  _err[:56])
    # The grounding rejection specifically must send the model back to the
    # transcript BEFORE it asks again. Telling it to ask is what lost
    # call-20260818-1112: the caller had said "office Abadan branch", the model
    # tried to save "Northside Branch" off its own record, and the call
    # escalated as unresolved with the answer sitting in the turns.
    check(any("RE-READ" in _e for _e in _inline_errs),
          "a rejection sends the model back to what the caller actually said")

    # /status decides, in one line, whether this call ever connected — and it
    # is the only line printed for a call that produced no transcript, so it is
    # the one place an operator has to be able to trust. It asked
    # `csid in _sessions` until 2026-08-18. Nothing writes _sessions on the
    # realtime path, so it was False for every realtime call and each one was
    # reported "NO CONVERSATION — nobody spoke to it", including 86-second
    # conversations that resolved.
    #
    # The invariant is not the registry's name. It is that whatever registry
    # that test consults must be marked on the COMMON path — unconditionally in
    # media_stream, not inside one branch of it — because that is what makes it
    # true of both call paths. Asserted structurally: a name equality would
    # pass again the moment someone swaps in another single-path registry.
    import agents.voice.twilio_worker as _tw
    _tw_ast = _ast.parse(_pl.Path(_tw.__file__).read_text(encoding="utf-8"))
    _fn = {n.name: n for n in _ast.walk(_tw_ast)
           if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    check({"status_callback", "media_stream"} <= set(_fn),
          "found both handlers before asserting about them",
          f"have {sorted(set(_fn) & {'status_callback', 'media_stream'})}")

    # Every `<call sid> in <registry>` test in the status handler. Keyed on the
    # SID because that is what a per-call registry is keyed by; it excludes the
    # handler's other membership test, `status in _never_connected`, which is a
    # lookup table of Twilio status strings and nothing to do with this. If the
    # SID variable is ever renamed out of that shape the set goes empty and the
    # population check below fails loudly rather than judging nothing.
    _probed = {c.comparators[0].id
               for c in _ast.walk(_fn["status_callback"])
               if isinstance(c, _ast.Compare)
               and any(isinstance(o, _ast.In) for o in c.ops)
               and isinstance(c.comparators[0], _ast.Name)
               and isinstance(c.left, _ast.Name)
               and "sid" in c.left.id.lower()}
    check(len(_probed) >= 1,
          "found the connectivity probe in /status", f"{sorted(_probed)}")

    # Names mutated by a top-level statement of media_stream — i.e. reached
    # whichever worker handles the call. Nested writes do not count; that is
    # precisely the bug.
    _common: set[str] = set()
    for _stmt in _fn["media_stream"].body:
        if not isinstance(_stmt, _ast.Expr) or not isinstance(_stmt.value, _ast.Call):
            continue
        _f = _stmt.value.func
        if isinstance(_f, _ast.Attribute) and isinstance(_f.value, _ast.Name):
            if _f.attr in ("add", "update", "__setitem__"):
                _common.add(_f.value.id)
    for _reg in _probed:
        check(_reg in _common,
              f"/status probes {_reg!r}, which media_stream marks on the "
              f"common path (both call paths reach it)",
              f"unconditionally marked: {sorted(_common)}")

    # A caller repeating themselves is telling you that is all they have. On a
    # live call they gave a street and a state twice, 36 seconds apart, and
    # save_branch was never called — 135 seconds, an answer given, nothing
    # recorded. Compared by word overlap so it survives the transcription
    # drifting "Lombard" -> "Lambert", and needs no vocabulary of its own.
    import types as _ns   # _t is rebound by an earlier loop variable
    _T = lambda t: _ns.SimpleNamespace(role="caller", text=t)
    _ss = lambda *ts: _ns.SimpleNamespace(turns=[_T(x) for x in ts])
    for _now, _prior, _want, _why in [
        ("He is working in Lambert Street in California.",
         ["He is working in Lombard Street in California."], True, "drifted transcription"),
        ("He's at the Northgate campus.",
         ["He's at the Northgate campus."], True, "verbatim"),
        ("He is working in Lambert Street in California.",
         ["I think I know but I'm not sure about the brand."], False, "different answer"),
        ("He's at the Northgate campus.",
         ["Yeah, I heard he's working in California."], False, "narrowing, not repeating"),
        # A repeated QUESTION is not a repeated answer — nudging the agent to
        # save "what do you want?" would be nonsense.
        ("What do you want now?", ["What do you want?"], False, "repeated question"),
        ("Hello", ["Hello"], False, "too short to mean anything"),
    ]:
        check(bool(rw.caller_repeated_answer(_now, _fake(_ss(*_prior)))) == _want,
              f"caller repeat ({_why}): {_now[:34]!r}")

    # A caller going to look something up has not refused. The give-up directive
    # is one-shot: once fired, the agent escalates on its next turn whatever they
    # say in between. On a live call that next turn was "can you please give me a
    # minute? I just need to check" — the most cooperative thing said on the
    # call — and it thanked them and hung up.
    for _t, _want in [
        ("Oh yeah, can you please give me a minute? I just need to check", True),
        ("Can you just wait for a moment, I'm getting another call.", True),
        ("Hold on, let me transfer you to the main branch.", True),
        ("Let me check.", True), ("one moment", True), ("bear with me", True),
        ("Sorry, wrong number.", False),
        ("I'm not allowed to give out that information.", False),
        ("Actually, he is not working now. He's retired.", False),
        ("He's at the Northgate campus.", False),
        ("Yeah, what do you want now?", False),
    ]:
        check(rw.is_hold_request(_t) == _want,
              f"hold request: {_want!s:5} for {_t[:40]!r}")

    # The greeting quoted INSIDE the context item must carry the same org and
    # name as the one the caller was told about. Called bare it fell back to the
    # defaults, so the banner printed "this is David" while the model was
    # instructed to open as "Alex" — and it said Alex. The org defaulted the same
    # way, hidden only because DEFAULT_ORG matched the configured value.
    _ctx = tpl.build_context(_probe, callback_number="", callback_email="",
                             org="Acme Health", agent_name="Jordan")
    _grt = tpl.build_greeting(_probe, org="Acme Health", agent_name="Jordan")
    check(_grt in _ctx, "context quotes the SAME greeting the caller hears")
    for _leak in ("Alex", "Definitive Healthcare"):
        check(_leak not in _ctx,
              f"context greeting does not fall back to the default {_leak!r}")

    # Stacked moves. The 46-word turn asked the caller to repeat themselves and
    # then answered the question it had just said it could not hear — four moves
    # in eighteen seconds. Counted as sentences, which needs no vocabulary: the
    # banned-phrase list for thinking-narration missed 2 of the 3 wordings
    # actually used ("let me respond to that for a moment", "let me say that
    # more clearly"), because "ways to narrate" is an open set, same as cities.
    _mk = lambda r, x: double(role=r, text=x)
    _turns = [
        _mk("agent", "Hi, this is Sarah, calling on behalf of X."),   # greeting, exempt
        _mk("caller", "Why should I tell you?"),
        _mk("agent", "Sorry, you're coming through faint — could you say that "
                     "again? It's a legitimate call. I'm Sarah on the directory "
                     "team."),
        _mk("agent", "Got it, thanks — which branch is Dr. Okafor at?"),
    ]
    _m = rw.conversation_metrics(_turns)
    check(_m["piled_turns"] == 1, "a stacked turn is counted", _m["piled_turns"])
    check(_m["longest_turn_sentences"] == 3, "longest turn measured in sentences",
          _m["longest_turn_sentences"])
    # The greeting is a fixed line, not a pile-up, and must not inflate the count.
    _only_greeting = [_mk("agent", "One. Two. Three.")]
    check(rw.conversation_metrics(_only_greeting)["piled_turns"] == 0,
          "the greeting is exempt from the pile-up count")
    # The prompt and the METRIC must agree on what a pile-up is. They did not.
    # Pacing said "ONE MOVE PER TURN — answer, or ask, or acknowledge, one of
    # them, then stop", while Shape Of A Turn marked the bare acknowledgement
    # ("Got it.") as Wrong and gave three worked examples of reaction-plus-ask.
    # Two sections classifying the same utterance oppositely is not a style
    # preference the model can satisfy; it is an arbitration it has to perform
    # on every turn, and examples beat prose, so Shape won inconsistently.
    # call-20260818-1112 measured 50% stapling — that is the arbitration, not
    # disobedience.
    #
    # piled_turns has ALWAYS counted >=3 sentences, so the measurement always
    # permitted reaction-plus-ask. The prompt rule was the outlier, and it was
    # also redundant: ONE ASK PER TURN already forbids the real hazard, in
    # countable terms, without colliding with anything.
    _flat_pace = " ".join(tpl.instructions.split())
    check("ONE ASK PER TURN" in _flat_pace,
          "the countable rule survives — one ASK per turn")
    check("ONE MOVE PER TURN" not in _flat_pace,
          "the rule that contradicted it is gone")
    # A bare reaction is Wrong and reaction-plus-ask is Right. Assert the prompt
    # still says so, since that is the half that had examples and won.
    check('Wrong: "Got it."' in _flat_pace,
          "a bare acknowledgement is still marked Wrong")
    # And the prompt's threshold must match the metric's: three or more.
    check("Three or more" in _flat_pace or "three or more" in _flat_pace,
          "prompt names the same pile-up threshold the metric counts (>=3)")
    check(rw.conversation_metrics(
              [_mk("agent", "Greeting."), _mk("agent", "Got it. Which branch is she at?")]
          )["piled_turns"] == 0,
          "reaction-plus-ask is not a pile-up by the metric either")
    check("NEVER ABOUT IT" in tpl.instructions,
          "thinking-narration judged by a test, not a phrase list")

    # Prompt-echo detection is DERIVED from the text we actually send, not
    # copied by hand. The old list held "forage ai" for days after the
    # organisation was renamed — a duplicated list rots silently every time the
    # original changes, and the prompt has been edited eleven times this week.
    from agents.voice.tools import _prompt_echoes as _echoes, save_branch as _sb
    from agents.experiment.memory import CallMemory as _CM
    _derived = _echoes()
    check(len(_derived) > 500,
          "echo phrases derived from the live prompt", f"{len(_derived)} entries")
    check("forage ai" not in _derived,
          "renamed org no longer lingers in the echo list")
    # A phrase actually in the current prompt must be caught...
    _live = "never claim to be a nurse a doctor"
    check(not _sb(_CM("t"), _live).get("ok"),
          "an echo of the live prompt is rejected")
    # ...and four-word sequences are long enough that real branch names, which
    # are short proper nouns, cannot collide with them.
    for _real in ("Northgate Campus", "Riverside Clinic", "Baptist Medical Center",
                  "1420 Beacon Street", "Downtown East", "Methodist Medical Center"):
        check(_sb(_CM("t"), _real).get("ok") is True,
              f"real branch name still accepted: {_real!r}")

    # Two requests for the same fact in one turn. The rule was "EXACTLY ONE
    # question mark per turn", which this passes with a single "?" — the same
    # blind spot the ask detector had, left in the prompt after the detector was
    # fixed. From a live call, and the trailing question is vaguer than the
    # statement it repeats: having just named two options, "which one is it?"
    # sounds like asking the caller to choose between them.
    for _t, _want in [
        ("I need the specific branch name or street address where Dr. Okafor "
         "sees patients. Which one is it?", True),
        ("I'm trying to find out which branch she works at. Do you know?", True),
        ("Do you know which branch she's working out of these days?", False),
        ("Sure, we keep a doctor directory — which branch is she at?", False),
        ("Thanks, I have that as the Riverside branch. Have a good day.", False),
        (tpl.build_greeting(Doctor(doctor_name="Dr. Jane Okafor",
                                   hospital_name="Northside Medical Group"),
                            org=settings.org_name), False),
    ]:
        check(rw._double_ask(_t) == _want, f"double-ask: {_want!s:5} for {_t[:40]!r}")
    check("ONE ASK PER TURN" in tpl.instructions,
          "rule counts asks, not question marks")
    check("EXACTLY ONE question mark per turn" not in tpl.instructions,
          "the question-mark-counting rule is gone")

    # Wrong organisation. A branch saved against the wrong hospital is corrupt
    # data, and it is the one failure grounding cannot see: every word can be
    # genuinely quoted from the caller and the record still be wrong, because
    # the call reached somewhere else. On a live call the record said "Northside
    # Medical Group", the caller answered "Thank you for calling the Methodist
    # Medical Center", nothing noticed, and the agent invented an address for it.
    def _sess(rec, *turns):
        return double(
            doctor=double(hospital_name=rec),
            turns=[double(role="caller", text=t) for t in turns])
    _R = "Northside Medical Group"
    for _turn, _want, _why in [
        ("Thank you for calling the Methodist Medical Center.", True, "different org"),
        ("You've reached Riverside Clinic, how can I help?",    True, "different org"),
        ("This is Mercy General Hospital.",                     True, "different org"),
        # Silence is the norm, not a signal — most people never name the place,
        # and firing on absence would block nearly every call.
        ("Thank you for calling Northside Medical Center.", False, "same place, other suffix"),
        ("Northside, this is Amy.",                        False, "person's name, right place"),
        ("Hello, this is Amy.",                            False, "person's name, no org"),
        ("Hello dear, how can I help you?",                False, "no self-identification"),
        ("Yes, speaking.",                                 False, "no org named"),
        ("this is the medical center",                     False, "generic words only"),
    ]:
        check(bool(rw.hospital_mismatch(_sess(_R, _turn))) == _want,
              f"wrong-organisation check ({_why}): {_turn[:38]!r}")

    # ── ...and later evidence has to be able to correct it ───────────────────
    # This returned on the FIRST differing claim, so a mismatch raised at
    # pickup could never be resolved however the rest of the call went.
    #
    # call-20260820-1440: caller answered "Hi, this is North Medical Group",
    # record said "Northside Medical Group", the save was blocked with
    # "NEED: which place this call actually reached". The agent asked. The
    # caller answered "This is Northside Medical Group." Nothing consumed it —
    # the agent escalated a second later with a reason the caller had just
    # contradicted, and a genuine branch (Mission Bay Clinic, 1825 Fourth
    # Street, grounding clean) was thrown away.
    #
    # What this deliberately does NOT do is decide whether "North" and
    # "Northside" are the same name. That is unanswerable from a transcript and
    # normalising them would be inventing data. It answers only the question
    # the rejection asked.
    #
    # And it is NOT "the last utterance wins" — which would be its own
    # bad-data bug. The negatives below are the point of the change.
    for _want, _why, _turns in [
        (False, "confirmed as the recorded org after the mismatch",
         ("Hi, this is North Medical Group, this is Varun.",
          "It's the Mission Bay Clinic, 1825 Fourth Street.",
          "This is Northside Medical Group.")),
        (True,  "never corrected",
         ("Thank you for calling the Methodist Medical Center.",
          "She's at the downtown site.")),
        # NAMING the record is not IDENTIFYING as it. Both of these contain
        # "Northside" and neither may clear a real mismatch.
        (True,  "a denial that names the record does not clear it",
         ("Thank you for calling the Methodist Medical Center.",
          "We're not Northside Medical Group.")),
        (True,  "a passing mention does not clear it",
         ("Thank you for calling the Methodist Medical Center.",
          "Dr. Okafor isn't at Northside any more.")),
        # Right place first, different org later, is a TRANSFER, not a
        # correction — so the confirmation must come after the claim to count.
        (True,  "confirmation BEFORE a differing claim does not clear it",
         ("You've reached Northside Medical Group.",
          "This is Methodist Medical Center.")),
        (False, "no organisation named at all is not a mismatch",
         ("Yes, speaking.", "She's at the Mission Bay clinic.")),
    ]:
        check(bool(rw.hospital_mismatch(_sess(_R, *_turns))) == _want,
              f"organisation state ({_why})")

    # Verbatim repeats. The agent said "Of course, take your time." twice in one
    # call, and the cause was not the model ignoring the no-repetition rule — the
    # prompt ordered that exact string: 'say ONLY "Of course, take your time."'.
    # A specific literal beats a general rule, correctly. So the failure mode to
    # guard is the SHAPE: any single quoted sentence the prompt commands will be
    # repeated verbatim the moment its situation recurs, and hold requests,
    # thanks and closings all recur.
    _scripted = _re.findall(
        r'say (?:ONLY|only|exactly)\s+["“]([^"”]{6,})["”]',
        tpl.instructions)
    check(not _scripted,
          "no single sentence is scripted verbatim in the prompt",
          "; ".join(_scripted)[:60] if _scripted else "")
    # The recurring cases must offer alternatives, not one wording.
    _hold = tpl.instructions[tpl.instructions.find("Hold request"):][:700]
    check(_hold.count('"') >= 8,
          "hold acknowledgement offers several wordings, not one")
    check("PICK A DIFFERENT ONE EACH TIME" in _hold,
          "hold guidance requires varying the wording")
    # The brevity over-correction must not come back. It was restored on purpose
    # to isolate the VAD variable, and that test is finished.
    for _dead in ("UNDER 15 WORDS", "under 15 words", "about eight",
                  "brilliant", "have a good evening"):
        check(_dead not in tpl.instructions,
              f"retired brevity/British wording absent: {_dead!r}")
    # Statement-form asks. The agent started using these once the brevity
    # rules were relaxed, and the budget counted 3 on a call where the caller
    # complained about being asked the same thing repeatedly.
    for text, expected in [
        ("I'm just trying to find out which branch she works at these days.", True),
        ("Could you tell me the branch name she is listed under?", True),
        ("When you have it, you can just say the branch name.", True),
        ("Of course, take your time.", False),
        ("You're right, that was irritating. I'll wait while you check.", False),
    ]:
        check(rw._is_location_ask(text) == expected,
              f"soft-ask detector: {expected!s:5} for {text[:44]!r}")

    # ── Thanking someone FOR a location is not asking for one ────────────────
    # _NOT_AN_ASK stripped the acknowledgement WORD and left the noun it
    # governs, so "Thanks for the location." became " for the location." and
    # counted as an ask. call-20260820-1915: seven location_asks against a
    # limit of four, and the verbatim-ask nudge telling the agent to "stop
    # stapling it on" about a sentence that asks nothing. Harmless that call —
    # holds had reset the budget — but an inflated count ends a call early on
    # one without holds.
    #
    # The distinction is grammatical: the noun is the acknowledgement's OBJECT,
    # not part of a fresh request.
    for _t, _want in [
        # the live bug and its family
        ("Thanks for the location.", False),
        ("Thanks for that location.", False),
        ("Thank you for the address.", False),
        ("Appreciate the branch info.", False),
        ("Thanks for the branch name — take care.", False),
        ("Thanks for that Mission Bay address.", False),
        # AN ACKNOWLEDGEMENT THAT GOES ON TO ASK IS STILL AN ASK. This is what
        # the residue test exists for and the fix must not eat it.
        ("Thanks — I still need the branch name.", True),
        ("Got it, I'm just trying to find the practice location.", True),
        ("Thanks for the address. I still need the branch name.", True),
        # ...and it must not jump a clause boundary. Without the negative
        # lookahead the two-word gap swallowed "and which" and the question
        # after it — a missed ask lets the agent pester someone, which is the
        # expensive direction.
        ("Great — and which campus is that, do you know.", True),
        ("Thanks — so which branch is that.", True),
        ("Perfect, and what location does she use.", True),
        ("Got it. But which site is she at.", True),
        ("Thanks for that — which branch though.", True),
        # A REQUEST WHOSE VERB SITS FURTHER FROM THE ACKNOWLEDGEMENT. The
        # two-word gap is the ceiling for an object; widen it and this gets
        # swallowed whole ("can you confirm the branch" is four words), and
        # a swallowed ask is an ask the budget never counts.
        ("Thanks — can you confirm the branch.", True),
        ("Got it. Do you happen to have the branch name.", True),
    ]:
        check(rw._is_location_ask(_t) == _want,
              f"ack-takes-object: {_want!s:5} for {_t[:44]!r}")

    # Escalation reasons that assert a fact about the doctor must be grounded.
    # A live call recorded escalate(reason="doctor deceased") after the caller
    # said only "he's not working right now".
    from core.models import TranscriptTurn as _TT2

    class _Sess:
        def __init__(self, said):
            self.turns = [_TT2(role="caller", text=said, timestamp="0")]

    not_working = _Sess("Actually, he's not working right now.")
    passed_away = _Sess("Oh, she passed away last year I'm afraid.")
    for reason, sess_, blocked in [
        ("doctor deceased", not_working, True),
        ("doctor retired", not_working, True),
        ("declined to share", not_working, False),
        ("could not obtain the location", not_working, False),
        ("doctor deceased", passed_away, False),
    ]:
        got = bool(rw._ungrounded_escalation(reason, _fake(sess_)))
        check(got == blocked,
              f"escalate {reason!r}: {'blocked' if blocked else 'allowed'} "
              f"given {sess_.turns[0].text[:34]!r}")

    # ── The inverse: an answer the caller GAVE, thrown away ──────────────────
    # Every guard above blocks a false positive — recording something that did
    # not happen. call-20260818-1112 failed the other way and nothing caught
    # it: the caller said "office Abadan branch" on turn two, save_branch was
    # called with "Northside Branch" (reshaped from the hospital name in the
    # model's own context), grounding correctly rejected it, the ask budget
    # correctly ran out, and the call escalated
    # reason="caller engaged but never provided a location".
    #
    # Every component behaved and the recorded reason is false. For a
    # data-collection product that is the expensive direction: a wrong row can
    # be found, a discarded answer looks identical to a receptionist who would
    # not say.
    import types as _ty2

    def _cs(*texts, rms=0.05):
        s = _ty2.SimpleNamespace()
        s.turns = [_ty2.SimpleNamespace(role="caller", text=t, audio_rms=rms)
                   for t in texts]
        s.doctor = Doctor(doctor_name="Dr. Jane Okafor",
                        hospital_name="Northside Medical Group")
        s.org_name, s.agent_name = "Definitive Healthcare", "David"
        return s

    _R = "caller engaged but never provided a location"
    for _want, _label, _s in [
        # The call itself, verbatim.
        (True,  "the call that lost an answer", _cs(
            "Hello, David. May I know why are you calling? Is it an emergency call?",
            "office Abadan branch, Need anything?",
            "But I wanted to know other than the branch details, do you know "
            "anything about the doctors?",
            "So can you share those details with me?")),
        (True,  "street address",        _cs("She's over on Lombard Street, I believe.")),
        (True,  "named campus",          _cs("That'd be the Northgate campus.")),
        (True,  "suburb and clinic",     _cs("Try the Jubilee Hills clinic.")),
        (True,  "turn-initial name",     _cs("Abadan branch is where she is.")),
        # Capitalisation is a CONJUNCT, not a requirement — a transcript that
        # came back caseless must not silently disable the detector.
        (True,  "caseless transcript",   _cs("she's at the northgate campus")),
        # False positives here are the expensive kind: they gate an escalation,
        # so one that fires on ordinary talk strands the agent on a call it
        # cannot end. These are the utterances that broke the first version.
        (False, "pure deflection",       _cs("Hello?", "Who is this?", "I can't help with that.")),
        (False, "flat refusal",          _cs("We don't give out that information.", "Sorry, no.")),
        (False, "generic word only",     _cs("She works at the branch.", "Just the main office.")),
        (False, "our other office",      _cs("She might be at our other office, I'm not sure.")),
        (False, "which location",        _cs("Which location did you want? I don't know.")),
        (False, "the office is closed",  _cs("The office is closed today, sorry.")),
        (False, "sentence-initial adj",  _cs("Closed. The branch is closed right now.")),
        # Our own record echoed back is not an answer FROM the call — it is the
        # exact material the fabrication was reshaped from.
        (False, "hospital on record",    _cs("This is Northside Medical Group hospital, how can I help?")),
        (False, "doctor's own name",     _cs("Okafor? At the clinic here, I think.")),
        (False, "the agent's persona",   _cs("Hi David, the office is closed.")),
        (False, "nothing transcribed",   _cs()),
        # Must clear the same hint-echo bar a real save has to clear.
        (False, "bare term, dead air",   _cs("Northgate", rms=0.001)),
    ]:
        check(bool(rw._discarded_location(_R, _fake(_s))) == _want,
              f"discarded-answer detector: {_want!s:5} for {_label}",
              rw._candidate_location(_fake(_s))[:52])

    # The reason gate. Reasons describing the CALL are the agent's own
    # observation; a place name in the transcript says nothing about whether
    # they are true, and blocking them would strand the agent.
    _gs = _cs("She's at the Northgate campus.")
    for _reason, _want in [
        ("caller engaged but never provided a location", True),
        ("caller does not know", True),
        ("could not obtain the location", True),
        ("wrong number", False), ("voicemail", False),
        ("declined to share", False), ("no response", False),
    ]:
        check(bool(rw._discarded_location(_reason, _fake(_gs))) == _want,
              f"reason gate: {_reason!r} "
              f"{'checked' if _want else 'exempt'}")

    # One-shot. A guard that can refuse forever is a call that cannot be ended,
    # and this detector is conservative rather than infallible. The session flag
    # is what bounds it — assert it exists and starts False, so the call site's
    # `not sess._discard_blocked` cannot silently become a permanent block.
    check(rw.RealtimeSession.__init__.__code__.co_consts is not None
          and "_discard_blocked" in rw.RealtimeSession(
              "CA00000000000000000000000000block",
              Doctor(doctor_name="Dr. X", hospital_name="Y")).__dict__,
          "session carries the one-shot flag that bounds the block")
    check(rw.RealtimeSession(
              "CA00000000000000000000000000block2",
              Doctor(doctor_name="Dr. X", hospital_name="Y"))._discard_blocked is False,
          "the discarded-answer block starts unfired")

    # ── THE TRANSCRIPT DECIDES, NOT THE MODEL'S WORDING ──────────────────────
    # _candidate_location used to run only when the reason matched
    # _NO_LOCATION_CLAIMS — a phrase whitelist checked against text the model
    # composes freely. call-20260821-1152: the caller said "She works at
    # Mission Bay clinic in San Francisco, but I'm not sure which location that
    # is", the model escalated with "caller could not provide...", and the list
    # holds "did not provide" but not "could not provide". One word, guard
    # silent, and a branch that grounds cleanly — it saved on the call eight
    # minutes earlier — was thrown away.
    #
    # POLARITY IS THE FIX. An inclusion list fails toward a lost answer; an
    # exemption list fails toward one blocked turn against a one-shot flag.
    def _dsess(*turns, rms=0.15):
        return double(
            doctor=double(hospital_name="Northside Medical Group",
                          doctor_name="Dr. Jane Okafor"),
            org_name="Definitive Healthcare", agent_name="David",
            turns=[double(role="caller", text=t, audio_rms=rms) for t in turns])

    for _want, _label, _reason, _turns in [
        # a usable location was supplied -> the escalation must be BLOCKED
        (True,  "valid branch", "caller does not know",
         ("She works at the Mission Bay clinic.",)),
        (True,  "branch + city", "caller does not know",
         ("She works at Mission Bay clinic in San Francisco.",)),
        (True,  "THE LIVE CALL: location + hedge, still usable",
         "caller could not provide a specific branch name or address for "
         "Mission Bay Clinic in San Francisco",
         ("Hi, I'm Varun. just a moment, let me clarify that.",
          "She works at Mission Bay clinic in San Francisco, but I'm not sure "
          "which location that is.")),
        (True,  "branch given, agent claims none",
         "caller never provided a location", ("It's the Northgate campus.",)),
        # WORDING VARIANTS over the same supplied location. Every one of these
        # is a phrasing the model can reach for and only two were on the list.
        (True,  "wording: could not provide", "could not provide a branch",
         ("She works at the Mission Bay clinic.",)),
        (True,  "wording: did not provide", "caller did not provide it",
         ("She works at the Mission Bay clinic.",)),
        (True,  "wording: unable to obtain", "unable to obtain the branch",
         ("She works at the Mission Bay clinic.",)),
        (True,  "wording: declined to specify", "declined to specify the site",
         ("She works at the Mission Bay clinic.",)),
        (True,  "wording: no usable detail", "no usable detail was given",
         ("She works at the Mission Bay clinic.",)),

        # LEGITIMATE escalations — these must stay ALLOWED.
        (False, "caller refuses", "caller refused, policy",
         ("We don't give that out, sorry.",)),
        (False, "caller does not know", "caller does not know",
         ("I honestly don't know where she works.",)),
        (False, "no location ever named",
         "caller engaged but never provided a location",
         ("Yes, speaking.", "Okay.", "Sure, no problem.")),
        (False, "doctor left", "doctor no longer works here",
         ("She left the practice last year.",)),
        (False, "only our OWN record echoed back", "caller does not know",
         ("This is Northside Medical Group, how can I help?",)),
        (False, "location only as a QUESTION", "caller does not know",
         ("Is she in San Francisco? I really couldn't say.",)),
        # WRONG ORGANISATION. A place IS named — it is the wrong organisation
        # itself — so a transcript-only rule would block this and strand the
        # agent. hospital_mismatch exempts it structurally.
        (False, "wrong organisation named", "reached the wrong organisation",
         ("Thank you for calling the Methodist Medical Center.",)),
        # CALL-SHAPE EXITS with a location present. Found in the existing suite
        # rather than in my own matrix: a voicemail greeting names the practice
        # and a wrong number names the bakery, so ignoring the reason entirely
        # would strand the agent on exactly the calls it must be able to end.
        (False, "voicemail, location present", "voicemail",
         ("You've reached the Northgate campus, leave a message.",)),
        (False, "wrong number, location present", "wrong number",
         ("This is the bakery on Mission Bay campus, love.",)),
        (False, "no response, location present", "no response",
         ("She's at the Northgate campus.",)),
    ]:
        check(bool(rw._discarded_location(_reason, _dsess(*_turns))) == _want,
              f"discard gate: {'block' if _want else 'allow':5} — {_label}")

    print("\n" + "=" * 66)
    print("  OpenAI handshake — a transient stall must not cost the call")
    print("=" * 66)
    # Call CAd1a20b, 2026-08-18: the callee picked up, the OpenAI handshake
    # stalled past the websockets default of 10s, and the call ended having
    # played nothing. A probe from the same machine minutes later connected in
    # 1.7s — so one bad TCP setup cost a live call with a real person on it.
    # There was no retry and the timeout was whatever the library chose.
    _hs_out: dict = {}
    _hs_sent, _hs_sess = await run_call(script_happy_path(), out=_hs_out,
                                        connect_failures=1)
    check(_hs_out["attempts"]["n"] == 2,
          "a failed handshake is retried once", f"{_hs_out['attempts']['n']} attempts")
    check(_hs_sess is not None and _hs_sess.memory.get("resolved"),
          "and the call still completes normally after the retry")
    # Two attempts, not unlimited: the callee is listening to silence the whole
    # time, so the budget is bounded by their patience, not by optimism.
    _hs_out2: dict = {}
    try:
        await run_call(script_happy_path(), out=_hs_out2, connect_failures=2)
        _gave_up = False
    except Exception:
        _gave_up = True
    check(_gave_up and _hs_out2["attempts"]["n"] == 2,
          "two failures gives up rather than holding a silent line open",
          f"{_hs_out2['attempts']['n']} attempts")
    check(rw._OAI_CONNECT_TIMEOUT_S < 10.0,
          "per-attempt timeout is tighter than the library default",
          f"{rw._OAI_CONNECT_TIMEOUT_S}s")

    print("\n" + "=" * 66)
    print("  Doctor routing — by CallSid, with no shared global")
    print("=" * 66)
    # `pending_doctor` was one module global that /answer read. Two calls in
    # flight would share it and the second would ask about the first one's
    # doctor — corrupt data, no error. It never fired because the global itself
    # made concurrency impossible, which is the worst kind of safe: the thing
    # keeping the bug dormant is the first thing a batch runner removes.
    #
    # Removing it surfaced why it had survived: register_call had NO CALLERS.
    # The SID map was never populated, so every call in the programme's history
    # resolved through the fallback. The safe path existed, was tested, and was
    # dead code.
    check(not hasattr(_tw, "pending_doctor"),
          "the shared pending_doctor global is gone")
    # The population that must route by SID: whatever places a call has to
    # register it, or /answer resolves nothing and hangs up.
    _rt_src = _pl.Path(_pl.Path(rw.__file__).parent.parent.parent / "run_twilio.py").read_text(encoding="utf-8")
    check("register_call(" in _rt_src,
          "run_twilio binds the CallSid it gets back from Twilio")

    _tw._doctor_by_sid.clear()
    _dA = Doctor(doctor_name="Dr. A", hospital_name="Hospital A")
    _dB = Doctor(doctor_name="Dr. B", hospital_name="Hospital B")
    _tw.register_call("CA_aaa", _dA)
    _tw.register_call("CA_bbb", _dB)
    _dr_a, _dr_b = await _tw._doctor_for("CA_aaa"), await _tw._doctor_for("CA_bbb")
    check(_dr_a is not None and _dr_b is not None
          and _dr_a.doctor_name == "Dr. A" and _dr_b.doctor_name == "Dr. B",
          "two calls in flight resolve to their own doctors")
    # Twilio retries webhooks. A second /answer for the same SID must resolve
    # to the same doctor, so the lookup must not consume the entry.
    _dr_again = await _tw._doctor_for("CA_aaa")
    check(_dr_again is not None and _dr_again.doctor_name == "Dr. A",
          "a retried webhook still resolves — the lookup does not pop")
    # An unknown SID waits, then gives up. Hanging up beats calling a hospital
    # about a record we cannot identify.
    _t0 = time.monotonic()
    _unknown = await _tw._doctor_for("CA_never_registered")
    _waited = time.monotonic() - _t0
    check(_unknown is None, "an unregistered CallSid resolves to nothing")
    check(_waited >= _tw._ROUTING_WAIT_S * 0.5,
          "and it waited for a late registration before giving up",
          f"{_waited:.2f}s")
    _tw._doctor_by_sid.clear()

    print("\n" + "=" * 66)
    print("  response.create — one door, three policies")
    print("=" * 66)
    # Six call sites, each with its own guard conditions. Two shipped without
    # checking _response_active and BOTH caused dead air on live calls
    # (97ff46d, then the empty-response re-request on 2026-08-11). That is one
    # missing abstraction, not two bugs — the seventh site would have been
    # another coin-flip.
    #
    # The invariant is not "the helper exists". It is that the helper is the
    # ONLY door: a new site that sends response.create directly gets none of
    # the guards and fails exactly the way the previous two did. Enforced
    # structurally, so it holds for code nobody has written yet.
    # ACROSS THE WHOLE PACKAGE, not just this module. _create_response moved to
    # agents/voice/grounding.py in the 2026-08-26 split, and an invariant that
    # only read realtime_worker would have reported "sent from nothing" and
    # passed the day someone added a second door in one of the new modules.
    # The claim was never about a file: response.create has ONE sender anywhere.
    _senders = set()
    _pkg_trees = [_ast.parse(_p.read_text(encoding="utf-8"))
                  for _p in sorted(_pl.Path(rw.__file__).parent.glob("*.py"))]
    for _mod_tree in _pkg_trees:
        for _fn_node in _ast.walk(_mod_tree):
            if not isinstance(_fn_node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            for _c in _ast.walk(_fn_node):
                if isinstance(_c, _ast.Constant) and _c.value == "response.create":
                    _senders.add(_fn_node.name)
    check(_senders == {"_create_response"},
          "response.create is sent from exactly one function",
          f"sent from {sorted(_senders)}")

    # And every site actually goes through it. A lower bound: sites get added,
    # and an equality here would fail on correct code.
    # Across the package for the same reason as the sender scan: the six call
    # sites are now split between realtime_worker and grounding.
    _calls = [n for _t in _pkg_trees for n in _ast.walk(_t)
              if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
              and n.func.id == "_create_response"]
    check(len(_calls) >= 6, "all six known sites route through the helper",
          f"{len(_calls)} call sites")
    # Every call names itself. A guard that silently does nothing looks exactly
    # like a guard that works — this module has been bitten by that three times,
    # so refusals are logged and the log needs to say which site was refused.
    check(all(any(k.arg == "why" for k in c.keywords) for c in _calls),
          "every site passes `why`, so a refusal names itself in the log")

    # The policy matrix. THE SITES DO NOT SHARE ONE POLICY: the goodbye and its
    # retry fire *because* the call is done, so a helper that refused on
    # sess.done would silently kill them — the exact silent no-op it exists to
    # prevent, at the two sites hardest to test.
    class _FakeWS:
        def __init__(self): self.sent = []
        async def send(self, raw): self.sent.append(json.loads(raw))

    async def _policy(active, done, **kw):
        _s = rw.RealtimeSession("CA000000000000000000000000polic",
                                Doctor(doctor_name="Dr. P"))
        _s._response_active, _s.done = active, done
        _ws = _FakeWS()
        ok = await rw._create_response(_ws, _s, why="test", **kw)
        return ok, len(_ws.sent)

    for _active, _done, _kw, _want, _label in [
        (False, False, {},                          True,  "idle call: allowed"),
        (True,  False, {},                          False, "response in flight: refused"),
        (False, True,  {},                          False, "call closing: refused"),
        (False, True,  {"allow_when_done": True},   True,  "goodbye: allowed though done"),
        (True,  True,  {"allow_when_done": True,
                        "allow_when_active": True}, True,  "goodbye from tool handler: allowed"),
        (True,  True,  {"allow_when_done": True},   False, "done-override alone does not cover in-flight"),
    ]:
        _ok, _n = await _policy(_active, _done, **_kw)
        check(_ok is _want and _n == (1 if _want else 0),
              f"policy: {_label}", f"sent={_n}")

    # ── Do not stack a reply on top of audio still playing ───────────────────
    # STILL PLAYING is not STILL GENERATING, and _response_active only knows
    # the second. OpenAI produces a reply far faster than realtime — a 6.25s
    # turn arrives in about a second — and every delta is forwarded to Twilio
    # immediately, so the rest sits in Twilio's queue long after OpenAI calls
    # the response done.
    #
    # Creating the next one then does not talk OVER the caller; it APPENDS, and
    # they hear an unbroken monologue with no gap to speak into. Measured on
    # call-20260819-2006: 21.3s of audio generated, 15.6s actually reaching the
    # callee, and a transcript showing three identical questions inside one
    # 50-word turn. She hung up at 36s.
    async def _play_policy(playing_s, **kw):
        _s = rw.RealtimeSession("CA0000000000000000000000playbk",
                                Doctor(doctor_name="Dr. Q"))
        _s._playback_ends_at = time.monotonic() + playing_s
        _ws2 = _FakeWS()
        ok = await rw._create_response(_ws2, _s, why="test", **kw)
        return ok, len(_ws2.sent)
    _ok, _n = await _play_policy(3.0)
    check(_ok is False and _n == 0,
          "a reply is not created while audio is still playing out")
    _ok, _n = await _play_policy(-1.0)
    check(_ok is True and _n == 1,
          "and is created once the queue has drained")
    # The closing sites are exempt: a goodbye that waits for the queue is a
    # goodbye that arrives after the line is being torn down.
    _ok, _n = await _play_policy(3.0, allow_when_done=True)
    check(_ok is True and _n == 1,
          "the goodbye is still allowed while audio plays out")

    # ── ...and the gate above never fired on the case that mattered ──────────
    # Found 2026-08-20. _create_response is the one place a reply is created BY
    # US, and the ordinary turn is not created by us — OpenAI's server VAD
    # makes it and the first this process sees is audio arriving. So the
    # playback gate has been correct, tested, and unreachable for the common
    # path since the day it shipped.
    #
    # call-20260820-1230, from the flush log: block 5 sent at 71.90s ran to
    # 76.95s; block 6 began sending at 76.30s. Twilio queues rather than mixes,
    # so the callee heard 7.35 unbroken seconds and spent it saying "Hello?",
    # "campus", "Hello," into a line that never paused. Blocks 7/8 repeated it
    # 0.45s apart.
    #
    # The fix cannot SLEEP: this runs in the OpenAI event pump, which must keep
    # reading for barge-in, response.done and tool calls. It queues silence to
    # Twilio instead, which lands the gap in the caller's ear and blocks
    # nothing here.
    _sil = base64.b64decode(rw._TWILIO_SILENCE_FRAME)
    check(len(_sil) == 160 and set(_sil) == {0xFF},
          "the breath frame is 20ms of mu-law silence", f"{len(_sil)} bytes")
    class _BreathWS:
        def __init__(self): self.sent = []
        async def send_text(self, s): self.sent.append(json.loads(s))
    _bs = rw.RealtimeSession("CA0000000000000000000000breath",
                             Doctor(doctor_name="Dr. Q"))
    _bs.stream_sid = "MZtest"
    _bw = _BreathWS()
    await rw._send_breath(_bw, _bs, rw._STACK_BREATH_S)
    check(len(_bw.sent) == int(rw._STACK_BREATH_S * 1000 / 20),
          "the breath is queued as 20ms media frames",
          f"{len(_bw.sent)} frames for {rw._STACK_BREATH_S}s")
    check(all(m["event"] == "media" and m["streamSid"] == "MZtest"
              for m in _bw.sent),
          "and addressed to the live stream")
    # No stream, no frames — never send media to a stream that is not up.
    _bs2 = rw.RealtimeSession("CA000000000000000000000breth2",
                              Doctor(doctor_name="Dr. Q"))
    _bw2 = _BreathWS()
    await rw._send_breath(_bw2, _bs2, rw._STACK_BREATH_S)
    check(not _bw2.sent, "no breath before the stream exists")

    # The gap must be long enough to read as a turn ending. OpenAI's VAD needs
    # realtime_silence_ms to call the CALLER done; the callee gets at least the
    # same to recognise their opening.
    check(rw._STACK_BREATH_S >= settings.realtime_silence_ms / 1000.0,
          "the gap is at least as long as the VAD's own end-of-turn window",
          f"{rw._STACK_BREATH_S}s vs {settings.realtime_silence_ms/1000.0}s")

    # And the wiring, which is the half that has been wrong before. The clock
    # for the new audio must come from the QUEUE's end plus the gap, not from
    # time.monotonic() — using "now" under-reports the queue by exactly the
    # overlap that caused the bug, and _playback_ends_at is derived from it.
    # ANCHORED ON THE FUNCTION, not on a file. The first-delta site moved to
    # agents/voice/audio.py with _handle_audio_delta; asking the interpreter
    # which source belongs to that function cannot drift when it moves again.
    _src_stack = inspect.getsource(rw._handle_audio_delta)
    _hook = _src_stack[_src_stack.find("if _first_delta_sent_at is None:"):][:3600]
    check("_send_breath" in _hook and "sess._playback_ends_at" in _hook,
          "the first-delta site consults the queue and inserts the gap")
    check("_first_delta_sent_at = (sess._playback_ends_at" in _hook,
          "and dates the new audio from the queue's end, not from now")
    check("not sess.done" in _hook,
          "closing is exempt, as it is in _create_response")

    # The matrix above tests the HELPER. It does not test the WIRING, and that
    # is the half that matters: setting the closing site's overrides to False
    # left every policy check passing while the goodbye was silently dropped —
    # the same guard-aimed-at-half-the-code shape as the rejection messages and
    # the /status probe. So assert the behaviour end to end.
    #
    # A resolved call owes the caller a spoken goodbye. In the happy path the
    # tool fires with no audio in that response, so the goodbye has to be
    # requested explicitly — greeting + goodbye = 2. Naive overrides on the
    # closing site drop it to 1 and hang up in silence, which is the bug
    # 6f0930a exists to prevent.
    _hp_sent, _ = await run_call(script_happy_path())
    _hp_creates = [m for m in _hp_sent if m.get("type") == "response.create"]
    check(len(_hp_creates) >= 2,
          "a resolved call still requests its spoken goodbye",
          f"{len(_hp_creates)} response.create on the happy path")

    print("\n" + "=" * 66)
    print("  WRITE-BACK — a resolved call must reach the Doctor record")
    print("=" * 66)
    # The programme exists to enrich a client directory. Until 2026-08-18 a
    # resolved call wrote a CallRecord and never touched the Doctor that
    # started it: Source.VOICE was assigned nowhere in the repo, and the
    # enrichment stopped at the call log. Invisible at N=1, because the branch
    # is read out of the call artifact by hand; structural the moment this
    # becomes the batch job it is meant to be.
    def _sess_for(doctor, *, branch=None, city=None):
        s = rw.RealtimeSession("CA0000000000000000000000000wb01", doctor)
        if branch:
            s.memory.update(branch=branch, resolved=True)
        if city:
            s.memory.update(city=city)
        return s

    # The case the CLI actually produces. run_twilio.py takes --doctor,
    # --hospital and --to; there is no --specialization, so a PERFECT call
    # still leaves the record failing is_complete(). It must not be recorded
    # as COMPLETE (the record contradicts that) nor silently downgraded with
    # no reason attached.
    _d_cli = Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside Medical Group")
    _r_cli = _sess_for(_d_cli, branch="Abadan Branch")._enrich_doctor(
        "Abadan Branch", rw.Outcome.COMPLETE)
    check(_d_cli.source is rw.Source.VOICE, "resolved call sets Source.VOICE")
    check(_d_cli.branch == "Abadan Branch", "resolved call writes the branch onto the Doctor")
    check(_d_cli.status is rw.DoctorStatus.PARTIALLY_VERIFIED,
          "no specialization -> PARTIALLY_VERIFIED, not COMPLETE", _d_cli.status.value)
    check(_r_cli["missing_for_complete"] == ["specialization"],
          "and the reason is named, not left as a bare False",
          f"{_r_cli['missing_for_complete']}")
    check(_d_cli.enriched_at is not None and _d_cli.enriched_at.tzinfo is not None,
          "enriched_at is set and timezone-aware")

    # A record that IS otherwise usable. VERIFIED means "confirmed by >=1 extra
    # source", which is exactly what a successful call establishes.
    _d_full = Doctor(doctor_name="Dr. A", hospital_name="H", specialization="Cardiology")
    _sess_for(_d_full, branch="Northgate Campus", city="Atlanta")._enrich_doctor(
        "Northgate Campus", rw.Outcome.COMPLETE)
    check(_d_full.status is rw.DoctorStatus.VERIFIED,
          "complete record + confirmed branch -> VERIFIED", _d_full.status.value)
    check(_d_full.city == "Atlanta", "city is carried across when the call captured one")

    # An unresolved call. The record still has no branch, which is all this
    # says — the reason lives in the call artifact, not the directory row.
    _d_no = Doctor(doctor_name="Dr. B", hospital_name="H")
    _sess_for(_d_no)._enrich_doctor(None, rw.Outcome.NONE)
    check(_d_no.status is rw.DoctorStatus.MISSING_BRANCH,
          "unresolved call -> MISSING_BRANCH", _d_no.status.value)
    check(_d_no.source is rw.Source.WEBSITE,
          "an unresolved call does NOT claim voice as the source",
          _d_no.source.value)

    # Upsert, not append. Re-calling the same doctor must update the row.
    # Keyed on (doctor_name, hospital_name): the same doctor at two hospitals
    # is two rows, and nothing else in the record is stable enough to key on.
    _wb_dir = _ARTEFACTS / "writeback"
    _wb_dir.mkdir(parents=True, exist_ok=True)
    (_wb_dir / "doctors.json").unlink(missing_ok=True)
    # save() moved to agents/voice/session.py and resolves json_dir in THAT
    # module's namespace; patching only the worker's re-export would leave the
    # artifact going to the real directory while the test read an empty one.
    with mock.patch.object(rw, "json_dir", lambda: _wb_dir), \
         mock.patch.object(_rwsession, "json_dir", lambda: _wb_dir):
        _s = _sess_for(Doctor(doctor_name="Dr. C", hospital_name="H1"))
        _s._write_doctor_directory({"doctor_name": "Dr. C", "hospital_name": "H1",
                                    "branch": "First"})
        _s._write_doctor_directory({"doctor_name": "Dr. C", "hospital_name": "H1",
                                    "branch": "Second"})
        _s._write_doctor_directory({"doctor_name": "Dr. C", "hospital_name": "H2",
                                    "branch": "Other hospital"})
        _rows = json.loads((_wb_dir / "doctors.json").read_text())
    check(len(_rows) == 2, "same doctor+hospital upserts; a different hospital appends",
          f"{len(_rows)} rows")
    check(next(r["branch"] for r in _rows
               if r["hospital_name"] == "H1") == "Second",
          "the re-call overwrote the earlier row rather than duplicating it")

    print("\n" + "=" * 66)
    print("  GROUNDING — never save a location the caller did not say")
    print("=" * 66)
    # A live call produced save_branch({'branch':'Riverside Clinic',
    # 'city':'Atlanta'}) when the caller had said only "Hello" and "Okay, next
    # slide, please". "Riverside Campus" was an EXAMPLE in the prompt; the
    # model reshaped it into a fabricated result, marked the call resolved and
    # hung up. Nothing downstream could distinguish it from a real answer.
    from core.models import TranscriptTurn as _TT

    class _FakeSess:
        def __init__(self, lines):
            self.turns = [_TT(role="caller", text=t, timestamp="00:00:00")
                          for t in lines]

    grounding_cases = [
        (["Hello.", "Okay, next slide, please."],
         {"branch": "Riverside Clinic", "city": "Atlanta"}, True),
        (["She's at the Northgate campus."],
         {"branch": "Northgate Campus"}, False),
        (["He works out of the Jubilee Hills office."],
         {"branch": "Jubilee Hills", "city": "Hyderabad"}, True),
        (["He's at 1420 Beacon Street in Boston."],
         {"branch": "1420 Beacon Street", "city": "Boston"}, False),
        (["Yes this is Northside."],
         {"branch": "the main branch"}, True),
        # No transcript at all: absence of evidence is not evidence of
        # fabrication, so do not block every save on a bad-audio call.
        (["[...]"], {"branch": "Anything At All"}, False),
    ]
    for lines, args, expect_blocked in grounding_cases:
        blocked = bool(rw._ungrounded_terms(args, _fake(_FakeSess(lines))))
        check(blocked == expect_blocked,
              f"{'blocks' if expect_blocked else 'allows'} {args.get('branch')!r} "
              f"given caller said {lines[0][:32]!r}")

    print("\n" + "=" * 66)
    print("  BRANCH VALIDATOR — a city is not a branch")
    print("=" * 66)
    # A live call saved branch="New York branch", city="New York" as RESOLVED.
    # Every word passed individually, so the all-filler check never fired, and
    # a useless record entered the dataset looking clean.
    from agents.voice.tools import save_branch
    from agents.experiment.memory import CallMemory
    validator_cases = [
        ("New York branch",           "New York",     False),
        ("London Branch",             "London",       False),
        ("New York City",             None,           False),
        ("Boston",                    None,           False),
        ("the Boston office",         None,           False),
        ("Texas",                     None,           False),
        ("Riverside Campus",          "Riverside",    False),
        ("Northgate Campus",          None,           True),
        ("Riverside Campus",          "Los Angeles",  True),
        ("Jubilee Hills",             None,           True),
        ("1420 Beacon Street",        "Boston",       True),
        ("Mercy General South Campus", "Sacramento",  True),
    ]
    for branch, city_arg, expect_ok in validator_cases:
        mem = CallMemory(call_id="validator-test")
        mem.clear()
        got_ok = bool(save_branch(mem, branch, city=city_arg).get("ok"))
        check(got_ok == expect_ok,
              f"{'accepts' if expect_ok else 'rejects'} {branch!r}"
              + (f" with city={city_arg!r}" if city_arg else ""))

    # "<Placename> branch" — the city list can never keep up ("New York branch"
    # was caught, "Newark branch" was saved on a live call). The shape is the
    # reliable signal. Push back once, then accept, because a group with one
    # Newark office genuinely does call it the Newark branch and rejecting
    # outright would loop forever.
    for branch in ("Newark branch", "the Boston office", "London Branch"):
        mem = CallMemory(call_id="validator-test")
        mem.clear()
        first = bool(save_branch(mem, branch).get("ok"))
        second = bool(save_branch(mem, branch).get("ok"))
        check(not first and second,
              f"{branch!r}: asks once, then accepts on retry")
        check(bool(mem.get("branch_needed_clarification")),
              f"{branch!r}: flagged as having needed clarification")

    for branch, city_arg in (("Northgate Campus", None),
                             ("1420 Beacon Street", "Boston"),
                             ("Mercy General South Campus", "Sacramento")):
        mem = CallMemory(call_id="validator-test")
        mem.clear()
        check(bool(save_branch(mem, branch, city=city_arg).get("ok")),
              f"{branch!r}: real site name accepted first time")

    print("\n" + "=" * 66)
    print("  SCENARIO 4 — invalid branch ('the branch') must be rejected")
    print("=" * 66)
    sent3, sess3 = await run_call(script_invalid_branch())
    check(not sess3.memory.get("resolved"),
          "generic word rejected, call not resolved")
    outputs = [json.loads(m["item"]["output"]) for m in sent3
               if m.get("type") == "conversation.item.create"
               and m["item"].get("type") == "function_call_output"]
    check(outputs and outputs[0].get("ok") is False,
          "save_branch returned an error the model can act on",
          outputs[0].get("error", "") if outputs else "no tool output")

    print("\n" + "=" * 66)
    print("  HINT ECHO — a bare prompt word is not a location")
    print("=" * 66)
    # call-20260821-1705. The caller said "hmm"; the transcriber had no
    # lexical content to decode, sampled its own conditioning prompt, and
    # emitted "Suite." Re-decoding that recorded 0.55s four times returned
    # 'campus', 'Suite,', the hint verbatim, and Urdu script — outputs that
    # disagree on identical bytes, which is the proof nothing was recovered
    # from the audio.
    _live_hint = get_template("forage_data_collection").transcribe_hint
    _vocab = rw._hint_vocabulary(_live_hint)
    # RETIRED 2026-08-26. The live hint is empty, so this guard is INERT BY
    # DESIGN rather than broken: it exists to catch the transcriber sampling our
    # conditioning prompt, and there is no longer a prompt to sample. Feeding it
    # _RETIRED_HINT_TEXT to keep it "armed" was tried and reverted — on one word
    # it then refuses a caller who answers "which branch?" with "Baptist", and
    # the call that prompted the retirement was to New York Baptist Hospital.
    # See _hint_vocabulary. Near-silent fabrications remain covered by
    # _audio_was_silent and _audio_carried_nothing, which never read the hint.
    check(len(_vocab) > 5,
          "the bare-word guard keeps its vocabulary after the hint is retired",
          f"{len(_vocab)} words — from _RETIRED_VOCAB_TEXT, not the live hint")
    check(rw._is_bare_hint_word("Suite", _live_hint),
          "the call-1705 'Suite' is still refused with the hint empty",
          "retiring the hint must not cost the protection it bought")
    # AND THE SPLIT HOLDS. Health systems left the live hint on 2026-08-20 and
    # were never covered here; folding them in during the retirement would
    # refuse a real one-word answer. The call that prompted all this was to New
    # York Baptist Hospital.
    for _system in ("Baptist", "Mercy", "Methodist", "Providence"):
        check(not rw._is_bare_hint_word(_system, _live_hint),
              f"a caller may still answer 'which branch?' with {_system!r}",
              "_RETIRED_HINT_TEXT must not reach a one-word test")

    # GENERIC, NOT A BLACKLIST — the property that made the retirement safe, and
    # now proved with TWO synthetic hints because the live one is empty. Each
    # must protect its own words and not the other's; no hardcoded list passes
    # both halves, and a list smuggled in via the retired text fails the second.
    _synth_hint = "Location words: zzyzx, quiggle."
    _other_hint = "Location words: fnord, blorp."
    check(rw._is_bare_hint_word("Zzyzx", _synth_hint),
          "a word from a SYNTHETIC hint is rejected — the guard follows the hint")
    check(not rw._is_bare_hint_word("Zzyzx", _other_hint),
          "and is allowed under a different hint that never contained it")
    check(rw._is_bare_hint_word("Fnord", _other_hint),
          "which protects its own words instead")
    check(not rw._is_bare_hint_word("Baptist", ""),
          "and with no hint at all nothing is protected — not even retired text",
          "a permanent one-word list here discards real single-word answers")
    # ANCHORED ON THE SYMBOL. _is_bare_hint_word moved to grounding.py, and
    # reading realtime_worker for it made find() return -1 — the slice then went
    # empty and this check passed while testing nothing at all. A vacuous pass
    # is worse than a failure: it reports coverage that no longer exists.
    _gfn = inspect.getsource(rw._is_bare_hint_word)
    _gbody = _gfn[_gfn.rfind('"""') + 3:]
    check("suite" not in _gbody.lower() and "campus" not in _gbody.lower(),
          "no vocabulary is hardcoded in the guard body")

    # ONE BARE WORD ONLY. Every word of the live hint must be refused alone...
    for _w in sorted(_vocab):
        check(rw._is_bare_hint_word(_w, _live_hint),
              f"bare hint word rejected: {_w!r}")
    check(rw._is_bare_hint_word("Suite.", _live_hint),
          "trailing punctuation does not smuggle one through")
    # ...and never merely for appearing inside a real site name.
    for _b in ("Downtown East", "1420 Beacon Street", "Northgate Campus",
               "Riverside Clinic", "Baptist Medical Center",
               "Methodist Medical Center", "Mercy General South Campus",
               "Suite 200, Beacon Street"):
        check(not rw._is_bare_hint_word(_b, _live_hint),
              f"multi-word location survives despite hint words: {_b!r}")
    # A real one-word place name is not hint vocabulary and must survive.
    for _b in ("Northgate", "Riverside", "Jubilee"):
        check(not rw._is_bare_hint_word(_b, _live_hint),
              f"real one-word name survives: {_b!r}")
    # And the validator underneath is untouched by any of it.
    for _b, _city in (("Northgate Campus", None), ("Riverside Clinic", None),
                      ("Baptist Medical Center", None),
                      ("Methodist Medical Center", None),
                      ("Downtown East", None),
                      ("1420 Beacon Street", "Boston")):
        _m = CallMemory(call_id="hint-echo-test")
        _m.clear()
        check(bool(save_branch(_m, _b, city=_city).get("ok")),
              f"save_branch still accepts {_b!r}")

    # THE RECORDED FIXTURE, END TO END. Source-level checks cannot see the
    # branch disabled, and every other gate is genuinely false here: the
    # fabricated word IS on the transcript, so grounding passes; there is no
    # address to drop; the organisation matches. If this gate stops firing,
    # the save goes through and only this check notices.
    _es = rw.RealtimeSession("CA000000000000000000hintecho",
                             Doctor(doctor_name="Dr. Jane Okafor",
                                    hospital_name="Northside Medical Group"))
    _es.transcribe_hint = _live_hint
    _es.turns = [rw.TranscriptTurn(role="caller", text="Suite.",
                                   timestamp="00:00:00", audio_rms=0.0383)]
    check(not rw._ungrounded_terms({"branch": "Suite"}, _es)
          and not rw._address_dropped({"branch": "Suite"}, _es)
          and not rw.hospital_mismatch(_es),
          "all three existing gates pass 'Suite' — this one is load-bearing")
    _ew = _TcWS()
    await rw._handle_tool_call(
        {"name": "save_branch", "call_id": "he1",
         "arguments": json.dumps({"branch": "Suite"})}, _es, _ew, {}, True)
    _eo = [json.loads(m["item"]["output"]) for m in _ew.sent
           if m.get("type") == "conversation.item.create"
           and m["item"].get("type") == "function_call_output"]
    check(_eo and _eo[0].get("ok") is False,
          "the hmm fixture 'Suite' is refused at the tool call",
          _eo[0].get("error", "") if _eo else "no tool output")
    check(not _es.memory.get("branch"), "and nothing reached the record")
    check(_es.memory.get("untrusted_location") == "Suite",
          "and the refusal is recorded, not silently dropped")
    # Terse machinery, never speakable prose — on call-20260818-1112 the agent
    # read one of these out to a caller.
    _err = _eo[0].get("error", "") if _eo else ""
    check("NEED:" in _err and "|" in _err,
          "the rejection is terse machinery the model can act on")
    # The same session must still save a real location afterwards, or the
    # guard has cost the call rather than protected it.
    _es.turns.append(rw.TranscriptTurn(role="caller",
                                       text="It's Mission Bay Clinic.",
                                       timestamp="00:00:01", audio_rms=0.15))
    _ew2 = _TcWS()
    await rw._handle_tool_call(
        {"name": "save_branch", "call_id": "he2",
         "arguments": json.dumps({"branch": "Mission Bay Clinic"})},
        _es, _ew2, {}, True)
    check(_es.memory.get("branch") == "Mission Bay Clinic",
          "and the real branch still saves on the very next attempt")

    # NOT A SHORT-UTTERANCE FILTER. Nothing on the audio path moved: a real
    # "Yes."/"Okay."/"Sure." at the fixture's own rms still reaches the model.
    check(not rw._audio_was_silent(0.0383)
          and not rw._audio_carried_nothing(0.0383, 0.147),
          "the fixture's rms is still above both audio thresholds")
    for _short in ("Yes.", "Okay.", "Sure.", "No."):
        check(not rw._is_bare_hint_word(_short, _live_hint),
              f"{_short!r} is not hint vocabulary and is never quarantined")

    print("\n" + "=" * 66)
    print("  BACKCHANNEL ECHO — our own mm-hm must not come back as speech")
    print("=" * 66)
    # The risk the config comment named and could not test for: a callee on
    # speakerphone hears our clip and their mic returns it. It cannot be found
    # in the transcript afterwards — the clips are "mm-hm"/"okay"/"right"/
    # "sure" and a caller saying "Okay." is the same string — so it has to be
    # stopped at the audio.
    check(isinstance(settings.realtime_backchannels, bool),
          "the echo guard is tested whether or not the feature is on")
    # Only a defect when the feature is ON. Clips are per-voice and only
    # cedar has them; asserting them unconditionally made a voice change
    # fail the suite and, worse, made four mutation results unreadable —
    # they reported CAUGHT off this failure rather than their own test.
    _clips = _bc.available(settings.realtime_voice)
    check(_clips > 0 or not settings.realtime_backchannels,
          f"backchannels are off, or the shipping voice has clips",
          f"voice={settings.realtime_voice!r} clips={_clips} "
          f"enabled={settings.realtime_backchannels}")
    # The silent-no-op is the real risk: enabling the feature on a voice
    # with no clips does nothing at all and logs nothing.
    check(not (settings.realtime_backchannels and _clips == 0),
          "the feature is never enabled on a voice that has no clips")

    # INDEPENDENT OF realtime_echo_gate. That gate is consulted only under
    # sess.agent_speaking, which a backchannel never sets; routing the echo
    # guard through it would mean the shipped default ("pass") disabled it.
    _eg_src = re.search(r"def _above_echo_floor.*?(?=\ndef |\nasync def )",
                        _PKG_SRC, re.S)
    check(_eg_src and "realtime_echo_gate" not in
          _eg_src.group(0)[_eg_src.group(0).rfind(chr(34)*3) + 3:],
          "the echo floor does not consult realtime_echo_gate")
    _saved_mode = settings.realtime_echo_gate
    try:
        for _mode in ("pass", "energy", "drop"):
            settings.realtime_echo_gate = _mode
            # 0x2a is a real signal (rms 0.164, the caller band); 0xff is
            # mu-law zero (rms 0.000), which is what an idle line sends.
            _loud = bytes([0x2a]) * 160
            _quiet = bytes([0xff]) * 160
            check(rw._above_echo_floor(_loud) and not rw._above_echo_floor(_quiet),
                  f"echo floor separates speech from silence under gate={_mode!r}")
    finally:
        settings.realtime_echo_gate = _saved_mode
    check(rw._above_echo_floor(b"") is False,
          "an empty frame is not mistaken for speech")

    # THE WINDOW IS SIZED FROM THE CLIP, not a constant. A longer clip must
    # hold the window open longer, or the tail of it comes back ungated.
    # Any voice that HAS clips — this tests the window arithmetic, not
    # which voice is shipping. Tying it to settings.realtime_voice made
    # the suite crash the moment the voice changed to one without clips.
    _clip_voice = next((v for v in ('cedar', 'marin', settings.realtime_voice)
                        if _bc.available(v)), None)
    _clip = _bc.pick(_clip_voice) if _clip_voice else None
    import base64 as _b64e
    _clip_s = len(_b64e.b64decode(_clip)) / 8000.0 if _clip else 0.0
    check(_clip is None or 0.1 <= _clip_s <= 1.2,
          f"the clip is a noise, not a turn (voice={_clip_voice})",
          f"{_clip_s:.2f}s" if _clip else "no clips installed for any voice")
    _rw_all = _PKG_SRC
    _k = _rw_all.find("sess._backchannel_mute_until = (")
    _inj = _rw_all[_k:_k + 220] if _k > 0 else ""
    check("len(base64.b64decode(_payload))" in _inj,
          "the mute window is derived from the clip's own length")
    check("_BACKCHANNEL_ECHO_MARGIN_S" in _inj,
          "plus a margin for the acoustic round trip")
    check(rw._BACKCHANNEL_ECHO_MARGIN_S > 0,
          "and that margin is non-zero", f"{rw._BACKCHANNEL_ECHO_MARGIN_S}s")

    # THE COUNTER IS THE WHOLE POINT OF THE LIVE TEST. Without it a call
    # cannot distinguish "no echo happened" from "echo happened and we never
    # noticed", because both look identical in the transcript.
    _rw_src = _PKG_SRC
    check("_backchannel_echo_frames += 1" in _rw_src,
          "suppressed frames are counted, not silently discarded")
    check('"backchannel_echo_frames"' in _rw_src
          and '"backchannels_sent"' in _rw_src,
          "and both numbers reach the call artifact")
    # BEHAVIOURAL, because the source check above cannot see the branch
    # disabled: `if False and time.time() < ...` leaves every substring in
    # place. That mutation survived the whole suite until this was added.
    _bs = rw.RealtimeSession("CA00000000000000000echotest",
                             Doctor(doctor_name="Dr. Jane Okafor",
                                    hospital_name="Northside Medical Group"))
    _loudf = bytes([0x2a]) * 160        # rms 0.164 — a person talking
    _quietf = bytes([0xff]) * 160       # rms 0.000 — mu-law zero
    check(not rw._is_own_backchannel_echo(_bs, _quietf),
          "outside the window, even silence is forwarded untouched")
    _bs._backchannel_mute_until = time.time() + 5
    check(rw._is_own_backchannel_echo(_bs, _quietf),
          "inside the window, a below-floor frame is withheld")
    check(not rw._is_own_backchannel_echo(_bs, _loudf),
          "inside the window, REAL SPEECH still gets through")
    _bs._backchannel_mute_until = time.time() - 0.01
    check(not rw._is_own_backchannel_echo(_bs, _quietf),
          "the window closes on its own — it cannot eat the rest of the call")
    _bc_site_src = _plb.Path(rw.__file__).read_text(encoding="utf-8")
    _site = _bc_site_src[_bc_site_src.find("if _is_own_backchannel_echo(sess, raw_bytes):"):][:180]
    check("_backchannel_echo_frames += 1" in _site and "continue" in _site,
          "and the media loop counts the frame before dropping it")

    # NOT A MUTE. The caller is mid-utterance by construction — a clip only
    # fires _BACKCHANNEL_AFTER_S into their turn — so real speech must pass.
    # Their measured level on the Twilio channel across live calls was
    # 0.079-0.240 against a floor of 0.020.
    check(settings.realtime_echo_rms < 0.079,
          "the floor sits below every caller level measured on a live call",
          f"floor {settings.realtime_echo_rms} vs quietest measured 0.079")

    print("\n" + "=" * 66)
    print("  TURN DETECTION — whichever ships, the payload must be coherent")
    print("=" * 66)
    # Deliberately NOT asserting which detector wins. semantic_vad shipped on
    # an argument and a probe that only proved the account accepts it, and the
    # Twilio recordings then measured it 0.6s SLOWER than server_vad (3.25s vs
    # 2.67s true acoustic gap). A test that pinned the winner would have
    # passed throughout. What this pins instead is that the choice stays
    # answerable: the payload matches the setting, and the measurement that
    # decided it is still written down where the next person will meet it.
    _cfg = rw.build_audio_config(
        transcribe_model=settings.realtime_transcribe_model,
        transcribe_hint=get_template("forage_data_collection").transcribe_hint,
        audio_format="pcm", noise_reduction=settings.realtime_noise_reduction,
        turn_detection=settings.realtime_turn_detection,
        eagerness=settings.realtime_vad_eagerness,
        voice=settings.realtime_voice, silence_ms=settings.realtime_silence_ms)
    # ── THE TWO LEGS ARE NEGOTIATED SEPARATELY ──────────────────────────────
    # Inbound stays mu-law (the model is the consumer; nothing we insert helps
    # it) while outbound asks for PCM16 so there is something left to condition.
    # A single format for both is what foreclosed output conditioning, and the
    # agent losing 2-3.4 kHz to the human caller is what that cost.
    _split = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="x",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar",
        silence_ms=400, output_format="pcm")
    check(_split["input"]["format"]["type"] == "audio/pcmu",
          "inbound stays g711 passthrough")
    check(_split["output"]["format"] == {"type": "audio/pcm", "rate": 24_000},
          "outbound asks for PCM16 24k so it can be conditioned",
          _split["output"]["format"])
    # AND THE DEFAULT IS THE OLD BEHAVIOUR. check_realtime.py probes variants
    # with one format and no opinion about the other; if omitting output_format
    # silently changed the outbound leg, every probe would be measuring
    # something other than what it named.
    _one = rw.build_audio_config(
        transcribe_model="gpt-4o-transcribe", transcribe_hint="x",
        audio_format="pcmu", noise_reduction="near_field",
        turn_detection="server_vad", eagerness="medium", voice="cedar")
    check(_one["output"]["format"]["type"] == "audio/pcmu",
          "omitting output_format leaves both legs on the inbound format")

    # ── OUTBOUND CONDITIONING ───────────────────────────────────────────────
    from agents.voice.outbound_audio import OutboundConditioner
    from agents.experiment.audio_utils import _mulaw_decode as _mud
    from agents.experiment.audio_utils import _mulaw_encode

    # A deterministic stand-in for speech: three formants, amplitude-modulated
    # so the compressor has something to act on. No external file, so this
    # check means the same thing on any machine.
    _oa_t = np.arange(int(24_000 * 3.0)) / 24_000.0
    _oa_sig = ((0.5 * np.sin(2 * np.pi * 220 * _oa_t)
            + 0.3 * np.sin(2 * np.pi * 1_400 * _oa_t)
            + 0.2 * np.sin(2 * np.pi * 2_800 * _oa_t))
           * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * _oa_t)))
    # NORMALISED TO THE OPERATING POINT the makeup gain is calibrated for.
    # Real agent speech measures -16 dBFS on the wire; the raw fixture lands at
    # -11.2, and at that level the compressor correctly pulls it DOWN 4 dB. A
    # level-preservation check run at the wrong level is measuring the
    # compressor doing its job and calling it a regression.
    _NOMINAL = -16.0
    _oa_sig = _oa_sig * (10 ** (_NOMINAL / 20) / np.sqrt(np.mean(_oa_sig ** 2)))
    _oa_pcm = (np.clip(_oa_sig, -1, 1) * 32767).astype(np.int16).tobytes()

    # THE SEAM TEST, which is the whole reason this is a stateful object.
    # The old _convert_oai_to_twilio resampled each 400ms delta independently
    # and restarted its anti-alias filter at every boundary: 100% of the
    # resulting error energy sat within 5ms of a seam, peak 0.09 — a -15 dB
    # transient 2.5 times a second. Byte-identical is the bar, not "close".
    _OA_CH = int(0.4 * 24_000) * 2
    _oa_whole = OutboundConditioner().process(_oa_pcm)
    _oa_c = OutboundConditioner()
    _oa_chunked = b"".join(_oa_c.process(_oa_pcm[i:i + _OA_CH])
                        for i in range(0, len(_oa_pcm), _OA_CH))
    check(_oa_c.enabled, "the conditioner has scipy and is actually running",
          getattr(_oa_c, "disabled_reason", ""))
    check(_oa_whole == _oa_chunked,
          "chunked deltas produce byte-identical audio to one-shot — no seam",
          f"{len(_oa_whole)} vs {len(_oa_chunked)} bytes")

    # AND WITH CHUNKS THAT ARE NOT A MULTIPLE OF THE DECIMATION RATIO. 24k->8k
    # keeps every third sample; a chunk length indivisible by 3 shifts which
    # sample that is, and a phase that jumps between chunks is a click at every
    # seam — the same defect arriving by a different route. 1000 bytes = 500
    # samples, and 500 % 3 != 0.
    _oa_c2 = OutboundConditioner()
    _oa_ragged = b"".join(_oa_c2.process(_oa_pcm[i:i + 1000])
                       for i in range(0, len(_oa_pcm), 1000))
    check(_oa_ragged == _oa_whole,
          "ragged chunk sizes keep decimation phase — still byte-identical",
          f"{len(_oa_ragged)} vs {len(_oa_whole)}")

    def _oa_band(sig, lo, hi, sr=8_000):
        n = 256
        fr = [sig[i:i + n] for i in range(0, len(sig) - n, n // 2)]
        en = np.array([np.sqrt(np.mean(f ** 2)) for f in fr])
        sel = [f for f, e in zip(fr, en) if e >= np.percentile(en, 92)]
        w = np.hanning(n)
        S = np.zeros(n // 2 + 1)
        for f in sel:
            S += np.abs(np.fft.rfft(f * w)) ** 2
        q = np.fft.rfftfreq(n, 1 / sr)
        return 100 * S[(q >= lo) & (q < hi)].sum() / S.sum()

    # THE POINT OF THE EXERCISE. Presence must actually lift 2-3.4 kHz, the
    # band that carries s/t/f/sh and the one the agent was measured to have at
    # half the human caller's level on the same line.
    _oa_plain = _mud(OutboundConditioner(presence_db=0.0, compress=False)
                  .process(_oa_pcm))
    _oa_lift = _mud(_oa_whole)
    check(_oa_band(_oa_lift, 2_000, 3_400) > _oa_band(_oa_plain, 2_000, 3_400) * 1.5,
          "presence lift raises 2-3.4 kHz by more than half again",
          f"{_oa_band(_oa_plain, 2000, 3400):.2f}% -> {_oa_band(_oa_lift, 2000, 3400):.2f}%")

    # LEVEL IS PRESERVED, NOT RAISED. The makeup gain is calibrated so
    # conditioning changes spectrum and dynamics and leaves loudness alone —
    # a stage that quietly made the agent quieter would work against the
    # complaint it exists to fix. Half a dB either way.
    _oa_lvl = lambda s: 20 * np.log10(float(np.sqrt(np.mean(s ** 2))))
    check(abs(_oa_lvl(_oa_lift) - _oa_lvl(_oa_plain)) < 1.0,
          "conditioning preserves level to within 1 dB at the nominal -16 dBFS",
          f"{_oa_lvl(_oa_plain):.1f} -> {_oa_lvl(_oa_lift):.1f} dBFS")

    # AND IS LEVEL-DEPENDENT BY DESIGN, which is worth pinning rather than
    # leaving as a surprise. The makeup gain is a constant — it has to be, since
    # a per-chunk normalise would pump — so preservation holds AT the operating
    # point and the compressor pulls everything else toward it. Louder in comes
    # out quieter, quieter in comes out louder. That is the dynamic-range
    # reduction being bought; it only becomes a defect if OpenAI's output level
    # drifts far from -16 dBFS, and this check is where that would show up.
    def _oa_at(db):
        s = _oa_sig * (10 ** (db / 20) / np.sqrt(np.mean(_oa_sig ** 2)))
        p = (np.clip(s, -1, 1) * 32767).astype(np.int16).tobytes()
        return _oa_lvl(_mud(OutboundConditioner().process(p)))
    # Asserted as the RATIO, which is the number actually configured. A 16 dB
    # spread of input should emerge as roughly 16/ratio. Anything looser is not
    # really testing the compressor, and an earlier cut of this check asserted
    # the loud case came out quieter than the nominal one — which compression
    # does not do and must not: it narrows the range, it does not invert it.
    _loud, _mid, _quiet = _oa_at(-8.0), _oa_at(-16.0), _oa_at(-24.0)
    _measured = (-8.0 - -24.0) / (_loud - _quiet)
    check(_loud > _mid > _quiet,
          "compression narrows the range without inverting it",
          f"-8 -> {_loud:.1f}, -16 -> {_mid:.1f}, -24 -> {_quiet:.1f} dBFS")
    check(abs(_measured - 3.0) < 0.5,
          "and it narrows it at the configured 3:1",
          f"16 dB in spans {_loud - _quiet:.1f} dB out = {_measured:.2f}:1")

    # NOTHING DRIVEN INTO THE MU-LAW CEILING by the boost. mu-law clipping is
    # not graceful, and a presence lift is exactly the thing that would cause it.
    check(float(np.mean(np.abs(_oa_lift) >= 0.99)) < 0.001,
          "the lift does not drive the signal into the mu-law ceiling",
          f"{100 * float(np.mean(np.abs(_oa_lift) >= 0.99)):.3f}% at ceiling")

    # DURATION AND THE DOWNSTREAM CLOCK. _wire_samples counts the recording
    # buffer, and a truncation on barge-in is bounded by it — so conditioned
    # output has to measure the same number of seconds it went in as.
    check(abs(rw._wire_samples(_oa_whole) / 8_000 - 3.0) < 0.01,
          "3s in, 3s out — the truncation and recording clock still agrees",
          f"{rw._wire_samples(_oa_whole) / 8000:.3f}s")

    # NO SCIPY -> OFF, NOT APPROXIMATE. audio_utils.resample falls back to
    # np.interp with no anti-alias filter at all, which is silent and badly
    # aliased. Unconditioned audio is merely dull; aliased audio is worse than
    # the problem being solved.
    _nofilt = OutboundConditioner.__new__(OutboundConditioner)
    _nofilt.enabled = False
    _out = _nofilt.process(_oa_pcm[:_OA_CH])
    check(len(_out) == _OA_CH // 2 // 3,
          "with no scipy it still emits correctly-sized mu-law, unconditioned",
          f"{len(_out)} bytes")

    # A TRUNCATED FRAME MUST NOT KILL THE CALL. np.frombuffer raises on a
    # buffer that is not a whole number of int16s, and this runs inside the
    # OpenAI event pump — an exception there takes the audio path down mid-call.
    check(len(OutboundConditioner().process(b"\x01\x02\x03")) >= 0,
          "an odd-length delta is trimmed, not raised")

    # ── THE FORMAT MATRIX ───────────────────────────────────────────────────
    # Four combinations of inbound/outbound format, and a one-second agent block
    # has to measure one second in all four. An earlier cut derived
    # "is the agent buffer mu-law" from the settings and got two of the four
    # wrong — reporting 1.0s as 0.167s, which is a garbled agent channel in the
    # WAV and a barge-in truncation computed against the wrong clock.
    #
    # It is right now because it is no longer derived: the agent buffer holds
    # what was sent to Twilio, and Twilio's wire is mu-law 8k on every call.
    _pcm24 = np.zeros(24_000, dtype=np.int16).tobytes()          # 1.0s
    _mu8 = _mulaw_encode(np.zeros(8_000, dtype=np.float32))      # 1.0s
    _keep = (settings.realtime_audio_format, settings.realtime_output_format)
    try:
        for _inb in ("pcmu", "pcm"):
            for _outb in ("pcmu", "pcm"):
                settings.realtime_audio_format = _inb
                settings.realtime_output_format = _outb
                _eff = rw._effective_output_format()
                _delivered = _pcm24 if _eff == "pcm" else _mu8
                _blob = (OutboundConditioner().process(_delivered)
                         if rw._outbound_conditioned() else _delivered)
                _secs = (rw._agent_wire_samples(_blob)
                         / rw._agent_wire_sample_rate())
                check(abs(_secs - 1.0) < 0.01,
                      f"in={_inb} out={_outb}: a 1s agent block measures 1s",
                      f"{_secs:.3f}s via {_eff}")

        # AND THE NEGOTIATION NEVER ASKS FOR PCM IT CANNOT FILTER. Without
        # scipy there is no anti-alias filter, and decimating unfiltered audio
        # folds 4-8 kHz back into speech — worse than the dull line the whole
        # exercise set out to fix. audio_utils.resample contains exactly this
        # trap already (silent np.interp fallback), which is why the check sits
        # upstream of the conversion rather than inside it.
        settings.realtime_output_format = "pcm"
        # PATCHED WHERE IT IS READ. _outbound_conditioned moved to
        # agents/voice/audio.py and closes over that module's binding, so
        # rebinding the worker's re-export no longer reaches it — the simulation
        # would silently do nothing and the check would pass for the wrong
        # reason. Same value, same origin (outbound_audio.DISABLED_REASON), one
        # module along. Both names are restored so neither module is left lying.
        _was = _rwaudio.OUTBOUND_UNAVAILABLE
        _was_rw = rw.OUTBOUND_UNAVAILABLE
        try:
            _rwaudio.OUTBOUND_UNAVAILABLE = "ImportError: simulated"
            rw.OUTBOUND_UNAVAILABLE = "ImportError: simulated"
            check(rw._effective_output_format() == "pcmu"
                  and rw._outbound_conditioned() is False,
                  "no conditioner -> negotiate mu-law, never PCM we can't filter")
        finally:
            _rwaudio.OUTBOUND_UNAVAILABLE = _was
            rw.OUTBOUND_UNAVAILABLE = _was_rw
        check(rw._effective_output_format() == "pcm",
              "and it comes back when the conditioner is available")

        # ── AND THE CALL ARTIFACT SAYS WHICH LEG RAN ────────────────────────
        # The entire case for the pcm leg is that it is falsifiable: flip one
        # setting, place two calls, compare. That comparison is made days later
        # against the artifacts, so a call that does not record which leg it ran
        # on is not evidence for either side — and audio_settings recorded
        # turn_detection, voice and noise_reduction while omitting the one
        # setting the session was actually changing.
        #
        # EVALUATED, NOT GREPPED, and the difference is the whole point. A
        # source check for '"output_format":' passes just as happily on
        #
        #     "output_format": settings.realtime_output_format
        #
        # which is the defect: without scipy that setting still reads "pcm"
        # while the call negotiated mu-law and sent nothing through the
        # conditioner. The artifact would assert conditioning on a call that had
        # none, and the A/B it exists to serve would compare pcm against pcm.
        # So the dict literal is lifted out of save() and evaluated under both
        # worlds — testing the expression the record actually uses instead of a
        # re-implementation of it that can agree while both are wrong.
        _sav = inspect.getsource(rw.RealtimeSession.save)
        _i = _sav.index('"audio_settings":')
        _i = _sav.index("{", _i)
        _d, _j = 0, _i
        for _j in range(_i, len(_sav)):
            _d += (_sav[_j] == "{") - (_sav[_j] == "}")
            if _d == 0:
                break
        _as_src = _sav[_i:_j + 1]

        def _recorded():
            return eval(_as_src, {**vars(rw), "settings": settings})

        settings.realtime_output_format = "pcm"
        _live = _recorded()
        check(_live.get("output_format") == "pcm"
              and _live.get("output_conditioned") is True,
              "the artifact records the conditioned leg when it ran",
              f"{_live.get('output_format')} / {_live.get('output_conditioned')}")
        check("output_downgraded" not in _live,
              "and claims no downgrade when there was none")

        # Same move as above: the recorded expression calls into audio.py.
        _was = _rwaudio.OUTBOUND_UNAVAILABLE
        _was_rw = rw.OUTBOUND_UNAVAILABLE
        try:
            _rwaudio.OUTBOUND_UNAVAILABLE = "ImportError: simulated"
            rw.OUTBOUND_UNAVAILABLE = "ImportError: simulated"
            _down = _recorded()
        finally:
            _rwaudio.OUTBOUND_UNAVAILABLE = _was
            rw.OUTBOUND_UNAVAILABLE = _was_rw
        # THE ONE THAT CATCHES THE COPIED SETTING. Here settings still says
        # "pcm" and the truth is "pcmu"; only a record that evaluates the
        # effective format can tell them apart.
        check(_down.get("output_format") == "pcmu"
              and _down.get("output_conditioned") is False,
              "a silently downgraded call is recorded as the mu-law it really "
              "was, not the pcm it asked for",
              f"setting said {settings.realtime_output_format!r}, "
              f"record said {_down.get('output_format')!r}")
        check(bool(_down.get("output_downgraded")),
              "and it carries the reason, so a dull call points at scipy "
              "instead of at the prompt",
              f"{_down.get('output_downgraded')!r}")

        # The inbound leg is recorded too — the two are separate settings now,
        # and attributing latency to one of them needs both on the record.
        settings.realtime_audio_format = "pcm"
        check(_recorded().get("input_format") == "pcm",
              "the inbound leg is on the record as well")
        settings.realtime_audio_format = "pcmu"

        # ── IS ANY OF THIS ACTUALLY WIRED IN? ───────────────────────────────
        # Every check above drives OutboundConditioner directly, so all of them
        # keep passing if the delta handler never calls it — which is precisely
        # what a broken wiring looks like, and a mutation run proved it: deleting
        #
        #     if _outbound_conditioned():
        #         raw_pcm = sess.outbound.process(raw_pcm)
        #
        # left the ENTIRE suite green. Raw PCM16 would have gone onto a wire that
        # reads it as mu-law — noise on every call — and nothing here noticed.
        # Same shape as the unwired _turn_asserts call site, caught the same way.
        settings.realtime_audio_format, settings.realtime_output_format = "pcmu", "pcm"

        class _TwWS:
            def __init__(self): self.sent = []
            async def send_text(self, s): self.sent.append(json.loads(s))

        _dsess_wired = rw.RealtimeSession("CA00000000000000000000wired1",
                                    Doctor(doctor_name="Dr. Jane Okafor"))
        _dsess_wired.stream_sid = "MZtest"
        _dsess_wired._stream_start_time = _dt.now()
        _dws = _TwWS()
        _dpcm = (np.clip(_oa_sig, -1, 1) * 32767).astype(np.int16).tobytes()
        _dbuf: list = []
        _dstate = rw._AudioDelta(0, None, None, None, False, None)
        _dstate = await rw._handle_audio_delta(
            {"delta": base64.b64encode(_dpcm).decode(), "item_id": "item_1"},
            _dsess_wired, _dws, _dbuf, _dstate)

        _media = [m for m in _dws.sent if m.get("event") == "media"]
        check(len(_media) == 1, "the delta handler forwarded exactly one media frame",
              f"{len(_media)}")
        _payload = base64.b64decode(_media[0]["media"]["payload"]) if _media else b""
        # mu-law is one byte per sample at 8k; the input was PCM16 at 24k, so a
        # correctly conditioned payload is exactly one sixth of the bytes in.
        check(len(_payload) == len(_dpcm) // 6,
              "and what reached Twilio is conditioned mu-law, not the raw PCM",
              f"{len(_payload)} bytes out of {len(_dpcm)} in "
              f"(raw-PCM passthrough would be {len(_dpcm)})")
        check(_dbuf and b"".join(_dbuf) == _payload,
              "the recording buffer holds exactly what was sent, not the source")
        check(_dstate.samples_this_response == len(_dpcm) // 6,
              "and the sample count that bounds a truncation counts sent samples",
              f"{_dstate.samples_this_response}")
    finally:
        settings.realtime_audio_format, settings.realtime_output_format = _keep

    _td = _cfg["input"]["turn_detection"] if "input" in _cfg else _cfg["turn_detection"]
    check(_td.get("type") == settings.realtime_turn_detection,
          f"the shipping detector reaches session.update: "
          f"{settings.realtime_turn_detection!r}")
    check(_td.get("interrupt_response") is True,
          "barge-in stays on under either detector — the caller can cut in")
    if settings.realtime_turn_detection == "server_vad":
        check(_td.get("silence_duration_ms") == settings.realtime_silence_ms,
              "server_vad is sent the silence timer it actually waits on",
              f"{_td.get('silence_duration_ms')}ms")
        check("eagerness" not in _td,
              "and NOT eagerness, which it would ignore")
        # 360ms interrupted a caller mid-sentence. Anything at or under that is
        # a setting already measured to be wrong.
        check(settings.realtime_silence_ms > 360,
              "the silence timer clears the measured truncation floor",
              f"{settings.realtime_silence_ms}ms vs 360ms")
    else:
        check("silence_duration_ms" not in _td,
              "semantic_vad is sent no silence timer — it has none")
        check(_td.get("eagerness") == settings.realtime_vad_eagerness,
              "and eagerness is sent, which is the only knob it has",
              f"{_td.get('eagerness')!r}")

    # THE MEASUREMENT MUST SURVIVE THE REVERT. Both flips so far were argued,
    # not measured; the numbers are the only thing that stops a third.
    _cfg_src = _plb.Path(rw.settings.__class__.__module__.replace(".", "/") + ".py")
    _cfg_txt = (_cfg_src.read_text(encoding="utf-8") if _cfg_src.exists()
                else _plb.Path("core/config.py").read_text(encoding="utf-8"))
    check("2.67s" in _cfg_txt and "3.25s" in _cfg_txt,
          "the true acoustic gaps for both detectors are recorded in config")
    check("Twilio recording" in _cfg_txt or "recordings" in _cfg_txt,
          "and it says where they came from, not just what they were")
    # The breath between stacked replies must still outlast the detector's own
    # wait, or a second reply lands before the callee registers the first.
    check(rw._STACK_BREATH_S >= settings.realtime_silence_ms / 1000.0,
          "the stacking breath still outlasts the silence timer",
          f"{rw._STACK_BREATH_S}s vs {settings.realtime_silence_ms / 1000.0}s")


    print("\n" + "=" * 66)
    print("  THE MIRROR — what OpenAI got, and only that")
    print("=" * 66)
    # call-20260821-1856. The backchannel echo guard withheld 173 frames from
    # OpenAI while _caller_pcm kept them, so our byte index ran 3.46s ahead of
    # OpenAI's ms clock. _utterance_slice then read past every utterance into
    # mu-law silence, reported rms=0.000244 — the fingerprint its own docstring
    # names — and the quarantine deleted the caller's real answers.
    #
    # Two buffers now, because they answer different questions: the recording
    # wants every frame on a gapless timeline, the measurement wants OpenAI's.
    _mirror_src = _plb.Path(rw.__file__).read_text(encoding="utf-8")
    check(_mirror_src.count("sess._caller_oai_pcm.append(") == 1,
          "the mirror is appended in exactly ONE place")
    _at = _mirror_src.find("sess._caller_oai_pcm.append(")
    _send = _mirror_src.find("input_audio_buffer.append", _at)
    check(0 < _send - _at < 260,
          "and that place is immediately before the send to OpenAI",
          f"{_send - _at} chars apart")
    # Nothing may drop a frame between the append and the send.
    check("continue" not in _mirror_src[_at:_send],
          "no drop sits between the mirror append and the send")
    # And the slicer must read the mirror, never the recording buffer.
    # _utterance_slice moved to agents/voice/audio.py. Read it through the
    # symbol rather than by regex over a file: the assertion is about what the
    # slicer READS, and that claim does not depend on which module holds it.
    _slice_src = inspect.getsource(rw._utterance_slice)
    _slice_body = _slice_src[_slice_src.rfind(chr(34) * 3) + 3:]
    check("_caller_pcm" not in _slice_body.replace("_caller_oai_pcm", ""),
          "_utterance_slice reads the mirror, not the recording buffer")

    # THE INVARIANT, DRIVEN. Source checks cannot see a drop re-introduced
    # above the append; this counts bytes on both sides of the real decision.
    class _MirrorSess:
        def __init__(self):
            self._caller_pcm = []
            self._caller_oai_pcm = []
            self._backchannel_mute_until = 0.0
            self._backchannel_echo_frames = 0

    def _feed(sess, frames, mute_from=-1, mute_to=-1):
        """The media loop's decision, byte for byte."""
        for i, f in enumerate(frames):
            sess._caller_pcm.append(f)                    # recording: always
            if 0 <= mute_from <= i < mute_to:
                sess._backchannel_echo_frames += 1
                continue                                   # withheld
            sess._caller_oai_pcm.append(f)                # mirror: forwarded

    _quiet = bytes([0xff]) * 160
    _talk = bytes([0x2a]) * 160
    _mir = _MirrorSess()
    _feed(_mir, [_talk] * 50)
    check(len(b"".join(_mir._caller_oai_pcm)) == len(b"".join(_mir._caller_pcm)),
          "with nothing withheld the two buffers agree exactly")
    _mir = _MirrorSess()
    _feed(_mir, [_talk] * 20 + [_quiet] * 173 + [_talk] * 20,
          mute_from=20, mute_to=193)
    _drift = len(b"".join(_mir._caller_pcm)) - len(b"".join(_mir._caller_oai_pcm))
    check(_drift == 173 * 160,
          "the recording keeps the withheld frames — the timeline stays gapless",
          f"{_drift} bytes")
    check(_mir._backchannel_echo_frames == 173,
          "and every withheld frame is counted")
    # The mirror is what the slicer indexes. Its length must equal what OpenAI
    # was fed, so a turn at the END is still found rather than read past.
    check(len(b"".join(_mir._caller_oai_pcm)) == 40 * 160,
          "the mirror holds exactly the forwarded frames",
          f"{len(b''.join(_mir._caller_oai_pcm))} bytes")

    # REGRESSION FIXTURE: the exact drift from call-20260821-1856. 173 frames
    # of 20ms at 24kHz PCM16 is 3.46s, which is what pushed the read past the
    # end of every utterance.
    check(abs(173 * 960 / 48 / 1000 - 3.46) < 0.01,
          "173 withheld frames is the 3.46s of drift that call measured")

    # And the offset the mirror makes structural: nothing is forwarded before
    # listening is enabled, so OpenAI's ms zero is the mirror's byte zero.
    check("sum(len(c) for c in sess._caller_oai_pcm)" in _mirror_src,
          "_listen_start_bytes is computed on the mirror, not the recording")

    print("\n" + "=" * 66)
    print("  DETECTOR LAG — measured from audio_end_ms, never assumed")
    print("=" * 66)
    # Two calls in a row were reported wrong by a constant that was reasoned
    # about instead of measured. server_vad's 0.7s charged to semantic_vad
    # inflated every gap on call-20260821-1856; removing it on call-20260821-
    # 1931 reported 0.81s while the Twilio recording measures 3.67s. The
    # instrument moved opposite to the thing it measures, and the second
    # version hid a real regression: across six recordings the TRUE acoustic
    # gap is 2.67s under server_vad and 3.25s under semantic_vad.
    _lag_src = _PKG_SRC
    check(not hasattr(rw, "_vad_hold_s"),
          "the per-detector constant is gone, not merely unused")
    check("realtime_silence_ms / 1000" not in _lag_src,
          "no site turns a SETTING into a reported latency")
    _stop = _lag_src[_lag_src.find("speech_stopped\":"):][:2600]
    check("audio_end_ms" in _stop and "_caller_stopped_at = time.monotonic() - _lag_s" in _stop,
          "the reply clock is backdated to when the caller actually stopped")
    check("sess._caller_oai_pcm" in _stop and "_listen_start_bytes" in _stop,
          "and the lag is computed against the mirror, which makes it exact")
    check("max(0.0, min(" in _stop,
          "the lag is clamped — a disagreeing buffer must not become a latency")

    # BEHAVIOURAL: the same arithmetic the handler runs.
    def _lag_of(end_ms, have_bytes, base=0):
        _bpms = rw._wire_bytes_per_ms()
        return max(0.0, min((have_bytes - (base + end_ms * _bpms)) / (_bpms * 1000.0), 10.0))

    _bpms_t = rw._wire_bytes_per_ms()
    check(abs(_lag_of(1000, int(1000 * _bpms_t))) < 1e-9,
          "no buffered audio past the stop -> zero lag")
    check(abs(_lag_of(1000, int(1500 * _bpms_t)) - 0.5) < 0.01,
          "500ms buffered after the caller stopped -> 0.5s of detector lag",
          f"{_lag_of(1000, int(1500 * _bpms_t)):.3f}s")
    check(_lag_of(1000, int(900 * _bpms_t)) == 0.0,
          "a negative lag clamps to zero rather than crediting the future")
    check(_lag_of(0, int(60_000 * _bpms_t)) == 10.0,
          "and a runaway one clamps rather than poisoning the median")

    # The artifact must carry the MEASUREMENT, and only when there is one.
    check('"detector_lag_s"' in _lag_src and '"vad_hold_s"' not in _lag_src,
          "the artifact reports a measured detector lag, not a setting")
    _art = _lag_src[_lag_src.find('"detector_lag_s"'):][:200]
    check("self.detector_lags" in _art and "else None" in _art,
          "and reports nothing rather than zero when nothing was measured")

    # The felt gap must never be reported as smaller than the detector's own
    # share of it — that is the shape of both previous mistakes.
    _ls = rw.RealtimeSession("CA000000000000000000000laggy",
                             Doctor(doctor_name="Dr. Jane Okafor",
                                    hospital_name="Northside Medical Group"))
    for _v in (0.0, 0.35, 2.9):
        _ls.detector_lags.append(_v)
        _ls.note_reply_latency(_v + 0.4)
    check(all(f >= d for f, d in zip(_ls.reply_latencies, _ls.detector_lags)),
          "every felt gap is at least the detector lag inside it")
    check(len(_ls.reply_latencies) == 3,
          "a zero-lag reply is still recorded — the floor cannot drop fast ones",
          f"{len(_ls.reply_latencies)} of 3")

    print("\n" + "=" * 66)
    print("  ASK BUDGET — a caller who names a place has engaged")
    print("=" * 66)
    # call-20260821-1931: the caller gave "Mission Bay Clinic, 1825 4th Street",
    # the live transcript mangled it to "Ford Street", grounding rejected the
    # model's correct reading, the re-ask hit the 4-ask limit, and the give-up
    # directive fired. The caller then repeated it cleanly — and grounding
    # PASSES on that transcript — but the agent had been told to stop.
    async def _budget_after_save(value):
        _bsess = rw.RealtimeSession("CA00000000000000000000budget",
                                    Doctor(doctor_name="Dr. Jane Okafor",
                                           hospital_name="Northside Medical Group"))
        _bsess.turns = [rw.TranscriptTurn(role="caller", timestamp="00:00:00",
                                          audio_rms=0.13, text=t)
                        for t in ("Okay, she's in San Francisco.",
                                  "It's the Mission Bay Clinic, 1825 Ford Street.")]
        _bsess._unanswered_asks = settings.realtime_max_unanswered_asks
        _bsess._asks_without_progress = settings.realtime_max_asks_without_progress
        _bsess._give_up_sent = True
        _bsess._give_up_at_turn = 2
        await rw._handle_tool_call(
            {"name": "save_branch", "call_id": "b1",
             "arguments": json.dumps({"branch": value, "city": "San Francisco"})},
            _bsess, _TcWS(), {}, True)
        return _bsess

    _bs_ok = await _budget_after_save("Mission Bay Clinic, 1825 4th Street")
    check(_bs_ok._unanswered_asks == 0 and _bs_ok._asks_without_progress == 0
          and not _bs_ok._give_up_sent,
          "a REJECTED save still resets BOTH counters — the caller answered",
          f"unanswered={_bs_ok._unanswered_asks} "
          f"no_progress={_bs_ok._asks_without_progress} "
          f"give_up={_bs_ok._give_up_sent}")
    check(not _bs_ok.memory.get("branch"),
          "and the rejection still stands — grounding is not weakened")
    # The reset must not be free: an empty value is not an answer.
    _bs_empty = await _budget_after_save("")
    check(_bs_empty._unanswered_asks == settings.realtime_max_unanswered_asks
          and _bs_empty._give_up_sent,
          "an empty branch resets nothing — that is not a caller answering")
    # And the OTHER budget must still bound a model that keeps offering junk.
    check(rw._MAX_SAVE_REJECTIONS >= 2,
          "rejected saves remain bounded by their own budget, which counts UP",
          f"{rw._MAX_SAVE_REJECTIONS}")
    _rej_src = _plb.Path(rw.__file__).read_text(encoding="utf-8")
    _reset = _rej_src[_rej_src.find("if str(args.get(\"branch\") or \"\").strip():"):][:420]
    check("_save_rejections" not in _reset,
          "and the save-rejection counter is NOT reset here — only the ask budget")

    import agents.voice.objectives as obj
    import agents.voice.tools as _tools
    import agents.voice.templates as _templates

    print("\n" + "=" * 66)
    print('  "YES" IS AN ANSWER — to a yes/no ask, and not to a place ask')
    print("=" * 66)
    # _is_filler_reply('Yes.') was True unconditionally, so
    # _caller_answered_since skipped that turn, the budget counted as though
    # nobody had spoken and the give-up directive fired. 'No.' returned False.
    # Only the POSITIVE answer was discarded — the one the client is calling to
    # collect. The function is global, so no template could have changed it.
    _PLACE = frozenset({obj.AnswerKind.PLACE})
    _CHOICE = frozenset({obj.AnswerKind.CHOICE})
    for _txt, _under_place, _under_choice, _why in [
        ("Yes.",   True,  False, "the whole bug in one string"),
        ("Yeah",   True,  False, "same word, spoken"),
        ("Yep.",   True,  False, ""),
        ("Yes, okay.", True, False, "padded with an acknowledgement"),
        # Unchanged in BOTH directions — the old behaviour on every other
        # string has to survive, or this is a new discard rather than a fix.
        ("No.",    False, False, "a refusal is information in either world"),
        ("Nope",   False, False, ""),
        ("Not sure, I'd have to check.", False, False, "UNSURE is an answer"),
        ("We're full — you'd be number twenty-one.", False, False,
         "the queue position the client actually wants"),
        ("She's at the Mission Bay Clinic.", False, False, "content is content"),
        # OUR OWN BACKCHANNEL CLIPS. mm-hm / okay / right / sure come back up
        # the line off a speakerphone and there is no way to tell them from the
        # caller saying the same thing — see _BACKCHANNEL_ECHO_MARGIN_S. They
        # stay filler even when a yes/no answer is what we asked for, or the
        # agent could answer its own question.
        ("Mm-hm.", True,  True,  "echo hazard: our own clip"),
        ("Okay.",  True,  True,  "echo hazard: our own clip"),
        ("Right.", True,  True,  "echo hazard: our own clip"),
        ("Sure.",  True,  True,  "echo hazard: our own clip"),
        ("Hello?", True,  True,  "a repair signal, not an answer"),
        ("Sorry, say again?", True, True, ""),
    ]:
        check(rw._is_filler_reply(_txt, "David", _PLACE) is _under_place,
              f"place ask: filler={_under_place!s:5} {_txt[:34]!r}", _why)
        check(rw._is_filler_reply(_txt, "David", _CHOICE) is _under_choice,
              f"yes/no ask: filler={_under_choice!s:5} {_txt[:34]!r}", _why)
    # No pending ask in view: judged exactly as it always was.
    check(rw._is_filler_reply("Yes.", "David") is True,
          "with no ask in view the old verdict stands — a bare yes is not a place",
          "the only ask this agent has ever made is for a place")

    # WHICH ASK IS IT. Form, not vocabulary: both of these name an office.
    _objv = obj.default_objective()
    for _ask, _want, _why in [
        ("Which branch is Dr. Okafor working out of?", _PLACE, "wh-form"),
        ("Could you tell me which office she practises at?", _PLACE, ""),
        ("I'm trying to confirm the address she works from.", _PLACE,
         "statement-form request"),
        ("Do you know which office she's at?", _PLACE,
         "opens with an auxiliary and is still a request for a place"),
        # The live clarification path. save_branch pushes back on "<City>
        # branch" asking for "confirmation this is their only location there",
        # the model asks exactly this, and the receptionist says "Yes."
        ("Is that your only office there?", _CHOICE, "the branch-clarification ask"),
        ("That's the only one, right?", _CHOICE, "tag question"),
        ("Are you accepting new patients at that location?", _CHOICE,
         "the field the next script adds"),
    ]:
        check(rw.expected_answers(_ask, _objv) == _want,
              f"expects {sorted(k.value for k in _want)}: {_ask[:46]!r}", _why)
    # One turn asking two things. Discarding either answer is the expensive
    # direction, so the expectation is a SET.
    _both = rw.expected_answers(
        "Are you accepting new patients — and which office is that?", _objv)
    check(_both == frozenset({obj.AnswerKind.PLACE, obj.AnswerKind.CHOICE}),
          "a compound ask entitles them to answer either half",
          f"{sorted(k.value for k in _both)}")

    # The four states, and the ORDER that keeps them apart.
    for _said, _want in [
        ("Yes, we are.", obj.ChoiceAnswer.YES),
        ("Yep, taking new patients right now.", obj.ChoiceAnswer.YES),
        ("No, we're not accepting new patients.", obj.ChoiceAnswer.NO),
        ("We're full, but I can put you on the waitlist.", obj.ChoiceAnswer.WAITLIST),
        ("You'd be number twenty-one in the queue.", obj.ChoiceAnswer.WAITLIST),
        ("We're full right now.", obj.ChoiceAnswer.WAITLIST),
        ("I'm not sure, you'd have to ask the doctor.", obj.ChoiceAnswer.UNSURE),
        ("It depends on the insurance.", obj.ChoiceAnswer.UNSURE),
        ("She's at the Mission Bay Clinic.", None),
    ]:
        check(obj.classify_choice(_said) == _want,
              f"choice={_want.value if _want else 'none':8} {_said[:44]!r}",
              "a queue position recorded as 'no' is the one fact the client "
              "would have acted on" if _want is obj.ChoiceAnswer.WAITLIST else "")
    # A value the process cannot recognise is not evidence — the same rule
    # grounding applies to a branch name, one field type later.
    _acc = obj.Field(name="accepting", memory_key="note_accepting",
                     kind=obj.AnswerKind.CHOICE, probe=obj.ACCEPTING_ASK,
                     spoken="new-patient status")
    check(_acc.present(_fake({"note_accepting": "waitlist"})) is True,
          "a CHOICE field holding a real state is collected")
    check(_acc.present(_fake({"note_accepting": "waitlist — number 21"})) is False,
          "and only the CANONICAL state — the tool canonicalises, so anything "
          "else in the field is a bug in the tool, not a sentence to re-read")
    check(_acc.present(_fake({"note_accepting": "The doctor is out sick today."})) is False,
          "a CHOICE field holding an unclassifiable value is NOT collected",
          "otherwise the objective resolves on a value nobody can read")

    print("\n" + "=" * 66)
    print("  SUCCESS CONDITION — declared by the template, not by save_branch")
    print("=" * 66)
    # save_branch() was the only function anywhere that set resolved=True.
    _mem_direct = rw.CallMemory("test-save-branch-direct")
    _mem_direct.clear()
    _tools.save_branch(_mem_direct, "Northgate Campus")
    check(_mem_direct.get("branch") == "Northgate Campus",
          "save_branch still records the field")
    check(_mem_direct.get("resolved") is None,
          "and no longer decides the call — nothing set resolved",
          f"{_mem_direct.get('resolved')!r}")
    # FIND, PROVE, JUDGE — locate the body first, so this cannot pass by
    # failing to find save_branch at all.
    _tsrc = _plb.Path(_tools.__file__).read_text(encoding="utf-8")
    _sb_body = _tsrc[_tsrc.find("def save_branch("):_tsrc.find("def note_info(")]
    check("memory.update(branch=branch" in _sb_body,
          "the save_branch body was located (this check is worth something)")
    # Checked as an ASSIGNMENT shape, not a bare substring — the body's own
    # comment explains why `resolved` no longer appears there, and the comment
    # saying so is not the defect the comment is warning about.
    check("resolved=" not in _sb_body.replace(" ", ""),
          "and nothing in it assigns resolved= any more")
    # Through run_tool, the verdict is derived and every existing reader of
    # memory['resolved'] still works.
    _mem_rt = rw.CallMemory("test-run-tool-outcome")
    _mem_rt.clear()
    _tools.run_tool("save_branch", _mem_rt, {"branch": "Northgate Campus"})
    check(_mem_rt.get("resolved") is True and _mem_rt.get("outcome") == "complete",
          "run_tool derives the verdict after the tool, from the objective",
          f"resolved={_mem_rt.get('resolved')!r} outcome={_mem_rt.get('outcome')!r}")

    # A two-field objective — the shape the next script has. PARTIAL is
    # EXPRESSIBLE, and whether it counts as success is declared, not decided
    # here: the client lead has not answered that for branch-without-accepting.
    _two = obj.CallObjective(fields=(obj.branch_field(), _acc))
    _lenient = obj.CallObjective(fields=(obj.branch_field(), _acc),
                                 success_at=obj.Outcome.PARTIAL)
    _snap_branch = _fake({"branch": "Northgate Campus"})
    # Canonical states, because Field.present is a membership check now: the
    # TOOL turns "yes, we are" into "yes", so a raw sentence sitting in the
    # field would be a tool bug rather than a value to re-read.
    _snap_acc = _fake({"note_accepting": "yes"})
    _snap_both = _fake({"branch": "Northgate Campus", "note_accepting": "yes"})
    check(_two.outcome(_fake({})) is obj.Outcome.NONE, "nothing collected -> NONE")
    check(_two.outcome(_snap_branch) is obj.Outcome.PARTIAL,
          "branch without the accepting status -> PARTIAL, not a failure",
          _two.outcome(_snap_branch).label)
    check(_two.outcome(_snap_acc) is obj.Outcome.PARTIAL,
          "accepting status without a branch -> PARTIAL, not NOTHING",
          "this is the call that recorded as NOT RESOLVED with real data in it")
    check(_two.outcome(_snap_both) is obj.Outcome.COMPLETE,
          "both fields -> COMPLETE")
    check(_two.missing(_snap_branch) == ("accepting",),
          "and the artifact names what is missing", f"{_two.missing(_snap_branch)}")
    check(_two.is_success(_snap_branch) is False
          and _lenient.is_success(_snap_branch) is True,
          "whether a partial is success is the TEMPLATE's call, not this file's",
          "success_at is the open question, declared rather than answered")
    check(_two.is_success(_snap_both) is True and _lenient.is_success(_fake({})) is False,
          "a lenient objective still requires something to have been collected")

    # The field write-back is no longer gated on the call-level verdict.
    _d_part = Doctor(doctor_name="Dr. B", hospital_name="H",
                     specialization="Cardiology")
    _s_part = rw.RealtimeSession("CA000000000000000000000partial", _d_part)
    _s_part.objective = _two
    _s_part.memory.update(branch="Northgate Campus")
    _rec_part = _s_part._enrich_doctor("Northgate Campus",
                                       _two.outcome(_s_part.memory))
    check(_d_part.branch == "Northgate Campus" and _d_part.source is rw.Source.VOICE,
          "a PARTIAL call still writes the branch it did get",
          "the old gate was `resolved and branch`, so a partial wrote nothing")
    check(_d_part.status is rw.DoctorStatus.PARTIALLY_VERIFIED,
          "but does not claim the record is VERIFIED", _d_part.status.value)
    check(_rec_part["status_before"] != _rec_part["status"],
          "and the change is recorded on the row")
    # Escalating after a save must not delete the half that worked.
    _mem_esc = rw.CallMemory("test-escalate-after-save")
    _mem_esc.clear()
    _tools.run_tool("save_branch", _mem_esc, {"branch": "Northgate Campus"})
    _tools.run_tool("escalate", _mem_esc,
                    {"reason": "could not obtain the new-patient status"})
    check(_mem_esc.get("branch") == "Northgate Campus"
          and _mem_esc.get("escalated") is True
          and _mem_esc.get("resolved") is True,
          "escalate records how the call ENDED and does not zero what it got",
          f"resolved={_mem_esc.get('resolved')!r}")
    # A template declaring a field no tool can write would report PARTIAL for
    # ever and blame the caller for it.
    _orphan = obj.CallObjective(fields=(obj.branch_field(),
                                        obj.Field(name="ghost",
                                                  memory_key="ghost",
                                                  kind=obj.AnswerKind.FREE,
                                                  probe=obj.LOCATION_NOUN),))
    check(obj.unwritable_fields(_orphan) == ("ghost",),
          "a field no tool writes is caught, not left to look like a refusal")
    for _name, _tpl in _templates.TEMPLATES.items():
        check(obj.unwritable_fields(_tpl.objective) == (),
              f"every field {_name} declares is one a tool can write")
        check(_tpl.objective.fields, f"{_name} declares what it collects")

    print("\n" + "=" * 66)
    print("  ASK BUDGET — spent by silence, not by answers")
    print("=" * 66)

    async def _drive_asks(asks, replies, *, objective=None):
        """Alternate agent asks and caller replies through the real handler."""
        _s = rw.RealtimeSession("CA00000000000000000000budget2",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.agent_name = "David"
        if objective is not None:
            _s.objective = objective
        _w = _TcWS()
        for _i, _ask in enumerate(asks):
            await rw._handle_agent_transcript({"transcript": _ask}, _s, _w, "", False)
            if _i < len(replies) and replies[_i] is not None:
                _s.add_turn("caller", replies[_i])
        _texts = [m["item"]["content"][0]["text"] for m in _w.sent
                  if m.get("type") == "conversation.item.create"
                  and m.get("item", {}).get("role") == "user"]
        _gave_up = any(m in t for m in rw.GIVE_UP_MARKERS.values() for t in _texts)
        return _s, _gave_up, _texts

    # THE HAPPY PATH: four DIFFERENT asks about the same field, four answers.
    # Under the old counter — which spent budget on ANSWERED asks — four
    # legitimate back-and-forths on one doctor was already a dead call; the
    # next script adds a second and third field per doctor on top of this, so
    # whatever survives here has to survive with room to spare. All four are
    # asks _is_location_ask actually recognises (each names a location noun),
    # since that recognition is what gates the whole mechanism today — see
    # the note on _is_location_ask about the one-field template it currently
    # serves.
    _hp_asks = [
        "Which branch is Dr. Okafor working out of?",
        "Is that the only office she has, or is there another location?",
        "Does she see patients at a different site as well?",
        "What's the street address for that branch?",
    ]
    _hp_replies = ["She's at the Mission Bay Clinic.", "Yes, that's the only one.",
                   "No, just that one.", "1825 Fourth Street."]
    _hp_sess, _hp_gave_up, _ = await _drive_asks(_hp_asks, _hp_replies)
    check(not _hp_gave_up,
          "four asks the caller ANSWERED do not end the call",
          "this is the new script's happy path, and it is per doctor")
    check(_hp_sess._unanswered_asks == 0,
          "nothing was charged to the budget", f"{_hp_sess._unanswered_asks}")
    check(_hp_sess._asks_without_progress == len(_hp_asks),
          "the no-progress ceiling still counted them — engaging is not "
          "supplying", f"{_hp_sess._asks_without_progress}")

    # THE INTERACTION with the filler filter, which is the point of doing both
    # changes together. Every reply here is a bare "Yes." to a yes/no ask: the
    # answer the client is calling to collect. Under the old filter each one
    # read as silence, so the counter that ends the call was reading nothing.
    #
    # This is the branch-clarification path that already exists in save_branch
    # today — "possibly the city restated ... confirmation this is their only
    # location there" — so every ask here is a real CHOICE-shaped turn the
    # current single-field template actually produces, not a hypothetical one.
    # Six, not five: the FIRST location-bearing ask in a call always counts as
    # answered regardless of the reply (there is no predecessor turn for it to
    # be unanswered against — see _last_ask_turn_idx's docstring), so proving
    # the silence case below needs one more than the ceiling to spare.
    _yn_asks = [
        "Is that your only office there?",
        "Is that the only branch, or is there another location?",
        "Is this the only site she practises from?",
        "Is that branch still her main office?",
        "Is the campus on Main Street her only location?",
        "Is that where she usually sees patients, at that address?",
    ]
    _yn_sess, _yn_gave_up, _ = await _drive_asks(_yn_asks, ["Yes."] * len(_yn_asks))
    check(not _yn_gave_up,
          "six yes/no asks answered 'Yes.' do not end the call",
          "with the old global filler filter every one of these was silence")
    check(_yn_sess._unanswered_asks == 0,
          "a bare 'Yes.' to a yes/no ask advances nothing",
          f"{_yn_sess._unanswered_asks}")

    # AND SILENCE STILL ENDS THE CALL. Same asks, nothing but "Hello."
    # coming back — the barge-in repair case, which must still be bounded.
    _sil_sess, _sil_gave_up, _sil_texts = await _drive_asks(
        _yn_asks, ["Hello?"] * len(_yn_asks))
    check(_sil_gave_up,
          "five asks answered with nothing but 'Hello?' DO end the call",
          "the budget still exists; it counts the right thing now")
    check(_sil_sess._unanswered_asks >= settings.realtime_max_unanswered_asks,
          "the unanswered counter is what ran out",
          f"{_sil_sess._unanswered_asks}/{settings.realtime_max_unanswered_asks}")
    check(_sil_sess._give_up_trigger == "unanswered"
          and rw.GIVE_UP_MARKERS["unanswered"] in " ".join(_sil_texts),
          "and the directive says so — they did not answer, they did not refuse",
          _sil_sess._give_up_trigger)
    check(rw.GIVE_UP_REASONS["unanswered"] in " ".join(_sil_texts),
          "the escalate reason it dictates is true of this call",
          "'caller engaged but never provided a location' was not")

    # THE OTHER WAY A CALL FAILS TO END: they answer every ask and supply
    # nothing. The old counter bounded this by accident, so removing it means
    # stating the bound.
    _chat = ["We get a lot of calls like this.", "It's been a busy morning.",
             "That's how it goes some days.", "The doctors are all in today.",
             "We had the phones down last week.", "It's quieter after lunch.",
             "Everyone's at a conference next week.", "That happens a lot.",
             "We're a big practice.", "Always something."]
    _ceil_asks = [f"Which office is Dr. Okafor at on {d}?" for d in
                  ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "the weekend", "holidays", "most weeks", "this week",
                   "next week")]
    _nc_sess, _nc_gave_up, _nc_texts = await _drive_asks(_ceil_asks, _chat)
    check(_nc_gave_up, "a caller who engages and never supplies still ends up "
                       "closed out — there is no duration cap to catch this")
    check(_nc_sess._give_up_trigger == "no_progress"
          and rw.GIVE_UP_MARKERS["no_progress"] in " ".join(_nc_texts),
          "and it is reported as no progress, not as silence",
          _nc_sess._give_up_trigger)
    check(settings.realtime_max_asks_without_progress
          > settings.realtime_max_unanswered_asks,
          "the liveness ceiling sits above the budget, or it would be the "
          "budget", f"{settings.realtime_max_asks_without_progress} vs "
                    f"{settings.realtime_max_unanswered_asks}")
    check(settings.realtime_max_asks_without_progress >= len(_hp_asks) + 1,
          "and above the happy path it has to survive",
          f"{settings.realtime_max_asks_without_progress} vs {len(_hp_asks)}")
    # Progress resets it, which is what makes ONE ceiling work for several
    # doctors in one call.
    _pr_sess, _, _ = await _drive_asks(_ceil_asks[:3], _chat[:3])
    check(_pr_sess._asks_without_progress == 3, "asks accumulate without progress")
    _pr_sess.reset_ask_budget("collected branch")
    check(_pr_sess._asks_without_progress == 0 and _pr_sess._unanswered_asks == 0
          and not _pr_sess._give_up_sent and _pr_sess._vetting_reasks == 0,
          "and one collected field clears every counter, for the next doctor",
          "three hand-written reset sites disagreed about which to clear")

    print("\n" + "=" * 66)
    print("  TEMPLATE 3 — provider verification: branch AND new-patient status")
    print("=" * 66)
    _PV = _templates.PROVIDER_VERIFICATION
    _PVO = _templates.PROVIDER_VERIFICATION_OBJECTIVE

    # WHAT IT COLLECTS. Two fields, branch first, both required.
    check([f.name for f in _PVO.fields][:3] == ["identity", "branch", "accepting"],
          "declares identity FIRST, then branch, then accepting",
          f"{[f.name for f in _PVO.fields]}")
    check(all(f.required for f in _PVO.fields),
          "all of them are required — conditionally for the later two, which is "
          "what required_when expresses")
    _acc_field = _PVO.field_named("accepting")
    check(_acc_field is not None
          and _acc_field.kind is obj.AnswerKind.CHOICE
          and _acc_field.probe is obj.ACCEPTING_ASK,
          "the accepting field is CHOICE-kind and probes with ACCEPTING_ASK",
          "reusing what objectives.py already defines, not a second pattern")
    # THE FAILURE unwritable_fields() EXISTS FOR. Run against the real objective:
    # a memory_key no tool writes reports PARTIAL for ever and reads, in the
    # artifact, as the caller declining to answer.
    check(obj.unwritable_fields(_PVO) == (),
          "every field it declares is one a tool actually writes",
          "a key nothing writes is a template bug wearing a receptionist's "
          "clothes")
    check(_acc_field is not None
          and _acc_field.memory_key == obj.NEW_PATIENT_STATUS_KEY,
          "and the field points at the key the tool writes, by constant not "
          "by spelling")
    # THE OPEN QUESTION, left open and asserted as such. If this ever flips to
    # PARTIAL it should be because someone decided, not because it drifted.
    check(_PVO.success_at is obj.Outcome.COMPLETE,
          "success_at is still STRICT — branch-without-status is not a success",
          "flagged in the diff; change it when the client answers, not before")
    _pv_branch_only = _fake({obj.IDENTITY_STATUS_KEY: "confirmed",
                             "branch": "Mission Bay Clinic"})
    check(_PVO.outcome(_pv_branch_only) is obj.Outcome.PARTIAL
          and _PVO.is_success(_pv_branch_only) is False,
          "so branch-only reports PARTIAL and resolved=False",
          "partial is EXPRESSIBLE either way — that is the point of the "
          "three-valued outcome")
    check(_PVO.missing(_pv_branch_only) == ("accepting",),
          "and the artifact names only what is actually still owed",
          "scheduling and referral are not owed until accepting comes back yes")

    # FCC: OUR IDENTIFICATION FIRST. Neither client script names the caller or
    # the organisation; ours must, and their disclaimer folds in after it.
    _pv_greet = _PV.build_greeting(
        Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside"),
        org="Definitive Healthcare", agent_name="Alex")
    _ident_at = min(_pv_greet.find("Alex"), _pv_greet.find("Definitive Healthcare"))
    _disclaim_at = _pv_greet.lower().find("not calling to book")
    check(_ident_at >= 0 and "Definitive Healthcare" in _pv_greet,
          "the greeting names the caller AND the organisation", _pv_greet[:60])
    check(_disclaim_at > _ident_at,
          "the client's not-booking line comes AFTER the identification, "
          "not instead of it", _pv_greet)
    check("on behalf of" in _pv_greet,
          "and still says 'on behalf of' — it does not claim employment")

    print("\n" + "-" * 66)
    print("  save_new_patient_status — four states, and a value it can read")
    print("-" * 66)
    for _status, _want_ok in [("yes", True), ("no", True), ("waitlist", True),
                              ("unsure", True), ("maybe", False), ("", False)]:
        _m = rw.CallMemory(f"test-nps-{_status or 'empty'}")
        _m.clear()
        _r = _tools.save_new_patient_status(_m, _status, heard="whatever")
        check(bool(_r.get("ok")) is _want_ok,
              f"status={_status!r:10} accepted={_want_ok}",
              str(_r.get("error", ""))[:50])
        if _want_ok:
            check(_m.get(obj.NEW_PATIENT_STATUS_KEY) == _status,
                  f"  and {_status!r} is stored under the declared key")
    # A whole sentence rather than a state word is classified, not bounced — the
    # model saying "they're full but there's a waitlist" is a real answer.
    _m_sent = rw.CallMemory("test-nps-sentence")
    _m_sent.clear()
    _r_sent = _tools.save_new_patient_status(
        _m_sent, "they're full but you'd be number 21", heard="you'd be number 21")
    check(_r_sent.get("ok") and _m_sent.get(obj.NEW_PATIENT_STATUS_KEY) == "waitlist",
          "a sentence is classified rather than rejected — and a queue position "
          "is WAITLIST, not no",
          f"{_m_sent.get(obj.NEW_PATIENT_STATUS_KEY)!r}")
    check(_m_sent.get(f"{obj.NEW_PATIENT_STATUS_KEY}_heard") == "you'd be number 21",
          "their own words are kept beside the state",
          "the state is what the directory filters on; the wording is what a "
          "reviewer needs to believe it")
    # An unreadable value must not satisfy the objective — same rule save_branch
    # applies to a branch name.
    _m_junk = rw.CallMemory("test-nps-junk")
    _m_junk.clear()
    _m_junk.update(**{obj.NEW_PATIENT_STATUS_KEY: "the doctor is out sick"})
    check(_PVO.outcome(_m_junk) is obj.Outcome.NONE,
          "a value the process cannot classify is NOT collected",
          "otherwise the objective resolves on something no reader can act on")

    print("\n" + "-" * 66)
    print("  Grounding the status — why classify_choice(blob) is not enough")
    print("-" * 66)

    def _status_sess(turns, *, asked=True):
        _s = rw.RealtimeSession("CA000000000000000000000status",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _PVO
        _s.agent_name = "Alex"
        if asked:
            _s.add_turn("agent", "Is Dr. Okafor accepting new patients?")
        for _text, _rms in turns:
            _s.turns.append(rw.TranscriptTurn(role="caller", text=_text,
                                              timestamp="00:00:00",
                                              audio_rms=_rms))
        return _s

    # A REAL ANSWER ON REAL AUDIO GROUNDS.
    check(rw._ungrounded_status({"status": "yes"},
                                _status_sess([("Yes, she is taking new patients.", 0.14)]))
          == "",
          "a real 'yes' on real audio grounds")
    check(rw._ungrounded_status({"status": "waitlist"},
                                _status_sess([("We're full, but I can put you on the list.", 0.13)]))
          == "",
          "and so does a waitlist answer in their own words")

    # THE BLOB FAILURE, which is the whole reason this is not
    # classify_choice(everything the caller said). "Yes, speaking." at pickup
    # is not an answer to a question that was asked afterwards.
    _blob = _status_sess([], asked=False)
    _blob.turns.append(rw.TranscriptTurn(role="caller", text="Yes, speaking.",
                                         timestamp="00:00:00", audio_rms=0.14))
    _blob.add_turn("agent", "Is Dr. Okafor accepting new patients?")
    _blob.turns.append(rw.TranscriptTurn(role="caller", text="One moment.",
                                         timestamp="00:00:02", audio_rms=0.14))
    check(obj.classify_choice("Yes, speaking.") is obj.ChoiceAnswer.YES,
          "a blob check WOULD have found a yes in this call",
          "which is what makes the next assertion worth something")
    check(rw._ungrounded_status({"status": "yes"}, _blob) != "",
          "but 'Yes, speaking.' said BEFORE the question does not ground a yes",
          "callers say yes constantly for other reasons — a status is two bits "
          "and has no distinctiveness to protect it")

    # ASKED BACK IS NOT TOLD. Same predicate the location check uses.
    check(rw._ungrounded_status(
              {"status": "yes"},
              _status_sess([("Accepting new patients — is that what you're asking?", 0.14)]))
          != "",
          "a receptionist asking the question BACK does not ground the answer")

    # THE BARE TOKEN ON DEAD AIR. For this field the audio measurement is the
    # only signal left, because a genuine answer IS bare.
    check(rw._ungrounded_status({"status": "yes"},
                                _status_sess([("Yes.", 0.002)])) != "",
          "a bare 'Yes.' on silent audio is refused — that is what a "
          "transcription artefact looks like")
    check(rw._ungrounded_status({"status": "yes"},
                                _status_sess([("Yes.", 0.14)])) == "",
          "but a bare 'Yes.' on real audio is a real answer",
          "bare is the NORMAL shape of this answer; only the audio separates them")
    # Unmeasured audio gets the benefit of the doubt, like every other guard here.
    check(rw._ungrounded_status({"status": "yes"},
                                _status_sess([("Yes.", None)])) == "",
          "an unmeasured turn is not treated as fabricated",
          "absence of measurement is not evidence")
    # Nothing transcribed at all -> do not block. Same conservative direction.
    check(rw._ungrounded_status({"status": "yes"}, _status_sess([])) == "",
          "no caller speech since the ask -> the guard stands down rather than "
          "blocking every save")
    # ...BUT "NOTHING SINCE THE ASK" ON A CALL THAT IS TRANSCRIBING IS NOT
    # SILENCE EITHER. call-20260825-1731: the agent asked "Is this Dr. Reyes,
    # Oncology, at Lakeview Medical?", the wait for the transcript timed out,
    # this branch stood down, and identity saved CONFIRMED on the model's own
    # unchecked string - heard: "Okay." - which classify_identity does not even
    # read as a confirmation. The `detail` guard on the SAME tool call reported
    # "'reyes', 'oncology' never appeared in the caller transcript".
    #
    # Three seconds later the caller said "Yes, Dr. Rayef is our oncologist." -
    # the real answer, surname mangled - and because identity was already
    # confirmed, _wrong_doctor_named never ran and the spelling repair never
    # fired. One permissive branch disabled the whole chain under it.
    _silent_since = _status_sess([("Yeah, it's a good time.", 0.14)])
    _silent_since.turns.append(rw.TranscriptTurn(
        role="agent", text="Is Dr. Okafor accepting new patients?",
        timestamp="00:00:00"))
    check(rw._ungrounded_status({"status": "yes"}, _silent_since) != "",
          "nothing since the ask, on a call that IS transcribing -> blocked",
          "standing down accepts the field on no evidence at all, which is "
          "strictly weaker than the asked-back case this function already "
          "refuses twenty lines below")
    check(not _silent_since.unverified_quotes,
          "and a refusal is not recorded as an unchecked quote",
          "nothing was accepted, so there is no quote to mark")

    print("\n" + "-" * 66)
    print("  A save whose evidence is still in flight")
    print("-" * 66)
    # call-20260826-1422: six saves, six waits, SIX TIMEOUTS, zero landed. Every
    # rejection was followed inside the same second by the caller transcript
    # carrying the answer. The model read "nothing has been transcribed" as
    # "they did not answer", apologised, and re-asked twice on a happy path.
    #
    # The wait is left exactly as it was. What changed is what happens when it
    # expires: the decision is deferred to the event that carries the evidence
    # instead of being taken without it.
    import agents.voice.evidence as _ev_mod

    class _DefWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    def _pending_sess(ask="Is Dr. Okafor accepting new patients?"):
        """A caller turn that has stopped but not yet transcribed."""
        _s = rw.RealtimeSession("CA0000000000000000000defer",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _PVO
        _s.agent_name = "Alex"
        # A REAL EARLIER TURN, because _ungrounded_status stands down entirely
        # when NOTHING has ever transcribed on the call — that branch exists for
        # a line transcription cannot read, and a fixture without it tests the
        # fallback rather than the race.
        _s.add_turn("agent", "Is this Dr. Okafor at Northside Medical Group?")
        _s.turns.append(rw.TranscriptTurn(
            role="caller", text="Yes, that's right.",
            timestamp="00:00:00", audio_rms=0.11))
        # And an OPEN save gate: accepting is gated on a confirmed identity, so
        # without this the gate refuses first and the grounding path never runs.
        _s.memory.update(doctor_identity="confirmed", new_patient_status="")
        _s.add_turn("agent", ask)
        _s.add_turn("caller", "[...]")
        # The transcriber has NOT answered for this placeholder yet — the exact
        # state _transcript_pending exists to name.
        _s._placeholder_at = time.monotonic()
        _s._transcript_at = 0.0
        return _s

    def _land(sess, text):
        """The transcript arrives: replace the placeholder, as the handler does."""
        for _i in range(len(sess.turns) - 1, -1, -1):
            if sess.turns[_i].role == "caller" and sess.turns[_i].text == "[...]":
                sess.turns[_i] = rw.TranscriptTurn(
                    role="caller", text=text, timestamp="00:00:00", audio_rms=0.12)
                break
        sess._transcript_at = time.monotonic()

    async def _propose(sess, ws, tool="save_new_patient_status", **args):
        await rw._handle_tool_call(
            {"call_id": "d1", "name": tool, "arguments": json.dumps(args)},
            sess, ws, {}, False)

    # The 1.5s wait is real and is deliberately NOT changed by this work; it is
    # shortened here so six scenarios do not cost nine seconds of suite time.
    # What is under test is the branch taken when it expires, not its length.
    # THE BLOCKING WAIT THESE WERE PINNED AROUND IS GONE. Six scenarios used
    # to be run with _TRANSCRIPT_WAIT_S dropped to 0.05 so the suite did not
    # spend nine seconds sleeping. Measured over 119 call artifacts the wait
    # never once returned early — 14 waits, 12 timeouts, 0 landed — so it was
    # deleted and the deferral below is now the whole mechanism, which is what
    # these scenarios were really testing all along.
    # ── 1. DELAYED TRANSCRIPT: the answer is late, not absent ────────────
    _s1 = _pending_sess(); _w1 = _DefWS()
    await _propose(_s1, _w1, status="yes",
                   heard="Yes, she's taking on new patients right now.")
    check(_s1._deferred_save is not None,
          "a save proposed before its transcript is HELD, not refused",
          "refusing here is what produced the apology and the re-ask")
    check(_s1.memory.get("new_patient_status") in (None, ""),
          "and nothing is written while it is held",
          f"{_s1.memory.get('accepting_new_patients')!r}")
    _out1 = [m for m in _w1.sent
             if m.get("item", {}).get("type") == "function_call_output"]
    _res1 = json.loads(_out1[0]["item"]["output"]) if _out1 else {}
    check(_res1.get("ok") is True and _res1.get("pending") is True,
          "the model is told it is in hand, not that nobody answered",
          f"{_res1}")
    check("ask again" not in json.dumps(_res1).lower()
          or "do not ask again" in json.dumps(_res1).lower(),
          "and is not invited to re-ask")

    _land(_s1, "Yes, she's taking on new patients right now.")
    await rw._resolve_deferred_save(_s1, _w1)
    check(_s1.memory.get("new_patient_status") == "yes",
          "when the words land the held save is applied",
          f"{_s1.memory.get('accepting_new_patients')!r}")
    check(_s1.deferred_saves and _s1.deferred_saves[-1]["outcome"] == "applied",
          "and the artifact says it was applied late, not silently",
          f"{_s1.deferred_saves}")
    check(not any("ask them again" in json.dumps(m).lower() for m in _w1.sent),
          "and NO re-ask directive is injected — the question was answered")

    # ── 2. MISSING TRANSCRIPT: nothing may be written ────────────────────
    _s2 = _pending_sess(); _w2 = _DefWS()
    await _propose(_s2, _w2, status="yes", heard="Yes, she is.")
    check(_s2._deferred_save is not None, "a save with no transcript is held")
    # The call ends here. Nothing arrives, nothing resolves.
    check(_s2.memory.get("new_patient_status") in (None, ""),
          "a transcript that never comes writes NOTHING",
          "this is the fabrication the deferral must not introduce")
    _held2 = _s2._deferred_save
    check(_held2 is not None and _held2["args"]["status"] == "yes",
          "and the unresolved save is still on the session for the artifact")

    # ── 3. CONTRADICTORY TRANSCRIPT: the words refuse it ─────────────────
    _s3 = _pending_sess(); _w3 = _DefWS()
    await _propose(_s3, _w3, status="yes", heard="Yes, she is.")
    _land(_s3, "No, she is not taking any new patients at the moment.")
    await rw._resolve_deferred_save(_s3, _w3)
    check(_s3.memory.get("new_patient_status") != "yes",
          "a 'yes' the caller contradicted is NOT saved",
          f"{_s3.memory.get('accepting_new_patients')!r}")
    check(_s3.deferred_saves and _s3.deferred_saves[-1]["outcome"] == "contradicted",
          "recorded as contradicted, with the guard's reason",
          f"{_s3.deferred_saves}")
    check(any("not been saved" in json.dumps(m).lower() for m in _w3.sent),
          "and NOW the model is told to ask again — there is evidence for it",
          "the re-ask this mechanism prevents is the UNEVIDENCED one")

    # ── 4. ALREADY ANSWERED: the answer lands after the guard ran ────────
    # The shape of the live failure, end to end: ask, propose, evidence
    # arrives late, and the question must not be put again.
    _s4 = _pending_sess(); _w4 = _DefWS()
    await _propose(_s4, _w4, status="yes",
                   heard="Yes, she's taking on new patients right now.")
    _land(_s4, "Yes, she's taking on new patients right now.")
    await rw._resolve_deferred_save(_s4, _w4)
    check(_s4.memory.get("new_patient_status") == "yes"
          and not any("ask them again" in json.dumps(m).lower() for m in _w4.sent),
          "an answered question is saved and never re-asked",
          f"saved={_s4.memory.get('accepting_new_patients')!r} "
          f"injections={[m for m in _w4.sent if m.get('type') == 'conversation.item.create']}")

    # ── 5. SCHEDULING TAKES THE SAME PATH ────────────────────────────────
    # Both tools that failed on the live call, not just the first.
    _s5 = _pending_sess("Can a new patient book an appointment right now?")
    _s5.memory.update(new_patient_status="yes")   # scheduling's gate
    _w5 = _DefWS()
    await _propose(_s5, _w5, tool="save_scheduling_status", status="yes",
                   heard="Yes, you can book online or call us directly also.")
    check(_s5._deferred_save is not None,
          "save_scheduling_status defers on the same condition")
    _land(_s5, "Yes, you can book online or call us directly also.")
    await rw._resolve_deferred_save(_s5, _w5)
    check(_s5.memory.get("scheduling_status") == "yes",
          "and applies when the words land",
          f"{_s5.memory.get('scheduling_available')!r}")

    # -- THE DEFERRED PATH CAN NOW END THE CALL - call-20260827-1010 --------
    # The synchronous handler sets sess.done when a successful save completes
    # the objective. This path runs the SAME tool, for real, a turn later, and
    # had no such check: on 1010 `outcome=complete` printed at 10:11:34 and the
    # call then ran another 24 seconds and four agent turns, ending on a
    # "Take care." the model had already said once.
    _sC = _pending_sess(); _wC = _DefWS()
    _sC.memory.update(branch="Riverside Campus")   # the last field outstanding
    await _propose(_sC, _wC, status="no", heard="No, she's not.")
    check(_sC._deferred_save is not None,
          "held while the words are in flight, as before")
    check(_sC._close_after_response is False,
          "and nothing is closing yet - the save has not actually happened")
    _land(_sC, "No, she's not taking new patients.")
    await rw._resolve_deferred_save(_sC, _wC)
    check(_sC.memory.get("new_patient_status") == "no",
          "the held save applies when the words land")
    check(_sC._close_after_response is True,
          "and an objective completing on THAT save now closes the call",
          "before this, the deferred path could not end a call at all")
    # THE FLAG, NOT THE GOODBYE. This runs inside the caller-transcript
    # handler, which shares no mutable state with the event loop. Injecting the
    # closing here would leave `_closing_sent` False, so the in-flight
    # response's own response.done would read "done, nothing pending" and hang
    # up ON the goodbye we just asked for.
    check(not any("goodbye" in json.dumps(m) for m in _wC.sent),
          "the goodbye is NOT injected here",
          "`_closing_sent` is a local of the event loop and the in-flight "
          "response has not spoken yet - both are answerable one event later")
    check(_sC.done is False,
          "and sess.done is not set here either - response.done owns that")

    # STILL PARTIAL: NO CLOSE. The mutation that matters, because a flag that
    # is always set hangs up on every deferred save the call ever makes.
    _sP = _pending_sess(); _wP = _DefWS()
    await _propose(_sP, _wP, status="no", heard="No, she's not.")
    _land(_sP, "No, she's not taking new patients.")
    await rw._resolve_deferred_save(_sP, _wP)
    check(_sP.memory.get("new_patient_status") == "no",
          "a save that leaves the branch outstanding still applies")
    check(_sP._close_after_response is False,
          "but does not close the call - the branch is still missing",
          f"missing={_PVO.missing(_sP.memory)}")

    # The other half lives in the event loop, where `_closing_sent` is a local
    # and the last agent turn exists. Asserted on the source, because those two
    # are in scope nowhere else - the same reason the response.create policy
    # checks in this suite read source.
    # Reads the whole package: the response.done handler moved to lifecycle.py
    # on 2026-08-27, and an assertion pinned to one FILE is an assertion that
    # goes quiet the day the code is extracted — which is the moment it is most
    # worth having.
    _wsrc = _PKG_SRC
    _cb = _wsrc[_wsrc.index("if sess._close_after_response and not sess.done:"):]
    _cb = _cb[:_cb.index("if sess.done:")]
    check("_sounded_like_a_goodbye" in _cb,
          "the loop re-runs the goodbye-shape test before hanging up",
          "hanging up on a question left a caller answering a dead line")
    check("endswith(\"?\")" in _cb,
          "and it is the same test: an utterance ending in '?' is not a "
          "farewell")
    check("_closing_sent = True" in _cb,
          "the closing it asks for is bookkept, so THIS response.done is "
          "consumed and the goodbye's own one reaches the hang-up branch")
    check("sess._response_active" in _cb,
          "and it stands down while a response is already in flight",
          "asking for the goodbye into a live response is the collision this "
          "module has been bitten by twice; the flag survives to try again")

    # ── EVIDENCE PRESENT AND WRONG IS STILL REFUSED IMMEDIATELY ─────────────
    # The deferral must key on "the words have not arrived", never on "the
    # guard objected". A version that deferred every objection would hold a
    # real contradiction until the call ended and re-ask nothing.
    _s6 = _pending_sess(); _w6 = _DefWS()
    _land(_s6, "No, she is not.")          # words are IN, and they say no
    await _propose(_s6, _w6, status="yes", heard="Yes, she is.")
    check(_s6._deferred_save is None,
          "with the transcript already in hand a bad save is refused ON THE SPOT",
          "deferring a contradiction would delay every correction to the hangup")
    _out6 = [m for m in _w6.sent
             if m.get("item", {}).get("type") == "function_call_output"]
    check(_out6 and json.loads(_out6[0]["item"]["output"]).get("ok") is False,
          "and the model is told so immediately",
          f"{_out6}")

    print("\n" + "-" * 66)
    print("  A BRANCH save whose evidence is still in flight")
    print("-" * 66)
    # The deferral landed on the choice fields on 2026-08-26 and save_branch was
    # missed out of it. call-20260827-0942 is what that cost, twice on one call:
    #
    #   waited 1.50s for the transcript and it never came
    #   BLOCKED {"branch": "Riverside campus"}
    #   CALLER : He works out at Riverside Campus.        <- one line later
    #
    #   waited 1.50s ... never came
    #   BLOCKED {"branch": "Riverside campus, 1477 10th Street"}
    #           (numbers 10, 1477 not in what the caller said)
    #   CALLER : I think it's 1477 10th Street.           <- one line later
    #
    # transcript_waits recorded two timeouts and deferred_saves was null: the
    # hold existed and this path could not reach it.
    import agents.voice.evidence as _ev_mod2

    class _BrWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    def _br_sess(ask="Which branch does Dr. Jennifer work out of?"):
        _s = rw.RealtimeSession("CA00000000000000000branch",
                                Doctor(doctor_name="Dr. Jennifer",
                                       hospital_name="New York Baptist Hospital"))
        _s.objective = _PVO
        _s.add_turn("agent", "Is this Dr. Jennifer's office?")
        _s.turns.append(rw.TranscriptTurn(role="caller", text="Yes, that's right.",
                                          timestamp="00:00:00", audio_rms=0.11))
        _s.memory.update(doctor_identity="confirmed")
        _s.add_turn("agent", ask)
        _s.add_turn("caller", "[...]")
        _s._placeholder_at = time.monotonic()
        _s._transcript_at = 0.0
        return _s

    def _br_land(sess, text):
        for _i in range(len(sess.turns) - 1, -1, -1):
            if sess.turns[_i].role == "caller" and sess.turns[_i].text == "[...]":
                sess.turns[_i] = rw.TranscriptTurn(
                    role="caller", text=text, timestamp="00:00:00", audio_rms=0.12)
                break
        sess._transcript_at = time.monotonic()

    async def _br_propose(sess, ws, **args):
        await rw._handle_tool_call(
            {"call_id": "b1", "name": "save_branch", "arguments": json.dumps(args)},
            sess, ws, {}, False)

    # Same deletion as above: no wait left to pin, the deferral is the whole
    # mechanism these races are about.
    # ── 0942 race 1: the site name ──────────────────────────────────────
    _b1, _bw1 = _br_sess(), _BrWS()
    await _br_propose(_b1, _bw1, branch="Riverside campus")
    check(_b1._deferred_save is not None,
          "a branch proposed before its transcript is HELD, not blocked",
          "this path rejected, and the caller was asked a third time")
    check(not _b1.memory.get("branch"),
          "and nothing is written while it is held",
          f"{_b1.memory.get('branch')!r}")
    _out1 = [m for m in _bw1.sent
             if m.get("item", {}).get("type") == "function_call_output"]
    check(_out1 and json.loads(_out1[0]["item"]["output"]).get("pending") is True,
          "the model is told it is in hand, not that it invented the name",
          f"{_out1}")

    _br_land(_b1, "He works out at Riverside Campus.")
    await rw._resolve_deferred_save(_b1, _bw1)
    check(_b1.memory.get("branch") == "Riverside campus",
          "and when the words land the branch is saved",
          f"{_b1.memory.get('branch')!r}")
    check(_b1.deferred_saves and _b1.deferred_saves[-1]["outcome"] == "applied",
          "recorded as applied late, not silently",
          f"{_b1.deferred_saves}")

    # ── 0942 race 2: the digits ─────────────────────────────────────────
    # The rejection there was specifically "numbers 10, 1477 not in what the
    # caller said" - the digit guard firing on words that had not arrived.
    _b2, _bw2 = _br_sess("Could you give the address?"), _BrWS()
    await _br_propose(_b2, _bw2, branch="Riverside campus, 1477 10th Street")
    check(_b2._deferred_save is not None,
          "a branch with digits is held on the same condition")
    _br_land(_b2, "I think it's 1477 10th Street at the Riverside campus.")
    await rw._resolve_deferred_save(_b2, _bw2)
    check(_b2.memory.get("branch") == "Riverside campus, 1477 10th Street",
          "and the digits ground once the words are actually there",
          f"{_b2.memory.get('branch')!r}")

    # ── a branch the caller never said is still refused ─────────────────
    _b3, _bw3 = _br_sess(), _BrWS()
    await _br_propose(_b3, _bw3, branch="Eastside Clinic")
    _br_land(_b3, "He works out at Riverside Campus.")
    await rw._resolve_deferred_save(_b3, _bw3)
    check(not _b3.memory.get("branch"),
          "a site name the words do not support is NOT saved",
          f"{_b3.memory.get('branch')!r}")
    check(_b3.deferred_saves and _b3.deferred_saves[-1]["outcome"] == "contradicted",
          "recorded as contradicted", f"{_b3.deferred_saves}")
    check(any("not been saved" in json.dumps(m).lower() for m in _bw3.sent),
          "and NOW the model is told to ask again — there is evidence for it")

    # ── EVIDENCE ALREADY IN HAND IS STILL REFUSED ON THE SPOT ──────────────
    # The hold must key on "the words have not arrived", never on "the guard
    # objected". Deferring a real fabrication would delay every correction to
    # the hangup — and 'Eastside Clinic' is a name a live agent has invented.
    _b4, _bw4 = _br_sess(), _BrWS()
    _br_land(_b4, "He works out at Riverside Campus.")
    await _br_propose(_b4, _bw4, branch="Eastside Clinic")
    check(_b4._deferred_save is None,
          "with the transcript in hand a fabricated branch is blocked immediately",
          "deferring a fabrication would be a regression, not a fix")
    _out4 = [m for m in _bw4.sent
             if m.get("item", {}).get("type") == "function_call_output"]
    check(_out4 and json.loads(_out4[0]["item"]["output"]).get("ok") is False,
          "and the model is told so at once", f"{_out4}")

    # ── THE PREDICATE ITSELF ────────────────────────────────────────────────
    _sp = _pending_sess()
    check(rw._transcript_pending(_sp) is True,
          "a placeholder the transcriber has not answered for is PENDING")
    _sp._transcript_at = _sp._placeholder_at + 0.01
    check(rw._transcript_pending(_sp) is False,
          "one it HAS answered for — and the answer was discarded — is not",
          "waiting on that spends the ceiling for evidence that will never come")
    _sp2 = _pending_sess(); _land(_sp2, "Yes, she is.")
    check(rw._transcript_pending(_sp2) is False,
          "and a turn with real words is not pending either")

    print("\n" + "-" * 66)
    print("  The caller asking to hang up")
    print("-" * 66)
    # call-20260826-1656 ran 193s; the last 23 were the caller saying bye four
    # times while the agent said it back. sess.done had two triggers - objective
    # COMPLETE and escalate - and neither was reachable by the caller.
    for _t in ("Thank you bye Cut the call.", "Bye-bye.", "Bye, I said.",
               "Goodbye.", "okay bye", "please cut the call", "end the call now",
               "can you hang up"):
        check(rw._caller_ends_call(_t), f"caller ending recognised: {_t!r}")

    # NO SENTIMENT. "How many times you will tell me bro?" is the clearest
    # statement of intent on that whole call and it is deliberately NOT matched:
    # reading frustration is a judgement, and one that hangs up on a caller
    # mid-sentence is the expensive direction.
    for _t in ("How many times you will tell me bro?",
               "By the way, she is at Riverside.",       # \bbye\b needs the e
               "I will pass you by the front desk.",
               "Hahaha.", "Yes, she is taking new patients.",
               "No, nothing else."):
        check(not rw._caller_ends_call(_t),
              f"not treated as an ending: {_t!r}",
              "a false positive here hangs up on someone mid-sentence")

    class _EndWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    def _end_sess(prior_agent="Is a referral required?"):
        _s = rw.RealtimeSession("CA000000000000000000000end",
                                Doctor(doctor_name="Dr. Jennifer",
                                       hospital_name="New York Baptist Hospital"))
        _s.objective = _PVO
        _s.add_turn("agent", prior_agent)
        _s.add_turn("caller", "[...]")
        _s._placeholder_at = time.monotonic()
        return _s

    async def _land(sess, ws, text):
        await rw._handle_caller_transcript(
            {"transcript": text}, sess, ws)

    # ── the flag is set, and nothing else is ────────────────────────────────
    _e1, _w1 = _end_sess(), _EndWS()
    await _land(_e1, _w1, "Thank you bye. Cut the call.")
    check(_e1.done is True, "a caller farewell sets sess.done",
          "the only flag that reaches the hangup branch")
    check(_e1.ended_by_caller and "cut the call" in _e1.ended_by_caller.lower(),
          "and the caller's own words are recorded for the artifact",
          f"{_e1.ended_by_caller!r}")

    # RESTRAINT IS THE DESIGN. The caller just spoke, so OpenAI's VAD already
    # opened a response; that one plays the farewell and its response.done
    # reaches the hangup. Injecting a second goodbye here would collide with the
    # one in flight - the collision this module has been bitten by twice.
    check(not any(m.get("type") == "response.create" for m in _w1.sent),
          "and NO second response is created from this path",
          f"{[m.get('type') for m in _w1.sent]}")
    check(not any("goodbye" in json.dumps(m).lower() for m in _w1.sent),
          "and no goodbye is injected either")

    # ── an ordinary turn is untouched ──────────────────────────────────────
    _e2, _w2 = _end_sess(), _EndWS()
    await _land(_e2, _w2, "Yes, a referral is always required.")
    check(_e2.done is False and _e2.ended_by_caller is None,
          "an ordinary answer does not end the call",
          f"done={_e2.done} ended_by={_e2.ended_by_caller!r}")

    # ── ENDING MUST NOT COST A FIELD THE CALLER JUST GAVE ──────────────────
    # The check sits AFTER _resolve_deferred_save on purpose: a save held for
    # evidence that lands in this very turn is applied first, and only then does
    # the call close. Getting this order wrong loses the last answer of every
    # call that ends politely - "yes she is, bye" is an ordinary way to speak.
    _e3, _w3 = _end_sess("Is Dr. Jennifer accepting new patients?"), _EndWS()
    _e3.memory.update(doctor_identity="confirmed", new_patient_status="")
    _e3.turns.insert(0, rw.TranscriptTurn(role="caller", text="Yes, that's right.",
                                          timestamp="00:00:00", audio_rms=0.11))
    _e3._deferred_save = {"name": "save_new_patient_status",
                          "args": {"status": "yes", "heard": "Yes, she is."},
                          "why": "held for evidence", "at": time.monotonic(),
                          "asked_turns": len(_e3.turns)}
    await _land(_e3, _w3, "Yes, she is taking new patients. Okay bye.")
    check(_e3.memory.get("new_patient_status") == "yes",
          "the held save still lands on the turn that ends the call",
          f"{_e3.memory.get('new_patient_status')!r}")
    check(_e3.done is True, "and the call still closes after it")

    # ── the hangup branch is what consumes the flag ────────────────────────
    # ANCHORED ON THE CLOSING SITE, not on the first `if sess.done:` in the
    # file - that one is the barge-in guard, and slicing from it measured the
    # wrong block entirely.
    _cl_at = _PKG_SRC.find("Closing done — waiting")
    _cl = _PKG_SRC[max(0, _cl_at - 1200):_cl_at + 700]
    check(_cl_at > 0 and "done_event.set()" in _cl and "twilio_ws.close()" in _cl,
          "sess.done reaches a real hangup, not just a flag",
          "the branch that drains the audio and closes the socket")
    check('"ended_by_caller": self.ended_by_caller' in _PKG_SRC,
          "and the reason reaches the call artifact",
          "a hangup we were ASKED for must be distinguishable from one we chose")

    print("\n" + "-" * 66)
    print("  Claiming a save that never happened")
    print("-" * 66)
    # call-20260826-1650: the agent said "I'll go with Eastside Clinic" and the
    # artifact records 'eastside' as never transcribed. branch is null - the
    # guard held on the WRITE - but the caller was told a site name they never
    # gave, because guards gate the tool call and never the speech.
    for _c in ("Got it, thanks for clarifying - I'll go with Eastside Clinic.",
               "Okay, thanks for that - let me just note the location you mentioned.",
               "I have noted the branch as Eastside Clinic.",
               "Thanks - let me just capture the referral details clearly.",
               "I'll put you down as Riverside.",
               "I'll mark that as the Eastside site."):
        check(rw._claims_saved(_c),
              f"claim caught: {_c[:46]!r}",
              "a claim that names the FIELD escaped the pronoun-only pattern")

    # THE OLD PATTERN STILL WORKS. Widening must not trade one phrasing family
    # for another - every alternation that was there before is still there.
    for _c in ("I'll note that.", "That's saved.", "all set", "we're done",
               "that's all I needed", "I have everything I need",
               "got it noted", "I'm saving that"):
        check(rw._claims_saved(_c), f"still caught: {_c!r}")

    # AND ORDINARY TURNS ARE UNTOUCHED. A false positive here injects a
    # correction telling the model it lied; on a turn that claimed nothing that
    # is a nudge spent for no reason, and this module bounds those for a reason.
    for _c in ("Which branch does Dr. Okafor work out of?",
               "Thanks for that location - is she taking new patients?",
               "Got it, thanks for confirming - let me check one more detail with you.",
               "Could you give me the exact site name or the street address?",
               "Let me ask you one more thing.",
               "Sorry, the line cut out - could you say that again?",
               "Is a referral required, or does it depend on insurance?"):
        check(not rw._claims_saved(_c),
              f"not a save claim: {_c[:46]!r}",
              "asking for a value is not claiming to have stored one")

    # BOTH CONSUMERS STILL REQUIRE THAT NOTHING WAS SAVED. That is what makes
    # the widening safe: the guard can only fire where the field really is
    # empty, so a false positive costs a nudge that was arguably due anyway.
    check("_claims_saved(_said) and not sess.done" in _PKG_SRC,
          "the grounding consumer still gates on the call not being finished")
    check('_claims_saved(text) and not sess.memory.get("branch")' in _PKG_SRC,
          "and the turns consumer still gates on the branch being unsaved",
          "without that gate a widened pattern would nag on correct closings")

    print("\n" + "-" * 66)
    print("  response.create cannot race an active response")
    print("-" * 66)
    # call-20260826-1422 at 14:24:56: a deferred tool-result create fired as two
    # caller transcripts landed and lost to a response OpenAI's own VAD had
    # already opened — conversation_already_has_active_response. Our guard read
    # _response_active, which is set from the response.created we RECEIVE, so
    # for one round trip a response existed that nothing local could see.
    class _SerWS:
        def __init__(self): self.sent = []
        async def send(self, m): self.sent.append(json.loads(m))

    def _ser_sess():
        _s = rw.RealtimeSession("CA00000000000000000000serial",
                                Doctor(doctor_name="Dr. Jane Okafor"))
        _s._playback_ends_at = 0.0
        return _s

    _ss, _sw = _ser_sess(), _SerWS()
    check(await rw._create_response(_sw, _ss, why="test") is True,
          "an idle session may create a response")
    check(_ss._response_active is True,
          "and is marked active AT THE SEND, not a round trip later",
          "the gap between send and response.created is where the second "
          "create used to slip through")
    _before = len(_sw.sent)
    check(await rw._create_response(_sw, _ss, why="test again") is False,
          "a second create while one is in flight is refused")
    check(len(_sw.sent) == _before,
          "and nothing goes on the wire", f"{_sw.sent}")

    # THE VAD WINDOW. OpenAI opens a response at every speech_stopped because
    # create_response defaults to true, and we hear about it a round trip later.
    _vs, _vw = _ser_sess(), _SerWS()
    _vs._vad_response_due_until = time.monotonic() + 2.0
    check(await rw._create_response(_vw, _vs, why="deferred tool result") is False,
          "no create while OpenAI's VAD is opening one for the turn just ended",
          "this is the exact 14:24:56 collision")
    check(not _vw.sent, "nothing sent into that window", f"{_vw.sent}")

    # THE RECOVERY SITES MUST STILL FIRE IN IT. They exist because the expected
    # response did NOT arrive; refusing them for one that is expected inverts
    # their purpose, and the first cut of this fix silenced the watchdog.
    _rs, _rw_ws = _ser_sess(), _SerWS()
    _rs._vad_response_due_until = time.monotonic() + 2.0
    check(await rw._create_response(_rw_ws, _rs, why="silence watchdog",
                                    allow_when_vad_pending=True) is True,
          "the silence watchdog is exempt — it fires BECAUSE nothing came")
    for _why, _src in [("silence watchdog", rw._silence_watchdog),
                       ("owed substance", rw._silence_watchdog)]:
        check("allow_when_vad_pending=True" in inspect.getsource(_src),
              f"and the {_why} call site actually passes the exemption")
    # Moved with response.done to lifecycle.py on 2026-08-27; the recovery is
    # exempt for the same reason the other two are — it fires BECAUSE nothing
    # came, so a "wait, something may be coming" guard would silence it.
    check("allow_when_vad_pending=True"
          in inspect.getsource(rw._handle_response_done),
          "as does the empty-response re-request, now in _handle_response_done")

    # response.created closes the window, so a normal turn is not throttled.
    _loop_txt = inspect.getsource(rw._oai_to_twilio)
    check("sess._vad_response_due_until = 0.0" in _loop_txt,
          "response.created closes the VAD window",
          "a window only an event can close is a call that never speaks again")
    check("_vad_response_due_until = time.monotonic() + 2.0" in _loop_txt,
          "and speech_stopped opens it")

    print("\n" + "-" * 66)
    print("  call-20260826-1422, the happy path it should have been")
    print("-" * 66)
    # Every field of the live call, with the caller's real words, driven through
    # the real tool handler. The call resolved — after two unnecessary re-asks
    # and 151 seconds. This asserts the same five fields land from the same
    # sentences, so a future change cannot quietly cost one of them.
    _reg_sess = rw.RealtimeSession("CA0000000000000000000regres",
                              Doctor(doctor_name="Dr. Alan Reyes",
                                     hospital_name="Lakeview Medical",
                                     specialization="Oncology"))
    _reg_sess.objective = _PVO
    _reg_sess.agent_name = "David"
    _regw = _SerWS()

    async def _reg_turn(ask, said, tool, **args):
        _reg_sess.add_turn("agent", ask)
        _reg_sess.turns.append(rw.TranscriptTurn(role="caller", text=said,
                                            timestamp="00:00:00", audio_rms=0.12))
        _reg_sess._transcript_at = time.monotonic()
        _reg_sess._placeholder_at = 0.0
        await rw._handle_tool_call(
            {"call_id": "r", "name": tool, "arguments": json.dumps(args)},
            _reg_sess, _regw, {}, False)

    await _reg_turn("Is this Dr. Reyes, Oncology, at Lakeview Medical?",
                "Yes, Dr. Reyes is one of our oncologists.",
                "save_doctor_identity", identity="confirmed",
                heard="Yes, Dr. Reyes is one of our oncologists.",
                detail="Oncology")
    check(_reg_sess.memory.get("doctor_identity") == "confirmed",
          "identity confirmed from the caller's own sentence",
          f"{_reg_sess.memory.get('doctor_identity')!r}")

    await _reg_turn("Do you know which branch Dr. Reyes is working out of?",
                "She is at the West Side Campus 1476 8th Street",
                "save_branch", branch="West Side Campus 1476 8th Street")
    check(_reg_sess.memory.get("branch"),
          "the branch and street number are saved, not called a fabrication",
          f"{_reg_sess.memory.get('branch')!r}")
    check("1476" in str(_reg_sess.memory.get("branch") or ""),
          "and the house number survives verbatim",
          f"{_reg_sess.memory.get('branch')!r}")

    await _reg_turn("Is Dr. Reyes currently taking new patients?",
                "Yes, she's taking on new patients right now.",
                "save_new_patient_status", status="yes",
                heard="Yes, she's taking on new patients right now.")
    check(_reg_sess.memory.get("new_patient_status") == "yes",
          "accepting = yes", f"{_reg_sess.memory.get('new_patient_status')!r}")

    await _reg_turn("Can a new patient get an appointment scheduled right now?",
                "Yes, you can book online or call us directly also.",
                "save_scheduling_status", status="yes",
                heard="Yes, you can book online or call us directly also.")
    check(_reg_sess.memory.get("scheduling_status") == "yes",
          "scheduling = yes", f"{_reg_sess.memory.get('scheduling_status')!r}")

    await _reg_turn("Is a referral needed, always, or only in certain situations?",
                "Only on, it depends upon the situation.",
                "save_referral_requirement", requirement="depends",
                heard="Only on, it depends upon the situation.",
                depends_on="the situation")
    check(_reg_sess.memory.get("referral_status") == "depends",
          "referral = depends", f"{_reg_sess.memory.get('referral_status')!r}")

    _reg_out = _PVO.outcome(_reg_sess.memory)
    check(_reg_out is rw.Outcome.COMPLETE,
          "and the call reports COMPLETE on all five fields",
          f"{_reg_out.name} — collected {sorted(_PVO.collected(_reg_sess.memory))}")
    check(not _reg_sess.deferred_saves and _reg_sess._deferred_save is None,
          "with nothing held — every transcript was already in hand",
          "the deferral must be inert on a call that never races")


    # THE ONE CASE THAT MUST STILL STAND DOWN: a call where nothing has ever
    # rendered. Transcription is broken - a poor line, or a model without it -
    # and refusing would block every save for the rest of the call for a reason
    # the caller had no part in. That is a lost row, the expensive direction.
    _never = _status_sess([])
    check(rw._ungrounded_status({"status": "yes", "heard": "Yes, we are."},
                                _never) == "",
          "no caller speech has EVER transcribed -> the guard still stands down",
          "blocking here loses every row on a line that will not render")
    check(_never.unverified_quotes == [{"field": "status", "value": "yes",
                                        "heard": "Yes, we are."}],
          f"but the quote is marked as never checked "
          f"({_never.unverified_quotes})",
          "selection never ran, so `heard` is the model's own words - storing "
          "it unmarked is how a model-authored quote became the provenance of "
          "a confirmed identity on 1731")

    # SILENCE AND ASKED-BACK ARE DIFFERENT VERDICTS, and the difference is the
    # one place this guard is deliberately stricter than the location one. A
    # branch still has to survive the blob check underneath; a status has no
    # second gate, so "they spoke and none of it answered" must block.
    _spoke_no_answer = _status_sess([("Sorry, what's this regarding?", 0.14)])
    check(rw._ungrounded_status({"status": "yes"}, _spoke_no_answer) != "",
          "they SPOKE since the ask and none of it answered -> blocked",
          "standing down here would let any of the four states be saved at "
          "that moment, with nothing else checking it")
    check("only asked back" in rw._ungrounded_status(
              {"status": "yes"}, _spoke_no_answer),
          "and the reason distinguishes it from silence, in the record")

    # ── NEVER ASKED AT ALL ──────────────────────────────────────────────────
    # With no ask there is no anchor, so every turn from pickup is in scope and
    # reason 1 creeps back in. These pin both halves of that case: the guard
    # still honours an answer volunteered before the question, and it does not
    # accept a bare affirmative that was never about new patients.
    _never_asked = _status_sess([("Yes, speaking.", 0.14)], asked=False)
    check(obj.classify_choice("Yes, speaking.") is obj.ChoiceAnswer.YES,
          "'Yes, speaking.' classifies as YES on its own",
          "which is exactly why the unprompted path cannot take it")
    check(rw._ungrounded_status({"status": "yes"}, _never_asked) != "",
          "never asked + 'Yes, speaking.' at pickup does NOT ground a yes",
          "the single most common opening utterance in this corpus — it is the "
          "phrase the retired hint echoed four times")
    # But a genuinely volunteered answer, given before the question, still
    # grounds. That is what the unanchored path exists for.
    _volunteered = _status_sess(
        [("She's at Mission Bay, and we're not taking new patients right now.", 0.14)],
        asked=False)
    check(rw._ungrounded_status({"status": "no"}, _volunteered) == "",
          "but an answer VOLUNTEERED before the question still grounds",
          "people do answer before being asked, and that turn is about the "
          "thing")
    # And the topical requirement applies only when there was no ask — once we
    # have asked, a bare "Yes." is the normal shape of the answer and grounds.
    check(rw._ungrounded_status({"status": "yes"},
                                _status_sess([("Yes.", 0.14)])) == "",
          "the topical requirement does NOT apply once the question was asked",
          "or the normal shape of a real answer would be refused")

    print("\n" + "-" * 66)
    print("  The call does not end on the first field")
    print("-" * 66)

    async def _pv_tool(sess, name, args):
        await rw._handle_tool_call(
            {"name": name, "call_id": "pv1", "arguments": json.dumps(args)},
            sess, _TcWS(), {}, True)
        return sess

    def _pv_session():
        _s = rw.RealtimeSession("CA0000000000000000000000pvdone",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _PVO
        _s.agent_name = "Alex"
        # IDENTITY FIRST, because the script now gates everything on it — a
        # session that skips it is a call that never established which doctor,
        # and the objective is right to hold everything back.
        _s.memory.update(**{obj.IDENTITY_STATUS_KEY: "confirmed"})
        _s.add_turn("agent", "Which branch does Dr. Okafor work out of?")
        _s.turns.append(rw.TranscriptTurn(
            role="caller", text="She's at the Mission Bay Clinic.",
            timestamp="00:00:00", audio_rms=0.14))
        return _s

    # THE BUG THIS WOULD HAVE BEEN. `sess.done` was set by name — a successful
    # save_branch ended the call by definition — which on this template would
    # hang up before the second question was ever asked.
    _pv_s = await _pv_tool(_pv_session(), "save_branch",
                           {"branch": "Mission Bay Clinic"})
    check(_pv_s.memory.get("branch") == "Mission Bay Clinic",
          "the branch saves on template 3")
    check(not _pv_s.done,
          "and the call is NOT over — there is a second question to ask",
          "setting done by tool name would hang up on the caller mid-script")
    check(_PVO.outcome(_pv_s.memory) is obj.Outcome.PARTIAL,
          "the call reads as PARTIAL at this point, not as finished")
    # Now the second field lands and the objective IS met.
    _pv_s.add_turn("agent", "Is she accepting new patients?")
    _pv_s.turns.append(rw.TranscriptTurn(
        role="caller", text="Yes, we are taking new patients.",
        timestamp="00:00:01", audio_rms=0.14))
    _pv_s = await _pv_tool(_pv_s, "save_new_patient_status",
                           {"status": "yes", "heard": "Yes, we are taking new patients."})
    check(_pv_s.memory.get(obj.NEW_PATIENT_STATUS_KEY) == "yes",
          "the status saves")
    # STILL NOT OVER. Answering "yes" is what OPENS questions 3 and 4, so a
    # two-field call cannot be complete here — this is the conditional gate
    # doing its job from the other direction.
    check(not _pv_s.done and _PVO.outcome(_pv_s.memory) is obj.Outcome.PARTIAL,
          "a YES answer opens two more questions rather than ending the call",
          f"outcome={_pv_s.memory.get('outcome')!r} done={_pv_s.done}")
    _pv_s.add_turn("agent", "Can a new patient get an appointment scheduled?")
    _pv_s.turns.append(rw.TranscriptTurn(
        role="caller", text="Yes, we can book them in next week.",
        timestamp="00:00:02", audio_rms=0.14))
    _pv_s = await _pv_tool(_pv_s, "save_scheduling_status",
                           {"status": "yes", "heard": "we can book them in next week"})
    check(not _pv_s.done, "still not over after the third field")
    _pv_s.add_turn("agent", "Is a referral needed?")
    _pv_s.turns.append(rw.TranscriptTurn(
        role="caller", text="It depends on their insurance.",
        timestamp="00:00:03", audio_rms=0.14))
    _pv_s = await _pv_tool(_pv_s, "save_referral_requirement",
                           {"requirement": "depends",
                            "heard": "It depends on their insurance.",
                            "depends_on": "their insurance"})
    check(_PVO.outcome(_pv_s.memory) is obj.Outcome.COMPLETE
          and _pv_s.memory.get("resolved") is True,
          "the objective is COMPLETE and the call resolved",
          f"outcome={_pv_s.memory.get('outcome')!r}")
    check(_pv_s.done, "and NOW the call is over — after the FOURTH field")
    # The other path: a NO answer completes the call two questions early, and
    # that is not a shortfall.
    _pv_no = await _pv_tool(_pv_session(), "save_branch",
                            {"branch": "Mission Bay Clinic"})
    _pv_no.add_turn("agent", "Is she accepting new patients?")
    _pv_no.turns.append(rw.TranscriptTurn(
        role="caller", text="No, she's not taking new patients.",
        timestamp="00:00:02", audio_rms=0.14))
    _pv_no = await _pv_tool(_pv_no, "save_new_patient_status",
                            {"status": "no", "heard": "No, she's not taking new patients."})
    check(_PVO.outcome(_pv_no.memory) is obj.Outcome.COMPLETE and _pv_no.done,
          "a NO call is COMPLETE and over after two fields, not PARTIAL",
          f"outcome={_pv_no.memory.get('outcome')!r} done={_pv_no.done}")
    check(_pv_no.memory.get("resolved") is True,
          "and it reports resolved — the receptionist answered everything asked")
    # The one-field template is unchanged: save_branch still ends that call.
    _b1_s = rw.RealtimeSession("CA00000000000000000000branch1",
                               Doctor(doctor_name="Dr. Jane Okafor",
                                      hospital_name="Northside Medical Group"))
    _b1_s.objective = _templates.FORAGE_DATA_COLLECTION.objective
    _b1_s.turns.append(rw.TranscriptTurn(
        role="caller", text="She's at the Mission Bay Clinic.",
        timestamp="00:00:00", audio_rms=0.14))
    _b1_s = await _pv_tool(_b1_s, "save_branch", {"branch": "Mission Bay Clinic"})
    check(_b1_s.done,
          "on the branch-only template save_branch still ends the call",
          "the change is that the OBJECTIVE decides, not that it never ends")

    print("\n" + "-" * 66)
    print("  The ask budget reaches the second field")
    print("-" * 66)
    # The counters were already objective-agnostic. The GATE feeding them was
    # not: nothing reached the budget except through _is_location_ask, so on
    # this template every ask about new patients was invisible to it.
    _acc_ask = "Is Dr. Okafor accepting new patients?"
    check(not rw._is_location_ask(_acc_ask),
          "an accepting-status ask names no location, so the old gate missed it",
          "which is why it needed generalising rather than confirming")
    _pv_gate = rw.RealtimeSession("CA00000000000000000000pvgate",
                                  Doctor(doctor_name="Dr. Jane Okafor",
                                         hospital_name="Northside"))
    _pv_gate.objective = _PVO
    check(rw._is_objective_ask(_acc_ask, _pv_gate),
          "the objective-aware gate sees it on template 3")
    _b_gate = rw.RealtimeSession("CA000000000000000000000bgate",
                                 Doctor(doctor_name="Dr. Jane Okafor",
                                        hospital_name="Northside"))
    _b_gate.objective = _templates.FORAGE_DATA_COLLECTION.objective
    check(not rw._is_objective_ask(_acc_ask, _b_gate),
          "and does NOT see it on a template that does not collect it",
          "a template's budget counts asks for ITS fields, not for every field "
          "any template has")
    check(rw._is_objective_ask("Which branch is she at?", _b_gate),
          "while the location ask still counts everywhere it did before")

    print("\n" + "=" * 66)
    print("  CONDITIONALLY REQUIRED — a 'no' call is COMPLETE, not PARTIAL")
    print("=" * 66)
    # call-20260824-1604 hung up after two fields because the objective said
    # COMPLETE while the prompt was still walking a four-question script. The
    # objective won, as it should — so the objective has to describe the script.
    # The hard part is that questions 3 and 4 only exist when the answer to 2
    # was yes: required=True would leave a correct "not accepting" call
    # permanently PARTIAL, blaming a receptionist who answered everything asked.
    _PVO4 = _templates.PROVIDER_VERIFICATION_OBJECTIVE
    check([f.name for f in _PVO4.fields]
          == ["identity", "branch", "accepting", "scheduling", "referral"],
          "all five fields are declared, in the order the script asks them",
          f"{[f.name for f in _PVO4.fields]}")
    check(obj.unwritable_fields(_PVO4) == (),
          "every one of the four is written by a tool",
          "a declared field nothing writes is PARTIAL for ever, silently")
    check(obj.invalid_conditions(_PVO4) == (),
          "and every conditional gate is structurally sound",
          f"{obj.invalid_conditions(_PVO4)}")

    _CONF = {obj.IDENTITY_STATUS_KEY: "confirmed"}
    _pv_paths = [
        ("nothing yet",  {**_CONF},
         obj.Outcome.PARTIAL,  ("branch", "accepting"), ("scheduling", "referral")),
        ("branch only",  {**_CONF, "branch": "Riverside Campus"},
         obj.Outcome.PARTIAL,  ("accepting",),          ("scheduling", "referral")),
        # THE CASE THAT DECIDED THE DESIGN. A front desk that says "no, we're
        # not taking anyone" has answered the call completely.
        ("accepting=no", {**_CONF, "branch": "Riverside Campus",
                          obj.NEW_PATIENT_STATUS_KEY: "no"},
         obj.Outcome.COMPLETE, (),                      ("scheduling", "referral")),
        ("waitlist",     {**_CONF, "branch": "Riverside Campus",
                          obj.NEW_PATIENT_STATUS_KEY: "waitlist"},
         obj.Outcome.COMPLETE, (),                      ("scheduling", "referral")),
        ("unsure",       {**_CONF, "branch": "Riverside Campus",
                          obj.NEW_PATIENT_STATUS_KEY: "unsure"},
         obj.Outcome.COMPLETE, (),                      ("scheduling", "referral")),
        # And when it IS yes, the two extra questions become required.
        ("accepting=yes", {**_CONF, "branch": "Riverside Campus",
                           obj.NEW_PATIENT_STATUS_KEY: "yes"},
         obj.Outcome.PARTIAL,  ("scheduling", "referral"), ()),
        ("yes + sched",  {**_CONF, "branch": "Riverside Campus",
                          obj.NEW_PATIENT_STATUS_KEY: "yes",
                          obj.SCHEDULING_STATUS_KEY: "yes"},
         obj.Outcome.PARTIAL,  ("referral",),           ()),
        ("all four",     {**_CONF, "branch": "Riverside Campus",
                          obj.NEW_PATIENT_STATUS_KEY: "yes",
                          obj.SCHEDULING_STATUS_KEY: "yes",
                          obj.REFERRAL_STATUS_KEY: "depends"},
         obj.Outcome.COMPLETE, (),                      ()),
    ]
    for _label, _mem, _want_out, _want_missing, _want_na in _pv_paths:
        _m = _fake(_mem)
        check(_PVO4.outcome(_m) is _want_out,
              f"{_label:14} -> {_want_out.label}", _PVO4.outcome(_m).label)
        check(_PVO4.missing(_m) == _want_missing,
              f"{_label:14}    missing={list(_want_missing)}",
              f"{_PVO4.missing(_m)}")
        check(_PVO4.not_applicable(_m) == _want_na,
              f"{_label:14}    n/a={list(_want_na)}",
              f"{_PVO4.not_applicable(_m)}")
    # "Never applied" and "asked and got nothing" are different facts, and the
    # spoken directive must not confuse them: on a NO call the agent must not
    # announce it failed to get a referral rule for a question never asked.
    _no_call = _fake({obj.IDENTITY_STATUS_KEY: "confirmed",
                      "branch": "Riverside Campus",
                      obj.NEW_PATIENT_STATUS_KEY: "no"})
    check(_PVO4.missing_spoken(_no_call) == "",
          "a completed NO call has nothing to apologise for out loud",
          f"{_PVO4.missing_spoken(_no_call)!r}")
    _yes_call = _fake({obj.IDENTITY_STATUS_KEY: "confirmed",
                       "branch": "Riverside Campus",
                       obj.NEW_PATIENT_STATUS_KEY: "yes"})
    check("referral" in _PVO4.missing_spoken(_yes_call),
          "but a YES call that stopped early names what it still owes",
          f"{_PVO4.missing_spoken(_yes_call)!r}")

    # THE CHECK THAT PAYS FOR THE DECLARATIVE SPELLING. Each of these fails in
    # the COMPLETE-too-early direction, which is the one nobody notices.
    _mk = lambda **kw: obj.Field(name=kw.pop("name"), memory_key="note_x",
                                 kind=obj.AnswerKind.CHOICE,
                                 probe=obj.ACCEPTING_ASK,
                                 states=obj.CHOICE_STATES, **kw)
    _gate_ok = obj.Field(name="accepting", memory_key=obj.NEW_PATIENT_STATUS_KEY,
                         kind=obj.AnswerKind.CHOICE, probe=obj.ACCEPTING_ASK,
                         states=obj.CHOICE_STATES, required=True)
    for _label, _flds, _want in [
        ("gate names a field that does not exist",
         (_gate_ok, _mk(name="x", required_when=obj.RequiredWhen("nope", frozenset({"yes"})))),
         True),
        ("gate on a value the gate field cannot hold",
         (_gate_ok, _mk(name="x", required_when=obj.RequiredWhen("accepting", frozenset({"ye"})))),
         True),
        ("gate on a field that is not itself required",
         (obj.Field(name="accepting", memory_key=obj.NEW_PATIENT_STATUS_KEY,
                    kind=obj.AnswerKind.CHOICE, probe=obj.ACCEPTING_ASK,
                    states=obj.CHOICE_STATES, required=False),
          _mk(name="x", required_when=obj.RequiredWhen("accepting", frozenset({"yes"})))),
         True),
        ("empty gate — never required",
         (_gate_ok, _mk(name="x", required_when=obj.RequiredWhen("accepting", frozenset()))),
         True),
        ("a sound gate",
         (_gate_ok, _mk(name="x", required_when=obj.RequiredWhen("accepting", frozenset({"yes"})))),
         False),
    ]:
        _bad = bool(obj.invalid_conditions(obj.CallObjective(fields=_flds)))
        check(_bad is _want, f"invalid_conditions catches: {_label}",
              f"{obj.invalid_conditions(obj.CallObjective(fields=_flds))}")
    # A gate that cannot be resolved must NOT quietly complete the call.
    _broken = obj.CallObjective(fields=(
        _gate_ok, _mk(name="x", required_when=obj.RequiredWhen("nope", frozenset({"yes"})))))
    check(_broken.outcome(_fake({obj.NEW_PATIENT_STATUS_KEY: "yes"}))
          is obj.Outcome.PARTIAL,
          "and a broken gate leaves the call PARTIAL rather than COMPLETE",
          "the failure has to be visible, not silently done")

    print("\n" + "-" * 66)
    print("  The two new fields: tools, states, grounding")
    print("-" * 66)
    for _tool, _val, _key, _want in [
        ("save_scheduling_status", "waitlist", obj.SCHEDULING_STATUS_KEY, "waitlist"),
        ("save_scheduling_status", "not until January", obj.SCHEDULING_STATUS_KEY, None),
        ("save_referral_requirement", "always", obj.REFERRAL_STATUS_KEY, "always"),
        ("save_referral_requirement", "only for some insurers",
         obj.REFERRAL_STATUS_KEY, "depends"),
        ("save_referral_requirement", "no referral needed",
         obj.REFERRAL_STATUS_KEY, "no"),
        ("save_referral_requirement", "purple", obj.REFERRAL_STATUS_KEY, None),
    ]:
        _m = rw.CallMemory(f"t4-{_tool}-{_val[:12]}")
        _m.clear()
        _r = _tools.TOOL_IMPLS[_tool](_m, _val, heard=_val)
        if _want is None:
            check(not _r.get("ok"), f"{_tool}({_val!r}) rejected",
                  str(_r.get("error"))[:52])
        else:
            check(_r.get("ok") and _m.get(_key) == _want,
                  f"{_tool}({_val!r}) -> {_want}", f"{_m.get(_key)!r}")
    # The referral vocabulary is its OWN, not the accepting one relabelled.
    check(obj.classify_referral("only for some plans") is obj.ReferralAnswer.DEPENDS,
          "'depends' is a first-class referral state",
          "the conditionality IS the answer the client acts on")
    check(obj.REFERRAL_STATES != obj.CHOICE_STATES,
          "and referral does not share the accepting field's states",
          f"{sorted(obj.REFERRAL_STATES)}")

    # -- "It's depend upon situation." - call-20260827-1130 -----------------
    # A correct DEPENDS answer the classifier could not read, so the deferred
    # save was REFUSED on the transcript that bears it out and the call ended
    # PARTIAL with referral missing. Neither half was absent on its own: the
    # bare stem is covered when followed by "on", and "upon" is covered when
    # the stem carries its "s". Only the conjunction fell through - and that is
    # ordinary Indian English, which is what the callees on this project speak.
    for _t in ["It's depend upon situation.", "It's depend on situation.",
               "It depends upon the situation.", "Depending upon the plan.",
               "It would depend upon their insurance."]:
        check(obj.classify_referral(_t) is obj.ReferralAnswer.DEPENDS,
              f"a DEPENDS answer is read as one: {_t!r}",
              f"got {obj.classify_referral(_t)!r}")
    # THE PREPOSITION IS REQUIRED, and this is the mutation that says why.
    # Widening the stem to `depend\w*` reaches inside "dependent", and DEPENDS
    # is tested BEFORE ALWAYS - so the loose version turns a correct
    # classification into a wrong one on a sentence that says the opposite.
    check(obj.classify_referral("It is not dependent on anything, always "
                                "required.") is obj.ReferralAnswer.ALWAYS,
          "and a negated 'dependent' is still ALWAYS, not DEPENDS",
          "the loose stem would claim this sentence for the wrong state")
    check(obj.classify_referral("No referral needed.") is obj.ReferralAnswer.NO
          and obj.classify_referral("Yes, always required.")
              is obj.ReferralAnswer.ALWAYS,
          "with the other three states untouched")
    _ref_field = _PVO4.field_named("referral")
    check(_ref_field is not None and _ref_field.present(
              _fake({obj.REFERRAL_STATUS_KEY: "waitlist"})) is False,
          "so an accepting-field state does not satisfy the referral field",
          "each field validates against its OWN vocabulary")

    # Grounding, anchored to each field's own ask.
    def _ground_sess(agent_ask, caller, rms=0.14):
        _s = rw.RealtimeSession("CA00000000000000000000ground4",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _PVO4
        _s.agent_name = "Alex"
        _s.add_turn("agent", agent_ask)
        _s.turns.append(rw.TranscriptTurn(role="caller", text=caller,
                                          timestamp="00:00:00", audio_rms=rms))
        return _s

    check(rw._ungrounded_scheduling(
              {"status": "yes"},
              _ground_sess("Can a new patient get an appointment scheduled?",
                     "Yes, we can book them in next week.")) == "",
          "a real scheduling answer grounds")
    check(rw._ungrounded_scheduling(
              {"status": "yes"},
              _ground_sess("Can a new patient get an appointment scheduled?",
                     "Yes.", rms=0.002)) != "",
          "a bare scheduling 'Yes.' on silent audio does not",
          "bare is the normal shape, so the audio carries the whole load")
    check(rw._ungrounded_referral(
              {"requirement": "depends"},
              _ground_sess("Is a referral needed?",
                     "It depends on their insurance.")) == "",
          "a real referral answer grounds")
    check(rw._ungrounded_referral(
              {"requirement": "always"},
              _ground_sess("Is a referral needed?",
                     "It depends on their insurance.")) != "",
          "and claiming ALWAYS when they said DEPENDS is refused",
          "the classified state has to match the one being saved")
    # Each guard is anchored to its OWN ask — a scheduling answer must not
    # ground a referral claim just because both came after some question.
    check(rw._ungrounded_referral(
              {"requirement": "no"},
              _ground_sess("Can a new patient get an appointment scheduled?",
                     "No, not at the moment.")) != "",
          "an answer to the SCHEDULING question does not ground a REFERRAL claim",
          "each field anchors on its own probe, not on 'the last thing asked'")

    print("\n" + "-" * 66)
    print("  A template must not promise a question it cannot ask")
    print("-" * 66)
    # call-20260824-1604 said "let me quickly pin down what that means for
    # scheduling" and hung up, because the prompt walked a four-question script
    # while the objective declared two. Prompt and objective disagreed and the
    # objective won — silently, mid-promise.
    #
    # So: if a template's instructions raise a topic, the objective must have a
    # field for it. Checked against the field probes themselves, which are the
    # same patterns the ask budget and the grounding guards use, so there is one
    # definition of "asking about scheduling" and not three.
    _TOPIC_PROBES = {
        "location":   obj.LOCATION_NOUN,
        "accepting":  obj.ACCEPTING_ASK,
        "scheduling": obj.SCHEDULING_ASK,
        "referral":   obj.REFERRAL_ASK,
    }
    for _name, _tpl in _templates.TEMPLATES.items():
        _declared = {f.probe for f in _tpl.objective.fields}
        for _topic, _probe in _TOPIC_PROBES.items():
            _raised = bool(_probe.search(_tpl.instructions))
            _has_field = _probe in _declared
            check(not (_raised and not _has_field),
                  f"{_name}: raises {_topic!r} only if it declares a field for it",
                  "the prompt promised a question the objective cannot end on — "
                  "exactly what hung up call-20260824-1604")
    # And the reverse: a declared field the prompt never asks about would be
    # collected by luck or not at all.
    for _name, _tpl in _templates.TEMPLATES.items():
        for _f in _tpl.objective.fields:
            check(bool(_f.probe.search(_tpl.instructions)),
                  f"{_name}: actually asks for the {_f.name!r} it declares")
    # Every save tool a template's prompt names must exist, or the model is
    # being told to call something that will come back 'unknown tool'.
    import re as _re4
    for _name, _tpl in _templates.TEMPLATES.items():
        for _mentioned in set(_re4.findall(r"\bsave_[a-z_]+", _tpl.instructions)):
            check(_mentioned in _tools.TOOL_IMPLS,
                  f"{_name}: names a real tool ({_mentioned})",
                  f"not in {sorted(_tools.TOOL_IMPLS)}")

    print("\n" + "-" * 66)
    print("  back_to_back_asks counted a healthy call")
    print("-" * 66)
    # call-20260824-1604 scored 1 on a flawless exchange. The loop skipped past
    # caller turns, so prev_agent_asked carried across the answer and any two
    # agent turns that both asked something counted. Tolerable on a one-question
    # script; on a four-question script every good call trips it, and a metric
    # that fires on the good case is the one people stop reading.
    _healthy = [
        rw.TranscriptTurn(role="agent", timestamp="1",
                          text="Do you know which branch Dr. Okafor works out of?"),
        rw.TranscriptTurn(role="caller", timestamp="2",
                          text="Yeah, she's at the Riverside campus."),
        rw.TranscriptTurn(role="agent", timestamp="3",
                          text="Got it — is Dr. Okafor currently taking new patients?"),
    ]
    check(rw.conversation_metrics(_healthy)["back_to_back_asks"] == 0,
          "two scripted questions with an answer between them count 0",
          f"{rw.conversation_metrics(_healthy)['back_to_back_asks']}")
    _into_silence = [
        rw.TranscriptTurn(role="agent", timestamp="1",
                          text="Which branch does she work out of?"),
        rw.TranscriptTurn(role="caller", timestamp="2", text="Hello?"),
        rw.TranscriptTurn(role="agent", timestamp="3",
                          text="Which branch is Dr. Okafor at?"),
    ]
    check(rw.conversation_metrics(_into_silence)["back_to_back_asks"] == 1,
          "but asking again into filler still counts 1",
          "that is the defect the metric is for, and it survives the fix")
    _no_reply = [
        rw.TranscriptTurn(role="agent", timestamp="1", text="Which branch is she at?"),
        rw.TranscriptTurn(role="agent", timestamp="2", text="Which campus is that?"),
    ]
    check(rw.conversation_metrics(_no_reply)["back_to_back_asks"] == 1,
          "and so does asking twice with nothing at all in between")

    print("\n" + "=" * 66)
    print("  A TOKEN NOBODY SAID, RIDING ALONG ON ONE THAT WAS")
    print("=" * 66)
    # call-20260824-2014. The transcriber rendered "Riverside campus" as "She
    # resides at campus", so grounding twice refused a value whose only content
    # word was 'Riverside' — correctly, on the evidence it had. The model then
    # offered "Riverside Campus, 1825 4th Street" and that was ACCEPTED, on the
    # street number, which the caller really had said.
    #
    # The accept is right and stays. What was wrong is the stamp: the row went
    # to the directory as "verified against caller transcript" while the one
    # distinctive word in it had never been transcribed at all. Same hole the
    # digit rule closed for numbers ("because 'bay' appeared, and one word was
    # enough"), alphabetic half.
    def _ride_sess(turns):
        _s = rw.RealtimeSession("CA00000000000000000000ridalg",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        for _t in turns:
            _s.turns.append(rw.TranscriptTurn(role="caller", text=_t,
                                              timestamp="00:00:00", audio_rms=0.06))
        return _s

    _rs = _ride_sess(["She resides at campus",
                      "He is just a very busy campus.",
                      "Oh I think it is 1825 4th street."])
    _rargs = {"branch": "Riverside Campus, 1825 4th Street"}
    check(rw._ungrounded_terms(_rargs, _rs) == "",
          "the value is still ACCEPTED — the caller did say the address",
          "blocking it would discard a correct answer, the expensive direction")
    check(rw._rode_along(_rargs, _rs) == ["riverside"],
          "but the word nobody was transcribed saying is named",
          f"{rw._rode_along(_rargs, _rs)}")
    # A fully corroborated value reports nothing, or the signal is noise.
    check(rw._rode_along({"branch": "1825 4th Street"}, _rs) == [],
          "a fully grounded value rides along on nothing")
    check(rw._rode_along({"branch": "Riverside Campus"},
                         _ride_sess(["She resides at the Riverside campus."])) == [],
          "and a value the caller DID say reports nothing either")
    # Generic nouns are not the signal — they are stopwords for grounding and
    # must not be reported as unverified.
    check("campus" not in rw._rode_along(_rargs, _rs),
          "generic place nouns are not reported — they are not evidence either way")
    # The stamp itself has to carry it, because the stamp is what a reviewer
    # reads months later.
    _stamp_sess = _ride_sess(["She resides at campus",
                              "Oh I think it is 1825 4th street."])
    await rw._handle_tool_call(
        {"name": "save_branch", "call_id": "r1",
         "arguments": json.dumps(_rargs)}, _stamp_sess, _TcWS(), {}, True)
    _g = str(_stamp_sess.memory.get("grounding") or "")
    check(_stamp_sess.memory.get("branch") == _rargs["branch"],
          "the branch still saves")
    check("riverside" in _g.lower() and "EXCEPT" in _g,
          "and the grounding stamp no longer claims the whole value was verified",
          _g[:100])
    check(_stamp_sess.memory.get("rode_along") == ["riverside"],
          "with the tokens recorded as data, not only as prose",
          "so 'which rows contain a word nobody said' is answerable by query")

    print("\n" + "=" * 66)
    print("  A CLEAN YES, REFUSED THREE TIMES — and the accept that was worse")
    print("=" * 66)
    # call-20260824-2014. The caller said "Ah, yes, she's taking the new
    # patients." — clean transcript, no ambiguity — and the status guard said
    # "nothing the caller said reads as that answer", three times. Two separate
    # rigidities, both the same shape as _is_filler_reply judging "Yes." on its
    # words alone.
    for _txt, _want, _why in [
        ("Ah, yes, she's taking the new patients.", "yes",
         "the live failure: ^\\W* could not cross the 'Ah', and the phrase "
         "form had no room for 'the'"),
        ("Oh yeah, she is.", "yes", "same lead-in, different interjection"),
        ("Well, yes.", "yes", ""),
        ("Um, yes, taking on new patients.", "yes", ""),
        ("She's taking any new patients she can get.", "yes",
         "three words of slack inside the phrase"),
        # THE ONE THAT WAS WORSE, and was live: a practice REFUSING new
        # patients classified as accepting them, because `not taking` demanded
        # adjacency and 'currently' broke it.
        ("She's not currently taking new patients.", "no",
         "was YES — a refusal recorded as an acceptance"),
        ("He's not seeing new patients at the moment.", "no",
         "was None — missed entirely"),
        ("She isn't currently accepting new patients.", "no", ""),
        ("We don't take new patients any more.", "no", ""),
        # And nothing that already worked may move.
        ("Yes, she is.", "yes", ""), ("Yes, we are.", "yes", ""),
        ("No.", "no", ""), ("Nope.", "no", ""),
        ("We're full, but I can put you on the waitlist.", "waitlist", ""),
        ("You'd be number twenty-one in the queue.", "waitlist", ""),
        ("I'm not sure, you'd have to ask.", "unsure", ""),
        ("It depends on the insurance.", "unsure", ""),
        ("She is at the Riverside campus.", None, "not an answer to this ask"),
        ("Okay.", None, ""), ("and", None, ""),
    ]:
        _got = obj.classify_choice(_txt)
        _gv = _got.value if _got else None
        check(_gv == _want, f"choice={_want!s:8} {_txt[:44]!r}", _why)
    # WHERE THE NEGATION GUARD ACTUALLY EARNS ITS PLACE. The widened NO pattern
    # handles "not currently taking" on its own, so these are the cases that
    # reach the YES pattern and have to be flipped by _negated_before: a
    # negator that is nowhere near an accepting-verb, in front of a bare
    # "we are". Without them the guard is untested and its mutation passes.
    for _txt, _want in [("I don't think we are.", "no"),
                        ("I wouldn't say we are.", "no")]:
        _g = obj.classify_choice(_txt)
        check((_g.value if _g else None) == _want,
              f"negation reaches the bare affirmative: {_txt!r}",
              "the NO pattern does not match this — only _negated_before does")
    # Negation is CLAUSE-scoped, not whole-string: the affirmative has to be
    # inside the negated clause to be flipped.
    check(obj._negated_before("She's not currently taking new patients.",
                              len("She's not currently ")) is True,
          "a negator earlier in the same clause flips the affirmative")
    check(obj._negated_before("Ah, yes, she's taking the new patients.", 4) is False,
          "and an interjection before it does not")

    print("\n" + "-" * 66)
    print("  The agent must not talk its own claim into the record")
    print("-" * 66)
    # THE WORSE HALF of the same call. After three refusals the status DID
    # save — not because anything was verified, but because the agent said
    # "I heard you say she's taking the new patients", that matched
    # ACCEPTING_ASK, the anchor moved past every caller turn that had answered,
    # the evidence window emptied, and the guard took its own "no evidence
    # since the ask" branch and stood down.
    check(obj.ACCEPTING_ASK.search("I heard you say she's taking the new patients."),
          "the restatement does match the TOPIC probe",
          "which is why a bare probe match was the wrong anchor")
    check(rw._is_ask_for("I heard you say she's taking the new patients.",
                         obj.ACCEPTING_ASK) is False,
          "but it is NOT an ask, so it no longer moves the anchor",
          "the model cannot move the goalposts by talking — same principle as "
          "_ungrounded_terms excluding the agent's words from `heard`")
    check(rw._is_ask_for("Is Dr. Okafor taking new patients right now?",
                         obj.ACCEPTING_ASK) is True,
          "while the real question still does")
    check(rw._is_ask_for("Just to confirm, she is taking new patients?",
                         obj.ACCEPTING_ASK) is True,
          "and so does a confirmation QUESTION — a yes after it is a real answer")
    # The read-back list must not swallow the agent's commonest ask phrasing.
    check(rw._is_location_ask("I'm trying to confirm which branch Dr. Okafor "
                              "works out of.") is True,
          "'trying to confirm which branch' is an ASK, not a read-back",
          "a bare 'to confirm' in the read-back list would stop the budget "
          "counting the phrasing the agent uses most")

    # ── PROMISING TO ASK IS NOT ASKING — call-20260827-1010 ────────────────
    # The same hole as the read-back above, entered in the future tense. The
    # agent said this, asked nobody anything, and it scored as an ask for the
    # `accepting` field: it stamped _field_ask_at, so the FIRST real
    # new-patient question forty seconds later came out as a RE-ASK; it spent
    # a slot of the ask budget through _is_objective_ask; and it is the anchor
    # _ungrounded_status measures its evidence window from.
    _promise = ("Thanks for that - I'm just noting Riverside Campus now, "
                "then I'll ask about new patients.")
    check(obj.ACCEPTING_ASK.search(_promise),
          "the promise does match the TOPIC probe",
          "which is why a bare probe match was never enough")
    check(rw._is_ask_for(_promise, obj.ACCEPTING_ASK) is False,
          "but a promise to ask LATER is not an ask now")
    # The other half of the same sentence: "I'm just noting X" is the read-back
    # list in the present progressive, and X carries the location noun.
    check(rw._is_location_ask(_promise) is False,
          "and filing a value as you say it is a read-back, not a branch ask")
    _PV_OBJ = get_template("provider_verification").objective
    for _t in ["I'll ask about new patients in a moment.",
               "Next I'll ask whether she's accepting new patients.",
               "Then I'll ask about the branch.",
               "After that I'll ask which office she works out of."]:
        check(rw._is_objective_ask(_t, double(objective=_PV_OBJ)) is False,
              f"a deferred promise asks nothing: {_t!r}")
    # THE NARROWNESS IS THE POINT, and these are the mutations that matter.
    # Over-exempting loses an ask, and a lost ask lets the agent pester people.
    check(rw._is_ask_for("Is Dr. Okafor taking new patients right now?",
                         obj.ACCEPTING_ASK) is True,
          "the real question is untouched")
    check(rw._is_ask_for("I'll ask about new patients.", obj.ACCEPTING_ASK)
          is True,
          "a bare 'I'll ask' with no deferral marker still counts as an ask",
          "a receptionist would simply answer it; the marker is required")
    check(rw._is_ask_for("Then I'll ask about new patients. Which branch is "
                         "she at", obj.ACCEPTING_ASK) is False
          and rw._is_location_ask("Then I'll ask about new patients. Which "
                                  "branch is she at") is True,
          "the promise is consumed to its own sentence, so a real ask beside "
          "it survives")
    check(rw._is_ask_for("Then I'll ask about new patients - are you taking "
                         "any?", obj.ACCEPTING_ASK) is True,
          "and a question mark in the promise's own sentence makes it an ask")

    # End to end, on the real turn sequence from the call.
    def _npseq(n):
        _s = rw.RealtimeSession("CA00000000000000000000npseq",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
        for _role, _text in [
            ("agent", "Thanks for the location — is Dr. Okafor taking new patients right now?"),
            ("caller", "Ah, yes, she's taking the new patients."),
            ("agent", "Alright, thanks for confirming that."),
            ("caller", "Okay."),
            ("agent", "I heard you say she's taking the new patients."),
        ][:n]:
            _s.turns.append(rw.TranscriptTurn(
                role=_role, text=_text, timestamp="00:00:00",
                audio_rms=0.06 if _role == "caller" else None))
        return _s

    check(rw._ungrounded_status({"status": "yes"}, _npseq(2)) == "",
          "the answer grounds on the turn it was actually given",
          "it took three refusals and a lucky stand-down on the live call")
    # And after the agent's restatement the evidence is STILL the caller's turn,
    # not an empty window.
    _after = _npseq(5)
    check(rw._ungrounded_status({"status": "yes"}, _after) == "",
          "and still grounds after the agent restates it")
    check(rw._ungrounded_status({"status": "no"}, _after) != "",
          "while a status the caller never gave is still refused",
          "the window did not empty, so the guard can still judge")

    print("\n" + "=" * 66)
    print("  A SPACE IS NOT A DIFFERENT ANSWER")
    print("=" * 66)
    # call-20260824-2113: caller said "east side clinic", model saved "Eastside
    # Clinic". `clinic` is a grounding stopword, so `eastside` was the only
    # content word left, and it is not a SUBSTRING of "east side". Rejected four
    # times — twice while the caller repeated themselves verbatim — and the call
    # recorded "could not obtain the location" about someone who answered
    # immediately, repeated it on request, and confirmed it.
    def _east(turns):
        _s = rw.RealtimeSession("CA0000000000000000000east2",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        for _t in turns:
            _s.turns.append(rw.TranscriptTurn(role="caller", text=_t,
                                              timestamp="00:00:00", audio_rms=0.08))
        return _s

    _es = _east(["He works at the east side clinic.",
                 "Yeah, he works at the east side clinic. Yeah, that's it."])
    for _v in ("Eastside Clinic", "eastside clinic", "East Side Clinic",
               "East-Side Clinic"):
        check(rw._ungrounded_terms({"branch": _v}, _es) == "",
              f"grounds however the spaces fall: {_v!r}")
        check(rw._rode_along({"branch": _v}, _es) == [],
              f"  and nothing is reported as riding along on {_v!r}",
              "the word DID ground; reporting it would be crying wolf")
    # The normalisations this covers are the common shape of US branch names.
    for _said, _saved in [("the north side office", "Northside"),
                          ("mid town clinic", "Midtown"),
                          ("saint marys hospital", "Saint Marys"),
                          ("the west-side annex", "Westside")]:
        check(rw._ungrounded_terms({"branch": _saved}, _east([_said])) == "",
              f"{_saved!r} grounds on {_said!r}")

    # IT IS NOT FUZZY MATCHING, and this is the assertion that says so. Every
    # letter must still appear in the same order, which is why it cannot do what
    # a similarity threshold would have done.
    check(rw._ungrounded_terms({"branch": "Riverside Campus"},
                               _east(["She resides at campus",
                                      "He is just a very busy campus."])) != "",
          "'Riverside' still does NOT ground on 'resides at'",
          "the letters differ, not just the spaces — the case a threshold "
          "could not separate is untouched")
    check(rw._ungrounded_terms({"branch": "Riverside Clinic"},
                               _east(["Hello. Okay, next slide, please."])) != "",
          "and the documented fabrication is still refused")
    check(rw._ungrounded_terms(
              {"branch": "Mission Bay Clinic, 1855 Fourth Street"},
              _east(["it's 1825 4th street"])) != "",
          "the invented house number is still refused",
          "digits keep their own exact comparison — the collapse does not "
          "route through the digit rule")
    check(rw._collapse("East-Side, Clinic!") == "eastsideclinic",
          "the collapse keeps letters and digits and drops everything else",
          rw._collapse("East-Side, Clinic!"))
    check(rw._grounded_in("eastside", "the east side clinic") is True
          and rw._grounded_in("riverside", "she resides at campus") is False,
          "boundary-insensitive, sequence-sensitive — both halves of the claim")

    # ── call-20260825-1425: a race, not a judgement ──────────────────────────
    # The caller said "Same at Riverside campus 7th street" at 14:26:06. That
    # turn was still the `[...]` placeholder when save_branch ran a second
    # later, and _asserted_caller_text skips placeholders — so `riverside` went
    # into the record as a word nobody was heard to say, in the same artifact
    # that carries the sentence, timestamped one second BEFORE the save.
    #
    # Nothing here was wrong about the evidence it had. Transcription lands
    # after the audio, so the verdict was decided against a transcript that was
    # still filling in, and then never looked at again. The answer to a race is
    # to ask again once the racing is over.
    _rc = _east(["[...]"])
    _rc.memory.clear()
    _rc.memory.update(branch="Riverside Campus")
    _rc.memory.update(grounding=rw._grounding_verdict(
        rw._rode_along({"branch": "Riverside Campus"}, _rc),
        heard_any=False))
    _at_save = _rc.memory.get("grounding")
    check("SKIPPED" in _at_save,
          f"at save time there was nothing to check against ({_at_save[:40]!r})",
          "the placeholder is not silence — it is a transcript that has not "
          "arrived")
    # The transcript lands, one second late.
    _rc.turns[0] = rw.TranscriptTurn(
        role="caller", text="Same at Riverside campus 7th street.",
        timestamp="00:00:00", audio_rms=0.08)
    rw._revisit_grounding(_rc)
    check(_rc.memory.get("grounding") == "verified against caller transcript",
          f"re-read on the finished transcript, the save is verified "
          f"({_rc.memory.get('grounding')!r})",
          "the corroborating sentence is sitting in the artifact — a verdict "
          "written once, at save time, could never see it")
    check(_rc.memory.get("grounding_at_save") == _at_save,
          "and what was believed DURING the call is kept beside it",
          "a verdict that silently improves after the fact cannot be "
          "audited: 'did the guard fire on this call' has to stay answerable")

    # RE-DECIDED, NOT RELAXED. It runs the same rule against a better
    # transcript, so it moves in both directions — a term that never arrived
    # is still an exception at the end of the call.
    _rc2 = _east(["Hello. Okay, next slide, please."])
    _rc2.memory.clear()
    _rc2.memory.update(branch="Riverside Clinic",
                       grounding="verified against caller transcript")
    rw._revisit_grounding(_rc2)
    check("EXCEPT" in (_rc2.memory.get("grounding") or "")
          and "riverside" in (_rc2.memory.get("grounding") or ""),
          f"a word that never arrived is still called out at the end "
          f"({_rc2.memory.get('grounding')!r})",
          "if the re-read could only clear exceptions it would be a way of "
          "waiting out the guard")
    # ── A CONTESTED SAVE MUST NOT STAMP ITSELF "VERIFIED" ─────────────────
    # call-20260827-1010. The model tried to save "Riverside Campus"; the guard
    # refused it against a transcript reading "Private site campus"; the
    # escalation guard then pushed the model to save the TRANSCRIPT'S wording,
    # and that save stamped a bare "verified against caller transcript". The
    # row reached doctors.json as status="verified" and the block reached
    # nothing durable at all — "HALLUCINAT", "REJECTED" and "blocked" each
    # scored 0 against the artifact JSON.
    check(rw._grounding_verdict([], True) == "verified against caller transcript",
          "an uncontested save reads exactly as it always did",
          "the qualifier must cost nothing on a clean call")
    _cont = rw._grounding_verdict([], True,
                                  [{"value": "Riverside Campus", "why": "x"}])
    check(_cont.startswith("CONTESTED"),
          f"but a save made after a rejection says so ({_cont[:40]!r})")
    check("Riverside Campus" in _cont,
          "and names the value that was refused, not just the count",
          "a reviewer reading the row needs the candidate to compare against")
    # The refusal itself is durable now, and it is what the verdict reads.
    _rej = _east(["Private site campus."])
    _rej.memory.clear()
    check(_rej.branch_rejections == [],
          "a session starts with no branch rejections")
    _rej.branch_rejections.append({"value": "Riverside Campus", "why": "x",
                                   "at": "10:10:43"})
    _rej.memory.update(branch="Private site campus",
                       grounding="verified against caller transcript")
    rw._revisit_grounding(_rej)
    check("CONTESTED" in (_rej.memory.get("grounding") or ""),
          f"and the record-time re-read keeps the contest "
          f"({(_rej.memory.get('grounding') or '')[:40]!r})",
          "_revisit_grounding re-decides from scratch, so a qualifier it did "
          "not know about would be silently dropped there")

    # And a call with nothing saved has no verdict to revise.
    _rc3 = _east(["She's at the Northgate campus."])
    _rc3.memory.clear()
    rw._revisit_grounding(_rc3)
    check(_rc3.memory.get("grounding") is None
          and _rc3.memory.get("grounding_at_save") is None,
          "a call that saved nothing gets no grounding verdict invented for it")

    print("\n" + "-" * 66)
    print("  The goodbye retry goes through the one response.create site")
    print("-" * 66)
    # call-20260824-2113 logged conversation_already_has_active_response on the
    # closing retry. The retry site is NOT the cause — it defers to the watchdog
    # and calls the helper — but that is worth pinning, because the fix for the
    # in-handler-sleep version of this race was to route it here.
    _rt_src = _PKG_SRC
    _rt_block = _rt_src[_rt_src.find("_retry_at = sess._goodbye_retry_at"):][:900]
    check("_create_response(" in _rt_block,
          "the goodbye retry requests its response through _create_response",
          "a raw oai_ws.send here is the regression this pins")
    check("allow_when_done=True" in _rt_block,
          "and declares allow_when_done — it fires BECAUSE the call is closing",
          "the default policy would refuse it and drop the line in silence")
    check("already in flight" in _rt_block,
          "and treats a refusal as 'nothing to retry', not as a failure")
    # The residual race is inherent: _response_active is only as fresh as the
    # last event we read, and OpenAI can create a response before we hear about
    # it. Reported as benign rather than as an API ERROR.
    _err_block = _rt_src[_rt_src.find('elif event_type == "error":'):][:2600]
    check("conversation_already_has_active_response" in _err_block,
          "the server-side view of that race is recognised")
    check("Goodbye retry raced" in _err_block,
          "and reported as benign rather than as an API ERROR",
          "printing API ERROR for an expected, already-handled race is how a "
          "log teaches people to ignore it")

    print("\n" + "=" * 66)
    print("  `heard` IS SELECTED FROM THE TRANSCRIPT, NOT TAKEN ON TRUST")
    print("=" * 66)
    # call-20260824-2116. `heard` exists so the record shows what was SAID
    # rather than what was concluded, and it arrived model-authored and
    # unchecked while all three tool schemas told the model it was "checked
    # against the call transcript". Nothing checked it. The model inserted
    # clauses nobody uttered, and a fabricated quote is worse than a wrong
    # status: it reads as verbatim to whoever audits the row.
    def _hsess(pairs):
        _s = rw.RealtimeSession("CA00000000000000000000heard1",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
        for _role, _t in pairs:
            _s.turns.append(rw.TranscriptTurn(
                role=_role, text=_t, timestamp="00:00:00",
                audio_rms=0.09 if _role == "caller" else None))
        return _s

    # THE TWO LIVE FABRICATIONS, verbatim from the call.
    _h1 = _hsess([("agent", "Got it — are they taking new patients right now?"),
                  ("caller", "Yeah, definitely, you can reach out to them.")])
    _a1 = {"status": "yes",
           "heard": "Yeah, definitely, they're taking new patients also. "
                    "You can reach out to them."}
    check(rw._ungrounded_status(_a1, _h1) == "", "the status still grounds")
    check(_a1["heard"] == "Yeah, definitely, you can reach out to them.",
          "and `heard` is REPLACED with the caller's real turn",
          f"{_a1['heard']!r}")
    check("taking new patients also" not in _a1["heard"],
          "the invented clause is gone, not flagged",
          "selection removes the failure mode; validation would only catch it")

    _h2 = _hsess([("agent", "can a new patient actually get an appointment "
                            "scheduled right now?"),
                  ("caller", "Yeah, you need to book through online or call."),
                  ("caller", "Please do that.")])
    _a2 = {"status": "yes",
           "heard": "Yeah, you need to book through online or call from the "
                    "front desk. Please do that."}
    check(rw._ungrounded_scheduling(_a2, _h2) == "", "the status still grounds")
    check(_a2["heard"] == "Yeah, you need to book through online or call.",
          "and `heard` is the corroborating turn, not the model's version",
          f"{_a2['heard']!r}")
    check("front desk" not in _a2["heard"],
          "'from the front desk' — a phrase absent from the whole call — is gone")
    # A model that quotes correctly is unaffected: the selected turn IS its text.
    _a3 = {"status": "yes", "heard": "Yeah, definitely, you can reach out to them."}
    check(rw._ungrounded_status(_a3, _h1) == ""
          and _a3["heard"] == "Yeah, definitely, you can reach out to them.",
          "an honest quote survives unchanged — selection is not a penalty")
    # A REJECTED save leaves heard alone; there is no corroborating turn to take.
    _a4 = {"status": "no", "heard": "She's not taking anyone."}
    check(rw._ungrounded_status(_a4, _h1) != "",
          "a status the caller never gave is still refused")
    check(_a4["heard"] == "She's not taking anyone.",
          "and nothing is selected for a save that did not happen")

    print("\n" + "-" * 66)
    print("  `detail` is dropped when it carries words nobody said")
    print("-" * 66)
    # detail/depends_on cannot be fixed by selection — the field is a SUMMARY by
    # construction, so no single caller turn is the right thing to copy in. It
    # gets the fallback instead: word-level, because a summary legitimately
    # reorders and drops words, and a verbatim-substring rule would reject every
    # honest one.
    check(rw._ungrounded_detail({"detail": "Book online or call the front desk."},
                                _h2, "detail") == ["front", "desk"],
          "the live fabricated qualifier is caught, word by word",
          "'desk' appears nowhere in the call")
    check(rw._ungrounded_detail({"detail": "book online or call"}, _h2,
                                "detail") == [],
          "an honest summary of the same turn passes",
          "reordering and dropping words is what a summary IS")
    check(rw._ungrounded_detail({"detail": ""}, _h2, "detail") == [],
          "an absent qualifier is not a fabrication")
    check(rw._ungrounded_detail({"detail": "the front-desk"}, _h2,
                                "detail") == ["front", "desk"],
          "and the collapse applies here too — front-desk is front desk")
    # End to end: the save SURVIVES, only the qualifier is dropped.
    _h3 = _hsess([("agent", "can a new patient get an appointment scheduled?"),
                  ("caller", "Yeah, you need to book through online or call.")])
    # THE FIXTURE HAS TO REACH THE POINT IN THE CALL WHERE THIS IS ASKABLE.
    # scheduling is gated on accepting='yes', which is gated on
    # identity='confirmed' — a call cannot know how to book a new patient in
    # before it knows the practice takes them, and cannot know that before it
    # knows whose practice it is. This test is about the qualifier trimmer, so
    # it stands the call up where the trimmer runs; without these two the gate
    # HOLDS the save and the assertion below reads as the trimmer discarding
    # it, which is a different defect entirely.
    _h3.memory.update(**{obj.IDENTITY_STATUS_KEY: "confirmed",
                         obj.NEW_PATIENT_STATUS_KEY: "yes"})
    await rw._handle_tool_call(
        {"name": "save_scheduling_status", "call_id": "d1",
         "arguments": json.dumps({
             "status": "yes",
             "heard": "Yeah, you need to book through online or call.",
             "detail": "Book online or call the front desk."})},
        _h3, _TcWS(), {}, True)
    check(_h3.memory.get(obj.SCHEDULING_STATUS_KEY) == "yes",
          "the verified status is still saved",
          "refusing the whole call over a footnote would throw away a real answer")
    check(_h3.memory.get(f"{obj.SCHEDULING_STATUS_KEY}_detail")
          == "Book online or call",
          "the invented words are cut and the rest of the qualifier is kept",
          "discarding it whole cost a queue position on call-20260825-0922")
    check("desk" not in str(_h3.memory.get(f"{obj.SCHEDULING_STATUS_KEY}_detail")),
          "and the word nobody said is gone from what was stored")
    check(_h3.memory.get("scheduling_grounding_dropped_words") == ["front", "desk"],
          "and the drop is recorded, not silent",
          "a field quietly emptied is as invisible as the fabrication was")

    print("\n" + "-" * 66)
    print("  The values reach the record at all")
    print("-" * 66)
    # call-20260824-2116 recorded outcome=complete and collected=[all four] and
    # NOT ONE of the three status values: they were written to CallMemory — a
    # one-hour scratchpad — and never copied into the artifact. The call wrote
    # down THAT it succeeded and not WHAT it learned.
    _pf = _hsess([])
    _pf.memory.update(branch="Riverside Campus")
    _pf.memory.update(**{obj.NEW_PATIENT_STATUS_KEY: "yes",
                         f"{obj.NEW_PATIENT_STATUS_KEY}_heard":
                             "Yeah, definitely, you can reach out to them."})
    _pf.memory.update(**{obj.SCHEDULING_STATUS_KEY: "yes",
                         f"{obj.SCHEDULING_STATUS_KEY}_detail": "book online or call"})
    _pf.memory.update(**{obj.REFERRAL_STATUS_KEY: "always",
                         f"{obj.REFERRAL_STATUS_KEY}_depends_on": "primary care doctor"})
    _cf = _pf.collected_fields()
    check(sorted(_cf) == ["accepting", "branch", "referral", "scheduling"],
          "every declared field appears with its value",
          f"{sorted(_cf)}")
    check(_cf["accepting"]["value"] == "yes"
          and _cf["referral"]["value"] == "always",
          "the states themselves, which the artifact simply did not have")
    check(_cf["accepting"]["heard"] == "Yeah, definitely, you can reach out to them.",
          "with the caller's own words beside them")
    check(_cf["referral"]["depends_on"] == "primary care doctor",
          "and the qualifier under its own name, not flattened into one field")
    # Derived from the objective, so a fifth field cannot be forgotten here —
    # which is precisely how the first three went missing.
    _one = obj.CallObjective(fields=(obj.branch_field(),))
    _pf.objective = _one
    check(sorted(_pf.collected_fields()) == ["branch"],
          "and it follows the objective, not a hand-written list of keys",
          "the omission that lost three fields was a list nobody updated")

    # AND IT HAS TO REACH THE RECORD. Exercising collected_fields() proves the
    # collector works, not that anything calls it — removing the one line that
    # wires it into the artifact left every check above passing, which is the
    # same fake-coverage shape as a mutation that survives.
    _d_cf = Doctor(doctor_name="Dr. C", hospital_name="H", specialization="Cardiology")
    _s_cf = rw.RealtimeSession("CA0000000000000000000cfld", _d_cf)
    _s_cf.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
    _s_cf.memory.update(branch="Riverside Campus")
    _s_cf.memory.update(**{obj.NEW_PATIENT_STATUS_KEY: "yes"})
    _rec_cf = _s_cf._enrich_doctor("Riverside Campus", obj.Outcome.PARTIAL)
    check("collected_fields" in _rec_cf,
          "the directory row carries the non-branch fields",
          f"{sorted(_rec_cf)}")
    check(_rec_cf["collected_fields"].get("accepting", {}).get("value") == "yes",
          "with their values, on the row the client actually reads",
          "doctors.json had branch and city and nothing else this call learned")
    _wsrc = _PKG_SRC
    _rec_block = _wsrc[_wsrc.find('"call_id":        self.call_id,'):][:1400]
    check('"fields":' in _rec_block and "collected_fields()" in _rec_block,
          "and the call artifact record is wired to the collector",
          "collected=[...] without the values is a call that recorded THAT it "
          "succeeded and not WHAT it learned")

    # The schemas must not promise a check that does not exist.
    _tsrc2 = _plb.Path(_tools.__file__).read_text(encoding="utf-8")
    check("checked against the call transcript" not in _tsrc2.lower(),
          "and no tool schema still claims `heard` is checked",
          "it is REPLACED, which is a different and stronger promise")

    print("\n" + "=" * 66)
    print("  call-20260825-0915 — four defects on one waitlist call")
    print("=" * 66)

    def _wl(pairs):
        _s = rw.RealtimeSession("CA00000000000000000000wlist",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
        for _r, _t in pairs:
            _s.turns.append(rw.TranscriptTurn(
                role=_r, text=_t, timestamp="00:00:00",
                audio_rms=0.09 if _r == "caller" else None))
        return _s

    print("\n" + "-" * 66)
    print("  1. selection took a fragment because it took the LAST match")
    print("-" * 66)
    # The VAD split the caller's final answer, so the last turn classifying as
    # WAITLIST was the scrap "The status waitlist is" — which went into the
    # record as the quotation justifying the state.
    _frag = _wl([("agent", "could you say if she's taking new patients right now?"),
                 ("caller", "Yeah"),
                 ("caller", "Yeah, no no, we are full right now, so."),
                 ("caller", "You"),
                 ("caller", "The status waitlist is")])
    _fa = {"status": "waitlist", "heard": "whatever the model wrote"}
    check(rw._ungrounded_status(_fa, _frag) == "", "the status still grounds")
    check(_fa["heard"] == "Yeah, no no, we are full right now, so.",
          "the LONGEST matching turn is selected, not the last",
          f"{_fa['heard']!r}")
    check(_fa["heard"] != "The status waitlist is",
          "so a mid-sentence fragment is no longer the record's evidence",
          "last-wins put exactly this scrap in the artifact")
    # Not a flip to first-wins: a fragment can arrive first just as easily, and
    # every candidate already asserts the same state, so the only question left
    # is which is the fullest statement of it.
    _first = _wl([("agent", "is she taking new patients?"),
                  ("caller", "waitlist"),
                  ("caller", "We are full right now and you would be number 21.")])
    _fb = {"status": "waitlist", "heard": "x"}
    check(rw._ungrounded_status(_fb, _first) == ""
          and _fb["heard"] == "We are full right now and you would be number 21.",
          "and a fragment arriving FIRST is not selected either",
          f"{_fb['heard']!r}")
    # Ties go to the later turn — same claim, same length, prefer the one they
    # most recently stood behind.
    _tie = _wl([("agent", "is she taking new patients?"),
                ("caller", "We are full!"), ("caller", "We are FULL!")])
    _fc = {"status": "waitlist", "heard": "x"}
    rw._ungrounded_status(_fc, _tie)
    check(_fc["heard"] == "We are FULL!", "ties go to the later turn",
          f"{_fc['heard']!r} (both are {len('We are full!')} chars)")

    print("\n" + "-" * 66)
    print("  2. detail is trimmed, not discarded")
    print("-" * 66)
    # call-20260825-0922: caller "you will be the number 21", model "you would
    # be number 21". One verb tense emptied the field and took the queue
    # position with it.
    _d1 = _wl([("agent", "is she taking new patients?"),
               ("caller", "you will be the number 21")])
    _a1 = {"status": "waitlist", "detail": "you would be number 21"}
    _dropped, _ = rw._strip_ungrounded_detail(_a1, _d1, "detail")
    check(list(_dropped) == [],
          "will/would is an inflection of an auxiliary, not an invention",
          f"{list(_dropped)}: reporting it cost the queue position once already")
    check(_a1["detail"] == "you would be number 21",
          "so the qualifier survives WHOLE, not trimmed to a fragment",
          f"{_a1['detail']!r}")
    # Framing words the model uses to narrate provenance are not content.
    _d2 = _wl([("agent", "is she taking new patients?"),
               ("caller", "we are full right now, but I can put you on the "
                          "list. You would be number 21.")])
    _a2 = {"status": "waitlist",
           "detail": "You'd said earlier you would be number 21"}
    rw._strip_ungrounded_detail(_a2, _d2, "detail")
    check(_a2["detail"] == "you would be number 21",
          "and the remainder reads cleanly once they are gone",
          f"{_a2['detail']!r}")
    # A DELETION MUST NOT REWRITE THE CLAIM. An ungrounded negator drops the
    # whole qualifier — trimming it would assert the opposite.
    _d3 = _wl([("agent", "is she taking new patients?"),
               ("caller", "we are accepting new patients until January")])
    _a3 = {"status": "yes", "detail": "not accepting until January"}
    _dr3, _why3 = rw._strip_ungrounded_detail(_a3, _d3, "detail")
    check(list(_dr3) == ["not"],
          "only the negator is ungrounded — everything else WAS said",
          f"{list(_dr3)}: so a strip would leave a fluent, inverted sentence")
    check(_a3["detail"] == "",
          "and the whole qualifier is dropped rather than trimmed",
          "trimming would have produced 'accepting until January' — the "
          "opposite of what the model wrote, reading as if a human wrote it")
    check("negator" in _why3 or "claims" in _why3,
          "and the reason says why, rather than looking like a normal trim")
    # A remainder with nothing left in it is emptied — and recorded as emptied.
    _d4 = _wl([("agent", "is she taking new patients?"), ("caller", "we are full")])
    _a4 = {"status": "waitlist", "detail": "completely swamped indefinitely"}
    _dr4, _why4 = rw._strip_ungrounded_detail(_a4, _d4, "detail")
    check(_a4["detail"] == "" and _dr4,
          "nothing informative left -> emptied, with the words recorded",
          "a quietly blank field reads like a caller who volunteered nothing")
    check("dropped whole" in _why4,
          f"and the reason distinguishes emptied from trimmed ({_why4!r})",
          "the same empty string is reached by a trim that removed the last "
          "word and by a qualifier that was never usable — a reviewer reading "
          "the artifact can only tell them apart from this")
    # Danglers left by a deletion are trimmed so the field stays legible.
    _d5 = _wl([("agent", "can a new patient book in?"),
               ("caller", "Yeah, you need to book through online or call.")])
    _a5 = {"status": "yes", "detail": "Book online or call the front desk."}
    rw._strip_ungrounded_detail(_a5, _d5, "detail")
    check(_a5["detail"] == "Book online or call",
          "and a trailing function word left by the cut is trimmed",
          f"{_a5['detail']!r}")
    # A fully grounded qualifier is untouched.
    _d6 = _wl([("agent", "is she taking new patients?"),
               ("caller", "you will be number 21 on the list")])
    _a6 = {"status": "waitlist", "detail": "number 21 on the list"}
    _dr6, _ = rw._strip_ungrounded_detail(_a6, _d6, "detail")
    check(not _dr6 and _a6["detail"] == "number 21 on the list",
          "an honest qualifier passes through unchanged")

    print("\n" + "-" * 66)
    print("  3. a waitlist answer that never uses the ask's vocabulary")
    print("-" * 66)
    # The agent had asked for a BRANCH, so nothing matched ACCEPTING_ASK and the
    # never-asked path applied. The caller's textbook waitlist answer contains
    # none of "accepting", "taking new" or "new patients" — it says full, list,
    # number 21 — and the old topical test threw it out. Twice.
    _wa = _wl([("agent", "Hi, this is David... Do you know which branch Dr. "
                         "Okafor works out of?"),
               ("caller", "They have a waitlist. That's the Midtown office."),
               ("agent", "could you tell me the actual location name or the "
                         "street address for that site?"),
               ("caller", "Yeah, we are full right now, but I can put you on "
                          "the list. You would be number 21.")])
    check(not any(rw._is_ask_for(t.text, obj.ACCEPTING_ASK)
                  for t in _wa.turns if t.role == "agent"),
          "no agent turn asked about new patients — the never-asked path")
    check(not obj.ACCEPTING_ASK.search(
              "Yeah, we are full right now, but I can put you on the list. "
              "You would be number 21."),
          "and the answer contains none of the ask's vocabulary",
          "which is what the old topical test was testing for")
    check(rw._ungrounded_status({"status": "waitlist", "heard": "x"}, _wa) == "",
          "it grounds anyway — the turn states the condition in its own words",
          "refused twice on the live call, and the queue position was lost")
    # The rule the topical test was defending still holds.
    _bare = _wl([("caller", "Yes, speaking.")])
    check(rw._ungrounded_status({"status": "yes", "heard": "x"}, _bare) != "",
          "a bare 'Yes, speaking.' with no ask is STILL refused",
          "it classifies on its opening token and asserts nothing")
    # BOTH POLARITY FAMILIES, and only as a DISCOURSE MARKER. "No, not at the
    # moment." answered the SCHEDULING question; with only the yes-family
    # stripped it stood alone as a referral NO, because a bare "no" is a valid
    # referral answer. And the delimiter matters: in "no referral needed" the
    # word is a determiner carrying the meaning, not a preface to it.
    check(obj.states_in_its_own_right("No, not at the moment.", "no",
                                      obj.classify_referral) is False,
          "a bare 'No,' does not stand alone as a REFERRAL answer",
          "each field classifies with its own vocabulary, not classify_choice")
    check(obj.states_in_its_own_right("no referral needed", "no",
                                      obj.classify_referral) is True,
          "but 'no referral needed' does — there the 'no' is the content",
          "an undelimited polarity word is a determiner, not a preface")
    # WHERE THE TWO VOCABULARIES DISAGREE, which is the only place the
    # classifier argument can be shown to be load-bearing: this is an ACCEPTING
    # answer, and classify_choice reads it as NO while classify_referral does
    # not recognise it at all.
    check(obj.classify_choice("we are not taking anyone") is obj.ChoiceAnswer.NO
          and obj.classify_referral("we are not taking anyone") is None,
          "the two vocabularies genuinely disagree on this sentence")
    check(obj.states_in_its_own_right("we are not taking anyone", "no",
                                      obj.classify_referral) is False,
          "so it does NOT stand alone as a referral answer",
          "defaulting to classify_choice would let an accepting answer ground "
          "a referral claim nobody was asked for")
    for _t, _st, _want in [
        ("Yes, speaking.", "yes", False),
        ("Yeah", "yes", False),
        ("Yeah, we are full right now, but I can put you on the list.",
         "waitlist", True),
        ("We're full right now.", "waitlist", True),
        ("we are not taking anyone", "no", True),
    ]:
        check(obj.states_in_its_own_right(_t, _st) is _want,
              f"stands alone={_want!s:5} {_t[:44]!r}",
              "strip the leading yes and see whether it still says the same")

    print("\n" + "-" * 66)
    print("  4. 'McDonald office' — where it came from")
    print("-" * 66)
    # The caller said "That's the Midtown office."; the model saved "McDonald
    # office". The guard caught it. The question was the source.
    _mc = "mcdonald"
    for _n, _tpl in _templates.TEMPLATES.items():
        _sent = (_tpl.instructions + _tpl.greeting + _tpl.transcribe_hint).lower()
        check(_mc not in _sent,
              f"{_n}: 'McDonald' is not in anything sent to the model")
    check(not any(_mc in _g for _g in _tools._prompt_echoes()),
          "nor in the derived prompt-echo grams")
    check(_templates.clean_doctor_name("Dr. Jane Okafor") == "Jane Okafor",
          "and clean_doctor_name produces nothing like it from this doctor",
          "the .title() hypothesis is ruled out — 'McDonald' lives only in that "
          "function's DOCSTRING, which is never transmitted")
    # It is a hallucination, and the guard is what stands between it and the
    # directory. That guard must keep working.
    _mcs = _wl([("agent", "which branch does Dr. Okafor work out of?"),
                ("caller", "That's the Midtown office.")])
    check(rw._ungrounded_terms({"branch": "McDonald office"}, _mcs) != "",
          "a branch nobody said is still refused",
          "this is the guard doing its job, not a defect to fix")
    check(rw._ungrounded_terms({"branch": "Midtown office"}, _mcs) == "",
          "while the branch they DID say grounds")

    print("\n" + "=" * 66)
    print("  SPECIALTY — the disambiguator, carried through to the call")
    print("=" * 66)
    # Confirmed with the client-side contact 2026-08-25: two doctors of the same
    # name at one hospital is the ordinary case, and the specialty is how a
    # receptionist knows which is meant. Both client scripts open
    # "Dr. [Name], [Specialty]" for exactly that reason.
    _d_spec = Doctor(doctor_name="Dr. Jane Okafor",
                     hospital_name="Northside Medical Group",
                     specialization="Cardiology")
    _ctx_spec = _templates.PROVIDER_VERIFICATION.build_context(
        _d_spec, callback_number="", callback_email="", org="Forage AI",
        agent_name="David")
    check("Cardiology" in _ctx_spec,
          "the specialty reaches CALL CONTEXT")
    check("Dr. Okafor, Cardiology" in _ctx_spec,
          "with the agent told to SAY it when identifying the doctor",
          "stating the fact alone reads as a form field, not as the half of "
          "the name that identifies the person")
    check("same surname" in _ctx_spec or "tells them apart" in _ctx_spec,
          "and told WHY, so it survives a turn where they sound unsure")
    # Absent is absent — never sent as "unknown", which invites the agent to
    # say so out loud to a receptionist.
    _d_nospec = Doctor(doctor_name="Dr. Jane Okafor",
                       hospital_name="Northside Medical Group")
    _ctx_nospec = _templates.PROVIDER_VERIFICATION.build_context(
        _d_nospec, callback_number="", callback_email="", org="Forage AI",
        agent_name="David")
    check("Specialty" not in _ctx_nospec,
          "no specialty -> the line is omitted, not sent as 'unknown'")

    # THE LONG-STANDING is_complete() GAP. REQUIRED_FOR_COMPLETE has always
    # named specialization and nothing ever supplied it, so every doctor this
    # agent resolved failed on that one field and was filed PARTIALLY_VERIFIED
    # however good the call was — see missing_for_complete()'s own docstring.
    def _resolve_with(spec):
        _d = Doctor(doctor_name="Dr. Jane Okafor",
                    hospital_name="Northside Medical Group",
                    specialization=spec)
        _s = rw.RealtimeSession("CA0000000000000000000spec2", _d)
        _s.memory.update(branch="Riverside Campus")
        _s._enrich_doctor("Riverside Campus", obj.Outcome.COMPLETE)
        return _d

    _d_no = _resolve_with(None)
    check(_d_no.missing_for_complete() == ["specialization"]
          and _d_no.status is rw.DoctorStatus.PARTIALLY_VERIFIED,
          "without a specialty a perfect call still cannot reach COMPLETE",
          "the gap, unchanged — and now nameable rather than mysterious")
    _d_yes = _resolve_with("Cardiology")
    check(_d_yes.is_complete() and _d_yes.missing_for_complete() == [],
          "with one, the record is COMPLETE at last",
          f"{_d_yes.missing_for_complete()}")
    check(_d_yes.status is rw.DoctorStatus.VERIFIED,
          "and the resolved doctor is finally filed VERIFIED, not PARTIALLY",
          _d_yes.status.value)

    # THE CACHE PREFIX STAYS CLEAN. All of this is per-call context; none of it
    # may reach the static instructions, or every campaign switch is a cold
    # cache.
    for _n, _tpl in _templates.TEMPLATES.items():
        for _leak in ("Jane", "Okafor", "Northside", "Forage AI", "David"):
            check(_leak not in _tpl.instructions,
                  f"{_n}: {_leak!r} stays out of the cached instructions")

    # THE CLI HAS TO ACTUALLY WIRE IT. Nothing in this suite executes
    # run_twilio.py, so --specialty could be advertised in --help and then
    # dropped on the floor while every check above still passed — a mutation
    # that removed exactly that wiring was invisible until this was added.
    # Read from source, because importing the module places a call.
    _rt_cli = _plb.Path("run_twilio.py").read_text(encoding="utf-8")
    check('"--specialty"' in _rt_cli,
          "run_twilio.py accepts --specialty")
    check("specialization=args.specialty" in _rt_cli,
          "and passes it into the Doctor it builds",
          "an accepted flag that reaches nothing is worse than no flag")

    print("\n" + "-" * 66)
    print("  GREETING — the client contact's own wording")
    print("-" * 66)
    # Her exact sanction: "you can say I'm calling on behalf of Forage AI to
    # verify some information that was missed on our website."
    _g = _templates.PROVIDER_VERIFICATION.build_greeting(
        _d_spec, org="Forage AI", agent_name="David")
    check("verify some information that was missed on our website" in _g,
          "the greeting uses her phrasing", _g)
    check("check a provider listing" not in _g,
          "and not the wording it replaced, which was ours")
    # The two things kept around it, each for a stated reason.
    _ident = min(_g.find("David"), _g.find("Forage AI"))
    check(_ident >= 0 and _g.index("verify some information") > _ident,
          "the identification still comes FIRST",
          "an automated call opens with the real caller and the organisation "
          "it represents — not negotiable against a preferred wording")
    check("not calling to book anything" in _g,
          "and the not-booking clause is kept",
          "it is from her own script and does real work: asking whether a "
          "doctor takes new patients sounds like someone trying to become one")
    check("on behalf of" in _g,
          "still 'on behalf of' — no employment claim")
    check(_g.rstrip().endswith("?"),
          "and it still ends on the ask, so the callee has a turn to take")

    print("\n" + "=" * 66)
    print("  IDENTITY — the question the script never asked")
    print("=" * 66)
    # From the client-side contact: "First level of check will be — is this Dr.
    # John Smith's office? ... If we don't know which doctor they're talking
    # about, accepting new patients makes no sense." The objective had four
    # fields and none of them established that the right doctor at the right
    # practice had been reached.
    _PVI = _templates.PROVIDER_VERIFICATION_OBJECTIVE
    check([f.name for f in _PVI.fields][0] == "identity",
          "identity is the FIRST field", f"{[f.name for f in _PVI.fields]}")
    check(obj.unwritable_fields(_PVI) == (),
          "every field is written by a tool — run BEFORE wiring, not after")
    check(obj.invalid_conditions(_PVI) == (),
          "and every gate in the chain is sound",
          f"{obj.invalid_conditions(_PVI)}")
    # The chain: branch and accepting gate on identity, scheduling and referral
    # on accepting. Two deep, which the validator used to refuse outright.
    for _fn in ("branch", "accepting"):
        _f = _PVI.field_named(_fn)
        check(_f is not None and _f.required_when is not None
              and _f.required_when.field == "identity",
              f"{_fn} is gated on identity")
    for _fn in ("scheduling", "referral"):
        _f = _PVI.field_named(_fn)
        check(_f is not None and _f.required_when is not None
              and _f.required_when.field == "accepting",
              f"{_fn} stays gated on accepting — a two-deep chain")

    # ── The gate ENFORCED, not merely declared ───────────────────────────────
    # call-20260825-1437. Everything above passed on that call and the branch
    # and the new-patient status were both filed anyway, for a doctor the call
    # never confirmed — reaching doctors.json stamped source=voice,
    # status=partially_verified, with `missing: ["identity"]` sitting in the
    # same artifact saying so. RequiredWhen decided whether a field was
    # REQUIRED; nothing decided whether it could be WRITTEN, and those are
    # different questions. run_tool is the one place with both the objective
    # and the memory, so it is where the second one gets asked.
    def _gm(name):
        _m = rw.CallMemory(name)
        _m.clear()
        return _m

    # PENDING — the gate field has no answer yet. HELD, not discarded: the
    # caller really did say this, and throwing away a real answer is this
    # project's expensive direction of failure.
    _g1 = _gm("test-gate-pending")
    _r1 = _tools.run_tool("save_branch", _g1, {"branch": "Eastside Clinic"},
                          objective=_PVI)
    check(not _r1.get("ok") and _g1.get("branch") is None,
          "a branch is NOT filed while identity is unsettled",
          f"this is 1437: {_g1.get('branch')!r} reached the directory under a "
          f"doctor nobody had confirmed")
    check([d["field"] for d in (_g1.get("deferred_saves") or [])] == ["branch"],
          f"it is HELD ({_g1.get('deferred_saves')})",
          "refusing outright would discard an answer the caller gave, which "
          "looks identical afterwards to a receptionist who would not say")
    check("HELD" in (_r1.get("error", "") + _r1.get("need", "")),
          f"and the model is TOLD it is held ({_r1})",
          "otherwise it goes back and asks a question already answered")

    # ...and the moment the gate opens, the held value is applied. No second
    # ask, no lost answer.
    _tools.run_tool("save_doctor_identity", _g1,
                    {"identity": "confirmed", "heard": "Yes, that's her office."},
                    objective=_PVI)
    check(_g1.get("branch") == "Eastside Clinic",
          f"the held branch lands the moment identity is confirmed "
          f"({_g1.get('branch')!r})",
          "held and never applied is the same lost answer with extra steps")
    check(_g1.get("deferred_applied") == ["branch"]
          and not _g1.get("deferred_saves"),
          f"and the flush is recorded, not silent ({_g1.get('deferred_applied')})")

    # CLOSED — the gate field is answered, and answered AGAINST this field.
    # Nothing to hold: identity=not_here means the doctor is not at this
    # practice, so a branch collected there belongs to nobody and no later turn
    # on this call can make it belong to somebody.
    _g2 = _gm("test-gate-closed")
    _tools.run_tool("save_doctor_identity", _g2,
                    {"identity": "not_here", "heard": "No, she left last year."},
                    objective=_PVI)
    _r2 = _tools.run_tool("save_branch", _g2, {"branch": "Eastside Clinic"},
                          objective=_PVI)
    check(not _r2.get("ok") and _g2.get("branch") is None
          and not _g2.get("deferred_saves"),
          "a branch is REFUSED outright once identity came back not_here",
          "holding it would be waiting for a turn that cannot come")
    check([d["why"] for d in (_g2.get("deferred_dropped") or [])]
          == ["identity=not_here"],
          f"and the drop says which answer closed it "
          f"({_g2.get('deferred_dropped')})",
          "a value that evaporates between the caller saying it and the "
          "artifact being written is the invisibility every guard here was "
          "retrofitted for")

    # An UNGATED field is untouched by any of this — identity itself has no
    # gate, and neither has any field of either branch template.
    _g3 = _gm("test-gate-ungated")
    check(_tools.run_tool("save_doctor_identity", _g3,
                          {"identity": "confirmed", "heard": "Speaking."},
                          objective=_PVI).get("ok"),
          "the ungated field saves normally",
          "a gate that blocks the field opening the gate deadlocks the call")

    # The model correcting itself supersedes; it does not queue. Replaying a
    # retracted value would write the one they took back.
    _g4 = _gm("test-gate-supersede")
    _tools.run_tool("save_branch", _g4, {"branch": "Eastside Clinic"},
                    objective=_PVI)
    _tools.run_tool("save_branch", _g4, {"branch": "Riverside Campus"},
                    objective=_PVI)
    check(len(_g4.get("deferred_saves") or []) == 1,
          f"a second attempt at a held field replaces the first "
          f"({_g4.get('deferred_saves')})")
    _tools.run_tool("save_doctor_identity", _g4,
                    {"identity": "confirmed", "heard": "Yes."}, objective=_PVI)
    check(_g4.get("branch") == "Riverside Campus",
          f"and what lands is the one they stood behind last "
          f"({_g4.get('branch')!r})",
          "replaying both in order writes the value they retracted")

    _K = obj.IDENTITY_STATUS_KEY
    for _lbl, _mem, _out in [
        ("nothing",      {},                      obj.Outcome.NONE),
        ("wrong_number", {_K: "wrong_number"},    obj.Outcome.COMPLETE),
        ("not_here",     {_K: "not_here"},        obj.Outcome.COMPLETE),
        ("unsure",       {_K: "unsure"},          obj.Outcome.COMPLETE),
        ("confirmed",    {_K: "confirmed"},       obj.Outcome.PARTIAL),
    ]:
        check(_PVI.outcome(_fake(_mem)) is _out,
              f"identity={_lbl:13} -> {_out.label}",
              _PVI.outcome(_fake(_mem)).label)
    # A denied identity makes the whole rest of the script not-applicable —
    # asking a bakery which branch Dr. Okafor works from is not a question.
    _denied = _fake({_K: "wrong_number"})
    check(set(_PVI.not_applicable(_denied))
          == {"branch", "accepting", "scheduling", "referral"},
          "and everything downstream is n/a, not missing",
          f"{_PVI.not_applicable(_denied)}")
    check(_PVI.missing_spoken(_denied) == "",
          "so the agent has nothing to apologise for out loud",
          "it did not fail to get a branch; there was no branch to get")

    # THE TWO NEGATIVES ARE DIFFERENT OUTCOMES.
    check(obj.IdentityAnswer.NOT_HERE.value != obj.IdentityAnswer.WRONG_NUMBER.value,
          "not_here and wrong_number are separate states",
          "one says the listing is wrong and the number is fine; the other "
          "says the number is wrong. Collapsing them sends somebody to "
          "re-verify a number that was never the problem")
    for _t, _want in [
        ("Yes, this is Dr. Okafor's office.", "confirmed"),
        ("Speaking.", "confirmed"),
        ("No, she doesn't work here.", "not_here"),
        ("There is no one by that name.", "not_here"),
        ("She left last year.", "not_here"),
        # The specialty mismatch, which reads nothing like a denial.
        ("We have a Dr. Smith but he's a dermatologist.", "not_here"),
        ("You've got the wrong number.", "wrong_number"),
        ("This is a bakery.", "wrong_number"),
        ("I'm not sure, I'm new here.", "unsure"),
        # Offering to look is not a denial.
        ("We have a few doctors but I can check.", "unsure"),
        ("It is 1426 7th Street.", None),
    ]:
        _g = obj.classify_identity(_t)
        check((_g.value if _g else None) == _want,
              f"identity={_want!s:12} {_t[:42]!r}")

    # The probe must not collide with the branch ask — office and practice are
    # LOCATION_NOUN too.
    check(obj.IDENTITY_ASK.search("Is this Dr. Okafor's office?") is not None,
          "the identity ask is recognised")
    check(obj.IDENTITY_ASK.search("Which branch does she work out of?") is None,
          "and a BRANCH ask is not — it would anchor the guard on the wrong turn")

    # ── "is this THE OFFICE FOR Dr. X" — call-20260827-1428 ────────────────
    # Third recurrence of one defect: the probe demanded "is this" immediately
    # followed by the title, the model softened the wording, and the evidence
    # anchor never moved off the FIRST identity question. The window still held
    # "No, I'm just receptionist"; the caller's clean "Yeah, it is." — which
    # classify_identity reads as CONFIRMED on its own — was refused, the agent
    # asked a third time, and the receptionist answered "Yes, I said".
    check(obj.classify_identity("Yeah, it is. How can I help you?")
          is obj.IdentityAnswer.CONFIRMED,
          "the caller's answer was always readable — the ASK was not",
          "which is why the fix is the probe, not the vocabulary")
    for _q in ["Thanks — is this the office for Dr. Jennifer, Cardiology?",
               "Right, thanks — just to be clear, is this the office for "
               "Dr. Jennifer, Cardiology?",
               "Is this the practice of Dr. Reyes?",
               "Is this the office of Dr. Okafor?"]:
        check(rw._is_ask_for(_q, obj.IDENTITY_ASK),
              f"a softened identity ask is still an identity ask: {_q[:52]!r}")
    # THE LOOKAHEAD IS THE MUTATION THAT MATTERS. With a bare gap this probe
    # swallows a BRANCH ask and anchors the identity guard on the wrong turn —
    # the exact collision the two checks above this block exist to prevent.
    # `office`/`practice`/`clinic` stay allowed because they name WHOSE office
    # it is; `branch`/`campus`/`site` never do.
    for _q in ["Is this the branch Dr. Jennifer works out of?",
               "Is this the campus Dr. Reyes works from?",
               "Is this the site Dr. Okafor sees patients at?"]:
        check(not obj.IDENTITY_ASK.search(_q),
              f"and a branch ask naming the doctor is still NOT one: {_q[:50]!r}")

    # ── "Right now no." — the same call, the other probe ──────────────────
    # The NO branch anchored the bare token at the string start while YES used
    # _LEAD_IN, so "Right now yes." classified and "Right now no." did not: the
    # affirmative was heard and the negative thrown away. A held save was
    # REFUSED against the transcript that bears it out, and the receptionist
    # had to restate it.
    #
    # Fixed at BOTH ends rather than by growing _LEAD_IN, because enumerating
    # lead-ins is the losing game these probes keep losing. What every one of
    # them has in common is the polarity word LAST — which is exactly what a
    # determiner never is.
    for _t in ["Right now no.", "Right now, no.", "At the moment no.",
               "Unfortunately no.", "Sorry, no.", "I think no.", "Ah, no."]:
        check(obj.classify_choice(_t) is obj.ChoiceAnswer.NO,
              f"a padded no is still a no: {_t!r}",
              f"got {obj.classify_choice(_t)!r}")
    check(obj.classify_referral("Right now no.") is obj.ReferralAnswer.NO,
          "and the referral vocabulary gets the same parity")
    # THE DETERMINER MUST NOT FLIP, which is what bounds the trailing form.
    for _t, _want in [("There is no waitlist.", "waitlist"),
                      ("We're full, but I can put you on the list.", "waitlist"),
                      ("No idea.", "unsure"),
                      ("She is accepting new patients.", "yes"),
                      ("Right now yes.", "yes"),
                      ("No she isn't.", "no"),
                      ("No we are not taking patients.", "no")]:
        _g = obj.classify_choice(_t)
        check(_g is not None and _g.value == _want,
              f"{_t!r} -> {_want}", f"got {_g!r}")
    # AND THE ONE THIS FOUND ON THE WAY PAST. "No problem" is an
    # acknowledgement, not an answer, and the bare leading token was reading it
    # as a refusal — turning a sentence that says the practice IS taking
    # patients into a row that says it is not. Pre-existing, in the same three
    # lines, and the wrong-row failure this whole system exists to prevent.
    check(obj.classify_choice("No problem, she's taking new patients.")
          is obj.ChoiceAnswer.YES,
          "'No problem' is an acknowledgement, not a refusal",
          "it used to classify NO — a YES answer recorded as its opposite")

    # ── THE PROBE SWEEP, 2026-08-27 ───────────────────────────────────────
    # Run across every remaining probe after the same two shapes were found
    # twice on call-20260827-1428. It found four more, and one of them was the
    # worst bug in objectives.py.
    #
    # 1. `n'?t\b` HAD NO LEFT BOUNDARY. Written for "isn't"/"isnt", it matched
    #    the tail of every word ending in "nt" — moment, patient, appointment,
    #    current, different, urgent, front, want, recent, department. So an
    #    affirmative anywhere after the word "patient" in its own clause was
    #    flipped: "Any patient we are happy to see." classified NO, a practice
    #    agreeing to see someone recorded as a refusal. PRE-EXISTING on the
    #    shipped patterns; the trailing-affirmative form only made it findable.
    for _w in ["moment", "patient", "appointment", "current", "different",
               "urgent", "front", "want", "recent", "department"]:
        check(not obj._NEGATOR.search(_w),
              f"{_w!r} is not a negator", "it merely ends in 'nt'")
    check(obj.classify_choice("Any patient we are happy to see.")
          is obj.ChoiceAnswer.YES,
          "an affirmative after the word 'patient' is not negated",
          "this classified NO — the wrong-row failure, on shipped patterns")
    # AND REAL NEGATION STILL FLIPS, with and without the transcriber's
    # apostrophe. A closed class of a dozen contractions is the one place a
    # list IS the right tool — unlike the ways a question can be phrased.
    for _t in ["She's not currently taking new patients.",
               "We aren't taking anyone.", "We arent taking anyone.",
               "She isn't accepting.", "We don't take new patients.",
               "We dont take new patients."]:
        check(obj.classify_choice(_t) is obj.ChoiceAnswer.NO,
              f"real negation still reads as NO: {_t!r}")

    # 2. AFFIRMATIVE/NEGATIVE PARITY, tested on the SAME sentence. The first
    #    cut of the trailing form went onto NO alone, which inverted the
    #    asymmetry rather than removing it — "At the moment no." classified and
    #    "At the moment yes." did not. Both polarities, every time.
    for _a, _b in [("Right now yes.", "Right now no."),
                   ("At the moment yes.", "At the moment no."),
                   ("Unfortunately yes.", "Unfortunately no."),
                   ("Sorry, yes.", "Sorry, no."),
                   ("I think yes.", "I think no."),
                   ("Probably yes.", "Probably no.")]:
        check(obj.classify_choice(_a) is obj.ChoiceAnswer.YES
              and obj.classify_choice(_b) is obj.ChoiceAnswer.NO,
              f"both polarities read on the same shape: {_a!r} / {_b!r}",
              f"{obj.classify_choice(_a)!r} / {obj.classify_choice(_b)!r}")

    # 3. LOCATION_NOUN was missing `clinic` — the word IDENTITY_ASK already
    #    treats as a place, and the branch question is the primary field of
    #    this project. "Which clinic does she work out of?" was not a location
    #    ask, so the budget did not count it and the anchor did not move.
    for _t in ["Which clinic does she work out of?",
               "Which facility is she based at?",
               "Which building does she see patients in?",
               "Which centre is she at?",
               "The medical center on Broadway."]:
        check(obj.LOCATION_NOUN.search(_t),
              f"a place named in the words a front desk uses: {_t[:44]!r}")
    # `clinics?`, not `clinic\w*`: the shared prompt says "route you to
    # clinical staff", and a probe matching that reads staff as a location.
    check(not obj.LOCATION_NOUN.search("route you to clinical staff"),
          "'clinical' is not a place")

    # 4. ACCEPTING_ASK demanded `taking`/`new` touch.
    check(obj.ACCEPTING_ASK.search("Is she taking anybody new?"),
          "slack between the verb and 'new'")
    check(not obj.ACCEPTING_ASK.search(
              "Are you taking a message for the doctor?"),
          "and two words of slack does not reach across to an unrelated 'new'")

    # ── note_info WAS WRITE-ONLY — call-20260827-1516 ─────────────────────
    # It did memory.update(**{f"note_{key}": value}), returned ok, and NOTHING
    # anywhere read a note_ key back out: not the artifact, not doctors.json,
    # not the summary. On 1516 the caller said it was a bad time and asked to
    # be rung back; the agent captured the window and called note_info twice,
    # and the call filed outcome=none, collected=[]. The one actionable thing
    # it learned survived only as prose inside the transcript array.
    _nsess = rw.RealtimeSession("CA00000000000000000000notes1",
                                Doctor(doctor_name="Dr. Alan Reyes",
                                       hospital_name="Lakeview Medical"))
    _tools.note_info(_nsess.memory, "callback_time",
                     "Caller requested callback in the afternoon.")
    _tools.note_info(_nsess.memory, "email", "directory@example.com")
    _tools.note_info(_nsess.memory, "blank", "")
    check(_nsess.notes() == {"callback_time": "Caller requested callback in "
                                              "the afternoon.",
                             "email": "directory@example.com"},
          f"note_info values are readable back off the session "
          f"({_nsess.notes()})",
          "a tool that reports ok and whose output nothing can see is the "
          "acts-and-leaves-no-trace defect wearing a third hat")
    check("note_" not in json.dumps(_nsess.notes()),
          "the memory-namespacing prefix does not leak into the record")
    check(rw.RealtimeSession("CA00000000000000000000notes2",
                             Doctor(doctor_name="X", hospital_name="Y")
                             ).notes() == {},
          "and a call that noted nothing reports nothing")
    # BOTH SERIALISERS. The artifact is what a reviewer reads; doctors.json is
    # what a person acts on, and a call whose only product was "ring back this
    # afternoon" has to put it on the row, not only on the artifact.
    _ssrc = _PKG_SRC[_PKG_SRC.find("def notes(self)"):]
    for _where, _hint in [('"notes":          self.notes() or None,', "artifact"),
                          ('"notes":            self.notes(),', "doctor record")]:
        check(_where in _PKG_SRC,
              f"notes reach the {_hint}",
              "one serialiser is how this got lost the first time")

    # ── SIGNED OFF WITHOUT ENDING ANYTHING — the same call ────────────────
    # sess.done moves on exactly four events and every one is a TOOL or the
    # caller; the agent saying goodbye was invisible. On 1516 it said "No
    # problem — take care." at 15:17:00, called nothing, and the call ran
    # another twenty seconds until the CALLER ended it.
    for _t in ["No problem — take care.", "Take care.", "Bye now.",
               "Have a good day.",
               "Alright, thanks for your time earlier — goodbye."]:
        check(rw._spoken_farewell(_t), f"a sign-off is recognised: {_t!r}")
    # `take care OF` is a promise to act. Reading it as goodbye would inject
    # the escalate directive into the middle of a call that is going fine.
    for _t in ["I'll take care of that.", "I will take care of it for you.",
               "Got it — which branch does she work out of?",
               "Got it — we'll follow up this afternoon."]:
        check(not rw._spoken_farewell(_t),
              f"and an ordinary turn is not: {_t!r}")
    # THE CORRECTION IS THE TOOL, NEVER A HANG-UP. escalate is what writes the
    # reason — on 1516 it was the only record of why the call produced nothing
    # — so cutting the line at the farewell trades twenty seconds of politeness
    # for a call with no outcome at all.
    _fsrc = _PKG_SRC[_PKG_SRC.find("if (_spoken_farewell(_said) and not sess.done"):][:1200]
    check(_fsrc and "Call escalate now" in _fsrc,
          "the directive asks for the tool",
          "hanging up here would lose the reason the call ended")
    check("sess.done = True" not in _fsrc and "twilio_ws" not in _fsrc,
          "and it neither ends the call itself nor touches the wire")
    check("_farewell_nudged" in _fsrc and "farewell_without_close" in _fsrc,
          "one-shot, and recorded so it cannot fire invisibly")

    # ── THE PRECEDENCE TRAP ───────────────────────────────────────────────
    # _NEGATOR's own docstring gives this sentence as the one that must be YES,
    # and it returned NO. The clause machinery was right and was never reached:
    # `we'?re not` carries NO OBJECT — it does not say what they are not doing
    # — and the NO branch is tested before YES, so a negative about walk-ins
    # beat the answer to the question actually asked. The tell was that
    # "we don't do walk-ins, but yes…" returned YES, because `don't` is not one
    # of the NO branch's bare literals.
    check(obj.classify_choice(
              "we're not doing walk-ins, but she is taking new patients")
          is obj.ChoiceAnswer.YES,
          "an affirmative in a LATER clause is the operative answer",
          "the sentence _NEGATOR's docstring promises and did not deliver")
    check(obj.classify_choice(
              "we don't do walk-ins, but yes she is taking new patients")
          is obj.ChoiceAnswer.YES,
          "and the form that already worked still does",
          "if this ever fails the fix went in at the wrong layer")
    # NARROWER THAN THE YES PATTERN, DELIBERATELY. Only an affirmative naming
    # NEW PATIENTS may overturn a stated negative — the answer has to be about
    # the thing that was asked. These are the mutations that matter: a general
    # affirmative, or a second negative, must both leave it NO.
    for _t in ["we're not taking new patients, but she is happy to see "
               "existing ones.",
               "we're not doing walk-ins, and we're not taking new patients",
               "We're not taking new patients.",
               "She's not currently taking new patients."]:
        check(obj.classify_choice(_t) is obj.ChoiceAnswer.NO,
              f"a stated negative is not overturned by: {_t[:46]!r}",
              f"got {obj.classify_choice(_t)!r}")
    # A CLAUSE BOUNDARY IS REQUIRED, so this can never fire inside the
    # negative's own clause — "not taking new patients" contains the very
    # phrase _EXPLICIT_YES looks for.
    check(not obj._affirmed_after("we're not taking new patients", 9),
          "an affirmative in the SAME clause is not a later answer")

    # ── THIRD-PERSON ACCEPTANCE ───────────────────────────────────────────
    # `we'?re accepting` covered the receptionist speaking for the practice and
    # nothing else, so the natural way to answer a question ABOUT the doctor
    # returned None and the answer was discarded.
    for _t in ["She is accepting.", "She's accepting.", "They are accepting.",
               "The doctor is accepting.", "We're accepting."]:
        check(obj.classify_choice(_t) is obj.ChoiceAnswer.YES,
              f"third person is an answer too: {_t!r}",
              f"got {obj.classify_choice(_t)!r}")
    # Copula-anchored, not subject-anchored — enumerating subjects is the same
    # losing game as enumerating phrasings. The negated forms stay NO because
    # that branch is tested first.
    for _t in ["She is not accepting.", "She isn't accepting.",
               "They are not accepting."]:
        check(obj.classify_choice(_t) is obj.ChoiceAnswer.NO,
              f"and its negation is still NO: {_t!r}",
              f"got {obj.classify_choice(_t)!r}")

    print("\n" + "-" * 66)
    print("  check_refusals — finding the next probe gap without a person")
    print("-" * 66)
    # Seven probe gaps have been found on this project and every one the same
    # way: a person read a console log. The offline scan replaces the FINDING,
    # not the judging — the alternative considered was classifying intent with
    # a second LLM on the live path, and the arithmetic killed it (60
    # classifications on a median 81s call = 44 RPM from one call against a 10
    # RPM ceiling, on a reply path measured at 1.52s median), before the deeper
    # objection that a model checking a model's reading of the same words is
    # not a guardrail.
    import check_refusals as _cr

    # A REFUSAL ALONE IS NOT A FINDING. Most refusals are the guard working,
    # and a scan that flags them all is a scan nobody runs twice. What indicts
    # one is what the call went on to do with the field.
    def _call(**kw):
        base = {"call_id": "call-test", "conversation": {"agent_turns": 3},
                "missing": [], "collected": []}
        base.update(kw)
        return base
    _ref = [{"tool": "save_new_patient_status", "why": "no", "heard": "Right now no.",
             "at": "00:00:00", "args": {}}]
    check(_cr.audit_dict(_call(save_refusals=_ref, missing=["accepting"]))[0]["verdict"]
          == "COST",
          "refused and the field never landed -> COST",
          "the guard did not delay the answer, it destroyed it")
    check(_cr.audit_dict(_call(save_refusals=_ref, collected=["accepting"]))[0]["verdict"]
          == "PREMATURE",
          "refused and the field landed later -> PREMATURE",
          "the caller was made to say it again in words the probe recognised")
    check(_cr.audit_dict(_call(save_refusals=_ref)) == [],
          "a refusal on a field the call never needed is not a finding",
          "flagging every refusal is how a scan stops being read")
    check(_cr.audit_dict(_call(missing=["accepting"])) == [],
          "and a call with no refusals is silent")
    # THE PHRASING IS THE DELIVERABLE — the whole point is handing a human the
    # exact string to take to the pattern.
    _f = _cr.audit_dict(_call(save_refusals=_ref, missing=["accepting"]))[0]
    check(_f["heard"] == "Right now no.",
          "the caller's exact words come out with the finding",
          "a refusal without them says a guard fired; with them it says which "
          "phrasing the probe could not read")
    # THREE SOURCES. A guard can refuse at three moments and the artifact
    # records each differently; reading one is how a scan reports "nothing to
    # see" on a call that lost a field.
    check(len(_cr._refusals(_call(
              save_refusals=_ref,
              deferred_saves=[{"tool": "save_branch", "outcome": "contradicted",
                               "why": "x", "args": {"heard": "y"}}],
              branch_rejections=[{"value": "Riverside", "why": "z"}]))) == 3,
          "on-the-spot, contradicted-after-the-wait and branch rejections all "
          "reach the scan")
    check(not _cr._refusals(_call(deferred_saves=[
              {"tool": "save_branch", "outcome": "applied"}])),
          "and a deferred save that APPLIED is not a refusal")
    # AGAINST THE REAL CORPUS: it must find the two gaps fixed by hand today,
    # and nothing else. A scan with false positives is one that gets ignored.
    import pathlib as _pl2
    _real = sorted(_pl2.Path("data/3 cases jsons").glob("call-*.json"))
    _found = [f for _p in _real for f in _cr.audit(_p)]
    _ids = {(f["call_id"][:18], f["field"]) for f in _found}
    check(("call-20260827-1130", "referral") in _ids,
          "the corpus scan finds \"It's depend upon situation\" (COST)",
          f"{sorted(_ids)}")
    check(("call-20260827-1428", "accepting") in _ids,
          'and finds "Right now, no." (PREMATURE)')
    check(len(_found) == 2,
          f"and flags nothing else across {len(_real)} calls",
          f"{[(f['call_id'], f['field']) for f in _found]}")

    # ── A MODULE'S RE-EXPORTED SURFACE MUST BE DECLARED ───────────────────
    # Every private name consumed by another module in this package has to
    # appear in its own module's __all__. Without it the checker reports the
    # module's whole reason for existing as unused: Pylance greys the name AT
    # ITS DEFINITION, and a hint storm is how a real warning gets buried.
    #
    # Found by hand twice on 2026-08-27 — lifecycle.py shipped with no __all__
    # at all, and grounding.py had been missing `_spoken_farewell` since the
    # farewell guard landed. Nothing in this suite could observe either, which
    # is the whole argument for making the process enforce it.
    import ast as _ast, pathlib as _pl, collections as _co
    _pkg = _pl.Path("agents/voice")
    _defined, _imported = {}, _co.defaultdict(set)
    for _p in sorted(_pkg.glob("*.py")):
        _tree = _ast.parse(_p.read_text(encoding="utf-8"))
        _names = set()
        for _n in _tree.body:
            if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                _names.add(_n.name)
            elif isinstance(_n, _ast.Assign):
                _names |= {_t.id for _t in _n.targets if isinstance(_t, _ast.Name)}
            elif isinstance(_n, _ast.AnnAssign) and isinstance(_n.target, _ast.Name):
                _names.add(_n.target.id)
        _defined[_p.stem] = {_x for _x in _names
                             if _x.startswith("_") and not _x.startswith("__")}
        for _n in _ast.walk(_tree):
            if isinstance(_n, _ast.ImportFrom) and (_n.module or "").startswith("agents.voice."):
                _imported[_n.module.split(".")[-1]] |= {_a.name for _a in _n.names}
    _undeclared = {}
    for _mod, _used in sorted(_imported.items()):
        _priv = _used & _defined.get(_mod, set())
        if not _priv:
            continue
        _t = _ast.parse((_pkg / f"{_mod}.py").read_text(encoding="utf-8"))
        _av = next((_n for _n in _t.body if isinstance(_n, _ast.Assign)
                    and any(getattr(_x, "id", "") == "__all__" for _x in _n.targets)), None)
        _listed = set() if _av is None else {_e.value for _e in _av.value.elts
                                             if isinstance(_e, _ast.Constant)}
        if _priv - _listed:
            _undeclared[_mod] = sorted(_priv - _listed)
    check(not _undeclared,
          "every module declares the private names other modules import from it",
          f"undeclared: {_undeclared}" if _undeclared else "")
    # AND THE CHECK ITSELF HAS TO BE ABLE TO FAIL — an absence assertion over a
    # scan that finds nothing passes for free. Prove the scan sees the surface
    # it is guarding.
    check(len(_imported.get("evidence", set())) > 20
          and len(_imported.get("lifecycle", set())) >= 3,
          "and the scan really sees the cross-module surface",
          f"evidence={len(_imported.get('evidence', set()))} "
          f"lifecycle={len(_imported.get('lifecycle', set()))}")

    print("\n" + "-" * 66)
    print("  Cited in source, untested until now")
    print("-" * 66)
    # A scan for call IDs found 58 distinct calls cited across agents/voice and
    # 53 of them carried by a check in here. These are the ones that were not:
    # history with nothing standing behind it, which is the half of "move the
    # trauma logs into the tests" that was straightforwardly right.

    # ── call-20260806-2029: the largest unit wins ─────────────────────────
    # conversation_metrics counts sentence repeats first, then clause repeats
    # NOT already inside a counted sentence. Counting clauses INSTEAD was the
    # first attempt: a short sentence splits into sub-threshold clauses and
    # vanishes, and checked against the whole call history that swap silently
    # dropped a real repeat while fixing three others.
    import agents.voice.metrics as _metrics

    def _reps(_s):
        return _metrics.conversation_metrics(
            [rw.TranscriptTurn(role="agent", text=_s, timestamp="00:00:00"),
             rw.TranscriptTurn(role="agent", text=_s, timestamp="00:00:00")]
        )["repeated_sentences"]
    for _s in ["Okay, no problem, I will.", "Right, so, let me see.",
               "Sure, one moment, hang on."]:
        check(_reps(_s) == 1,
              f"a repeat whose every CLAUSE is sub-threshold still counts: "
              f"{_s!r}",
              "counting clauses instead of sentences loses exactly this shape")
    # ...and the other half of "largest unit wins": one sentence said twice is
    # ONE repetition, not one for the sentence plus one for each of its clauses.
    check(_reps("Could I get the branch name, or the street address please?")
          == 1,
          "a long repeat with long clauses is counted once, not per clause")
    check(_reps("Sure, no rush.") == 0,
          "and three words is below the threshold either way",
          "the threshold is what stops every 'okay' scoring as a repeat")

    # ── call-20260819-1323: the hint came back as a caller turn ───────────
    # "Mercy Hospital" — the FIRST health system in the transcription hint —
    # arrived at audio_rms 0.011 on a call where the callee never spoke at all,
    # and the agent answered it. The hint is prepended to the transcriber's own
    # context, so anything in it can come back out as transcript.
    _echo = lambda _txt, _rms: rw._is_hint_echo(
        rw.TranscriptTurn(role="caller", text=_txt, timestamp="00:00:00",
                          audio_rms=_rms), ["mercy"], _rms)
    check(_echo("Mercy Hospital", 0.011),
          "a bare hint term on near-silent audio is quarantined",
          "call-1323: the callee never spoke and the agent answered anyway")
    # BOTH SIGNALS MUST FAIL, and these are the mutations that say so. Either
    # one alone would throw away a real answer from a quiet caller, or accept a
    # fabrication from a loud one.
    check(not _echo("Mercy Hospital", 0.09),
          "the same words on real audio are a real answer")
    check(not _echo("She's at the Mercy Hospital campus on 5th", 0.011),
          "and a quiet turn that says MORE than the term is not an echo")
    check(not _echo("Mercy Hospital", None),
          "an unmeasured turn gets the benefit of the doubt",
          "absence of measurement is not evidence of fabrication")

    # ── call-20260820-1732: a phantom "Yes." on dead air ──────────────────
    # 0.7s of near-silence produced a whole receptionist greeting. A genuine
    # status answer IS bare — "Yes." is the normal shape — so the usual
    # "did they say more than the term" discriminator is satisfied by every
    # true answer, and the audio measurement is the only signal left.
    def _statsess(_rms):
        _s = rw.RealtimeSession("CA0000000000000000000000sil01",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _templates.get_template("provider_verification").objective
        _s.memory.update(doctor_identity="confirmed")
        _s.add_turn("agent", "Is Dr. Okafor taking new patients?")
        _s.turns.append(rw.TranscriptTurn(role="caller", text="Yes.",
                                          timestamp="00:00:00", audio_rms=_rms))
        return _s
    check(rw._ungrounded_status({"status": "yes", "heard": "Yes."},
                                _statsess(0.0002)),
          "a bare status token on silent audio is refused",
          "this is squarely what the transcriber fabricates")
    check(not rw._ungrounded_status({"status": "yes", "heard": "Yes."},
                                    _statsess(0.09)),
          "the same answer on real speech is accepted")
    check(not rw._ungrounded_status({"status": "yes", "heard": "Yes."},
                                    _statsess(None)),
          "and an unmeasured turn is not refused on an unmeasurement")
    for _n, _t in _templates.TEMPLATES.items():
        _has = _t.objective.field_named("identity") is not None
        check(bool(obj.IDENTITY_ASK.search(_t.instructions)) is _has,
              f"{_n}: raises identity only if it declares the field")

    # Grounding, with selection.
    def _idsess(pairs):
        _s = rw.RealtimeSession("CA00000000000000000000ident",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _PVI
        for _r, _t in pairs:
            _s.turns.append(rw.TranscriptTurn(
                role=_r, text=_t, timestamp="00:00:00",
                audio_rms=0.09 if _r == "caller" else None))
        return _s

    _ia = {"identity": "confirmed", "heard": "a model paraphrase"}
    check(rw._ungrounded_identity(_ia, _idsess([
              ("agent", "Is this Dr. Okafor's office?"),
              ("caller", "Yes, that's us.")])) == "",
          "a real confirmation grounds")
    check(_ia["heard"] == "Yes, that's us.",
          "and heard is selected from the transcript", f"{_ia['heard']!r}")
    check(rw._ungrounded_identity({"identity": "confirmed", "heard": "x"},
                                  _idsess([("agent", "Is this Dr. Okafor's office?"),
                                           ("caller", "It is 1426 7th Street.")])) != "",
          "a confirmation the caller never gave is refused")

    # The greeting asks permission before asking anything else.
    _gp = _templates.PROVIDER_VERIFICATION.build_greeting(
        Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside"),
        org="Forage AI", agent_name="David")
    check("good time" in _gp.lower(),
          "the opener asks whether now is a good time", _gp)
    check(_gp.index("Forage AI") < _gp.lower().index("good time"),
          "after the identification, not before it")
    check("book anything" in _gp,
          "and the not-booking clause is still there")

    # THE EVICTION DID NOT MOVE BRANCH GROUNDING. forage_data_collection is the
    # control — it still carries the full location block and the full doctor
    # block — so a disagreement between the two templates would BE the damage.
    def _bverdict(_objective, _turns, _value):
        _s = rw.RealtimeSession("CA0000000000000000000ctrl2",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        _s.objective = _objective
        for _t in _turns:
            _s.turns.append(rw.TranscriptTurn(role="caller", text=_t,
                                              timestamp="0", audio_rms=0.09))
        return (rw._ungrounded_terms({"branch": _value}, _s) == "",
                tuple(rw._rode_along({"branch": _value}, _s)))

    for _turns, _value, _why in [
        (["She's at the Riverside campus."], "Riverside Campus", "clean save"),
        (["He works at the east side clinic."], "Eastside Clinic", "space collapse"),
        (["She resides at campus"], "Riverside Campus", "ASR mangled"),
        (["Hello. Okay, next slide, please."], "Riverside Clinic", "fabrication"),
        (["it's 1825 4th street"], "Mission Bay Clinic, 1855 Fourth Street",
         "invented house number"),
        (["office Abadan branch"], "Northside Branch", "reshaped hospital name"),
        (["Yeah, just a moment."], "Downtown", "nothing said"),
    ]:
        check(_bverdict(_templates.FORAGE_DATA_COLLECTION.objective, _turns, _value)
              == _bverdict(_PVI, _turns, _value),
              f"eviction left branch grounding unmoved: {_why}",
              "forage_data_collection still has the evicted blocks, so a "
              "disagreement here would be the damage")

    print("\n" + "=" * 66)
    print("  call-20260825-1226 — identity confirmed the wrong doctor")
    print("=" * 66)
    # The record said Dr. Okafor. The caller said "that's right, Dr. Kapoor is
    # one of our cardiologists." identity saved CONFIRMED, because the guard
    # classified the affirmative and never looked at the name. Okafor and
    # Kapoor are not the same person, and this field exists to answer exactly
    # that question — the two-John-Smiths case it was built for is the same
    # shape, and a check that accepts Kapoor for Okafor cannot separate those.
    def _nm(caller, doctor="Dr. Jane Okafor"):
        _s = rw.RealtimeSession("CA00000000000000000000name",
                                Doctor(doctor_name=doctor,
                                       hospital_name="Northside Medical Group",
                                       specialization="Cardiology"))
        _s.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
        _s.turns.append(rw.TranscriptTurn(
            role="agent", timestamp="0",
            text="Is this Dr. Okafor, Cardiology, at Northside Medical Group?"))
        _s.turns.append(rw.TranscriptTurn(role="caller", text=caller,
                                          timestamp="0", audio_rms=0.09))
        return _s

    _live = "Yeah, this is, yeah, that's right, Dr. Kapoor is one of our cardiologists."
    check(obj.classify_identity(_live) is obj.IdentityAnswer.CONFIRMED,
          "the affirmative alone still classifies as confirmed",
          "which is why the vocabulary could never have caught this")
    check(rw._ungrounded_identity({"identity": "confirmed", "heard": "x"},
                                  _nm(_live)) != "",
          "but confirming is now REFUSED — they named a different doctor",
          "the highest-priority defect: a row confirmed against the wrong "
          "person, attached to a real practice")
    check("kapoor" in rw._ungrounded_identity(
              {"identity": "confirmed", "heard": "x"}, _nm(_live)).lower(),
          "and the refusal names the surname it heard")
    for _c, _refuse, _why in [
        ("Yes, that's Dr. Okafor's office.", False, "our doctor, possessive"),
        ("Yes, Dr Okafor works here.", False, "no full stop after Dr"),
        ("Yes, speaking.", False, "nobody named — silence is not a mismatch"),
        ("That's right.", False, "nobody named"),
        ("Yes, we have a Dr. Smith here.", True, "a different doctor"),
        ("Yes, Doctor Kapoor is here.", True, "spelled-out title"),
    ]:
        _v = rw._ungrounded_identity({"identity": "confirmed", "heard": "x"},
                                     _nm(_c))
        check(bool(_v) is _refuse,
              f"{'refuse' if _refuse else 'accept'}: {_c[:40]!r}", _why)
    # The check is on CONFIRMING. A different doctor being named is evidence
    # FOR not_here, not against it.
    check(rw._surnames_named("Dr. Kapoor is one of our cardiologists") == ["kapoor"],
          "the surname extractor reads the name off the title")
    check(rw._surnames_named("She's at the Riverside campus.") == [],
          "and finds none where none is claimed")
    # A POSSESSIVE IS A SUFFIX, NOT A CHARACTER SET. `.rstrip("'s")` reads like
    # it removes "'s" and removes every trailing apostrophe and s instead, so
    # "Reyes" came back "reye". Live on call-20260825-1625: the caller said
    # "Dr. Reyes is an oncologist" — right doctor, clean transcript — and the
    # guard answered "they named 'reye', and the doctor on this call is
    # 'reyes'", refusing a correct confirmation and spending a turn spelling a
    # name nobody had got wrong. Every surname ending in s: Reyes, Jones,
    # Hayes, Brooks, Sanders. The fixtures above are Okafor, Kapoor and Smith,
    # which is why none of them could show it.
    for _txt, _want, _why in [
        ("Dr. Reyes is an oncologist.", ["reyes"], "the s is part of the name"),
        ("Yes, that's Dr. Okafor's office.", ["okafor"], "a real possessive"),
        ("Dr. Jones' office is upstairs.", ["jones"], "a bare trailing "
         "apostrophe is still a possessive"),
        ("Dr. Hayes has left.", ["hayes"], "and again with a different name"),
    ]:
        check(rw._surnames_named(_txt) == _want,
              f"{_txt[:34]!r} -> {_want}", f"{rw._surnames_named(_txt)}: {_why}")
    # End to end: the name the guard was built to protect no longer accuses
    # itself.
    check(rw._ungrounded_identity(
              {"identity": "confirmed",
               "heard": "Yes, Dr. Reyes is our oncologist."},
              _nm("Yes, Dr. Reyes is our oncologist.",
                  doctor="Dr. Alan Reyes")) == "",
          "an s-final surname confirms against itself",
          "the guard refused our own doctor on a clean transcript, which is a "
          "false accusation and costs a turn spelling a correct name")

    print("\n" + "-" * 66)
    print("  call-20260825-1620 - the guard asked before the evidence existed")
    print("-" * 66)
    # THE MODEL HEARS AUDIO; THE GUARDS READ TRANSCRIPTS; THE TRANSCRIPT LAGS.
    # All three of these lines are inside the same second of that call's log:
    #
    #   16:21:37  'eastside' never appeared in the caller transcript
    #   16:21:37  HALLUCINATED BRANCH BLOCKED: {'branch': 'Eastside Clinic'}
    #   16:21:37  CALLER : He's at the Eastside clinic.
    #
    # The first answer was a blocking wait, and these checks used to exercise
    # it. THE MEASUREMENT THAT WAIT ASKED FOR KILLED IT - across 119 call
    # artifacts it ran 14 times, timed out 12 and landed 0, never once doing
    # its job while costing 1.5s a time (the whole of `ours 1.53s` in the 3.44s
    # reply on call-20260827-1010). What is left is the predicate both it and
    # the deferral asked, and the deferral itself, which is tested above
    # against the real 0942 races.
    def _pend(turns, answered=False):
        _s = rw.RealtimeSession("CA0000000000000000000000000tw01",
                                Doctor(doctor_name="Dr. Alan Reyes",
                                       hospital_name="Lakeview Medical"))
        for _r, _t in turns:
            _s.add_turn(_r, _t)
        _s._placeholder_at = time.monotonic()
        # The transcriber answering AFTER the placeholder is what "still in
        # flight" is the absence of.
        _s._transcript_at = _s._placeholder_at + (0.01 if answered else -0.01)
        return _s

    check(rw._transcript_pending(
              _pend([("agent", "which branch is he working out of?"),
                     ("caller", "[...]")])) is True,
          "words still in flight are recognised as still in flight",
          "the 1620 case: the guard is about to judge a turn the transcriber "
          "has not answered for")
    check(rw._transcript_pending(
              _pend([("caller", "He's at the Eastside clinic.")])) is False,
          "a turn that has landed is not pending")
    # call-20260825-1712: the transcriber replied and the reply was junk. The
    # placeholder stays standing and looks identical from outside - this is the
    # condition the old wait burned its ceiling on, twice.
    check(rw._transcript_pending(
              _pend([("caller", "[...]")], answered=True)) is False,
          "a placeholder the transcriber has ALREADY answered for is not "
          "pending - the evidence exists and is nothing")
    check(rw._transcript_pending(_pend([("agent", "Take care.")])) is False,
          "and an agent turn on the end is not a caller turn in flight")

    # -- ESCALATE: THE ONE TOOL THAT CANNOT DEFER --------------------------
    # Five of the six tools the deleted wait covered are saves, and a save
    # whose guard objects mid-flight is HELD. escalate ends the call instead,
    # so nothing catches it later: _discarded_location would read a transcript
    # that has not caught up and the answer the caller just gave would be
    # invisible to the guard whose whole job is noticing it.
    class _EscWS:
        def __init__(self): self.sent = []
        async def send(self, s): self.sent.append(json.loads(s))

    async def _try_escalate(sess, ws):
        await rw._handle_tool_call(
            {"call_id": "e1", "name": "escalate",
             "arguments": json.dumps({"reason": "caller engaged but never "
                                                "provided a location"})},
            sess, ws, {}, False)

    _esc = _pend([("agent", "which branch is he working out of?"),
                  ("caller", "[...]")])
    _esc._unanswered_asks = 4
    _ews = _EscWS()
    await _try_escalate(_esc, _ews)
    # The result rides INSIDE a conversation.item.create, not as a top-level
    # message — reading the outer type finds nothing and the check passes for
    # the wrong reason on a guard that never fired.
    _out = [m["item"] for m in _ews.sent
            if m.get("item", {}).get("type") == "function_call_output"]
    check(bool(_out) and json.loads(_out[-1]["output"]).get("ok") is False,
          "escalating while their last turn is still transcribing is refused",
          f"{_out and _out[-1]['output']}")
    check(any("still transcribing" in json.dumps(m) for m in _ews.sent),
          "and the model is told to wait for the words, not to ask again")
    check(_esc._unanswered_asks == 4,
          "the ask budget is NOT reset - it ran out for its own reasons and "
          "this says nothing about whether they were engaging",
          f"unanswered={_esc._unanswered_asks}")
    # ONE-SHOT, or a stalled transcriber is a call that cannot be ended.
    _ews2 = _EscWS()
    await _try_escalate(_esc, _ews2)
    check(not any("still transcribing" in json.dumps(m) for m in _ews2.sent),
          "the hold is one-shot; the second attempt is allowed through",
          "a guard that can refuse forever cannot end a call")

    print("")
    print("-" * 66)
    print("  calls 1433/1437 — spell the name, because matching cannot")
    print("-" * 66)
    # Same guard, the opposite cause. On 1226 the receptionist really did name
    # somebody else. On 1433 and 1437 the RECEPTIONIST WAS RIGHT and the line
    # was wrong: the record said Reyes and the transcript produced "Dr. Riaz",
    # "Dr. Yes" and "Dr. Ayers" across the two calls. Both ended unconfirmed.
    #
    # No string comparison rescues this and it is worth saying why, because the
    # obvious fix is to reach for one. Soundex and metaphone match Riaz to
    # Reyes and miss Ayers entirely; edit distance misses all three; and every
    # threshold loose enough to admit Ayers also admits Kapoor for Okafor,
    # which is the 1226 defect being reintroduced to fix 1437. The two calls
    # want opposite tolerances from one comparison.
    #
    # So the repair is not a comparison. Spell the name a letter at a time and
    # ask them to confirm the letters — a channel the transcriber cannot mangle
    # the same way, and one a receptionist can answer plainly.
    check(rw._spell_out("Reyes") == "R-E-Y-E-S",
          "the name is offered one letter at a time")
    for _garble in ["Yeah, Dr. Riaz is here.", "That's Dr. Yes.",
                    "Yes, Dr. Ayers speaking."]:
        _v = rw._ungrounded_identity({"identity": "confirmed", "heard": "x"},
                                     _nm(_garble, doctor="Dr. Ana Reyes"))
        check("R-E-Y-E-S" in _v,
              f"the refusal of {_garble[:26]!r} asks for the letters",
              "refusing without a repair is how both calls spent their whole "
              "budget on one question and hung up with nothing")

    # The repair only counts when the agent ACTUALLY spells it. Set where the
    # agent's turn is recorded, not where the rejection is written — a model
    # that ignores the instruction must not advance the scan, or the next
    # "yes" confirms against a name nobody ever heard.
    check(rw._spelled_out("It's R-E-Y-E-S, Reyes.", "reyes")
          and rw._spelled_out("spelled R E Y E S", "reyes")
          and not rw._spelled_out("Dr. Reyes, cardiology", "reyes"),
          "spelling it out is recognised, and saying it plainly is not",
          "the plain name is the sound that was already mangled — if that "
          "counted as spelling, the repair would be satisfied by the failure")

    # AND THEN THE CONFIRMATION LANDS. This is the whole point: the mangled
    # surname from earlier in the call must stop refusing an answer the caller
    # has now given against our actual letters.
    _sp = _nm("Yeah, Dr. Riaz is here.", doctor="Dr. Ana Reyes")
    _sp.turns.append(rw.TranscriptTurn(
        role="agent", timestamp="0",
        text="Let me spell it — R-E-Y-E-S. Is that the name you have?"))
    _sp._name_spelled_at = len(_sp.turns)
    _sp.turns.append(rw.TranscriptTurn(role="caller", timestamp="0",
                                       audio_rms=0.09,
                                       text="Oh, Reyes, yes, that's her."))
    check(rw._ungrounded_identity({"identity": "confirmed",
                                   "heard": "Oh, Reyes, yes, that's her."},
                                  _sp) == "",
          "after the letters, their yes confirms",
          "scanning the whole call finds the mangled turn again and refuses "
          "forever — that is the loop 1437 died in")
    # The refusal is not simply switched off. A DIFFERENT doctor named after
    # the letters is the practice answering, not the line.
    _sp2 = _nm("Yeah, sure.", doctor="Dr. Ana Reyes")
    _sp2._name_spelled_at = len(_sp2.turns)
    _sp2.turns.append(rw.TranscriptTurn(
        role="caller", timestamp="0", audio_rms=0.09,
        text="No, we've got a Dr. Whitfield, no Reyes here."))
    _v2 = rw._ungrounded_identity({"identity": "confirmed", "heard": "x"}, _sp2)
    check(_v2 != "" and "not_here" in _v2,
          "and a different name AFTER the letters is refused and routed to "
          "not_here",
          "asking a third time spends the call on a question already "
          "answered; not_here is the valuable negative result")

    # ── The stored quote must be the turn the answer actually rests on ──────
    # call-20260825-1620 saved identity=confirmed with
    #
    #   heard: "Yes, Dr. Rayaz is our oncologist."
    #
    # which is the turn the guard REFUSED, three turns before the agent spelled
    # R-E-Y-E-S and the caller said "Yes, the same doctor." The confirmation
    # rests entirely on that later turn; the stored quote names the doctor this
    # call had just decided was the wrong one. `heard` is what a reviewer reads
    # as the reason the row says what it says, so this is a row that documents
    # itself with its own counter-evidence.
    #
    # Selection picks the fullest MATCHING turn, and the mangled one is simply
    # the longer sentence. Nothing was wrong with the choice; the window it
    # chose from reached back past the point where the question was settled.
    _pv = _nm("Yes, Dr. Rayaz is our oncologist.", doctor="Dr. Ana Reyes")
    _pv.turns.append(rw.TranscriptTurn(
        role="agent", timestamp="0",
        text="The name we have is R-E-Y-E-S - is that the same doctor?"))
    _pv._name_spelled_at = len(_pv.turns)
    _pv.turns.append(rw.TranscriptTurn(role="caller", timestamp="0",
                                       audio_rms=0.09,
                                       text="Yes, the same doctor."))
    _pa = {"identity": "confirmed", "heard": "Yes, Dr. Reyes is our oncologist."}
    check(rw._ungrounded_identity(_pa, _pv) == "",
          "after the letters the confirmation stands")
    check(_pa["heard"] == "Yes, the same doctor.",
          f"and the stored quote is the turn it rests on ({_pa['heard']!r})",
          "the longer pre-spelling turn names the doctor this call rejected - "
          "storing it makes the row cite its own counter-evidence")

    # THE FLOOR MUST NOT BECOME A WAY TO ACCEPT WITH NO EVIDENCE. Spelling the
    # name and claiming confirmed before they answer is refused, not stood
    # down on: the permissive "nothing transcribed since we asked" branch is
    # the wrong verdict here, because we put a specific question to them and
    # the only turns before the floor are the ones the spelling superseded.
    _pv2 = _nm("Yes, Dr. Rayaz is our oncologist.", doctor="Dr. Ana Reyes")
    _pv2.turns.append(rw.TranscriptTurn(
        role="agent", timestamp="0",
        text="The name we have is R-E-Y-E-S - is that the same doctor?"))
    _pv2._name_spelled_at = len(_pv2.turns)
    _v3 = rw._ungrounded_identity({"identity": "confirmed", "heard": "x"}, _pv2)
    check(_v3 != "" and "spelled the name out" in _v3,
          f"confirming before they answer the letters is refused ({_v3[:60]!r})",
          "standing down would accept a confirmation whose only support is the "
          "turn the spelling was performed to supersede")

    # And a call where the name was never spelled is untouched by any of this -
    # the floor is 0, so the anchor is exactly where it always was.
    _pv3 = _nm("Yes, that's Dr. Okafor's office.")
    _pa3 = {"identity": "confirmed", "heard": "invented"}
    check(rw._ungrounded_identity(_pa3, _pv3) == ""
          and _pa3["heard"] == "Yes, that's Dr. Okafor's office.",
          f"an unspelled call selects as before ({_pa3['heard']!r})",
          "the floor must be inert on every call that never needed it")

    # ── call-20260825-1712: the ask the probe could not see ─────────────────
    # The agent asked "Great - are you able to confirm THIS IS Dr. Reyes,
    # Oncology, at Lakeview Medical?" and IDENTITY_ASK matched nothing, because
    # it only knew the interrogative order ("is this Dr."). With no ask to
    # anchor on, `since` stayed 0, the never-asked branch applied, and
    #
    #   caller: "Yes, Dr. Reyes is our oncologist."
    #
    # was REFUSED - twice - for not standing on its own once the leading yes is
    # stripped. Three turns later the model reworded to "Is this Dr. Reyes'
    # line...", which did match, and a bare "Yes," was accepted immediately.
    # The guard threw away the good evidence and took the weakest available,
    # which is the exact inversion of what it is for.
    for _phrasing in [
        "Great - are you able to confirm this is Dr. Reyes, Oncology, at "
        "Lakeview Medical?",
        "Can you confirm this is Dr. Reyes, Oncology, at Lakeview Medical?",
        "Am I through to Dr. Reyes at Lakeview Medical?",
    ]:
        _as = rw.RealtimeSession("CA0000000000000000000000000as01",
                                 Doctor(doctor_name="Dr. Alan Reyes",
                                        hospital_name="Lakeview Medical",
                                        specialization="Oncology"))
        _as.objective = _templates.PROVIDER_VERIFICATION_OBJECTIVE
        _as.add_turn("caller", "Yeah, it's a good time.")
        _as.add_turn("agent", _phrasing)
        _as.turns.append(rw.TranscriptTurn(
            role="caller", text="Yes, Dr. Reyes is our oncologist.",
            timestamp="0", audio_rms=0.09))
        _aa = {"identity": "confirmed", "heard": "Yes, Dr. Reyes is our oncologist."}
        check(rw._ungrounded_identity(_aa, _as) == "",
              f"the ask is recognised: {_phrasing[:44]!r}",
              "unrecognised, the answer is judged as if nobody had asked - and "
              "a proper answer is exactly what fails that test")
        check(_aa["heard"] == "Yes, Dr. Reyes is our oncologist.",
              f"and the quote is their real answer ({_aa['heard']!r})",
              "anchored at 0, an earlier unrelated turn is in the window too")

    # The probe stays NARROW, which is the constraint the pattern was written
    # under: office/practice/clinic are LOCATION_NOUN, so a loose identity
    # probe would match every branch ask and anchor the identity guard on the
    # wrong turn. Named doctor required; the agent's own greeting excluded.
    for _not_identity in [
        "Which branch does Dr. Reyes work out of?",
        "Thanks - which branch does Dr. Reyes work out of?",
        "Do you know which branch Dr. Okafor works out of?",
        "Is she taking new patients?",
        "Is there a waitlist or another way to book?",
        "Hi, this is David, calling on behalf of Forage AI to verify some "
        "information that was missed on our website.",
    ]:
        check(not rw._is_ask_for(_not_identity, obj.IDENTITY_ASK),
              f"still not an identity ask: {_not_identity[:42]!r}",
              "a probe that matches a branch ask anchors the identity guard on "
              "the wrong turn, which is what the narrowness is protecting")

    # VISIBLE. Both calls ended unconfirmed with nothing anywhere saying the
    # name was why — the only trace was a memory key nothing read and no
    # artifact carried. "Never asked" and "asked, and the name came back wrong
    # three times" are different results and looked identical.
    _vis = _nm("Yeah, Dr. Riaz is here.", doctor="Dr. Ana Reyes")
    rw._ungrounded_identity({"identity": "confirmed", "heard": "x"}, _vis)
    rw._ungrounded_identity({"identity": "confirmed", "heard": "x"}, _vis)
    check(_vis.name_mismatches == [{"heard": "riaz", "ours": "reyes",
                                    "said": "Yeah, Dr. Riaz is here.",
                                    "after_spelling": False}],
          f"the wrong name is recorded once, with ours beside it "
          f"({_vis.name_mismatches})",
          "a guard that refuses invisibly cannot be reviewed after the call")

    print("\n" + "-" * 66)
    print("  A number said as a word is still that number")
    print("-" * 66)
    # The caller said "Riverside Campus Seventh Street" twice. The model wrote
    # "7th". The digit rule said "number 7 not in what the caller said" and
    # refused three times; the branch that finally saved was a bare "Riverside"
    # with the campus and the street both lost. The map already knew
    # seventh -> 7; only the caller's side was not consulting it.
    def _dg(turns):
        _s = rw.RealtimeSession("CA00000000000000000000dgt",
                                Doctor(doctor_name="Dr. Jane Okafor",
                                       hospital_name="Northside Medical Group"))
        for _t in turns:
            _s.turns.append(rw.TranscriptTurn(role="caller", text=_t,
                                              timestamp="0", audio_rms=0.09))
        return _s

    _spoken = _dg(["Yes, Riverside Campus Seventh Street.", "Yes, Seventh Street."])
    check(rw._ungrounded_terms({"branch": "Riverside Campus, 7th Street"},
                               _spoken) == "",
          "'7th' grounds on a caller who said 'Seventh'",
          "normalisation, not tolerance — the same number, a different notation")
    check(rw._ungrounded_terms({"branch": "Riverside Campus Seventh Street"},
                               _spoken) == "",
          "and so does the spelled form, as it always did")
    # THE ZERO-TOLERANCE RULE MUST NOT WEAKEN. This is the guard that exists
    # because a house number nobody said reached the directory.
    _addr = _dg(["it's 1825 4th street"])
    check(rw._ungrounded_terms(
              {"branch": "Mission Bay Clinic, 1855 Fourth Street"}, _addr) != "",
          "an invented house number is still refused")
    check(rw._ungrounded_terms(
              {"branch": "Mission Bay Clinic, 1825 4th Street"}, _addr) == "",
          "and the correct one still grounds")
    check(rw._ungrounded_terms({"branch": "Ninth Street Clinic, 9th Street"},
                               _dg(["it is on Seventh Street"])) != "",
          "a DIFFERENT number said as a word is still caught",
          "'9th' does not ground on 'Seventh' — the map normalises, it does "
          "not blur")

    print("\n" + "-" * 66)
    print("  cardiology / cardiologists — one fact, two parts of speech")
    print("-" * 66)
    _cs_sess = _nm("that's right, Dr. Okafor is one of our cardiologists.")
    _ca = {"detail": "cardiology"}
    _cd, _ = rw._strip_ungrounded_detail(_ca, _cs_sess, "detail")
    check(list(_cd) == [] and _ca["detail"] == "cardiology",
          "the specialty survives when they said the practitioner form",
          "suffix-stripping cannot turn cardiologists into cardiology; a "
          "shared six-character prefix can")
    check(rw._grounded_loosely("cardiology", "one of our cardiologists"),
          "prefix matching covers the family")
    check(not rw._grounded_loosely("downtown", "please download the form"),
          "and does not reach across unrelated words",
          "they part company inside the first six characters")
    # It stays a QUALIFIER-only loosening. Branch grounding is untouched.
    check(rw._ungrounded_terms({"branch": "Cardiology Campus"},
                               _dg(["she is one of our cardiologists"])) != "",
          "a branch is NOT grounded by prefix — that field keeps exact matching",
          "a loosening there costs a wrong address, which is a different price")

    print("\n" + "=" * 66)
    print("  FAILED" if FAILURES else "  ALL PASSED")
    print("=" * 66 + "\n")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
