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
import json
import pathlib
import re
import sys
import tempfile
import time
import types
from unittest import mock

import core.bootstrap  # noqa: F401  (UTF-8 stdout on Windows)

# soundfile is only needed to write the WAV; stub it so this runs anywhere.
if "soundfile" not in sys.modules:
    try:
        import soundfile  # noqa: F401
    except ImportError:
        _sf = types.ModuleType("soundfile")
        _sf.write = lambda *a, **k: None
        sys.modules["soundfile"] = _sf

from core.models import Doctor
import agents.voice.realtime_worker as rw


# ── Fakes ─────────────────────────────────────────────────────────────────────

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


async def run_call(script, out=None, connect_failures=0):
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
         mock.patch.dict("sys.modules",
                         {"twilio.rest": types.SimpleNamespace(Client=_NoTwilio)}), \
         mock.patch.object(rw.RealtimeSession, "__init__", spy_init):
        doctor = Doctor(doctor_name="Dr. Jane Okafor",
                        hospital_name="Northside Medical Group",
                        specialization="Cardiology")
        await asyncio.wait_for(
            rw.handle_realtime(twilio, "CA000000000000000000000000testsid", doctor),
            timeout=30,
        )
    return sent, captured.get("session")


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
                    _claimed_done_nudged=False, memory={},
                    _caller_speaking_since=None, agent_speaking=False,
                    _backchannel_done_this_utterance=False,
                    _last_backchannel_at=0.0, _last_backchannel_clip=None,
                    _backchannels_sent=0, stream_sid="MZtest",
                    listen_enabled=asyncio.Event())
        base.update(over)
        s = types.SimpleNamespace(**base)
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
        _wd = asyncio.create_task(rw._silence_watchdog(_ws, _sess, _done))
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
    # And it must stand down the moment they speak.
    _sess2 = _wd_sess(_agent_quiet_since=None)
    _ws2, _done2 = _WS(), asyncio.Event()
    with mock.patch.object(rw, "_SILENCE_PROMPT_AFTER", 0.0):
        _wd2 = asyncio.create_task(rw._silence_watchdog(_ws2, _sess2, _done2))
        await asyncio.sleep(1.2)
        _done2.set()
        await asyncio.wait_for(_wd2, timeout=2)
    check(not _ws2.sent, "no prompt while the caller has the turn", len(_ws2.sent))

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
          "no response.create while one is already generating", len(_ws3.sent))

    # Two thresholds. Mid-conversation a pause is someone thinking, and seven
    # seconds of thinking room is right. Straight after the opening line there is
    # nothing to think about, and seven seconds of dead air on a cold call is
    # when people hang up.
    check(rw._SILENCE_PROMPT_FIRST < rw._SILENCE_PROMPT_AFTER,
          "first silence is given less rope than a mid-call one",
          f"{rw._SILENCE_PROMPT_FIRST}s vs {rw._SILENCE_PROMPT_AFTER}s")
    # The cap is for the CALL. Resetting it whenever the caller spoke meant a
    # callee who says "hello?" and nothing else could be prompted forever.
    _src = pathlib.Path("agents/voice/realtime_worker.py").read_text(encoding="utf-8")
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
    _sp_vals = [re.search(r"=\s*([^\n=]+)$", l).group(1).strip().rstrip(",)")
                for l in _sp_lines]
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
    # Positive control: the proper nouns a hint legitimately exists for are
    # still there, so gutting the hint fails rather than passing this section.
    for _noun in ("Kaiser Permanente", "Cleveland Clinic", "campus",
                  "medical center", "boulevard"):
        check(_noun.lower() in _hint.lower(),
              f"hint still primes the vocabulary it is for: {_noun!r}")

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
        _got = bool(rw._ungrounded_terms(_args, types.SimpleNamespace(turns=_turns)))
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
    _mkr = lambda r, x: types.SimpleNamespace(role=r, text=x)
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
    class _S:  # minimal session: only _caller_pcm is read
        def __init__(self, chunks): self._caller_pcm = chunks
    # Speech at 1000-3000ms, then silence. The old code, marking the position
    # late, would have taken the tail.
    _sess_sl = _S([_silence[:int(1000*_bpms)], _loud, _silence])
    _cut = rw._utterance_slice(_sess_sl, 1000, 3000, fallback_chunk_pos=2)
    _rms_cut = rw._loudest_window_rms(rw._wire_to_pcm16(_cut))
    _tail = b"".join(_sess_sl._caller_pcm[2:])
    _rms_tail = rw._loudest_window_rms(rw._wire_to_pcm16(_tail))
    check(_rms_cut > 10 * max(_rms_tail, 1e-9),
          "OpenAI's timestamps cut the SPEECH, not the silence after it",
          f"timestamped {_rms_cut:.4f} vs arrival-time {_rms_tail:.6f}")
    # Missing or nonsensical timestamps fall back rather than measuring nothing.
    check(rw._utterance_slice(_sess_sl, None, None, 0) == b"".join(_sess_sl._caller_pcm),
          "no timestamps -> fall back to the chunk position")
    check(rw._utterance_slice(_sess_sl, 99_000, 99_500, 0) == b"".join(_sess_sl._caller_pcm),
          "out-of-range timestamps fall back rather than slicing nothing")

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
                        _plb.Path(rw.__file__).read_text(encoding="utf-8"), re.S)
    check(_ct_src and "_audio_carried_nothing" in _ct_src.group(0)
          and "_reads_as_hint_vocabulary" in _ct_src.group(0),
          "the drop requires both signals together")

    # ── A silent drop must leave a trace ─────────────────────────────────────
    # Two turns were dropped on call-20260819-2006 and the artifact recorded
    # nothing — the only evidence was a terminal that happened to still be open.
    check("suppressed_echoes" in _plb.Path(rw.__file__).read_text(encoding="utf-8")
          .split("\"grounding\":")[1][:600],
          "dropped turns are written into the call artifact")

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
    _rw_src2 = _plb.Path(rw.__file__).read_text(encoding="utf-8")
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
    _mk_c = lambda t: types.SimpleNamespace(role="caller", text=t, audio_rms=0.18)
    _addr1847 = types.SimpleNamespace(turns=[
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
    _no_addr = types.SimpleNamespace(turns=[
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
        check(rw._address_offered(types.SimpleNamespace(turns=[_mk_c(_t)])) is None,
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
    _hf = lambda t: types.SimpleNamespace(role="caller", text=t, audio_rms=0.15)
    _addr_sess = types.SimpleNamespace(turns=[
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
                         _plb.Path(rw.__file__).read_text(encoding="utf-8"), re.S)
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
                         _plb.Path(rw.__file__).read_text(encoding="utf-8"), re.S)
    check(_ht_body and "_claimed_done_at = time.time()" in _ht_body.group(0),
          "the transcript handler only records the claim, it does not correct it")
    check(_wd_body and "_claimed_done_nudged" in _wd_body.group(0),
          "the watchdog makes the correction, once the state has settled")
    check("sess._claimed_done_at = 0.0" in _plb.Path(rw.__file__).read_text(encoding="utf-8"),
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
        _raw = _b64.b64decode(_a)
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
                        _plb.Path(rw.__file__).read_text(encoding="utf-8"), re.S)
    check(_wd_src and "add_turn" not in _wd_src.group(0),
          "and the watchdog that sends it does not record one either")
    # Guard rails on when it may fire.
    check(rw._BACKCHANNEL_AFTER_S >= 2.0,
          "it waits until the caller is genuinely mid-utterance",
          f"{rw._BACKCHANNEL_AFTER_S}s")
    check(rw._BACKCHANNEL_COOLDOWN_S > rw._BACKCHANNEL_AFTER_S,
          "and cannot fire twice in quick succession — that is a tic",
          f"cooldown {rw._BACKCHANNEL_COOLDOWN_S}s")
    # OFF by default and gated in the watchdog, not merely by absent clips.
    # It has never run on a phone line, and there is a specific untested
    # interaction: a callee on speakerphone may have their mic pick the
    # backchannel back up, and with realtime_echo_gate="pass" nothing would
    # suppress it — the agent could transcribe its own noise as caller speech.
    # Off so the next call tests ONE new thing rather than two.
    check(settings.realtime_backchannels is False,
          "backchannels default OFF — untested on a live line")
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
    _sess_lv = types.SimpleNamespace(turns=[
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
    _thin = types.SimpleNamespace(turns=[
        _TT(role="caller", text="Mercy", audio_rms=0.05)])
    check(rw._caller_speech_level(_thin) is None,
          "no adaptive level until there are enough measured turns",
          f"{rw._MIN_TURNS_FOR_ADAPTIVE} needed")
    # The absolute stays a FLOOR: on a uniformly quiet call a fraction of a
    # small number must not drive the threshold toward zero.
    _quiet = types.SimpleNamespace(turns=[
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
    # The wire format must match on both legs, or the session speaks one codec
    # and Twilio hears another — silence on a connected call.
    check(session["audio"]["input"]["format"] == session["audio"]["output"]["format"],
          "input and output audio formats agree")
    expected_fmt = ("audio/pcmu" if settings.realtime_audio_format == "pcmu"
                    else "audio/pcm")
    check(session["audio"]["input"]["format"]["type"] == expected_fmt,
          f"audio format is {expected_fmt}")
    check(session["audio"]["input"]["transcription"]["language"] == "en",
          "transcription pinned to en")
    check(session.get("max_output_tokens") == settings.realtime_max_response_tokens,
          "response token cap set")

    ctx = items[0]["item"]["content"][0]["text"]
    check("CALL CONTEXT" in ctx, "per-call facts sent as a conversation item")
    check("Dr. Jane Okafor" in ctx, "context names the doctor")
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
        check(_probe.hospital_name not in _g,
              f"{_name}: opener does not spend itself confirming the hospital",
              _g[:52])
        check(_g.rstrip().endswith("?"),
              f"{_name}: opener ends on the ask, handing over the turn", _g[-40:])
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
        check(any(s in _low5 for s in ("do you know", "any chance",
                                       "would you know", "i'm hoping",
                                       "could you tell")),
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
        check("EMERGENCY" in _f3 and "nothing urgent" in _f3,
              f"{_n3}: prompt answers the is-this-an-emergency question")
        # It opened that same turn with "It's just me, calling on behalf of..."
        # — a phrase that identifies nobody, from a stranger on their phone.
        check("it's just me" in _f3.lower(),
              f"{_n3}: prompt bans the identifies-nobody non-answer")
        # "Need anything?" — an open offer, answered with nothing.
        check("They OFFER to help" in _f3,
              f"{_n3}: prompt takes an offer of help instead of returning it")
        # "Can you share those details with me?" — answered by reciting what was
        # on the record, to someone unverified.
        check("collect this information, not" in _f3,
              f"{_n3}: prompt refuses to read the record back to the caller")
        # And the rejection-handling rule that lost the branch.
        check("RE-READ WHAT THEY ACTUALLY SAID" in _f3,
              f"{_n3}: prompt re-reads the transcript before re-asking")
    check("Never claim to have noted, saved, or recorded a location you were" in flat,
          "cannot claim to have saved a location it never got")
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
    _worker_src = pathlib.Path("agents/voice/realtime_worker.py").read_text(
        encoding="utf-8")
    # The first word may be short. '[a-z]{4,}' silently skipped every directive
    # opening with a three-letter word, which was both of the ones beginning
    # "you ..." — including the ask-budget directive that ENDS the call. The
    # test reported 5 paths while the module had 7, so the point of deriving
    # them from source (catch a new path the day it lands) was defeated by the
    # pattern used to derive them.
    _injected = set(re.findall(r'"\(system: ([a-z][a-z ]{9,})', _worker_src))
    _declared = _worker_src.count('"(system: ')
    check(len(_injected) == _declared,
          "every injected directive is found, none skipped by the pattern",
          f"{len(_injected)} found vs {_declared} in source")
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

    probe = Doctor(doctor_name="Dr. Jane Okafor",
                   hospital_name="Northside Medical Group")
    disclosed = TEMPLATES["forage_ai_disclosed"]
    # The claim is that automation is DISCLOSED, not that one particular phrase
    # is used. Asserting on "automated assistant" pinned the wording, and the
    # wording is now banned outright — the agent must not describe itself in
    # assistant register anywhere. Check the disclosure, not the phrasing.
    _dg = disclosed.build_greeting(probe)
    check("automated" in _dg.lower(),
          "forage_ai_disclosed announces automation upfront", _dg[:56])
    check("automated assistant" not in _dg.lower(),
          "forage_ai_disclosed discloses without the assistant register", _dg[:56])
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
    for _name, _val in vars(rw).items():
        if isinstance(_val, _re.Pattern):
            check(not any(ord(c) < 32 and c not in "\t\n" for c in _val.pattern),
                  f"no control characters in {_name}")
    # Plain string constants too, not just regexes. A 0x01 sentinel landed in
    # _ABBREV_MARK and the regex-only guard could not see it — the third control
    # byte to reach this file. Read renders them invisibly, so nothing catches
    # these by eye.
    for _name, _val in vars(rw).items():
        if isinstance(_val, str) and not _name.startswith("__"):
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
    from agents.voice.memory import CallMemory as _Mem
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
    for _node in _ast.walk(_ast.parse(_pl.Path(rw.__file__).read_text(encoding="utf-8"))):
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
                    _p.value for _p in _v.values if isinstance(_p, _ast.Constant)))
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
        check(bool(rw.caller_repeated_answer(_now, _ss(*_prior))) == _want,
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
    import types as _t
    _mk = lambda r, x: _t.SimpleNamespace(role=r, text=x)
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
    from agents.voice.memory import CallMemory as _CM
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
    import types as _types
    def _sess(rec, *turns):
        return _types.SimpleNamespace(
            doctor=_types.SimpleNamespace(hospital_name=rec),
            turns=[_types.SimpleNamespace(role="caller", text=t) for t in turns])
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
        got = bool(rw._ungrounded_escalation(reason, sess_))
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
        check(bool(rw._discarded_location(_R, _s)) == _want,
              f"discarded-answer detector: {_want!s:5} for {_label}",
              rw._candidate_location(_s)[:52])

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
        check(bool(rw._discarded_location(_reason, _gs)) == _want,
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
    check((await _tw._doctor_for("CA_aaa")).doctor_name == "Dr. A"
          and (await _tw._doctor_for("CA_bbb")).doctor_name == "Dr. B",
          "two calls in flight resolve to their own doctors")
    # Twilio retries webhooks. A second /answer for the same SID must resolve
    # to the same doctor, so the lookup must not consume the entry.
    check((await _tw._doctor_for("CA_aaa")).doctor_name == "Dr. A",
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
    _rw_tree = _ast.parse(_pl.Path(rw.__file__).read_text(encoding="utf-8"))
    _senders = set()
    for _fn_node in _ast.walk(_rw_tree):
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
    _calls = [n for n in _ast.walk(_rw_tree)
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
    def _sess_for(doctor, *, branch=None, city=None, resolved=False):
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
    _r_cli = _sess_for(_d_cli, branch="Abadan Branch")._enrich_doctor("Abadan Branch", True)
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
        "Northgate Campus", True)
    check(_d_full.status is rw.DoctorStatus.VERIFIED,
          "complete record + confirmed branch -> VERIFIED", _d_full.status.value)
    check(_d_full.city == "Atlanta", "city is carried across when the call captured one")

    # An unresolved call. The record still has no branch, which is all this
    # says — the reason lives in the call artifact, not the directory row.
    _d_no = Doctor(doctor_name="Dr. B", hospital_name="H")
    _sess_for(_d_no)._enrich_doctor(None, False)
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
    with mock.patch.object(rw, "json_dir", lambda: _wb_dir):
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
        blocked = bool(rw._ungrounded_terms(args, _FakeSess(lines)))
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
    from agents.voice.memory import CallMemory
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
    print("  FAILED" if FAILURES else "  ALL PASSED")
    print("=" * 66 + "\n")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
