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
import json
import pathlib
import sys
import tempfile
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


async def run_call(script):
    """Run one scripted call, return (messages_sent_to_oai, session_memory).

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

    with mock.patch.object(rw.websockets, "connect",
                           lambda *a, **k: FakeConn(FakeOAI(script, sent))), \
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
        # The hospital-confirmation question was ignored by 10 of 11 callees and
        # the check it stood for now lives in hospital_mismatch(), so the opening
        # no longer spends its one question on it.
        check("?" not in _g, f"{_name}: opener ends flat, callee speaks next", _g[:52])
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
    check("automated assistant" in disclosed.build_greeting(probe),
          "forage_ai_disclosed announces automation upfront")
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
    # written 0x08 into a regex twice; the first time only _LOCATION_ASK was
    # guarded, and the second landed in a pattern the guard did not cover.
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
    check("ONE MOVE PER TURN" in tpl.instructions,
          "prompt requires one move per turn")
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
