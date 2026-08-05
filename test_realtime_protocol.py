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
import sys
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


async def run_call(script):
    """Run one scripted call, return (messages_sent_to_oai, session_memory)."""
    sent = []
    twilio = FakeTwilio([{"event": "start", "start": {"streamSid": "MZtest"}}])
    captured = {}

    real_init = rw.RealtimeSession.__init__

    def spy_init(self, call_sid, doctor):
        real_init(self, call_sid, doctor)
        captured["session"] = self

    with mock.patch.object(rw.websockets, "connect",
                           lambda *a, **k: FakeConn(FakeOAI(script, sent))), \
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
    check(session["audio"]["input"]["turn_detection"]["type"] == "server_vad",
          "turn detection configured")
    check(session["audio"]["input"]["transcription"]["language"] == "en",
          "transcription pinned to en")
    check(session.get("max_output_tokens") == settings.realtime_max_response_tokens,
          "response token cap set")

    ctx = items[0]["item"]["content"][0]["text"]
    check("CALL CONTEXT" in ctx, "per-call facts sent as a conversation item")
    check("Dr. Jane Okafor" in ctx, "context names the doctor")
    check("automated assistant" in ctx, "greeting discloses automation")
    # Match on the concept, not one wording — the greeting says "this call's
    # recorded" now, and a contraction change must not fail a disclosure test.
    check("record" in ctx.lower(), "greeting discloses recording")
    check(any(i["item"].get("type") == "function_call_output" for i in items),
          "tool result returned to the model")

    # A truthful script must never recite an unreachable callback number.
    from agents.voice.templates import is_usable_callback_number
    if not is_usable_callback_number(settings.callback_number):
        check(settings.callback_number not in ctx,
              "unusable CALLBACK_NUMBER withheld from the call context")
        check("NONE AVAILABLE" in ctx,
              "agent told explicitly that no callback number exists")
    check("NEVER invent, guess, or approximate a phone number" in tpl.instructions,
          "instructions forbid inventing a phone number")
    check("repeat it plainly and in full" in tpl.instructions,
          "mid-call identity re-ask is handled")
    check("identity and contact facts are exempt" in tpl.instructions,
          "identity facts exempt from the no-repetition rule")
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
    check("never deferred to a later turn" in tpl.instructions,
          "disclosures cannot be postponed to a later turn")

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
