"""Does the ENFORCEMENT close what the prompt could not?

    python scripts/check_cadence_loop.py             # 3 takes per scenario
    python scripts/check_cadence_loop.py 5

── WHY THIS EXISTS, AND WHY render_ab_demo COULD NOT ANSWER IT ─────────────
render_ab_demo measures the PROMPT: it opens a session, seeds an exchange,
asks for one turn and stops. Every guard in this repo lives in
_handle_agent_transcript, which that path never reaches. So on 2026-09-04
three rounds of A/B showed the rebuilt prompt not moving the enquiry-bot
cadence and said nothing at all about whether the fix worked — the prompt is
half the change and the renders could only see that half.

This drives the loop the call actually runs:

    seed the exchange -> response.create -> read the agent's turn
    -> run the REAL predicates over it (_ack_opener, _housekeeping_turn,
       _narrated_the_reply)
    -> if one fires, inject the REAL directive (cadence_directive)
    -> response.create -> read what the model says next

and prints both turns. What it asks is narrow and worth stating plainly: given
the turn the model does produce, does the corrective directive get a better
one. It is not a phone call, there is no audio path, no VAD, no barge-in and
no tools, so nothing here proves what happens on a live call.

── WHAT IT FOUND ON THE RUN THAT MOTIVATED IT ──────────────────────────────
The first version of the cadence directives named only the failure — "start on
the thing you are actually saying". Nine corrections, none of them good:

    turn 1  "Okay, thanks for confirming that."
    turn 2  "Okay."
    turn 2  "Let me just get the location sorted out with you."
    turn 2  "Thanks for confirming that this is Dr. Okafor's office."

Passing the outstanding objective field into the directive changed it:

    turn 1  "Okay, thanks for confirming that."
    turn 2  "Which office does Dr. Okafor see people at?"
    turn 1  "Sure, let me share that with you."
    turn 2  "January 18, 1977."

Six of nine good, two partial, one poor. That is the difference between a
directive that says what not to do and one that says what to say, and it is
not visible from any text test — which is the argument for keeping this script
rather than the numbers it produced.

── NOTHING IS COPIED ───────────────────────────────────────────────────────
The predicates and the directives are IMPORTED. An earlier draft held its own
copies of the directive strings and was still testing them after two of the
three started interpolating the objective — a hand-copy that goes stale while
continuing to pass is this repo's most documented failure shape.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.bootstrap  # noqa: F401,E402  (UTF-8 stdout on Windows)

from core.config import settings                              # noqa: E402
from core.models import Doctor                                # noqa: E402
from agents.voice.realtime_worker import (                    # noqa: E402
    REALTIME_URL, build_audio_config,
)
from agents.voice.templates import get_template               # noqa: E402
from agents.voice.grounding import (                          # noqa: E402
    _ack_opener, _reacted_to_news, _housekeeping_turn,
    _narrated_the_reply, cadence_directive,
)


def _verdict(objective, text: str, prev_ack: str, memory: dict):
    """Which guard fires, and the directive it would inject.

    THE ORDER MATTERS HERE AND NOT IN turns.py. There the three are
    independent `if` blocks and two can fire on one turn; this returns one so
    the run reads cleanly, so a turn that is both narration and an ack-run is
    reported under the more specific of the two.
    """
    if _narrated_the_reply(text):
        return "reply_narration", cadence_directive("reply_narration")
    want = objective.next_spoken(memory)
    if _housekeeping_turn(text):
        return "housekeeping", cadence_directive("housekeeping", want)
    # THE EXEMPTION LIVES WITH THE GUARD, for the reason the directive wording
    # does: this script replays the real predicates against the real model, and
    # a copy that had not grown the exemption would be measuring a guard that
    # no longer ships.
    if _ack_opener(text) and prev_ack and not _reacted_to_news(text):
        return "ack_run", cadence_directive("ack_run", want)
    return "", ""


async def _turn(ws) -> str:
    """One response.create, returning the FIRST spoken item's transcript.

    First item only, because that is all the call path plays — everything
    after it is muted. Reading them all would report words no callee hears.
    """
    await ws.send(json.dumps({"type": "response.create"}))
    first, said = "", ""
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
        t = m.get("type")
        if t == "response.output_audio.delta" and not first:
            first = m.get("item_id", "")
        elif t == "response.output_audio_transcript.done":
            if not first or m.get("item_id") == first:
                said = m.get("transcript") or said
                first = first or m.get("item_id", "")
        elif t == "response.done":
            return said.strip()
        elif t == "error":
            raise SystemExit(f"turn failed: {m.get('error')}")


async def _run_one(objective, prior, instructions, context, voice,
                   prev_ack, memory):
    import websockets

    hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}
    url = REALTIME_URL.format(model=settings.realtime_model)
    async with websockets.connect(url, additional_headers=hdr) as ws:
        await asyncio.wait_for(ws.recv(), timeout=20)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": instructions,
            # NO TOOLS, deliberately: a model holding save_branch may call it
            # instead of speaking, and this is measuring speech.
            "audio": build_audio_config(
                transcribe_model=settings.realtime_transcribe_model,
                transcribe_hint="", audio_format="pcmu", output_format="pcm",
                noise_reduction="off", turn_detection="server_vad",
                eagerness="medium", voice=voice),
            "max_output_tokens": 1200}}))
        # WAIT FOR session.updated, or the first turn runs under the default
        # persona — the same trap render_ab_demo documents.
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if m.get("type") == "session.updated":
                break
            if m.get("type") == "error":
                raise SystemExit(f"session.update rejected: {m.get('error')}")
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": context}]}}))
        # An assistant item carries "output_text", a user item "input_text".
        # Server-side checked; "text" is the obvious guess and it is wrong.
        for role, said in prior:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": role, "content": [
                    {"type": ("output_text" if role == "assistant"
                              else "input_text"), "text": said}]}}))
        t1 = await _turn(ws)
        fired, directive = _verdict(objective, t1, prev_ack, memory)
        if not fired:
            return t1, "", ""
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": directive}]}}))
        return t1, fired, await _turn(ws)


def _scenarios(template, doctor):
    """Real exchanges off the corpus, with the memory the call would hold.

    THE MEMORY IS NOT DECORATION: two of the three directives interpolate the
    next outstanding field off it, so a scenario with the wrong keys tests a
    directive that would never go out. The keys are the objective's
    memory_keys, not the field names.
    """
    greet = template.build_greeting(doctor)
    opened = [("assistant", greet),
              ("user", f"Yes, this is Dr. {doctor.doctor_name.split()[-1]}'s "
                       f"office.")]
    return {
        # -1306: "Got it, thanks for confirming that. Let me check one more
        # thing." — five of eleven turns on that call were this shape.
        "confirm": (opened, "", {"doctor_identity": "confirmed"}),
        # -1236: answered a date of birth with "Mm-hmm, one moment while I
        # answer that. April 21, 1984."
        "dob": (opened + [
            ("user", "Okay, can I get your first and last name?"),
            ("assistant", "I'm not a patient here yet — I'm just looking. "
                          "It's Emile Keswick."),
            ("user", "Okay, and your date of birth?")],
            "thanks", {"doctor_identity": "confirmed"}),
        # -2121: "Thanks for checking — I'll ask one quick thing."
        "bad_news": (opened + [
            ("user", "She sees people at our Northgate clinic."),
            ("user", "Unfortunately she's completely booked and not accepting "
                     "new patients right now.")],
            "okay", {"doctor_identity": "confirmed",
                     "branch": "Northgate clinic",
                     "new_patient_status": "no"}),
    }


async def main() -> int:
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    takes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    template = get_template("patient_discovery")
    doctor = Doctor(doctor_name="Dr. Jane Okafor",
                    hospital_name="Northside Medical Group")
    # Ignored by PatientPersonaTemplate; passed so the call type-checks.
    context = template.build_context(
        doctor, callback_number="", callback_email="")
    print(f"\nmodel  {settings.realtime_model}"
          f"\nvoice  {settings.realtime_voice}"
          f"\ntakes  {takes} per scenario\n")
    fired_n = 0
    for tag, (prior, prev_ack, memory) in _scenarios(template, doctor).items():
        print(f"\n{'=' * 74}\n{tag}\n{'=' * 74}")
        for i in range(takes):
            t1, fired, t2 = await _run_one(
                template.objective, prior, template.instructions, context,
                settings.realtime_voice, prev_ack, memory)
            print(f"  take {i + 1}\n    turn 1 : {t1}")
            if fired:
                fired_n += 1
                print(f"    GUARD  : {fired}\n    turn 2 : {t2}")
            else:
                print("    GUARD  : (none fired — turn 1 stands)")
    # NO PASS/FAIL. Whether turn 2 is better is a judgement about conversation
    # and this script does not get to make it — a threshold here would be a
    # number nobody trusts on a sample of nine. Read the pairs.
    print(f"\n{fired_n} guard firings. Read the pairs: the question is whether "
          f"turn 2 is what a person would have said.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
