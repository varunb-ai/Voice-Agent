"""Render the A/B persona clips — the flat delivery against the humble one.

    python scripts/render_ab_demo.py                       # marin, both versions
    python scripts/render_ab_demo.py --voice marin --voice cedar
    python scripts/render_ab_demo.py --also-raw            # keep the 24k source too

Writes data/demo_audio/<voice>_<version>_<line>.wav and prints a table of what
was actually said. No phone call, no Twilio, no tunnel: this opens the same
WebSocket the call opens, sends the same session.update, and captures the audio.
A full matrix is a couple of minutes and a few cents.

── WHAT THE EXPERIMENT IS ──────────────────────────────────────────────────
The question the client asked is not "is version B nicer". It is:

    can the SAME voice model produce a different vocal delivery from a
    different persona prompt, or is it flat whatever the prompt says?

So the LINE IS HELD FIXED and only the session instructions vary. Two renders
of "Hi, I'm looking for a new doctor..." — one under the prompt as it shipped
this morning, one under the persona fix — isolate delivery from wording. If
they sound identical, the answer is no and no amount of rewording will help;
that is a real finding and this script is how it gets settled rather than
argued about.

── WHY VERSION A IS RECONSTRUCTED AND NOT WRITTEN OUT BY HAND ──────────────
An A/B against a prompt somebody typed from memory proves nothing: it compares
the new persona against a strawman, and the strawman always loses. Version A
here is the REAL pre-fix prompt, rebuilt by substituting the two superseded
blocks back into the live template. If either substitution fails to match,
this script REFUSES TO RUN rather than rendering two copies of version B and
labelling one of them "before" — see _version_a(), and see the repo's own
history of a str.replace that silently no-opped and shipped anyway.

── WHY IT GOES THROUGH OutboundConditioner ─────────────────────────────────
24kHz studio PCM is not what anybody on a phone hears. The clips are decimated
to 8kHz and conditioned exactly as a live call conditions them, so the
comparison is made on the leg the callee actually gets. Judging a telephony
persona on wideband audio would flatter both versions and rank them wrongly.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.bootstrap  # noqa: F401,E402  (UTF-8 stdout on Windows)

from core.audio_utils import _mulaw_decode                      # noqa: E402
from core.config import settings                                # noqa: E402
from core.models import Doctor                                  # noqa: E402
from agents.voice.outbound_audio import OutboundConditioner     # noqa: E402
from agents.voice.realtime_worker import (                      # noqa: E402
    REALTIME_URL, build_audio_config,
)
from agents.voice.templates import get_template                 # noqa: E402
from agents.voice.objectives import sentences                   # noqa: E402

_TWILIO_SR = 8_000


# ── Version A: the prompt exactly as it was before the persona fix ──────────
# Verbatim, including the indentation, because these are matched against the
# live template with str.replace and a near-miss silently matches nothing.

_OLD_SHARED_RULE = """\
- EVERY SENTENCE MUST BE IN THE CONVERSATION, NEVER ABOUT IT. The test: delete
  the sentence — if the caller loses no information, it should not be said.
  A sentence that narrates what you are doing, how you are speaking, or how you
  intend to reply is a sentence about the conversation: "let me think", "one
  second", "hmm", "okay so". The ways to make that move are endless, so judge
  by the test and not by the wording. Natural pauses are fine; narrating them
  is not."""

_NEW_SHARED_RULE = """\
- EVERY SENTENCE MUST BE IN THE CONVERSATION, NEVER ABOUT IT. The test: delete
  the sentence — if the caller loses no information, it should not be said.
  A sentence that narrates what you are doing, how you are speaking, or how you
  intend to reply is a sentence about the conversation: "let me think", "one
  second", "let me check". The ways to make that move are endless, so judge by
  the test and not by the wording. A HESITATION IS NOT A SENTENCE and the test
  does not reach one; your tone above sets how much you hesitate."""

_OLD_TONE = """\
- A slightly hesitant person making a personal call about themselves. Not brisk,
  not efficient, not doing admin — this one matters to you a little.
- Warm and a bit unsure, the way anyone is ringing a practice cold: mildly
  apologetic for taking their time, genuinely grateful for small help."""

_NEW_TONE = """\
- A NATURAL VOICE, SUBTLY UNDER THE WEATHER. You are a real person calling a
  clinic while not feeling well. The illness shows in small, human ways:
  slightly lower energy than a healthy caller, a touch of tiredness, a little
  less confidence when explaining yourself, a faint edge of concern. It is
  FELT through delivery, never performed — no sad voice, no drama, no crying,
  no weakness, no monotone, nothing slow or acted. If a listener thinks "this
  person doesn't feel completely well and seems a bit concerned", that is
  exactly right. If they think "this person is doing a sad voice", you failed.
- POLITE AND HUMBLE, SEEKING HELP. You are politely trying to get medical
  assistance, not casually inquiring and not chatting with a friend. Ordinary
  courtesy throughout; customer-service brightness never.
- NORMAL INSIDE A SENTENCE. Once started, a sentence flows at a natural
  conversational pace — no gaps inside it, no dragging, no rushing, nothing
  drawn out. All the unwellness lives in energy and tone, never in broken flow.
- HESITATION SITS AT A BOUNDARY, NEVER MID-SENTENCE. A brief natural beat —
  a small pause, a soft "Well", "Mm-hmm", "Yeah" — is human and welcome BEFORE
  you answer, when you are thinking, unsure, or recalling something. ONE may
  also sit at the JOIN where your reaction hands over to your question:
  "Thanks for checking, and — is there a waiting list?" That join is a clause
  boundary, not the middle of a sentence. NEVER drop a filler into continuous
  speech, NEVER one after every reply of theirs, NEVER the same one twice.
- When you are not sure of something, let it show mildly — a softer answer, a
  slight uncertainty, a brief beat before you commit — while staying clear and
  conversational the whole time.
- Gratitude is quiet, not cheerful. NEVER say "sorry" unless you genuinely
  misheard them. And never perk up mid-call."""


def _version_a(current: str) -> str:
    """The live instructions with the persona fix reversed out.

    RAISES rather than returning something plausible. A silent no-op here
    produces two renders of version B, one of them captioned "before", and the
    demo then shows a difference that does not exist — which is worse than no
    demo. Both blocks must be found.
    """
    out = current
    for label, new, old in (("shared anti-narration rule", _NEW_SHARED_RULE, _OLD_SHARED_RULE),
                            ("_TONE_PATIENT", _NEW_TONE, _OLD_TONE)):
        if new not in out:
            raise SystemExit(
                f"render_ab_demo: cannot rebuild version A — the {label} in "
                f"templates.py no longer matches the copy held here. Update the "
                f"constant in this file to the text you replaced, or version A "
                f"would silently render as version B."
            )
        out = out.replace(new, old)
    return out


# ── The lines ───────────────────────────────────────────────────────────────
# REAL LINES THE AGENT ACTUALLY SAYS, not invented demo copy. The greeting is
# pulled from the template itself so it cannot drift; the other two are quoted
# in _PATIENT_REASON and _PATIENT_CLOSE. Nothing here is the name-and-DOB
# intake answer: that turn is the compound-disclaimer defect fixed on
# 2026-09-03, and demonstrating a persona on a line that used to be broken
# invites the wrong conversation.

_REASON_LINE = ("Oh, nothing urgent. Just a standard checkup. I'm just looking "
                "for a new doctor right now.")
_CLOSE_LINE = "Let me just figure out my schedule, and I'll call back. Thanks!"


# ── TWO KINDS OF CLIP, because the brief contains two different questions ───
#
# SCRIPTED — the words are fixed and only the persona varies. This is the
#   controlled experiment and the one that answers "can the same voice model
#   deliver the same sentence differently?". Nothing else can answer it: if the
#   words move, any difference a listener hears has two causes.
#
# NATURAL — the model is given the exchange so far and answers in its own
#   words. This is the PHRASING question, which is a separate client
#   requirement, and it cannot be asked with the words held fixed.
#
# THE SPLIT IS NOT TIDINESS, IT IS THE SECOND THING THIS SCRIPT GOT WRONG.
# Every render opens a fresh session, so at turn one the model is told — by the
# prompt under test, correctly — to open with the greeting, and the first run
# produced the greeting for all three tags. Seeding the prior turns fixed that
# and exposed the next layer: with history in place, a "say this line"
# conversation item reads as something the RECEPTIONIST said, and the model
# answers it in character ("Oh, okay. Let me answer why I'm calling."). Moving
# the directive into response.create.instructions did not fix it either — a
# 5,900-token prompt about how to conduct a call beats a one-line recital
# request, which is the prompt being strong rather than the prompt being wrong.
#
# So only the GREETING is scripted, and it is the right one to script: it is a
# fixed line in the template, it is the first thing every callee hears, and it
# renders identically worded on every attempt. The mid-call turns stop
# pretending to be controlled and become what they actually are.
_PRIOR_ANSWERED = "Yes, this is Dr. Okafor's office."
_PRIOR_WHY = "Sure — can I ask what you're looking to be seen for?"
_PRIOR_ACCEPTING = ("She is taking new patients, yes. Would you like me to "
                    "get you booked in?")

SCRIPTED, NATURAL = "scripted", "natural"


def _lines(template, doctor: Doctor) -> list[tuple[str, str, str, list]]:
    """(tag, mode, line or "", prior turns) — role/text pairs, oldest first."""
    greet = template.build_greeting(doctor)
    return [
        # Turn one. Nothing to seed: this IS the opener.
        ("greeting", SCRIPTED, greet, []),
        ("reason", NATURAL, "",
         [("assistant", greet), ("user", _PRIOR_ANSWERED), ("user", _PRIOR_WHY)]),
        ("close", NATURAL, "",
         [("assistant", greet), ("user", _PRIOR_ANSWERED),
          ("user", _PRIOR_ACCEPTING)]),
    ]


# ── Rendering ───────────────────────────────────────────────────────────────

async def _render(voice: str, instructions: str, context: str,
                  text: str, prior: list) -> tuple[bytes, str]:
    """One line, one persona. Returns (PCM16 24kHz, transcript)."""
    import websockets

    url = REALTIME_URL.format(model=settings.realtime_model)
    # additional_headers, NOT extra_headers: websockets renamed it in v14 and
    # this repo runs 15.x, where the old name is a TypeError at connect time.
    hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}

    async with websockets.connect(url, additional_headers=hdr) as ws:
        await asyncio.wait_for(ws.recv(), timeout=20)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": instructions,
            # NO TOOLS. Nothing here should be saved, and a model holding
            # save_branch on a scripted line may call it instead of speaking.
            "audio": build_audio_config(
                transcribe_model=settings.realtime_transcribe_model,
                transcribe_hint="",
                audio_format="pcmu",
                # PCM ON THE WAY OUT, which is what gives the conditioner
                # something to work on — asking for pcmu here would hand back
                # 8k mu-law that has already thrown away the band being fixed.
                output_format="pcm",
                noise_reduction="off",
                turn_detection="server_vad",
                eagerness="medium",
                voice=voice),
            # 200 TRUNCATED REAL LINES. Audio output bills tokens fast, and at
            # 200 the greeting fit while a mid-call turn came back as "Sorry,
            # I" and stopped — a cut-off clip that reads as the persona
            # trailing away rather than as a cap being hit. Generous, because
            # nothing here is paying for a long tail.
            "max_output_tokens": 1200,
        }}))
        # WAIT FOR session.updated. Sending response.create into the gap runs
        # the render under whatever the session was before, which on the first
        # clip is the default persona and no voice selection at all.
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if m.get("type") == "session.updated":
                break
            if m.get("type") == "error":
                raise SystemExit(f"session.update rejected: {m.get('error')}")

        # PER-CALL CONTEXT FIRST, exactly as the live call sends it: a user
        # item carrying input_text, landing after the cached instructions
        # prefix. Without it the persona has no synthetic name, no doctor and
        # no practice, and a natural turn invents all three.
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": context}]}}))

        # The turns that already happened go in next, so a mid-call line is
        # not asked for at turn one. THE TWO ROLES DO NOT SHARE A CONTENT TYPE:
        # an assistant item carries "output_text", a user item "input_text".
        # This is checked server-side and rejects the whole render —
        #     Invalid value: 'text'. Value must be 'output_text'.
        # — which is the good failure. It is written down because "text" is the
        # obvious guess and it is wrong.
        for role, said_before in prior:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": role, "content": [
                    {"type": ("output_text" if role == "assistant"
                              else "input_text"),
                     "text": said_before}]}}))
        if text:
            # SCRIPTED. `response.create.instructions` REPLACES the session
            # instructions for this response, so the persona is RESENT inside
            # it rather than merely appended to. A and B still differ by
            # exactly the persona text and by nothing else, which is the only
            # property the A/B needs; the directive must never go in ALONE,
            # which would strip the persona and render both versions the same.
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "instructions": (
                        f"{instructions}\n\n"
                        f"# This turn\n"
                        f"Say this line and only this line, word for word, in "
                        f"your own voice and manner: {text}\n"
                        f"Do not add anything and do not reply to anything. "
                        f"Say exactly those words and stop."),
                }}))
        else:
            # NATURAL. Bare, so the SESSION instructions apply — the persona
            # under test, unmodified, answering the exchange seeded above in
            # whatever words it chooses. An instructions override here would
            # replace the very thing being measured.
            await ws.send(json.dumps({"type": "response.create"}))

        # ONE SPOKEN ITEM, exactly as the live call plays it. The model can
        # emit several assistant items in a single response, and the call path
        # mutes everything after the first — so concatenating them here would
        # render a clip containing audio no callee would ever hear. The first
        # version of this script did that and produced 19.55s of audio under a
        # three-word transcript, which is how it was noticed.
        #
        # Keyed on item_id off the deltas themselves rather than on a counter,
        # because the items interleave with the transcript events and only the
        # id says which audio belongs to which.
        chunks: list[bytes] = []
        first_item = ""
        said = ""
        dropped: list[str] = []
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            t = m.get("type")
            # gpt-realtime-2 emits response.output_audio.delta. The older
            # response.audio.delta never arrives, and a loop waiting on it
            # collects nothing and writes an empty file.
            if t == "response.output_audio.delta" and m.get("delta"):
                item = m.get("item_id", "")
                if not first_item:
                    first_item = item
                if item == first_item:
                    chunks.append(base64.b64decode(m["delta"]))
            elif t == "response.output_audio_transcript.done":
                if m.get("item_id", "") == first_item:
                    said = m.get("transcript", "") or said
                elif m.get("transcript"):
                    dropped.append(m["transcript"])
            elif t == "response.done":
                break
            elif t == "error":
                raise SystemExit(f"render failed: {m.get('error')}")

    if dropped:
        # NOT SILENT. A muted second item is the model piling two moves into
        # one turn, which is a thing to see rather than to tidy away — the same
        # reason the call artifact records dropped_second_items instead of
        # simply discarding them.
        print(f"    (2nd item muted, as on a call: {dropped[0][:60]!r})")
    return b"".join(chunks), said


def _speech_seconds(samples: np.ndarray, sr: int) -> float:
    """Seconds of the clip that carry voice, at 20ms resolution.

    RAW DURATION IS THE WRONG MEASURE and it took four runs to see why. A clip
    is speech plus the silence around it, and the silence moves for reasons
    that have nothing to do with the persona — where the model chose to breathe,
    how long the tail ran. Duration on the same sentence swung both directions
    across runs and reversed the apparent winner.

    Voiced time on a FIXED line is speaking rate, which is the thing the
    persona is supposed to change. Same threshold the call path uses for a
    frame that carries nothing.
    """
    win = int(0.02 * sr)
    if win <= 0 or samples.size < win:
        return 0.0
    n = samples.size // win
    rms = np.sqrt((samples[:n * win].reshape(n, win) ** 2).mean(axis=1))
    return float((rms > 0.01).sum()) * 0.02


def _write_wav(path: Path, pcm24: bytes, *, conditioned: bool) -> tuple[float, float]:
    """Write one clip. Returns (duration, voiced seconds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if conditioned:
        # The live chain, in one shot. The conditioner is stateful across
        # chunks by design, but its own tests record that chunked output is
        # byte-identical to single-shot, so feeding the whole clip is exact.
        mulaw = OutboundConditioner().process(pcm24)
        samples = _mulaw_decode(mulaw)
        sr = _TWILIO_SR
    else:
        samples = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 24_000
    # 16-BIT LINEAR, ALWAYS. Handing mu-law bytes to wave.setsampwidth(1)
    # writes a header claiming 8-bit unsigned PCM over a mu-law payload, and
    # every player renders that as loud distortion — a broken demo that looks
    # like a broken agent.
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return len(pcm16) / float(sr), _speech_seconds(samples, sr)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", default=None,
                    help="repeatable; defaults to REALTIME_VOICE")
    ap.add_argument("--doctor", default="Dr. Jane Okafor")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", action="append", default=None,
                    metavar="TAG",
                    help="repeatable; render only these lines (greeting, "
                         "reason, close). --only greeting with a high --takes "
                         "is the cheap way to settle the delivery question, "
                         "since it is the only scripted line.")
    ap.add_argument("--takes", type=int, default=1,
                    help="renders per condition. >1 is how the delivery claim "
                         "gets measured rather than asserted: one take cannot "
                         "separate the persona from run-to-run variation, and "
                         "on this script it reversed the apparent winner "
                         "between runs.")
    ap.add_argument("--also-raw", action="store_true",
                    help="also keep the unconditioned 24kHz source")
    args = ap.parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    voices = args.voice or [settings.realtime_voice]
    for v in voices:
        if v in ("marin", "cedar") and settings.realtime_model != "gpt-realtime-2":
            raise SystemExit(
                f"voice {v!r} requires REALTIME_MODEL=gpt-realtime-2, "
                f"currently {settings.realtime_model!r}")

    out_dir = args.out or (Path(__file__).resolve().parent.parent
                           / "data" / "demo_audio")
    template = get_template("patient_discovery")
    doctor = Doctor(doctor_name=args.doctor, hospital_name="Northside Medical Group")
    context = template.build_context(doctor)

    versions = [("A-flat", _version_a(template.instructions)),
                ("B-humble", template.instructions)]
    lines = _lines(template, doctor)
    if args.only:
        wanted = {t.strip().lower() for t in args.only}
        unknown = wanted - {ln[0] for ln in lines}
        if unknown:
            raise SystemExit(f"--only: no such line {sorted(unknown)}; "
                             f"choose from {[ln[0] for ln in lines]}")
        lines = [ln for ln in lines if ln[0] in wanted]

    print(f"\nmodel   {settings.realtime_model}")
    print(f"voices  {', '.join(voices)}")
    print(f"out     {out_dir}\n")

    rows = []
    for voice in voices:
        for version, instructions in versions:
            for tag, mode, text, prior in lines:
                for take in range(1, args.takes + 1):
                    pcm24, said = await _render(voice, instructions, context,
                                                text, prior)
                    if not pcm24:
                        print(f"  {voice}/{version}/{tag} take {take}: "
                              f"NO AUDIO — skipped")
                        continue
                    suffix = f"_take{take}" if args.takes > 1 else ""
                    stem = f"{voice}_{version}_{tag}{suffix}"
                    path = out_dir / f"{stem}.wav"
                    dur, voiced = _write_wav(path, pcm24, conditioned=True)
                    if args.also_raw:
                        _write_wav(out_dir / f"{stem}.raw24k.wav", pcm24,
                                   conditioned=False)
                    rows.append((voice, version, tag, mode, dur, voiced,
                                 said, text))
                    print(f"  {path.name}  {dur:.2f}s  ({voiced:.2f}s voiced)")

    # ── What was actually said ──────────────────────────────────────────────
    # Printed because the audio is the deliverable but the TEXT is the control:
    # if version B rendered different words, the clips differ for two reasons
    # at once and the delivery claim is no longer isolated.
    print("\n" + "=" * 78)
    print("  TRANSCRIPTS — the words are the control; only delivery should vary")
    print("=" * 78)
    for voice, version, tag, mode, dur, voiced, said, want in rows:
        print(f"\n  {voice:6s} {version:9s} {tag:9s} {mode:8s} "
              f"{dur:5.2f}s  {voiced:5.2f}s voiced")
        print(f"    {said or '(no transcript)'}")

    # DID IT SAY THE LINE IT WAS ASKED FOR? The first run of this script
    # rendered the greeting for all three tags — every clip was the opener,
    # because a fresh session is at turn one and the prompt says to open with
    # the greeting. The audio was fine and the experiment was worthless, and
    # nothing would have shown it except reading six transcripts. Now it is
    # checked: a low word overlap means the model answered the CALL rather than
    # the request, and those clips are not comparable.
    # NATURAL TURNS ARE SUPPOSED TO DIFFER IN WORDING, so they reach this with
    # an empty `want` and fall out at the length test below. Only scripted
    # clips carry a line to be held to.
    wrong = []
    for voice, version, tag, _m, _d, _v, said, want in rows:
        a = {w.strip(".,!?—'\"").lower() for w in (said or "").split()}
        b = {w.strip(".,!?—'\"").lower() for w in want.split()}
        if not b:
            continue
        if len(a & b) / len(b) < 0.6:
            wrong.append((voice, version, tag, want, said))
    if wrong:
        print("\n" + "!" * 78)
        print("  OFF-SCRIPT — these clips did not say the line they were given")
        print("!" * 78)
        for voice, version, tag, want, said in wrong:
            print(f"\n  {voice}/{version}/{tag}")
            print(f"    asked : {want}")
            print(f"    said  : {said or '(nothing)'}")
        print("\n  Do NOT put these in the A/B: the words differ as well as the")
        print("  delivery, so any difference a listener hears has two causes.")

    # ELLIPSIS WATCH. objectives.sentences() splits on [.!?] followed by
    # whitespace, so a literal "..." reads as a sentence boundary: "Hi
    # there... good afternoon." counts as three sentences and would fire
    # piled_turns (>=3) on an ordinary turn. Nothing breaks — those metrics are
    # measure-only — but they are how this very change gets judged, so a model
    # that reaches for ellipsis on its own needs to be seen rather than
    # discovered later in a confusing artifact.
    flagged = [(v, ver, tag, s) for v, ver, tag, _m, _d, _v, s, _w in rows
               if "..." in (s or "") or len(sentences(s or "")) >= 3]
    if flagged:
        print("\n" + "-" * 78)
        print("  ELLIPSIS / SENTENCE-COUNT WATCH")
        print("-" * 78)
        for v, ver, tag, s in flagged:
            print(f"  {v}/{ver}/{tag}: {len(sentences(s))} sentences"
                  f"{' — contains ...' if '...' in s else ''}")
        print("\n  These turns would read as pile-ups to conversation_metrics.")
        print("  Measure-only, so no call behaviour changes — but if this is")
        print("  common, protect '...' in objectives.sentences() before")
        print("  trusting piled_turns on any call run under this persona.")

    # ── The measurement ─────────────────────────────────────────────────────
    # ON THE SCRIPTED LINE ONLY. Voiced seconds on identical words is speaking
    # rate; voiced seconds on different words is just a different sentence, so
    # the natural turns are excluded from the comparison they cannot support.
    #
    # THE SPREAD IS PRINTED BESIDE THE MEAN, and that is the point of the whole
    # table. Four single-take runs of this script disagreed about which version
    # was slower, twice each way. A mean whose takes overlap the other version's
    # takes has not shown anything, and printing only the mean is how that goes
    # unnoticed — so if the ranges overlap, the honest report is "judge by ear".
    scripted = [r for r in rows if r[3] == SCRIPTED]
    if scripted and args.takes > 1:
        print("\n" + "=" * 78)
        print("  SPEAKING RATE on the fixed line — same words, so only "
              "delivery varies")
        print("=" * 78)
        print(f"\n  {'voice':8s} {'version':10s} {'takes':>5s} "
              f"{'voiced mean':>12s} {'min':>7s} {'max':>7s}")
        for voice in voices:
            for version, _ in versions:
                vs = [r[5] for r in scripted if r[0] == voice and r[1] == version]
                if not vs:
                    continue
                print(f"  {voice:8s} {version:10s} {len(vs):5d} "
                      f"{sum(vs)/len(vs):11.2f}s {min(vs):6.2f}s {max(vs):6.2f}s")
        for voice in voices:
            a = [r[5] for r in scripted if r[0] == voice and r[1] == "A-flat"]
            b = [r[5] for r in scripted if r[0] == voice and r[1] == "B-humble"]
            if not (a and b):
                continue
            if max(a) < min(b) or max(b) < min(a):
                slower = "B-humble" if sum(b) > sum(a) else "A-flat"
                print(f"\n  {voice}: SEPARATED — every {slower} take is slower "
                      f"than every take of the other. The persona moved the "
                      f"delivery.")
            else:
                print(f"\n  {voice}: OVERLAPPING ranges — this many takes do "
                      f"NOT show a rate difference. Do not claim one; play the "
                      f"clips and judge by ear, or raise --takes.")

    print(f"\n{len(rows)} clips in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
