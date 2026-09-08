"""Does the persona actually change the VOICE, or only the words?

    python scripts/compare_persona_delivery.py                  # 3 takes
    python scripts/compare_persona_delivery.py 5

Writes data/demo_audio/persona_<tone>_<case>_takeN.wav and prints an acoustic
table: speaking rate, energy, pitch movement, pauses, emphasis.

── THE BRIEF ASKED FOR ONE EXPERIMENT AND IT IS TWO ────────────────────────
The request was to compare

    A  "Okay. Is there a waiting list?"
    B  "Oh… okay. I was hoping she might be taking new patients. Is there a
        waiting list?"

on pace, energy, pitch movement, emphasis and pauses. Those are two different
sentences, so any difference a listener hears has two causes and the
measurement cannot separate them — the same confound render_ab_demo documents.
Run as one test it would "prove" the persona works by reading out the fact
that B has more words in it.

So it is split, and both halves are needed:

  SCRIPTED — the SAME line under both personas. The words cannot move, so
    every difference is delivery. This is the half that answers "can this voice
    model sound mildly unwell at all, or is it flat whatever the prompt says?"
    That question has a real chance of answering NO, which is why it is asked
    separately and first.

  NATURAL — the model answers the same bad news in its own words under both
    personas. This is the wording half, and it is where the brief's B line
    would come from if the persona produces it at all.

── THE CONTROL IS REAL, NOT INVENTED ──────────────────────────────────────
The neutral side substitutes _TONE_ADMIN — the tone block templates 1-3
actually ship, "a capable, friendly person doing a quick piece of admin" — in
place of _TONE_PATIENT. Everything else in the prompt is byte-identical. An
A/B against a neutral persona typed from memory compares the persona against a
strawman, and the strawman always loses.

── WHAT IS MEASURED, AND WHAT IT CANNOT SETTLE ────────────────────────────
Pitch is tracked by autocorrelation on 40 ms frames over 70-300 Hz, which is
adequate for 8 kHz telephony and not a research-grade tracker. Every figure is
reported with its per-take spread, and a difference is only worth claiming
where the ranges separate — a single take cannot tell a persona from run-to-run
variation, which is the mistake that reversed the apparent winner between two
runs of render_ab_demo.

READ THE F0 COLUMNS WITH SUSPICION ON SHORT LINES. The first run came back
with a median of 171 Hz +-105 on a six-word scripted line: a half-range that
wide is the tracker producing octave errors across takes, not the voice moving
by an octave. On clips this short the F0 columns carry almost no information
and the honest conclusion has to rest on s/word, RMS, crest and pauses. If
pitch movement is the question, lengthen the line rather than trusting these
two columns.

None of this says whether it SOUNDS right. Play the files.
"""
from __future__ import annotations

import asyncio
import base64
import json
import statistics
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
from agents.voice import templates as T                         # noqa: E402

SR = 8_000


def _neutral(instructions: str) -> str:
    """The live prompt with the patient persona swapped for the admin tone.

    RAISES rather than returning something plausible, exactly as
    render_ab_demo._version_a does: a silent no-op here renders both sides
    under the same persona and reports "no difference", which is the one
    conclusion this script must never reach by accident.
    """
    if T._TONE_PATIENT not in instructions:
        raise SystemExit(
            "compare_persona_delivery: _TONE_PATIENT is not in the built "
            "prompt, so the neutral control cannot be made by substitution. "
            "Both sides would render identically and the table would read as "
            "'the persona does nothing'.")
    return instructions.replace(T._TONE_PATIENT, T._TONE_ADMIN)


# ── The turn under test ─────────────────────────────────────────────────────
# call-20260904-1245, near enough verbatim: this is the moment the brief is
# about, and the one where a person's voice would actually move.
_BAD_NEWS = ("Unfortunately she's completely booked and not accepting new "
             "patients right now.")
# The scripted line is EMOTIONALLY LOADED BUT STATES NO EMOTION, and it is long
# enough for prosody to have somewhere to happen. Both matter.
#
# The first version of this test used the brief's neutral line, "Okay. Is there
# a waiting list?" — six words, no interior pauses possible, and the pitch
# tracker had four voiced frames to work with. A delivery difference has almost
# nowhere to show on a line that short, so "no difference" was close to
# unfalsifiable.
#
# It also must not contain the emotion as WORDS. Handing the persona side
# "I'm not feeling great at the moment" to read would let it score by reciting
# the state rather than delivering it, which is the exact failure the persona
# block itself bans. Disappointment is carried by the situation here, and both
# sides read the identical text.
_SCRIPTED = (
    "I was really hoping she'd be taking people on. It's taken me a while to "
    "find someone local, so that's a shame. Is there a waiting list I could "
    "get on, or should I try somewhere else?")
_SCRIPTED_SHORT = "Okay. Is there a waiting list?"


async def _render(voice, instructions, context, prior, text):
    import websockets

    hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}
    url = REALTIME_URL.format(model=settings.realtime_model)
    async with websockets.connect(url, additional_headers=hdr) as ws:
        await asyncio.wait_for(ws.recv(), timeout=20)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "instructions": instructions,
            "audio": build_audio_config(
                transcribe_model=settings.realtime_transcribe_model,
                transcribe_hint="", audio_format="pcmu", output_format="pcm",
                noise_reduction="off", turn_detection="server_vad",
                eagerness="medium", voice=voice),
            "max_output_tokens": 1200}}))
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
        for role, said in prior:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": role, "content": [
                    {"type": ("output_text" if role == "assistant"
                              else "input_text"), "text": said}]}}))
        if text:
            # The persona is RESENT inside the override, never replaced by the
            # directive alone — otherwise the scripted side renders with no
            # persona at all and the comparison is against nothing.
            await ws.send(json.dumps({"type": "response.create", "response": {
                "instructions": (
                    f"{instructions}\n\n# This turn\nSay this line and only "
                    f"this line, word for word, in your own voice and manner: "
                    f"{text}\nDo not add anything and do not reply to "
                    f"anything. Say exactly those words and stop.")}}))
        else:
            await ws.send(json.dumps({"type": "response.create"}))

        chunks, first, said = [], "", ""
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            t = m.get("type")
            if t == "response.output_audio.delta" and m.get("delta"):
                if not first:
                    first = m.get("item_id", "")
                if m.get("item_id", "") == first:
                    chunks.append(base64.b64decode(m["delta"]))
            elif t == "response.output_audio_transcript.done":
                if m.get("item_id", "") == first:
                    said = m.get("transcript", "") or said
            elif t == "response.done":
                break
            elif t == "error":
                raise SystemExit(f"render failed: {m.get('error')}")
        return b"".join(chunks), said.strip()


def _to_phone(pcm24: bytes) -> np.ndarray:
    """24 kHz studio PCM -> the 8 kHz the callee is actually played.

    Through OutboundConditioner, because judging a telephony persona on
    wideband audio flatters both sides and can rank them wrongly.

    ── PASS THE BYTES. THE FIRST VERSION DECODED THEM FIRST AND WAS SILENT ────
    It did this:

        cond.process(np.frombuffer(pcm24, dtype=np.int16)
                     .astype(np.float32) / 32768.0)

    `process` takes PCM16 BYTES and opens with `np.frombuffer(pcm16,
    dtype=np.int16)`. Handed a float32 array, numpy reinterprets its buffer as
    int16 — no error, no warning, twice as many samples, and the result is
    noise with a nearly constant level.

    It did not look broken. The clips played, the durations were plausible,
    and the table filled in. What it produced was emphasis 0.5 dB and zero
    pauses in EVERY condition — which reads exactly like "the model is flat
    whatever the prompt says", and was used to conclude that. Measured
    correctly the same renders give ~4 dB of emphasis and 7-13 interior
    pauses. A whole experiment was reported off it.

    The tell was there and was misread: RMS came out -14.6 dBFS ±0.1 in all
    eight cells. Two personas and two voices agreeing to a tenth of a dB is
    not a finding about personas, it is a signal that is not speech.
    """
    cond = OutboundConditioner()
    if not isinstance(pcm24, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"_to_phone needs PCM16 bytes, got {type(pcm24).__name__} — "
            f"OutboundConditioner.process reinterprets anything else through "
            f"np.frombuffer and returns noise without raising.")
    return _mulaw_decode(cond.process(pcm24))


# ── Acoustic measures ───────────────────────────────────────────────────────

def _f0(x: np.ndarray, sr: int = SR) -> list[float]:
    """Autocorrelation pitch per 40 ms frame, 70-300 Hz, voiced frames only.

    OCTAVE-CORRECTED, and it needs to be. The first run of this script reported
    a median of 171 Hz ±105 on a six-word line — a half-range that wide is the
    tracker picking the half-frequency peak on some takes and not others, and
    it made the two pitch columns pure noise. Autocorrelation is prone to it
    because the peak at 2T is nearly as tall as the one at T.
    """
    n = int(0.040 * sr)
    hop = n // 2
    lo, hi = int(sr / 300), int(sr / 70)
    raw = []
    for i in range(0, max(0, len(x) - n), hop):
        f = x[i:i + n] - np.mean(x[i:i + n])
        if np.sqrt(np.mean(f ** 2)) < 0.01:
            continue
        ac = np.correlate(f, f, mode="full")[n - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if not len(seg):
            continue
        k = int(np.argmax(seg)) + lo
        # A weak peak is an unvoiced frame, not a low note.
        if ac[k] / ac[0] > 0.35:
            raw.append(sr / k)
    if len(raw) < 4:
        return raw
    # Fold halves and doubles toward the utterance's own centre. The centre is
    # the median of the RAW values, which survives a minority of octave errors;
    # folding then pulls that minority in rather than discarding them, so a
    # genuinely wide-ranging utterance is not flattened.
    c = statistics.median(raw)
    out = []
    for v in raw:
        for cand in (v, v * 2.0, v / 2.0):
            if 0.6 * c <= cand <= 1.7 * c:
                out.append(cand)
                break
        else:
            out.append(v)
    return out


def _pauses(x: np.ndarray, floor=0.012, minlen=0.06, sr: int = SR):
    """Interior silences: (count, total seconds). Leading/trailing discarded."""
    n = int(0.010 * sr)
    lvl = np.array([np.sqrt(np.mean(x[i:i + n] ** 2))
                    for i in range(0, max(0, len(x) - n), n)])
    voiced = lvl >= floor
    if not voiced.any():
        return 0, 0.0
    a, b = int(np.argmax(voiced)), len(voiced) - int(np.argmax(voiced[::-1]))
    runs, cur = [], 0
    for v in voiced[a:b]:
        if v:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    keep = [r * 0.010 for r in runs if r * 0.010 >= minlen]
    return len(keep), float(sum(keep))


def measure(x: np.ndarray, words: int, sr: int = SR) -> dict:
    n = int(0.010 * sr)
    lvl = np.array([np.sqrt(np.mean(x[i:i + n] ** 2))
                    for i in range(0, max(0, len(x) - n), n)])
    voiced = lvl[lvl >= 0.012]
    f0 = _f0(x, sr)
    npause, spause = _pauses(x, sr=sr)
    dur = len(x) / sr
    voiced_s = len(voiced) * 0.010
    # EMPHASIS as the spread of level ACROSS the utterance, in dB, measured on
    # 100 ms windows. Crest factor is a peak-to-average ratio and is dominated
    # by a single plosive; what "emphasis" means to a listener is how much the
    # level moves from one syllable group to the next, which is this.
    w = int(0.100 * sr)
    blocks = [float(np.sqrt(np.mean(x[i:i + w] ** 2)))
              for i in range(0, max(0, len(x) - w), w)]
    bdb = [20 * np.log10(b) for b in blocks if b >= 0.012]
    return {
        "dur": dur,
        "voiced": voiced_s,
        "rate": (voiced_s / words) if words else float("nan"),
        "rms_db": 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12),
        "crest": (20 * np.log10(float(np.max(np.abs(x))) + 1e-12)
                  - 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)),
        "emph": statistics.pstdev(bdb) if len(bdb) > 2 else float("nan"),
        "f0_med": statistics.median(f0) if f0 else float("nan"),
        "f0_iqr": ((statistics.quantiles(f0, n=4)[2]
                    - statistics.quantiles(f0, n=4)[0])
                   if len(f0) > 3 else float("nan")),
        "npause": npause,
        "spause": spause,
    }


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


_COLS = [("rate", "s/word", ".3f"), ("rms_db", "RMS dBFS", ".1f"),
         ("emph", "emphasis dB", ".1f"), ("f0_med", "F0 med Hz", ".0f"),
         ("f0_iqr", "F0 IQR Hz", ".0f"), ("npause", "pauses", ".1f"),
         ("spause", "pause s", ".2f")]


def _vals(ms, k):
    return [m[k] for m in ms if m[k] == m[k]]


def _row(label, ms):
    cells = ""
    for k, _, fmt in _COLS:
        vs = _vals(ms, k)
        cells += (f"{'—':>16}" if not vs else
                  f"{statistics.mean(vs):>9{fmt}} ±"
                  f"{(max(vs) - min(vs)) / 2:<6{fmt}}")
    return f"  {label:<12}{cells}"


def _verdict(a, b):
    """Which measures separate? Non-overlapping ranges, stated as a rule.

    NOT a t-test. n is small and the takes are not independent draws from
    anything tidy — what is being asked is the weaker, honest question: do the
    two sets of takes even occupy different ground. If the ranges overlap, a
    difference in the means is not something to report as an effect.
    """
    out = []
    for k, name, fmt in _COLS:
        va, vb = _vals(a, k), _vals(b, k)
        if len(va) < 2 or len(vb) < 2:
            continue
        if max(va) < min(vb) or max(vb) < min(va):
            d = statistics.mean(vb) - statistics.mean(va)
            out.append(f"{name} ({d:+{fmt}})")
    return out


async def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="?", type=int, default=3)
    ap.add_argument("--voice", action="append", default=None,
                    help="repeatable; defaults to REALTIME_VOICE. marin and "
                         "cedar are measured to take steering differently, so "
                         "a null result on one is not a null result.")
    ap.add_argument("--short", action="store_true",
                    help="use the six-word line instead of the long one — the "
                         "version the first run used, kept so its null result "
                         "can be reproduced rather than argued about")
    ap.add_argument("--natural-takes", type=int, default=None,
                    help="takes for the words-free half (default: takes//2)")
    # WHICH HALVES TO RUN. Both default to everything, so every existing
    # invocation renders exactly what it rendered before. They exist because
    # the full cross product is voices x 2 cases x 2 tones x takes: asking for
    # "5 voices, 3 takes" spends 60 renders when the experiment wants 15, and
    # on company-funded credit that difference is the whole decision.
    ap.add_argument("--cases", default="scripted,natural",
                    help="comma-separated subset of scripted,natural "
                         "(default: both)")
    ap.add_argument("--tones", default="neutral,persona",
                    help="comma-separated subset of neutral,persona "
                         "(default: both)")
    args = ap.parse_args()

    _cases = {c.strip() for c in args.cases.split(",") if c.strip()}
    _tones = {s.strip() for s in args.tones.split(",") if s.strip()}
    _bad = ((_cases - {"scripted", "natural"})
            | (_tones - {"neutral", "persona"}))
    if _bad or not _cases or not _tones:
        raise SystemExit(
            f"--cases must be from scripted,natural and --tones from "
            f"neutral,persona; got {sorted(_bad)!r}")

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    takes = args.takes
    nat_takes = (args.natural_takes if args.natural_takes is not None
                 else max(2, takes // 2))
    voices = args.voice or [settings.realtime_voice]
    for v in voices:
        if v in ("marin", "cedar") and settings.realtime_model != "gpt-realtime-2":
            raise SystemExit(f"voice {v!r} requires REALTIME_MODEL=gpt-realtime-2")
    line = _SCRIPTED_SHORT if args.short else _SCRIPTED
    tpl = T.get_template("patient_discovery")
    doctor = Doctor(doctor_name="Dr. Jane Okafor",
                    hospital_name="Northside Medical Group")
    # Ignored by PatientPersonaTemplate; passed so the call type-checks.
    ctx = tpl.build_context(doctor, callback_number="", callback_email="")
    greet = tpl.build_greeting(doctor)
    prior = [("assistant", greet),
             ("user", "Yes, this is Dr. Okafor's office."),
             ("user", "She sees people at our Northgate clinic."),
             ("user", _BAD_NEWS)]
    out_dir = Path(__file__).resolve().parent.parent / "data" / "demo_audio"

    versions = [(n, i) for n, i in
                (("neutral", _neutral(tpl.instructions)),
                 ("persona", tpl.instructions))
                if n in _tones]
    _planned = sum((takes if c == 'scripted' else nat_takes)
                   for c in sorted(_cases)) * len(versions) * len(voices)
    print()
    print(f'PLAN   cases {",".join(sorted(_cases))}   '
          f'tones {",".join(n for n, _ in versions)}')
    print(f'PLAN   RENDERS {_planned}  (voices x cases x tones x takes)')
    print(f"\nmodel  {settings.realtime_model}\nvoices {', '.join(voices)}"
          f"\ntakes  {takes} scripted / {nat_takes} natural, per voice per tone"
          f"\nline   {len(line.split())} words\nout    {out_dir}\n")

    results: dict = {}
    for voice in voices:
        for case, text, n in (("scripted", line, takes),
                              ("natural", "", nat_takes)):
            if case not in _cases:
                continue
            for tone, ins in versions:
                ms, saids = [], []
                for t in range(1, n + 1):
                    pcm, said = await _render(voice, ins, ctx, prior, text)
                    if not pcm:
                        print(f"  {voice}/{tone}/{case} take {t}: NO AUDIO")
                        continue
                    x = _to_phone(pcm)
                    _write(out_dir /
                           f"persona_{voice}_{tone}_{case}_take{t}.wav", x)
                    ms.append(measure(x, len((said or text).split())))
                    saids.append(said)
                results[(voice, case, tone)] = (ms, saids)

    hdr = "  " + f"{'':<12}" + "".join(f"{nm:>16}" for _, nm, _ in _COLS)
    for voice in voices:
        for case in ("scripted", "natural"):
            a = results.get((voice, case, "neutral"), ([], []))
            b = results.get((voice, case, "persona"), ([], []))
            if not a[0] and not b[0]:
                continue
            print(f"\n{'=' * 122}")
            print(f"  {voice.upper()} · {case.upper()}"
                  + ("   — same words both sides; every difference is delivery"
                     if case == "scripted"
                     else "   — the model's own words; wording AND delivery move"))
            print("=" * 122)
            print(hdr)
            for tone, (ms, _) in (("neutral", a), ("persona", b)):
                if ms:
                    print(_row(tone, ms))
            if len(versions) < 2:
                # ONE ARM RAN. A verdict line here would read as a null
                # result about a comparison that was never made, which is
                # the one output shape this file exists to avoid.
                print()
                print(f'  ONE TONE ARM ONLY ({versions[0][0]}) — no '
                      f'comparison was run, so there is no verdict.')
            else:
                sep = _verdict(a[0], b[0])
                print("\n  SEPARATES: " + (", ".join(sep) if sep else
                                           "nothing — every range overlaps"))
            if case == "natural":
                print("  transcripts")
                for tone, (_, ss) in (("neutral", a), ("persona", b)):
                    for s in ss:
                        print(f"    {tone:<8} {s}")

    print(f"\n  Mean ±half-range. SEPARATES lists only measures whose ranges do "
          f"not overlap at all;\n  a difference in means with overlapping "
          f"ranges is not an effect worth reporting.")
    print("  Then play the files — none of this says whether it sounds right.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
