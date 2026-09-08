"""Is this voice steerable on DELIVERY at all — and if so, by how much?

    python scripts/probe_delivery_ceiling.py            # 6 takes per arm
    python scripts/probe_delivery_ceiling.py 8 --voice marin --voice cedar

── THE QUESTION compare_persona_delivery COULD NOT ANSWER ──────────────────
That script compares `_TONE_PATIENT` against `_TONE_ADMIN` and reports
"SEPARATES: nothing" — twice now, before and after the 2026-09-04 rebuild of
the block. That result is consistent with TWO very different worlds:

    A. the model does not take delivery direction from the prompt at all
    B. it does, and both arms are simply mild

A and B call for opposite responses. In A no wording of a tone block will ever
work and emotion has to come from voice choice or the audio layer. In B the
block is just under-dosed and the lever is real. **Comparing two mild variants
cannot tell them apart**, and every persona experiment on this project so far
has been exactly that comparison.

So this adds the arm that was missing: a deliberately EXTREME direction, to
find the ceiling. It is an INSTRUMENT, NOT A PROPOSED PROMPT. Nothing here is
meant to ship — the extreme arm exists to answer "does the needle move when it
is pushed as hard as it can be pushed", and its wording is chosen to be
unmistakable rather than good. If even that arm does not separate, world A is
true and no further tone-block edit is worth writing. If it does separate, the
lever exists and the only remaining question is dosage.

── FOUR ARMS, ONE LINE ─────────────────────────────────────────────────────
Every arm reads the SAME words, so nothing in the table can be an artefact of
one arm having more to say — the confound that split compare_persona_delivery
into a scripted half and a natural half.

    admin     _TONE_ADMIN, the block templates 1-3 actually ship
    patient   _TONE_PATIENT as it currently stands
    none      the tone slot emptied — is the block doing anything vs NOTHING?
    extreme   an unmistakable delivery direction (the ceiling probe)

`none` matters as much as `extreme`: if `patient` cannot be separated from an
EMPTY slot, the block is not earning the tokens it costs, which is a decision
about the prompt budget and not only about emotion.

Measurement, the WAV writing and the overlap rule are imported from
compare_persona_delivery rather than copied — a second implementation of
"which measures separate" is the two-places bug this repo keeps paying for,
and the F0 caveats in that file's docstring apply here unchanged.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.voice import templates as T                       # noqa: E402
from core.config import settings                              # noqa: E402
from core.models import Doctor                                # noqa: E402
from scripts.compare_persona_delivery import (                # noqa: E402
    _BAD_NEWS,
    _COLS,
    _SCRIPTED,
    _render,
    _vals,
    _row,
    _to_phone,
    _verdict,
    _write,
    measure,
)


# THE CEILING PROBE. Deliberately overdone, and deliberately about SOUND rather
# than about feeling: "tired" is a description the model can acknowledge
# without changing anything, while a rate and a pitch direction are things the
# vocoder has to act on. If this does not move the numbers, nothing milder
# will.
#
# NOT A CANDIDATE BLOCK. It would be a bad prompt — it asks for a performance,
# which is the exact failure the brief calls out ("do not make the patient
# dramatic, weak, theatrical"). It is here to bound the instrument.
_TONE_EXTREME = """\
- SPEAK NOTICEABLY SLOWLY AND QUIETLY. Draw the words out. Let your voice sit
  low and fall away at the end of every sentence. Leave a clear beat before you
  start speaking and another one in the middle of what you say. You are worn
  out and it is audible in every syllable: soft, heavy, unhurried, with none of
  the lift a healthy person's voice has. Never sound brisk, never sound bright,
  never sound energetic."""

_TONE_NONE = "- Speak naturally."


# ── THE DOSE LADDER ─────────────────────────────────────────────────────────
# The ceiling run settled that marin DOES take delivery direction (6/6 measures
# moved as commanded, p=0.016) and that `patient` and `none` are both
# indistinguishable from `admin`. So the shipped block is not inert-by-nature,
# it is under-dosed — and the question is no longer "does this work" but "how
# hard do you have to push before it moves, and is that dose still a person".
#
# THESE ARE CANDIDATES, unlike _TONE_EXTREME. Each one is small, and each names
# the four things the experiment showed the vocoder acts on — RATE, LOUDNESS,
# PITCH, PAUSES — instead of the adjectives about feeling that measured as
# nothing. Every one keeps the brief's clamp: not dramatic, not weak, not
# theatrical, no performed illness.
#
# THE LADDER IS THE POINT. One candidate would only tell you whether that one
# works. Three graded ones locate the threshold, and _dose_position reports
# where each sits between admin (0.0) and extreme (1.0), which is the number
# the choice actually turns on.
_TONE_DOSE_1 = """\
- A SHADE SLOWER AND SOFTER THAN YOUR NORMAL BASELINE. Not slow, not tired-
  sounding: a step under your usual pace and a step under your usual volume,
  the way anyone talks when they are not quite well. Pitch moves normally.
  Nothing else about your delivery changes."""

# Close to the wording proposed in review, kept nearly verbatim on purpose: it
# is a well-judged middle rung and the point of a ladder is that the rungs are
# not all mine.
_TONE_DOSE_2 = """\
- SLIGHTLY SLOWER, SOFTER AND LESS BRIGHT THAN YOUR NORMAL BASELINE. Speak at
  a slightly slower pace, with slightly reduced vocal energy and a modestly
  softer delivery. Let an occasional natural pause fall before you carry on
  with a thought. Keep pitch movement natural but a little subdued. Do not
  perform sadness or weakness."""

_TONE_DOSE_3 = """\
- CLEARLY SLOWER, SOFTER AND FLATTER THAN A HEALTHY CALLER, WITHOUT ACTING.
  Take your pace down a step and your volume down a step. Let a sentence
  settle at the end rather than lift. Leave a real beat before you answer
  something, and let one fall inside a longer sentence. None of the brightness
  of a well person — and none of the performance of an unwell one."""


# WHICH WAY EACH MEASURE SHOULD MOVE IF _TONE_EXTREME IS BEING OBEYED, read
# off its own words. Emphasis is not in here because the block does not command
# it either way, and scoring a measure nobody asked for would dilute the test.
_COMMANDED = {
    "rate":   +1,   # "speak noticeably slowly"        -> seconds per word UP
    "rms_db": -1,   # "and quietly"                    -> dBFS DOWN
    "f0_med": -1,   # "let your voice sit low"         -> median F0 DOWN
    "f0_iqr": -1,   # "fall away", none of the lift    -> spread DOWN
    "npause": +1,   # "leave a clear beat"             -> more pauses
    "spause": +1,   # "and another one in the middle"  -> longer total pause
}


def _direction(base: list, arm: list) -> tuple:
    """How many measures moved the way the instruction asked, and how likely.

    ── WHY THIS EXISTS, AND IT IS THE MOST IMPORTANT FUNCTION HERE ──────────
    _verdict asks whether two sets of takes occupy disjoint ground. That is the
    right question for "is this difference worth reporting", and it is the
    WRONG question for "does this voice respond to direction at all", because
    it throws away the one thing that separates a real effect from noise at
    small n: whether the movement is COHERENT.

    The first valid run of this script had `extreme` moving on every single
    measure in exactly the direction _TONE_EXTREME commands — slower, quieter,
    lower, flatter, more pauses, longer pauses — and _verdict reported
    "nothing, every range overlaps", so the script printed NO CEILING and
    recommended abandoning prompt-level delivery control. That recommendation
    was wrong and this is what catches it.

    Six independent measures under a null of random sign: 6/6 is p = 1/64.
    Not a strong claim, and it does not need to be — the alternative on offer
    was "no effect whatsoever", and one coherent run refutes that.
    """
    import math
    n = 0
    for k, want in _COMMANDED.items():
        va, vb = _vals(base, k), _vals(arm, k)
        if len(va) < 2 or len(vb) < 2:
            continue
        d = statistics.mean(vb) - statistics.mean(va)
        if d != 0 and (d > 0) == (want > 0):
            n += 1
    total = len(_COMMANDED)
    p = sum(math.comb(total, i) for i in range(n, total + 1)) / 2 ** total
    return n, total, p


def _dose_precision(base: list, top: list) -> float:
    """Roughly how many dose units one arm's sampling noise is worth.

    ── READ THIS BEFORE TRUSTING A DOSE NUMBER ─────────────────────────────
    `_dose_position` divides an arm's movement by the admin->extreme span, and
    on this voice that span is SMALL on most measures while the take-to-take
    spread is not. Measured on the completed 14-take run:

        measure   admin->extreme span   take spread   spread in dose units
        rate                    0.017         0.018                   1.06
        rms_db                 -0.900         1.100                   1.22
        f0_med                -15.000        16.000                   1.07
        npause                  2.500         2.500                   1.00
        f0_iqr                -18.000        13.000                   0.72
        spause                  1.140         0.530                   0.46

    On four of six, ordinary variation between takes is as large as the ENTIRE
    scale the dose is expressed on. So a dose of 0.29 against a dose of 0.42 is
    not a ranking, it is two draws from the same noise — and printing them to
    two decimals invites exactly the comparison they cannot support.

    THE DIRECTION TEST IS NOT AFFECTED and that is not luck: it aggregates the
    SIGN across six measures instead of trusting any one magnitude, which is
    what makes it survive per-measure noise this large. Use `_direction` to ask
    whether a dose works at all; do not use `_dose_position` to rank two doses
    that both do.

    Returns the median per-measure noise in dose units. Above ~0.5 the ranking
    is not meaningful at this sample size, and the honest way to choose between
    rungs is to listen to them.
    """
    out = []
    for k in _COMMANDED:
        va, vt = _vals(base, k), _vals(top, k)
        if min(len(va), len(vt)) < 3:
            continue
        span = statistics.mean(vt) - statistics.mean(va)
        if span == 0:
            continue
        # half-range, the same spread the table prints beside each mean
        half = (max(va) - min(va)) / 2
        out.append(abs(half / span))
    return statistics.median(out) if out else float("inf")


def _dose_position(base: list, arm: list, top: list) -> Optional[float]:
    """Where this arm sits between admin (0.0) and extreme (1.0).

    NOT A RANKING AT SMALL n. See _dose_precision, which reports how much of
    this scale is noise — on marin at 14 takes it is most of it.

    THE NUMBER THE DOSAGE CHOICE TURNS ON, and neither of the other two
    functions reports it. `_verdict` says whether ranges separate — at this n
    it says "no" for everything except nothing. `_direction` says whether the
    movement is coherent — it says "yes" for any dose that works at all, so it
    cannot rank them. What is missing is HOW FAR, and that is a fraction.

    Per commanded measure: (arm - admin) / (extreme - admin), so 0.0 is
    indistinguishable from the neutral block and 1.0 matches the deliberately
    overdone one. Averaged over the six, and the MEDIAN is taken rather than
    the mean because one measure with a tiny denominator can otherwise throw
    the whole figure.

    A measure whose extreme movement is negligible is skipped: dividing by it
    reports a ratio of two noise terms as a dose.
    """
    frac = []
    for k, _ in _COMMANDED.items():
        va, vb, vt = _vals(base, k), _vals(arm, k), _vals(top, k)
        if min(len(va), len(vb), len(vt)) < 2:
            continue
        a, b, t = (statistics.mean(v) for v in (va, vb, vt))
        span = t - a
        # 2% of the baseline is the floor for "the extreme arm moved this at
        # all". Below it the denominator is noise and the ratio is meaningless.
        if abs(span) < abs(a) * 0.02:
            continue
        frac.append((b - a) / span)
    return statistics.median(frac) if frac else None


def _is_verbatim(said: str, want: str) -> bool:
    """Did the take read the line, allowing for punctuation and transcription?

    Word sequence after stripping everything that is not a letter or a space,
    with a 90% prefix match rather than equality: the transcriber drops a
    trailing word often enough that demanding exact identity would throw away
    good takes, and a take that reads 34 of 37 words in order is reading the
    line. A take that answered in its own words shares almost no prefix and is
    nowhere near this bar.
    """
    def norm(s):
        return [w for w in "".join(
            c if (c.isalpha() or c.isspace()) else " "
            for c in (s or "").lower()).split() if w]
    a, b = norm(said), norm(want)
    if not a or not b:
        return False
    keep = int(len(b) * 0.9)
    return a[:keep] == b[:keep]


def _swap(instructions: str, block: str) -> str:
    """Put `block` in the tone slot. Refuses if the slot cannot be found.

    Same guard compare_persona_delivery._neutral carries, and for the same
    reason: a substitution that silently misses produces four identical arms
    and a confident null result. That is the most expensive failure this script
    has, because it looks exactly like the finding.
    """
    if T._TONE_PATIENT not in instructions:
        raise SystemExit(
            "probe_delivery_ceiling: _TONE_PATIENT is not in the built prompt "
            "— the substitution would silently no-op and every arm would be "
            "the same audio. Re-sync before trusting any number here.")
    return instructions.replace(T._TONE_PATIENT, block)


async def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="?", type=int, default=6)
    ap.add_argument("--voice", action="append", default=None)
    ap.add_argument("--with-none", action="store_true",
                    help="add the empty-tone-slot arm (already answered: "
                         "indistinguishable from admin)")
    args = ap.parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    voices = args.voice or [settings.realtime_voice]
    tpl = T.get_template("patient_discovery")
    doctor = Doctor(doctor_name="Dr. Jane Okafor",
                    hospital_name="Northside Medical Group")
    # Ignored by PatientPersonaTemplate; passed so the call type-checks.
    ctx = tpl.build_context(doctor, callback_number="", callback_email="")
    prior = [("assistant", tpl.build_greeting(doctor)),
             ("user", "Yes, this is Dr. Okafor's office."),
             ("user", "She sees people at our Northgate clinic."),
             ("user", _BAD_NEWS)]
    out_dir = Path(__file__).resolve().parent.parent / "data" / "demo_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    # `none` is dropped from the default ladder: the ceiling run already
    # settled that it is indistinguishable from admin, and every arm costs
    # `takes` renders. --with-none puts it back.
    arms = [("admin",   _swap(tpl.instructions, T._TONE_ADMIN)),
            ("patient", tpl.instructions)]
    if args.with_none:
        arms.append(("none", _swap(tpl.instructions, _TONE_NONE)))
    arms += [("dose1",   _swap(tpl.instructions, _TONE_DOSE_1)),
             ("dose2",   _swap(tpl.instructions, _TONE_DOSE_2)),
             ("dose3",   _swap(tpl.instructions, _TONE_DOSE_3)),
             ("extreme", _swap(tpl.instructions, _TONE_EXTREME))]

    print(f"\nmodel  {settings.realtime_model}"
          f"\nvoices {', '.join(voices)}"
          f"\ntakes  {args.takes} per arm"
          f"\nline   {len(_SCRIPTED.split())} words, identical in every arm\n")

    for voice in voices:
        print("=" * 118)
        print(f"  {voice.upper()}  — same words in all four arms; every "
              f"difference is delivery")
        print("=" * 118)
        results: dict[str, list] = {}
        attempted: dict[str, int] = {}
        kept: list = []
        for name, instructions in arms:
            ms, ok = [], 0
            for i in range(args.takes):
                # _render returns (pcm_bytes, transcript). Unpacking it is not
                # optional bookkeeping: _to_phone raises on anything that is
                # not PCM16 bytes precisely because passing it the wrong object
                # produced a confident null result once already — numpy
                # reinterpreted the buffer, gave back noise at a constant
                # level, and the run was used to conclude the model was flat.
                pcm, said = await _render(voice, instructions, ctx, prior,
                                          _SCRIPTED)
                if not pcm:
                    continue
                x = _to_phone(pcm)
                _write(out_dir / f"ceiling_{voice}_{name}_take{i + 1}.wav", x)
                # ── ONLY VERBATIM TAKES ENTER THE TABLE ─────────────────────
                # The premise of this whole comparison is that the words are
                # identical in every arm, so a take that answered in its own
                # words instead is not a slightly noisy sample — it is a
                # different measurement. The first run of this script kept
                # them, and the rows were built from 2-4 verbatim takes out of
                # 6: it printed "NO CEILING", confidently, off arms that were
                # not reading the same line.
                #
                # A 6,890-token prompt about conducting a phone call fights a
                # one-line recital request and wins some of the time — the
                # documented reason render_ab_demo cannot run a wording A/B
                # through a recital. The response is to OVERSAMPLE and discard,
                # not to average across takes that answer different questions.
                if not _is_verbatim(said, _SCRIPTED):
                    continue
                ok += 1
                # WHICH FILES ARE THE COMPARABLE ONES. Every take is written to
                # disk, including the ones discarded above, so a person or a
                # script picking files afterwards has no way to tell a take
                # that read the line from one that answered in its own words —
                # and comparing those by ear is the listening version of the
                # bug the verbatim filter just fixed in the table.
                kept.append({"arm": name, "take": i + 1,
                             "file": f"ceiling_{voice}_{name}_take{i + 1}.wav",
                             "said": said})
                # Word count from the TAKE, not from the script. Dividing a
                # take's duration by 37 when it said nine words reports a
                # speaking rate that never happened.
                ms.append(measure(x, len((said or _SCRIPTED).split())))
            results[name] = ms
            attempted[name] = args.takes
        # The manifest, beside the audio it describes.
        (out_dir / f"ceiling_{voice}_manifest.json").write_text(
            json.dumps(kept, indent=1), encoding="utf-8")
        header = "".join(f"{n:>16}" for _, n, _ in _COLS)
        print(f"  {'arm':<10}{header}")
        for name, _ in arms:
            print(_row(f"  {name}", results[name]))
        print("  verbatim takes kept: " + "  ".join(
            f"{n} {len(results[n])}/{attempted[n]}" for n, _ in arms))
        _thin = [n for n, _ in arms if len(results[n]) < 3]
        if _thin:
            print(f"  ** {', '.join(_thin)} kept fewer than 3 verbatim takes. "
                  f"Those rows are not evidence — re-run with more takes "
                  f"before drawing any conclusion from them. **")

        # AGAINST `admin`, THE CONTROL THAT SHIPS. Every arm is judged against
        # the same baseline so the comparisons are commensurable.
        print()
        base = results["admin"]
        for name, _ in arms[1:]:
            sep = _verdict(base, results[name])
            print(f"  {name:<9} vs admin : "
                  + (", ".join(sep) if sep else
                     "nothing — every range overlaps"))
        # THE ONE THAT DECIDES WHAT TO DO NEXT.
        ext = _verdict(base, results["extreme"])
        print()
        # THE CONCLUSION IS GATED ON HAVING THE EVIDENCE FOR IT. "No ceiling"
        # is the expensive claim — it ends a line of work — and the first run
        # printed it off 3 verbatim takes. A null result from a thin sample is
        # not a null result.
        # DIRECTION, NOT ONLY SEPARATION. Read _direction's docstring before
        # changing anything here: the overlap test alone called this wrong once
        # already, on the first valid run.
        _top = results["extreme"]
        for name, _ in arms[1:]:
            _n, _t, _p = _direction(base, results[name])
            _pos = _dose_position(base, results[name], _top)
            _posf = "   n/a" if _pos is None else f"{_pos:6.2f}"
            print(f"  {name:<9} direction  : {_n}/{_t} measures moved as "
                  f"commanded (p={_p:.3f})   dose {_posf}  "
                  f"(0.00 = admin, 1.00 = extreme)")
        _prec = _dose_precision(base, _top)
        print()
        if _prec > 0.5:
            print(f"  ** THE DOSE COLUMN CANNOT RANK THESE ARMS. One arm's "
                  f"sampling noise is worth\n     ~{_prec:.2f} dose units — "
                  f"the scale runs 0 to 1, so most of it is noise at this\n"
                  f"     sample size. Two doses differing by less than that "
                  f"are the same draw.\n     `direction` still holds (it "
                  f"aggregates signs, not magnitudes): use it to ask\n"
                  f"     whether a rung works at all, and your EARS to choose "
                  f"between rungs that do. **")
        else:
            print(f"  Dose precision ~{_prec:.2f} units — differences larger "
                  f"than that are readable.")
        print("  Either way the last step is listening: no column here can "
              "tell a believable\n  patient from an actor, and that is the "
              "property being bought.")
        _n, _t, _p = _direction(base, results["extreme"])
        print()
        if len(base) < 4 or len(results["extreme"]) < 4:
            print("  NOT ENOUGH VERBATIM TAKES TO CONCLUDE ANYTHING. "
                  f"admin {len(base)}, extreme {len(results['extreme'])}; "
                  f"4 each is the minimum.")
            print("  Re-run with more takes. Do NOT read the lines above as a "
                  "null result.")
        elif ext or _p <= 0.05:
            print(f"  CEILING FOUND — this voice DOES take delivery direction."
                  f"  {_n}/{_t} measures moved the commanded way (p={_p:.3f})"
                  + (f"; ranges separate on {', '.join(ext)}" if ext else
                     "; no single range separates, but the movement is"
                     " coherent, which noise is not."))
            print("  So a tone block CAN work and the shipped one is "
                  "under-dosed rather than inert. The open question is dosage,"
                  " and whether a dose that works is one you want to ship.")
        else:
            print(f"  NO CEILING — an unmistakable direction moved {_n}/{_t} "
                  f"measures the commanded way (p={_p:.3f}), which is chance.")
            print("  Stop editing the tone block. Emotion has to come from "
                  "voice choice or the audio layer.")
    print("\n  Mean +-half-range. SEPARATES lists only measures whose ranges do "
          "not overlap at all.\n  Then play the files — none of this says "
          "whether it sounds right.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
