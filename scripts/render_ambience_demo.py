"""Render the ambience A/B/C/D so it can be settled by ear rather than argued.

    python scripts/render_ambience_demo.py            # real agent speech
    python scripts/render_ambience_demo.py --offline  # no API call, synthesised
    python scripts/render_ambience_demo.py --db -40 --duck-db -55 --attack 50

Writes to data/demo_audio/:

    ambience_on.wav    the four cases, in order, as the receptionist hears them
    ambience_off.wav   the identical speech with the feature off
    ambience_bed.wav   the bed alone, un-ducked, to hear what is being added

── WHAT MAKES THIS AN HONEST DEMO ──────────────────────────────────────────
It goes through the SAME AmbienceMixer a call uses, pulled one 20 ms frame at
a time exactly as the pump pulls it, and the file is the mu-law that came out.
Not a re-render at a nicer sample rate, not the mix computed a second way for
the demo: the bytes in ambience_on.wav are the bytes that would have gone to
Twilio, decoded once to make a WAV. That is the same rule the delta path holds
itself to — the recording is what was played, or it is worthless.

ambience_off.wav is the SAME conditioned speech through the same conditioner
with the mixer absent, so the pair isolates one variable. The gaps in it are
digital silence, which is what the line is today.

── THE FOUR CASES, AND WHERE TO LISTEN ─────────────────────────────────────
The timeline is one continuous take, printed with timestamps when it runs:

    A  agent speaking      the bed is pushed down under the voice
    B  agent silent        a quiet room, not a dead channel
    C  agent stops         B arrives by a fade, not a switch
    D  agent starts again  and leaves by one

C and D are the ones worth concentrating on, because they are the two the
cheap implementation gets wrong: a bed that stops and starts is more obviously
artificial than no bed at all. A short pause INSIDE the second utterance is
included on purpose — the bed should not come up for it.

── HEADPHONES, AND EXPECT IT TO BE SUBTLE ──────────────────────────────────
The bed sits at -45 dBFS. On laptop speakers in a normal room it is at or
below the noise floor of the room you are sitting in, and you will conclude it
does nothing. That is a property of the playback, not of the file.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.bootstrap  # noqa: F401,E402  (UTF-8 stdout on Windows)

from core.audio_utils import _mulaw_decode, _mulaw_encode      # noqa: E402
from core.config import settings                               # noqa: E402
from agents.voice import ambience as amb                       # noqa: E402
from agents.voice.outbound_audio import OutboundConditioner    # noqa: E402

_SR8 = 8_000
_SR24 = 24_000

# The line the agent says. Long enough to hold a natural internal pause, and
# it is the project's own opening rather than a lorem-ipsum: delivery is part
# of what is being judged.
_LINE = ("Hi, I'm looking for a new doctor for my mum, and I wanted to check "
         "whether Dr. Okafor is taking new patients at the moment.")

_SECOND = "Right, and does she need a referral from a GP first?"


# ── The timeline ─────────────────────────────────────────────────────────────
# (seconds of silence, then an utterance) — silence first so the file OPENS on
# case B and the listener hears the room before anything else happens.
def _timeline(u1: np.ndarray, u2: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """(label, lead-in silence seconds, voice) in order."""
    # The pause inside the second utterance is cut from the utterance itself
    # rather than inserted between two renders, so the two halves are one
    # continuous piece of speech with a comma-length gap in it — which is what
    # the hold has to survive.
    half = len(u2) // 2
    return [
        ("B  quiet room, nothing said yet",        2.5, np.zeros(0, dtype=np.float32)),
        ("A/D agent speaks - bed ducks under it",  0.0, u1),
        ("C  agent stops - bed fades back in",     3.0, np.zeros(0, dtype=np.float32)),
        ("D  agent speaks again",                  0.0, u2[:half]),
        ("   ...a 150ms pause INSIDE the sentence", 0.15, u2[half:]),
        ("C  and stops for good",                  3.5, np.zeros(0, dtype=np.float32)),
    ]


def _synth(seconds: float) -> bytes:
    """PCM16 24k that behaves like speech: formants, syllables, and gaps.

    The offline stand-in. It is not pretty and is not meant to be — what the
    duck reacts to is level over time, and this has the right level over time.
    """
    n = int(seconds * _SR24)
    t = np.arange(n) / _SR24
    rng = np.random.default_rng(7)
    car = (0.5 * np.sin(2 * np.pi * 130 * t)
           + 0.35 * np.sin(2 * np.pi * 720 * t)
           + 0.25 * np.sin(2 * np.pi * 1_900 * t)
           + 0.12 * np.sin(2 * np.pi * 2_900 * t))
    # Syllables at ~4.5 Hz with real gaps between words, and a slower phrase
    # envelope on top so it is not a machine gun.
    syl = np.clip(np.sin(2 * np.pi * 4.5 * t) * 1.6, 0.0, 1.0)
    phrase = 0.55 + 0.45 * np.clip(np.sin(2 * np.pi * 0.35 * t + 1.0), -1, 1)
    x = car * syl * phrase
    x += rng.normal(0, 0.004, n)                      # breath
    x *= 10 ** (-16.0 / 20) / np.sqrt(np.mean(x ** 2))
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


async def _speak(text: str, voice: str) -> bytes:
    """Real agent speech as PCM16 24k, through the same socket a call opens."""
    from scripts.render_ab_demo import _render          # noqa: PLC0415
    instructions = (
        "You are on a phone call. Say exactly the line you are given, once, "
        "in your natural speaking voice. Do not add anything to it.")
    pcm, said = await _render(voice, instructions, "You are on a phone call.",
                              text, [])
    print(f"    said: {said[:70]!r}")
    return pcm


def _write(path: Path, mulaw: bytes) -> float:
    """Decode the wire bytes once and write them. No second rendering."""
    pcm = _mulaw_decode(mulaw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR8)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return len(mulaw) / _SR8


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="synthesise the speech instead of calling the API")
    ap.add_argument("--voice", default=settings.realtime_voice)
    ap.add_argument("--db", type=float, default=None, help="resting level dBFS")
    ap.add_argument("--duck-db", type=float, default=None)
    ap.add_argument("--attack", type=float, default=None, help="ms")
    ap.add_argument("--release", type=float, default=None, help="ms")
    ap.add_argument("--hold", type=float, default=None, help="ms")
    args = ap.parse_args()

    if args.db is not None:      settings.realtime_ambience_db = args.db
    if args.duck_db is not None: settings.realtime_ambience_duck_db = args.duck_db
    if args.attack is not None:  settings.realtime_ambience_attack_ms = int(args.attack)
    if args.release is not None: settings.realtime_ambience_release_ms = int(args.release)
    if args.hold is not None:    settings.realtime_ambience_hold_ms = int(args.hold)
    settings.realtime_ambience = True

    offline = args.offline or not settings.openai_api_key
    if offline and not args.offline:
        print("no OPENAI_API_KEY — falling back to synthesised speech")
    if offline:
        pcm1, pcm2 = _synth(4.5), _synth(3.0)
    else:
        print(f"rendering two lines in {args.voice}...")
        pcm1 = asyncio.run(_speak(_LINE, args.voice))
        pcm2 = asyncio.run(_speak(_SECOND, args.voice))

    # CONDITIONED ONCE, BY THE LIVE CONDITIONER, and the two utterances share
    # one instance exactly as one call does — its filters, decimation phase and
    # compressor envelope carry between them.
    cond = OutboundConditioner()
    u1 = cond.process_pcm8(pcm1)
    u2 = cond.process_pcm8(pcm2)
    print(f"conditioner: {'on' if cond.enabled else 'OFF - ' + cond.disabled_reason}, "
          f"{len(u1) / _SR8:.1f}s + {len(u2) / _SR8:.1f}s of speech")

    mixer = amb.build(settings)
    if mixer is None:
        print("ambience did not build — is data/ambience/room_tone.wav there? "
              "run scripts/make_room_tone.py")
        return 1

    # ── drive the mixer exactly as the pump drives it ───────────────────────
    on, off, marks = bytearray(), bytearray(), []
    at = 0.0
    for label, lead_s, voice in _timeline(u1, u2):
        marks.append((at, label))
        if lead_s:
            mixer.push_silence(lead_s)
            off += b"\xff" * int(lead_s * _SR8)
            at += lead_s
        if voice.size:
            mixer.push_voice(voice)
            off += _mulaw_encode(voice)
            at += voice.size / _SR8
    # Frames, one at a time, until the queue is empty — the pump's own loop.
    while mixer.queued_samples > 0:
        on += mixer.next_frame()
    # Pad the OFF file to the same length so the two line up in an editor.
    if len(off) < len(on):
        off += b"\xff" * (len(on) - len(off))

    out = Path(__file__).resolve().parent.parent / "data" / "demo_audio"
    d_on = _write(out / "ambience_on.wav", bytes(on))
    _write(out / "ambience_off.wav", bytes(off[:len(on)]))

    # The bed alone, un-ducked, so "what is even being added" is answerable.
    bare = amb.AmbienceMixer(
        amb.RoomTone.load(str(Path(__file__).resolve().parent.parent
                              / "data" / "ambience" / "room_tone.wav")),
        amb.Ducker(ambient_db=settings.realtime_ambience_db,
                   duck_db=settings.realtime_ambience_db,
                   attack_ms=1, release_ms=1, hold_ms=0))
    _write(out / "ambience_bed.wav",
           b"".join(bare.next_frame() for _ in range(int(d_on * 50))))

    # ── what to listen for, and where ───────────────────────────────────────
    print(f"\nwrote {out}{chr(92)}ambience_on.wav   ({d_on:.1f}s) "
          f"<- what the receptionist hears")
    print(f"      {out}{chr(92)}ambience_off.wav  (same speech, feature off)")
    print(f"      {out}{chr(92)}ambience_bed.wav  (the bed alone, un-ducked)")
    print(f"\nsettings: bed {settings.realtime_ambience_db:.0f} dBFS, "
          f"duck {settings.realtime_ambience_duck_db:.0f} dBFS, "
          f"attack {settings.realtime_ambience_attack_ms}ms, "
          f"release {settings.realtime_ambience_release_ms}ms, "
          f"hold {settings.realtime_ambience_hold_ms}ms")
    print("\n  time    case")
    for t, label in marks:
        print(f"  {int(t) // 60}:{t % 60:05.2f}  {label}")

    # ── and the same thing as numbers, for anyone who wants both ────────────
    # Measured from the FILE, not from the gain state: a trace read out of the
    # ducker would agree with itself whatever the mixer actually wrote.
    sig = _mulaw_decode(bytes(on))
    win = int(0.10 * _SR8)
    lvl = np.array([_db(sig[i:i + win]) for i in range(0, len(sig) - win, win)])
    quiet = lvl[lvl < -35]
    print(f"\n  measured from ambience_on.wav, 100ms windows:")
    print(f"    speech windows   {float(np.max(lvl)):.1f} dBFS peak")
    print(f"    quiet windows    {float(np.median(quiet)):.1f} dBFS median "
          f"({len(quiet)} of {len(lvl)})")
    print(f"    quietest window  {float(np.min(lvl)):.1f} dBFS "
          f"(a dead channel measures -inf; mu-law's smallest step is -72.2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
