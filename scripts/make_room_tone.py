"""Build the ambient room tone the outbound mixer lays under the call.

    python scripts/make_room_tone.py --from room_noise.mp3     # a real recording
    python scripts/make_room_tone.py --synth                   # no source file
    python scripts/make_room_tone.py --from x.mp3 --seconds 30 --crossfade 3

Mono PCM16 at 8 kHz - the rate the Twilio leg actually runs at, so the mixer
never resamples and nothing above the g711 ceiling is carried around only to be
thrown away. agents/voice/ambience.py reads it once per call and loops it for
the life of the call.

-- THE SHIPPED ASSET, AND WHERE IT CAME FROM -------------------------------
    Pixabay "Room Noise", id 58390, 2:00
    https://pixabay.com/sound-effects/film-special-effects-room-noise-58390/
    sha256(mp3) 0464b60b7bc0e157030d7aa06a7f435b93118376d682631bb92e8d83b47b9d7e

    Pixabay Content License: free for commercial use, no attribution required.
    It forbids redistributing content UNCHANGED on a standalone basis, which is
    not what this is - the clip is high-passed, decimated to 8 kHz, windowed,
    crossfaded and mixed 45 dB under a phone call.

    The source mp3 is deliberately NOT in the repo. The URL and the checksum
    above are what make the asset re-derivable; 2.3 MB of source carried around
    for a 469 KB artefact is not.

-- IT WAS CHOSEN BY MEASUREMENT, AND THE NAMES WOULD HAVE MISLED YOU -------
Eight candidates were scored on what this use actually needs. The obvious pick
by name - "Empty Room Tone 4", also the longest - carries a 25.6 dB whine at
2.5 kHz. "Room Tone Office 12" has a 28.3 dB hum at 592 Hz. The winner is the
flattest thing in the list:

    clip                          tonality   level range   note
    Room Noise 58390                 1.5 dB       1.1 dB   <- shipped
    Room Ambience.wav                9.1 dB      15.9 dB   an event in it
    Big Room Ambience 1             15.2 dB      15.1 dB
    Room Tone 1                     17.2 dB      12.2 dB
    Ambient Empty Room Noise        18.7 dB       2.7 dB
    room tone 42                    19.3 dB      11.0 dB
    005487 Empty Room Tone 4        25.6 dB      10.4 dB   whine at 2.5 kHz
    Room Tone Office 12             28.3 dB       3.6 dB   hum at 592 Hz

"tonality" is the loudest narrow peak over its own neighbours in the long-term
spectrum: a hum or a whine stands 15-30 dB proud where noise sits a couple of
dB. "level range" is p95-p5 of the per-100ms level.

A spectral-autocorrelation "harmonic comb" test was also run and THROWN AWAY.
On controls it scored pure pink noise 0.774 against 0.520 for a bed with a real
50 Hz comb buried in it - it was measuring the detrending, not the signal. An
uncontrolled statistic that happens to be alarming is not evidence, and it
would have rejected the best clip in the set.

-- THE LOOP IS CROSSFADED, AND FOR A RECORDING IT HAS TO BE ----------------
`--synth` builds the bed as the inverse FFT of a random-phase spectrum, which
is circularly continuous BY CONSTRUCTION - one period of a periodic function,
so there is no seam to hide and no crossfade needed. A recording has no such
property: sample N-1 and sample 0 are unrelated, and butting them together is a
click once per loop, forever.

So `--from` takes seconds+crossfade of audio and folds the tail back into the
head with equal-power (sqrt) weights, which holds RMS across the blend because
the two stretches are uncorrelated. Both paths are then checked the same way:
the step across the wrap has to sit inside the file's own distribution of
adjacent sample steps, and THE SCRIPT REFUSES TO WRITE if it does not. An asset
with an audible loop point is worse than no asset, because the failure only
turns up on a live call.

-- THE WINDOW IS CHOSEN, NOT TAKEN FROM THE TOP ---------------------------
A two-minute recording can hold one cough, one chair, one door. Taking the
first 33 seconds is how that ends up under every call. `--from` scores every
candidate window by its own level range, keeps the calmest, and prints where it
landed so the choice is visible rather than implicit.

SEEDED, so `--synth` re-runs produce the identical file, and `--from` is
deterministic given the same source.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.bootstrap  # noqa: F401,E402  (UTF-8 stdout on Windows)

_SR = 8_000
_SEED = 20260904

# Band. The g711 passband is 300-3400 Hz; shaping just inside it means the
# level we set is the level that survives the codec.
_HP_HZ, _HP_ORDER = 250.0, 2
_LP_HZ, _LP_ORDER = 3_300.0, 4

# Room modes for --synth. (centre Hz, lift dB, gaussian width Hz). Wide and
# gentle: a narrow high-gain peak is a whistle, and a whistle is a tell.
_MODES = ((420.0, 5.0, 110.0), (1_150.0, 3.5, 220.0))

# Slow breathing for --synth. +/-1.5 dB, everything under 0.35 Hz.
_DRIFT_DB = 1.5
_DRIFT_HZ = 0.35

# Level the FILE is written at. Not the level it plays at - ambience.RoomTone
# normalises to unit RMS on load and the dB settings take it from there - so
# this only has to be high enough that 16-bit quantisation in the asset is
# irrelevant once it is scaled down to -45 dBFS. -18 dBFS leaves ~78 dB over
# the quantiser and peaks short of full scale by more than noise's own crest
# factor can close.
_FILE_RMS_DB = -18.0


# -- --synth -----------------------------------------------------------------

def _circular_noise(n, rng, shape):
    """One period of a periodic random signal with the given magnitude shape.

    Random phase, fixed magnitude, inverse real FFT. DC is zeroed (a noise bed
    with an offset is a click when it loops) and the Nyquist bin is left real,
    which irfft requires.
    """
    mag = shape.astype(np.float64).copy()
    phase = rng.uniform(0.0, 2.0 * np.pi, mag.size)
    mag[0] = 0.0
    phase[0] = 0.0
    if n % 2 == 0:
        phase[-1] = 0.0
    return np.fft.irfft(mag * np.exp(1j * phase), n)


def _band_shape(freq):
    """Pink, band-limited to the telephone band, with two room modes."""
    f = np.maximum(freq, 1e-6)
    pink = 1.0 / np.sqrt(np.maximum(f, 20.0))
    hp = (f / _HP_HZ) ** _HP_ORDER / np.sqrt(1.0 + (f / _HP_HZ) ** (2 * _HP_ORDER))
    lp = 1.0 / np.sqrt(1.0 + (f / _LP_HZ) ** (2 * _LP_ORDER))
    mag = pink * hp * lp
    for fc, gain_db, width in _MODES:
        lift = 10.0 ** (gain_db / 20.0) - 1.0
        mag = mag * (1.0 + lift * np.exp(-0.5 * ((f - fc) / width) ** 2))
    return mag


def render_synth(seconds: float, seed: int = _SEED) -> np.ndarray:
    """float64, exactly one loop period long, circularly continuous."""
    n = int(round(seconds * _SR))
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n, 1.0 / _SR)

    bed = _circular_noise(n, rng, _band_shape(freq))
    bed /= np.sqrt(np.mean(bed ** 2))

    # The drift envelope, circular for the same reason the bed is: multiplying
    # a circular signal by a non-circular one puts the seam straight back.
    drift = _circular_noise(n, rng, np.exp(-0.5 * (freq / _DRIFT_HZ) ** 2))
    drift /= max(np.max(np.abs(drift)), 1e-12)
    return bed * 10.0 ** (_DRIFT_DB * drift / 20.0)


# -- --from ------------------------------------------------------------------

def _decode(path: Path) -> tuple[np.ndarray, int]:
    """Any audio file -> mono float64 at its own rate.

    soundfile first because it is exact for wav/flac; PyAV for the compressed
    formats it cannot open. ffmpeg is not required and is not installed here.
    """
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return x.mean(axis=1), int(sr)
    except Exception:
        pass
    import av
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        sr = int(st.rate or 48_000)
        rs = av.audio.resampler.AudioResampler(format="dbl", layout="mono", rate=sr)
        out = []
        for fr in c.decode(st):
            for r in rs.resample(fr):
                out.append(r.to_ndarray().ravel())
        for r in (rs.resample(None) or []):
            out.append(r.to_ndarray().ravel())
    if not out:
        raise SystemExit(f"no audio decoded from {path}")
    return np.concatenate(out), sr


def _to_8k(x: np.ndarray, sr: int) -> np.ndarray:
    """Down to 8 kHz with a real anti-alias filter.

    resample_poly, NOT the streaming decimator in outbound_audio: this is an
    offline build with the whole signal in hand, so there are no chunk seams to
    carry filter state across and no reason to hand-roll it.
    """
    if sr == _SR:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), _SR)
    return resample_poly(x, _SR // g, sr // g)


def _highpass(x: np.ndarray) -> np.ndarray:
    """Kill DC and the sub-band rumble g711 discards anyway.

    The shipped source carries a -0.017 DC offset, which is a step at the loop
    point and level spent on something no phone will reproduce.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, _HP_HZ / (_SR / 2), btype="high", output="sos")
    return sosfiltfilt(sos, x)


def _calmest_window(x: np.ndarray, need: int) -> int:
    """Start index of the steadiest `need` samples, by per-second level range.

    A recording can hold one cough, one chair, one door; taking the top of the
    file is how that ends up under every call.
    """
    if x.size <= need:
        return 0
    w = _SR
    lv = np.array([20 * np.log10(max(float(np.sqrt(np.mean(x[i:i + w] ** 2))), 1e-12))
                   for i in range(0, x.size - w, w)])
    span = max(1, need // w)
    best, at = np.inf, 0
    for i in range(0, len(lv) - span + 1):
        seg = lv[i:i + span]
        score = float(seg.max() - seg.min())
        if score < best:
            best, at = score, i
    return at * w


def _crossfade_loop(x: np.ndarray, n: int, fade: int) -> np.ndarray:
    """Fold the tail back into the head so a modulo read has no seam.

    Equal-power (sqrt) weights, which hold RMS across the blend because the two
    stretches are uncorrelated. `x` must be at least n + fade long.

    THE FADE GOES ON THE HEAD, not the tail, and that is the whole trick:
    out[n-1] is x[n-1] and out[0] is x[n], which are ADJACENT IN THE SOURCE. A
    reader wrapping from the last sample to the first therefore crosses a step
    the recording already contained.
    """
    out = x[:n].copy()
    if fade <= 0:
        return out
    w = np.linspace(0.0, 1.0, fade, endpoint=False)
    out[:fade] = x[:fade] * np.sqrt(w) + x[n:n + fade] * np.sqrt(1.0 - w)
    return out


def ingest(path: Path, seconds: float, fade_s: float) -> np.ndarray:
    need = int(round(seconds * _SR))
    fade = int(round(fade_s * _SR))
    raw, sr = _decode(path)
    print(f"  decoded  {raw.size / sr:.1f}s at {sr} Hz "
          f"(sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}...)")
    x = _highpass(_to_8k(raw, sr))
    if x.size < need + fade:
        raise SystemExit(f"source is {x.size / _SR:.1f}s, need "
                         f"{(need + fade) / _SR:.1f}s for a {seconds:.0f}s loop "
                         f"with a {fade_s:.0f}s crossfade")
    at = _calmest_window(x, need + fade)
    print(f"  window   {at / _SR:.0f}s-{(at + need + fade) / _SR:.0f}s "
          f"(the calmest {(need + fade) / _SR:.0f}s in the file)")
    return _crossfade_loop(x[at:at + need + fade], need, fade)


# -- shared ------------------------------------------------------------------

def write_wav(path: Path, sig: np.ndarray, rms_db: float | None = None) -> None:
    x = sig
    if rms_db is not None:
        x = x * 10.0 ** (rms_db / 20.0) / np.sqrt(np.mean(x ** 2))
    pcm = np.clip(x, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes((pcm * 32767.0).astype(np.int16).tobytes())


def _report(sig: np.ndarray) -> bool:
    """Measure what was built. False means the seam is audible.

    MEASURED FROM THE SIGNAL, not asserted from the parameters - a build script
    that prints its own inputs back is not telling anyone anything.
    """
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))
    step = np.abs(np.diff(sig))
    wrap = abs(float(sig[0] - sig[-1]))
    p999 = float(np.percentile(step, 99.9))
    print(f"  rms      {20 * np.log10(rms):+.1f} dBFS")
    print(f"  peak     {20 * np.log10(peak):+.1f} dBFS "
          f"(crest {20 * np.log10(peak / rms):.1f} dB)")
    print(f"  wrap     {wrap:.6f}  vs median adjacent step "
          f"{float(np.median(step)):.6f}, p99.9 {p999:.6f}")
    ok = wrap <= p999
    print(f"  seam     {'SEAMLESS' if ok else 'AUDIBLE - refusing to write'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from", dest="src", default=None,
                     help="an audio file to build the bed from (mp3, wav, ...)")
    src.add_argument("--synth", action="store_true",
                     help="synthesise instead, when there is no source file")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--crossfade", type=float, default=3.0,
                    help="loop crossfade, seconds (--from only; --synth needs none)")
    ap.add_argument("--seed", type=int, default=_SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="also write room_tone_preview.wav at the -45 dBFS "
                         "playing level, which is what it actually sounds like")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else root / "data" / "ambience" / "room_tone.wav"

    if args.src:
        print(f"building from {args.src}")
        sig = ingest(Path(args.src), args.seconds, args.crossfade)
    else:
        if not args.synth:
            ap.error("pass --from <audio file>, or --synth to synthesise one")
        print("synthesising")
        sig = render_synth(args.seconds, args.seed)

    sig = sig * 10.0 ** (_FILE_RMS_DB / 20.0) / np.sqrt(np.mean(sig ** 2))
    if not _report(sig):
        return 1

    write_wav(out, sig)
    print(f"wrote {out}  ({args.seconds:.0f}s, {out.stat().st_size / 1024:.0f} KB)")
    if args.preview:
        # INTO demo_audio, NOT next to the asset. data/ambience/*.wav is
        # re-admitted to git so the bed itself ships; a preview written there
        # would be committed alongside it on the next add. demo_audio stays
        # ignored, which is what a render is.
        pv = root / "data" / "demo_audio" / (out.stem + "_preview.wav")
        write_wav(pv, sig, rms_db=-45.0)
        print(f"wrote {pv}  (at the -45 dBFS playing level)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
