"""Conditioning for the audio the caller actually hears.

WHY THIS EXISTS, measured rather than assumed. On six Twilio dual-channel
recordings the agent carries roughly HALF the energy the human caller carries
in 2-3.4 kHz — the band that holds s, t, f and sh, which is what "crisp" means:

    agent    80-300Hz 32.1%   .3-1k 49.7%   1-2k 16.5%   2-3.4k 1.42%
    caller   80-300Hz 45.1%   .3-1k 48.3%   1-2k  3.8%   2-3.4k 2.66%

Same call, same codec, same instrument, so the comparison is controlled. Two
things make it a property of what we SEND rather than of the line: the agent
measures 1.0-1.6% on every single call while callers scatter from 0.0% to 7.1%,
and the caller's voice has already been through the mobile network's AMR uplink
before Twilio recorded it — a handicapped reference that the agent still loses
to.

The agent is also about 5 dB peakier (crest 19-21 dB against the caller's
13-16 dB), so at matched peak level it sits lower against line noise.

WHAT THIS DOES NOT CLAIM. It does not make the call wideband — 8 kHz g711 is
the ceiling on PSTN and nothing here changes that. It moves the energy we do
send into the band that carries intelligibility, and it evens out the peaks so
average level rises without the peaks clipping.

THE TARGET IS FALSIFIABLE, which is the point of picking the caller as the
reference rather than a taste judgement: match the person on the other end of
the same line. Measured offline on call-20260825-1847's agent audio —

    as sent today                       2-3.4kHz  0.98%
    + presence lift                     2-3.4kHz  2.74%
    + compression                       2-3.4kHz  3.26%
    human caller, same call                       3.50%

── WHY THIS IS NOT A FUNCTION ──────────────────────────────────────────────
Every stage here carries state across chunk boundaries, and that is the whole
reason this is a class and not the one-line `_convert_oai_to_twilio` it
replaces. That function resampled EACH 400ms delta independently, so the
anti-alias filter restarted cold at every seam:

    error RMS   0.00108      SNR 48.6 dB      <- looks harmless
    error peak  0.09014                       <- a -15 dB transient
    share of error energy within 5ms of a chunk boundary:  100%
        (those boundaries are 2% of the samples)

The aggregate SNR is fine and the distribution is not: every bit of that error
sits at the seams, 2.5 times a second. It was inert only because pcmu
passthrough bypassed the function entirely — turning conditioning on without
fixing it would have traded a dull line for a ticking one.
"""
from __future__ import annotations

import logging

import numpy as np

from core.audio_utils import _mulaw_encode

log = logging.getLogger(__name__)

# 24000 / 8000 = 3 EXACTLY, which is why decimation is used here rather than
# scipy.signal.resample_poly. resample_poly has no public way to carry filter
# state between calls, so a streaming caller must either re-filter overlapping
# history or accept the seam artefact above. An integer ratio needs neither: a
# stateful low-pass at 24 kHz followed by taking every third sample is the same
# operation with a handle on its own memory.
_OAI_SR = 24_000
_TWILIO_SR = 8_000
_DECIM = _OAI_SR // _TWILIO_SR

# Presence band and lift. From the offline experiment, not from taste.
_PRESENCE_LO_HZ = 1_800.0
_PRESENCE_HI_HZ = 3_400.0
_PRESENCE_DB = 5.0

# Anti-alias corner. The g711 band is 300-3400 Hz and Nyquist after decimation
# is 4 kHz, so everything above 3.4 kHz in OpenAI's 24 kHz output has to be gone
# before the third sample is taken or it folds back into speech. Elliptic, not
# Butterworth: the transition from 3.4 kHz to 4 kHz is narrow, and an order-8
# Butterworth only reaches about 23 dB of rejection by 4 kHz. 0.5 dB of
# passband ripple buys 60 dB of stopband, which is the right trade here.
_AA_HZ = 3_400.0
_AA_ORDER = 8
_AA_PASS_RIPPLE_DB = 0.5
_AA_STOP_ATTEN_DB = 60.0

# Compressor. Threshold sits below the measured -16 dBFS speech level so the
# ratio acts on ordinary speech rather than only on peaks.
_COMP_THRESH_DB = -26.0
_COMP_RATIO = 3.0
_COMP_ATTACK_S = 0.005
_COMP_RELEASE_S = 0.080

# MAKEUP IS A CONSTANT, not a per-chunk normalise: gain set from whatever was
# loudest in each 400ms window would pump audibly between windows.
#
# It was once computed from the compressor parameters at a -16 dBFS operating
# point. That under-compensated by ~1.5 dB — the formula assumes the compressor
# sees RMS, but the envelope follower sits above RMS — so the computed constant
# is gone and the value below is measured. Swept on call-20260825-1847:
# assumes the compressor sees the RMS level; the envelope follower sits above
# RMS and below peak, so the real gain reduction on speech is larger than the
# formula predicts. Swept against call-20260825-1847's agent audio:
#
#     makeup   2-3.4kHz   level      crest    at ceiling
#      6.67       3.41%   -17.7 dB   16.3 dB     0.000%
#      8.17       3.43%   -16.2 dB   15.7 dB     0.001%
#      9.67       3.40%   -14.7 dB   14.2 dB     0.003%
#     (input, unconditioned:  0.99%  -16.2 dB   16.0 dB)
#
# 8.17 is the LEVEL-PRESERVING value: output lands on the same -16.2 dBFS the
# unconditioned path produces, so this stage changes spectrum and dynamics and
# explicitly does NOT change loudness. Conditioning that quietly made the agent
# quieter would be working against the complaint it was built for.
#
# Louder is available and is deliberately not taken. 9.67 buys another 1.5 dB
# of crest reduction, which would help on a noisy mobile leg — but it is a
# separate decision from the presence fix, it was not what the experiment
# tested, and raising output level on a call whose levels differ from this one
# is how a ceiling gets hit. Change it on evidence from a live call, not here.
_MAKEUP_DB = 8.17

# Headroom below full scale, so the presence lift cannot drive mu-law into its
# ceiling on a loud syllable. mu-law clipping is not graceful.
_CEILING = 0.95


# ── scipy, imported ONCE, AT MODULE LEVEL ───────────────────────────────────
# NOT lazily inside __init__ or process(), and the reason is a real failure
# rather than style. A deferred import runs inside whatever context the caller
# happens to be in, and the test suite wraps calls in
#
#     mock.patch.dict("sys.modules", {...})
#
# which snapshots sys.modules on entry and RESTORES it on exit. scipy imported
# for the first time inside that block is evicted when the block ends, and
# scipy's compiled extensions cannot be loaded a second time in one process:
#
#     ImportError: cannot load module more than once per process
#
# The conditioner then disabled itself for the rest of the run and every audio
# check downstream measured the unconditioned path while looking like it had
# measured the conditioned one. Importing here binds scipy when this module is
# first imported, which is before any patching, and the whole failure mode goes
# away.
#
# The filter design is deterministic and constant, so it is done once here too
# rather than per call.
_SOSFILT = None
_SOSFILT_ZI = None
_AA_SOS = None
_PRESENCE_SOS = None
DISABLED_REASON = ""

try:
    from scipy.signal import butter, ellip, sosfilt, sosfilt_zi

    _nyq = _OAI_SR / 2.0
    _AA_SOS = ellip(_AA_ORDER, _AA_PASS_RIPPLE_DB, _AA_STOP_ATTEN_DB,
                    _AA_HZ / _nyq, btype="low", output="sos")
    _PRESENCE_SOS = butter(2, [_PRESENCE_LO_HZ / _nyq, _PRESENCE_HI_HZ / _nyq],
                           btype="band", output="sos")
    _SOSFILT, _SOSFILT_ZI = sosfilt, sosfilt_zi
except Exception as exc:                      # pragma: no cover - env-dependent
    # THE REASON IS KEPT, not swallowed. A conditioner that silently turns
    # itself off looks exactly like one that is working — audio still flows,
    # the call still completes, and the only symptom is the dull line this
    # module exists to fix. Whoever finds it off has to be told why in the same
    # breath. Recording this is what surfaced the sys.modules failure above,
    # which had otherwise presented as "the tests pass".
    DISABLED_REASON = f"{type(exc).__name__}: {exc}"
    log.warning("[Realtime] outbound audio conditioning unavailable — %s",
                DISABLED_REASON)


def _design():
    """Filter sections, or None when scipy is unavailable.

    Returning None rather than falling back to something approximate is
    deliberate — see the fallback in audio_utils.resample, which drops to
    np.interp with no anti-alias filter at all. That is silent and badly
    aliased, and it is exactly the failure this module exists to avoid. Better
    to send unconditioned audio, which is merely dull, than aliased audio.
    """
    if _AA_SOS is None or _PRESENCE_SOS is None or _SOSFILT_ZI is None:
        return None
    return _AA_SOS, _PRESENCE_SOS, _SOSFILT_ZI


class OutboundConditioner:
    """One per call. Stateful, and must be fed the deltas in order.

    `process` takes one PCM16 24 kHz chunk as it arrives from OpenAI and returns
    the mu-law 8 kHz bytes to hand Twilio. Chunk boundaries are invisible in the
    output: the filters keep their memory, the decimator keeps its phase, and
    the compressor keeps its envelope.
    """

    def __init__(self, *, presence_db: float = _PRESENCE_DB,
                 compress: bool = True) -> None:
        self.enabled = False
        self._presence_gain = 10.0 ** (presence_db / 20.0) - 1.0
        self._compress = compress
        self._makeup = 10.0 ** (_MAKEUP_DB / 20.0)
        # Decimation phase. A chunk whose length is not a multiple of 3 would
        # otherwise shift which samples are kept, and a phase that jumps
        # between chunks is a click at every seam — the same defect as the
        # filter restarting, arriving by a different route.
        self._phase = 0
        self._env = 0.0
        self._atk = float(np.exp(-1.0 / (_COMP_ATTACK_S * _TWILIO_SR)))
        self._rel = float(np.exp(-1.0 / (_COMP_RELEASE_S * _TWILIO_SR)))
        self.disabled_reason = ""
        d = _design()
        if d is None:
            self.disabled_reason = DISABLED_REASON or "unknown"
            log.warning("[Realtime] outbound audio conditioning is OFF, "
                        "sending unprocessed — %s", self.disabled_reason)
            return
        self._aa, self._presence, _zi = d
        # Filter memory, initialised to the steady state for a zero input so
        # the first chunk does not open with a transient.
        self._zi_aa = _zi(self._aa) * 0.0
        self._zi_pres = _zi(self._presence) * 0.0
        self.enabled = True

    # ── the chain ────────────────────────────────────────────────────────────

    def process(self, pcm16: bytes) -> bytes:
        """PCM16 24 kHz in, mu-law 8 kHz out."""
        return _mulaw_encode(self.process_pcm8(pcm16))

    def process_pcm8(self, pcm16: bytes) -> np.ndarray:
        """The same chain, stopping one step short: conditioned float32 at 8 kHz.

        SPLIT OUT SO THE ENCODE HAPPENS ONCE. When ambience is on, the bed is
        mixed into this signal before it is encoded (agents/voice/ambience.py),
        and mu-law is the wrong domain to add anything in — the mixer would
        have to decode what this just encoded, sum, and encode again, putting a
        second quantisation on every sample of speech for no reason.

        `process` is unchanged and still returns bytes, so every existing
        caller and the whole seam suite are untouched: the encode simply moved
        from the end of this function to the start of that one.
        """
        # ODD LENGTH IS A TRUNCATED FRAME, NOT A CRASH. np.frombuffer raises
        # ValueError on a buffer that is not a whole number of int16s, and this
        # runs inside the OpenAI event pump — an exception here takes down the
        # audio path mid-call. Drop the stray byte and carry on; half a sample
        # is inaudible and a dead call is not.
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        # `sosfilt is None` is implied by `not self.enabled` and pyright cannot
        # see that, so it is re-asserted here rather than suppressed. Cheap, and
        # a suppression would be a standing lie about an invariant that could
        # later stop holding.
        sosfilt = _SOSFILT
        if not self.enabled or sosfilt is None or x.size == 0:
            # UNREACHABLE IN PRODUCTION, and deliberately so: without a filter
            # _outbound_conditioned() returns False, the session negotiates
            # μ-law and this object is never fed. It is kept for direct callers
            # (the tests among them) and it DOES alias — decimating unfiltered
            # audio folds 4-8 kHz back into speech. That is the whole reason the
            # availability check lives upstream rather than here.
            return x[::_DECIM] if x.size else np.zeros(0, dtype=np.float32)

        # 1. PRESENCE, at 24 kHz where the band is still intact. Added to the
        #    dry signal rather than replacing it, so this is a shelf-like lift
        #    and not a band-pass — the rest of the voice is untouched.
        band, self._zi_pres = sosfilt(self._presence, x, zi=self._zi_pres)
        y = x + band * self._presence_gain

        # 2. ANTI-ALIAS, before anything is thrown away.
        y, self._zi_aa = sosfilt(self._aa, y, zi=self._zi_aa)

        # 3. DECIMATE, phase carried across the boundary.
        start = (-self._phase) % _DECIM
        y = y[start::_DECIM]
        self._phase = (self._phase + x.size) % _DECIM

        # 4. COMPRESS, at 8 kHz — cheaper, and the envelope belongs on the
        #    signal that is actually sent.
        if self._compress and y.size:
            y = self._apply_compression(y)

        return np.clip(y, -_CEILING, _CEILING)

    def _apply_compression(self, y: np.ndarray) -> np.ndarray:
        """Envelope follower with asymmetric attack/release, state carried.

        The loop is over 8 kHz samples — 8000 a second, a few hundred
        microseconds of Python per second of audio — which is well inside the
        budget for a path that already spends milliseconds on filtering. A
        vectorised release with a separate attack pass was tried and is harder
        to read for no measurable gain.
        """
        env = np.empty_like(y)
        e = self._env
        for i, v in enumerate(np.abs(y)):
            coef = self._atk if v > e else self._rel
            e = coef * e + (1.0 - coef) * float(v)
            env[i] = e
        self._env = e
        env_db = 20.0 * np.log10(np.maximum(env, 1e-6))
        over = np.maximum(0.0, env_db - _COMP_THRESH_DB)
        gain = 10.0 ** ((-over * (1.0 - 1.0 / _COMP_RATIO)) / 20.0)
        return y * gain * self._makeup
