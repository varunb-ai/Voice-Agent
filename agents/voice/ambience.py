"""Room tone on the outbound leg, ducked under the agent's voice.

WHAT THE RECEPTIONIST HEARS TODAY, between our turns, is digital silence — not
a quiet line, *nothing*. Twilio is sent no frames at all when no response is
playing, and 0xFF mu-law is a dead channel. A person on the other end of a real
phone call is never in that. The absence is one of the tells that survives
after latency, prosody and backchannels are all fixed.

── WHY THIS IS A PUMP AND NOT A GAIN STAGE ─────────────────────────────────
Twilio QUEUES media frames, it does not mix them. That single fact decides the
whole architecture, and it is the same fact `_playback_ends_at` rests on: audio
plays at realtime from the moment the first frame is handed over, and anything
sent afterwards lands *behind* it.

So "keep a bed under the call" cannot be done by adding frames alongside the
voice — they would play after it. The bed and the voice must be the SAME
samples. Something has to own the wire and emit one frame every 20 ms whether
or not the agent is speaking, with everything else mixed into it.

That also rules out the cheap version of the duck. If the bed were a separate
stream that stopped when a response began, the stop could not be faded: by the
time a delta arrives the previous ambience frames are already queued and cannot
be un-sent. The duck would be a hard cut at voice onset — precisely the
ambience-ON/OFF/ON artefact this exists to avoid — and the bed would be *gone*
for the whole turn rather than reduced. Ducking requires mixing; mixing
requires ownership.

── WHERE IT SITS ───────────────────────────────────────────────────────────
    OpenAI PCM16 24k ─► OutboundConditioner.process_pcm8()  (EQ, AA, /3, comp)
                                     │  float32 @ 8k, unencoded
                                     ▼
    room tone ──► [ duck envelope + mixer ] ──► mu-law 8k ──► Twilio
                                     ▲
                        voice level, measured from the signal

AFTER conditioning and BEFORE the mu-law encode, which is the only point where
both signals are linear, at the same rate, and nothing downstream is calibrated
on them. Mixing before the conditioner would feed a noise bed to a compressor
whose threshold and makeup were measured on speech; mixing after the encode
would mean decoding and re-encoding the voice for no reason.

── WHY THE LEVEL FLOOR IS NOT A FREE PARAMETER ─────────────────────────────
mu-law's smallest non-zero magnitude is 8/32768 = -72.2 dBFS, which is also
its step size near zero. (0xFF, the silence frame, decodes to exactly 0 — the
two are different facts. -72.2 is the same 0.000244 this codebase knows as the
audio_rms of a turn with nothing under it.) At the -45 dBFS resting level the
bed has ~27 dB above that quantum, which is ample for noise. A -58 dBFS duck target is about two and a half of those
steps.

A first draft of this note claimed the ducked bed is "not representable at all"
under speech. That is wrong and the measurement says so: mu-law's step is only
coarse near the PEAKS, speech has a 13-16 dB crest factor, and the step near
zero is fine — mixed against an identical render with a silent bed, 22% of
bytes still differ under -16 dBFS speech, and 45% under -25 dBFS. The bed is
not erased by the duck; it is made inaudible by it, which is the entire job.

What the numbers buy is still mostly the SHAPE OF THE TRANSITION, which is why
attack and release are the settings worth tuning and the floor is not — but
"the floor does nothing" would be the opposite overstatement. Below about
-70 dB there is genuinely nothing left to send.

── WHAT THE BED ACTUALLY IS ────────────────────────────────────────────────
data/ambience/room_tone.wav: 30s of Pixabay "Room Noise" 58390, high-passed,
decimated to 8 kHz, cut to the calmest 33s window in the source and crossfaded
into a loop. Chosen out of eight candidates by measurement — 1.5 dB of tonality
and 1.1 dB of level range across the whole two-minute source, where the clip
the name would have picked has a 25.6 dB whine at 2.5 kHz. Pixabay Content
License. scripts/make_room_tone.py carries the URL, the checksum and the table.

── UNTESTED ON A LIVE CALL ─────────────────────────────────────────────────
Every number in here is a starting point argued from the codec and from the
existing conditioning measurements. None of it has been through a Twilio
recording yet. scripts/render_ambience_demo.py renders the four cases by ear
and is the thing to run before believing any of it.
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_utils import _mulaw_encode

log = logging.getLogger(__name__)

_SR = 8_000
# Twilio's media frames are 20 ms. 160 samples of 8 kHz mu-law, one byte each.
FRAME_SAMPLES = 160
FRAME_S = FRAME_SAMPLES / _SR

# The same ceiling OutboundConditioner clips to. Mixing happens after its
# clip, so the sum can exceed it again by the bed's amplitude; re-clipping here
# keeps mu-law out of its own ceiling, which is not a graceful place to be.
_CEILING = 0.95

# ── The voice detector ───────────────────────────────────────────────────────
# "An OpenAI chunk exists" is the wrong signal and the spec is right to refuse
# it: a response's deltas contain the pauses between its own words, and a chunk
# boundary carries no information about whether anybody is talking. This reads
# the CONDITIONED SIGNAL — the actual samples about to go on the wire.
#
# The gate can afford to be low. When no response is playing the mixer is fed
# exact zeros, so the envelope is exactly 0 and the gate cannot trip; its only
# job is to stay tripped through the quiet parts INSIDE a response, where the
# floor is the model's own output noise and not silence. -50 dBFS sits ~34 dB
# under speech and ~22 dB over the mu-law quantum.
_VOICE_GATE_DB = -50.0

# Peak follower decay. Fast enough to track syllables, slow enough that the
# gate does not chatter inside a single vowel.
_ENV_DECAY_S = 0.020

# The detector runs on 2 ms sub-blocks rather than per sample. The GAIN is
# still per-sample smooth — see _ramp — so this costs nothing audible: 2 ms of
# resolution on a 75 ms attack is a fortieth of the ramp. It buys a serial loop
# of 500 iterations a second instead of 8000, which matters because this runs
# inside a task with a 20 ms deadline.
_DETECT_BLOCK = 16


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class RoomTone:
    """A looping bed, normalised so its gain setting means what it says.

    Normalised to unit RMS at load, so `gain = 10**(db/20)` produces exactly
    that RMS on the wire. Without this the dB settings would be relative to
    whatever level the asset happened to be rendered at, and re-rendering the
    asset would silently change every configured level.

    THE LOOP SEAM IS SOLVED AT BUILD TIME, not here. scripts/make_room_tone.py
    folds the tail of the clip back into its head with an equal-power
    crossfade, so wrapping is a plain modulo read: no per-frame crossfade, no
    second read pointer, and nothing to get wrong on the frame where the wrap
    happens. It refuses to write an asset whose wrap step falls outside the
    file's own distribution of adjacent steps, and the test here re-checks the
    same thing on the shipped file rather than trusting the builder.

    (`--synth` produces a bed that needs no crossfade at all, being one period
    of a periodic function by construction. The shipped asset is a real
    recording and has no such property, which is why the crossfade exists.)
    """

    def __init__(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float32).ravel()
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        # A silent or absent asset is a bed of nothing, which is today's
        # behaviour. It must not become a divide by zero.
        self._buf = (arr / rms).astype(np.float32) if rms > 1e-9 else arr
        self._pos = 0

    @property
    def size(self) -> int:
        return int(self._buf.size)

    @property
    def pos(self) -> int:
        """Where the loop is. Continuous for the whole call, never reset."""
        return self._pos

    def next(self, n: int) -> np.ndarray:
        """`n` samples, wrapping as many times as it takes."""
        if self._buf.size == 0:
            return np.zeros(n, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            take = min(n - filled, self._buf.size - self._pos)
            out[filled:filled + take] = self._buf[self._pos:self._pos + take]
            filled += take
            self._pos = (self._pos + take) % self._buf.size
        return out

    @classmethod
    def load(cls, path: str) -> "RoomTone":
        """Mono 8 kHz PCM16 WAV -> RoomTone. A missing file is an empty bed.

        NOT .ulaw, which is this repo's convention for wire assets like the
        backchannel clips, and the difference is the point: those are pushed to
        Twilio verbatim, this one is a MIXER INPUT and is never sent as-is.
        Storing it mu-law would quantise it once on the way in and again after
        mixing, and at -45 dBFS the first of those is doing real damage to a
        signal whose whole character is its noise floor.
        """
        p = Path(path)
        if not p.is_file():
            log.warning("[Ambience] room tone not found at %s — bed is silent", p)
            return cls(np.zeros(0, dtype=np.float32))
        try:
            with wave.open(str(p), "rb") as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    log.warning("[Ambience] %s is not mono PCM16 — bed is silent", p)
                    return cls(np.zeros(0, dtype=np.float32))
                if w.getframerate() != _SR:
                    log.warning("[Ambience] %s is %d Hz, not %d — bed is silent",
                                p, w.getframerate(), _SR)
                    return cls(np.zeros(0, dtype=np.float32))
                raw = w.readframes(w.getnframes())
        except Exception as exc:                  # pragma: no cover - io-dependent
            log.warning("[Ambience] could not read %s: %s — bed is silent", p, exc)
            return cls(np.zeros(0, dtype=np.float32))
        return cls(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


class Ducker:
    """Smooth gain on the bed, driven by the voice signal and by a hold.

    THE HOLD IS WHAT STOPS THE PUMPING, and it is a separate mechanism from the
    release rather than a longer release. A release long enough to bridge the
    gap between two words would also take a second to come back after a turn
    actually ends, which is the opposite of the requirement. So: the state
    machine stays ducked for `hold_ms` after the voice drops below the gate,
    and only then does the release ramp start. A breath mid-sentence never
    reaches the ramp at all; the end of a turn reaches it once.

    The ramp is ONE-POLE IN dB, not in amplitude. The spec is written in dB and
    the range here is only ~13 dB, but a linear-amplitude fade over that range
    spends most of its time in the top few dB and reads as front-loaded. dB is
    also what makes `attack_ms` mean something checkable: it is the time to
    cover 90% of the gap, so the test can assert both that the gain moved and
    that it did not jump.
    """

    def __init__(self, *, ambient_db: float, duck_db: float,
                 attack_ms: float, release_ms: float, hold_ms: float,
                 sr: int = _SR) -> None:
        self.ambient_db = float(ambient_db)
        self.duck_db = float(duck_db)
        self._sr = sr
        # 90% of the gap in the configured time: 1 - e^-2.303 = 0.9.
        self._atk = self._coef(attack_ms, 2.302585)
        self._rel = self._coef(release_ms, 2.302585)
        self._hold_blocks = max(0, int(hold_ms / 1000.0 * sr / _DETECT_BLOCK))
        self._gate = _db_to_lin(_VOICE_GATE_DB)
        self._env_decay = float(np.exp(-_DETECT_BLOCK / (_ENV_DECAY_S * sr)))
        # State, carried across every frame for the life of the call.
        self._env = 0.0
        self._held = 0
        self._db = self.ambient_db      # the call opens on a quiet room
        self.ducked_blocks = 0          # diagnostics only

    def _coef(self, ms: float, decades: float) -> float:
        """Per-SAMPLE one-pole coefficient, and per-sample is the whole point.

        This was written per SUB-BLOCK first, to match the loop it is chosen
        in, and _ramp then raised it to the power of the SAMPLE index. The two
        disagreed by the block size: a configured 75 ms attack covered 90% of
        its range in 5 ms, and a 300 ms release in about 19 ms. Both are hard
        switches with a slope on them, which is the exact artefact this module
        exists to avoid — and the bed still ended up at the right level, so
        every endpoint assertion passed while the transitions were wrong.

        The test that caught it is the one that measures TIME rather than
        endpoints. The release had no such check and was equally broken; it has
        one now.
        """
        n = max(1.0, float(ms) / 1000.0 * self._sr)
        return float(np.exp(-decades / n))

    @property
    def gain_db(self) -> float:
        return self._db

    def gains(self, voice: np.ndarray) -> np.ndarray:
        """Per-sample linear gain for the bed under this block of voice.

        `voice` is the conditioned outbound signal for one frame — the samples
        that are actually about to be sent, which is what "measured voice
        level" has to mean if it is to mean anything.
        """
        n = voice.size
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        out_db = np.empty(n, dtype=np.float32)
        at = 0
        while at < n:
            blk = voice[at:at + _DETECT_BLOCK]
            peak = float(np.max(np.abs(blk))) if blk.size else 0.0
            # Instant attack, exponential decay: a peak follower, because the
            # question is "is there speech here", not "how loud is it".
            self._env = peak if peak > self._env else self._env * self._env_decay
            if self._env >= self._gate:
                self._held = self._hold_blocks
            elif self._held > 0:
                self._held -= 1
            ducking = self._env >= self._gate or self._held > 0
            if ducking:
                self.ducked_blocks += 1
            target = self.duck_db if ducking else self.ambient_db
            coef = self._atk if target < self._db else self._rel
            out_db[at:at + blk.size] = self._ramp(target, coef, blk.size)
            at += _DETECT_BLOCK
        return (10.0 ** (out_db / 20.0)).astype(np.float32)

    def _ramp(self, target: float, coef: float, n: int) -> np.ndarray:
        """Closed form of the one-pole over a constant-target run.

        g[k] = target + (g0 - target) * coef**k, vectorised. The recursion is
        serial but its solution is not, so the per-sample smoothness costs one
        numpy power per sub-block instead of a Python loop over every sample.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        k = np.arange(1, n + 1, dtype=np.float32)
        vals = target + (self._db - target) * (coef ** k)
        self._db = float(vals[-1])
        return vals


class AmbienceMixer:
    """Owns the outbound wire. Everything that makes a sound feeds this.

    The voice queue holds conditioned 8 kHz float samples waiting to be sent.
    Deltas arrive far faster than realtime — a 6 s reply lands in about a
    second — so the queue is where that burst now waits instead of in Twilio's,
    which is what makes a barge-in `clear` able to stop audio that has not been
    played yet. `flush_voice` is the hook the three clear sites need; without
    it a cancelled response would keep draining out of here after Twilio had
    dropped its own queue.
    """

    def __init__(self, tone: RoomTone, ducker: Ducker) -> None:
        self.tone = tone
        self.ducker = ducker
        self._voice: list[np.ndarray] = []
        self._head = 0
        self.frames_sent = 0
        self.voice_frames = 0
        # Every frame this emitted, in order and contiguous — the tape of what
        # the receptionist was actually played, bed included. save() uses it so
        # the recording cannot become a document of a different mix.
        self.sent: list[bytes] = []
        self.started_at: Optional[float] = None

    # ── inputs ──────────────────────────────────────────────────────────────

    def push_voice(self, pcm8: np.ndarray) -> None:
        """Queue conditioned 8 kHz float samples for playout."""
        arr = np.asarray(pcm8, dtype=np.float32).ravel()
        if arr.size:
            self._voice.append(arr)

    def push_silence(self, seconds: float) -> None:
        """Queue a deliberate gap — the stacked-reply breath.

        Silence rather than nothing, so the gap keeps its place in the queue
        and the bed plays over it. A breath with room tone in it is the version
        a person would actually leave.
        """
        n = int(max(0.0, seconds) * _SR)
        if n:
            self._voice.append(np.zeros(n, dtype=np.float32))

    def flush_voice(self) -> int:
        """Drop everything queued. Returns the samples thrown away.

        The bed is untouched: a barge-in silences the agent, it does not hang
        up the room.
        """
        left = self.queued_samples
        self._voice.clear()
        self._head = 0
        return left

    @property
    def queued_samples(self) -> int:
        return sum(a.size for a in self._voice) - self._head

    # ── output ──────────────────────────────────────────────────────────────

    def _take_voice(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        while filled < n and self._voice:
            head = self._voice[0]
            take = min(n - filled, head.size - self._head)
            out[filled:filled + take] = head[self._head:self._head + take]
            filled += take
            self._head += take
            if self._head >= head.size:
                self._voice.pop(0)
                self._head = 0
        return out

    def next_frame(self) -> bytes:
        """One 20 ms mu-law frame: voice if there is any, bed either way."""
        voice = self._take_voice(FRAME_SAMPLES)
        if np.any(voice):
            self.voice_frames += 1
        gain = self.ducker.gains(voice)
        bed = self.tone.next(FRAME_SAMPLES) * gain
        frame = _mulaw_encode(np.clip(voice + bed, -_CEILING, _CEILING))
        self.frames_sent += 1
        self.sent.append(frame)
        return frame


def build(settings_obj) -> Optional[AmbienceMixer]:
    """A mixer, or None when ambience is off — and None is the whole safety net.

    Off means the pump is never started and every send site takes the branch it
    took before this module existed, so "disabled is byte-for-byte unchanged"
    is true by construction rather than by a gain of zero somewhere. A zero
    gain would still route the audio through a decode, a mix and a re-encode.
    """
    if not getattr(settings_obj, "realtime_ambience", False):
        return None
    path = getattr(settings_obj, "realtime_ambience_path", "") or str(
        Path(__file__).resolve().parent.parent.parent
        / "data" / "ambience" / "room_tone.wav")
    tone = RoomTone.load(path)
    if tone.size == 0:
        log.warning("[Ambience] enabled but no usable room tone — staying off")
        return None
    duck = Ducker(
        ambient_db=getattr(settings_obj, "realtime_ambience_db", -45.0),
        duck_db=getattr(settings_obj, "realtime_ambience_duck_db", -58.0),
        attack_ms=getattr(settings_obj, "realtime_ambience_attack_ms", 75),
        release_ms=getattr(settings_obj, "realtime_ambience_release_ms", 300),
        hold_ms=getattr(settings_obj, "realtime_ambience_hold_ms", 300),
    )
    log.info("[Ambience] on — %.1fs of tone, %.0f dB resting, %.0f dB ducked",
             tone.size / _SR, duck.ambient_db, duck.duck_db)
    return AmbienceMixer(tone, duck)


__all__ = ["AmbienceMixer", "Ducker", "RoomTone", "FRAME_SAMPLES", "FRAME_S",
           "build"]
