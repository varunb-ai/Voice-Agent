"""Backchannels — the "mm-hm" a listener makes while the other person talks.

A human listener is not silent. While you speak they emit short tokens — mm-hm,
right, sure — and their absence is one of the things that makes a voice agent
feel like a machine holding a walkie-talkie. The callee gets no evidence anyone
is there until the agent's whole turn arrives, which on this rig is 1.9-3.1s
after they stop.

THE AUDIO DOES NOT COME FROM THE MODEL. It is pre-rendered and injected
straight into the Twilio media stream. That matters:

  * A `response.create` mid-utterance would collide with turn detection, be
    cancelled by the caller's own speech, and cost a response — the thing
    barge-in exists to prevent.
  * OpenAI's server VAD listens to the CALLER stream, so audio we push toward
    Twilio is invisible to it. The conversation state is untouched.
  * It costs nothing per call and cannot make the model say anything.

A backchannel is deliberately NOT a turn: it is never added to the transcript,
never counted by conversation_metrics, and never seen by the grounding guards.
It is a noise, not a move.

Clips are raw 8kHz mu-law, matching what Twilio wants on the wire, stored per
voice so the "mm-hm" is in the same voice as the rest of the call — a different
voice grunting would be worse than silence. Generate them with
scripts/make_backchannels.py.

If no clips exist for the configured voice this module returns nothing and the
feature is simply off. That is the intended failure mode: silence is the
current behaviour, so an absent clip set costs nothing.
"""
from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
_DIR = _ROOT / "data" / "backchannels"

# Loaded once per process, keyed by voice.
_CACHE: dict[str, list[bytes]] = {}


def _clips(voice: str) -> list[bytes]:
    if voice in _CACHE:
        return _CACHE[voice]
    folder = _DIR / voice
    clips: list[bytes] = []
    if folder.is_dir():
        for p in sorted(folder.glob("*.ulaw")):
            try:
                data = p.read_bytes()
            except Exception:
                continue
            # 8kHz mu-law is 8000 bytes/second. Anything beyond ~1.2s is not a
            # backchannel, it is a turn, and talking over someone for a second
            # and a half is worse than staying quiet.
            if 800 <= len(data) <= 9600:
                clips.append(data)
    _CACHE[voice] = clips
    return clips


def available(voice: str) -> int:
    """How many usable clips exist for this voice."""
    return len(_clips(voice))


def pick(voice: str, exclude: Optional[str] = None) -> Optional[str]:
    """A base64 mu-law payload ready for a Twilio `media` event, or None.

    `exclude` is the base64 payload returned last time, NOT raw bytes — the
    caller holds what this returned. The first version annotated it as bytes
    and compared it directly against the raw clips, so the exclusion never
    matched and the same clip could repeat. Caught by a flaky test; the
    annotation is now what the call site actually passes.

    Avoids repeating the clip used last. Saying the identical "mm-hm" twice in
    one call is the same tell as the identical hold acknowledgement the prompt
    already warns about — people vary, and the repeat is what gives it away.
    """
    clips = _clips(voice)
    if not clips:
        return None
    # `exclude` is the base64 payload handed back from last time; the clips are
    # raw bytes. Comparing the two directly never matches, so the exclusion did
    # nothing and the same clip could come up twice in a row — caught by a
    # flaky test, which is the only way a 1-in-N bug announces itself.
    prev: Optional[bytes] = None
    if exclude:
        try:
            prev = base64.b64decode(exclude)
        except Exception:
            prev = None
    choices = [c for c in clips if c != prev] or clips
    return base64.b64encode(random.choice(choices)).decode()
