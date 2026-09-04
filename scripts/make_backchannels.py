"""Render backchannel clips in the SAME voice the calls use.

    python scripts/make_backchannels.py            # current REALTIME_VOICE
    python scripts/make_backchannels.py --voice marin --voice cedar

Writes 8kHz mu-law to data/backchannels/<voice>/*.ulaw, which is exactly what
Twilio wants on the wire — agents/voice/backchannel.py reads them and pushes
them straight into the media stream during a call.

Why pre-render at all, rather than asking the model mid-call: a
`response.create` while the caller is talking collides with turn detection, is
cancelled by their own speech, and costs a response. See the module docstring
in backchannel.py.

Why the same voice: a different voice grunting "mm-hm" between the agent's
sentences is worse than saying nothing. That is why these are per-voice folders
and why the loader is keyed on REALTIME_VOICE.

Costs a few cents once per voice. Re-run only if you change voices or want
different tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings                      # noqa: E402
from agents.voice.realtime_worker import (            # noqa: E402
    REALTIME_URL, build_audio_config,
)

# Short, non-committal, and none of them a reply. A backchannel must not be
# answerable — "right" invites nothing, whereas "I see, go on" is a turn and
# would leave the caller waiting for the agent to continue.
# NON-LEXICAL ONLY, and that is a correction paid for on a live call.
# call-20260903-1259 shipped with "okay", "right", "sure", "yeah" and "got it"
# in the pool. A word carries meaning: heard while the caller is mid-sentence,
# "Got it" reads as the agent agreeing to something, or as a signal to stop
# talking. "Mm-hm" cannot be mistaken for either — it says only "still here".
#
# NARROWED FURTHER ON THE CLIENT'S READ. "Uh-huh" and "Mhm" survived the first
# purge, but they are still SEMANTIC acknowledgement tokens — in a healthcare
# call an "uh-huh" can be heard as agreement to what was just said ("we can
# start the record, then"). The pool is now the two neutral nasal tokens only:
# mm-hm and mm. Each rendered twice, because `pick`'s no-repeat exclusion needs
# something to choose from — the identical grunt twice in one call is exactly
# the tell the module docstring warns about.
TOKENS = {
    "mmhm":  "Mm-hm.",
    "mmhm2": "Mm-hm.",
    "mm":    "Mm.",
    "mm2":   "Mm.",
}


# ── Trimming ─────────────────────────────────────────────────────────────────
# THE MODEL PADS ITS ANSWER WITH SILENCE AND THE PADDING IS NOT FREE.
# Measured 2026-09-04 on the shipped clips: marin carried 36-67% trailing
# silence (mm.ulaw 1.20s wall for 0.40s of sound), cedar 0-6% for the same
# tokens. Nothing distinguishes them but the take — the renderer wrote whatever
# came back — so cedar being clean was luck, not a property of the voice.
#
# IT COSTS TWICE OVER.
#   * turns.py sizes the echo-mute window from the clip's WALL length:
#     `mute_until = now + len(payload)/8000 + _BACKCHANNEL_ECHO_MARGIN_S`.
#     A 1.20s clip therefore withholds quiet inbound frames for 1.60s while the
#     sound lasts 0.40s — and withholding inbound frames during a backchannel
#     is the exact mechanism that discarded 402 frames (~8s) of a real caller
#     on call-20260903-1259. Trimming right-sizes that window with no logic
#     change, because the window is derived from the length.
#   * The 9600-byte ceiling above is meant to stop a clip becoming a turn. On
#     padded audio it was measuring silence and letting a token through at the
#     ceiling that was mostly nothing.
#
# BYTE SLICING, NO RE-ENCODE. 8kHz mu-law is one byte per sample, so a sample
# index IS a byte index: decode only to FIND the boundaries, then slice the
# original bytes. Re-encoding would requantise every sample to fix the ends.
def trim(audio: bytes, *, tail_s: float = 0.04, floor_db: float = 38.0) -> bytes:
    """Strip leading and trailing silence, keeping a short tail.

    `tail_s` is deliberately not zero: cutting on the last frame above the
    floor clips the decay of a nasal token and makes "mm" end abruptly, which
    is audible in a way the silence never was.

    Returns the input unchanged if it is empty or silent throughout — a clip
    that measures as pure silence is a render failure for the length gate to
    reject, not something to slice to nothing here.
    """
    import numpy as np
    from core.audio_utils import _mulaw_decode

    win = 80                                    # 10ms at 8kHz
    n = len(audio) // win * win
    if n == 0:
        return audio
    x = _mulaw_decode(audio)[:n].reshape(-1, win).astype(np.float64)
    db = 20.0 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-9)
    voiced = np.flatnonzero(db > db.max() - floor_db)
    if voiced.size == 0:
        return audio
    start = int(voiced[0]) * win
    end = min(len(audio), (int(voiced[-1]) + 1) * win + int(tail_s * 8000))
    return audio[start:end]


async def _render_one(voice: str, name: str, text: str, out_dir: Path) -> bool:
    """One token, ONE FRESH SESSION. The realtime model will not repeat a
    token inside one session — ask for "Mm-hm." twice and the second response
    comes back silent (0.00s, skipped), which is how the first narrowed pool
    render lost clips: marin ended up with a single mm-hm and `pick`'s
    no-repeat exclusion had nothing to choose. A new session per token gives
    every token its own honest take."""
    import websockets
    url = REALTIME_URL.format(model=settings.realtime_model)
    hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}
    async with websockets.connect(url, additional_headers=hdr) as ws:
        await asyncio.wait_for(ws.recv(), timeout=20)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            # No tools, no call script: say the token and nothing else.
            "instructions": ("You are a speech renderer. Say exactly the words "
                             "given, once, in a warm neutral tone. Add nothing."),
            "audio": build_audio_config(
                transcribe_model=settings.realtime_transcribe_model,
                transcribe_hint="", audio_format="pcmu",
                noise_reduction="off", turn_detection="server_vad",
                eagerness="medium", voice=voice),
            "max_output_tokens": 32,
        }}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if m.get("type") == "session.updated":
                break

        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": f"Say exactly: {text}"}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        chunks: list[bytes] = []
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            t = m.get("type")
            if t == "response.output_audio.delta" and m.get("delta"):
                chunks.append(base64.b64decode(m["delta"]))
            elif t == "response.done":
                break
    raw = b"".join(chunks)
    audio = trim(raw)
    # 8000 bytes = 1 second of mu-law. Anything longer is a sentence,
    # not a backchannel, and would talk over the caller.
    #
    # GATED ON THE TRIMMED TOKEN, and that is the point of trimming here rather
    # than at load time: this bound is about how long we talk over the caller,
    # and untrimmed it was measuring the model's trailing silence as if it were
    # speech. marin/mm came back 1.20s and passed at the ceiling while carrying
    # 0.40s of "mm" and 0.80s of nothing.
    if not (800 <= len(audio) <= 9600):
        print(f"  {voice}/{name}: {len(audio)/8000:.2f}s — skipped "
              f"(outside the 0.1-1.2s backchannel range)")
        return False
    (out_dir / f"{name}.ulaw").write_bytes(audio)
    _note = (f"  (trimmed {len(raw)/8000:.2f}s -> {len(audio)/8000:.2f}s)"
             if len(audio) != len(raw) else "")
    print(f"  {voice}/{name}.ulaw  {len(audio)/8000:.2f}s{_note}")
    return True


async def render(voice: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, text in TOKENS.items():
        if await _render_one(voice, name, text, out_dir):
            written += 1
    return written


def retrim(out_dir: Path) -> int:
    """Trim clips already on disk, in place. No API call, no new take.

    Separate from a re-render because the two are not interchangeable: a
    re-render costs money AND returns a different take, so a voice that has
    been listened to and approved would have to be approved again. Trimming
    removes silence from the take that is already there.

    Idempotent — a trimmed clip trims to itself, so this is safe to re-run.
    """
    changed = 0
    for p in sorted(out_dir.glob("*.ulaw")):
        before = p.read_bytes()
        after = trim(before)
        if len(after) == len(before):
            print(f"  {p.parent.name}/{p.name}  {len(before)/8000:.2f}s  already tight")
            continue
        if not (800 <= len(after) <= 9600):
            print(f"  {p.parent.name}/{p.name}  trims to {len(after)/8000:.2f}s "
                  f"— OUTSIDE the 0.1-1.2s range, left alone")
            continue
        p.write_bytes(after)
        changed += 1
        print(f"  {p.parent.name}/{p.name}  {len(before)/8000:.2f}s -> "
              f"{len(after)/8000:.2f}s  (-{(len(before)-len(after))/8000:.2f}s)")
    return changed


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", default=None,
                    help="repeatable; defaults to REALTIME_VOICE")
    ap.add_argument("--retrim", action="store_true",
                    help="trim silence off the clips already on disk and exit. "
                         "No API call and no new take: use this after changing "
                         "the trim, or on clips rendered before it existed.")
    args = ap.parse_args()
    voices = args.voice or [settings.realtime_voice]
    root = Path(__file__).resolve().parent.parent / "data" / "backchannels"

    if args.retrim:
        total = 0
        for v in voices:
            d = root / v
            if not d.is_dir():
                print(f"\nno clips for voice={v}")
                continue
            print(f"\nretrimming voice={v}")
            total += retrim(d)
        print(f"\n{total} clip(s) trimmed under {root}")
        return 0

    total = 0
    for v in voices:
        print(f"\nrendering backchannels for voice={v}")
        total += await render(v, root / v)
    print(f"\n{total} clips written under {root}")
    print("Backchannels are now live on calls using those voices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
