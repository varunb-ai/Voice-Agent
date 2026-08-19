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
TOKENS = {
    "mmhm":   "Mm-hm.",
    "right":  "Right.",
    "sure":   "Sure.",
    "okay":   "Okay.",
    "yeah":   "Yeah.",
    "gotit":  "Got it.",
}


async def render(voice: str, out_dir: Path) -> int:
    import websockets
    url = REALTIME_URL.format(model=settings.realtime_model)
    hdr = {"Authorization": f"Bearer {settings.openai_api_key}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

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

        for name, text in TOKENS.items():
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
            audio = b"".join(chunks)
            # 8000 bytes = 1 second of mu-law. Anything longer is a sentence,
            # not a backchannel, and would talk over the caller.
            if not (800 <= len(audio) <= 9600):
                print(f"  {voice}/{name}: {len(audio)/8000:.2f}s — skipped "
                      f"(outside the 0.1-1.2s backchannel range)")
                continue
            (out_dir / f"{name}.ulaw").write_bytes(audio)
            print(f"  {voice}/{name}.ulaw  {len(audio)/8000:.2f}s")
            written += 1
    return written


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", default=None,
                    help="repeatable; defaults to REALTIME_VOICE")
    args = ap.parse_args()
    voices = args.voice or [settings.realtime_voice]
    root = Path(__file__).resolve().parent.parent / "data" / "backchannels"
    total = 0
    for v in voices:
        print(f"\nrendering backchannels for voice={v}")
        total += await render(v, root / v)
    print(f"\n{total} clips written under {root}")
    print("Backchannels are now live on calls using those voices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
