"""FastAPI server — handles Telnyx Call Control events and media streaming.

Call flow:
  1. run_telnyx.py places outbound call via Telnyx API
  2. Hospital answers → Telnyx sends POST /webhook (call.answered)
  3. We tell Telnyx to stream audio to WS /stream/<call_id>
  4. WebSocket: receive μ-law audio → Whisper STT → VoiceBrain → pyttsx3 TTS
               → send μ-law audio back → hospital hears the agent
  5. Brain done → hangup → Agent 5 saves recording
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.models import Doctor, TranscriptTurn
from agents.experiment.brain import VoiceBrain
from agents.experiment.memory import CallMemory
from agents.experiment.audio_utils import telnyx_to_float32, float32_to_telnyx, wav_to_float32, resample
from agents.recording.agent import record_call

log = logging.getLogger(__name__)
app = FastAPI()

# ── Telnyx API calls ──────────────────────────────────────────────────────────

_TELNYX = "https://api.telnyx.com/v2"

def _hdrs() -> dict:
    return {"Authorization": f"Bearer {settings.telnyx_api_key}",
            "Content-Type": "application/json"}


async def _start_stream(call_control_id: str, ws_url: str) -> None:
    async with httpx.AsyncClient() as c:
        await c.post(
            f"{_TELNYX}/calls/{call_control_id}/actions/streaming_start",
            headers=_hdrs(),
            json={"stream_url": ws_url, "stream_track": "both_tracks"},
            timeout=10,
        )


async def _hangup(call_control_id: str) -> None:
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"{_TELNYX}/calls/{call_control_id}/actions/hangup",
                         headers=_hdrs(), timeout=10)
    except Exception:
        pass


# ── Per-call session ──────────────────────────────────────────────────────────

# silence detection thresholds
_SILENCE_RMS       = 0.008   # audio quieter than this = silence
_SILENCE_FRAMES    = 35      # ~700 ms of silence = end of utterance
_MIN_SPEECH_FRAMES = 12      # require 240ms of actual speech before processing


class _Session:
    def __init__(self, ccid: str, doctor: Doctor):
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id   = f"call-{ts}-{ccid[-4:]}"
        self.ccid      = ccid
        self.doctor    = doctor
        self.start_dt  = datetime.now()
        self.memory    = CallMemory(call_id=self.call_id)
        self.memory.clear()
        self.memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
        self.brain     = VoiceBrain(doctor, self.memory, use_llm=True)
        self.turns:    list[TranscriptTurn] = []
        self.ws:       Optional[WebSocket] = None
        # audio accumulation
        self.speech_buf:    list[np.ndarray] = []  # speech frames waiting for transcription
        self.all_audio:     list[np.ndarray] = []  # full call audio for recording
        self.silence_count: int = 0
        self.speech_count:  int = 0
        self.processing:    bool = False           # True while Whisper/brain/TTS running

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(TranscriptTurn(role=role, text=text, timestamp=self._ts()))

    async def send_speech(self, text: str) -> None:
        """TTS → μ-law → WebSocket back to Telnyx (hospital hears this)."""
        if not self.ws:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            from agents.experiment.tts_local import synthesize
            await asyncio.to_thread(synthesize, text, tmp.name)
            arr, sr = wav_to_float32(tmp.name)
            arr_16k = resample(arr, sr, 16_000)
            self.all_audio.append(arr_16k)          # save for recording
            payload = float32_to_telnyx(arr_16k)
            await self.ws.send_text(
                json.dumps({"event": "media", "media": {"payload": payload}})
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    async def save(self) -> None:
        """Agent 5 — build and persist CallRecord."""
        import soundfile as sf
        duration = int((datetime.now() - self.start_dt).total_seconds())
        audio_path: Optional[str] = None
        if self.all_audio:
            merged = np.concatenate(self.all_audio)
            p = Path("data/recordings") / f"{self.call_id}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), merged, 16_000)
            audio_path = str(p)

        snap = self.memory.snapshot()
        snap["transcript"] = [t.model_dump() for t in self.turns]
        record, backend = record_call(
            snap, call_id=self.call_id,
            audio_path=audio_path, duration_seconds=duration,
            use_llm=False, persist=True,
        )
        log.info(
            "Saved %s | branch=%s resolved=%s backend=%s",
            self.call_id, record.branch, record.resolved, backend,
        )
        _print_summary(record, backend, self.turns)


# ── In-memory registry ────────────────────────────────────────────────────────

_sessions: dict[str, _Session] = {}
pending_doctor: Optional[Doctor] = None   # set by run_telnyx.py before dial


# ── Webhook ───────────────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    body  = await request.json()
    etype = body.get("data", {}).get("event_type", "")
    pl    = body.get("data", {}).get("payload", {})
    ccid  = pl.get("call_control_id", "")

    if etype == "call.answered":
        doc = pending_doctor
        if not doc:
            await _hangup(ccid)
            return JSONResponse({"ok": True})
        sess = _Session(ccid, doc)
        _sessions[ccid] = sess
        log.info("Call answered — starting stream for %s", ccid)
        # Derive WebSocket URL from public server URL
        ws_base = (settings.server_public_url
                   .replace("https://", "wss://")
                   .replace("http://",  "ws://"))
        await _start_stream(ccid, f"{ws_base}/stream/{ccid}")

    elif etype == "call.hangup":
        sess = _sessions.pop(ccid, None)
        if sess:
            await sess.save()

    return JSONResponse({"ok": True})


# ── WebSocket media stream ────────────────────────────────────────────────────

@app.websocket("/stream/{ccid}")
async def media_stream(ws: WebSocket, ccid: str):
    await ws.accept()
    sess = _sessions.get(ccid)
    if not sess:
        await ws.close()
        return
    sess.ws = ws

    # Send greeting first
    greeting = sess.brain.greet()
    sess.add_turn("agent", greeting)
    log.info("Agent: %s", greeting)
    await sess.send_speech(greeting)

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            if msg.get("event") != "media":
                continue
            if msg["media"].get("track") != "inbound":
                continue          # skip our own outbound audio echoed back
            if sess.processing:
                continue          # drop frames while brain/TTS are busy

            arr = telnyx_to_float32(msg["media"]["payload"])   # float32 at 16 kHz
            sess.all_audio.append(arr)

            rms = float(np.sqrt(np.mean(arr ** 2)))
            if rms < _SILENCE_RMS:
                sess.silence_count += 1
                sess.speech_count   = 0
            else:
                sess.silence_count  = 0
                sess.speech_count  += 1
                sess.speech_buf.append(arr)

            # End of utterance: speech detected then fell silent
            uttered = (
                sess.silence_count >= _SILENCE_FRAMES
                and len(sess.speech_buf) >= _MIN_SPEECH_FRAMES
            )
            if not uttered:
                continue

            speech = np.concatenate(sess.speech_buf)
            sess.speech_buf.clear()
            sess.silence_count = 0
            sess.processing = True

            async def handle_turn(audio: np.ndarray) -> None:
                try:
                    caller_text = await asyncio.to_thread(_transcribe, audio)
                    if not caller_text:
                        return
                    log.info("Caller: %s", caller_text)
                    sess.add_turn("caller", caller_text)
                    reply = sess.brain.handle(caller_text)
                    sess.add_turn("agent", reply)
                    log.info("Agent:  %s", reply)
                    await sess.send_speech(reply)
                    if sess.brain.done:
                        await asyncio.sleep(1.5)    # let last audio finish
                        await _hangup(sess.ccid)
                finally:
                    sess.processing = False

            asyncio.create_task(handle_turn(speech))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Stream error: %s", e, exc_info=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _transcribe(audio_16k: np.ndarray) -> str:
    from agents.experiment.stt_whisper import transcribe_array
    return transcribe_array(audio_16k)


def _print_summary(record, backend: str, turns: list[TranscriptTurn]) -> None:
    from rich.console import Console
    from rich.table import Table
    c = Console()
    c.rule("[bold]Call Complete")
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("[dim]Call ID[/dim]",     record.call_id)
    t.add_row("[dim]Doctor[/dim]",      record.doctor_name)
    t.add_row("[dim]Hospital[/dim]",    record.hospital_name or "—")
    t.add_row("[dim]Branch[/dim]",
              f"[green]{record.branch}[/green]" if record.branch else "[dim]not obtained[/dim]")
    t.add_row("[dim]Resolved[/dim]",
              "[green]Yes[/green]" if record.resolved else "[red]No[/red]")
    t.add_row("[dim]Duration[/dim]",    f"{record.duration_seconds}s")
    t.add_row("[dim]Audio file[/dim]",  record.audio_path or "—")
    t.add_row("[dim]Saved to[/dim]",    backend)
    c.print(t)
    c.print("\n[bold]Summary:[/bold]", record.summary)
    c.print("\n[bold]Transcript:[/bold]")
    for turn in turns:
        color = "cyan" if turn.role == "agent" else "green"
        c.print(f"  [dim]{turn.timestamp}[/dim]  [{color}]{turn.role.title()}[/{color}]  {turn.text}")
