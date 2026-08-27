"""FastAPI server — handles Vonage Voice API WebSocket audio streaming.

Vonage flow:
  1. run_vonage.py places outbound call via Vonage REST API (JWT auth)
  2. Vonage calls answer_url -> we return NCCO with WebSocket connect
  3. Hospital answers -> Vonage opens WebSocket to /stream/<conversation_uuid>
  4. Binary frames = raw PCM16 at 16kHz -> Whisper -> VoiceBrain -> pyttsx3 -> send back
  5. Brain done -> close WebSocket -> call ends -> Agent 5 saves recording
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from core.config import settings
from core.models import Doctor, TranscriptTurn
from agents.experiment.brain import VoiceBrain
from agents.experiment.memory import CallMemory
from agents.experiment.audio_utils import wav_to_float32, resample
from agents.recording.agent import record_call

log = logging.getLogger(__name__)
app = FastAPI()

# Vonage streams 16kHz PCM16 linear by default (we request it in NCCO)
VONAGE_SR  = 16_000
WHISPER_SR = 16_000

# Dynamic end-of-speech detection (each frame ≈ 20ms)
_SILENCE_RMS          = 0.008
_MIN_SPEECH_FRAMES    = 10
_END_SILENCE_FRAMES   = 150   # 3 seconds silence after speech = caller done
_MAX_UTTERANCE_FRAMES = 500   # 10 second hard cap


# ── audio helpers ─────────────────────────────────────────────────────────────

def _bytes_to_float32(data: bytes) -> np.ndarray:
    """Raw PCM16 bytes -> float32 numpy array."""
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def _float32_to_bytes(samples: np.ndarray) -> bytes:
    """float32 -> raw PCM16 bytes for Vonage."""
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()


# ── per-call session ──────────────────────────────────────────────────────────

class _Session:
    def __init__(self, conv_uuid: str, doctor: Doctor):
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id    = f"call-{ts}-{conv_uuid[-4:]}"
        self.conv_uuid  = conv_uuid
        self.doctor     = doctor
        self.start_dt   = datetime.now()
        self.memory     = CallMemory(call_id=self.call_id)
        self.memory.clear()
        self.memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
        self.brain      = VoiceBrain(doctor, self.memory, use_llm=True)
        self.turns:     list[TranscriptTurn] = []
        self.ws:        Optional[WebSocket] = None
        self.speech_buf:           list[np.ndarray] = []
        self.all_audio:            list[np.ndarray] = []
        self.silence_ct:           int = 0
        self.speech_ct:            int = 0
        self.total_speech_in_turn: int = 0
        self.processing:           bool = False
        self._preroll:             deque = deque(maxlen=6)

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(TranscriptTurn(
            role=role, text=text,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))

    async def send_speech(self, text: str) -> None:
        """TTS -> PCM16 bytes -> WebSocket binary frame -> hospital hears agent."""
        if not self.ws:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            from agents.experiment.tts_local import synthesize
            await asyncio.to_thread(synthesize, text, tmp.name)
            arr, sr = wav_to_float32(tmp.name)
            arr_16k = resample(arr, sr, VONAGE_SR)
            self.all_audio.append(arr_16k)
            await self.ws.send_bytes(_float32_to_bytes(arr_16k))
            # Wait for audio to finish playing before accepting caller input
            duration_sec = len(arr_16k) / VONAGE_SR
            await asyncio.sleep(duration_sec + 0.3)
            self.speech_buf.clear()
            self._preroll.clear()
            self.silence_ct = 0
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    async def save(self) -> None:
        import soundfile as sf
        duration = int((datetime.now() - self.start_dt).total_seconds())
        audio_path: Optional[str] = None
        if self.all_audio:
            merged = np.concatenate(self.all_audio)
            p = Path("data/recordings") / f"{self.call_id}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), merged, VONAGE_SR)
            audio_path = str(p)
        snap = self.memory.snapshot()
        snap["transcript"] = [t.model_dump() for t in self.turns]
        record, backend = record_call(
            snap, call_id=self.call_id,
            audio_path=audio_path, duration_seconds=duration,
            use_llm=False, persist=True,
        )
        _print_summary(record, backend, self.turns)


# registry
_sessions:      dict[str, _Session] = {}
pending_doctor: Optional[Doctor]    = None


# ── Vonage answer webhook — return NCCO ───────────────────────────────────────

@app.post("/webhook/answer")
async def answer(request: Request):
    """Vonage calls this when the hospital picks up. Return NCCO to connect WebSocket."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    params = dict(request.query_params)
    conv_uuid = body.get("conversation_uuid") or params.get("conversation_uuid", "unknown")

    doc = pending_doctor
    if not doc:
        return JSONResponse([{"action": "talk", "text": "No session available."}])

    sess = _Session(conv_uuid, doc)
    _sessions[conv_uuid] = sess
    log.info("Vonage answer webhook: conv=%s", conv_uuid)

    ws_url = f"{settings.server_public_url.replace('https://', 'wss://')}/stream/{conv_uuid}"
    ncco = [
        {
            "action": "connect",
            "endpoint": [{
                "type": "websocket",
                "uri": ws_url,
                "content-type": "audio/l16;rate=16000",
                "headers": {"conv_uuid": conv_uuid},
            }],
        }
    ]
    return JSONResponse(ncco)


# ── Vonage event webhook ───────────────────────────────────────────────────────

@app.post("/webhook/events")
async def events(request: Request):
    try:
        body = await request.json()
        log.info("Vonage event: %s", body.get("status", body))
    except Exception:
        pass
    return JSONResponse({"ok": True})


# ── WebSocket (Vonage audio stream) ───────────────────────────────────────────

@app.websocket("/stream/{conv_uuid}")
async def voicebot_stream(ws: WebSocket, conv_uuid: str):
    await ws.accept()

    sess = _sessions.get(conv_uuid)
    if not sess:
        # fallback: create session if answer webhook hadn't fired yet
        doc = pending_doctor
        if not doc:
            await ws.close()
            return
        sess = _Session(conv_uuid, doc)
        _sessions[conv_uuid] = sess

    sess.ws = ws
    log.info("Vonage WebSocket connected: %s", conv_uuid)

    # Send greeting immediately
    greeting = sess.brain.greet()
    sess.add_turn("agent", greeting)
    log.info("Agent: %s", greeting)
    await sess.send_speech(greeting)

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                # Binary frame = raw PCM16 audio from caller
                if sess.processing:
                    continue

                arr = _bytes_to_float32(msg["bytes"])
                sess.all_audio.append(arr)

                rms = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0
                if rms < _SILENCE_RMS:
                    sess.silence_ct += 1
                    if sess.speech_ct == 0:
                        sess._preroll.append(arr)
                    sess.speech_ct = 0
                else:
                    if sess.speech_ct == 0:
                        sess.speech_buf.extend(sess._preroll)
                        sess._preroll.clear()
                    sess.silence_ct = 0
                    sess.speech_ct += 1
                    sess.speech_buf.append(arr)

                if (sess.silence_ct >= _END_SILENCE_FRAMES
                        and len(sess.speech_buf) >= _MIN_SPEECH_FRAMES):
                    speech = np.concatenate(sess.speech_buf)
                    sess.speech_buf.clear()
                    sess._preroll.clear()
                    sess.silence_ct = 0
                    sess.processing = True

                    async def handle(audio: np.ndarray) -> None:
                        try:
                            text = await asyncio.to_thread(_transcribe, audio)
                            if not text:
                                return
                            log.info("Caller: %s", text)
                            sess.add_turn("caller", text)
                            reply = await asyncio.to_thread(sess.brain.handle, text)
                            sess.add_turn("agent", reply)
                            log.info("Agent:  %s", reply)
                            await sess.send_speech(reply)
                            if sess.brain.done:
                                await asyncio.sleep(2)
                                await ws.close()
                        finally:
                            sess.processing = False

                    asyncio.create_task(handle(speech))

            elif "text" in msg and msg["text"]:
                # Text frame = metadata from Vonage (ignore)
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Stream error: %s", e, exc_info=True)
    finally:
        # Renamed from `sess`: see exotel_worker for why.
        _ending = _sessions.pop(conv_uuid, None)
        if _ending:
            await _ending.save()


# ── health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


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
    t.add_row("[dim]Call ID[/dim]",    record.call_id)
    t.add_row("[dim]Doctor[/dim]",     record.doctor_name)
    t.add_row("[dim]Hospital[/dim]",   record.hospital_name or "—")
    t.add_row("[dim]Branch[/dim]",
              f"[green]{record.branch}[/green]" if record.branch else "[dim]not obtained[/dim]")
    t.add_row("[dim]Resolved[/dim]",
              "[green]Yes[/green]" if record.resolved else "[red]No[/red]")
    t.add_row("[dim]Duration[/dim]",   f"{record.duration_seconds}s")
    t.add_row("[dim]Audio[/dim]",      record.audio_path or "—")
    t.add_row("[dim]Saved to[/dim]",   backend)
    c.print(t)
    c.print("\n[bold]Summary:[/bold]", record.summary)
    c.print("\n[bold]Transcript:[/bold]")
    for turn in turns:
        color = "cyan" if turn.role == "agent" else "green"
        c.print(f"  [dim]{turn.timestamp}[/dim]  [{color}]{turn.role.title()}[/{color}]  {turn.text}")
