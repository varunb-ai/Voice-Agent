"""FastAPI server — handles Exotel voice bot via WebSocket audio streaming.

Exotel flow:
  1. run_exotel.py places outbound call via Exotel REST API
  2. Hospital answers → Exotel opens WebSocket to /stream/<call_sid>
  3. We receive PCM audio → Whisper STT → VoiceBrain → pyttsx3 TTS → send back
  4. Brain done → close WebSocket → call ends → Agent 5 saves recording
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.models import Doctor, TranscriptTurn
from agents.experiment.brain import VoiceBrain
from core.memory import CallMemory
from core.audio_utils import wav_to_float32, resample
from agents.recording.agent import record_call

log = logging.getLogger(__name__)
app = FastAPI()

# Exotel streams 8kHz PCM16 (linear, not μ-law)
EXOTEL_SR  = 8_000
WHISPER_SR = 16_000

# Dynamic end-of-speech detection (each frame ≈ 20ms)
_SILENCE_RMS          = 0.008
_MIN_SPEECH_FRAMES    = 10
_END_SILENCE_FRAMES   = 150   # 3 seconds silence after speech = caller done
_MAX_UTTERANCE_FRAMES = 500   # 10 second hard cap


# ── audio helpers ─────────────────────────────────────────────────────────────

def _exotel_to_float32(payload_b64: str) -> np.ndarray:
    """Exotel base64 PCM16 → float32 at 16kHz."""
    raw = base64.b64decode(payload_b64)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return resample(arr, EXOTEL_SR, WHISPER_SR)


def _float32_to_exotel(samples_16k: np.ndarray) -> str:
    """float32 at 16kHz → base64 PCM16 at 8kHz for Exotel."""
    arr_8k  = resample(samples_16k, WHISPER_SR, EXOTEL_SR)
    pcm16   = (np.clip(arr_8k, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode()


# ── per-call session ──────────────────────────────────────────────────────────

class _Session:
    def __init__(self, call_sid: str, doctor: Doctor):
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id    = f"call-{ts}-{call_sid[-4:]}"
        self.call_sid   = call_sid
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
        """TTS → PCM16 → base64 → WebSocket → hospital hears agent."""
        if not self.ws:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            from agents.experiment.tts_local import synthesize
            await asyncio.to_thread(synthesize, text, tmp.name)
            arr, sr = wav_to_float32(tmp.name)
            arr_16k = resample(arr, sr, WHISPER_SR)
            self.all_audio.append(arr_16k)
            payload = _float32_to_exotel(arr_16k)
            await self.ws.send_text(json.dumps({
                "event": "media",
                "media": {"payload": payload},
            }))
            # Wait for audio to finish playing before accepting caller input
            duration_sec = len(arr_16k) / WHISPER_SR
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
            sf.write(str(p), merged, WHISPER_SR)
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


# ── WebSocket (Exotel voicebot stream) ───────────────────────────────────────

@app.websocket("/stream/{call_sid}")
async def voicebot_stream(ws: WebSocket, call_sid: str):
    await ws.accept()

    doc = pending_doctor
    if not doc:
        await ws.close()
        return

    sess = _Session(call_sid, doc)
    _sessions[call_sid] = sess
    sess.ws = ws
    log.info("Exotel stream connected: %s", call_sid)

    # Send greeting as soon as stream opens
    greeting = sess.brain.greet()
    sess.add_turn("agent", greeting)
    log.info("Agent: %s", greeting)
    await sess.send_speech(greeting)

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            event = msg.get("event", "")

            if event == "media":
                if sess.processing:
                    continue
                arr = _exotel_to_float32(msg["media"]["payload"])
                sess.all_audio.append(arr)

                rms = float(np.sqrt(np.mean(arr ** 2)))
                if rms >= _SILENCE_RMS:
                    if sess.total_speech_in_turn == 0:
                        sess.speech_buf.extend(sess._preroll)
                        sess._preroll.clear()
                    sess.silence_ct = 0
                    sess.speech_ct += 1
                    sess.total_speech_in_turn += 1
                    sess.speech_buf.append(arr)
                else:
                    sess.silence_ct += 1
                    sess.speech_ct = 0
                    if sess.total_speech_in_turn == 0:
                        sess._preroll.append(arr)

                caller_done = (
                    sess.total_speech_in_turn >= _MIN_SPEECH_FRAMES
                    and sess.silence_ct >= _END_SILENCE_FRAMES
                )
                too_long = sess.total_speech_in_turn >= _MAX_UTTERANCE_FRAMES

                if caller_done or too_long:
                    if not sess.speech_buf:
                        sess.total_speech_in_turn = 0
                        continue
                    speech = np.concatenate(sess.speech_buf)
                    sess.speech_buf.clear()
                    sess._preroll.clear()
                    sess.silence_ct = 0
                    sess.speech_ct = 0
                    sess.total_speech_in_turn = 0
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

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Stream error: %s", e, exc_info=True)
    finally:
        # Renamed from `sess`: reusing the name here made the checker treat
        # the live session above as Optional for the whole function.
        _ending = _sessions.pop(call_sid, None)
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
