"""FastAPI server — handles Twilio Voice webhooks and bidirectional media streaming.

Call flow:
  1. run_twilio.py places outbound call via Twilio REST API
  2. Hospital answers → Twilio POSTs to /answer → we return TwiML with <Connect><Stream>
  3. Twilio opens WebSocket to /stream/<call_sid>
  4. We receive μ-law audio → WebRTC VAD → Groq Whisper STT → VoiceBrain LLM → Piper TTS → send back
  5. Brain done → close WebSocket → Twilio hangs up → Agent 5 saves recording
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from core.config import settings
from core.models import Doctor, TranscriptTurn
from core.memory import CallMemory
from core.audio_utils import telnyx_to_float32, float32_to_telnyx, wav_to_float32, resample

# ── Classic pipeline dependencies (USE_REALTIME=false only) ──────────────────
# webrtcvad, the Piper/Whisper brain and the recording agent are needed only by
# the classic VAD→STT→LLM→TTS path. A speech-to-speech realtime deployment
# should not have to install them, so import failures are deferred until the
# classic path is actually used rather than crashing the server at startup.
try:
    import webrtcvad
    from agents.experiment.brain import VoiceBrain
    from agents.recording.agent import record_call
    _CLASSIC_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _e:            # pragma: no cover - depends on install set
    webrtcvad = None                 # type: ignore[assignment]
    VoiceBrain = None                # type: ignore[assignment]
    record_call = None               # type: ignore[assignment]
    _CLASSIC_IMPORT_ERROR = _e

log = logging.getLogger(__name__)
app = FastAPI()


# ── Twilio webhook authentication ────────────────────────────────────────────
# Every public endpoint here is reachable by anyone who knows the ngrok URL.
# Twilio signs each request with the account auth token; verifying that
# signature is the only thing that proves a POST actually came from Twilio.
# Without it, /recording_ready in particular will hand your Twilio credentials
# to any URL an attacker supplies.

async def _verify_twilio_signature(request: Request, form) -> bool:
    """Validate X-Twilio-Signature. Returns True if the request is authentic.

    Twilio signs the exact URL it was configured with. Behind a tunnel, the URL
    the app sees is not that URL, so it has to be rebuilt from
    SERVER_PUBLIC_URL. Several near-miss variants are plausible (trailing
    slash, query string, forwarded scheme), so each is tried and the failure
    path logs enough to tell which assumption was wrong.
    """
    enforcing = settings.twilio_validate_webhooks

    if not settings.twilio_auth_token:
        log.error("TWILIO_AUTH_TOKEN is unset — cannot validate")
        return not enforcing
    try:
        from twilio.request_validator import RequestValidator
    except ImportError:
        log.error("twilio package missing — cannot validate webhook signature")
        return not enforcing

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        log.warning("No X-Twilio-Signature header on %s", request.url.path)
        return not enforcing

    base = settings.server_public_url.strip().rstrip("/")
    path = request.url.path
    query = request.url.query
    candidates = [
        base + path,
        base + path + ("?" + query if query else ""),
        str(request.url),
        str(request.url).replace("http://", "https://", 1),
    ]

    params = {k: v for k, v in form.multi_items()} if hasattr(form, "multi_items") else dict(form)
    validator = RequestValidator(settings.twilio_auth_token)

    for candidate in candidates:
        if validator.validate(candidate, params, signature):
            return True

    # Diagnose regardless of whether we're enforcing, so a single call in
    # non-enforcing mode still yields the information needed to fix this.
    print("\n" + "!" * 64, flush=True)
    print("  TWILIO SIGNATURE MISMATCH", flush=True)
    print(f"  path           : {path}", flush=True)
    print(f"  sig received   : {signature}", flush=True)
    for c in candidates:
        print(f"  tried          : {c}", flush=True)
        print(f"    -> computed  : {validator.compute_signature(c, params)}", flush=True)
    print(f"  params         : {sorted(params)}", flush=True)
    print(f"  auth token     : ...{settings.twilio_auth_token[-4:]} "
          f"(len {len(settings.twilio_auth_token)})", flush=True)
    print(f"  enforcing      : {enforcing}", flush=True)
    if enforcing:
        print("  -> REJECTED. Set TWILIO_VALIDATE_WEBHOOKS=false in .env to test,", flush=True)
        print("     then re-enable before deploying — it is what stops", flush=True)
        print("     /recording_ready leaking your Twilio credentials.", flush=True)
    else:
        print("  -> ALLOWED (validation disabled). Re-enable before deploying.", flush=True)
    print("!" * 64 + "\n", flush=True)
    return not enforcing


def _forbidden() -> Response:
    return Response(status_code=403, content="invalid twilio signature")


def _is_twilio_recording_url(url: str) -> bool:
    """Only ever send Twilio credentials to Twilio's own API host."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc in _ALLOWED_RECORDING_HOSTS


_ALLOWED_RECORDING_HOSTS = {"api.twilio.com"}


def _form_str(form, key: str, default: str = "") -> str:
    """One Twilio form field, as the text it always is.

    `FormData.get` is typed `UploadFile | str | None` because a multipart
    body MAY carry a file, and every call-sid, recording URL and status in
    this module flows out of it into a dict key, a path fragment or an HTTP
    request. Twilio signs and sends urlencoded webhooks and never posts a
    file to these endpoints, so this narrows to the case that actually
    occurs.

    A non-string yields the DEFAULT rather than str(value): the repr of an
    UploadFile is not a call sid, and letting one reach _doctor_for or a
    filename would turn a type error into a wrong file on disk. Behaviour is
    unchanged for every value Twilio actually sends.
    """
    v = form.get(key, default)
    return v if isinstance(v, str) else default


def _require_classic() -> None:
    """Raise a clear error if the classic pipeline is used without its deps."""
    if _CLASSIC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "USE_REALTIME=false needs the classic pipeline dependencies "
            f"(missing: {_CLASSIC_IMPORT_ERROR}). Install them, or set "
            "USE_REALTIME=true to run speech-to-speech."
        ) from _CLASSIC_IMPORT_ERROR


# ── WebRTC VAD (dynamic end-of-speech) ───────────────────────────────────────
# Mode 2 = medium-aggressive. Works frame-by-frame using spectral features,
# not a fixed timer — responds the moment the caller truly stops talking.
_VAD = webrtcvad.Vad(2) if webrtcvad is not None else None

_VAD_FRAME_SAMPLES = 320          # 20ms at 16kHz — WebRTC VAD required size
_MIN_SPEECH_FRAMES = 16           # need ≥320ms of real speech — prevents tiny noise bursts and garbled fragments
_MAX_SPEECH_FRAMES = 600          # safety cap: 12s max utterance
_POST_TTS_COOLDOWN = 1.5          # seconds of dead-zone after agent speaks — long enough for Twilio buffer to drain so agent voice doesn't echo back as "caller speech"

# Thinking fillers played while LLM is processing (> 1.5s) — avoids dead air
_THINKING_FILLERS = [
    "Oh sure, one sec!",
    "Just a moment!",
    "Oh, let me check that!",
    "Right, one moment!",
]


def _adaptive_eos(speech_frames: int) -> int:
    """Return how many silent frames to wait before deciding the caller has stopped.
    Tighter thresholds now that LLM is Groq (~400ms) — less dead air to fill."""
    if speech_frames < 10:    # < 200ms — single word ("Yes", "Jubilee Hills")
        return 3              # 60ms silence → respond immediately
    if speech_frames < 40:    # 200ms–800ms — short sentence
        return 6              # 120ms silence
    if speech_frames < 150:   # 800ms–3s — normal answer
        return 10             # 200ms silence
    return 14                 # > 3s — long answer → 280ms to confirm done


def _vad_is_speech(chunk: np.ndarray) -> bool:
    """Run WebRTC VAD on a 320-sample (20ms@16kHz) float32 chunk."""
    if _VAD is None:
        return False
    if len(chunk) < _VAD_FRAME_SAMPLES:
        chunk = np.pad(chunk, (0, _VAD_FRAME_SAMPLES - len(chunk)))
    elif len(chunk) > _VAD_FRAME_SAMPLES:
        chunk = chunk[:_VAD_FRAME_SAMPLES]
    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    try:
        return _VAD.is_speech(pcm, 16000)
    except Exception:
        return False


# ── per-call session ──────────────────────────────────────────────────────────

class _Session:
    """Classic VAD→STT→LLM→TTS call session. Not used when USE_REALTIME=true."""

    def __init__(self, call_sid: str, doctor: Doctor):
        _require_classic()
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id     = f"call-{ts}-{call_sid[-4:]}"
        self.call_sid    = call_sid
        self.stream_sid  = ""
        self.doctor      = doctor
        self.start_dt    = datetime.now()
        self.memory      = CallMemory(call_id=self.call_id)
        self.memory.clear()
        self.memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
        # Guaranteed non-None by _require_classic() above; restated for the
        # type checker, which cannot narrow through that call.
        assert VoiceBrain is not None
        self.brain       = VoiceBrain(doctor, self.memory, use_llm=True)
        self.turns:      list[TranscriptTurn] = []
        self.ws:         Optional[WebSocket] = None

        # audio state
        self.speech_buf:   list[np.ndarray] = []   # speech frames being accumulated
        self._call_t0:     float = time.monotonic() # call start reference for time-aligned mixing
        self._first_frame_t: Optional[float] = None # monotonic time of first inbound media frame
        self.agent_segs:   list[tuple[float, np.ndarray]] = []  # (offset_sec, audio) — agent TTS
        self.caller_audio: list[np.ndarray] = []   # continuous inbound audio from call start
        self._preroll:     deque = deque(maxlen=12) # silence frames before speech starts (240ms buffer)
        self._vad_buf:     np.ndarray = np.array([], dtype=np.float32)  # chunk accumulator
        self._speech_ct:   int  = 0    # VAD-confirmed speech frames this utterance
        self._end_ct:      int  = 0    # consecutive non-speech frames after speech
        self.processing:   bool = False
        self.speaking:     bool = False
        self._barge_in:    bool = False   # caller spoke while agent was talking → cut TTS
        self._barge_frames: int = 0       # consecutive high-energy frames while agent speaking

        self._listen_after: float = 0.0  # monotonic clock — ignore audio until this time
        self._twilio_rec_sid: Optional[str] = None  # Twilio-side recording SID
        self._tts_cache: dict[str, np.ndarray] = {}
        self._filler_count:         int = 0  # max 1 thinking filler per call — repeated fillers sound robotic
        self._silence_probe_count:  int = 0  # max 2 "are you still there?" probes per call
        self._last_reply: str = ""          # deduplication — prevent saying same thing twice in a row
        threading.Thread(target=self._prebuild_tts, daemon=True).start()

    def _prebuild_tts(self) -> None:
        import re
        from agents.experiment.tts_local import synthesize
        from core.audio_utils import wav_to_float32, resample
        from agents.experiment.prompts import _FIRST_ASK, _REPEAT_ASK, _HOLD_ACKS, _CLOSINGS, _GREETINGS
        from core.config import settings
        clean = re.sub(r"^Dr\.?\s+", "", self.doctor.doctor_name, flags=re.I).strip()
        tod = "morning"  # pre-warm all time variants below
        from agents.experiment.prompts import _GREETINGS
        all_greetings = [
            g.format(time_of_day=tod, hospital=self.doctor.hospital_name or "your hospital",
                     org=settings.org_name)
            for g in _GREETINGS
            for tod in ("morning", "afternoon", "evening")
        ]
        phrases = [
            *all_greetings,
            *[v.format(doctor=clean) for v in _FIRST_ASK],
            *[v.format(doctor=clean) for v in _REPEAT_ASK],
            *_HOLD_ACKS,
            *_CLOSINGS,
            "Oh thanks so much for your time. Have a great day!",
            "Oh my apologies — I must have the wrong number. Sorry about that, have a great day!",
            "Oh no worries at all — thanks so much for your time. Have a great day!",
            "Oh I see, thanks for letting me know. Have a great day!",
            # Thinking fillers — must match _THINKING_FILLERS exactly so cache hits
            *_THINKING_FILLERS,
        ]
        for text in phrases:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                synthesize(text, tmp.name)
                arr, sr = wav_to_float32(tmp.name)
                self._tts_cache[text] = resample(arr, sr, 16_000)
            except Exception as e:
                log.warning("TTS pre-cache failed for %r: %s", text[:50], e)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        log.info("TTS cache ready — %d phrases", len(self._tts_cache))

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(TranscriptTurn(
            role=role, text=text,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))

    def _reset_vad(self) -> None:
        self.speech_buf.clear()
        self._preroll.clear()
        self._vad_buf  = np.array([], dtype=np.float32)
        self._speech_ct = 0
        self._end_ct   = 0

    async def send_speech(self, text: str) -> None:
        if not self.ws or not self.stream_sid:
            return
        try:
            self.speaking = True
            self._barge_in = False
            self._barge_frames = 0
            interrupted = False
            offset = time.monotonic() - self._call_t0

            await self.ws.send_text(json.dumps({"event": "clear", "streamSid": self.stream_sid}))

            if text in self._tts_cache:
                # ── Cache hit: instant playback ───────────────────────────────
                arr_16k = self._tts_cache[text]
                log.info("TTS cache hit: %s", text[:60])
                self.agent_segs.append((offset, arr_16k))
                await self.ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": float32_to_telnyx(arr_16k)},
                }))
                duration_sec = len(arr_16k) / 16_000
                # Poll for barge-in while audio plays
                total_wait = duration_sec + 0.20
                waited = 0.0
                while waited < total_wait:
                    await asyncio.sleep(0.02)
                    waited += 0.02
                    if self._barge_in:
                        try:
                            await self.ws.send_text(json.dumps(
                                {"event": "clear", "streamSid": self.stream_sid}
                            ))
                        except Exception:
                            pass
                        log.info("TTS interrupted — caller started speaking")
                        interrupted = True
                        break
            else:
                # ── Streaming TTS: first chunk plays in ~200ms ────────────────
                log.info("TTS streaming: %s", text[:60])
                from agents.experiment.tts_local import synthesize_stream_chunks
                import numpy as np
                cache_parts: list[np.ndarray] = []
                duration_sec = 0.0

                async for ulaw_b64, f32_16k in synthesize_stream_chunks(text):
                    if self._barge_in:
                        try:
                            await self.ws.send_text(json.dumps(
                                {"event": "clear", "streamSid": self.stream_sid}
                            ))
                        except Exception:
                            pass
                        log.info("TTS interrupted mid-stream — caller started speaking")
                        interrupted = True
                        break
                    await self.ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": ulaw_b64},
                    }))
                    cache_parts.append(f32_16k)
                    duration_sec += len(f32_16k) / 16_000

                if cache_parts:
                    arr_16k = np.concatenate(cache_parts)
                    self._tts_cache[text] = arr_16k  # cache for instant replay next time
                    self.agent_segs.append((offset, arr_16k))

                # Small drain wait — Twilio buffers a few hundred ms of audio
                if not interrupted:
                    await asyncio.sleep(0.35)

            self._barge_in = False
            self._barge_frames = 0
            self._reset_vad()
            self.speaking = False
            self._listen_after = time.monotonic() + (0.05 if interrupted else _POST_TTS_COOLDOWN)
            # Reset silence timer after Sarah finishes speaking — gives caller a fresh window to respond
            self._last_caller_speech_t = time.monotonic()
        except Exception as e:
            log.error("send_speech error: %s", e, exc_info=True)
            self.speaking = False

    def _start_twilio_recording(self) -> None:
        """Ask Twilio to record the live call — both sides, exactly as the phone call sounds."""
        if not settings.twilio_account_sid or not settings.twilio_auth_token:
            return
        try:
            from twilio.rest import Client as TwilioClient
            tc = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            rec = tc.calls(self.call_sid).recordings.create()
            self._twilio_rec_sid = rec.sid
            log.info("Twilio recording started: %s", rec.sid)
        except Exception as e:
            log.warning("Could not start Twilio recording: %s", e)

    async def _download_twilio_recording(self) -> Optional[str]:
        """Poll until Twilio finishes processing the recording, then download it."""
        if not self._twilio_rec_sid:
            return None
        import requests as req
        try:
            from twilio.rest import Client as TwilioClient
            tc = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            for attempt in range(8):          # poll up to ~16 seconds
                await asyncio.sleep(2)
                rec = await asyncio.to_thread(tc.recordings(self._twilio_rec_sid).fetch)
                log.info("Twilio recording status (attempt %d): %s", attempt + 1, rec.status)
                if rec.status == "completed":
                    url = (
                        f"https://api.twilio.com/2010-04-01/Accounts/"
                        f"{settings.twilio_account_sid}/Recordings/{self._twilio_rec_sid}.wav"
                    )
                    resp = await asyncio.to_thread(
                        req.get, url,
                        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        base = Path(__file__).resolve().parent.parent.parent / "data" / "3 cases voice"
                        base.mkdir(parents=True, exist_ok=True)
                        p = base / f"{self.call_id}.wav"
                        p.write_bytes(resp.content)
                        log.info("Twilio recording saved: %s (%d bytes)", p, len(resp.content))
                        return str(p)
                    log.warning("Twilio recording download failed: HTTP %s", resp.status_code)
                    return None
                if rec.status == "failed":
                    log.warning("Twilio recording failed")
                    return None
        except Exception as e:
            log.error("Twilio recording download error: %s", e, exc_info=True)
        return None

    async def save(self) -> None:
        import soundfile as sf
        duration = int((datetime.now() - self.start_dt).total_seconds())
        audio_path: Optional[str] = None
        log.info("Saving call — duration=%ds twilio_rec=%s", duration, self._twilio_rec_sid)

        # Primary: Twilio-side recording — the real call exactly as it happened
        audio_path = await self._download_twilio_recording()

        # Fallback: reconstruct from our local tracks if Twilio recording failed
        if not audio_path:
            log.info("Twilio recording unavailable — building local mix as fallback")
            try:
                if self.caller_audio or self.agent_segs:
                    SR = 16_000
                    total_samples = max(duration * SR, 1)

                    caller_track = np.zeros(total_samples, dtype=np.float32)
                    if self.caller_audio:
                        caller_start = 0
                        if self._first_frame_t is not None:
                            caller_start = int((self._first_frame_t - self._call_t0) * SR)
                            caller_start = max(0, min(caller_start, total_samples - 1))
                        caller = np.concatenate(self.caller_audio)
                        end = min(caller_start + len(caller), total_samples)
                        caller_track[caller_start:end] = caller[:end - caller_start]

                    agent_track = np.zeros(total_samples, dtype=np.float32)
                    for offset_sec, audio in self.agent_segs:
                        start = int(offset_sec * SR)
                        end   = min(start + len(audio), total_samples)
                        if start < total_samples:
                            agent_track[start:end] = audio[:end - start]

                    for offset_sec, audio in self.agent_segs:
                        a_start = int(offset_sec * SR)
                        a_end   = min(a_start + len(audio) + int(0.15 * SR), total_samples)
                        if a_start < total_samples:
                            caller_track[a_start:a_end] = 0.0

                    mixed = caller_track * 0.8 + agent_track * 0.8
                    peak = np.max(np.abs(mixed))
                    if peak > 1.0:
                        mixed /= peak

                    base = Path(__file__).resolve().parent.parent.parent / "data" / "3 cases voice"
                    base.mkdir(parents=True, exist_ok=True)
                    p = base / f"{self.call_id}.wav"
                    sf.write(str(p), mixed, SR)
                    audio_path = str(p)
                    log.info("Fallback audio saved: %s", p)
                else:
                    log.warning("No audio captured — skipping WAV save")
            except Exception as e:
                log.error("Audio save failed: %s", e, exc_info=True)
        snap = self.memory.snapshot()
        snap["transcript"] = [t.model_dump() for t in self.turns]
        n_turns     = len([t for t in self.turns if t.role == "caller"])
        cost_usd    = _calc_cost(duration, n_turns)
        # Same invariant as VoiceBrain above: _Session cannot exist without
        # _require_classic() having passed.
        assert record_call is not None
        record, backend = record_call(
            snap, call_id=self.call_id,
            audio_path=audio_path, duration_seconds=duration,
            use_llm=False, persist=True, cost_usd=cost_usd,
        )
        _print_summary(record, backend, self.turns)


# ── registry ──────────────────────────────────────────────────────────────────

_sessions:      dict[str, _Session] = {}
_call_id_by_sid: dict[str, str]    = {}   # CallSid → call_id for recording filename
# CallSid → Doctor for realtime calls. Kept separate from _sessions because the
# realtime path builds no classic _Session. Popped when the media stream opens.
_pending_realtime_doctor: dict[str, Doctor] = {}

# When Twilio told us the callee picked up, per CallSid (time.monotonic()).
#
# "First audio 1.08s after response.create" was the only greeting figure this
# project had, and it starts the clock at OUR request — after /answer, after
# the media WebSocket is opened, after Twilio's stream-start handshake. The
# question actually being asked is "how long between them pressing answer and
# hearing a voice", and nothing measured it, so the pre-warm's effect on it was
# never established either way. /answer fires the instant the call is answered,
# which is as close to the pickup as this side of the wire can get.
_answered_at: dict[str, float] = {}

# CallSids whose Twilio Media Stream actually opened. This is the ONLY registry
# both call paths write, which is the whole reason it exists: /status used to
# ask `csid in _sessions` to decide whether anyone had spoken to the agent, but
# _sessions is written on the classic path alone — the realtime branch of
# /answer deliberately builds no _Session. The test was therefore False for
# every realtime call, and each one, including a flawless 86-second
# conversation, was reported as "NO CONVERSATION — nobody spoke to it".
#
# Written where the fact happens: Twilio opens this socket only after the call
# is answered, so its absence still catches the 2026-08-13 case (completed at
# 14s having never fetched /answer). Discarded in /status.
_media_opened: set[str] = set()

# ── Doctor routing ───────────────────────────────────────────────────────────
# `pending_doctor` was a single module global: the caller set it, /answer read
# it. Two concurrent calls therefore shared one doctor, and the second would
# quietly ask about the first one's — corrupt data, no error. It could not fire
# because the global itself made concurrency impossible, which is the worst
# kind of safe: the bug is dormant, and the thing keeping it dormant is the
# thing a batch runner removes first.
#
# Routing is by CallSid, which Twilio returns when the call is created.
#
# The global is GONE as of 2026-08-18, and removing it exposed why it had
# survived: `register_call` had no callers anywhere. The SID map was never
# populated, so every call in the programme's history resolved through the
# fallback. The concurrency-safe path existed, was tested, and was dead —
# run_twilio.py now registers the SID it gets back from Twilio.
_doctor_by_sid: dict[str, Doctor] = {}
_routing_lock = threading.Lock()

# How long /answer will wait for register_call to catch up. Twilio has to ring
# the far end before it fetches /answer, so this window is milliseconds of work
# against seconds of ringing — but "cannot happen" is what the global was for,
# and a bounded wait costs nothing on the path where it never triggers.
_ROUTING_WAIT_S = 2.0


# The server's event loop, captured at startup. run_twilio.py places the call
# from a threading.Timer, so register_call runs on a DIFFERENT thread from
# uvicorn — scheduling the pre-warm needs the loop explicitly rather than
# asyncio.get_event_loop(), which would find no loop on that thread.
_LOOP: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def _capture_loop() -> None:
    global _LOOP
    _LOOP = asyncio.get_running_loop()


def register_call(call_sid: str, doctor: Doctor) -> None:
    """Bind a placed call to its doctor, and start warming a session.

    The pre-warm rides along here because this is the moment the call is
    placed — the phone is about to ring, and those seconds are otherwise dead.
    See realtime_worker.prewarm_realtime for what it buys and why failing is
    free.
    """
    with _routing_lock:
        _doctor_by_sid[call_sid] = doctor
    if _LOOP is not None and settings.use_realtime:
        from agents.voice.realtime_worker import prewarm_realtime
        # Fire and forget, onto the server's loop from this thread. Never
        # awaited: the call must not wait on it, and prewarm_realtime cannot
        # raise.
        asyncio.run_coroutine_threadsafe(prewarm_realtime(call_sid), _LOOP)


async def _doctor_for(call_sid: str) -> Optional[Doctor]:
    """Which doctor is this CallSid about?

    Waits briefly rather than falling back to a shared global: the only case
    the fallback ever covered was the webhook overtaking register_call, and
    waiting fixes that without letting two concurrent calls read one another's
    doctor. Returns None if the SID is genuinely unknown, which /answer turns
    into a hangup — a call about the wrong doctor is worse than no call.

    Does NOT pop. Twilio retries webhooks, and a second /answer for the same
    SID must resolve to the same doctor rather than falling off the end. The
    entry is discarded in /status with the other per-call registries.
    """
    deadline = time.monotonic() + _ROUTING_WAIT_S
    while True:
        with _routing_lock:
            doctor = _doctor_by_sid.get(call_sid)
        if doctor is not None or time.monotonic() >= deadline:
            return doctor
        await asyncio.sleep(0.02)


# ── TwiML webhook ─────────────────────────────────────────────────────────────

@app.post("/answer")
async def answer(request: Request):
    form = await request.form()
    if not await _verify_twilio_signature(request, form):
        log.warning("Rejected unsigned /answer request")
        return _forbidden()
    csid = _form_str(form, "CallSid")
    doc  = await _doctor_for(csid)
    if not doc:
        log.error("No doctor registered for CallSid %s after %.1fs — hanging up "
                  "rather than calling about an unknown record", csid, _ROUTING_WAIT_S)
        return Response(_twiml_hangup(), media_type="application/xml")

    if settings.use_realtime:
        # Do NOT build a classic _Session here. Its constructor spawns a thread
        # that Piper-synthesises ~40 phrases which the realtime path never uses,
        # and the session was discarded at /stream anyway. realtime_worker owns
        # its own RealtimeSession, created when the media stream opens.
        _pending_realtime_doctor[csid] = doc
        _answered_at[csid] = time.monotonic()
        log.info("Call answered (realtime): %s", csid)
    else:
        sess = _Session(csid, doc)
        _sessions[csid] = sess
        _call_id_by_sid[csid] = sess.call_id   # so /recording_ready can name the file correctly
        log.info("Call answered: %s", csid)

    ws_url = (settings.server_public_url
              .replace("https://", "wss://")
              .replace("http://",  "ws://")) + f"/stream/{csid}"
    return Response(_twiml_stream(ws_url), media_type="application/xml")


@app.post("/status")
async def status_callback(request: Request):
    form   = await request.form()
    if not await _verify_twilio_signature(request, form):
        log.warning("Rejected unsigned /status request")
        return _forbidden()
    status = _form_str(form, "CallStatus")
    csid   = _form_str(form, "CallSid")
    # print, not log.info: nothing configures logging for the uvicorn process,
    # so this went nowhere and the only thing on screen was uvicorn's access
    # line, "POST /status 200 OK" — which says a webhook arrived, not what it
    # said. Two calls on 2026-08-13 were never picked up and the terminal gave
    # no hint of it; the outcome had to be read back out of the Twilio API
    # afterwards. Same defect as the swallowed API errors in realtime_worker.
    #
    # A call nobody answers never reaches handle_realtime, so no transcript, no
    # artifact and no CALL ENDED block is printed. This webhook is the ONLY
    # place the outcome surfaces.
    _never_connected = {
        "no-answer": "nobody picked up — it rang out",
        "busy":      "line was busy",
        "failed":    "the call failed to connect",
        "canceled":  "the call was cancelled before it connected",
    }
    _dur = _form_str(form, "CallDuration") or "?"
    if status in _never_connected:
        print(f"\n  ☎️  NOT ANSWERED — {_never_connected[status]} "
              f"(status={status}, {_dur}s)\n", flush=True)
    elif status == "completed":
        # "completed" only means Twilio finished the call normally — it does
        # NOT mean the agent was ever on a live call. One on 2026-08-13 came
        # back completed at 14s having never fetched /answer at all. The honest
        # test is whether the Media Stream connected, which is what
        # _media_opened records, on both paths.
        #
        # This asked `csid in _sessions` until 2026-08-18. Nothing writes
        # _sessions on the realtime path, so the branch below fired on every
        # single realtime call and told the operator that nobody had spoken to
        # an agent that had just held a full conversation. Claiming a call
        # failed when it succeeded is worse than saying nothing: it discredits
        # the one line here that is supposed to be trustworthy.
        #
        # Note what the marker does and does not prove. It proves the call was
        # answered and audio was flowing. It does NOT prove a person spoke —
        # voicemail and answer-machine pickups open a media stream too — so the
        # wording below stops at what is actually known. Whether a human said
        # anything is a question the transcript answers, and the realtime path
        # already prints that in its own CALL ENDED block.
        if csid in _media_opened:
            print(f"  ☎️  call completed ({_dur}s)", flush=True)
        else:
            print(f"\n  ☎️  NO MEDIA STREAM — Twilio reports completed "
                  f"({_dur}s), but the audio stream never connected, so the "
                  f"agent was never on this call and there is no transcript. "
                  f"Rang out, or was answered by something that did not "
                  f"connect the stream.\n", flush=True)
    else:
        print(f"  ☎️  call status: {status} ({_dur}s)", flush=True)
    if status == "completed":
        # Realtime sessions save themselves in handle_realtime's finally block.
        # _call_id_by_sid is left in place — /recording_ready fires after this
        # and needs it to name the MP3; that handler pops it.
        _pending_realtime_doctor.pop(csid, None)
        # /stream pops this on a normal call; this is the unanswered path, where
        # /stream never runs and the entry would otherwise be immortal.
        _answered_at.pop(csid, None)
        # Read above, discarded here — a registry that is only ever added to is
        # a leak in a batch runner, which is the stated direction for this.
        _media_opened.discard(csid)
        with _routing_lock:
            _doctor_by_sid.pop(csid, None)
        sess = _sessions.pop(csid, None)
        if sess:
            await sess.save()
    return Response(_twiml_ok(), media_type="application/xml")


async def _download_twilio_recording(recording_url: str, call_sid: str, recording_sid: str) -> None:
    """Download the Twilio recording MP3 and save it to the data folder.

    `recording_url` arrives in a webhook POST body, i.e. it is attacker-supplied
    unless the signature checked out. This request carries the Twilio account SID
    and auth token in an Authorization header, so it must only ever be sent to
    Twilio's own host, and it must not follow redirects — a 302 off-host would
    forward those credentials to wherever the redirect points.
    """
    import httpx
    mp3_url = recording_url + ".mp3"
    if not _is_twilio_recording_url(mp3_url):
        log.error("Refusing to send Twilio credentials to non-Twilio host: %r", mp3_url)
        return
    dest_dir = Path(__file__).resolve().parent.parent.parent / "data" / "3 cases voice"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Name the file after the call_id so it matches the JSON (e.g. call-20260707-0017-e1a7.mp3)
    call_id = _call_id_by_sid.pop(call_sid, call_sid)
    dest = dest_dir / f"twilio-{call_id}.mp3"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                mp3_url,
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                follow_redirects=False,
            )
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            print(f"\n📼 Twilio recording saved → {dest}", flush=True)
        else:
            print(f"[Recording] Download failed: HTTP {resp.status_code} — {mp3_url}", flush=True)
    except Exception as e:
        print(f"[Recording] Download error: {e}", flush=True)


@app.post("/recording_ready")
async def recording_ready(request: Request):
    """Twilio calls this when a call recording is ready to download."""
    form           = await request.form()
    if not await _verify_twilio_signature(request, form):
        log.warning("Rejected unsigned /recording_ready request")
        return _forbidden()
    recording_url  = _form_str(form, "RecordingUrl")
    recording_sid  = _form_str(form, "RecordingSid")
    recording_status = _form_str(form, "RecordingStatus")
    call_sid       = _form_str(form, "CallSid")
    duration       = _form_str(form, "RecordingDuration")
    print(f"\n📼 Recording ready  : status={recording_status} dur={duration}s sid={recording_sid}", flush=True)
    if recording_status == "completed" and recording_url:
        asyncio.create_task(_download_twilio_recording(recording_url, call_sid, recording_sid))
    return Response(_twiml_ok(), media_type="application/xml")


def _twiml_stream(ws_url: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{ws_url}"/></Connect>'
        "</Response>"
    )

def _twiml_hangup() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'

def _twiml_ok() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response/>'


# ── WebSocket media stream ────────────────────────────────────────────────────

@app.websocket("/stream/{call_sid}")
async def media_stream(ws: WebSocket, call_sid: str):
    await ws.accept()
    # Both paths, before either branches — the marker means "Twilio connected
    # the media stream", and that is equally true whichever worker handles it.
    _media_opened.add(call_sid)

    # How long Twilio + the tunnel took to open this socket after /answer
    # returned the TwiML naming it.
    #
    # On call-20260820-1154 the callee heard nothing for 6.73s, of which 5.83s
    # was "Twilio setup" — a single black-box number covering TwiML fetch,
    # tunnel routing, TLS, and Twilio's own stream handshake. That is the
    # largest term in the dead air and the least understood, and no baseline
    # exists for it: pickup_to_greeting_s was only added the commit before.
    #
    # Splitting it here costs one subtraction and makes the next call answer
    # the question that cannot be answered now — is the tunnel slow, or is
    # Twilio? Deliberately NOT attributed in the wording: this measures the
    # span, and which half owns it is what the number is for.
    _ans = _answered_at.get(call_sid)
    if _ans is not None:
        print(f"[Twilio] Media socket open {time.monotonic() - _ans:.2f}s after "
              f"/answer returned (Twilio TwiML fetch + tunnel + TLS)", flush=True)

    # ── Realtime: speech-to-speech only, no fallback ──────────────────────────
    # The old code fell back to the classic VAD→STT→LLM→TTS pipeline on any
    # realtime error. That silently changed which system was on the phone
    # mid-call, on a WebSocket whose messages had already been partly consumed,
    # so it could not work anyway. Fail loudly instead.
    if settings.use_realtime:
        doctor = _pending_realtime_doctor.pop(call_sid, None)
        if not doctor:
            log.error("[Realtime] No pending doctor for CallSid %s", call_sid)
            await ws.close()
            return
        from agents.voice.realtime_worker import handle_realtime
        try:
            await handle_realtime(ws, call_sid, doctor,
                                  answered_at=_answered_at.pop(call_sid, None))
        except Exception as e:
            log.error("[Realtime] Call failed: %s", e, exc_info=True)
            try:
                await ws.close()
            except Exception:
                pass
        return

    sess = _sessions.get(call_sid)
    if not sess:
        await ws.close()
        return

    sess.ws = ws
    sess._last_caller_speech_t = time.monotonic()  # track when caller last spoke

    try:
        async for raw in ws.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event", "")

            if event == "start":
                sess.stream_sid = msg["start"]["streamSid"]
                log.info("Stream started: %s", sess.stream_sid)
                sess._last_caller_speech_t = time.monotonic()
                # Start Twilio native recording in background (non-blocking)
                asyncio.create_task(asyncio.to_thread(sess._start_twilio_recording))
                greeting = sess.brain.greet()
                sess.add_turn("agent", greeting)
                log.info("Agent: %s", greeting)
                await sess.send_speech(greeting)

            elif event == "media":
                if msg["media"].get("track") != "inbound":
                    continue

                # Always decode and record — never skip (skipping causes silent gaps in WAV)
                arr = telnyx_to_float32(msg["media"]["payload"])
                if sess._first_frame_t is None:
                    sess._first_frame_t = time.monotonic()
                sess.caller_audio.append(arr)

                if sess.processing:
                    continue

                # ── Barge-in detection (agent is speaking) ────────────────────
                if sess.speaking:
                    # Use WebRTC VAD (accurate voice detection) + low RMS fallback.
                    # Decay counter instead of hard-reset so natural inter-word pauses
                    # don't cancel out a real barge-in attempt.
                    rms      = float(np.sqrt(np.mean(arr ** 2)))
                    is_voice = _vad_is_speech(arr) if len(arr) >= _VAD_FRAME_SAMPLES else False
                    if is_voice or rms > 0.012:
                        sess._barge_frames += 1
                        sess._preroll.append(arr)
                        if sess._barge_frames >= 2:  # 2 × 20ms = 40ms of voice → interrupt
                            sess._barge_in = True
                    else:
                        sess._barge_frames = max(0, sess._barge_frames - 1)  # decay, not hard reset
                    continue

                # Post-TTS dead zone: accumulate preroll but don't start speech detection.
                # Prevents line noise and TTS echo from immediately triggering a new cycle.
                if time.monotonic() < sess._listen_after:
                    sess._preroll.append(arr)
                    continue

                # Accumulate into VAD chunk buffer
                sess._vad_buf = np.concatenate([sess._vad_buf, arr])

                # Process in 320-sample (20ms) chunks
                while len(sess._vad_buf) >= _VAD_FRAME_SAMPLES:
                    chunk = sess._vad_buf[:_VAD_FRAME_SAMPLES]
                    sess._vad_buf = sess._vad_buf[_VAD_FRAME_SAMPLES:]

                    is_speech = _vad_is_speech(chunk)

                    if is_speech:
                        sess._last_caller_speech_t = time.monotonic()  # reset silence timer
                        if sess._speech_ct == 0:
                            # Speech just started — prepend preroll to catch first syllable
                            sess.speech_buf.extend(sess._preroll)
                            sess._preroll.clear()
                        sess._speech_ct += 1
                        sess._end_ct     = 0
                        sess.speech_buf.append(chunk)

                    else:
                        if sess._speech_ct > 0:
                            # After speech — counting end-of-speech silence
                            sess._end_ct += 1
                            sess.speech_buf.append(chunk)  # include trailing silence
                        else:
                            # Before speech — update preroll window
                            sess._preroll.append(chunk)

                    # ── End-of-speech detected (adaptive) ────────────────────
                    caller_done = (sess._speech_ct >= _MIN_SPEECH_FRAMES
                                   and sess._end_ct >= _adaptive_eos(sess._speech_ct))
                    too_long    = sess._speech_ct >= _MAX_SPEECH_FRAMES

                    if (caller_done or too_long) and sess.speech_buf:
                        speech = np.concatenate(sess.speech_buf)
                        sess._reset_vad()
                        sess.processing = True

                        async def handle(audio: np.ndarray) -> None:
                            try:
                                # Reject audio that's too quiet — noise burst, not real speech
                                rms = float(np.sqrt(np.mean(audio ** 2)))
                                if rms < 0.006:
                                    log.info("STT skipped — low energy audio (RMS=%.4f)", rms)
                                    return
                                ctx = f"{sess.doctor.doctor_name} {sess.doctor.hospital_name or ''}"
                                text = await asyncio.to_thread(_transcribe, audio, ctx)
                                if not text:
                                    log.info("STT empty — skipping")
                                    return
                                # Single filler/conjunction word — caller paused mid-sentence.
                                # Skip LLM entirely so they can finish what they were saying.
                                _FILLERS = {"and","but","or","so","well","um","uh","hmm",
                                            "ah","okay","ok","yeah","right","like","just"}
                                if len(text.split()) == 1 and text.lower().strip(".,!?") in _FILLERS:
                                    log.info("Filler word only (%r) — waiting for caller to finish", text)
                                    return
                                log.info("Caller: %s", text)
                                sess.add_turn("caller", text)

                                # Run LLM in a task; play a filler only if it takes > 3s
                                # AND only once per call — repeated fillers sound robotic.
                                llm_task = asyncio.create_task(
                                    asyncio.to_thread(sess.brain.handle, text)
                                )
                                try:
                                    reply = await asyncio.wait_for(
                                        asyncio.shield(llm_task), timeout=3.0
                                    )
                                except asyncio.TimeoutError:
                                    if sess._filler_count == 0:
                                        import random
                                        await sess.send_speech(random.choice(_THINKING_FILLERS))
                                        sess._filler_count += 1
                                    reply = await llm_task

                                if not reply:
                                    reply = f"Could you tell me which branch Dr. {sess.doctor.doctor_name} is based at?"

                                # Deduplication — don't say the exact same thing twice in a row
                                if reply == sess._last_reply:
                                    clean = re.sub(r"^Dr\.?\s+", "", sess.doctor.doctor_name, flags=re.I).strip()
                                    reply = (f"Sorry about that — just to confirm: which branch or city "
                                             f"is Dr. {clean} currently working at?")
                                sess._last_reply = reply

                                sess.add_turn("agent", reply)
                                log.info("Agent: %s | done=%s", reply, sess.brain.done)
                                await sess.send_speech(reply)
                                if sess.brain.done:
                                    await asyncio.sleep(2.5)
                                    try:
                                        await ws.close()
                                    except Exception:
                                        pass  # already closed by Twilio — ignore
                            finally:
                                sess.processing = False

                        asyncio.create_task(handle(speech))
                        break  # exit inner while — new frames will be handled next event

                    # ── Silence timeout: caller hasn't responded for 8s ──────────
                    # Only fires when NOT in a speech segment, NOT processing, NOT speaking,
                    # and only up to 2 times per call to avoid nagging.
                    elif (sess._speech_ct == 0
                          and not sess.processing
                          and not sess.speaking
                          and sess._silence_probe_count < 2
                          and time.monotonic() - sess._last_caller_speech_t > 8.0
                          and time.monotonic() > sess._listen_after + 1.0):
                        sess._last_caller_speech_t = time.monotonic()  # reset so we don't spam
                        sess._silence_probe_count += 1
                        async def _probe_silence():
                            if not sess.speaking and not sess.processing:
                                await sess.send_speech("Oh, are you still there?")
                        asyncio.create_task(_probe_silence())

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "not connected" in str(e).lower() or "accept" in str(e).lower():
            log.debug("WebSocket closed by Twilio: %s", e)
        else:
            log.error("Stream runtime error: %s", e, exc_info=True)
    except Exception as e:
        log.error("Stream error: %s", e, exc_info=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _calc_cost(duration_seconds: int, n_caller_turns: int) -> float:
    """Total call cost in USD: gpt-4o-mini (estimated) + Twilio."""
    llm_in_tokens  = n_caller_turns * 800
    llm_out_tokens = n_caller_turns * 150
    llm_cost    = (llm_in_tokens / 1_000_000 * 0.15) + (llm_out_tokens / 1_000_000 * 0.60)
    twilio_cost = (duration_seconds / 60.0) * 0.0165
    return llm_cost + twilio_cost


def _transcribe(audio_16k: np.ndarray, context: str = "") -> str:
    from agents.experiment.stt_whisper import transcribe_array
    return transcribe_array(audio_16k, context=context)


def _print_summary(record, backend: str, turns: list[TranscriptTurn]) -> None:
    from rich.console import Console
    from rich.table import Table
    c = Console()
    c.rule("[bold]Call Complete")
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("[dim]Call ID[/dim]",   record.call_id)
    t.add_row("[dim]Doctor[/dim]",    record.doctor_name)
    t.add_row("[dim]Hospital[/dim]",  record.hospital_name or "—")
    t.add_row("[dim]Branch[/dim]",
              f"[green]{record.branch}[/green]" if record.branch else "[dim]not obtained[/dim]")
    t.add_row("[dim]Resolved[/dim]",
              "[green]Yes[/green]" if record.resolved else "[red]No[/red]")
    t.add_row("[dim]Duration[/dim]",  f"{record.duration_seconds}s")
    t.add_row("[dim]Audio[/dim]",     record.audio_path or "—")
    t.add_row("[dim]Saved to[/dim]",  backend)
    c.print(t)
    c.print("\n[bold]Summary:[/bold]", record.summary)
    c.print("\n[bold]Transcript:[/bold]")
    for turn in turns:
        color = "cyan" if turn.role == "agent" else "green"
        c.print(f"  [dim]{turn.timestamp}[/dim]  [{color}]{turn.role.title()}[/{color}]  {turn.text}")

    # ── Cost breakdown ────────────────────────────────────────────────
    dur     = record.duration_seconds or 0
    n_turns = len([t for t in turns if t.role == "caller"])
    llm_in_tokens  = n_turns * 800
    llm_out_tokens = n_turns * 150
    llm_cost    = (llm_in_tokens / 1_000_000 * 0.15) + (llm_out_tokens / 1_000_000 * 0.60)
    twilio_cost = (dur / 60.0) * 0.0165
    total       = record.cost_usd or (llm_cost + twilio_cost)
    c.print(f"\n[bold]Cost Breakdown:[/bold]")
    c.print(f"  Duration        : {dur}s ({dur/60:.1f} min)")
    c.print(f"  LLM turns       : {n_turns}  (~{llm_in_tokens} in / {llm_out_tokens} out tokens)")
    c.print(f"  gpt-4o-mini     : ${llm_cost:.4f}")
    c.print(f"  Twilio          : ${twilio_cost:.4f}")
    c.print(f"  [bold]TOTAL           : ${total:.4f}[/bold]")
