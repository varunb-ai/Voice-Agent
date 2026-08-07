"""OpenAI Realtime API bridge for Twilio voice calls — speech-to-speech only.

Architecture:
    Twilio WS  ←→  FastAPI  ←→  OpenAI Realtime WS
   (μ-law 8kHz)               (PCM16 24kHz)

One persistent WebSocket carries the whole conversation. Caller audio goes in,
agent audio comes out, and the model never round-trips through a separate STT
or TTS service. Inline `input_audio_transcription` runs alongside purely to
produce the written transcript — it is not in the conversational path, and
nothing waits on it.

Two things this module deliberately does NOT do, because both broke the
speech-to-speech guarantee:
  * no out-of-band whisper-1 HTTP transcription per caller turn
  * no fallback to the classic VAD→STT→LLM→TTS pipeline

Prompt caching: the session is configured once with a template's STATIC
instructions, and per-call facts are sent as the first conversation item. Never
put the doctor or hospital into `instructions`, and never pass a per-response
`instructions` override — either one moves the cache boundary and the whole
prefix is re-billed on every turn. See agents/voice/templates.py.

Enable with USE_REALTIME=true in .env.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
import websockets.exceptions
from fastapi import WebSocket

from core.config import settings
from core.models import Doctor, TranscriptTurn
from agents.voice.memory import CallMemory
from agents.voice.templates import get_template
from agents.voice.tools import run_tool, TOOL_SCHEMAS
from agents.voice.audio_utils import resample, _mulaw_decode, _mulaw_encode

log = logging.getLogger(__name__)

# Caller-utterance RMS below which the line is treated as too faint to trust.
# Clear phone speech measures roughly 0.03-0.08; a live call that produced a
# fabricated answer measured 0.004-0.012 throughout.
_LOW_AUDIO_RMS = 0.015

# Where call artefacts land. Indirected through functions so tests can point
# them at a temp directory — the protocol suite used to write real WAVs and
# JSON into data/ on every run, polluting the actual call records.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def audio_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases voice"


def json_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases jsons"


REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"
# Serialises the read-modify-write of master.json. See save().
_MASTER_LOCK = threading.Lock()

_TWILIO_SR = 8_000
_OAI_SR    = 24_000


# ── Audio format conversion ───────────────────────────────────────────────────

def _convert_twilio_to_oai(payload_b64: str) -> str:
    """Twilio base64(μ-law 8kHz) → base64(PCM16 24kHz) for OpenAI."""
    raw     = base64.b64decode(payload_b64)
    f32_8k  = _mulaw_decode(raw)
    f32_24k = resample(f32_8k, _TWILIO_SR, _OAI_SR)
    pcm16   = (np.clip(f32_24k, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode()


def _convert_oai_to_twilio(pcm16_b64: str) -> str:
    """OpenAI base64(PCM16 24kHz) → base64(μ-law 8kHz) for Twilio."""
    raw     = base64.b64decode(pcm16_b64)
    f32_24k = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    f32_8k  = resample(f32_24k, _OAI_SR, _TWILIO_SR)
    return base64.b64encode(_mulaw_encode(f32_8k)).decode()


# ── μ-law passthrough ─────────────────────────────────────────────────────────
# With REALTIME_AUDIO_FORMAT=pcmu the session speaks the same g711 μ-law Twilio
# already sends, so frames cross untouched in both directions: no μ-law decode,
# no 8k→24k resample inbound, no 24k→8k resample and re-encode outbound. That is
# two resamples removed from each 20ms frame, 50 frames a second, each way.
#
# Recording still needs linear PCM, but only to write a WAV — decoding μ-law at
# 8kHz is cheap and skips the resample entirely, so the recording is written at
# 8kHz rather than 24kHz. It is a phone call; 8kHz is the true bandwidth anyway.

def _passthrough_enabled() -> bool:
    return settings.realtime_audio_format == "pcmu"


def _wire_sample_rate() -> int:
    """Sample rate of whatever is stored in the recording buffers."""
    return _TWILIO_SR if _passthrough_enabled() else _OAI_SR


def _wire_to_pcm16(raw: bytes) -> np.ndarray:
    """Decode a recording-buffer chunk to float32, whatever format it holds."""
    if _passthrough_enabled():
        return _mulaw_decode(raw)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _wire_samples(raw: bytes) -> int:
    """Number of audio samples in a chunk. μ-law is 1 byte/sample, PCM16 is 2."""
    return len(raw) if _passthrough_enabled() else len(raw) // 2


def _loudest_window_rms(arr: np.ndarray, window_s: float = 0.3) -> float:
    """RMS of the LOUDEST window in an utterance, not the mean across it.

    Mean RMS over a whole utterance is dominated by the gaps between words. On
    a live call a perfectly audible caller — peak 0.098, twelve windows above
    0.02, every turn transcribing cleanly — measured 0.0016 by the mean and was
    told "you're coming through faint". Telling an audible person they cannot
    be heard is its own way to lose the call.

    The loudest window answers the question actually being asked: when they
    were speaking, was there enough signal? Utterances too short to judge
    return 0.0, which the caller treats as "no opinion".
    """
    if arr.size == 0:
        return 0.0
    sr = _wire_sample_rate()
    win = int(window_s * sr)
    if arr.size < win:
        return 0.0     # too short to judge — do not guess
    best = 0.0
    for start in range(0, arr.size - win + 1, max(1, win // 2)):
        seg = arr[start:start + win]
        best = max(best, float(np.sqrt(np.mean(seg ** 2))))
    return best


_LOCATION_ASK = re.compile(
    r"(which|what|where).{0,40}(branch|location|office|campus|site|"
    r"practis|practic|work)", re.I)


# Asks carrying no question mark. Once the brevity rules were relaxed the agent
# began asking in softer statement form — "I'm just trying to find out which
# branch she works at" — which is every bit a request, but the counter did not
# see it. The budget therefore read 3 asks on a call where the caller said
# "why are you keep on asking the same question? It's kind of irritating."
_SOFT_LOCATION_ASK = re.compile(
    r"(trying to (find|work) out|need to know|just need|could you tell me|"
    r"you can just say|let me know|hoping to (get|find))"
    r".{0,60}(branch|location|office|campus|site)", re.I)


# Turns that MENTION the location without asking for it: acknowledging a value
# just given, or signing off. Everything else that names a location is a request,
# whatever shape it takes.
_NOT_AN_ASK = re.compile(
    r"\b(thanks|thank you|got it|perfect|great|appreciate|have a (good|great)|"
    r"take care|goodbye|bye now|i'?ll (note|record|pass)|i have that|"
    r"that'?s all|no (problem|worries))\b", re.I)

# Reading back a value the caller already gave.
_CONFIRMS_VALUE = re.compile(
    r"\b(i have that as|i'?ve got that|i'?ll note|noted as|recorded as|"
    r"i'?ll put (that|it) down|so that'?s)\b", re.I)

_LOCATION_NOUN = re.compile(
    r"\b(branch|location|office|campus|site|address|practis\w*|practic\w*)\b", re.I)


def _is_location_ask(text: str) -> bool:
    """Is this agent turn asking where the doctor practises?

    Counts statement-form asks as well as questions. A request phrased politely
    is still a request, and the person on the other end experiences it as one.

    This used to be a whitelist of phrasings requiring a question mark, and it
    scored 0 asks on a call that asked four times — the agent had simply picked
    wordings that were not on the list ("trying to confirm" where the list held
    "trying to find out"). Enumerating phrasings cannot work: the model has more
    ways to ask than anyone can list.

    So it is inverted. Naming a location IS an ask unless the turn is plainly
    acknowledging or closing. This over-counts a little, which is the safe
    direction for a budget whose purpose is to stop the agent pestering people.
    """
    if not _LOCATION_NOUN.search(text):
        return False
    # Reading a value back is not asking for one.
    if "?" not in text and _CONFIRMS_VALUE.search(text):
        return False
    if "?" in text:
        return True
    # An acknowledgement that goes on to ask for something is still an ask, so
    # only a turn that is ENTIRELY acknowledgement is exempt.
    stripped = _NOT_AN_ASK.sub("", text)
    return bool(_LOCATION_NOUN.search(stripped))


def _ask_budget_outcome(turns: list, sent_at: Optional[int],
                        sent: bool, escalated: bool) -> dict:
    """What happened after the give-up directive was injected.

    The count alone is not enough. Thanking them and escalating in one turn is
    the directive working. Taking two turns where the first contains another
    question is the directive landing but not taking effect — a soft version of
    the model ignoring it outright, and worth telling apart, because the fix
    differs: a wording tweak versus enforcing the budget at the response level
    instead of asking nicely in a user turn.
    """
    if not sent or sent_at is None:
        return {"limit": settings.realtime_max_location_asks,
                "directive_sent": False, "verdict": "not needed"}

    after = [t for t in turns[sent_at:] if t.role == "agent"]
    asked_again = sum(1 for t in after if _is_location_ask(t.text))

    if not escalated:
        verdict = "IGNORED — directive sent, agent never escalated"
    elif asked_again:
        verdict = f"OBEYED LATE — asked {asked_again} more time(s) first"
    elif len(after) <= 1:
        verdict = "OBEYED — closed on the next turn"
    else:
        verdict = f"OBEYED — took {len(after)} turns, no further asks"

    return {
        "limit": settings.realtime_max_location_asks,
        "directive_sent": True,
        "agent_turns_after": len(after),
        "asked_again_after": asked_again,
        "escalated": escalated,
        "verdict": verdict,
        "turns_after": [t.text[:90] for t in after],
    }


def conversation_metrics(turns: list) -> dict:
    """Count the conversational failures that prose rules keep failing to stop.

    Three attempts at fixing the same behaviour by writing more forceful
    instructions, each one ignored, is evidence the marginal rule is doing
    less each time. These are constraints on the SHAPE of a turn rather than
    its content, which makes them detectable in code even though the prompt
    cannot reliably enforce them.

    Nothing here changes behaviour — you cannot unsay a turn. The point is to
    have a number, so the next prompt edit can be evaluated against the last
    one instead of by reading a transcript and forming an impression.

      stapled_questions  — agent asked a question in the same turn it answered
                           one of theirs. Six of these in one 111s call.
      back_to_back_asks  — agent asked in two consecutive turns.
      repeated_sentences — same agent sentence said more than once.
    """
    agent = [t for t in turns if t.role == "agent"]
    stapled = back_to_back = 0
    prev_agent_asked = False

    for i, turn in enumerate(turns):
        if turn.role != "agent":
            continue
        asks = "?" in turn.text
        prev_caller = next((turns[j] for j in range(i - 1, -1, -1)
                            if turns[j].role == "caller"), None)
        # Did the caller's most recent turn, immediately before this one, ask
        # something? Then answering AND asking in one breath is the failure.
        if asks and prev_caller is not None and i > 0 and turns[i - 1] is prev_caller \
                and "?" in prev_caller.text:
            stapled += 1
        if asks and prev_agent_asked:
            back_to_back += 1
        prev_agent_asked = asks

    seen: dict[str, int] = {}
    for t in agent:
        for sentence in re.split(r"(?<=[.!?])\s+", t.text.strip()):
            key = sentence.strip().lower()
            if len(key.split()) >= 4:
                seen[key] = seen.get(key, 0) + 1
    repeated = sum(n - 1 for n in seen.values() if n > 1)

    # Denominators, so counts can be compared across calls of different
    # difficulty. A hostile caller who answers nothing gives the agent six
    # chances to staple; a cooperative one gives it one. Raw counts make the
    # easy call look better when it may simply have had fewer opportunities.
    caller = [t for t in turns if t.role == "caller"]
    caller_questions = sum(1 for t in caller if "?" in t.text)
    return {
        # How many times it asked where the doctor practises. On the call that
        # exposed this it was six, with no location offered between any of
        # them — the number that says "it would not let go".
        "location_asks": sum(1 for t in agent if _is_location_ask(t.text)),
        "agent_turns": len(agent),
        "caller_turns": len(caller),
        "caller_questions": caller_questions,
        "question_turns": sum(1 for t in agent if "?" in t.text),
        "stapled_questions": stapled,
        # The rate is the comparable figure: of the times they asked something,
        # how often did the agent answer and ask back in the same breath?
        "staple_rate": round(stapled / caller_questions, 2) if caller_questions else None,
        "back_to_back_asks": back_to_back,
        "repeated_sentences": repeated,
    }


# Escalation reasons that assert a FACT about the doctor rather than describing
# how the call went. "declined to share" is an observation about the call and
# needs no evidence; "doctor deceased" is a claim about a real person and does.
_FACTUAL_ESCALATIONS = {
    "deceased": ("deceased", "died", "passed away", "passed", "late "),
    "retired":  ("retired", "retirement"),
    "left":     ("left", "no longer", "moved on", "resigned", "quit"),
    "relocated": ("relocated", "transferred", "moved to"),
    "on leave": ("on leave", "maternity", "sabbatical", "sick leave"),
}


def _ungrounded_escalation(reason: str, sess: "RealtimeSession") -> str:
    """Reject an escalation reason asserting something the caller never said.

    A live call ended with escalate(reason="doctor deceased") after the caller
    said only "actually, he's not working right now". Nobody said died, passed
    away, or deceased. save_branch was guarded against exactly this and
    escalate was not — so a fabricated claim about a named real person went
    into the record, where a reviewer would read it as fact.

    Only claims ABOUT THE DOCTOR are checked. Reasons describing the call
    itself ("declined to share", "wrong number", "no response") are the agent's
    own observation and need no corroboration.
    """
    heard = " ".join(t.text.lower() for t in sess.turns
                     if t.role == "caller" and t.text.strip() != "[...]")
    if not heard.strip():
        return ""
    low = reason.lower()
    for claim, markers in _FACTUAL_ESCALATIONS.items():
        if claim in low and not any(m in heard for m in markers):
            return (f"reason {reason!r} states the doctor is {claim}, which "
                    f"nobody said on this call")
    return ""


def _echo_gate_allows(raw: bytes) -> bool:
    """Should this caller frame reach OpenAI while the agent is speaking?

    Governs whether the caller can interrupt at all. See REALTIME_ECHO_GATE.
    """
    mode = settings.realtime_echo_gate
    if mode == "pass":
        return True
    if mode == "energy":
        arr = _wire_to_pcm16(raw)
        if arr.size == 0:
            return False
        return float(np.sqrt(np.mean(arr ** 2))) >= settings.realtime_echo_rms
    return False   # "drop"


# ── Tool schema conversion ────────────────────────────────────────────────────

def _realtime_tools() -> list[dict]:
    """Convert TOOL_SCHEMAS (chat format) → OpenAI Realtime flat format."""
    result = []
    for s in TOOL_SCHEMAS:
        if s["type"] == "function":
            fn = s["function"]
            result.append({
                "type": "function",
                "name": fn["name"],
                "description": fn["description"],
                "parameters": fn["parameters"],
            })
    return result


# ── Session audio configuration ───────────────────────────────────────────────

def build_audio_config(*, transcribe_model: str, transcribe_hint: str,
                       audio_format: str, noise_reduction: str,
                       turn_detection: str, eagerness: str,
                       voice: str, silence_ms: int = 500) -> dict:
    """Assemble the session.update `audio` block.

    Split out so check_realtime.py can probe variants against the live API
    without duplicating the shape — the settings below are empirical questions,
    not things to settle by reading.
    """
    fmt: dict = ({"type": "audio/pcmu"} if audio_format == "pcmu"
                 else {"type": "audio/pcm", "rate": _OAI_SR})

    if turn_detection == "semantic_vad":
        td: dict = {"type": "semantic_vad", "eagerness": eagerness}
    else:
        td = {
            "type": "server_vad",
            "threshold": 0.55,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_ms,
        }

    audio_in: dict = {
        "format": fmt,
        "transcription": {
            "model": transcribe_model,
            "language": "en",
            "prompt": transcribe_hint,
        },
        "turn_detection": td,
    }
    if noise_reduction and noise_reduction != "off":
        audio_in["noise_reduction"] = {"type": noise_reduction}

    return {"input": audio_in, "output": {"format": fmt, "voice": voice}}


# ── Grounding: a saved location must be one the caller actually said ─────────

# Words that carry no identifying information, so their presence in the
# transcript proves nothing about whether the caller named a real place.
_UNGROUNDED_STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "our", "their", "and",
    "branch", "branches", "office", "offices", "campus", "campuses",
    "clinic", "clinics", "center", "centre", "centers", "centres",
    "hospital", "location", "locations", "site", "sites", "medical",
    "building", "unit", "practice", "city", "street", "road", "avenue",
}


def _ungrounded_terms(args: dict, sess: "RealtimeSession") -> str:
    """Return a description of any branch/city term the caller never said.

    Empty string means everything checks out. Compares against the caller's
    own transcribed words only — the agent's words are excluded, or the model
    could ground a fabrication in its own earlier hallucination.

    If no caller speech was transcribed at all (every turn still a `[...]`
    placeholder) the check is skipped rather than blocking every save, since
    absence of transcript is not evidence of fabrication.
    """
    heard = " ".join(
        t.text.lower() for t in sess.turns
        if t.role == "caller" and t.text.strip() != "[...]"
    )
    if not heard.strip():
        return ""   # nothing transcribed — cannot judge, do not block

    missing = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        terms = [w.strip(".,!?-—'\"") for w in value.lower().split()]
        content = [w for w in terms if w and w not in _UNGROUNDED_STOPWORDS]
        if not content:
            continue
        # One content word appearing is enough — transcription is imperfect and
        # we would rather let a real answer through than block it.
        if not any(w in heard for w in content):
            missing.append(f"{field}={value!r}")
    return " and ".join(missing)


# ── Per-call session ──────────────────────────────────────────────────────────

class RealtimeSession:
    def __init__(self, call_sid: str, doctor: Doctor):
        ts               = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id     = f"call-{ts}-{call_sid[-4:]}"
        self.call_sid    = call_sid
        self.doctor      = doctor
        self.start_dt    = datetime.now()
        self.memory      = CallMemory(call_id=self.call_id)
        self.memory.clear()
        self.memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
        self.turns: list[TranscriptTurn] = []
        self.stream_sid  = ""
        self.done        = False
        # Gate: don't forward caller audio until the greeting finishes playing
        self.listen_enabled = asyncio.Event()
        # Gate: don't forward caller audio while the agent is speaking, plus an
        # echo cooldown after. Frames arriving in this window are dropped.
        self.agent_speaking = False

        # Agent PCM: list of (time_offset_from_stream_start_seconds, pcm16_bytes)
        # Timestamps let us place agent audio at the right position on the timeline
        self._agent_pcm: list[tuple[float, bytes]] = []
        # Caller PCM: continuous stream from Twilio — already timeline-aligned
        self._caller_pcm: list[bytes] = []
        # When the Twilio stream started (set on "start" event)
        self._stream_start_time: Optional[datetime] = None
        # Set when response.create for the greeting is sent; cleared once the
        # first audio delta arrives, so we measure the callee's dead air.
        self._greeting_requested_at: Optional[float] = None
        # Warn the model about a faint line at most once per call.
        self._low_audio_warned: bool = False
        # Asks for the location so far, and whether we have already told the
        # model to stop. See realtime_max_location_asks.
        self._location_asks: int = 0
        self._give_up_sent: bool = False
        # Turn index when the give-up directive was injected, so we can tell
        # afterwards whether the agent actually acted on it. The directive is
        # appended to the conversation and there is no second lever, so its
        # effectiveness has to be measured rather than assumed.
        self._give_up_at_turn: Optional[int] = None

        # Token usage tracking (from response.done events).
        # Cached tokens are counted SEPARATELY and billed at the cached rate —
        # they are the only direct evidence that prompt caching is working, so
        # they are tracked even though the totals include them.
        self._input_audio_tokens:        int = 0
        self._input_audio_cached_tokens: int = 0
        self._output_audio_tokens:       int = 0
        self._input_text_tokens:         int = 0
        self._input_text_cached_tokens:  int = 0
        self._output_text_tokens:        int = 0
        self._responses:                 int = 0

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(TranscriptTurn(
            role=role,
            text=text,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))

    def _cost_lines(self, duration_seconds: int) -> tuple[list[tuple[str, str, float]], float]:
        """Itemise the call cost. Returns ([(label, detail, usd)], total_usd).

        Cached input is billed at the cached rate, so the totals reflect whether
        prompt caching actually engaged rather than assuming it did or didn't.
        Rates come from core.config — verify them against current OpenAI pricing.
        """
        s = settings
        audio_in_fresh = max(0, self._input_audio_tokens - self._input_audio_cached_tokens)
        text_in_fresh  = max(0, self._input_text_tokens  - self._input_text_cached_tokens)

        rows = [
            ("Audio in",        f"{audio_in_fresh:,} tok",
             audio_in_fresh / 1_000_000 * s.price_audio_in),
            ("Audio in cached", f"{self._input_audio_cached_tokens:,} tok",
             self._input_audio_cached_tokens / 1_000_000 * s.price_audio_in_cached),
            ("Audio out",       f"{self._output_audio_tokens:,} tok",
             self._output_audio_tokens / 1_000_000 * s.price_audio_out),
            ("Text in",         f"{text_in_fresh:,} tok",
             text_in_fresh / 1_000_000 * s.price_text_in),
            ("Text in cached",  f"{self._input_text_cached_tokens:,} tok",
             self._input_text_cached_tokens / 1_000_000 * s.price_text_in_cached),
            ("Text out",        f"{self._output_text_tokens:,} tok",
             self._output_text_tokens / 1_000_000 * s.price_text_out),
            ("Telephony",       f"{duration_seconds/60:.2f} min",
             duration_seconds / 60.0 * s.price_telephony_per_min),
        ]
        return rows, sum(usd for _, _, usd in rows)

    def _calc_cost(self, duration_seconds: int) -> float:
        return self._cost_lines(duration_seconds)[1]

    def _cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache. 0.0 means caching never hit."""
        total = self._input_text_tokens + self._input_audio_tokens
        if not total:
            return 0.0
        cached = self._input_text_cached_tokens + self._input_audio_cached_tokens
        return cached / total

    def _print_cost(self, duration_seconds: int) -> None:
        rows, total = self._cost_lines(duration_seconds)
        hit = self._cache_hit_rate()

        print("\n" + "─" * 56, flush=True)
        print(f"  💰  CALL COST — {self.call_id}", flush=True)
        print(f"      model={settings.realtime_model}  template={settings.call_template}", flush=True)
        print("─" * 56, flush=True)
        print(f"  {'Duration':<17}{duration_seconds}s ({duration_seconds/60:.1f} min), "
              f"{self._responses} responses", flush=True)
        for label, detail, usd in rows:
            print(f"  {label:<17}{detail:>16}  → ${usd:.4f}", flush=True)
        print("─" * 56, flush=True)
        print(f"  {'TOTAL':<17}{'':>16}  → ${total:.4f}", flush=True)
        if duration_seconds:
            print(f"  {'per minute':<17}{'':>16}  → ${total / (duration_seconds/60):.4f}", flush=True)
        print(f"  {'cache hit rate':<17}{'':>16}    {hit:.1%}", flush=True)
        if hit < 0.20 and self._responses > 2:
            print("  ⚠  Low cache hit rate. Something per-call is leaking into the", flush=True)
            print("     prefix — check that `instructions` is static and that no", flush=True)
            print("     response.create carries an `instructions` override.", flush=True)
        print("─" * 56 + "\n", flush=True)

    async def save(self) -> None:
        import soundfile as sf
        # Recording buffers hold whatever the wire format is: 8kHz μ-law
        # under passthrough, 24kHz PCM16 otherwise.
        _SR = _wire_sample_rate()
        duration    = int((datetime.now() - self.start_dt).total_seconds())
        audio_path: Optional[str] = None

        # Build WAV from accumulated streams
        try:
            base_dir = audio_dir()
            base_dir.mkdir(parents=True, exist_ok=True)
            wav_path = base_dir / f"{self.call_id}.wav"

            # Caller: continuous stream from Twilio — already timeline-aligned (t=0 = stream start)
            caller_raw = b"".join(self._caller_pcm)
            caller = (_wire_to_pcm16(caller_raw)
                      if caller_raw else np.zeros(_SR, dtype=np.float32))

            # Agent: place each response block at its timestamp position.
            # Each block is a complete response — all deltas joined in order, so
            # PCM at 24 kHz naturally spans the correct duration from t_offset.
            n = max(len(caller), int(duration * _SR), _SR)
            agent = np.zeros(n, dtype=np.float32)
            print(f"[Realtime] Recording: caller={len(caller)} samples, "
                  f"agent_responses={len(self._agent_pcm)}, n={n}", flush=True)
            for (t_offset, pcm_bytes) in self._agent_pcm:
                start = int(t_offset * _SR)
                arr   = _wire_to_pcm16(pcm_bytes)
                end   = min(start + len(arr), n)
                if end > start:
                    agent[start:end] += arr[:end - start]
                print(f"  agent block: t={t_offset:.2f}s, dur={_wire_samples(pcm_bytes)/_SR:.2f}s, "
                      f"samples={len(arr)}", flush=True)

            # Pad to same length
            n = max(len(caller), len(agent))
            if len(caller) < n: caller = np.pad(caller, (0, n - len(caller)))
            if len(agent)  < n: agent  = np.pad(agent,  (0, n - len(agent)))

            # Soft-gate the caller channel:
            # During the agent's speaking windows, reduce caller volume to 10% so the echo
            # is barely audible. Outside those windows, keep caller at full volume.
            # This is better than hard-gating (zeroing) which silenced the caller entirely
            # when they responded immediately after the agent stopped speaking.
            soft_gate = np.ones(n, dtype=np.float32)
            for (t_off, pcm_b) in self._agent_pcm:
                s = int(t_off * _SR)
                # 0.3s tail after agent audio ends — covers residual phone echo
                e = min(s + _wire_samples(pcm_b) + int(0.30 * _SR), n)
                if e > s:
                    soft_gate[s:e] = 0.10   # 10% = echo barely audible, speech still recorded

            gated_caller = caller * soft_gate

            # Debug: log raw caller RMS in 3s chunks to see if voice is present
            chunk = 3 * _SR
            raw_rms = [float(np.sqrt(np.mean(caller[i:i+chunk]**2)))
                       for i in range(0, len(caller) - chunk, chunk)]
            print(f"[Realtime] Caller raw RMS per 3s: {[f'{r:.4f}' for r in raw_rms]}", flush=True)

            # Normalise each channel independently then mix to MONO.
            def _normalise(arr: np.ndarray, target: float = 0.7) -> np.ndarray:
                peak = np.max(np.abs(arr))
                return arr * (target / peak) if peak > 0.01 else arr
            agent_norm  = _normalise(agent)
            caller_norm = _normalise(gated_caller)
            mono = np.clip(agent_norm + caller_norm, -1.0, 1.0)
            sf.write(str(wav_path), mono, _SR)
            audio_path = str(wav_path)
            log.info("Realtime recording saved (mono, soft-gated caller): %s", wav_path)
        except Exception as e:
            log.error("Failed to save audio: %s", e, exc_info=True)

        # Save per-call JSON
        data_dir = json_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        branch  = self.memory.get("branch")
        resolved = bool(self.memory.get("resolved"))
        # clean_doctor_name strips "Dr." so we don't get "Dr. Dr. John"
        from agents.voice.templates import clean_doctor_name
        doctor_display = clean_doctor_name(self.doctor.doctor_name)
        summary = (
            f"Called {self.doctor.hospital_name} to verify Dr. {doctor_display}'s branch. "
            + (f"Branch confirmed: {branch}." if resolved else "Branch could not be confirmed.")
        )

        # Clean up transcript:
        # 1. Drop [...]  — placeholder never replaced (too short/quiet to transcribe)
        # 2. Drop short agent fragments (< 4 words, e.g. "Right, right —", "Got it,")
        #    that appear between caller turns — these are filler utterances, not full turns
        # 3. Merge consecutive agent turns within 5s (fragment + real response collapsed into one)
        # 4. Merge consecutive caller turns within 4s (VAD split one sentence into two)
        # 5. Drop duplicate agent turns (same text within 3s — barge-in double-fire)
        merged: list[TranscriptTurn] = []
        for turn in self.turns:
            # Drop untranscribed placeholders
            if turn.text.strip() == "[...]":
                continue

            if merged:
                prev = merged[-1]
                try:
                    prev_t = datetime.strptime(prev.timestamp, "%H:%M:%S")
                    cur_t  = datetime.strptime(turn.timestamp, "%H:%M:%S")
                    gap    = (cur_t - prev_t).total_seconds()
                except ValueError:
                    gap = 99

                # Merge consecutive agent turns within 5s (fragment collapsed into next real turn)
                if prev.role == "agent" == turn.role and gap <= 5:
                    prev_words = len(prev.text.strip().split())
                    if prev_words <= 4:
                        # prev is the fragment — drop it, keep the fuller current turn
                        merged[-1] = turn
                    else:
                        # both are substantial — merge (rare, but handles multi-sentence responses)
                        merged[-1] = TranscriptTurn(
                            role=prev.role,
                            text=prev.text.rstrip() + " " + turn.text.lstrip(),
                            timestamp=prev.timestamp,
                        )
                    continue

                # Merge split caller sentences (same role, within 4s)
                if prev.role == "caller" == turn.role and gap <= 4:
                    merged[-1] = TranscriptTurn(
                        role=prev.role,
                        text=prev.text.rstrip() + " " + turn.text.lstrip(),
                        timestamp=prev.timestamp,
                    )
                    continue

                # Drop duplicate agent turns (same text within 3s)
                if (prev.role == "agent" == turn.role and gap <= 3
                        and prev.text.strip() == turn.text.strip()):
                    continue

            merged.append(turn)

        # Final pass: drop lone agent fragments (≤ 3 words ending in — or ,)
        # e.g. "Right, right —" or "Got it," that weren't adjacent to another agent turn
        def _is_fragment(t: TranscriptTurn) -> bool:
            if t.role != "agent":
                return False
            txt = t.text.strip()
            return len(txt.split()) <= 3 and txt[-1:] in ("—", ",", "–")
        merged = [t for t in merged if not _is_fragment(t)]

        # Calculate cost before saving so it goes into JSON/DB
        cost_usd = self._calc_cost(duration)

        record = {
            "call_id":        self.call_id,
            "doctor_name":    self.doctor.doctor_name,
            "hospital_name":  self.doctor.hospital_name,
            "branch":         branch,
            "resolved":       resolved,
            "duration_seconds": duration,
            "cost_usd":       round(cost_usd, 6),
            "template":       settings.call_template,
            # How much to trust `branch`. "SKIPPED" means the caller's speech
            # never transcribed, so nothing verified the saved location against
            # what they actually said — filter on this before treating a batch
            # of results as clean.
            "grounding":      self.memory.get("grounding"),
            # Countable conversational failures. Prose rules against these have
            # been ignored across three prompt versions; measuring them makes
            # the next edit evaluable instead of impressionistic.
            "conversation":   conversation_metrics(merged),
            # Did the ask-budget directive actually work? It is injected as a
            # conversation item with no follow-up lever, so if the agent
            # acknowledges it and asks again there is nothing else to pull.
            # Recorded so the budget is evaluated, not trusted.
            "ask_budget": _ask_budget_outcome(
                self.turns, self._give_up_at_turn,
                self._give_up_sent, bool(self.memory.get("escalated"))),
            "branch_needed_clarification":
                bool(self.memory.get("branch_needed_clarification")),
            "model":          settings.realtime_model,
            # Recorded so latency across calls can be attributed to the settings
            # that produced it, instead of reconstructed from memory afterwards.
            "audio_settings": {
                "turn_detection": settings.realtime_turn_detection,
                "silence_ms":     settings.realtime_silence_ms,
                "eagerness":      settings.realtime_vad_eagerness,
                "voice":          settings.realtime_voice,
                "noise_reduction": settings.realtime_noise_reduction,
            },
            "usage": {
                "responses":         self._responses,
                "input_audio":       self._input_audio_tokens,
                "input_audio_cached": self._input_audio_cached_tokens,
                "output_audio":      self._output_audio_tokens,
                "input_text":        self._input_text_tokens,
                "input_text_cached": self._input_text_cached_tokens,
                "output_text":       self._output_text_tokens,
                "cache_hit_rate":    round(self._cache_hit_rate(), 4),
            },
            "transcript": [
                {"role": t.role, "text": t.text, "timestamp": t.timestamp}
                for t in merged
            ],
            "summary":     summary,
            "audio_path":  audio_path,
            "recorded_at": self.start_dt.isoformat(),
        }
        json_path = data_dir / f"{self.call_id}.json"
        json_path.write_text(json.dumps(record, indent=2))

        # Update master.json.
        #
        # Read-modify-write with no lock: two calls finishing together both read
        # the same list, both append their own entry, and the second write
        # silently discards the first. No error, no warning — a completed call
        # simply is not in the index.
        #
        # Paired deliberately with the CallSid routing fix in twilio_worker.
        # Fixing only that one would remove the module global that currently
        # makes concurrency impossible, turning this from dormant into live.
        master = data_dir / "master.json"
        with _MASTER_LOCK:
            try:
                existing = json.loads(master.read_text()) if master.exists() else []
            except Exception:
                existing = []
            existing.append({
                "call_id":          self.call_id,
                "time":             self.start_dt.isoformat(),
                "doctor":           self.doctor.doctor_name,
                "hospital":         self.doctor.hospital_name,
                "branch":           branch,
                "resolved":         resolved,
                "grounding":        self.memory.get("grounding"),
                "duration_seconds": duration,
                "cost_usd":         round(cost_usd, 6),
                "summary":          summary,
                "audio_path":       audio_path,
                "json_path":        f"data/3 cases jsons/{self.call_id}.json",
            })
            master.write_text(json.dumps(existing, indent=2))
        log.info("Realtime call saved: %s (resolved=%s branch=%s)", self.call_id, resolved, branch)

        # ── End-of-call summary ────────────────────────────────────────
        _W = 60
        print("\n" + "═" * _W, flush=True)
        print(f"  CALL ENDED  —  {self.call_id}", flush=True)
        print("─" * _W, flush=True)
        print(f"  Doctor   : {self.doctor.doctor_name}", flush=True)
        print(f"  Hospital : {self.doctor.hospital_name}", flush=True)
        print(f"  Duration : {duration}s", flush=True)
        if resolved:
            print(f"  Result   : ✅ RESOLVED — Branch: {branch}", flush=True)
        else:
            reason = self.memory.get("escalate_reason", "unknown")
            print(f"  Result   : ⚠️  NOT RESOLVED — {reason}", flush=True)
        print("─" * _W, flush=True)
        print("  TRANSCRIPT:", flush=True)
        for t in merged:
            role_label = "🤖 AGENT " if t.role == "agent" else "👤 Caller"
            print(f"  [{t.timestamp}] {role_label}: {t.text}", flush=True)
        print("─" * _W, flush=True)
        if audio_path:
            print(f"  Recording: {audio_path}", flush=True)
        print(f"  JSON     : data/3 cases jsons/{self.call_id}.json", flush=True)
        print("═" * _W + "\n", flush=True)

        m = conversation_metrics(merged)
        print(f"  CONVERSATION SHAPE", flush=True)
        print(f"    agent turns          {m['agent_turns']}", flush=True)
        print(f"    of which questions   {m['question_turns']}", flush=True)
        rate = f"{m['staple_rate']:.0%}" if m['staple_rate'] is not None else "n/a"
        print(f"    caller turns         {m['caller_turns']} "
              f"({m['caller_questions']} of them questions)", flush=True)
        print(f"    stapled onto answers {m['stapled_questions']} of "
              f"{m['caller_questions']}  ({rate})", flush=True)
        print(f"    asked twice running  {m['back_to_back_asks']}", flush=True)
        print(f"    repeated sentences   {m['repeated_sentences']}"
              f"{'   <- this is the one that correlates with a bad call' if m['repeated_sentences'] else ''}",
              flush=True)

        self._print_cost(duration)


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_realtime(twilio_ws: WebSocket, call_sid: str, doctor: Doctor) -> None:
    """Bridge Twilio WebSocket ↔ OpenAI Realtime API for a single call."""
    sess     = RealtimeSession(call_sid, doctor)
    template = get_template(settings.call_template)

    # Never let configured settings be silently ignored — someone set them for a
    # reason, and a call going out under the wrong org name or in the wrong
    # language is not recoverable once the callee has heard it.
    for warning in template.config_warnings(agent_language=settings.agent_language,
                                            org_name=settings.org_name):
        log.warning("[Realtime] %s", warning)
        print(f"\n  ⚠  {warning}\n", flush=True)

    # The organisation is a runtime value: it names whichever client's campaign
    # this call belongs to, and it reaches the model through the per-call
    # context item, never through the cached instructions.
    greeting = template.build_greeting(doctor, org=settings.org_name)
    context  = template.build_context(
        doctor,
        callback_number=settings.callback_number,
        callback_email=settings.callback_email,
        org=settings.org_name,
    )

    # Let /recording_ready name the downloaded MP3 after this call_id so audio,
    # JSON and transcript all share one identifier.
    from agents.voice import twilio_worker
    twilio_worker._call_id_by_sid[call_sid] = sess.call_id

    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    model   = settings.realtime_model

    # One model, named in config. The old fallback list meant the cost breakdown
    # could describe a different model than the one that actually served the call.
    conn   = websockets.connect(REALTIME_URL.format(model=model), additional_headers=headers)
    ws_obj = await conn.__aenter__()
    try:
        raw   = await asyncio.wait_for(ws_obj.recv(), timeout=10.0)
        first = json.loads(raw)
        if first.get("type") == "error":
            err = first.get("error", {})
            raise RuntimeError(f"{model} rejected the connection: {err.get('message')}")
        print(f"[Realtime] Connected: {model}", flush=True)
    except Exception:
        # Close the socket we opened — the old code leaked it on the timeout path.
        await conn.__aexit__(None, None, None)
        raise

    oai_ws_ctx = conn

    try:
        oai_ws = ws_obj

        # ── 2. Configure session — ONE message, everything in it ──────
        # Splitting this across two session.update calls churned the cached
        # prefix. `instructions` is the template's STATIC text: no doctor, no
        # hospital, no time of day. Those go in the conversation item at step 4.
        await oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type":         "realtime",
                "instructions": template.instructions,
                "tools":        _realtime_tools(),
                "audio": build_audio_config(
                    transcribe_model=settings.realtime_transcribe_model,
                    transcribe_hint=template.transcribe_hint,
                    audio_format=settings.realtime_audio_format,
                    noise_reduction=settings.realtime_noise_reduction,
                    turn_detection=settings.realtime_turn_detection,
                    eagerness=settings.realtime_vad_eagerness,
                    voice=settings.realtime_voice,
                    silence_ms=settings.realtime_silence_ms,
                ),
                "max_output_tokens": settings.realtime_max_response_tokens,
            },
        }))

        # ── 2b. Drain messages until session.updated or error ─────────
        for _ in range(10):
            session_confirmed = await asyncio.wait_for(oai_ws.recv(), timeout=10.0)
            sc = json.loads(session_confirmed)
            ev = sc.get("type", "")
            if ev == "error":
                err = sc.get("error", {})
                raise RuntimeError(f"session.update rejected: {err.get('code')} {err.get('message')}")
            if ev == "session.updated":
                print(f"[Realtime] Session configured — template={template.name} "
                      f"voice={settings.realtime_voice}", flush=True)
                break
        else:
            print("[Realtime] session.updated not received — continuing anyway", flush=True)

        # ── 3. Wait for Twilio stream to start ────────────────────────
        print("[Realtime] Waiting for Twilio stream start...", flush=True)
        async for raw_msg in twilio_ws.iter_text():
            msg = json.loads(raw_msg)
            if msg.get("event") == "start":
                sess.stream_sid = msg["start"]["streamSid"]
                sess._stream_start_time = datetime.now()
                print(f"[Realtime] Twilio stream started: {sess.stream_sid}", flush=True)
                # Start Twilio recording NOW — audio stream just opened, agent is about to speak.
                # Starting here (not in /answer) skips the ringing/setup gap entirely.
                async def _start_twilio_recording(csid=call_sid):
                    from twilio.rest import Client as TwilioClient
                    tw = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)

                    # Trial accounts reject the richer recording parameters.
                    # Fall back to a bare recording rather than losing it: the
                    # audio is the point, dual-channel and the callback are not.
                    try:
                        rec = await asyncio.to_thread(
                            lambda: tw.calls(csid).recordings.create(
                                recording_channels="dual",
                                trim="trim-silence",
                                recording_status_callback=settings.server_public_url + "/recording_ready",
                                recording_status_callback_method="POST",
                            )
                        )
                        print(f"[Recording] Started (dual channel): {rec.sid}", flush=True)
                        return
                    except Exception as e:
                        print(f"[Recording] Full options rejected ({e}) — "
                              f"retrying bare", flush=True)
                    try:
                        rec = await asyncio.to_thread(
                            lambda: tw.calls(csid).recordings.create()
                        )
                        print(f"[Recording] Started (mono, no callback): {rec.sid}. "
                              f"Fetch it from the Twilio console, or rely on the "
                              f"local mix written at the end of the call.", flush=True)
                    except Exception as e:
                        print(f"[Recording] Could not start: {e}. The local WAV "
                              f"mix will still be written.", flush=True)
                asyncio.create_task(_start_twilio_recording())
                break

        # ── Call start banner ──────────────────────────────────────────
        _W = 60
        print("\n" + "═" * _W, flush=True)
        print(f"  CALL STARTED  {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"  Doctor  : {doctor.doctor_name}", flush=True)
        print(f"  Hospital: {doctor.hospital_name}", flush=True)
        print(f"  Call ID : {sess.call_id}", flush=True)
        print("═" * _W, flush=True)
        print(f"  Greeting → {greeting}", flush=True)
        print("─" * _W + "\n", flush=True)

        # ── 4. Send per-call context, then ask for the opening line ───
        # The context item carries the doctor, hospital and the exact greeting.
        # It lands AFTER the cached instructions prefix, so varying it between
        # calls costs ~110 tokens instead of re-billing the whole prompt.
        # response.create deliberately carries NO `instructions` override — an
        # override replaces the session instructions for that response and puts
        # it on a different, uncacheable prefix.
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": context}],
            },
        }))
        await oai_ws.send(json.dumps({"type": "response.create"}))
        # First-audio latency is the dead air the callee hears after picking up.
        # Measured separately from mid-call latency because the first response
        # pays for an uncached prompt and any connection warm-up.
        sess._greeting_requested_at = time.monotonic()
        print("[Realtime] Context sent, greeting requested — starting audio loops", flush=True)

        # ── 5. Run both directions concurrently ───────────────────────
        # First leg to finish ends the call; the other is cancelled explicitly.
        # asyncio.gather() would leave the surviving leg orphaned, and the
        # Twilio leg only notices done_event when a new frame arrives — so if
        # the far end goes quiet it can hang and sess.save() never runs.
        done_event = asyncio.Event()
        legs = [
            asyncio.create_task(_twilio_to_oai(twilio_ws, oai_ws, sess, done_event),
                                name="twilio->oai"),
            asyncio.create_task(_oai_to_twilio(oai_ws, twilio_ws, sess, done_event),
                                name="oai->twilio"),
        ]
        try:
            _finished, pending = await asyncio.wait(
                legs, return_when=asyncio.FIRST_COMPLETED
            )
            done_event.set()
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=5.0)
        finally:
            for task in legs:
                if not task.done():
                    task.cancel()
        for task in legs:
            if task.done() and not task.cancelled() and task.exception():
                log.error("[Realtime] %s leg failed: %s", task.get_name(), task.exception())

    except websockets.exceptions.ConnectionClosed as e:
        log.info("[Realtime] OAI WebSocket closed: %s", e)
    except Exception as e:
        log.error("[Realtime] handle_realtime error: %s", e, exc_info=True)
    finally:
        try:
            await oai_ws_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        await sess.save()


# ── Twilio → OpenAI ───────────────────────────────────────────────────────────

async def _twilio_to_oai(
    twilio_ws: WebSocket,
    oai_ws,
    sess: RealtimeSession,
    done_event: asyncio.Event,
) -> None:
    """Forward Twilio inbound audio to OpenAI Realtime input buffer."""
    try:
        async for raw in twilio_ws.iter_text():
            if done_event.is_set():
                break
            msg   = json.loads(raw)
            event = msg.get("event", "")

            if event == "media":
                if msg["media"].get("track") != "inbound":
                    continue
                payload = msg["media"]["payload"]
                try:
                    if _passthrough_enabled():
                        # Twilio already speaks the session's format. Store the
                        # μ-law bytes as-is and forward the payload untouched.
                        raw_bytes = base64.b64decode(payload)
                        sess._caller_pcm.append(raw_bytes)
                        oai_payload = payload
                    else:
                        raw_bytes = base64.b64decode(payload)
                        pcm_24k = (resample(_mulaw_decode(raw_bytes), _TWILIO_SR, _OAI_SR) * 32767).astype(np.int16)
                        sess._caller_pcm.append(pcm_24k.tobytes())
                        oai_payload = base64.b64encode(pcm_24k.tobytes()).decode()
                    if not sess.listen_enabled.is_set():
                        continue
                    if sess.agent_speaking and not _echo_gate_allows(raw_bytes):
                        # Only reached under REALTIME_ECHO_GATE=drop|energy.
                        #
                        # Dropping every frame here made the agent
                        # uninterruptible. OpenAI's VAD can only fire on audio
                        # it receives, so with the gate shut
                        # input_audio_buffer.speech_started never arrives and
                        # the barge-in handler below is unreachable: a
                        # receptionist who starts talking gets talked over
                        # until the agent finishes its turn. That is the most
                        # robotic thing a voice agent can do, and no prompt
                        # wording fixes it.
                        #
                        # Default is now "pass". near_field noise reduction and
                        # semantic_vad both post-date this gate and may handle
                        # line echo on their own — an empirical question one
                        # call answers.
                        continue
                    await oai_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": oai_payload,
                    }))
                except Exception as e:
                    log.debug("[Realtime] Twilio→OAI audio error: %s", e)

            elif event == "stop":
                log.info("[Realtime] Twilio stream stopped")
                break

    except Exception as e:
        log.info("[Realtime] Twilio→OAI loop ended: %s", e)


# ── OpenAI → Twilio ───────────────────────────────────────────────────────────

async def _oai_to_twilio(
    oai_ws,
    twilio_ws: WebSocket,
    sess: RealtimeSession,
    done_event: asyncio.Event,
) -> None:
    """Forward OpenAI Realtime events to Twilio + handle tool calls."""
    _pending_tools: dict[str, dict] = {}
    _agent_text_buf       = ""
    _response_active      = False   # True while model is generating audio
    _response_had_audio   = False   # True if current response included any audio (model spoke)
    _barge_in_pending     = False   # True when we cancelled a response — skip its transcript
    _closing_sent         = False   # True after we send closing response.create — wait for its response.done
    _closing_retries      = 0       # a goodbye the caller talked over is not a goodbye
    # Tool results arrive on response.function_call_arguments.done, which fires
    # BEFORE response.done for the same response. Creating a response there
    # raises conversation_already_has_active_response, so defer it to response.done.
    _pending_response_create = False
    _caller_speaking       = False
    _speech_start_pcm_pos  = 0       # position in sess._caller_pcm when caller speech started
    _samples_this_response = 0       # PCM16 samples sent this response — used for dynamic echo cooldown
    _current_response_pcm: list[bytes] = []    # accumulate all deltas for one response
    _current_response_start: Optional[float] = None  # stream-relative time of first delta
    # monotonic clock when this response's first audio chunk went to Twilio;
    # playback ends at this + audio duration, which is what the echo gate needs
    _first_delta_sent_at: Optional[float] = None
    # id of the assistant item currently being spoken — needed to truncate it
    # to what the caller actually heard when they interrupt
    _current_item_id: Optional[str] = None
    # response ids already accounted for, so a repeated response.done cannot
    # double-count its tokens into the cost figure
    _counted_responses: set[str] = set()

    try:
        async for raw in oai_ws:
            msg        = json.loads(raw)
            event_type = msg.get("type", "")

            # ── Caller barge-in: cancel current response immediately ───────
            if event_type == "input_audio_buffer.speech_started":
                _caller_speaking = True
                _speech_start_pcm_pos = len(sess._caller_pcm)  # mark start of this utterance
                if sess.done:
                    continue  # don't interrupt the closing farewell
                if _response_active and not _barge_in_pending:
                    # Only cancel once per active response — prevents inflation
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✋ BARGE-IN  : caller interrupted agent", flush=True)
                    _barge_in_pending = True
                    _response_active  = False
                    sess.agent_speaking = False
                    # Flush whatever agent audio arrived before the cancel
                    if _current_response_pcm and _current_response_start is not None:
                        sess._agent_pcm.append((_current_response_start, b"".join(_current_response_pcm)))
                    _current_response_pcm.clear()
                    _current_response_start = None
                    try:
                        await oai_ws.send(json.dumps({"type": "response.cancel"}))
                        await twilio_ws.send_text(json.dumps({
                            "event": "clear", "streamSid": sess.stream_sid,
                        }))
                        # Cancelling stops generation but leaves the FULL
                        # response in OpenAI's conversation history, while the
                        # caller only heard the part that had played. The model
                        # then reasons as though it said things nobody heard —
                        # it won't repeat them, and may refer back to them
                        # ("as I mentioned") about words never spoken.
                        #
                        # Truncate the item to what actually reached the ear.
                        # Twilio plays what we send at realtime speed, so
                        # elapsed-since-first-chunk is the audio they heard,
                        # capped at what was generated.
                        if _current_item_id and _first_delta_sent_at is not None:
                            heard_ms = int((time.monotonic() - _first_delta_sent_at) * 1000)
                            generated_ms = int(_samples_this_response / _wire_sample_rate() * 1000)
                            audio_end_ms = max(0, min(heard_ms, generated_ms))
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.truncate",
                                "item_id": _current_item_id,
                                "content_index": 0,
                                "audio_end_ms": audio_end_ms,
                            }))
                            print(f"[Realtime] Truncated to {audio_end_ms}ms — "
                                  f"the model's context now matches what was heard",
                                  flush=True)
                    except Exception:
                        pass

            # ── Caller finished speaking ───────────────────────────────────
            elif event_type == "input_audio_buffer.speech_stopped":
                if _caller_speaking and sess.listen_enabled.is_set():
                    _caller_speaking = False
                    # Placeholder — filled in by the session's own inline
                    # transcription when conversation.item.input_audio_
                    # transcription.completed arrives. Nothing else fills it:
                    # the out-of-band whisper-1 HTTP fallback was removed to
                    # keep this path pure speech-to-speech. Placeholders that
                    # never resolve are dropped from the transcript in save().
                    sess.add_turn("caller", "[...]")

                    # Tell the model when the line is genuinely too quiet to
                    # trust. Left to its own judgement it does not report
                    # difficulty — on a live call a caller at roughly a tenth
                    # of normal phone level produced a confident fabrication
                    # rather than "sorry, I didn't catch that". Measured level
                    # is evidence the model does not otherwise have.
                    # If ANY caller turn has transcribed cleanly, the line is
                    # audible by definition — never tell them otherwise.
                    #
                    # This alarm has now fired falsely on three consecutive
                    # calls, twice in one of them, telling a caller measuring
                    # 0.0335 RMS that they were "coming through really faint".
                    # Measuring the loudest window instead of the mean was not
                    # enough: server VAD sometimes fires speech_started on
                    # noise, and the resulting slice contains no speech at all,
                    # so any energy measure of it reads as silence.
                    #
                    # Working transcription is the evidence that matters.
                    heard_clearly = any(
                        t.role == "caller" and t.text.strip() not in ("", "[...]")
                        for t in sess.turns)
                    utterance = b"".join(sess._caller_pcm[_speech_start_pcm_pos:])
                    if utterance and not sess._low_audio_warned and not heard_clearly:
                        arr = _wire_to_pcm16(utterance)
                        rms = _loudest_window_rms(arr)
                        if 0.0 < rms < _LOW_AUDIO_RMS:
                            sess._low_audio_warned = True
                            print(f"[Realtime] Caller audio very faint "
                                  f"(RMS {rms:.4f} vs ~0.03 typical) — telling "
                                  f"the agent to ask them to speak up", flush=True)
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": ("(system: the caller's line is very "
                                                 "faint and hard to make out. Ask them "
                                                 "to speak up or repeat. Do not guess "
                                                 "at anything you did not clearly hear.)"),
                                    }],
                                },
                            }))

            # ── Audio → Twilio ─────────────────────────────────────────────
            # gpt-realtime-2 uses response.output_audio.delta (not response.audio.delta)
            elif event_type == "response.output_audio.delta":
                delta = msg.get("delta", "")
                if delta:
                    sess.agent_speaking  = True
                    _response_active     = True
                    _response_had_audio  = True
                if delta and sess.stream_sid:
                    try:
                        raw_pcm = base64.b64decode(delta)
                        _samples_this_response += _wire_samples(raw_pcm)
                        if _first_delta_sent_at is None:
                            _first_delta_sent_at = time.monotonic()
                        _current_item_id = msg.get("item_id") or _current_item_id
                        # Buffer for recording: stamp only the first delta of this response.
                        # All deltas arrive fast (~0.2s for a 2s response) so we must NOT
                        # timestamp each chunk individually — they'd all pile up at the same
                        # position and overlap into distorted audio.  Instead we collect them
                        # and flush the whole block on response.done.
                        _current_response_pcm.append(raw_pcm)
                        if _current_response_start is None and sess._stream_start_time:
                            _current_response_start = (datetime.now() - sess._stream_start_time).total_seconds()
                        # Dead air the callee hears before the agent speaks.
                        if sess._greeting_requested_at is not None:
                            gap = time.monotonic() - sess._greeting_requested_at
                            sess._greeting_requested_at = None
                            print(f"[Realtime] First audio {gap:.2f}s after "
                                  f"response.create", flush=True)
                            if gap > 2.0:
                                print(f"[Realtime]   ^ that is dead air on the "
                                      f"callee's end before the greeting starts",
                                      flush=True)
                        twilio_payload = (delta if _passthrough_enabled()
                                          else _convert_oai_to_twilio(delta))
                        await twilio_ws.send_text(json.dumps({
                            "event":    "media",
                            "streamSid": sess.stream_sid,
                            "media":    {"payload": twilio_payload},
                        }))
                    except Exception as e:
                        log.error("[Realtime] audio send error: %s", e)
                elif delta and not sess.stream_sid:
                    log.warning("[Realtime] Audio delta received but stream_sid empty — dropped")

            # ── Agent transcript ───────────────────────────────────────────
            elif event_type == "response.output_audio_transcript.delta":
                _agent_text_buf += msg.get("delta", "")

            elif event_type == "response.output_audio_transcript.done":
                if _barge_in_pending:
                    # This transcript was cancelled — never fully heard, skip it
                    _barge_in_pending = False
                    _agent_text_buf = ""
                    continue
                text = (msg.get("transcript") or _agent_text_buf).strip()
                if text:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{ts}] 🤖 AGENT  : {text}", flush=True)
                    sess.add_turn("agent", text)

                    # Enforce the exit condition the prompt never had.
                    #
                    # A live call asked for the location six times in 111
                    # seconds. The caller engaged throughout but never refused,
                    # never said they did not know, was not a wrong number and
                    # was not voicemail — so none of the prompt's escalation
                    # triggers matched, and "never close until you have saved a
                    # location or escalated" left asking again as the only move.
                    # The repetition was a symptom of having no way out, not a
                    # phrasing failure, and no wording of a phrasing rule fixes
                    # it. A budget does.
                    if _is_location_ask(text) and not sess.done:
                        sess._location_asks += 1
                        if (sess._location_asks >= settings.realtime_max_location_asks
                                and not sess._give_up_sent):
                            sess._give_up_sent = True
                            sess._give_up_at_turn = len(sess.turns)
                            print(f"[Realtime] {sess._location_asks} asks with no "
                                  f"location — telling the agent to stop and "
                                  f"escalate", flush=True)
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": (
                                            f"(system: you have now asked for the "
                                            f"location {sess._location_asks} times "
                                            f"and have not been given one. Stop "
                                            f"asking. Thank them briefly, say "
                                            f"goodbye, and call escalate with "
                                            f"reason 'caller engaged but never "
                                            f"provided a location'.)"
                                        ),
                                    }],
                                },
                            }))
                _agent_text_buf = ""

            # ── Caller transcript — replace placeholder if transcription enabled ──
            elif event_type == "conversation.item.input_audio_transcription.completed":
                text = msg.get("transcript", "").strip()
                if text:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] 👤 CALLER : {text}", flush=True)
                    # Replace the most recent "[...]" placeholder with real text
                    for i in range(len(sess.turns) - 1, -1, -1):
                        if sess.turns[i].role == "caller" and sess.turns[i].text == "[...]":
                            sess.turns[i] = TranscriptTurn(
                                role="caller", text=text,
                                timestamp=sess.turns[i].timestamp,
                            )
                            break
                    else:
                        sess.add_turn("caller", text)

            # ── Tool call arguments streaming ──────────────────────────────
            elif event_type == "response.function_call_arguments.delta":
                call_id = msg.get("call_id", "")
                name    = msg.get("name", "")
                if call_id not in _pending_tools:
                    _pending_tools[call_id] = {"name": name, "args": ""}
                _pending_tools[call_id]["args"] += msg.get("delta", "")

            # ── Tool call complete ─────────────────────────────────────────
            elif event_type == "response.function_call_arguments.done":
                call_id  = msg.get("call_id", "")
                name     = msg.get("name", "")
                args_str = msg.get("arguments") or _pending_tools.get(call_id, {}).get("args", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                # Grounding check. On a live call the model called save_branch
                # with {'branch': 'Riverside Clinic', 'city': 'Atlanta'} when
                # the caller had said only "Hello" and "Okay, next slide,
                # please". "Riverside Campus" was an EXAMPLE in the prompt; the
                # model reshaped it into a fabricated result and hung up.
                # Nothing downstream could tell that record from a real one.
                #
                # So a location may only be saved if the caller actually said
                # it. Verified against the transcript, not the model's claim.
                if name == "save_branch":
                    # The check switches itself off when nothing was
                    # transcribed — correct, since absence of transcript is not
                    # evidence of fabrication and blocking would kill genuine
                    # saves on a bad line. But that is exactly the condition
                    # that produces fabrications: bad line -> no transcript ->
                    # guard off -> a location the model may have inferred gets
                    # written as fact. And with the out-of-band whisper
                    # fallback removed there is no second path to a transcript.
                    #
                    # So record it. A save that could not be verified must not
                    # be indistinguishable downstream from one that was.
                    heard_any = any(t.role == "caller" and t.text.strip() != "[...]"
                                    for t in sess.turns)
                    sess.memory.update(
                        grounding="verified against caller transcript" if heard_any
                        else "SKIPPED — no caller speech was transcribed on this "
                             "call, so the saved location could not be checked "
                             "against anything the caller actually said"
                    )
                    ungrounded = _ungrounded_terms(args, sess)
                    if ungrounded:
                        result = {
                            "ok": False,
                            "error": (
                                "REJECTED — that location does not appear "
                                "anywhere in what the caller said on this "
                                "call. Never save a location you were not "
                                "told. Ask them for it, and if they do not "
                                "give one, escalate."
                            ),
                        }
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                              f"🚫 HALLUCINATED BRANCH BLOCKED: {args}", flush=True)
                    else:
                        result = run_tool(name, sess.memory, args)
                elif name == "escalate":
                    bad = _ungrounded_escalation(args.get("reason", ""), sess)
                    if bad:
                        result = {"ok": False, "error": (
                            f"REJECTED — {bad}. Escalate with what actually "
                            f"happened on the call, not an inference about the "
                            f"doctor. If you are unsure why, say 'could not "
                            f"obtain the location'.")}
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                              f"🚫 UNGROUNDED ESCALATION BLOCKED: {args}",
                              flush=True)
                    else:
                        result = run_tool(name, sess.memory, args)
                else:
                    result = run_tool(name, sess.memory, args)
                # Report what the tool ACTUALLY did. This used to print
                # "✅ BRANCH SAVED" unconditionally, without looking at the
                # result — so a live call logged
                #     🚫 HALLUCINATED BRANCH BLOCKED: {'branch': 'Downtown'}
                #     ✅ BRANCH SAVED : {'branch': 'Downtown'}
                # one line apart. The guard had worked and nothing was saved,
                # but the log said otherwise. A safeguard that reports itself as
                # having failed is worse than no log at all: it sends you
                # hunting a bug that isn't there and hides the one that is.
                ts = datetime.now().strftime("%H:%M:%S")
                ok = bool(result.get("ok"))
                if name == "save_branch":
                    if ok:
                        print(f"\n[{ts}] ✅ BRANCH SAVED   : {args}", flush=True)
                    else:
                        print(f"\n[{ts}] ⛔ BRANCH REJECTED: {args}", flush=True)
                        print(f"          reason: {result.get('error', '')}", flush=True)
                elif name == "escalate":
                    label = "⚠️  ESCALATED     " if ok else "⛔ ESCALATE FAILED"
                    print(f"\n[{ts}] {label}: {args}", flush=True)
                elif name == "note_info":
                    print(f"[{ts}] {'📝 NOTE           ' if ok else '⛔ NOTE REJECTED  '}: {args}",
                          flush=True)
                else:
                    print(f"[{ts}] 🔧 TOOL           : {name}({args}) → {result}", flush=True)

                if name in ("save_branch", "escalate") and result.get("ok"):
                    sess.done = True

                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type":    "function_call_output",
                        "call_id": call_id,
                        "output":  json.dumps(result),
                    },
                }))

                if sess.done:
                    # "_response_had_audio" was being read as "the agent said
                    # goodbye", so the call hung up on whatever it happened to
                    # be saying. On a live call it asked "which office is Dr.
                    # Okafor working out of?", called save_branch in the same
                    # response, and hung up — leaving the caller answering a
                    # question to a dead line.
                    #
                    # An utterance ending in a question mark is not a farewell.
                    last_agent = next((t.text for t in reversed(sess.turns)
                                       if t.role == "agent"), "")
                    sounded_like_a_goodbye = bool(last_agent) and not last_agent.rstrip().endswith("?")

                    if _response_had_audio and sounded_like_a_goodbye:
                        # Model already said goodbye in its audio — don't inject another line
                        # The current response.done will trigger the close
                        _closing_sent = False
                    else:
                        # Tool fired with no spoken goodbye. Ask for one via a
                        # conversation item rather than a per-response
                        # `instructions` override — an override swaps out the
                        # session instructions and lands this response on a
                        # different, uncacheable prefix.
                        await oai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": "(say a brief warm goodbye now, then stop)",
                                }],
                            },
                        }))
                        await oai_ws.send(json.dumps({"type": "response.create"}))
                        _closing_sent = True  # skip tool-call response.done, close on closing's
                else:
                    _pending_response_create = True

            # ── Response done: extract token usage + check resolution ────
            elif event_type == "response.done":
                _response_active    = False
                _response_had_audio = False   # reset for next response
                sess._responses    += 1
                # "completed" | "cancelled" | "incomplete" | "failed". This was
                # never read, so a closing response the caller talked over was
                # indistinguishable from one that actually played, and the call
                # hung up on a goodbye nobody heard.
                _resp_status = ((msg.get("response") or {}).get("status")
                                or "completed")
                # A cancelled response may never emit transcript.done. Clearing
                # the flag only there meant it leaked into the NEXT response and
                # silently swallowed a real transcript line.
                _barge_in_pending   = False
                # Flush buffered agent audio as one contiguous block.
                # Placing it all at _current_response_start means the PCM runs at the correct
                # sample rate (24 kHz) from that point — no overlap, no gaps.
                if _current_response_pcm and _current_response_start is not None:
                    sess._agent_pcm.append((_current_response_start, b"".join(_current_response_pcm)))
                    print(f"[Realtime] Flushed agent response: {len(_current_response_pcm)} chunks, "
                          f"start={_current_response_start:.2f}s, "
                          f"dur={_samples_this_response/_wire_sample_rate():.2f}s", flush=True)
                _current_response_pcm.clear()
                _current_response_start = None
                # Dynamic echo cooldown: wait until audio finishes playing on the phone +
                # echo travel time.  response.done fires when the SERVER finishes generating
                # (fast), but the audio is still playing on the handset.  Using a fixed 0.5s
                # caused the agent to hear its own echo and generate a duplicate response.
                # Formula: playback_duration + 0.65s echo margin (min 0.5s for very short clips).
                # Wait until the audio has finished PLAYING on the handset,
                # then a small margin — and no longer, because caller audio is
                # dropped for this whole window.
                #
                # The old formula measured the wait from response.done, which
                # fires when the SERVER finishes generating. Generation runs
                # faster than realtime, so response.done lands well before
                # playback ends, and adding the full clip duration on top of it
                # over-waited by roughly the generation time — about 2s of
                # deafness added to every single turn, directly inflating the
                # measured 2.5-4s response latency.
                #
                # Playback ends at (first chunk sent) + (audio duration), since
                # Twilio plays what we send at realtime speed.
                _audio_seconds = _samples_this_response / _wire_sample_rate()
                if _first_delta_sent_at is not None:
                    _playback_ends_at = _first_delta_sent_at + _audio_seconds
                    _echo_cooldown = max(0.3, _playback_ends_at + 0.25 - time.monotonic())
                else:
                    _echo_cooldown = max(0.3, _audio_seconds + 0.25)
                _first_delta_sent_at = None
                _current_item_id = None
                _samples_this_response = 0
                async def _end_speaking_gate(s=sess, delay=_echo_cooldown):
                    await asyncio.sleep(delay)
                    s.agent_speaking = False
                    # Under REALTIME_ECHO_GATE=pass this window gates nothing —
                    # frames flow throughout — so announcing it as "now
                    # listening" was misleading output implying the caller had
                    # been unheard for 6.91s when they had not.
                    if settings.realtime_echo_gate != "pass":
                        print(f"[Realtime] Echo cooldown done ({delay:.2f}s) — "
                              f"listening for caller", flush=True)
                asyncio.create_task(_end_speaking_gate())
                # Account each response's tokens ONCE. A live call logged the
                # same usage line twice, identical to the token
                # (in_text=4572 cached=4416 in_audio=372 out_audio=108), and
                # counted 6 responses against 4 audio blocks. Every duplicate
                # inflates the cost figure — the one number this project has
                # been trying to get honest.
                _resp_id = msg.get("response", {}).get("id")
                if _resp_id and _resp_id in _counted_responses:
                    log.debug("[Realtime] duplicate response.done for %s — "
                              "usage already counted", _resp_id)
                    usage = {}
                else:
                    if _resp_id:
                        _counted_responses.add(_resp_id)
                    usage = msg.get("response", {}).get("usage", {})
                if usage:
                    details_in  = usage.get("input_token_details",  {})
                    details_out = usage.get("output_token_details", {})
                    sess._input_audio_tokens  += details_in.get("audio_tokens",  0)
                    sess._input_text_tokens   += details_in.get("text_tokens",   0)
                    sess._output_audio_tokens += details_out.get("audio_tokens", 0)
                    sess._output_text_tokens  += details_out.get("text_tokens",  0)
                    # Cached tokens — the only direct evidence that the prompt
                    # cache is engaging. Shape varies by API version: a flat
                    # `cached_tokens` plus an optional per-modality breakdown.
                    cached = details_in.get("cached_tokens_details") or {}
                    c_audio = cached.get("audio_tokens", 0)
                    c_text  = cached.get("text_tokens",  0)
                    if not (c_audio or c_text):
                        # No breakdown available — attribute the flat total to
                        # text, which is where the static prompt prefix lives.
                        c_text = details_in.get("cached_tokens", 0)
                    sess._input_audio_cached_tokens += c_audio
                    sess._input_text_cached_tokens  += c_text
                    print(f"[Realtime] usage: in_text={details_in.get('text_tokens', 0)} "
                          f"(cached {c_text})  in_audio={details_in.get('audio_tokens', 0)} "
                          f"(cached {c_audio})  out_audio={details_out.get('audio_tokens', 0)}",
                          flush=True)
                # Enable caller audio forwarding after first response (greeting) finishes
                if not sess.listen_enabled.is_set():
                    print("[Realtime] Greeting done — now listening to caller", flush=True)
                    sess.listen_enabled.set()
                # Deferred response.create from a tool result — safe now that the
                # previous response has completed.
                if _pending_response_create and not sess.done:
                    _pending_response_create = False
                    await oai_ws.send(json.dumps({"type": "response.create"}))
                if sess.done:
                    if _closing_sent:
                        # This is the tool-call response.done — closing response is being generated, wait for it
                        _closing_sent = False
                    elif _resp_status != "completed" and _closing_retries < 1:
                        # The goodbye was cancelled — the caller was still
                        # talking, so barge-in killed it. Hanging up here is
                        # what drops the line in silence. The goodbye item is
                        # still in the conversation; ask for it once more, after
                        # a beat so we are not talking over them again.
                        _closing_retries += 1
                        print(f"[Realtime] Closing response was {_resp_status} — "
                              f"caller talked over it. Retrying the goodbye once.",
                              flush=True)
                        await asyncio.sleep(0.8)
                        await oai_ws.send(json.dumps({"type": "response.create"}))
                    else:
                        # This is the closing response.done.
                        # Wait for the FULL audio to finish playing on the caller's phone before hanging up.
                        # _echo_cooldown = audio_duration + 0.65s, computed just above from _samples_this_response.
                        # Sleeping only 1s was cutting off the goodbye mid-sentence.
                        hangup_wait = max(_echo_cooldown, 1.5)
                        print(f"[Realtime] Closing done — waiting {hangup_wait:.1f}s for audio to finish playing", flush=True)
                        await asyncio.sleep(hangup_wait)
                        print("[Realtime] Hanging up now", flush=True)
                        done_event.set()
                        try:
                            await twilio_ws.close()
                        except Exception:
                            pass
                        break

            elif event_type == "error":
                err  = msg.get("error", {})
                code = err.get("code", "")
                msg_text = err.get("message", "")
                # Suppress harmless errors
                if code == "response_cancel_not_active":
                    pass
                elif "input_audio_transcription" in msg_text or "unknown_parameter" in code:
                    print(f"[Realtime] Transcription not supported on this model — caller turns will show as '[...]'", flush=True)
                else:
                    log.error("[Realtime] API error: %s %s", code, msg_text)

    except websockets.exceptions.ConnectionClosed:
        log.info("[Realtime] OAI WebSocket closed normally")
    except Exception as e:
        # This used to log at INFO with no traceback, so a bug in the event loop
        # silently ended the call and looked, from the caller's side, exactly
        # like a dropped line. A NameError here cost 12 test failures that all
        # pointed somewhere else.
        log.exception("[Realtime] OAI→Twilio loop CRASHED: %s", e)
        print(f"\n[Realtime] ❌ EVENT LOOP CRASHED — the call was cut short.\n"
              f"           {type(e).__name__}: {e}\n"
              f"{traceback.format_exc()}", flush=True)
