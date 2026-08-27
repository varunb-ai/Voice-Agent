"""Layer 3 — Speech to Text.

Primary:  Groq Whisper API (whisper-large-v3-turbo) — cloud GPU, far more accurate on phone audio.
Fallback: faster-whisper local model — used only if Groq API fails.

    audio (np array, 16kHz) ──▶ Groq Whisper API ──▶ text
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Union, cast, Any

import numpy as np

from core.config import settings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_COMPUTE = "int8"

_MEDICAL_PROMPT = (
    # Geographic locations only — NO hospital brand names.
    # Hospital names (Apollo, Wockhardt, KIMS, Yashoda etc.) cause Whisper to hallucinate them
    # into unclear speech (e.g. "I can't share" → "Wockhardt KIMS CARE Yashoda").
    # Callers say location names for branches — those are what we need Whisper to get right.
    "Jubilee Hills Banjara Hills Gachibowli Secunderabad Kondapur Madhapur Kukatpally Miyapur "
    "Hitech City Manikonda Begumpet Somajiguda Ameerpet Dilsukhnagar Uppal LB Nagar "
    "Andheri Bandra Powai Navi Mumbai Thane Malad Borivali Dadar Kurla Chembur "
    "Koramangala Whitefield Indiranagar Jayanagar Banashankari Electronic City HSR Layout "
    "Anna Nagar T Nagar Adyar Velachery Perambur Porur Vadapalani "
    "Noida Gurugram Dwarka Rohini Janakpuri Lajpat Nagar Saket "
    "North South East West Main Campus"
)


# ── OpenAI Whisper API (primary) ──────────────────────────────────────────────

def _transcribe_openai(audio_16k: np.ndarray, context: str = "") -> str:
    """Send audio to OpenAI Whisper API — whisper-1."""
    import soundfile as sf
    from openai import OpenAI

    audio_16k = _phone_filter(audio_16k)

    buf = io.BytesIO()
    sf.write(buf, audio_16k, 16000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    prompt = (context + " " + _MEDICAL_PROMPT).strip() if context else _MEDICAL_PROMPT

    client = OpenAI(api_key=settings.openai_api_key)
    result = client.audio.transcriptions.create(
        file=("audio.wav", buf),
        model="whisper-1",
        language="en",
        response_format="text",
        prompt=prompt,
    )
    return (result or "").strip()


# ── Local Whisper (fallback) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _tiny_model():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="auto", compute_type=_COMPUTE)


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(settings.whisper_model, device="auto", compute_type=_COMPUTE)


def _phone_filter(audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt
        low  = 300  / (sr / 2)
        high = 3400 / (sr / 2)
        sos = butter(4, [low, high], btype="band", output="sos")
        # sosfilt returns a tuple only when zi is passed; it is not.
        return cast(Any, sosfilt(sos, audio)).astype(np.float32)
    except ImportError:
        return audio


def _run_local(model, audio: np.ndarray, language: str) -> str:
    audio = _phone_filter(audio)
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=2,
        vad_filter=True,
        temperature=0.0,
        initial_prompt=_MEDICAL_PROMPT,  # location names only — no hospital brands
        condition_on_previous_text=False,
    )
    parts = []
    for s in segments:
        if getattr(s, "no_speech_prob", 0.0) > 0.50:
            continue
        parts.append(s.text.strip())
    return " ".join(parts).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_array(audio_16k: np.ndarray, language: str = "en", context: str = "") -> str:
    """Transcribe a float32 numpy array at 16 kHz.
    Primary: local faster-whisper (~150ms, zero network). Fallback: OpenAI whisper-1 API."""
    import logging
    log = logging.getLogger(__name__)

    peak = np.abs(audio_16k).max()
    if peak < 0.005:
        return ""
    if peak < 0.20:
        audio_16k = audio_16k * (0.6 / max(peak, 1e-6))

    if len(audio_16k) < 9_600:
        log.info("STT skipped — audio too short (%d samples)", len(audio_16k))
        return ""

    # Primary: local faster-whisper — ~150ms, no network round-trip
    try:
        result = _run_local(_model(), audio_16k, language)
        log.info("STT [local]: %r", result)
        if result and not _looks_garbled(result):
            return result
        log.info("STT [local] result looks garbled — falling back to OpenAI")
    except Exception as e:
        log.warning("STT [local] failed (%s) — falling back to OpenAI Whisper", e)

    # Fallback: OpenAI whisper-1 API — more accurate but adds ~700ms network latency
    if settings.openai_api_key:
        try:
            result = _transcribe_openai(audio_16k, context=context)
            log.info("STT [OpenAI]: %r", result)
            if result and not _looks_garbled(result):
                return result
        except Exception as e:
            log.warning("STT [OpenAI] also failed: %s", e)

    return ""


def transcribe(audio: Union[str, bytes], language: str = "en") -> str:
    """Single-pass transcription from a file path (used in tests)."""
    # whisper accepts a path or bytes here, not only an ndarray.
    result = _run_local(_model(), cast(Any, audio), language)
    return result if not _looks_garbled(result) else ""


# ── Hallucination filter ──────────────────────────────────────────────────────

_HALLUCINATION_FRAGMENTS = (
    # Prompt echo — Whisper outputs the initial_prompt when audio is unclear
    "doctor name branch", "branch location city", "hospital branch location",
    "branch city area", "doctor name branch city",
    "doctor name.", "doctor name ",  # short prompt echo
    # Generic Whisper noise hallucinations
    "the area", "right side", "located on", "thank you for watching",
    "please subscribe", "www.", ".com", "subtitles by", "transcribed by",
    "this video", "check out", "in this video",
    # Twilio trial-account automated announcement fragments
    "trial account", "free trial", "upgrade your account",
    "to remove this message", "press any key", "do it again",
    "going to have to", "going to do", "going to get a ring",
    "roll roll", "ring with a roll",
    # Common phone-line silence noise hallucinations
    "thank you for your support", "thank you very much for",
    "right all the people", "by the government",
    "in the corner", "you are in the corner",
    "currently we are going", "we are going to get",
    "and now, the", "hello and you know you need",
    "you are, you are, you are",  # VAD-loop artifacts
)


# Common English words that should appear in any real phone conversation.
# If a multi-word transcription has NONE of these, it's likely garbled audio/noise.
_COMMON_ENGLISH = frozenset({
    "the", "a", "an", "is", "at", "in", "on", "to", "of", "and", "or", "for",
    "i", "you", "it", "we", "he", "she", "they", "my", "your", "our", "his", "her",
    "yes", "no", "ok", "okay", "hi", "hello", "thank", "thanks", "please", "sorry",
    "doctor", "dr", "branch", "hospital", "working", "located", "here", "there",
    "this", "that", "not", "can", "will", "would", "have", "has", "had", "been",
    "don't", "doesn't", "isn't", "with", "from", "by", "as", "its", "are", "was",
    "one", "two", "three", "just", "right", "well", "so", "but", "if", "about",
    "know", "think", "work", "works", "see", "need", "call", "let", "hold", "time",
    "good", "day", "morning", "afternoon", "evening", "sir", "ma'am", "speak",
})


def _looks_garbled(text: str) -> bool:
    if not text:
        return True
    import re
    if re.search(r"(.)\1{3,}", text):
        return True
    if len(text) <= 3 and not any(c.isalpha() for c in text):
        return True
    tl = text.lower()
    if any(h in tl for h in _HALLUCINATION_FRAGMENTS):
        return True
    words = re.findall(r"\b\w+\b", tl)
    if len(words) >= 4 and len(set(words)) / len(words) < 0.55:
        return True
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if len(trigrams) != len(set(trigrams)):
            return True
    # All-unknown-word check: 6+ words with zero common English words → noise/garbled
    # Catches cases like "Sampakamenta Eustandes Patrimandas Sengit Edita..." from line noise
    if len(words) >= 6 and not any(w in _COMMON_ENGLISH for w in words):
        return True
    return False
