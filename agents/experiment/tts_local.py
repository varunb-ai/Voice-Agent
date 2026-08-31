"""Local CPU text-to-speech using Piper TTS.

Piper is fast, CPU-optimized, and has 100+ voice variants.
Voices are stored in data/voices/ as .onnx + .onnx.json files.

Available voices (set PIPER_VOICE in .env):
    en_US-amy-medium       female, American, natural
    en_US-ryan-high        male,   American, very natural
    en_US-lessac-high      female, American, professional
    en_GB-alba-medium      female, British

For random voice per call (anti-bot detection), set:
    PIPER_VOICE=random
"""
from __future__ import annotations

import random
import wave
from functools import lru_cache
from pathlib import Path

from core.config import settings

_VOICES_DIR = Path(__file__).parent.parent.parent / "data" / "voices"

# All downloaded voices
_ALL_VOICES = [
    "en_US-amy-medium",
    "en_US-ryan-high",
    "en_US-lessac-high",
    "en_GB-alba-medium",
]

# voice variants by gender
_FEMALE_VOICES = ["en_US-amy-medium", "en_US-lessac-high", "en_GB-alba-medium"]
_MALE_VOICES   = ["en_US-ryan-high"]


def _pick_voice() -> str:
    voice = getattr(settings, "piper_voice", "en_US-lessac-high")
    if voice == "random":
        return random.choice(_ALL_VOICES)
    if voice == "random_female":
        return random.choice(_FEMALE_VOICES)
    if voice == "random_male":
        return random.choice(_MALE_VOICES)
    return voice


@lru_cache(maxsize=8)
def _load_voice(voice_name: str):
    from piper import PiperVoice  # type: ignore[import-not-found]  # optional runtime dep
    model_path = _VOICES_DIR / f"{voice_name}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Voice model not found: {model_path}\n"
            f"Available: {[p.stem for p in _VOICES_DIR.glob('*.onnx')]}"
        )
    return PiperVoice.load(str(model_path))


def _synthesize_openai(text: str, out_path: str) -> str:
    """Render text to WAV using OpenAI TTS (nova voice). Returns out_path."""
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="wav",
    )
    response.stream_to_file(out_path)
    return out_path


def synthesize(text: str, out_path: str) -> str:
    """Render text to a WAV file. Uses OpenAI TTS (nova) if OPENAI_API_KEY is set, else Piper."""
    if settings.openai_api_key:
        return _synthesize_openai(text, out_path)
    voice_name = _pick_voice()
    voice = _load_voice(voice_name)
    with wave.open(out_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, set_wav_format=True)
    return out_path


async def synthesize_stream_chunks(text: str):
    """Async generator: yields (ulaw_b64, float32_16k) tuples as audio arrives from OpenAI.

    Runs the sync OpenAI PCM streaming in a thread pool so the async event loop
    is never blocked. First chunk arrives ~200ms after the call starts.
    Works with all openai-python SDK versions (iter_bytes, not aiter_bytes).
    """
    import asyncio
    import numpy as np
    from openai import OpenAI
    from core.audio_utils import resample, float32_to_telnyx

    OPENAI_SR = 24_000
    BYTES_PER_SAMPLE = 2
    CHUNK_SAMPLES = 4_800  # 200ms at 24kHz

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format="pcm",
            ) as resp:
                for raw in resp.iter_bytes(chunk_size=9_600):
                    loop.call_soon_threadsafe(queue.put_nowait, raw)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    producer_task = loop.run_in_executor(None, _producer)

    buffer = b""
    while True:
        raw = await queue.get()
        if raw is None:
            break
        buffer += raw
        while len(buffer) >= CHUNK_SAMPLES * BYTES_PER_SAMPLE:
            pcm = buffer[:CHUNK_SAMPLES * BYTES_PER_SAMPLE]
            buffer = buffer[CHUNK_SAMPLES * BYTES_PER_SAMPLE:]
            f32_24k = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            f32_16k = resample(f32_24k, OPENAI_SR, 16_000)
            yield float32_to_telnyx(f32_16k), f32_16k

    await producer_task  # wait for the thread to fully exit

    # Flush remaining buffer
    if len(buffer) >= BYTES_PER_SAMPLE:
        if len(buffer) % BYTES_PER_SAMPLE:
            buffer = buffer[: -(len(buffer) % BYTES_PER_SAMPLE)]
        f32_24k = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        f32_16k = resample(f32_24k, OPENAI_SR, 16_000)
        yield float32_to_telnyx(f32_16k), f32_16k


def speak(text: str) -> None:
    """Synthesize and play audio through speakers (blocking)."""
    import tempfile, subprocess, sys
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        synthesize(text, tmp.name)
        if sys.platform == "win32":
            subprocess.run(
                ["powershell", "-c", f'(New-Object Media.SoundPlayer "{tmp.name}").PlaySync()'],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(["aplay", tmp.name], check=True, capture_output=True)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def speak_and_save(text: str, out_path: str) -> str:
    """Synthesize, save to out_path, AND play through speakers."""
    synthesize(text, out_path)
    import subprocess, sys
    if sys.platform == "win32":
        subprocess.run(
            ["powershell", "-c", f'(New-Object Media.SoundPlayer "{out_path}").PlaySync()'],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(["aplay", out_path], check=True, capture_output=True)
    return out_path
