"""Audio codec helpers for Telnyx media streaming.

Telnyx sends/receives μ-law (PCMU) audio at 8 kHz.
Whisper and our TTS work at 16 kHz float32.
This module converts between the two.
"""
from __future__ import annotations

import base64
import numpy as np

TELNYX_SR = 8_000
WHISPER_SR = 16_000


# ── μ-law codec ───────────────────────────────────────────────────────────────

def _mulaw_decode(data: bytes) -> np.ndarray:
    """μ-law bytes → float32 PCM in [-1, 1]."""
    try:
        import audioop  # available in Python ≤ 3.12
        pcm = audioop.ulaw2lin(data, 2)
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except (ImportError, AttributeError):
        # Pure-numpy fallback (Python 3.13+)
        b = np.frombuffer(data, dtype=np.uint8).astype(np.int32) ^ 0xFF
        sign = (b >> 7) & 1
        exp  = (b >> 4) & 0x07
        mant = b & 0x0F
        mag  = (((mant << 1) | 1) << (exp + 2)).astype(np.float32)
        mag  = (mag - 264.0) / 32768.0
        return np.where(sign, -mag, mag).astype(np.float32)


def _mulaw_encode(samples: np.ndarray) -> bytes:
    """float32 PCM in [-1, 1] → μ-law bytes."""
    try:
        import audioop
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return audioop.lin2ulaw(pcm, 2)
    except (ImportError, AttributeError):
        s       = np.clip(samples, -1.0, 1.0)
        pcm     = (s * 32767).astype(np.int16)
        sign    = (pcm < 0).astype(np.uint8) << 7
        pcm_abs = np.minimum(np.abs(pcm).astype(np.int32) + 132, 32767)
        exp     = np.floor(np.log2(np.maximum(pcm_abs, 1))).astype(np.int32)
        exp     = np.clip(exp - 2, 0, 7)
        mant    = ((pcm_abs >> (exp + 1)) & 0x0F).astype(np.uint8)
        encoded = (~(sign | (exp.astype(np.uint8) << 4) | mant)).astype(np.uint8)
        return encoded.tobytes()


# ── resampling ────────────────────────────────────────────────────────────────

def resample(arr: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return arr
    try:
        # scipy gives proper sinc-based resampling — much cleaner than linear interp
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(to_sr, from_sr)
        return resample_poly(arr, to_sr // g, from_sr // g).astype(np.float32)
    except ImportError:
        n_out = int(len(arr) * to_sr / from_sr)
        return np.interp(
            np.linspace(0, len(arr) - 1, n_out),
            np.arange(len(arr)),
            arr,
        ).astype(np.float32)


# ── Telnyx payload helpers ────────────────────────────────────────────────────

def telnyx_to_float32(payload_b64: str) -> np.ndarray:
    """Telnyx base64 μ-law payload → float32 at 16 kHz (Whisper-ready)."""
    raw    = base64.b64decode(payload_b64)
    arr_8k = _mulaw_decode(raw)
    return resample(arr_8k, TELNYX_SR, WHISPER_SR)


def float32_to_telnyx(samples_16k: np.ndarray) -> str:
    """float32 at 16 kHz → base64 μ-law payload for Telnyx WebSocket."""
    arr_8k = resample(samples_16k, WHISPER_SR, TELNYX_SR)
    return base64.b64encode(_mulaw_encode(arr_8k)).decode()


def wav_to_float32(path: str) -> tuple[np.ndarray, int]:
    """Read any WAV file → (float32 array, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr
