"""Layer 6 — Text to Speech (CosyVoice 2, open source).

    text ──▶ CosyVoice 2 ──▶ speech (wav)

CosyVoice 2 is the current open-source quality leader and multilingual. It is a
heavier install (clone the repo + download the model from ModelScope), so this
adapter isolates it behind one synthesize() call. The LiveKit worker wraps this as
a streaming TTS source.

Setup (one time):
    git clone https://github.com/FunAudioLLM/CosyVoice
    pip install -r CosyVoice/requirements.txt
    # download model: iic/CosyVoice2-0.5B  (via modelscope)
Then set COSYVOICE_DIR / COSYVOICE_MODEL below or via env.
"""
from __future__ import annotations

import os
from functools import lru_cache

_COSYVOICE_DIR = os.getenv("COSYVOICE_DIR", "")        # path to cloned CosyVoice repo
_COSYVOICE_MODEL = os.getenv("COSYVOICE_MODEL", "iic/CosyVoice2-0.5B")
_DEFAULT_SPEAKER = os.getenv("COSYVOICE_SPEAKER", "english_female")


@lru_cache(maxsize=1)
def _engine():
    import sys
    if _COSYVOICE_DIR and _COSYVOICE_DIR not in sys.path:
        sys.path.append(_COSYVOICE_DIR)
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
    return CosyVoice2(_COSYVOICE_MODEL)


def synthesize(text: str, out_path: str, speaker: str | None = None) -> str:
    """Render `text` to a wav file at out_path. Returns the path."""
    import torchaudio  # CosyVoice dependency

    engine = _engine()
    speaker = speaker or _DEFAULT_SPEAKER
    chunks = list(engine.inference_sft(text, speaker, stream=False))
    # Concatenate streamed chunks into one waveform.
    import torch
    audio = torch.cat([c["tts_speech"] for c in chunks], dim=1)
    torchaudio.save(out_path, audio, engine.sample_rate)
    return out_path
