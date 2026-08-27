"""Bridges our open-source STT/TTS into LiveKit Agents' plugin interfaces.

LiveKit expects stt.STT and tts.TTS subclasses that stream frames. Our Whisper and
CosyVoice adapters are non-streaming (file/array in, text/wav out), so these
classes buffer a turn's audio and run the model once per utterance — simplest
correct integration; can be upgraded to true streaming later.

This is the one piece that is version-sensitive: the exact base-class methods
(`_recognize_impl`, `synthesize`, frame types) vary across livekit-agents releases.
Confirm signatures against your installed version before relying on it.
"""
from __future__ import annotations

from agents.experiment import stt_whisper, tts_cosyvoice


class WhisperSTT:
    """Wrap faster-whisper as a LiveKit STT. Buffers an utterance, transcribes once."""

    def __init__(self, language: str = "en"):
        self.language = language

    def transcribe_bytes(self, wav_bytes: bytes) -> str:
        # Write to a temp file because faster-whisper takes a path/array.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        try:
            return stt_whisper.transcribe(path, language=self.language)
        finally:
            os.unlink(path)

    # TODO(livekit): subclass livekit.agents.stt.STT and implement _recognize_impl
    #   to call transcribe_bytes() on the buffered frames for the current turn.


class CosyVoiceTTS:
    """Wrap CosyVoice 2 as a LiveKit TTS. Renders a reply to wav, streams it back."""

    def synthesize_to_file(self, text: str, out_path: str) -> str:
        return tts_cosyvoice.synthesize(text, out_path)

    # TODO(livekit): subclass livekit.agents.tts.TTS and implement synthesize()
    #   to yield audio frames from the rendered wav.
