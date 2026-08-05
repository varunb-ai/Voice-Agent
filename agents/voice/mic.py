"""Microphone capture for the local voice demo.

Records from the default mic into a temp wav file that Whisper can read. Uses
push-to-talk: recording starts immediately and stops when you press Enter — simpler
and more reliable than automatic silence detection for a first build.
"""
from __future__ import annotations

import queue
import tempfile
import threading

SAMPLE_RATE = 16000      # Whisper expects 16 kHz mono
CHANNELS = 1


def _flush_stdin() -> None:
    """Drain any buffered keypresses (e.g. Enter from the previous turn)."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass  # non-Windows: nothing to flush


def record_until_enter() -> str:
    """Record from the mic until the user presses Enter. Returns a wav file path."""
    import sounddevice as sd
    import soundfile as sf

    _flush_stdin()  # discard buffered Enter from previous turn

    frames: "queue.Queue" = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        frames.put(indata.copy())

    stop = threading.Event()

    def wait_for_enter():
        input()              # blocks this thread until Enter
        stop.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    collected = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", callback=callback):
        while not stop.is_set():
            try:
                collected.append(frames.get(timeout=0.1))
            except queue.Empty:
                pass
    # Drain anything left in the queue.
    while not frames.empty():
        collected.append(frames.get())

    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    if collected:
        import numpy as np
        audio = np.concatenate(collected, axis=0)
        sf.write(path, audio, SAMPLE_RATE)
    else:
        sf.write(path, [], SAMPLE_RATE)
    return path
