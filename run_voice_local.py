"""Local voice agent — mic & speakers, CPU-only, fully offline.

    mic -> Whisper (STT) -> VoiceBrain (rules/Qwen) -> speaker (TTS)

Records BOTH sides of every call:
  - Agent turns  : saved to wav via pyttsx3 (speak_and_save)
  - Caller turns : recorded from mic via sounddevice/soundfile
  - All clips are merged in order into one full-call wav at 16 kHz
  - Full transcript with HH:MM:SS timestamps
  - Summary, branch, hospital, date/time saved to PostgreSQL or data/calls.json

Modes:
    python run_voice_local.py              # full voice (default)
    python run_voice_local.py --text       # type answers (no mic, no audio)
    python run_voice_local.py --no-speak   # silent; text only (still records mic)

Voice config in .env:
    VOICE_GENDER  female | male
    VOICE_AGE     young | adult | senior
    VOICE_RATE    words/min (0 = auto)
    VOICE_VOLUME  0.0 - 1.0
"""
from __future__ import annotations

import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import core.bootstrap  # noqa: F401  must be first — fixes Windows UTF-8 console

from rich.console import Console
from rich.table import Table

from core.config import settings
from core.models import Doctor, TranscriptTurn
from agents.voice.brain import VoiceBrain
from agents.voice.memory import CallMemory
from agents.recording.agent import record_call

console = Console()

TARGET_SR = 16_000   # all clips resampled to this before merging


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_call_id() -> str:
    now = datetime.now()
    suffix = str(int(time.time()))[-4:]
    return f"call-{now.strftime('%Y%m%d-%H%M')}-{suffix}"


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _agent_wav(call_dir: Path, idx: int) -> str:
    return str(call_dir / f"turn_{idx:03d}_agent.wav")


def _say(text: str, speak: bool, save_to: Optional[str] = None) -> None:
    """Print the agent reply, optionally speak it and save it to a wav clip."""
    console.print(f"\n[bold cyan]  Agent:[/bold cyan] {text}\n")
    if not speak:
        return
    from agents.voice import tts_local
    if save_to:
        tts_local.speak_and_save(text, save_to)
    else:
        tts_local.speak(text)
    # Give audio time to finish and flush any buffered keypresses before next mic turn.
    time.sleep(0.8)


def _listen() -> tuple[str, Optional[str]]:
    """Record mic until Enter, transcribe. Returns (text, wav_path)."""
    from agents.voice import mic, stt_whisper
    console.print("[bold yellow]  >> Speak now — press  Enter  when you finish <<[/bold yellow]")
    wav = mic.record_until_enter()
    console.print("[dim]  (transcribing…)[/dim]")
    text = stt_whisper.transcribe(wav) or ""
    label = text if text else "(nothing heard)"
    console.print(f"[green]  Caller:[/green] {label}\n")
    return text, wav


def _to_float32_16k(path: str):
    """Read any wav → float32 numpy array at TARGET_SR. Returns array or None."""
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:           # stereo → mono
            data = data.mean(axis=1)
        if sr != TARGET_SR:         # resample with linear interpolation
            n_out = int(len(data) * TARGET_SR / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, n_out),
                np.arange(len(data)),
                data,
            ).astype("float32")
        return data
    except Exception:
        return None


def _merge_clips(clip_paths: list[str], out_path: str) -> bool:
    """Merge all audio clips (agent + caller) in order into one wav at 16 kHz."""
    try:
        import numpy as np
        import soundfile as sf

        SILENCE = 0.4                       # seconds of gap between turns
        gap = np.zeros(int(TARGET_SR * SILENCE), dtype="float32")
        chunks = []
        for p in clip_paths:
            if not p or not Path(p).exists():
                continue
            if Path(p).stat().st_size <= 44:    # empty / header-only wav
                continue
            arr = _to_float32_16k(p)
            if arr is not None and len(arr) > 0:
                chunks.append(arr)
                chunks.append(gap)

        if not chunks:
            return False

        combined = np.concatenate(chunks)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, combined, TARGET_SR)
        return True
    except Exception:
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Local voice agent with full call recording")
    ap.add_argument("--text",     action="store_true", help="type answers, no mic")
    ap.add_argument("--no-speak", action="store_true", help="don't play audio through speaker")
    ap.add_argument("--doctor",   default="")
    ap.add_argument("--hospital", default="")
    args = ap.parse_args()

    speak = not args.no_speak

    # ── intake ──
    console.rule("[bold]New Voice Call")
    doctor_name   = args.doctor.strip()   or console.input("[bold]  Doctor name  :[/bold] ").strip() or "Unknown Doctor"
    hospital_name = args.hospital.strip() or console.input("[bold]  Hospital name:[/bold] ").strip() or "Unknown Hospital"

    call_id  = _make_call_id()
    start_dt = datetime.now()
    start_ts = time.time()

    # Temp folder — each turn's wav clip lands here; merged at end
    call_dir = Path(tempfile.gettempdir()) / call_id
    call_dir.mkdir(parents=True, exist_ok=True)

    doctor = Doctor(doctor_name=doctor_name, specialization="Cardiology",
                    hospital_name=hospital_name)
    memory = CallMemory(call_id=call_id)
    memory.clear()
    memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
    brain = VoiceBrain(doctor, memory, use_llm=True)

    # ── banner ──
    console.rule(f"[bold]Call  {call_id}")
    console.print(
        f"  Doctor   : [bold]{doctor.doctor_name}[/bold]    "
        f"Hospital : [bold]{doctor.hospital_name}[/bold]\n"
        f"  Voice    : {settings.piper_voice}    "
        f"Whisper  : {settings.whisper_model}\n"
        f"  Started  : {start_dt.strftime('%Y-%m-%d  %H:%M:%S')}\n"
    )
    if args.text:
        console.print("[dim]  Text mode — type your replies and press Enter.[/dim]")
    else:
        console.print("[dim]  Voice mode — speak into your mic, press Enter to stop.[/dim]")
    console.print("[dim]  You are the hospital receptionist.[/dim]\n")

    # Pre-warm models so first turn has no loading delay
    console.print("[dim]  Loading voice models…[/dim]", end="\r")
    from agents.voice.tts_local import _load_voice, _pick_voice
    from agents.voice.stt_whisper import _model as _whisper_model
    _load_voice(_pick_voice())
    _whisper_model()
    console.print("[dim]  Models ready.          [/dim]\n")

    # All wav clips in call order (agent clips + caller clips interleaved)
    audio_clips: list[str] = []
    turns: list[TranscriptTurn] = []
    turn_idx = 0

    def _add(role: str, text: str) -> None:
        turns.append(TranscriptTurn(role=role, text=text, timestamp=_now_ts()))

    # ── greeting ──
    greeting = brain.greet()
    _add("agent", greeting)
    agent_wav = _agent_wav(call_dir, turn_idx)
    turn_idx += 1
    _say(greeting, speak, save_to=agent_wav if speak else None)
    if speak and Path(agent_wav).exists():
        audio_clips.append(agent_wav)

    # ── conversation loop ──
    while not brain.done:
        if args.text:
            user_text = console.input("[green]  You (type): [/green]")
            mic_wav = None
        else:
            user_text, mic_wav = _listen()
            if mic_wav and Path(mic_wav).exists():
                audio_clips.append(mic_wav)

        if user_text.strip().lower() in {"quit", "exit", "bye"}:
            break

        _add("caller", user_text)

        reply = brain.handle(user_text)
        _add("agent", reply)

        agent_wav = _agent_wav(call_dir, turn_idx)
        turn_idx += 1
        _say(reply, speak, save_to=agent_wav if speak else None)
        if speak and Path(agent_wav).exists():
            audio_clips.append(agent_wav)

    # ── build recording ──
    duration = int(time.time() - start_ts)

    # Merge ALL clips (agent + caller) into one full-call wav
    audio_path: Optional[str] = None
    if audio_clips:
        tmp_full = str(call_dir / "full_call.wav")
        if _merge_clips(audio_clips, tmp_full):
            audio_path = tmp_full   # archive_audio() copies it to data/recordings/

    # Patch snapshot transcript so timestamps are saved
    snap = memory.snapshot()
    snap["transcript"] = [t.model_dump() for t in turns]

    # Agent 5 — persist CallRecord (Postgres or JSON fallback)
    record, backend = record_call(
        snap,
        call_id=call_id,
        audio_path=audio_path,
        duration_seconds=duration,
        use_llm=True,
        persist=True,
    )

    # ── result panel ──
    console.print()
    console.rule("[bold]Call Saved")

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_row("[dim]Call ID[/dim]",    record.call_id)
    tbl.add_row("[dim]Doctor[/dim]",     record.doctor_name)
    tbl.add_row("[dim]Hospital[/dim]",   record.hospital_name or "—")
    tbl.add_row("[dim]Branch[/dim]",
                f"[bold green]{record.branch}[/bold green]"
                if record.branch else "[dim](not obtained)[/dim]")
    tbl.add_row("[dim]Resolved[/dim]",
                "[green]Yes[/green]" if record.resolved else "[red]No[/red]")
    tbl.add_row("[dim]Duration[/dim]",   f"{duration}s")
    tbl.add_row("[dim]Date / Time[/dim]", start_dt.strftime("%Y-%m-%d  %H:%M:%S"))
    tbl.add_row("[dim]Audio file[/dim]",
                record.audio_path or "[dim](text mode — no audio)[/dim]")
    tbl.add_row("[dim]Saved to[/dim]",   backend)
    console.print(tbl)

    console.print()
    console.print("[bold]Summary:[/bold]", record.summary)

    console.print()
    console.print("[bold]Full transcript:[/bold]")
    for t in turns:
        color = "cyan" if t.role == "agent" else "green"
        label = "Agent " if t.role == "agent" else "Caller"
        console.print(f"  [dim]{t.timestamp}[/dim]  [{color}]{label}[/{color}]  {t.text}")

    console.print()


if __name__ == "__main__":
    main()
