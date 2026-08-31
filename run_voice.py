"""CLI for Agent 4 — Voice Agent.

The real phone agent runs via the LiveKit worker (see agents/voice/worker.py and
its setup notes). This CLI exercises the *brain* offline: a simulated conversation
with a receptionist, driven by typed lines, so the reasoning/memory/tool-calling
loop is testable with zero telephony, STT, or TTS.

Usage:
    python run_voice.py --selftest        # scripted conversations, both outcomes
    python run_voice.py --interactive     # you type the receptionist's replies
"""
from __future__ import annotations

import argparse

import core.bootstrap  # noqa: F401  (UTF-8 console)
from rich.console import Console

from core.models import Doctor
from agents.experiment.brain import VoiceBrain
from core.memory import CallMemory

console = Console()


def _run_scripted(doctor: Doctor, caller_lines: list[str], call_id: str) -> None:
    memory = CallMemory(call_id=call_id)
    memory.clear()
    memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
    brain = VoiceBrain(doctor, memory, use_llm=False)   # offline heuristic brain

    console.print(f"[bold cyan]Agent:[/bold cyan] {brain.greet()}")
    for line in caller_lines:
        console.print(f"[bold magenta]Caller:[/bold magenta] {line}")
        reply = brain.handle(line)
        console.print(f"[bold cyan]Agent:[/bold cyan] {reply}")
        if brain.done:
            break

    snap = memory.snapshot()
    outcome = (
        f"[green]resolved → branch: {snap.get('branch')}[/green]"
        if snap.get("resolved")
        else "[yellow]escalated (no branch obtained)[/yellow]"
    )
    console.print(f"  result: {outcome}   [dim](memory backend: {memory.backend})[/dim]\n")


def _selftest() -> None:
    console.rule("[bold]Case A — receptionist knows the branch")
    _run_scripted(
        Doctor(doctor_name="John Smith", specialization="Cardiology", hospital_name="Apollo"),
        ["Hello, how can I help?",
         "Oh yes, Dr. John Smith works at our Dallas Medical Center branch."],
        call_id="selftest-A",
    )

    console.rule("[bold]Case B — nobody can confirm, agent escalates")
    _run_scripted(
        Doctor(doctor_name="Jane Doe", specialization="Pediatrics", hospital_name="Mercy"),
        ["Hi there.", "Hmm, I'm not sure.", "Sorry, I really don't know.",
         "No, there's no one here who would know right now."],
        call_id="selftest-B",
    )


def _interactive() -> None:
    doctor = Doctor(doctor_name="John Smith", specialization="Cardiology", hospital_name="Apollo")
    memory = CallMemory(call_id="interactive")
    memory.clear()
    brain = VoiceBrain(doctor, memory, use_llm=False)
    console.print(f"[bold cyan]Agent:[/bold cyan] {brain.greet()}")
    while not brain.done:
        try:
            line = input("Caller: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        console.print(f"[bold cyan]Agent:[/bold cyan] {brain.handle(line)}")
    console.print(f"[dim]result: {memory.snapshot()}[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent 4 — Voice Agent (brain, offline)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        _interactive()
    else:
        _selftest()


if __name__ == "__main__":
    main()
