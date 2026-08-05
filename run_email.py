"""CLI for Agent 3 — Email Agent.

Reads data/needs_followup.json (Agent 2's output) and runs the email state machine
for each doctor. In dry-run (default) no real mail is sent; with --live it uses the
SMTP/IMAP credentials in .env.

Usage:
    python run_email.py --selftest          # offline, exercises both branches
    python run_email.py --to hospital@x.com # dry-run over needs_followup.json
    python run_email.py --to hospital@x.com --live
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import core.bootstrap  # noqa: F401  (UTF-8 console; must precede rich)
from rich.console import Console

from core.models import Doctor
from agents.email.agent import run_email_flow, EmailStatus

console = Console()


def _show(state) -> None:
    color = {
        EmailStatus.COMPLETED: "green",
        EmailStatus.VOICE_CALL: "yellow",
    }.get(state.status, "cyan")
    console.print(
        f"[bold]{state.doctor.doctor_name}[/bold] → "
        f"[{color}]{state.status.value}[/{color}]"
        + (f"  (branch: {state.doctor.branch})" if state.doctor.branch else "")
    )
    for line in state.log:
        console.print(f"    [dim]· {line}[/dim]")


def _selftest() -> None:
    console.rule("[bold]Case A — hospital replies with the branch")
    doc_a = Doctor(doctor_name="John Smith", specialization="Cardiology",
                   hospital_name="Apollo")
    reply = "Hello, Dr. John Smith is associated with our Dallas Medical Center branch."
    state_a = run_email_flow(
        doc_a, "apollo@example.com", dry_run=True,
        reply_provider=lambda d: reply,        # inject a reply
    )
    _show(state_a)

    console.rule("[bold]Case B — no reply, escalates to Voice Agent")
    doc_b = Doctor(doctor_name="Jane Doe", specialization="Pediatrics",
                   hospital_name="Mercy")
    state_b = run_email_flow(
        doc_b, "mercy@example.com", dry_run=True,
        reply_provider=lambda d: None,         # never replies
    )
    _show(state_b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent 3 — Email Agent")
    ap.add_argument("--input", default="data/needs_followup.json")
    ap.add_argument("--to", help="hospital email address to contact")
    ap.add_argument("--live", action="store_true", help="actually send/read mail")
    ap.add_argument("--llm", action="store_true", help="use Qwen for generation/parsing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.to:
        ap.error("--to is required (or use --selftest)")

    path = Path(args.input)
    if not path.exists():
        ap.error(f"{path} not found — run Agent 2 first")
    doctors = [Doctor(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
    if not doctors:
        console.print("[yellow]no doctors need follow-up.[/yellow]")
        return

    for d in doctors:
        state = run_email_flow(
            d, args.to, use_llm=args.llm, dry_run=not args.live,
        )
        _show(state)


if __name__ == "__main__":
    main()
