"""Component 1 — Email Generation.

Template first (fast, consistent, no cost); Qwen3 only when asked for a richer or
context-specific message. A clean fixed template is the right default for a simple
"which branch is Dr. X at?" ask.
"""
from __future__ import annotations

from core import llm
from core.models import Doctor

_TEMPLATE = """\
Subject: Verifying Dr. {doctor}'s branch — {hospital}

Hello,

We are updating our physician directory and want to make sure our records for
your team are accurate.

Could you please confirm which branch Dr. {doctor}{spec} is affiliated with?

Thank you for your time.

Best regards,
Directory Verification Team
"""

_LLM_SYSTEM = (
    "You write short, polite, professional verification emails to hospitals asking "
    "which branch a named doctor works at. Plain text. Start with a 'Subject:' line."
)


def generate(doctor: Doctor, use_llm: bool = False) -> tuple[str, str]:
    """Return (subject, body). Template by default; Qwen if use_llm=True and reachable."""
    if use_llm and llm.is_available():
        prompt = (
            f"Doctor: {doctor.doctor_name}\n"
            f"Specialization: {doctor.specialization or 'unknown'}\n"
            f"Hospital: {doctor.hospital_name or 'the hospital'}\n"
            "Ask which branch this doctor is affiliated with."
        )
        try:
            text = llm.chat(prompt, system=_LLM_SYSTEM)
            return _split_subject(text)
        except Exception:
            pass  # fall through to template

    spec = f" ({doctor.specialization})" if doctor.specialization else ""
    text = _TEMPLATE.format(
        doctor=doctor.doctor_name,
        hospital=doctor.hospital_name or "your hospital",
        spec=spec,
    )
    return _split_subject(text)


def _split_subject(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    subject = "Verifying doctor branch"
    body_start = 0
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return subject, body
