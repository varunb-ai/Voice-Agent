"""Preflight for the Realtime voice agent — no phone call, no tokens generated.

Verifies, in order:
  1. settings resolve and the configured template renders
  2. the prompt-cache split is intact (nothing per-call in the instructions)
  3. the configured REALTIME_MODEL accepts a connection AND our session.update

Step 3 opens a WebSocket and sends one session.update. It never sends
response.create, so no audio or text is generated and nothing is billed beyond
the connection itself.

Usage:
    python check_realtime.py
    python check_realtime.py --probe     # also test the other Realtime models
"""
from __future__ import annotations

import argparse
import asyncio
import json

import core.bootstrap  # noqa: F401  (UTF-8 stdout on Windows)

from core.config import settings
from core.models import Doctor

OTHER_MODELS = ["gpt-realtime-2", "gpt-realtime-mini", "gpt-realtime"]

_PASS, _FAIL, _WARN = "  [PASS]", "  [FAIL]", "  [WARN]"
_failures = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global _failures
    if not ok:
        _failures += 1
    print(f"{_PASS if ok else _FAIL} {label}" + (f"  — {detail}" if detail else ""))
    return ok


async def try_model(model: str, instructions: str, tools: list) -> tuple[bool, str]:
    """Connect and push our real session config. Returns (ok, detail)."""
    import websockets

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
            if first.get("type") == "error":
                return False, first.get("error", {}).get("message", "rejected")

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": instructions,
                    "tools": tools,
                    "audio": {
                        "input": {
                            "transcription": {"model": "whisper-1", "language": "en"},
                            "turn_detection": {"type": "server_vad"},
                        },
                        "output": {"voice": settings.realtime_voice},
                    },
                    "max_output_tokens": settings.realtime_max_response_tokens,
                },
            }))
            for _ in range(10):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
                if msg.get("type") == "error":
                    err = msg.get("error", {})
                    return False, f"session.update rejected: {err.get('message')}"
                if msg.get("type") == "session.updated":
                    return True, "connected, session.update accepted"
            return False, "no session.updated received"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="also test the other Realtime models")
    args = ap.parse_args()

    print("\n" + "=" * 64)
    print("  REALTIME PREFLIGHT")
    print("=" * 64)

    # ── 1. Settings ──────────────────────────────────────────────────────
    print("\n1. Settings")
    for k in ("use_realtime", "call_template", "realtime_model", "realtime_voice",
              "agent_language", "org_name", "callback_number", "callback_email",
              "server_public_url", "twilio_from_number"):
        print(f"      {k:22s} {getattr(settings, k)}")

    check(settings.use_realtime, "USE_REALTIME is true",
          "set USE_REALTIME=true or this runs the classic pipeline")
    check(bool(settings.openai_api_key), "OPENAI_API_KEY present")
    check("555-01" not in settings.callback_number,
          "CALLBACK_NUMBER is a real number",
          f"{settings.callback_number} is in the reserved fictional range and is read aloud on voicemail")
    check(not settings.server_public_url.startswith("https://your-"),
          "SERVER_PUBLIC_URL is set to a real tunnel")

    # ── 2. Template ──────────────────────────────────────────────────────
    print("\n2. Template")
    from agents.voice.templates import get_template
    from agents.voice.tools import TOOL_SCHEMAS
    from agents.voice.realtime_worker import _realtime_tools

    try:
        tpl = get_template(settings.call_template)
    except ValueError as e:
        check(False, "template resolves", str(e))
        return 1
    print(f"      {tpl.name} — {tpl.description}")

    # AGENT_LANGUAGE vs template language. Not a failure — the template wins by
    # design — but it must never pass silently, because someone set that value
    # deliberately and a call in the wrong language cannot be taken back.
    warning = tpl.language_warning(settings.agent_language)
    if warning:
        print(f"{_WARN} {warning}")
    else:
        print(f"{_PASS} AGENT_LANGUAGE agrees with the template ({tpl.language})")

    doctor = Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside Medical Group",
                    specialization="Cardiology")
    greeting = tpl.build_greeting(doctor)
    context = tpl.build_context(doctor,
                                callback_number=settings.callback_number,
                                callback_email=settings.callback_email)

    print(f"\n      GREETING THE CALLEE WILL HEAR:\n        {greeting}\n")

    check("automated assistant" in greeting.lower(),
          "greeting discloses it is automated")
    check("recorded" in greeting.lower(), "greeting discloses recording")
    check("Okafor" not in tpl.instructions and "Northside" not in tpl.instructions,
          "instructions contain no per-call data (cache prefix is stable)")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        static = len(enc.encode(tpl.instructions)) + len(enc.encode(json.dumps(TOOL_SCHEMAS)))
        varying = len(enc.encode(context))
        print(f"      cacheable prefix {static} tok | per-call context {varying} tok")
    except ImportError:
        print("      (pip install tiktoken for a token count)")

    # ── 3. Live connection ───────────────────────────────────────────────
    print("\n3. Realtime connection (no response.create — nothing generated)")
    if not settings.openai_api_key:
        check(False, "cannot test connection", "no API key")
        return 1

    tools = _realtime_tools()
    ok, detail = await try_model(settings.realtime_model, tpl.instructions, tools)
    check(ok, f"REALTIME_MODEL={settings.realtime_model}", detail)

    if args.probe:
        print("\n      Other models on this account:")
        for m in OTHER_MODELS:
            if m == settings.realtime_model:
                continue
            ok_m, detail_m = await try_model(m, tpl.instructions, tools)
            print(f"        {'OK  ' if ok_m else 'no  '} {m:20s} {detail_m}")

    print("\n" + "=" * 64)
    if _failures:
        print(f"  {_failures} check(s) failed — fix before placing a call")
    else:
        print("  All checks passed. Safe to place a test call.")
    print("=" * 64 + "\n")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
