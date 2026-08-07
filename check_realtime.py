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


async def probe_audio_variants(instructions: str, tools: list, hint: str) -> None:
    """Ask the live API which audio settings it actually accepts.

    Format, noise reduction and turn detection are all things the docs either
    show only one example of or do not mention at all. Rather than reason about
    them, send each variant and see what session.update says. Costs nothing —
    no response is ever created.
    """
    import websockets
    from agents.voice.realtime_worker import build_audio_config

    base = dict(transcribe_model=settings.realtime_transcribe_model,
                transcribe_hint=hint, audio_format="pcm",
                noise_reduction="near_field", turn_detection="server_vad",
                eagerness="medium", voice=settings.realtime_voice)

    variants = [
        ("format: audio/pcmu (g711 passthrough)", {**base, "audio_format": "pcmu"}),
        ("format: audio/pcm 24k (current)",       {**base, "audio_format": "pcm"}),
        ("turn_detection: semantic_vad",          {**base, "turn_detection": "semantic_vad"}),
        ("turn_detection: server_vad (current)",  {**base, "turn_detection": "server_vad"}),
        ("noise_reduction: near_field (current)", {**base, "noise_reduction": "near_field"}),
        ("noise_reduction: far_field",            {**base, "noise_reduction": "far_field"}),
        ("noise_reduction: off",                  {**base, "noise_reduction": "off"}),
    ]

    url = f"wss://api.openai.com/v1/realtime?model={settings.realtime_model}"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    print("\n5. Audio configuration probe (no response created, nothing billed)")
    for label, kwargs in variants:
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                await asyncio.wait_for(ws.recv(), timeout=15.0)
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": instructions,
                        "tools": tools,
                        "audio": build_audio_config(**kwargs),
                        "max_output_tokens": settings.realtime_max_response_tokens,
                    },
                }))
                verdict = "no session.updated"
                for _ in range(10):
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
                    if m.get("type") == "error":
                        verdict = "REJECTED: " + (m.get("error", {}).get("message") or "")[:88]
                        break
                    if m.get("type") == "session.updated":
                        verdict = "accepted"
                        break
            mark = _PASS if verdict == "accepted" else _WARN
            print(f"{mark} {label:42} {verdict}")
        except Exception as e:
            print(f"{_WARN} {label:42} {type(e).__name__}: {e}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="also test the other Realtime models")
    ap.add_argument("--audio-probe", action="store_true",
                    help="ask the API which audio formats / VAD / noise "
                         "reduction settings it accepts")
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
    # marin/cedar exist only on gpt-realtime-2 and will be rejected elsewhere.
    if settings.realtime_voice in ("marin", "cedar"):
        check(settings.realtime_model == "gpt-realtime-2",
              f"voice {settings.realtime_voice!r} requires gpt-realtime-2",
              f"REALTIME_MODEL={settings.realtime_model} — use a legacy voice "
              f"(shimmer, alloy, sage...) or switch the model")
    # Not a blocker: the template degrades gracefully, telling callers there is
    # no phone line and giving the email. Fine for testing, weak for production.
    from agents.voice.templates import is_usable_callback_number
    if is_usable_callback_number(settings.callback_number):
        print(f"{_PASS} CALLBACK_NUMBER is a number someone could actually call")
    else:
        print(f"{_WARN} CALLBACK_NUMBER {settings.callback_number!r} is unusable. "
              f"The agent will say it has no phone line and give the email "
              f"instead. Does not block testing; get a real number before "
              f"calling hospitals.")

    check(not settings.server_public_url.startswith("https://your-"),
          "SERVER_PUBLIC_URL is not the placeholder default")

    # Checking the string isn't the default is nearly worthless — a stale ngrok
    # URL from a previous session passes that and then silently swallows every
    # webhook, so the call connects to dead air. Actually probe it.
    try:
        import httpx
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=False) as c:
            r = await c.get(settings.server_public_url.rstrip("/") + "/")
        # FastAPI answers 404 for GET / — that still proves the tunnel reaches us.
        print(f"{_PASS} SERVER_PUBLIC_URL is live (HTTP {r.status_code})")
    except Exception as e:
        print(f"{_WARN} SERVER_PUBLIC_URL is not reachable ({type(e).__name__}). "
              f"Expected if ngrok isn't running yet — but if it IS running, this "
              f"URL is stale. Restart ngrok, copy the new URL into .env, and "
              f"re-run. A stale URL means Twilio's webhooks go nowhere and the "
              f"callee hears silence.")

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

    # Settings the template declares but does not read. Not failures — the
    # template wins by design — but they must never pass silently, because
    # someone set them deliberately and a call cannot be taken back.
    warnings = tpl.config_warnings(agent_language=settings.agent_language)
    for w in warnings:
        print(f"{_WARN} {w}")
    if not warnings:
        print(f"{_PASS} AGENT_LANGUAGE agrees with the template")

    doctor = Doctor(doctor_name="Dr. Jane Okafor", hospital_name="Northside Medical Group",
                    specialization="Cardiology")
    greeting = tpl.build_greeting(doctor, org=settings.org_name)
    context = tpl.build_context(doctor,
                                callback_number=settings.callback_number,
                                callback_email=settings.callback_email,
                                org=settings.org_name)

    print(f"\n      GREETING THE CALLEE WILL HEAR:\n        {greeting}\n")

    # Template 1 is truthful about WHO and WHY — it names the organisation and
    # uses no pretext. Announcing automation upfront is the forage_ai_disclosed
    # variant, so assert per template rather than assuming one shape.
    check(settings.org_name.split()[0].lower() in greeting.lower(),
          f"greeting names the organisation ({settings.org_name})")
    check(settings.org_name not in tpl.instructions,
          "organisation kept out of the cached instructions")
    if tpl.name == "forage_ai_disclosed":
        check("automated" in greeting.lower(),
              "disclosed variant announces automation upfront")
        check("recorded" in greeting.lower(),
              "disclosed variant discloses recording upfront")
    else:
        flat_i = " ".join(tpl.instructions.split())
        check("say yes, you're an automated system" in flat_i
              or "confirm plainly and immediately" in flat_i,
              "confirms it is automated if asked point-blank")
        check("recorded" in flat_i, "confirms the call is recorded if asked")
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

    # ── 3. Twilio account ────────────────────────────────────────────────
    # All read-only REST calls. Catches the setup mistakes that otherwise show
    # up as a call that silently fails to connect.
    print("\n3. Twilio account (read-only, no call placed)")
    if not (settings.twilio_account_sid and settings.twilio_auth_token):
        check(False, "Twilio credentials present")
    else:
        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException

            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            acct = await asyncio.to_thread(
                lambda: client.api.accounts(settings.twilio_account_sid).fetch()
            )
            check(True, "credentials valid", f"{acct.friendly_name}")

            is_trial = (acct.type or "").lower() == "trial"
            if is_trial and settings.use_realtime:
                # NOT a hard failure — I had this wrong. One trial account here
                # could not open <Connect><Stream> (call placed, /answer 200,
                # line dropped after ~3s with no WebSocket), and I concluded
                # Media Streams was paid-only. Another trial account on this
                # same project runs it fine and has completed several calls.
                # Both report type=Trial, so trial status is not the
                # discriminator — newly created accounts appear to be
                # restricted until fully provisioned, matching the 401
                # "Policy evaluation failed" seen on the same account.
                print(f"{_WARN} TRIAL account. Media Streams (<Connect><Stream>) "
                      f"is what the realtime bridge runs on, and some trial "
                      f"accounts cannot open it — the call connects then drops "
                      f"after ~3s with no audio. If that happens, the account "
                      f"needs upgrading. Trial also plays a 'you have a trial "
                      f"account' message to the callee.")
            elif is_trial:
                print(f"{_WARN} TRIAL account — trial message plays to the callee, "
                      f"calls limited to your sign-up country.")
            else:
                print(f"{_PASS} full account — Media Streams available, "
                      f"no trial message")

            # Does the configured from-number belong to this account?
            #
            # Unverified trial accounts return 401 "Policy evaluation failed"
            # (code 20003) on several read endpoints, and can return an EMPTY
            # list here even when the account genuinely owns numbers — while
            # still happily placing calls from them. So an empty list proves
            # nothing and must not be reported as a missing number.
            try:
                numbers = await asyncio.to_thread(
                    lambda: client.incoming_phone_numbers.list(limit=20)
                )
                owned = [n.phone_number for n in numbers]
                if owned:
                    check(settings.twilio_from_number in owned,
                          f"TWILIO_FROM_NUMBER {settings.twilio_from_number} is on this account",
                          f"account owns: {owned}")
                else:
                    print(f"{_WARN} could not list this account's numbers — trial "
                          f"policy restricts the endpoint. Cannot confirm "
                          f"{settings.twilio_from_number} belongs here; a test "
                          f"call is the real check.")
            except Exception:
                print(f"{_WARN} number listing blocked by trial policy — "
                      f"cannot verify TWILIO_FROM_NUMBER from the API")

            # Trial accounts can only call verified numbers. Same caveat.
            try:
                verified = await asyncio.to_thread(
                    lambda: client.outgoing_caller_ids.list(limit=50)
                )
                v_numbers = [v.phone_number for v in verified]
                print(f"      verified to call: {v_numbers or '(none listed)'}")
                if is_trial and not v_numbers:
                    print(f"{_WARN} no verified caller IDs returned — either none "
                          f"are set, or the endpoint is policy-restricted")
            except Exception:
                print(f"{_WARN} verified caller ID list blocked by trial policy — "
                      f"if a call fails to connect, verify the destination number "
                      f"in the console")

        except TwilioRestException as e:
            check(False, "Twilio credentials valid", f"HTTP {e.status}: {e.msg}")
        except ImportError:
            check(False, "twilio package installed")
        except Exception as e:
            check(False, "Twilio check", f"{type(e).__name__}: {e}")

    # ── 4. Live connection ───────────────────────────────────────────────
    print("\n4. Realtime connection (no response.create — nothing generated)")
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

    if args.audio_probe:
        await probe_audio_variants(tpl.instructions, tools, tpl.transcribe_hint)

    print("\n" + "=" * 64)
    if _failures:
        print(f"  {_failures} check(s) failed — fix before placing a call")
    else:
        print("  All checks passed. Safe to place a test call.")
    print("=" * 64 + "\n")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
