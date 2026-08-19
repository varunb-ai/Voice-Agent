"""Start the Twilio voice bot server and place an outbound call.

Usage:
    python run_twilio.py --doctor "Dr. John" --hospital "Apollo" --to "+919876543210"

Requirements:
    - ngrok running: ngrok http 8000
    - SERVER_PUBLIC_URL set in .env to your ngrok URL
    - TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER in .env
"""
from __future__ import annotations

import argparse
import threading
import time

import core.bootstrap  # noqa: F401

from core.config import settings
from core.models import Doctor
import agents.voice.twilio_worker as worker


def _place_call(to_number: str, doctor: Doctor) -> None:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    answer_url = settings.server_public_url + "/answer"
    status_url = settings.server_public_url + "/status"

    # Trial accounts reject status_callback* with "trial accounts have limited
    # parameter access". Dropping them is safe on the realtime path: the call is
    # saved in handle_realtime's finally block, not from the /status webhook.
    # Only the classic pipeline relies on /status, and that isn't in use here.
    minimal = dict(to=to_number, from_=settings.twilio_from_number, url=answer_url)
    full = dict(
        minimal,
        method="POST",
        status_callback=status_url,
        status_callback_method="POST",
        status_callback_event=["completed"],
        # Recording is started when the stream opens — not here — so we don't
        # capture the ringing gap before the call connects.
    )

    def _explain(err: TwilioRestException) -> str:
        """Turn a Twilio error code into the thing to actually do about it."""
        text = (err.msg or "")
        if err.code == 573003 or "verified voice recipient" in text:
            return (
                f"{to_number} is not a verified caller ID for "
                f"{settings.twilio_from_number} on this account.\n"
                f"  On a trial account a verified recipient is tied to the trial "
                f"number that verified it, so replacing your number breaks the\n"
                f"  binding. Either re-verify the destination in the console, or "
                f"upgrade the account — which removes the restriction entirely\n"
                f"  and is required anyway for Media Streams."
            )
        if err.code == 21210 or "not a Twilio phone number" in text:
            return (f"{settings.twilio_from_number} is not on this account. "
                    f"Check TWILIO_FROM_NUMBER matches TWILIO_ACCOUNT_SID.")
        if "disallowed parameter" in text.lower():
            return ("Trial accounts reject some call parameters. Retried "
                    "without them and it still failed — see above.")
        return text

    try:
        call = client.calls.create(**full)
    except TwilioRestException as e:
        if "disallowed parameter" not in (e.msg or "").lower():
            print(f"\n  *** Could not place the call (Twilio {e.code}) ***")
            print(f"  {_explain(e)}\n")
            raise SystemExit(1)
        print("  Note     : trial account — retrying without status callbacks")
        try:
            call = client.calls.create(**minimal)
        except TwilioRestException as e2:
            # The fallback failing for a DIFFERENT reason chained two full
            # tracebacks together, burying a one-line configuration problem.
            print(f"\n  *** Could not place the call (Twilio {e2.code}) ***")
            print(f"  {_explain(e2)}\n")
            raise SystemExit(1) from None
    # Bind the call to its doctor by CallSid, immediately and before anything
    # else. This is what /answer resolves against; until 2026-08-18 nothing
    # called it and every call fell through to a module global, so two calls in
    # flight would have asked about one doctor. Twilio still has to ring the far
    # end before it fetches /answer, so the gap between create() and here is
    # microseconds against seconds — and /answer waits, so even that is covered.
    worker.register_call(call.sid, doctor)
    print(f"\n  Call SID : {call.sid}")
    print(f"  Calling  : {to_number}")
    print(f"  From     : {settings.twilio_from_number}")
    print(f"  Answer   : {answer_url}")
    print(f"  Status   : {status_url}\n")


def _warmup() -> None:
    """Pre-load models and verify APIs before placing a call."""
    import os, tempfile
    from core.config import settings

    if settings.use_realtime:
        # Realtime API handles STT+LLM+TTS in one WebSocket — nothing to pre-load locally.
        print("  Mode     : OpenAI Realtime API (gpt-realtime-2)")
        # Measured, not aspirational. Agent response latency across live
        # calls: 3.43s, 2.83s, 1.93s, 2.15s. Cost per completed call:
        # $0.06-$0.12. The banner used to claim "~300-500ms | ~$0.06/min",
        # which our own measurements contradict — and it is the first thing
        # printed on every run, so it is what ends up in a screenshot.
        print("  Latency  : ~2s agent response (measured 1.9-3.4s)")
        print("  Cost     : ~$0.06-0.12 per completed call (measured)")
        try:
            import websockets  # noqa: F401
        except ImportError:
            print("\n  *** ERROR: websockets not installed — run: pip install websockets ***\n")
            raise SystemExit(1)
        print("  Warmup   : skipped (Realtime API handles everything)\n")
        return

    print("  Warming up STT...", end=" ", flush=True)
    from agents.voice.stt_whisper import _model, _tiny_model
    _tiny_model()
    _model()
    print("done")

    print("  Warming up TTS...", end=" ", flush=True)
    from agents.voice.tts_local import synthesize
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    synthesize("Hello.", tmp.name)
    os.unlink(tmp.name)
    print("done")

    print("  Warming up LLM...", end=" ", flush=True)
    from core.llm import client as llm_client
    try:
        resp = llm_client().chat.completions.create(
            model    = settings.llm_model,
            messages = [{"role": "user", "content": "Say: ok"}],
            max_tokens = 5,
            timeout  = 10.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        print(f"done  [{settings.llm_model} → '{answer}']\n")
    except Exception as e:
        print(f"\n\n  *** LLM FAILED — calls will use dumb fallback! ***")
        print(f"  Model : {settings.llm_model}")
        print(f"  URL   : {settings.llm_base_url}")
        print(f"  Error : {e}\n")
        raise SystemExit(1)


def _check_ngrok_url_is_current() -> None:
    """Abort if SERVER_PUBLIC_URL doesn't match the running ngrok tunnel.

    ngrok hands out a new URL on every restart. A stale SERVER_PUBLIC_URL still
    passes every other check — the call places fine, then Twilio fetches the
    answer URL from a dead host and plays "we could not reach" to the callee.
    Nothing in the local logs shows a problem, because nothing ever arrives.
    Catch it here rather than burning a call to discover it.
    """
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as r:
            tunnels = _json.load(r).get("tunnels") or []
    except Exception:
        return  # ngrok not running locally, or a different tunnel tool — not our call

    live = [t.get("public_url") for t in tunnels
            if str(t.get("public_url", "")).startswith("https://")]
    if not live:
        return

    configured = settings.server_public_url.rstrip("/")
    if configured in [u.rstrip("/") for u in live]:
        return

    print("\n  *** SERVER_PUBLIC_URL is stale — call aborted ***")
    print(f"  .env points at : {configured}")
    print(f"  ngrok is now at: {live[0]}")
    print("\n  Fix:  python update_ngrok_url.py\n")
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor",   required=True,  help="Doctor name")
    ap.add_argument("--hospital", required=True,  help="Hospital name")
    ap.add_argument("--to",       required=True,  help="Hospital phone number e.g. +919876543210")
    ap.add_argument("--port",     default=8000,   type=int)
    args = ap.parse_args()

    doctor = Doctor(doctor_name=args.doctor, hospital_name=args.hospital)

    print(f"\n  Doctor   : {doctor.doctor_name}")
    print(f"  Hospital : {doctor.hospital_name}")
    print(f"  To       : {args.to}")
    print(f"  Server   : {settings.server_public_url}\n")

    _check_ngrok_url_is_current()
    _warmup()

    print("  Starting server and placing call...\n")

    # Place call after server starts (give uvicorn a moment to bind)
    threading.Timer(2.0, _place_call, args=(args.to, doctor)).start()

    import uvicorn
    uvicorn.run(worker.app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
