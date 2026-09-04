"""Start the Twilio voice bot server and place an outbound call.

Usage:
    python run_twilio.py --doctor "Dr. John" --hospital "Apollo" --to "+919876543210"

Requirements:
    - ngrok running: ngrok http 8000
    - SERVER_PUBLIC_URL set in .env to your ngrok URL
    - TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER in .env
"""
from __future__ import annotations
from typing import Any, cast

import argparse
import threading
import time

import core.bootstrap  # noqa: F401

from core.config import settings
from core.models import Doctor
from agents.voice.templates import resolve_specialty, split_doctor_specialty
import agents.voice.twilio_worker as worker


def _place_call(to_number: str, doctor: Doctor) -> str:
    """Place one outbound call. Returns the CallSid.

    The SID is returned rather than only printed because a batch runner needs
    it: it is the key `_call_id_by_sid` fills in at /answer, which is how
    anything outside this process learns that the callee picked up and which
    call artifact belongs to which row. The single-call path below ignores the
    return value, so this changes nothing for it.
    """
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
        # The kwargs dict is heterogeneous, so the SDK cannot see `to` and
        # `from_` as the strings they are.
        call = client.calls.create(**cast(Any, full))
    except TwilioRestException as e:
        if "disallowed parameter" not in (e.msg or "").lower():
            print(f"\n  *** Could not place the call (Twilio {e.code}) ***")
            print(f"  {_explain(e)}\n")
            raise SystemExit(1)
        print("  Note     : trial account — retrying without status callbacks")
        try:
            call = client.calls.create(**cast(Any, minimal))
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
    # call.sid is Optional on the SDK type; a create() that returned None
    # here would already have failed above.
    worker.register_call(cast(str, call.sid), doctor)
    print(f"\n  Call SID : {call.sid}")
    print(f"  Calling  : {to_number}")
    print(f"  From     : {settings.twilio_from_number}")
    print(f"  Answer   : {answer_url}")
    print(f"  Status   : {status_url}\n")
    return cast(str, call.sid)


def _warmup() -> None:
    """Pre-load models and verify APIs before placing a call."""
    import os, tempfile
    from core.config import settings

    if settings.use_realtime:
        # Realtime API handles STT+LLM+TTS in one WebSocket — nothing to pre-load locally.
        print("  Mode     : OpenAI Realtime API (gpt-realtime-2)")
        # Measured, not aspirational. Agent response latency across live
        # calls: 3.43s, 2.83s, 1.93s, 2.15s. The banner used to claim
        # "~300-500ms | ~$0.06/min", which our own measurements contradict —
        # and it is the first thing printed on every run, so it is what ends
        # up in a screenshot.
        #
        # THE COST RANGE MOVED WITH THE SCRIPT, not with the API's pricing.
        # $0.06-0.12 was measured on the one-field branch templates. Every
        # response re-reads the whole prompt, so a longer script costs more per
        # turn AND takes more turns: patient_discovery is ~5,300 tokens and
        # call-20260902-2207 came to $0.2066 over 130s (18 responses, 130,816
        # cached text tokens, 98.2% hit rate); the 88s provider_verification
        # call on 2026-08-26 was $0.169. A banner nobody can trust is worse
        # than none, and this one is read by whoever is deciding whether to
        # place a batch.
        print("  Latency  : ~2s agent response (measured 1.9-3.4s)")
        print("  Cost     : ~$0.10-0.25 per completed call (measured; "
              "scales with template length)")
        try:
            import websockets  # noqa: F401
        except ImportError:
            print("\n  *** ERROR: websockets not installed — run: pip install websockets ***\n")
            raise SystemExit(1)
        print("  Warmup   : skipped (Realtime API handles everything)\n")
        return

    print("  Warming up STT...", end=" ", flush=True)
    from agents.experiment.stt_whisper import _model, _tiny_model
    _tiny_model()
    _model()
    print("done")

    print("  Warming up TTS...", end=" ", flush=True)
    from agents.experiment.tts_local import synthesize
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
    # OPTIONAL, AND THE DISAMBIGUATOR WHEN PRESENT. Confirmed with the
    # client-side contact 2026-08-25: two doctors of the same name at one
    # hospital is the ordinary case, and the specialty is how the receptionist
    # knows which is meant — both client scripts open "Dr. [Name], [Specialty]".
    #
    # It also closes a gap open since this CLI was written:
    # Doctor.REQUIRED_FOR_COMPLETE names specialization, nothing ever supplied
    # it, so every doctor the voice agent resolved failed is_complete() on that
    # one field and was filed PARTIALLY_VERIFIED however good the call was —
    # exactly what missing_for_complete()'s docstring has described for weeks.
    ap.add_argument("--specialty", default=None,
                    help="Doctor's specialty, e.g. Cardiology. Optional, but it "
                         "is how a receptionist tells two doctors of the same "
                         "name apart, and without it a resolved record cannot "
                         "reach COMPLETE.")
    ap.add_argument("--port",     default=8000,   type=int)
    args = ap.parse_args()

    # A SPECIALTY TYPED INTO THE NAME IS RECOVERED, NOT SILENTLY OBEYED.
    # call-20260903-1126 ran as --doctor "Mark F. Abel Pediatric" with no
    # --specialty; the surname is the last token, so the agent spent the call
    # asking for "Dr. Pediatric" while the receptionist kept saying "Dr. Abel".
    # The name is cleaned for the greeting either way (clean_doctor_name), and
    # the specialty it was carrying lands in the field that exists for it —
    # where it becomes the disambiguator rather than nothing.
    #
    # SAID OUT LOUD, because a silent repair of the operator's input is how the
    # same typo gets made on every call in the batch.
    # Called on the raw string: a leading "Dr." cannot affect a test that only
    # ever looks at TRAILING tokens, so this needs no second copy of the
    # title-stripping regex that clean_doctor_name owns.
    _name, _found = split_doctor_specialty(args.doctor or "")
    _specialty = resolve_specialty(args.doctor, args.specialty)
    if _found:
        print(f"\n  ⚠️  '{_found}' read as a SPECIALTY, not part of the name."
              f"\n      name → {_name!r}"
              + (f", specialty → {_found!r}" if not args.specialty else
                 f"; --specialty {args.specialty!r} kept")
              + "\n      Pass --specialty next time and leave it out of --doctor.")

    doctor = Doctor(doctor_name=args.doctor, hospital_name=args.hospital,
                    specialization=_specialty or None)

    print(f"\n  Doctor   : {doctor.doctor_name}")
    if doctor.specialization:
        print(f"  Specialty: {doctor.specialization}")
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
