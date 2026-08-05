"""Copy the running ngrok tunnel's URL into SERVER_PUBLIC_URL in .env.

Free ngrok URLs change every restart, and a stale SERVER_PUBLIC_URL fails
quietly: Twilio's webhooks go to a dead host, the call connects, and the callee
hears silence. Hand-copying it every session is where that mistake comes from.

ngrok exposes a local API on port 4040 while running. This reads the public
URL from there and rewrites the one line in .env, leaving everything else --
including credentials and comments -- byte-identical.

Usage:
    # terminal 1
    C:\\Users\\varun\\Downloads\\ngrok\\ngrok.exe http 8000
    # terminal 2
    python update_ngrok_url.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

NGROK_API = "http://127.0.0.1:4040/api/tunnels"
ENV_PATH = Path(__file__).resolve().parent / ".env"
KEY = "SERVER_PUBLIC_URL"


def fetch_tunnel_url() -> str | None:
    """Return the https public URL of the running ngrok tunnel, if any."""
    try:
        with urllib.request.urlopen(NGROK_API, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None

    tunnels = data.get("tunnels") or []
    https = [t for t in tunnels if str(t.get("public_url", "")).startswith("https://")]
    chosen = https or tunnels
    return chosen[0].get("public_url") if chosen else None


def update_env(url: str) -> tuple[str | None, bool]:
    """Rewrite the SERVER_PUBLIC_URL line. Returns (old_value, changed)."""
    text = io.open(ENV_PATH, encoding="utf-8").read()
    pattern = re.compile(rf"^{KEY}=(.*)$", re.MULTILINE)
    match = pattern.search(text)

    if not match:
        # Key absent entirely — append it rather than silently doing nothing.
        with io.open(ENV_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{KEY}={url}\n")
        return None, True

    old = match.group(1).strip()
    if old == url:
        return old, False

    io.open(ENV_PATH, "w", encoding="utf-8", newline="").write(
        pattern.sub(f"{KEY}={url}", text, count=1)
    )
    return old, True


def main() -> int:
    if not ENV_PATH.exists():
        print(f"  .env not found at {ENV_PATH}")
        return 1

    url = fetch_tunnel_url()
    if not url:
        print("  ngrok does not appear to be running.")
        print("  Start it in another terminal, then re-run this:")
        print(r"      C:\Users\varun\Downloads\ngrok\ngrok.exe http 8000")
        return 1

    old, changed = update_env(url)
    if not changed:
        print(f"  {KEY} already correct: {url}")
    else:
        print(f"  old: {old or '(not set)'}")
        print(f"  new: {url}")
        print(f"  {KEY} updated in .env")
    print("\n  Next:  python check_realtime.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
