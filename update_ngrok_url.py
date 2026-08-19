"""Copy the running tunnel's public URL into SERVER_PUBLIC_URL in .env.

Tunnel URLs change every restart, and a stale SERVER_PUBLIC_URL fails quietly:
Twilio's webhooks go to a dead host, the call connects, and the callee hears
silence. Hand-copying it every session is where that mistake comes from.

TWO TUNNELS ARE SUPPORTED, ngrok first, then cloudflared.

cloudflared was added on 2026-08-19 because Windows Defender began deleting
ngrok.exe on sight -- Trojan:Win32/Kepavll!rfn, a heuristic detection, applied
to a binary winget had just downloaded from ngrok's own CDN with a verified
hash. It removed the copy in Downloads at 21:48 and the winget copy at 22:05,
seconds after ngrok self-updated. Whatever that detection is worth, a tool that
gets deleted between one command and the next cannot be depended on.

They report their URL by different means, and the difference matters:

  * ngrok runs a local API on port 4040, so the URL can be READ FROM THE
    RUNNING PROCESS. If it answers, the URL is live by construction.
  * cloudflared has no such API. Its quick-tunnel URL is only ever PRINTED, so
    it has to be run with --logfile and the URL parsed back out. A log file
    outlives the process that wrote it, so a stale log is a dead URL that looks
    perfectly good -- exactly the failure this script exists to prevent. Hence
    _LOG_MAX_AGE_S, and hence: run check_realtime.py afterwards either way. It
    fetches the URL and confirms THIS agent answers, which is the only check
    that cannot be fooled by a plausible-looking string.

Usage:
    # terminal 1 -- one of:
    ngrok http 8000
    cloudflared tunnel --url http://localhost:8000 --logfile tunnel.log
    # terminal 2
    python update_ngrok_url.py
    python check_realtime.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NGROK_API = "http://127.0.0.1:4040/api/tunnels"
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
KEY = "SERVER_PUBLIC_URL"

# Where `cloudflared --logfile` is told to write, in the order to try.
CF_LOGS = (ROOT / "tunnel.log", ROOT / "cloudflared.log")
# Older than this and the log is assumed to describe a tunnel that has since
# died. Twenty minutes: long enough to start the tunnel, get distracted, and
# come back; short enough that yesterday's log cannot supply today's URL.
_LOG_MAX_AGE_S = 20 * 60
_CF_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def fetch_ngrok_url() -> str | None:
    """The https public URL of the running ngrok tunnel, if any."""
    try:
        with urllib.request.urlopen(NGROK_API, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None

    tunnels = data.get("tunnels") or []
    https = [t for t in tunnels if str(t.get("public_url", "")).startswith("https://")]
    chosen = https or tunnels
    return chosen[0].get("public_url") if chosen else None


def fetch_cloudflared_url() -> tuple[str | None, str]:
    """The most recent trycloudflare URL from a FRESH cloudflared log.

    Returns (url, note). The note explains a refusal, because "no URL found"
    and "found one but the log is from this morning" are different problems and
    only one of them is fixed by starting the tunnel.
    """
    for log in CF_LOGS:
        if not log.exists():
            continue
        age = time.time() - log.stat().st_mtime
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = _CF_URL.findall(text)
        if not found:
            continue
        if age > _LOG_MAX_AGE_S:
            return None, (
                f"{log.name} has a URL but was last written {age/60:.0f} "
                f"minutes ago — that tunnel is almost certainly dead. Start "
                f"cloudflared again (it writes a fresh URL) and re-run this.")
        # Last wins: a restarted tunnel appends a new URL to the same file.
        return found[-1], ""
    return None, ""


def fetch_tunnel_url() -> tuple[str | None, str]:
    """(url, note) from whichever tunnel is up. ngrok first — it is verifiable."""
    url = fetch_ngrok_url()
    if url:
        return url, "ngrok"
    url, note = fetch_cloudflared_url()
    if url:
        return url, "cloudflared"
    return None, note


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

    url, note = fetch_tunnel_url()
    if not url:
        if note:
            print(f"  {note}")
        else:
            print("  No tunnel is running.")
        print("  Start one in another terminal, then re-run this:")
        print("      cloudflared tunnel --url http://localhost:8000 "
              "--logfile tunnel.log")
        print("      ngrok http 8000")
        print("\n  cloudflared MUST be given --logfile: its quick-tunnel URL "
              "is printed and nowhere else,")
        print("  so without the log there is nothing for this script to read.")
        return 1

    old, changed = update_env(url)
    print(f"  tunnel: {note}")
    if not changed:
        print(f"  {KEY} already correct: {url}")
    else:
        print(f"  old: {old or '(not set)'}")
        print(f"  new: {url}")
        print(f"  {KEY} updated in .env")
    # Not optional. This script proves a URL was PRINTED, not that anything
    # answers on it — and a dead URL that looks right is the exact failure it
    # was written to prevent.
    print("\n  Next:  python check_realtime.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
