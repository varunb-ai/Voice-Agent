"""Dial a phone number via Telnyx Call Control API."""
from __future__ import annotations

import httpx
from core.config import settings

_API = "https://api.telnyx.com/v2"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.telnyx_api_key}",
        "Content-Type": "application/json",
    }


def dial(to_number: str, webhook_url: str) -> str:
    """Place an outbound call. Returns call_control_id."""
    r = httpx.post(
        f"{_API}/calls",
        headers=_headers(),
        json={
            "connection_id": settings.telnyx_connection_id,
            "to":            to_number,
            "from":          settings.telnyx_from_number,
            "webhook_url":   webhook_url,
            "webhook_url_method": "POST",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["data"]["call_control_id"]


def hangup(call_control_id: str) -> None:
    try:
        httpx.post(
            f"{_API}/calls/{call_control_id}/actions/hangup",
            headers=_headers(),
            timeout=10,
        )
    except Exception:
        pass
