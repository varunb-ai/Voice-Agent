"""Components 2 & 4 — Sending (SMTP) and Reading (IMAP).

Wraps Python's stdlib smtplib / imaplib. A `dry_run` flag lets the whole pipeline
run with no mailbox configured: sends are logged instead of transmitted, and reply
reads return nothing. Real I/O happens only once EMAIL_ADDRESS/EMAIL_PASSWORD are
set in .env and dry_run=False.
"""
from __future__ import annotations

import email as email_lib
import imaplib
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Optional

from core.config import settings


@dataclass
class IncomingReply:
    from_addr: str
    subject: str
    body: str
    received_at: datetime


def _credentials_present() -> bool:
    return bool(settings.email_address and settings.email_password)


def send_email(to_addr: str, subject: str, body: str, dry_run: bool = False) -> bool:
    """Send via SMTP. Returns True on success. In dry_run, just prints and returns True."""
    if dry_run or not _credentials_present():
        print(f"[mailer:dry-run] would send to {to_addr} | subject={subject!r}")
        return True

    msg = EmailMessage()
    msg["From"] = settings.email_address
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.email_address, settings.email_password)
        server.send_message(msg)
    return True


def _decode(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def _extract_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return _decode(part)
        return ""
    return _decode(msg)


def fetch_replies_from(
    sender: str, since_uid: Optional[str] = None, dry_run: bool = False
) -> list[IncomingReply]:
    """Read unseen messages from `sender` via IMAP. Empty list in dry_run."""
    if dry_run or not _credentials_present():
        return []

    replies: list[IncomingReply] = []
    with imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port) as imap:
        imap.login(settings.email_address, settings.email_password)
        imap.select("INBOX")
        typ, data = imap.search(None, f'(UNSEEN FROM "{sender}")')
        if typ != "OK":
            return []
        for num in data[0].split():
            typ, msg_data = imap.fetch(num, "(RFC822)")
            if typ != "OK":
                continue
            msg = email_lib.message_from_bytes(msg_data[0][1])
            replies.append(
                IncomingReply(
                    from_addr=parseaddr(msg.get("From", ""))[1],
                    subject=msg.get("Subject", ""),
                    body=_extract_body(msg),
                    received_at=datetime.utcnow(),
                )
            )
    return replies
