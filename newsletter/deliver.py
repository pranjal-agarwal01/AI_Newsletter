"""Email delivery. Two providers, chosen automatically from the environment:

- resend: RESEND_API_KEY present -> Resend HTTP API. Use this for a real
  subscriber list sent from your own verified domain (SPF/DKIM/DMARC).
- gmail:  otherwise -> Gmail SMTP app password. Fine for a few recipients.

Force one with EMAIL_PROVIDER=resend|gmail.

Each subscriber gets their own message addressed only to them — recipients
never see the rest of the list. Both paths add List-Unsubscribe and Reply-To
headers (the 2026 bulk-sender rules expect them). Keep the send() signature
stable so a future SendGrid/Telegram adapter can drop in.
"""
from __future__ import annotations

import logging
import os
import smtplib
import time
from email.message import EmailMessage

import httpx

from .config import env

log = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


def provider() -> str:
    forced = os.getenv("EMAIL_PROVIDER", "").strip().lower()
    if forced in ("resend", "gmail"):
        return forced
    if os.getenv("RESEND_API_KEY"):
        return "resend"
    return "gmail"


def recipients() -> list[str]:
    """DIGEST_TO is a comma-separated list; falls back to the sending account."""
    default = os.getenv("GMAIL_ADDRESS") or os.getenv("MAIL_FROM", "")
    raw = env("DIGEST_TO", default)
    return [address.strip() for address in raw.split(",") if address.strip()]


def _unsubscribe_value() -> str | None:
    """List-Unsubscribe header value. MAIL_UNSUBSCRIBE may be an https:// URL
    (best — enables true one-click) or a mailto:. Returns None if unset."""
    target = os.getenv("MAIL_UNSUBSCRIBE", "").strip()
    if not target:
        return None
    return f"<{target}>"


def _reply_to() -> str | None:
    return os.getenv("MAIL_REPLY_TO", "").strip() or None


# --- Gmail (SMTP) ----------------------------------------------------------

def _gmail_build(sender: str, recipient: str, subject: str, html_body: str, text_body: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    if _reply_to():
        message["Reply-To"] = _reply_to()
    unsub = _unsubscribe_value()
    if unsub:
        message["List-Unsubscribe"] = unsub
        if unsub.startswith("<https"):
            message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    return message


def _gmail_send(subject: str, html_body: str, text_body: str, to_list: list[str]) -> tuple[int, list[str]]:
    sender = env("GMAIL_ADDRESS")
    password = env("GMAIL_APP_PASSWORD")
    sent, failed = 0, []
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        for recipient in to_list:
            try:
                smtp.send_message(_gmail_build(sender, recipient, subject, html_body, text_body))
                sent += 1
                log.info("deliver[gmail]: sent to %s", recipient)
            except smtplib.SMTPException as exc:
                failed.append(recipient)
                log.warning("deliver[gmail]: failed for %s (%s) — continuing", recipient, exc)
    return sent, failed


# --- Resend (HTTP API) -----------------------------------------------------

def _resend_payload(sender: str, recipient: str, subject: str, html_body: str, text_body: str) -> dict:
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "html": html_body,
        "text": text_body,
    }
    if _reply_to():
        payload["reply_to"] = _reply_to()
    unsub = _unsubscribe_value()
    if unsub:
        headers = {"List-Unsubscribe": unsub}
        if unsub.startswith("<https"):
            headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        payload["headers"] = headers
    return payload


def _resend_post(payload: dict, auth: dict) -> bool:
    """One recipient. Retries transient failures; returns True on success,
    False on a permanent per-recipient rejection (e.g. invalid address)."""
    backoffs = [2, 5]
    for attempt in range(len(backoffs) + 1):
        try:
            response = httpx.post(RESEND_URL, json=payload, headers=auth, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.TransportError(f"HTTP {response.status_code}")
            if response.status_code >= 400:
                log.warning(
                    "deliver[resend]: rejected %s (HTTP %d: %s)",
                    payload["to"][0], response.status_code, response.text[:200],
                )
                return False
            return True
        except httpx.TransportError as exc:
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
            else:
                log.warning("deliver[resend]: gave up on %s (%s)", payload["to"][0], exc)
                return False
    return False


def _resend_send(subject: str, html_body: str, text_body: str, to_list: list[str]) -> tuple[int, list[str]]:
    api_key = env("RESEND_API_KEY")
    sender = env("MAIL_FROM")  # e.g. "Your Digest <news@yourdomain.com>", domain must be verified
    auth = {"Authorization": f"Bearer {api_key}"}
    sent, failed = 0, []
    for recipient in to_list:
        payload = _resend_payload(sender, recipient, subject, html_body, text_body)
        if _resend_post(payload, auth):
            sent += 1
            log.info("deliver[resend]: sent to %s", recipient)
        else:
            failed.append(recipient)
    return sent, failed


# --- Public entry point ----------------------------------------------------

def send(subject: str, html_body: str, text_body: str) -> None:
    to_list = recipients()
    prov = provider()

    if prov == "resend":
        sent, failed = _resend_send(subject, html_body, text_body, to_list)
    else:
        sent, failed = _gmail_send(subject, html_body, text_body, to_list)

    if not sent:
        raise RuntimeError(
            f"[{prov}] could not deliver to any recipient ({', '.join(to_list)}). "
            "Check credentials and the addresses in DIGEST_TO."
        )
    log.info("deliver[%s]: '%s' delivered to %d of %d recipient(s)", prov, subject, sent, len(to_list))
    if failed:
        log.warning("deliver[%s]: undelivered addresses: %s", prov, ", ".join(failed))
