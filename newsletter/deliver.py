"""Email delivery adapter. Phase 1: Gmail SMTP with an app password.

Each subscriber gets their own message addressed only to them — recipients
never see the rest of the list. Keep the send() signature stable so
SendGrid/Telegram adapters can replace this later.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from .config import env

log = logging.getLogger(__name__)


def recipients() -> list[str]:
    """DIGEST_TO is a comma-separated list; falls back to the sending account."""
    raw = env("DIGEST_TO", env("GMAIL_ADDRESS"))
    return [address.strip() for address in raw.split(",") if address.strip()]


def _build(sender: str, recipient: str, subject: str, html_body: str, text_body: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    return message


def send(subject: str, html_body: str, text_body: str) -> None:
    sender = env("GMAIL_ADDRESS")
    password = env("GMAIL_APP_PASSWORD")
    to_list = recipients()

    sent, failed = 0, []
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        for recipient in to_list:
            try:
                smtp.send_message(_build(sender, recipient, subject, html_body, text_body))
                sent += 1
                log.info("deliver: sent to %s", recipient)
            except smtplib.SMTPException as exc:
                failed.append(recipient)
                log.warning("deliver: failed for %s (%s) — continuing", recipient, exc)

    if not sent:
        raise RuntimeError(
            f"Could not deliver to any recipient ({', '.join(to_list)}). "
            "Check GMAIL_APP_PASSWORD and the addresses in DIGEST_TO."
        )
    log.info("deliver: '%s' delivered to %d of %d recipient(s)", subject, sent, len(to_list))
    if failed:
        log.warning("deliver: undelivered addresses: %s", ", ".join(failed))
