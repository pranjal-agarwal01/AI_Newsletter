"""Fetches the full page for thin articles and extracts clean text with trafilatura.
Bounded and defensive: per-request timeout, streamed size cap, content-type check,
and an SSRF guard that refuses private/loopback/link-local/metadata hosts (matters
on self-hosted or cloud runners with an internal network)."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
import trafilatura

from . import db
from .models import Article

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 400   # articles with less stored text than this get enriched
MAX_TEXT_CHARS = 8000  # cap stored text so the LLM prompt stays bounded
MAX_BYTES = 3_000_000  # stop reading a response after ~3 MB
FETCH_TIMEOUT = httpx.Timeout(10, connect=5)  # fail fast when the network is down
OK_CONTENT = ("text/html", "application/xhtml", "text/plain", "application/xml", "text/xml")

HEADERS = {"User-Agent": "Mozilla/5.0 (personal newsletter bot; contact via profile)"}


def _host_is_public(host: str | None) -> bool:
    """True only if every resolved IP for host is a normal public address.
    Blocks localhost, private ranges, link-local (incl. cloud metadata
    169.254.169.254), reserved, multicast, and unspecified addresses."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _fetch_clean_text(client: httpx.Client, url: str) -> str | None:
    """Return extracted article text, or None if the URL is unsafe/unusable."""
    if urlparse(url).scheme not in ("http", "https"):
        return None
    if not _host_is_public(urlparse(url).hostname):
        log.warning("enrich: refusing non-public host: %s", url)
        return None
    with client.stream("GET", url) as response:
        response.raise_for_status()
        # re-validate after redirects — the final hop must also be public
        if not _host_is_public(urlparse(str(response.url)).hostname):
            log.warning("enrich: redirect landed on non-public host: %s", response.url)
            return None
        ctype = response.headers.get("content-type", "").lower()
        if ctype and not any(ctype.startswith(c) for c in OK_CONTENT):
            return None
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) >= MAX_BYTES:
                break
    return trafilatura.extract(body.decode("utf-8", errors="ignore")) or ""


def enrich_articles(conn, articles: list[Article]) -> None:
    transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(
        transport=transport, timeout=FETCH_TIMEOUT, follow_redirects=True,
        headers=HEADERS, max_redirects=5,
    ) as client:
        for article in articles:
            if len(article.raw_text) >= MIN_TEXT_CHARS:
                continue
            if article.source == "arXiv" or article.url.lower().endswith(".pdf"):
                continue
            try:
                text = _fetch_clean_text(client, article.url)
            except Exception:
                log.warning("enrich: fetch failed, keeping feed summary: %s", article.url)
                continue
            if text and len(text) > len(article.raw_text):
                article.raw_text = text[:MAX_TEXT_CHARS]
                db.set_raw_text(conn, article.id, article.raw_text)
                log.info("enrich: %s -> %d chars", article.url, len(article.raw_text))
