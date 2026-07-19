"""Fetches the full page for thin articles and extracts clean text with trafilatura.
Bounded on purpose: per-request timeout, char cap, failures skipped."""
from __future__ import annotations

import logging

import httpx
import trafilatura

from . import db
from .models import Article

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 400   # articles with less stored text than this get enriched
MAX_TEXT_CHARS = 8000  # cap stored text so the LLM prompt stays bounded
FETCH_TIMEOUT = httpx.Timeout(10, connect=5)  # fail fast when the network is down

HEADERS = {"User-Agent": "Mozilla/5.0 (personal newsletter bot; contact via profile)"}


def enrich_articles(conn, articles: list[Article]) -> None:
    transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(
        transport=transport, timeout=FETCH_TIMEOUT, follow_redirects=True, headers=HEADERS
    ) as client:
        for article in articles:
            if len(article.raw_text) >= MIN_TEXT_CHARS:
                continue
            if article.source == "arXiv" or article.url.lower().endswith(".pdf"):
                continue
            try:
                response = client.get(article.url)
                response.raise_for_status()
                text = trafilatura.extract(response.text) or ""
            except Exception:
                log.warning("enrich: fetch failed, keeping feed summary: %s", article.url)
                continue
            if len(text) > len(article.raw_text):
                article.raw_text = text[:MAX_TEXT_CHARS]
                db.set_raw_text(conn, article.id, article.raw_text)
                log.info("enrich: %s -> %d chars", article.url, len(article.raw_text))
