"""arXiv adapter: newest submissions in the configured categories via the Atom API.
Abstracts come with the response, so these items need no later enrichment."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import feedparser
import httpx

from ..models import RawItem

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"


class ArxivAdapter:
    name = "arxiv"

    def __init__(self, config: dict):
        self.categories = config.get("categories", ["cs.AI"])
        self.max_results = config.get("max_results", 40)

    def fetch(self) -> list[RawItem]:
        query = " OR ".join(f"cat:{c}" for c in self.categories)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.max_results),
        }
        try:
            transport = httpx.HTTPTransport(retries=3)
            with httpx.Client(transport=transport, timeout=30, follow_redirects=True) as client:
                response = client.get(API, params=params)
                response.raise_for_status()
        except Exception as exc:
            log.error("arxiv: query failed (%s: %s) — skipping this source", type(exc).__name__, exc)
            return []
        parsed = feedparser.parse(response.text)
        items: list[RawItem] = []
        for entry in parsed.entries:
            published = None
            if entry.get("published_parsed"):
                published = datetime.fromtimestamp(
                    time.mktime(entry.published_parsed), tz=timezone.utc
                )
            items.append(
                RawItem(
                    source="arXiv",
                    title=" ".join(entry.get("title", "").split()),
                    url=entry.get("link", ""),
                    published_at=published,
                    raw_text=" ".join(entry.get("summary", "").split()),
                )
            )
        log.info("arxiv: %d papers fetched", len(items))
        return items
