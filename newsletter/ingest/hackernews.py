"""Hacker News adapter: every story from the last `lookback_hours` above
`min_score` points, via the official HN search API (Algolia), filtered by
AI-related title keywords.

One request covers the whole window, so a story that peaked and left the front
page between two runs is still caught (the old top-stories snapshot missed
those, and needed one request per story).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import httpx

from ..models import RawItem

log = logging.getLogger(__name__)

API = "https://hn.algolia.com/api/v1/search"


class HackerNewsAdapter:
    name = "hackernews"

    def __init__(self, config: dict):
        self.lookback_hours = config.get("lookback_hours", 36)
        self.min_score = config.get("min_score", 80)
        self.patterns = [
            re.compile(rf"\b{re.escape(kw.strip())}s?\b", re.IGNORECASE)  # s? also matches plurals
            for kw in config.get("keywords", [])
        ]

    def _title_matches(self, title: str) -> bool:
        return any(p.search(title) for p in self.patterns)

    def fetch(self) -> list[RawItem]:
        since = int(time.time()) - self.lookback_hours * 3600
        params = {
            "tags": "story",
            "numericFilters": f"created_at_i>{since},points>={self.min_score}",
            "hitsPerPage": 1000,
        }
        try:
            transport = httpx.HTTPTransport(retries=3)
            with httpx.Client(transport=transport, timeout=30) as client:
                response = client.get(API, params=params)
                response.raise_for_status()
                hits = response.json()["hits"]
        except Exception as exc:
            log.error("hackernews: search failed (%s: %s) — skipping this source", type(exc).__name__, exc)
            return []

        items: list[RawItem] = []
        for hit in hits:
            title = hit.get("title") or ""
            if not self._title_matches(title):
                continue
            discussion = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            items.append(
                RawItem(
                    source="Hacker News",
                    title=title,
                    url=hit.get("url") or discussion,  # Ask/Show HN text posts have no url
                    published_at=datetime.fromtimestamp(hit.get("created_at_i", 0), tz=timezone.utc),
                    raw_text=hit.get("story_text") or "",
                )
            )
        log.info("hackernews: %d of %d stories matched keywords", len(items), len(hits))
        return items
