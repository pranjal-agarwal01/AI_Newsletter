"""RSS/Atom adapter: pulls the newest entries from configured blog feeds.

Feeds are fetched with httpx (connect/DNS retries built in) and the bytes are
handed to feedparser — feedparser's own fetching swallows network errors and
silently returns 0 entries, which hid a DNS outage once. Never again.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import feedparser
import httpx

from ..models import RawItem

log = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _strip_html(text: str) -> str:
    return TAG_RE.sub(" ", text or "").strip()


def _entry_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)


class RssAdapter:
    name = "rss"

    def __init__(self, config: dict):
        self.feeds = config.get("feeds", [])
        self.max_items_per_feed = config.get("max_items_per_feed", 15)

    def _parse_feed(self, client: httpx.Client, url: str):
        try:
            response = client.get(url)
            response.raise_for_status()
            return feedparser.parse(response.content)
        except Exception as exc:
            log.warning("rss: httpx fetch failed (%s), trying feedparser directly: %s", exc, url)
            return feedparser.parse(url)

    def fetch(self) -> list[RawItem]:
        items: list[RawItem] = []
        transport = httpx.HTTPTransport(retries=3)
        with httpx.Client(
            transport=transport, timeout=20, follow_redirects=True,
            headers={"User-Agent": BROWSER_UA},
        ) as client:
            for feed in self.feeds:
                try:
                    parsed = self._parse_feed(client, feed["url"])
                    entries = parsed.entries[: self.max_items_per_feed]
                    if not entries:
                        reason = getattr(parsed, "bozo_exception", None) or "feed returned no entries"
                        log.warning("rss: %s -> 0 entries (%s)", feed["name"], reason)
                        continue
                    for entry in entries:
                        url = entry.get("link")
                        title = entry.get("title", "").strip()
                        if not url or not title:
                            continue
                        items.append(
                            RawItem(
                                source=feed["name"],
                                title=title,
                                url=url,
                                published_at=_entry_datetime(entry),
                                raw_text=_strip_html(entry.get("summary", ""))[:4000],
                            )
                        )
                    log.info("rss: %s -> %d entries", feed["name"], len(entries))
                except Exception:
                    log.exception("rss: feed failed, skipping: %s", feed.get("url"))
        return items
