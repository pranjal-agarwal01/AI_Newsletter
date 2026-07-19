"""Hacker News adapter: top stories via the official Firebase API,
filtered by score and AI-related keywords."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from ..models import RawItem

log = logging.getLogger(__name__)

API = "https://hacker-news.firebaseio.com/v0"


class HackerNewsAdapter:
    name = "hackernews"

    def __init__(self, config: dict):
        self.max_stories = config.get("max_stories", 60)
        self.min_score = config.get("min_score", 80)
        keywords = config.get("keywords", [])
        self.patterns = [
            re.compile(rf"\b{re.escape(kw.strip())}\b", re.IGNORECASE) for kw in keywords
        ]

    def _title_matches(self, title: str) -> bool:
        return any(p.search(title) for p in self.patterns)

    def fetch(self) -> list[RawItem]:
        items: list[RawItem] = []
        transport = httpx.HTTPTransport(retries=3)
        with httpx.Client(transport=transport, timeout=15) as client:
            try:
                ids = client.get(f"{API}/topstories.json").json()[: self.max_stories]
            except Exception as exc:
                log.error("hackernews: failed to list top stories (%s) — skipping this source", exc)
                return items
            for story_id in ids:
                try:
                    story = client.get(f"{API}/item/{story_id}.json").json()
                except Exception:
                    log.warning("hackernews: item %s failed, skipping", story_id)
                    continue
                if not story or story.get("type") != "story":
                    continue
                title = story.get("title", "")
                score = story.get("score", 0)
                if score < self.min_score or not self._title_matches(title):
                    continue
                discussion = f"https://news.ycombinator.com/item?id={story_id}"
                items.append(
                    RawItem(
                        source="Hacker News",
                        title=title,
                        url=story.get("url") or discussion,
                        published_at=datetime.fromtimestamp(story.get("time", 0), tz=timezone.utc),
                        raw_text=story.get("text", ""),
                        extra={"score": score, "discussion": discussion},
                    )
                )
        log.info("hackernews: %d stories passed score/keyword filter", len(items))
        return items
