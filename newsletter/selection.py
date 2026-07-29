"""Phase-1 candidate selection: deterministic, cheap, no LLM.
Narrows all fresh unsent articles down to a bounded candidate list;
Claude makes the final editorial pick in summarize.py.

This module is the seam where the Phase-2 embedding-based ranking engine lands.
"""
from __future__ import annotations

import logging
import re

from .models import Article

log = logging.getLogger(__name__)

STOPWORDS = {
    "and", "the", "for", "with", "that", "this", "from", "are", "was", "can",
    "how", "what", "when", "your", "you", "not", "but", "all", "its", "own",
    "new", "best", "practices", "tools", "preparation",
}

MAX_PER_SOURCE = 10


def _profile_keywords(profile: dict) -> set[str]:
    text = " ".join(
        profile.get("interests", [])
        + profile.get("tech_stack", [])
        + [profile.get("target_role", "")]
    ).lower()
    words = re.findall(r"[a-z][a-z0-9+#-]{2,}", text)
    return {w for w in words if w not in STOPWORDS}


def _score(article: Article, keywords: set[str]) -> int:
    title = article.title.lower()
    body = article.raw_text[:1000].lower()
    score = 0
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, title):
            score += 3
        elif re.search(pattern, body):
            score += 1
    return score


def select_candidates(articles: list[Article], profile: dict, max_candidates: int) -> list[Article]:
    keywords = _profile_keywords(profile)
    scored = sorted(
        articles,
        key=lambda a: (_score(a, keywords), a.published_at or ""),
        reverse=True,
    )
    candidates: list[Article] = []
    per_source: dict[str, int] = {}
    for article in scored:
        if len(candidates) >= max_candidates:
            break
        if per_source.get(article.source, 0) >= MAX_PER_SOURCE:
            continue
        candidates.append(article)
        per_source[article.source] = per_source.get(article.source, 0) + 1
    log.info(
        "selection: %d fresh unsent -> %d candidates (per-source cap %d)",
        len(articles), len(candidates), MAX_PER_SOURCE,
    )
    return candidates
