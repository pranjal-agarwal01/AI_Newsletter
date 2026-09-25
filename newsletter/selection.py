"""Candidate selection: deterministic, cheap, no LLM.
Narrows all fresh unsent articles down to a bounded candidate list;
the model makes the final editorial pick in summarize.py.

1. Drop stories already sent (or queued twice) under a different URL, matched by title.
2. Rank by the profile's keywords: a title hit is worth 3, a body hit 1.
3. Take the top max_candidates, with a cap per source so one prolific
   source (arXiv publishes hundreds of papers a day) can't crowd out the rest.
"""
from __future__ import annotations

import logging
import re

from .models import Article

log = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """Lowercase alphanumerics only, so 'GPT‑5.6 is out!' == 'gpt-5.6 is out'."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _keyword_patterns(profile: dict) -> list[re.Pattern]:
    return [
        re.compile(rf"\b{re.escape(kw.strip().lower())}s?\b")  # s? also matches plurals
        for kw in profile.get("keywords", [])
        if kw.strip()
    ]


def _score(article: Article, patterns: list[re.Pattern]) -> int:
    title = article.title.lower()
    body = article.raw_text[:1000].lower()
    score = 0
    for pattern in patterns:
        if pattern.search(title):
            score += 3
        elif pattern.search(body):
            score += 1
    return score


def select_candidates(
    articles: list[Article],
    profile: dict,
    sent_titles: list[str],
    max_candidates: int,
    max_per_source: int,
    source_caps: dict[str, int],
) -> list[Article]:
    patterns = _keyword_patterns(profile)
    ranked = sorted(
        articles,
        key=lambda a: (_score(a, patterns), a.published_at or ""),
        reverse=True,
    )
    seen_titles = {normalize_title(t) for t in sent_titles}
    candidates: list[Article] = []
    per_source: dict[str, int] = {}
    duplicates = 0
    for article in ranked:
        if len(candidates) >= max_candidates:
            break
        key = normalize_title(article.title)
        if key in seen_titles:
            duplicates += 1
            continue
        if per_source.get(article.source, 0) >= source_caps.get(article.source, max_per_source):
            continue
        seen_titles.add(key)
        candidates.append(article)
        per_source[article.source] = per_source.get(article.source, 0) + 1
    log.info(
        "selection: %d fresh unsent -> %d candidates (%d same-story duplicates dropped; per source: %s)",
        len(articles), len(candidates), duplicates,
        ", ".join(f"{s} {n}" for s, n in sorted(per_source.items(), key=lambda kv: -kv[1])),
    )
    return candidates
