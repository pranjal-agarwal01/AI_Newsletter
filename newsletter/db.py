"""SQLite storage: articles, issues, and what was sent in each issue."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

from .config import DB_PATH
from .models import Article, RawItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    published_at  TEXT,
    raw_text      TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL UNIQUE,
    fetched_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at       TEXT NOT NULL,
    item_count    INTEGER NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS issue_items (
    issue_id    INTEGER NOT NULL REFERENCES issues(id),
    article_id  INTEGER NOT NULL REFERENCES articles(id),
    rank        INTEGER NOT NULL,
    PRIMARY KEY (issue_id, article_id)
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _normalize_url(url: str) -> str:
    """Lowercase only scheme + host (case-insensitive by spec); keep path/query
    case intact so /Case and /case stay distinct articles."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, parts.fragment))


def content_hash(item: RawItem) -> str:
    return hashlib.sha256(_normalize_url(item.url).encode("utf-8")).hexdigest()


def upsert_article(conn: sqlite3.Connection, item: RawItem) -> tuple[int, bool]:
    """Insert the item unless its hash exists. Returns (article_id, was_new)."""
    h = content_hash(item)
    row = conn.execute("SELECT id FROM articles WHERE content_hash = ?", (h,)).fetchone()
    if row:
        return row["id"], False
    published = item.published_at.astimezone(timezone.utc).isoformat() if item.published_at else None
    cur = conn.execute(
        "INSERT INTO articles (source, title, url, published_at, raw_text, content_hash, fetched_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            item.source,
            item.title,
            item.url,
            published,
            item.raw_text,
            h,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid, True


def set_raw_text(conn: sqlite3.Connection, article_id: int, text: str) -> None:
    conn.execute("UPDATE articles SET raw_text = ? WHERE id = ?", (text, article_id))
    conn.commit()


def unsent_recent_articles(conn: sqlite3.Connection, freshness_hours: int) -> list[Article]:
    """Articles inside the freshness window that were never part of a sent issue."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=freshness_hours)).isoformat()
    rows = conn.execute(
        """
        SELECT a.* FROM articles a
        WHERE a.id NOT IN (SELECT article_id FROM issue_items)
          AND COALESCE(a.published_at, a.fetched_at) >= ?
        ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
        """,
        (cutoff,),
    ).fetchall()
    return [
        Article(
            id=r["id"], source=r["source"], title=r["title"], url=r["url"],
            published_at=r["published_at"], raw_text=r["raw_text"],
        )
        for r in rows
    ]


def hours_since_last_issue(conn: sqlite3.Connection) -> float | None:
    """Hours since the most recent sent issue, or None if nothing was ever sent."""
    row = conn.execute("SELECT sent_at FROM issues ORDER BY sent_at DESC LIMIT 1").fetchone()
    if not row:
        return None
    last = datetime.fromisoformat(row["sent_at"])
    return (datetime.now(timezone.utc) - last).total_seconds() / 3600


def record_issue(
    conn: sqlite3.Connection,
    article_ids_ranked: list[int],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> int:
    cur = conn.execute(
        "INSERT INTO issues (sent_at, item_count, model, input_tokens, output_tokens)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            len(article_ids_ranked),
            model,
            input_tokens,
            output_tokens,
        ),
    )
    issue_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO issue_items (issue_id, article_id, rank) VALUES (?, ?, ?)",
        [(issue_id, aid, rank) for rank, aid in enumerate(article_ids_ranked, start=1)],
    )
    conn.commit()
    return issue_id
