"""SQLite storage: articles, issues, and what was sent in each issue."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
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


def upsert_article(conn: sqlite3.Connection, item: RawItem) -> bool:
    """Insert the item unless its hash exists. Returns True if it was new."""
    h = content_hash(item)
    if conn.execute("SELECT 1 FROM articles WHERE content_hash = ?", (h,)).fetchone():
        return False
    published = item.published_at.astimezone(timezone.utc).isoformat() if item.published_at else None
    conn.execute(
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
    return True


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


def recently_sent_titles(conn: sqlite3.Connection, days: int) -> list[str]:
    """Titles of articles sent in the last `days` days — used to catch the same
    story arriving again under a different URL (e.g. blog post + HN link)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT a.title FROM issue_items ii
        JOIN issues i ON i.id = ii.issue_id
        JOIN articles a ON a.id = ii.article_id
        WHERE i.sent_at >= ?
        """,
        (cutoff,),
    ).fetchall()
    return [r["title"] for r in rows]


def prune_old_text(conn: sqlite3.Connection, days: int) -> None:
    """Drop the stored body text of articles older than `days`. Old articles are
    never candidates again, but their text made up most of the git-committed
    database. Rows (title/URL/hash) stay so dedup keeps working."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cleared = conn.execute(
        "UPDATE articles SET raw_text = '' WHERE fetched_at < ? AND raw_text != ''", (cutoff,)
    ).rowcount
    conn.commit()
    if cleared:
        conn.execute("VACUUM")  # actually shrink the file; takes well under a second at this size


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
