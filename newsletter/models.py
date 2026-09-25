"""Shared data shapes passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawItem:
    """One item as returned by a source adapter, before storage."""

    source: str
    title: str
    url: str
    published_at: datetime | None = None
    raw_text: str = ""


@dataclass
class Article:
    """One stored article row, as read back from SQLite."""

    id: int
    source: str
    title: str
    url: str
    published_at: str | None
    raw_text: str
