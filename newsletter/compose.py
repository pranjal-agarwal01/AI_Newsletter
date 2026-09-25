"""Turns the structured digest into the final email bodies (HTML + plain text)."""
from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import TEMPLATES_DIR
from .models import Article
from .summarize import Digest

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)

# Byline shown in the email footer — edit these four lines to rebrand the issue.
AUTHOR = {
    "name": "Pranjal Agarwal",
    "tagline": "Automation & AI engineering",
    "portfolio": "https://portfolio-tau-five-zmhn056mhc.vercel.app/",
    "linkedin": "https://www.linkedin.com/in/pranjal-agarwal01",
}


def _domain(url: str) -> str:
    host = urlparse(url).netloc
    return host[4:] if host.startswith("www.") else host


def _edition(now: datetime) -> tuple[datetime, str]:
    """Label the issue in the reader's timezone. A run just after midnight is
    still the previous day's evening issue (scheduled runs can start late)."""
    if now.hour < 5:
        return now - timedelta(days=1), "Evening"
    return now, "Morning" if now.hour < 16 else "Evening"


def compose(digest: Digest, articles_by_id: dict[int, Article], tz: str) -> tuple[str, str, str]:
    """Returns (subject, html_body, text_body)."""
    day, edition = _edition(datetime.now(ZoneInfo(tz)))
    issue_date = day.strftime("%A, %B %d, %Y")
    subject = f"⚡ Your AI digest — {day.strftime('%b %d')} · {edition}"

    items = []
    for entry in digest.items:
        article = articles_by_id[entry.article_id]
        items.append(
            {
                "headline": entry.headline,
                "summary": entry.summary,
                "why_it_matters": entry.why_it_matters,
                "url": article.url,
                "source": article.source,
                "domain": _domain(article.url),
            }
        )

    html = _env.get_template("digest.html.j2").render(
        subject=subject, issue_date=issue_date, edition=edition,
        intro=digest.intro, items=items, author=AUTHOR,
    )

    lines = [f"Your AI digest — {issue_date} ({edition} edition)", "", digest.intro, ""]
    for i, item in enumerate(items, start=1):
        lines += [
            f"{i}. {item['headline']} ({item['source']})",
            f"   {item['summary']}",
            f"   Why it matters: {item['why_it_matters']}",
            f"   Full story: {item['url']}",
            "",
        ]
    lines += [
        "—",
        f"Curated & built by {AUTHOR['name']} — {AUTHOR['tagline']}",
        f"Portfolio: {AUTHOR['portfolio']}",
        f"LinkedIn: {AUTHOR['linkedin']}",
    ]
    text = "\n".join(lines)

    return subject, html, text
