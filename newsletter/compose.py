"""Turns the structured digest into the final email bodies (HTML + plain text)."""
from __future__ import annotations

from datetime import date
from urllib.parse import urlparse

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


def compose(digest: Digest, articles_by_id: dict[int, Article]) -> tuple[str, str, str]:
    """Returns (subject, html_body, text_body)."""
    issue_date = date.today().strftime("%A, %B %d, %Y")
    subject = f"⚡ Your AI digest — {date.today().strftime('%b %d')}"

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
        subject=subject, issue_date=issue_date, intro=digest.intro, items=items, author=AUTHOR
    )

    lines = [f"Your AI digest — {issue_date}", "", digest.intro, ""]
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
