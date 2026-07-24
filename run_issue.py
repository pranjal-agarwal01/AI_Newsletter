"""Orchestrator: runs one issue of the newsletter end to end.

    python run_issue.py            # full run: ingest -> ... -> send email
    python run_issue.py --dry-run  # everything except sending; writes out/digest-<date>.html
    python run_issue.py --force    # send even if an issue already went out today
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from newsletter import db, deliver
from newsletter.compose import compose
from newsletter.config import OUT_DIR, load_profile, load_sources
from newsletter.enrich import enrich_articles
from newsletter.ingest import build_adapters
from newsletter.selection import select_candidates
from newsletter.summarize import active_model, stub_digest, write_digest

log = logging.getLogger("run_issue")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and send one newsletter issue.")
    parser.add_argument("--dry-run", action="store_true", help="skip sending; write HTML to out/")
    parser.add_argument("--force", action="store_true", help="bypass the once-per-day guard")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="test mode without an API key: raw excerpts instead of Claude summaries (implies --dry-run)",
    )
    args = parser.parse_args()
    if args.no_llm:
        args.dry_run = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sources = load_sources()
    profile = load_profile()
    conn = db.connect()

    min_gap = sources.get("min_hours_between_issues", 11)
    gap = db.hours_since_last_issue(conn)
    if not args.dry_run and not args.force and gap is not None and gap < min_gap:
        log.info(
            "Last issue went out %.1f hours ago (minimum gap: %sh). Use --force to send anyway.",
            gap, min_gap,
        )
        return 0

    log.info("--- stage 1/6: ingest ---")
    new_count = 0
    for adapter in build_adapters(sources):
        try:
            items = adapter.fetch()
        except Exception:
            log.exception("ingest: adapter '%s' failed entirely; continuing", adapter.name)
            continue
        for item in items:
            _, was_new = db.upsert_article(conn, item)
            new_count += was_new
    log.info("ingest: %d new articles stored", new_count)

    log.info("--- stage 2/6: select candidates ---")
    fresh = db.unsent_recent_articles(conn, sources.get("freshness_hours", 36))
    candidates = select_candidates(fresh, profile, sources.get("max_candidates", 25))
    if not candidates:
        log.info("No fresh candidates today — nothing to send.")
        return 0

    log.info("--- stage 3/6: enrich candidate text ---")
    enrich_articles(conn, candidates)

    if args.no_llm:
        log.info("--- stage 4/6: summarize (skipped, --no-llm test mode) ---")
        digest, usage = stub_digest(candidates, profile), {"input_tokens": 0, "output_tokens": 0}
    else:
        log.info("--- stage 4/6: summarize with %s ---", active_model())
        try:
            digest, usage = write_digest(candidates, profile)
        except RuntimeError as exc:
            log.error(str(exc))
            return 1

    if not digest.items:
        log.info("Claude found nothing worth sending today.")
        return 0

    log.info("--- stage 5/6: compose ---")
    articles_by_id = {a.id: a for a in candidates}
    subject, html, text = compose(digest, articles_by_id)

    log.info("--- stage 6/6: deliver ---")
    if args.dry_run:
        OUT_DIR.mkdir(exist_ok=True)
        out_path = OUT_DIR / f"digest-{date.today().isoformat()}.html"
        out_path.write_text(html, encoding="utf-8")
        log.info("dry run: wrote %s (no email sent, no issue recorded)", out_path)
        return 0

    deliver.send(subject, html, text)
    issue_id = db.record_issue(
        conn,
        [item.article_id for item in digest.items],
        active_model(),
        usage["input_tokens"],
        usage["output_tokens"],
    )
    log.info(
        "Issue #%d sent: %d items, %d in / %d out tokens.",
        issue_id, len(digest.items), usage["input_tokens"], usage["output_tokens"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
