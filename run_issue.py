"""Orchestrator: runs one issue of the newsletter end to end.

    python run_issue.py            # full run: ingest -> ... -> send email
    python run_issue.py --dry-run  # everything except sending, on a copy of the DB; writes out/digest-<date>.html
    python run_issue.py --no-llm   # like --dry-run, but raw excerpts instead of an LLM call
    python run_issue.py --force    # send even if the last issue was recent or this one is thin
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date

from newsletter import db, deliver
from newsletter.compose import compose
from newsletter.config import DB_PATH, OUT_DIR, load_profile, load_sources
from newsletter.enrich import enrich_articles
from newsletter.ingest import build_adapters
from newsletter.selection import select_candidates
from newsletter.summarize import active_model, stub_digest, write_digest

log = logging.getLogger("run_issue")

DEDUP_DAYS = 14      # a story sent within this many days won't be sent again under another URL
KEEP_TEXT_DAYS = 14  # article body text older than this is dropped to keep the DB small


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and send one newsletter issue.")
    parser.add_argument("--dry-run", action="store_true", help="skip sending; write HTML to out/")
    parser.add_argument("--force", action="store_true", help="bypass the min-gap and min-items guards")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="test mode without an LLM call: raw excerpts instead of summaries (implies --dry-run)",
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
    if args.dry_run:
        # Work on a copy: the real DB is git-tracked and the cloud copy is the source of truth.
        OUT_DIR.mkdir(exist_ok=True)
        db_path = shutil.copyfile(DB_PATH, OUT_DIR / "dry-run.db")
    else:
        db_path = DB_PATH
    conn = db.connect(db_path)

    min_gap = sources.get("min_hours_between_issues", 8)
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
        new_count += sum(db.upsert_article(conn, item) for item in items)
    log.info("ingest: %d new articles stored", new_count)
    db.prune_old_text(conn, KEEP_TEXT_DAYS)

    log.info("--- stage 2/6: select candidates ---")
    min_items = 1 if args.force else sources.get("min_items", 1)
    candidates = select_candidates(
        db.unsent_recent_articles(conn, sources.get("freshness_hours", 72)),
        profile,
        sent_titles=db.recently_sent_titles(conn, DEDUP_DAYS),
        max_candidates=sources.get("max_candidates", 30),
        max_per_source=sources.get("max_per_source", 10),
        source_caps=sources.get("source_caps") or {},
    )
    if len(candidates) < min_items:
        log.info("Only %d fresh candidates (minimum %d) — skipping; they carry over to the next run.",
                 len(candidates), min_items)
        return 0

    log.info("--- stage 3/6: enrich candidate text ---")
    enrich_articles(conn, candidates)

    if args.no_llm:
        log.info("--- stage 4/6: summarize (skipped, --no-llm test mode) ---")
        digest, usage = stub_digest(candidates, profile), {"input_tokens": 0, "output_tokens": 0, "model": "none"}
    else:
        log.info("--- stage 4/6: summarize with %s ---", active_model())
        try:
            digest, usage = write_digest(candidates, profile)
        except RuntimeError as exc:
            log.error(str(exc))
            return 1

    if len(digest.items) < min_items:
        log.info("The model picked only %d stories (minimum %d) — skipping; unsent stories carry over.",
                 len(digest.items), min_items)
        return 0

    log.info("--- stage 5/6: compose ---")
    articles_by_id = {a.id: a for a in candidates}
    tz = profile.get("digest_preferences", {}).get("timezone", "UTC")
    subject, html, text = compose(digest, articles_by_id, tz)

    log.info("--- stage 6/6: deliver ---")
    if args.dry_run:
        out_path = OUT_DIR / f"digest-{date.today().isoformat()}.html"
        out_path.write_text(html, encoding="utf-8")
        log.info("dry run: wrote %s — subject: %s (no email sent, real DB untouched)", out_path, subject)
        return 0

    deliver.send(subject, html, text)
    issue_id = db.record_issue(
        conn,
        [item.article_id for item in digest.items],
        usage["model"],
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
