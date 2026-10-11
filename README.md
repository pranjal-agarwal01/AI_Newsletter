# Personalized AI Newsletter

A pipeline that collects AI news from RSS blogs, Hacker News, and arXiv, filters it against your profile, has an LLM pick and summarize the ~10 items most relevant to you, and emails the digest twice a day.

```
sources (RSS / HN / arXiv)
  -> ingestion adapters      newsletter/ingest/
  -> SQLite (dedup by URL)   newsletter/db.py -> newsletter.db
  -> selection filter        newsletter/selection.py   (deterministic, no LLM: same-story dedup,
                                                         keyword ranking, per-source caps)
  -> one LLM call            newsletter/summarize.py   (OpenRouter; free model with fallbacks)
  -> HTML email              newsletter/compose.py + deliver.py
```

## Setup (once)

1. **Python deps** (Python 3.10+):
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. **Secrets** — copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` (from https://openrouter.ai/settings/keys) — default model is
     NVIDIA Nemotron 3 Ultra (free); set `OPENROUTER_MODEL` to any slug (including Claude
     or GPT models) to change it.
   - `GMAIL_ADDRESS` + `GMAIL_APP_PASSWORD` — create the app password at
     myaccount.google.com → Security → 2-Step Verification → App passwords
   - `DIGEST_TO` — where the digest goes (defaults to your Gmail address)
3. **Personalize** `profile.json` — role, interests, goals, tone, and the `keywords`
   used to rank candidates before the LLM sees them.
4. Optionally edit `sources.yaml` to add/remove feeds or tune the caps.

## Run

```
.venv\Scripts\python run_issue.py --no-llm    # no API call: raw excerpts, writes out/digest-<date>.html
.venv\Scripts\python run_issue.py --dry-run   # full pipeline incl. the LLM, no email; writes out/digest-<date>.html
.venv\Scripts\python run_issue.py             # sends the email
.venv\Scripts\python run_issue.py --force     # sends even if the last issue was recent or this one is thin
.venv\Scripts\python run_issue.py --check     # only verifies the OpenRouter key and email login; sends nothing
```

`--dry-run` and `--no-llm` work on a throwaway copy (`out/dry-run.db`), so they never
modify the git-tracked `newsletter.db` — the cloud copy is the source of truth for
what's been sent.

Guards that keep the inbox sane:
- **Minimum gap** — a run refuses to send if the previous issue went out less than
  `min_hours_between_issues` (default 8) hours earlier, so late or manual runs can't double-send.
- **Minimum size** — if an issue would have fewer than `min_items` (default 4) stories,
  the run skips and those stories carry over to the next issue.
- **Same-story dedup** — a story already sent in the last 14 days (or queued twice) under
  a different URL, e.g. a blog post and its Hacker News link, is dropped by title.

## Schedule (GitHub Actions — runs in the cloud, PC can be off)

`.github/workflows/newsletter.yml` runs at 10:00 and 22:00 IST, installs dependencies,
runs the pipeline, and commits `newsletter.db` back to the repo so the cloud remembers
what it already sent.

One-time setup — add these **repository secrets**
(GitHub repo → Settings → Secrets and variables → Actions → New repository secret):

| Secret | Value |
|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter key |
| `GMAIL_ADDRESS` | the Gmail account that sends |
| `GMAIL_APP_PASSWORD` | the 16-char app password |
| `DIGEST_TO` | recipient address |

Test it from the repo's **Actions tab → AI Newsletter → Run workflow**.

When a run fails, the reason is posted as an annotation on the run's summary page
(visible without opening the logs). The usual cause is a revoked Gmail app password:
create a new one, confirm it locally with `run_issue.py --check`, then update the
`GMAIL_APP_PASSWORD` secret.

### On-time delivery (recommended)

GitHub's built-in cron is best-effort: in practice these runs start **2–5 hours late**
(the 10:00 issue lands around 14:30–15:30 IST, the 22:00 one after midnight). To get
them on time, have a free external scheduler press "Run workflow" for you:

1. **Create a token** — GitHub → Settings → Developer settings → Fine-grained tokens →
   Generate new token. Repository access: *only this repo*. Permissions: **Actions: Read and write**.
2. **Create two jobs on [cron-job.org](https://cron-job.org)** (free), at 10:00 and 22:00,
   timezone Asia/Kolkata:
   - URL: `https://api.github.com/repos/<owner>/<repo>/actions/workflows/newsletter.yml/dispatches`
   - Method: `POST`
   - Headers: `Authorization: Bearer <token>` and `Accept: application/vnd.github+json`
   - Body: `{"ref":"main"}`

Keep the GitHub cron as a backup: when the external trigger has already sent an issue,
the late GitHub run hits the minimum-gap guard and exits without sending.

## Cost accounting

Every sent issue logs its model and token usage in the `issues` table:

```
.venv\Scripts\python -c "import sqlite3; [print(dict(r)) for r in sqlite3.connect('newsletter.db').execute('SELECT * FROM issues').fetchall()]"
```

## Sending to a real list (Resend + your own domain)

Gmail SMTP is fine for a few recipients. For a real subscriber list, switch to
Resend — the code auto-detects it (if `RESEND_API_KEY` is set it uses Resend,
otherwise Gmail). One-time setup:

1. **Buy a domain** (Namecheap / Cloudflare / Porkbun, ~$1–12/yr).
2. **Create a Resend account** (resend.com) → add your domain → Resend shows
   SPF, DKIM, and DMARC DNS records.
3. **Paste those records** into your domain's DNS and wait for Resend to mark
   the domain "Verified".
4. **Create an API key** in Resend.
5. **Add these GitHub repository secrets** (Settings → Secrets and variables → Actions):

   | Secret | Value |
   |---|---|
   | `RESEND_API_KEY` | `re_...` from Resend |
   | `MAIL_FROM` | `Your AI Digest <news@yourdomain.com>` (on the verified domain) |
   | `MAIL_REPLY_TO` | where replies should go (optional) |
   | `MAIL_UNSUBSCRIBE` | an `https://` unsubscribe URL, or a `mailto:` (optional but recommended) |

   The moment `RESEND_API_KEY` exists, scheduled runs switch to Resend
   automatically — no code change. Remove it to fall back to Gmail.

**Before the domain is verified**, you can smoke-test locally with Resend's
sandbox: set `MAIL_FROM=onboarding@resend.dev` and send only to your own
verified email.

> Not yet built (next milestone): storage-backed subscriber management and a
> working one-click unsubscribe endpoint. Until then, recipients are the
> comma-separated `DIGEST_TO`, and `MAIL_UNSUBSCRIBE` points at a mailbox/URL
> you handle manually.

## Roadmap

Phase 2: move state off the git-tracked SQLite file to a hosted database, embedding-based
near-duplicate clustering and ranking (replacing `selection.py`), and subscriber management.
Phase 3: research agent, critic loop, click tracking, and Telegram delivery.
