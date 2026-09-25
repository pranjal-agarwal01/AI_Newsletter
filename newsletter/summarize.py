"""The single LLM call per issue: the model reads the subscriber profile plus the
candidate articles, picks the most relevant ones, and writes the digest copy.

Runs on OpenRouter (any model slug via OPENROUTER_MODEL — Claude, GPT, and the
free models all go through the same OpenAI-compatible endpoint).
"""
from __future__ import annotations

import json
import logging
import os
import time

import httpx
from pydantic import BaseModel, ValidationError

from .models import Article

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Tried in order if the primary model is unavailable (capacity, rate limit, or
# retired slug). Override with OPENROUTER_FALLBACK_MODELS (comma-separated),
# or set it empty to disable.
DEFAULT_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EXCERPT_CHARS = 1500
# Reasoning models count their thinking in completion tokens; 8000 was
# occasionally within a few hundred tokens of truncating the JSON.
MAX_OUTPUT_TOKENS = 16000


class DigestItem(BaseModel):
    article_id: int
    headline: str
    summary: str
    why_it_matters: str


class Digest(BaseModel):
    intro: str
    items: list[DigestItem]


def active_model() -> str:
    return os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL


SYSTEM_PROMPT = """You are the editor of a personalized AI newsletter with exactly one subscriber, described in the profile below. From the candidate articles, choose the ones genuinely worth this subscriber's time and write the digest.

Selection rules:
- Aim for {max_items} items. Include every candidate this subscriber would plausibly want to know about — model launches and pricing, new tools and integrations, agent security incidents, practical techniques. Leave out items with no real connection to their interests or goals (general tech news, politics, research outside their stack) rather than padding the digest to reach {max_items}.
- If several candidates cover the same news (e.g. a launch post and commentary on it), include only the most useful one.
- Research papers (source "arXiv"): include at most 2, and only if they have a concrete takeaway the subscriber could apply. Never lead the issue with one.
- Prefer variety: mix product launches, tools, and techniques rather than all of one kind.
- Order items by relevance to the subscriber, most relevant first.

Writing rules:
- headline: rewrite plainly; no clickbait.
- summary: 2-3 short sentences (about 60 words max) in plain language at the subscriber's experience level for that topic. No unexplained jargon. Only state facts present in the article text.
- why_it_matters: one sentence connecting the item to the subscriber's goals or stack. Don't force a connection that isn't there.
- intro: 1-2 sentences framing this issue for the subscriber.
- Match the tone in digest_preferences.

Subscriber profile:
{profile}

Respond with ONLY a JSON object, no markdown fences, no commentary, exactly this shape:
{{"intro": "...", "items": [{{"article_id": 123, "headline": "...", "summary": "...", "why_it_matters": "..."}}]}}
article_id must be copied from the candidate list."""


def max_items(profile: dict) -> int:
    return profile.get("digest_preferences", {}).get("max_items", 10)


def _candidate_block(article: Article) -> dict:
    return {
        "article_id": article.id,
        "source": article.source,
        "title": article.title,
        "url": article.url,
        "published_at": article.published_at,
        "excerpt": article.raw_text[:EXCERPT_CHARS],
    }


def _messages(candidates: list[Article], profile: dict) -> list[dict]:
    system = SYSTEM_PROMPT.format(max_items=max_items(profile), profile=json.dumps(profile, indent=2))
    user = "Candidate articles for this issue:\n\n" + json.dumps(
        [_candidate_block(a) for a in candidates], indent=2
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(text: str) -> dict:
    """Free models sometimes wrap JSON in fences or prose — take the outermost object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    return json.loads(text[start : end + 1])


class _Unavailable(Exception):
    """Signals 'this model can't serve the request right now, try the next one'."""


_TRANSIENT_HINTS = ("exhaust", "rate", "overload", "capacity", "temporarily", "timeout", "unavailable", "try again")


def _is_transient_error(err) -> bool:
    """OpenRouter sometimes returns HTTP 200 with an error object whose `code`
    is a 5xx/429 or whose message describes a capacity/rate problem."""
    if isinstance(err, dict):
        code = err.get("code")
        if isinstance(code, int) and (code == 429 or code >= 500):
            return True
        msg = str(err.get("message", "")).lower()
    else:
        msg = str(err).lower()
    return any(hint in msg for hint in _TRANSIENT_HINTS)


def _fallback_models() -> list[str]:
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS")
    if raw is not None:
        return [m.strip() for m in raw.split(",") if m.strip()]
    primary = active_model()
    return [m for m in DEFAULT_FALLBACK_MODELS if m != primary]


def _try_model(model: str, messages: list, headers: dict) -> tuple[Digest, dict]:
    """One model, with retry/backoff. Returns (Digest, usage) on success,
    raises _Unavailable to move to the next model, or RuntimeError for a
    failure no other model can fix (bad key)."""
    payload = {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "messages": messages}
    backoffs = [2, 5]
    note = "unknown error"
    for attempt in range(len(backoffs) + 1):
        retry = attempt < len(backoffs)
        try:
            response = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
        except httpx.TransportError as exc:
            note = f"connection ({type(exc).__name__})"
            if retry:
                time.sleep(backoffs[attempt]); continue
            raise _Unavailable(note)

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"OpenRouter rejected the key (HTTP {response.status_code}) — check OPENROUTER_API_KEY."
            )
        if response.status_code == 429 or response.status_code >= 500:
            note = f"HTTP {response.status_code}"
            if retry:
                time.sleep(backoffs[attempt]); continue
            raise _Unavailable(note)
        if response.status_code >= 400:
            # e.g. a retired free-model slug (404) — retrying won't help, but another model might
            raise _Unavailable(f"HTTP {response.status_code}: {response.text[:120]}")

        data = response.json()
        err = data.get("error")
        if err:
            note = f"upstream: {str(err)[:70]}"
            if retry and _is_transient_error(err):
                time.sleep(backoffs[attempt]); continue
            raise _Unavailable(note)

        raw_usage = data.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "model": model,
        }
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        try:
            return Digest.model_validate(_extract_json(content)), usage
        except (ValueError, ValidationError):
            note = "unparseable JSON"
            if retry:
                time.sleep(backoffs[attempt]); continue
            raise _Unavailable(note)
    raise _Unavailable(note)


def write_digest(candidates: list[Article], profile: dict) -> tuple[Digest, dict]:
    """Returns (digest, usage) where usage has input/output token counts and the model used."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or preview without a key: python run_issue.py --no-llm"
        )
    messages = _messages(candidates, profile)
    headers = {"Authorization": f"Bearer {api_key}"}

    digest = usage = None
    last_note = ""
    for model in [active_model()] + _fallback_models():
        try:
            digest, usage = _try_model(model, messages, headers)
            break
        except _Unavailable as exc:
            last_note = f"{model} ({exc})"
            log.warning("openrouter: %s unavailable, trying next model", last_note)
    if digest is None:
        raise RuntimeError(
            f"All OpenRouter models were unavailable — last: {last_note}. The free tier is likely "
            "temporarily exhausted; the next scheduled run will retry."
        )

    # Keep only valid, unique article IDs. The model occasionally repeats an id;
    # a duplicate would hit the (issue_id, article_id) primary key and crash the
    # run AFTER the email already went out — so dedupe here, before delivery.
    valid_ids = {a.id for a in candidates}
    seen: set[int] = set()
    unique_items = []
    for item in digest.items:
        if item.article_id in valid_ids and item.article_id not in seen:
            seen.add(item.article_id)
            unique_items.append(item)
    digest.items = unique_items[: max_items(profile)]

    log.info(
        "summarize: %s chose %d items (in=%d out=%d tokens)",
        usage["model"], len(digest.items), usage["input_tokens"], usage["output_tokens"],
    )
    return digest, usage


def stub_digest(candidates: list[Article], profile: dict) -> Digest:
    """No-LLM digest for testing the pipeline and email layout without an API
    call: top candidates as-is, raw excerpts instead of written summaries."""
    items = []
    for article in candidates[: max_items(profile)]:
        excerpt = " ".join(article.raw_text.split())
        summary = excerpt[:300] + ("…" if len(excerpt) > 300 else "")
        items.append(
            DigestItem(
                article_id=article.id,
                headline=article.title,
                summary=summary or "No article text available.",
                why_it_matters="(test mode — the LLM writes this line in a real run)",
            )
        )
    return Digest(
        intro="Test issue: these articles were picked by the keyword filter alone, in filter order. "
        "In a real run, the model chooses the best ones and writes the summaries.",
        items=items,
    )
