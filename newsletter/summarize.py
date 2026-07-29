"""The single LLM call per issue: the model reads the subscriber profile plus the
candidate articles, picks the most relevant ones, and writes the digest copy.

Two providers, chosen automatically from whichever key .env contains
(or forced with LLM_PROVIDER=openrouter|anthropic):
- openrouter: any OpenAI-compatible model slug via OPENROUTER_MODEL
- anthropic:  claude-sonnet-5 via the official SDK with structured outputs
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

ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Tried in order if the primary free model is capacity-exhausted mid-run.
# Override with OPENROUTER_FALLBACK_MODELS (comma-separated), or set it empty to disable.
DEFAULT_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EXCERPT_CHARS = 1500


class DigestItem(BaseModel):
    article_id: int
    headline: str
    summary: str
    why_it_matters: str


class Digest(BaseModel):
    intro: str
    items: list[DigestItem]


def provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in ("anthropic", "openrouter"):
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "none"


def active_model() -> str:
    if provider() == "anthropic":
        return ANTHROPIC_MODEL
    return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)


SYSTEM_PROMPT = """You are the editor of a personalized daily AI newsletter with exactly one subscriber, described in the profile below. Your job each day: from the candidate articles, choose the ones genuinely worth this subscriber's time and write the digest.

Selection rules:
- Aim for a FULL digest of {max_items} items. Select the {max_items} most relevant candidates; only send fewer if there genuinely aren't that many with any relevance to the subscriber.
- Include every item with real relevance to the subscriber's interests or goals. Only drop items that are exact duplicates of another selected item, or completely unrelated to the profile. When unsure, include it.
- Prefer variety: mix product launches, tools, and research rather than all of one kind.
- Order items by relevance to the subscriber, most relevant first.

Writing rules:
- headline: rewrite plainly; no clickbait.
- summary: 2-4 sentences at the depth matching the subscriber's experience level for that topic. Only state facts present in the article text.
- why_it_matters: one sentence connecting the item to the subscriber's goals or stack.
- intro: 1-2 sentences framing today's issue for this subscriber.
- Match the tone in digest_preferences.

Subscriber profile:
{profile}"""

JSON_INSTRUCTIONS = """

Respond with ONLY a JSON object, no markdown fences, no commentary, exactly this shape:
{"intro": "...", "items": [{"article_id": 123, "headline": "...", "summary": "...", "why_it_matters": "..."}]}
article_id must be copied from the candidate list."""


def _candidate_block(article: Article) -> dict:
    return {
        "article_id": article.id,
        "source": article.source,
        "title": article.title,
        "url": article.url,
        "published_at": article.published_at,
        "excerpt": article.raw_text[:EXCERPT_CHARS],
    }


def _prompts(candidates: list[Article], profile: dict) -> tuple[str, str, int]:
    max_items = profile.get("digest_preferences", {}).get("max_items", 10)
    system = SYSTEM_PROMPT.format(max_items=max_items, profile=json.dumps(profile, indent=2))
    user = "Candidate articles for today's issue:\n\n" + json.dumps(
        [_candidate_block(a) for a in candidates], indent=2
    )
    return system, user, max_items


def _extract_json(text: str) -> dict:
    """Free models sometimes wrap JSON in fences or prose — take the outermost object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    return json.loads(text[start : end + 1])


class _Transient(Exception):
    """Signals 'this model is temporarily unavailable, try the next one'."""


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


def _openrouter_try_model(model: str, messages: list, headers: dict) -> tuple[Digest, dict]:
    """One model, with retry/backoff. Returns (Digest, usage) on success,
    raises _Transient to move to the next model, or RuntimeError for a
    permanent failure (bad key, bad request)."""
    payload = {"model": model, "max_tokens": 8000, "messages": messages}
    backoffs = [2, 5]
    note = "unknown error"
    for attempt in range(len(backoffs) + 1):
        try:
            response = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
        except httpx.TransportError as exc:
            note = f"connection ({type(exc).__name__})"
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt]); continue
            raise _Transient(note)

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"OpenRouter rejected the key (HTTP {response.status_code}) — check OPENROUTER_API_KEY in .env."
            )
        if response.status_code == 429 or response.status_code >= 500:
            note = f"HTTP {response.status_code}"
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt]); continue
            raise _Transient(note)
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter rejected the request (HTTP {response.status_code}) — check OPENROUTER_MODEL in .env."
            )

        data = response.json()
        err = data.get("error")
        if err:
            if _is_transient_error(err):
                note = f"upstream: {str(err)[:70]}"
                if attempt < len(backoffs):
                    time.sleep(backoffs[attempt]); continue
                raise _Transient(note)
            raise RuntimeError(f"OpenRouter error: {err}")

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
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt]); continue
            raise _Transient(note)
    raise _Transient(note)


def _openrouter_digest(candidates: list[Article], profile: dict) -> tuple[Digest, dict]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env.")
    system, user, _ = _prompts(candidates, profile)
    messages = [
        {"role": "system", "content": system + JSON_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]
    headers = {"Authorization": f"Bearer {api_key}"}

    models = [active_model()] + _fallback_models()
    last_note = ""
    for model in models:
        try:
            return _openrouter_try_model(model, messages, headers)
        except _Transient as exc:
            last_note = f"{model} ({exc})"
            log.warning("openrouter: %s unavailable, trying next model", last_note)
    raise RuntimeError(
        f"All OpenRouter models were unavailable — last: {last_note}. The free tier is likely "
        "temporarily exhausted; the next scheduled run will retry."
    )


def _anthropic_digest(candidates: list[Article], profile: dict) -> tuple[Digest, dict]:
    import anthropic

    client = anthropic.Anthropic()
    system, user, _ = _prompts(candidates, profile)
    response = client.messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=Digest,
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": ANTHROPIC_MODEL,
    }
    return response.parsed_output, usage


def write_digest(candidates: list[Article], profile: dict) -> tuple[Digest, dict]:
    """Returns (digest, usage) where usage has input/output token counts."""
    prov = provider()
    if prov == "none":
        raise RuntimeError(
            "No LLM credentials found. Copy .env.example to .env and set "
            "OPENROUTER_API_KEY (or ANTHROPIC_API_KEY). "
            "Or preview without a key: python run_issue.py --no-llm"
        )
    if prov == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set in .env.")
    if prov == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env.")

    if prov == "openrouter":
        digest, usage = _openrouter_digest(candidates, profile)
    else:
        digest, usage = _anthropic_digest(candidates, profile)

    # Keep only valid, unique article IDs. The model occasionally repeats an id;
    # a duplicate would hit the (issue_id, article_id) primary key and crash the
    # run AFTER the email already went out — so dedupe here, before delivery.
    max_items = profile.get("digest_preferences", {}).get("max_items", 10)
    valid_ids = {a.id for a in candidates}
    seen: set[int] = set()
    unique_items = []
    for item in digest.items:
        if item.article_id in valid_ids and item.article_id not in seen:
            seen.add(item.article_id)
            unique_items.append(item)
    digest.items = unique_items[:max_items]

    log.info(
        "summarize: %s chose %d items (in=%d out=%d tokens)",
        usage.get("model", active_model()), len(digest.items),
        usage["input_tokens"], usage["output_tokens"],
    )
    return digest, usage


def stub_digest(candidates: list[Article], profile: dict) -> Digest:
    """No-LLM digest for testing without an API key: top candidates as-is,
    raw excerpts instead of written summaries."""
    max_items = profile.get("digest_preferences", {}).get("max_items", 10)
    items = []
    for article in candidates[:max_items]:
        excerpt = " ".join(article.raw_text.split())
        summary = excerpt[:300] + ("…" if len(excerpt) > 300 else "")
        items.append(
            DigestItem(
                article_id=article.id,
                headline=article.title,
                summary=summary or "No article text available.",
                why_it_matters="(test mode — the LLM writes this line once your API key is set)",
            )
        )
    return Digest(
        intro="Test issue: these articles were picked by the keyword filter alone, in filter order. "
        "With an API key, the model chooses the best ones and writes real summaries.",
        items=items,
    )
