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
- Pick at most {max_items} items; fewer is fine on a slow news day. Skip anything stale, hype-only, or irrelevant to the profile.
- Prefer variety: not all papers, not all product launches.
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


def _post_with_retries(payload: dict, headers: dict) -> httpx.Response:
    """POST to OpenRouter, retrying transient failures (DNS/connect/timeouts,
    429 rate limits, 5xx) with backoff. Raises RuntimeError with a plain
    message when all attempts fail — run_issue.py prints it as one line."""
    backoffs = [2, 5]
    last_note = ""
    for attempt in range(len(backoffs) + 1):
        try:
            response = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
            if response.status_code == 429 or response.status_code >= 500:
                last_note = f"HTTP {response.status_code}"
                raise httpx.TransportError(last_note)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"OpenRouter rejected the request (HTTP {exc.response.status_code}) — "
                "check OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
            ) from exc
        except httpx.TransportError as exc:
            last_note = last_note or f"{type(exc).__name__}: {exc}"
            if attempt < len(backoffs):
                wait = backoffs[attempt]
                log.warning("openrouter: attempt %d failed (%s), retrying in %ds", attempt + 1, last_note, wait)
                time.sleep(wait)
                last_note = ""
            else:
                if "429" in str(exc):
                    raise RuntimeError(
                        "OpenRouter rate limit hit (HTTP 429) — the free tier allows a limited "
                        "number of requests per day. Try again later or add credits."
                    ) from exc
                raise RuntimeError(
                    f"Could not reach OpenRouter after {len(backoffs) + 1} attempts "
                    f"({type(exc).__name__}) — check your internet connection and try again."
                ) from exc
    raise RuntimeError("unreachable")


def _openrouter_digest(candidates: list[Article], profile: dict) -> tuple[Digest, dict]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = active_model()
    system, user, _ = _prompts(candidates, profile)
    payload = {
        "model": model,
        "max_tokens": 8000,
        "messages": [
            {"role": "system", "content": system + JSON_INSTRUCTIONS},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_error: Exception | None = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in (1, 2):
        response = _post_with_retries(payload, headers)
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        raw_usage = data.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
        }
        content = data["choices"][0]["message"]["content"] or ""
        try:
            return Digest.model_validate(_extract_json(content)), usage
        except (ValueError, ValidationError) as exc:
            last_error = exc
            log.warning("openrouter: attempt %d returned unparseable JSON, retrying", attempt)
    raise RuntimeError(f"OpenRouter model {model} did not return valid digest JSON: {last_error}")


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
    if prov == "openrouter":
        digest, usage = _openrouter_digest(candidates, profile)
    else:
        digest, usage = _anthropic_digest(candidates, profile)

    max_items = profile.get("digest_preferences", {}).get("max_items", 10)
    valid_ids = {a.id for a in candidates}
    digest.items = [i for i in digest.items if i.article_id in valid_ids][:max_items]

    log.info(
        "summarize: %s chose %d items (in=%d out=%d tokens)",
        active_model(), len(digest.items), usage["input_tokens"], usage["output_tokens"],
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
