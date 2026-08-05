"""OpenAI-compatible LLM access (OpenRouter or Groq).

The RAG answer is generated via a *structured* call: we force JSON output and
validate it against a Pydantic schema. This keeps the model honest about
whether it could answer and which retrieved chunks it relied on, which in turn
lets the app attach citations deterministically from chunk metadata.

Robustness pattern (adapted from a prior project):
1. Ask for JSON mode via ``response_format={"type": "json_object"}``.
2. Parse; on malformed JSON try deterministic repair (``json_repair``).
3. On validation failure, do one bounded retry with an explicit correction note.
Transient network/rate-limit errors are retried by ``tenacity``.
"""

import json
import logging
from typing import Type, TypeVar

from json_repair import repair_json
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from rag_app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APIError)


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
)
def _chat(system_prompt: str, user_prompt: str, *, temperature: float) -> str:
    settings = get_settings()
    response = _client().chat.completions.create(
        model=settings.model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _strip_markdown_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            first_line, rest = raw.split("\n", 1)
            if first_line.strip().lower() in ("json", ""):
                raw = rest
    return raw.strip()


def call_structured(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    *,
    temperature: float = 0.1,
) -> T:
    """Call the LLM and validate its JSON reply against ``schema``."""

    def _attempt(extra_note: str = "") -> T:
        raw = _chat(system_prompt, user_prompt + extra_note, temperature=temperature)
        cleaned = _strip_markdown_fence(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = json.loads(repair_json(cleaned))
        return schema.model_validate(data)

    try:
        return _attempt()
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Structured call failed, retrying once: %s", exc)
        return _attempt(
            extra_note=(
                "\n\nYour previous reply was not valid JSON matching the required "
                "schema. Return ONLY a single valid JSON object, with no commentary "
                "and no markdown code fences."
            )
        )
