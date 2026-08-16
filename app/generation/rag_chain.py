"""Baseline RAG chain: dense retrieve -> prompt -> Groq JSON answer + citations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from app.config.settings import get_settings
from app.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.models.schemas import Citation, ResearchResponse, RetrievedChunk
from app.retrieval.vector_store import query as dense_query

logger = logging.getLogger(__name__)

_NOT_FOUND = (
    "The retrieved documents do not contain enough information to answer this question. "
    "I will not guess."
)
_GROQ_DOWN = (
    "The language-model service is temporarily unavailable. "
    "Please retry in a moment."
)
_RATE_LIMITED = (
    "The language-model service is rate-limited right now. "
    "Please retry shortly."
)


class RagUnavailableError(RuntimeError):
    """Groq is down or rate-limited; the API layer maps this to HTTP 503."""


def run_rag(
    query: str,
    k: int | None = None,
    hits: list[RetrievedChunk] | None = None,
    retrieve_fn=None,
) -> ResearchResponse:
    """Execute RAG: retrieve (or use provided hits) -> Groq JSON answer + citations."""
    if hits is None:
        if retrieve_fn is not None:
            hits = retrieve_fn(query)
        else:
            hits = dense_query(query, k=k)
    if not hits:
        return ResearchResponse(answer=_NOT_FOUND, citations=[])
    try:
        payload = _generate(query, hits)
    except RateLimitError as exc:
        logger.warning("Groq rate limited: %s", exc)
        raise RagUnavailableError(_RATE_LIMITED) from exc
    except APIConnectionError as exc:
        logger.warning("Groq connection error: %s", exc)
        raise RagUnavailableError(_GROQ_DOWN) from exc
    except APIStatusError as exc:
        logger.warning("Groq status error: %s", exc)
        if exc.status_code in {429}:
            raise RagUnavailableError(_RATE_LIMITED) from exc
        if exc.status_code in {500, 502, 503, 504}:
            raise RagUnavailableError(_GROQ_DOWN) from exc
        raise
    return _parse_generation(payload, hits)


def _generate(query: str, hits: list[RetrievedChunk]) -> str:
    settings = get_settings()
    api_key = settings["groq_api_key"]
    if not api_key:
        raise RagUnavailableError("GROQ_API_KEY is not set.")
    client = Groq(api_key=api_key)
    user_msg = build_rag_prompt(query, hits)
    models = [settings["groq_model"]]
    fallback = settings["groq_fallback_model"]
    if fallback and fallback not in models:
        models.append(fallback)
    last_error: Exception | None = None
    for model in models:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or "{}"
        except APIStatusError as exc:
            last_error = exc
            body = str(exc).lower()
            decommissioned = exc.status_code == 400 and (
                "decommission" in body or "does not exist" in body or "not found" in body
            )
            if decommissioned and model != models[-1]:
                logger.warning("Model %s unavailable; trying %s", model, models[-1])
                continue
            raise
    assert last_error is not None
    raise last_error


def _parse_generation(raw: str, hits: list[RetrievedChunk]) -> ResearchResponse:
    data = _extract_json(raw)
    answer = str(data.get("answer") or "").strip() or _NOT_FOUND
    allowed = {
        (h.chunk.source_document, h.chunk.document_type, h.chunk.page_number)
        for h in hits
    }
    citations: list[Citation] = []
    for item in data.get("citations") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_document") or "").strip()
        doc_type = str(item.get("document_type") or "").strip()
        page = str(item.get("page_number") or "").strip()
        if (source, doc_type, page) not in allowed:
            # Keep a citation if source+page match even if type is slightly off.
            match = next(
                (
                    (s, t, p)
                    for (s, t, p) in allowed
                    if s == source and p == page
                ),
                None,
            )
            if not match:
                continue
            source, doc_type, page = match
        citations.append(
            Citation(source_document=source, document_type=doc_type, page_number=page)
        )
    return ResearchResponse(answer=answer, citations=_dedupe(citations))


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _dedupe(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Citation] = []
    for cit in citations:
        key = (cit.source_document, cit.document_type, cit.page_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(cit)
    return out
