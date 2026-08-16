"""RAGAS-style metrics implemented against Groq + local embeddings.

ragas 0.4.3 (and 0.2.15) currently fail to import: they pull
``langchain_community.chat_models.vertexai``, which was removed from
langchain-community. DeepEval is the closest actively maintained alternative
if you want a packaged library later. These functions follow the published
RAGAS definitions (faithfulness, context precision, context recall, answer
relevancy) so the ablation table stays comparable to RAGAS papers.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from app.config.settings import get_settings
from app.embeddings.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

PLACEHOLDER_MARKERS = (
    "[FILL IN",
    "FILL IN FROM",
    "FILL IN once",
    "FILL IN —",
    "[FILL IN FROM ACTUAL",
)


def is_placeholder_answer(expected: str | None) -> bool:
    text = (expected or "").strip()
    if not text:
        return True
    upper = text.upper()
    return any(marker.upper() in upper for marker in PLACEHOLDER_MARKERS)


def _client() -> Groq:
    settings = get_settings()
    key = settings["groq_api_key"]
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def _chat_json(system: str, user: str, retries: int = 5) -> dict[str, Any]:
    settings = get_settings()
    models = [settings["groq_model"]]
    fallback = settings["groq_fallback_model"]
    if fallback and fallback not in models:
        models.append(fallback)
    client = _client()
    last: Exception | None = None
    for attempt in range(retries):
        for model in models:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = completion.choices[0].message.content or "{}"
                return _extract_json(raw)
            except RateLimitError as exc:
                last = exc
                wait = min(20.0, 2.0 ** attempt + 1)
                logger.warning("Judge rate-limited; sleeping %.1fs", wait)
                time.sleep(wait)
                break
            except APIStatusError as exc:
                last = exc
                body = str(exc).lower()
                if exc.status_code == 429:
                    time.sleep(min(20.0, 2.0 ** attempt + 1))
                    break
                decommissioned = exc.status_code == 400 and (
                    "decommission" in body or "does not exist" in body
                )
                if decommissioned and model != models[-1]:
                    continue
                raise
            except APIConnectionError as exc:
                last = exc
                time.sleep(2)
                break
    if last:
        raise last
    return {}


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


def _join_contexts(contexts: list[str]) -> str:
    parts = []
    for i, ctx in enumerate(contexts, start=1):
        parts.append(f"[{i}] {ctx[:2500]}")
    return "\n\n".join(parts)


def faithfulness(answer: str, contexts: list[str]) -> float | None:
    """Fraction of answer claims supported by retrieved context (RAGAS)."""
    if not (answer or "").strip():
        return None
    if not contexts:
        return 0.0
    data = _chat_json(
        "You evaluate RAG faithfulness. Return JSON only.",
        (
            "Break the ANSWER into atomic factual claims. For each claim, set supported=true "
            "only if the CONTEXT entails it. Ignore advice-refusals that do not assert facts; "
            "if there are no factual claims, return {\"claims\": []}.\n\n"
            f"CONTEXT:\n{_join_contexts(contexts)}\n\nANSWER:\n{answer}\n\n"
            'JSON schema: {"claims": [{"text": str, "supported": bool}]}'
        ),
    )
    claims = data.get("claims") or []
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
    return supported / len(claims)


def answer_relevancy(question: str, answer: str) -> float | None:
    """RAGAS-style: generate questions from the answer, cosine-sim to the original."""
    if not (answer or "").strip():
        return None
    data = _chat_json(
        "You reverse-engineer questions from an answer. Return JSON only.",
        (
            "Generate 3 questions that this ANSWER would be a direct response to.\n\n"
            f"ANSWER:\n{answer}\n\n"
            'JSON schema: {"questions": [str, str, str]}'
        ),
    )
    generated = [q for q in (data.get("questions") or []) if isinstance(q, str) and q.strip()]
    if not generated:
        return None
    q_vec = embed_query(question)
    other_vecs = embed_texts(generated)
    sims = [_cosine(q_vec, v) for v in other_vecs]
    return sum(sims) / len(sims)


def context_precision(question: str, contexts: list[str], reference: str | None = None) -> float | None:
    """RAGAS precision@k over retrieved chunks judged relevant to the question."""
    if not contexts:
        return 0.0
    ref_note = f"\nGround-truth answer (if useful): {reference}" if reference else ""
    data = _chat_json(
        "You judge whether retrieved passages help answer a question. Return JSON only.",
        (
            "For each CONTEXT chunk, relevant=true if it contains information that helps "
            "answer QUESTION (or supports a correct refusal when the question is out of scope).\n\n"
            f"QUESTION:\n{question}{ref_note}\n\n"
            f"CONTEXTS:\n{_join_contexts(contexts)}\n\n"
            'JSON schema: {"relevant": [bool, ...]}  (one bool per context, same order)'
        ),
    )
    flags = data.get("relevant")
    if not isinstance(flags, list) or len(flags) != len(contexts):
        flags = [False] * len(contexts)
    flags = [bool(x) for x in flags[: len(contexts)]]
    return _precision_at_k(flags)


def context_recall(reference: str, contexts: list[str]) -> float | None:
    """RAGAS: fraction of ground-truth statements attributable to retrieved context."""
    if not (reference or "").strip():
        return None
    if not contexts:
        return 0.0
    data = _chat_json(
        "You evaluate retrieval recall against a ground-truth answer. Return JSON only.",
        (
            "Split GROUND_TRUTH into atomic statements. attributed=true if some CONTEXT "
            "supports that statement.\n\n"
            f"GROUND_TRUTH:\n{reference}\n\n"
            f"CONTEXT:\n{_join_contexts(contexts)}\n\n"
            'JSON schema: {"statements": [{"text": str, "attributed": bool}]}'
        ),
    )
    statements = data.get("statements") or []
    if not statements:
        return None
    hit = sum(1 for s in statements if isinstance(s, dict) and s.get("attributed"))
    return hit / len(statements)


def _precision_at_k(relevance: list[bool]) -> float:
    """Mean of precision@i at each relevant rank (RAGAS context precision)."""
    if not any(relevance):
        return 0.0
    running = 0
    scores: list[float] = []
    for i, rel in enumerate(relevance, start=1):
        if rel:
            running += 1
            scores.append(running / i)
    return sum(scores) / len(scores)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, min(1.0, dot / (na * nb)))


def compute_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None,
) -> dict[str, float | None | bool]:
    """Score one example. Placeholder GT skips context_recall (and GT-conditioned precision)."""
    placeholder = is_placeholder_answer(expected_answer)
    reference = None if placeholder else expected_answer
    scores: dict[str, float | None | bool] = {
        "no_ground_truth_answer_yet": placeholder,
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "answer_relevancy": None,
    }
    try:
        scores["faithfulness"] = faithfulness(answer, contexts)
    except Exception as exc:
        logger.warning("faithfulness failed: %s", exc)
    time.sleep(0.6)
    try:
        scores["context_precision"] = context_precision(question, contexts, reference)
    except Exception as exc:
        logger.warning("context_precision failed: %s", exc)
    time.sleep(0.6)
    try:
        scores["answer_relevancy"] = answer_relevancy(question, answer)
    except Exception as exc:
        logger.warning("answer_relevancy failed: %s", exc)
    if not placeholder:
        time.sleep(0.6)
        try:
            scores["context_recall"] = context_recall(expected_answer or "", contexts)
        except Exception as exc:
            logger.warning("context_recall failed: %s", exc)
    return scores
