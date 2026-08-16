"""Cross-encoder reranking of retrieval candidates."""

from __future__ import annotations

from functools import lru_cache

from app.models.schemas import RetrievedChunk

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_MODEL_NAME)


def rerank(
    query_text: str,
    hits: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Score (query, chunk) pairs and keep the top_k."""
    if not hits:
        return []
    model = _load_reranker()
    pairs = [(query_text, hit.chunk.text[:4000]) for hit in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
    return [
        RetrievedChunk(chunk=hit.chunk, score=float(score))
        for hit, score in ranked[:top_k]
    ]
