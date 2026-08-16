"""BM25 + reciprocal rank fusion for hybrid retrieval."""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk, RetrievedChunk
from app.retrieval.vector_store import query as dense_query

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 over a chunk list (same corpus as a Chroma collection)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens) if chunks else None

    def query(self, query_text: str, k: int = 15) -> list[RetrievedChunk]:
        if not self.chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query_text))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        hits: list[RetrievedChunk] = []
        max_score = max((abs(s) for _, s in ranked), default=0.0) or 1.0
        for idx, score in ranked:
            hits.append(
                RetrievedChunk(chunk=self.chunks[idx], score=float(score) / max_score)
            )
        return hits


def reciprocal_rank_fusion(
    lists: list[list[RetrievedChunk]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Combine ranked lists with RRF. Final score is the fused rank score."""
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}
    for ranked in lists:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.chunk.chunk_id
            fused[cid] += 1.0 / (rrf_k + rank)
            by_id[cid] = hit
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(chunk=by_id[cid].chunk, score=score) for cid, score in ordered
    ]


def hybrid_query(
    query_text: str,
    bm25: BM25Index,
    collection_name: str,
    top_k: int = 5,
    candidate_k: int = 15,
) -> list[RetrievedChunk]:
    dense_hits = dense_query(
        query_text,
        k=candidate_k,
        collection_name=collection_name,
        min_relevance=0.0,
    )
    sparse_hits = bm25.query(query_text, k=candidate_k)
    return reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
