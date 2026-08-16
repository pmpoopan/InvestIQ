"""Dump unscored baseline RAG outputs for every golden-set question."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import REPO_ROOT, get_settings
from app.generation.rag_chain import RagUnavailableError, run_rag
from app.retrieval.vector_store import query as dense_query

logger = logging.getLogger(__name__)

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.json"
OUT_PATH = REPO_ROOT / "eval" / "results" / "baseline_raw_outputs.json"


def run_eval() -> Path:
    """Run the baseline pipeline over the golden set and write raw JSON (no scores)."""
    settings = get_settings()
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = payload.get("golden_set") or []
    results: list[dict] = []
    for item in items:
        qid = item.get("id")
        question = item.get("question") or ""
        logger.info("Running %s", qid)
        retrieved = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "score": hit.score,
                "source_document": hit.chunk.source_document,
                "document_type": hit.chunk.document_type,
                "page_number": hit.chunk.page_number,
                "section_heading": hit.chunk.section_heading,
            }
            for hit in dense_query(question)
        ]
        error = None
        answer = ""
        citations: list[dict] = []
        try:
            response = run_rag(question)
            answer = response.answer
            citations = [c.model_dump() for c in response.citations]
        except RagUnavailableError as exc:
            error = str(exc)
            answer = str(exc)
        results.append(
            {
                "id": qid,
                "category": item.get("category"),
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieved": retrieved,
                "error": error,
            }
        )
        time.sleep(0.35)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "meta": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "embedding_model": settings["embedding_model"],
                    "groq_model": settings["groq_model"],
                    "top_k": settings["top_k"],
                    "min_relevance": settings["min_relevance"],
                    "scored": False,
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (%s questions)", OUT_PATH, len(results))
    return OUT_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eval()
