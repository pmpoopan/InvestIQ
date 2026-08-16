"""Chroma wrapper: persist Phase 1 chunks and run dense top-k search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from app.config.settings import get_settings
from app.embeddings.embedder import embed_query, embed_texts
from app.models.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_BATCH = 64


def _client(path: str | None = None) -> chromadb.PersistentClient:
    settings = get_settings()
    persist = Path(path or settings["chroma_path"])
    persist.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist))


def get_collection(reset: bool = False):
    settings = get_settings()
    client = _client()
    name = settings["chroma_collection"]
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def chroma_metadata(chunk: Chunk) -> dict[str, str]:
    """Chroma only accepts str/int/float/bool; never None."""
    return {
        "chunk_id": chunk.chunk_id,
        "source_document": chunk.source_document,
        "document_type": chunk.document_type,
        "fund_name": chunk.fund_name or "",
        "amc_name": chunk.amc_name or "",
        "page_number": chunk.page_number,
        "section_heading": (chunk.section_heading or "")[:500],
        "chunk_type": chunk.chunk_type,
    }


def load_processed_chunks(processed_dir: str | Path | None = None) -> list[Chunk]:
    settings = get_settings()
    root = Path(processed_dir or settings["processed_dir"])
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def upsert_chunks(chunks: list[Chunk], reset: bool = False) -> int:
    """Embed and write chunks into Chroma. Returns the number stored."""
    if not chunks:
        return 0
    collection = get_collection(reset=reset)
    for start in range(0, len(chunks), _BATCH):
        batch = chunks[start : start + _BATCH]
        embeddings = embed_texts([c.text for c in batch])
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[chroma_metadata(c) for c in batch],
        )
        logger.info("Upserted chunks %s–%s", start + 1, start + len(batch))
    return len(chunks)


def index_processed_corpus(reset: bool = True) -> int:
    chunks = load_processed_chunks()
    logger.info("Indexing %s processed chunks", len(chunks))
    return upsert_chunks(chunks, reset=reset)


def query(query_text: str, k: int | None = None) -> list[RetrievedChunk]:
    """Dense-only similarity search (baseline: no hybrid, no rerank)."""
    settings = get_settings()
    top_k = k if k is not None else settings["top_k"]
    min_rel = settings["min_relevance"]
    collection = get_collection(reset=False)
    if collection.count() == 0:
        return []
    qvec = embed_query(query_text)
    raw = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    hits: list[RetrievedChunk] = []
    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        similarity = max(0.0, 1.0 - float(dist))
        if similarity < min_rel:
            continue
        meta = meta or {}
        chunk = Chunk(
            chunk_id=str(meta.get("chunk_id") or chunk_id),
            text=text or "",
            source_document=str(meta.get("source_document") or ""),
            document_type=meta.get("document_type") or "sid",
            fund_name=meta.get("fund_name") or None,
            amc_name=meta.get("amc_name") or None,
            page_number=str(meta.get("page_number") or ""),
            section_heading=meta.get("section_heading") or None,
            chunk_type=meta.get("chunk_type") or "prose",
        )
        hits.append(RetrievedChunk(chunk=chunk, score=similarity))
    return hits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = index_processed_corpus(reset=True)
    print(f"Indexed {n} chunks")


if __name__ == "__main__":
    main()
