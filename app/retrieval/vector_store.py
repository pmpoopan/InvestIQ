"""Vector store persistence and similarity search.

Phase 2 will implement a Chroma-backed store for the baseline RAG pipeline.
Later ablation work (hybrid search + rerank) will sit on top of this layer.
"""


def upsert_chunks(chunks: list) -> None:
    """Write chunk embeddings and metadata into the vector store.

    TODO(Phase 2): implement vector store upsert.
    """
    pass


def query(query_text: str, k: int = 5) -> list:
    """Retrieve nearest chunks for a query.

    TODO(Phase 2): implement dense retrieval (baseline).
    """
    pass
