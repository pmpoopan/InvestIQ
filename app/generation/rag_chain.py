"""Baseline RAG chain: retrieve -> prompt -> generate with citations.

Phase 2 will wire embeddings, the vector store, Groq/Llama generation, and
citation formatting into a single callable chain.
"""


def run_rag(query: str) -> dict:
    """Execute the baseline RAG pipeline for a single query.

    TODO(Phase 2): implement retrieval + generation chain.
    """
    pass
