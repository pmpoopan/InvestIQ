"""sentence-transformers wrapper for chunk and query embeddings."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings

# bge-small-en-v1.5 recommended query prefix (documents are embedded without it).
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embedder(model_name: str | None = None):
    settings = get_settings()
    return _load_model(model_name or settings["embedding_model"])


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Encode document chunks (no query instruction prefix)."""
    if not texts:
        return []
    model = get_embedder(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [row.tolist() for row in vectors]


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    """Encode a search query with the BGE instruction prefix."""
    model = get_embedder(model_name)
    vector = model.encode(
        _QUERY_PREFIX + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
