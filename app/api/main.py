"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.research import router as research_router

app = FastAPI(title="InvestIQ", version="0.4.0")
app.include_router(research_router)


@app.get("/health")
def health() -> dict:
    """Liveness; includes Chroma size when the index exists."""
    status: dict = {"status": "ok"}
    try:
        from app.retrieval.vector_store import get_collection

        status["chunk_count"] = get_collection().count()
    except Exception:
        status["chunk_count"] = None
    return status
