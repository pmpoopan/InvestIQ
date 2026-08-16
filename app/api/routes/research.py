"""Research query API: POST /api/research."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.generation.rag_chain import RagUnavailableError
from app.guardrails.pipeline import run_guarded
from app.models.schemas import ResearchRequest, ResearchResponse

router = APIRouter(tags=["research"])


@router.post("/api/research", response_model=ResearchResponse)
def query_research(payload: ResearchRequest) -> ResearchResponse:
    """Run a research query with Phase 4 guardrails around the RAG pipeline."""
    try:
        return run_guarded(payload.query)
    except RagUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
