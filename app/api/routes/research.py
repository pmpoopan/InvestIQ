"""Research query API: POST /api/research."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.generation.rag_chain import RagUnavailableError, run_rag
from app.models.schemas import ResearchRequest, ResearchResponse

router = APIRouter(tags=["research"])


@router.post("/api/research", response_model=ResearchResponse)
def query_research(payload: ResearchRequest) -> ResearchResponse:
    """Run a research query over ingested documents."""
    try:
        return run_rag(payload.query)
    except RagUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
