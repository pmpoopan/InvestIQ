"""Research query API routes.

Phase 2 will expose a query endpoint that runs the baseline RAG chain and
returns an answer plus citations.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query")
def query_research() -> dict:
    """Run a research query over ingested documents.

    TODO(Phase 2): accept a query payload and call the RAG chain.
    """
    pass
