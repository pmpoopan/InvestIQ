"""FastAPI application entrypoint.

Phase 2 will mount research routes and shared settings. Later phases add
guardrails, health checks, and production hardening.
"""

from fastapi import FastAPI

app = FastAPI(title="InvestIQ", version="0.0.0")


@app.get("/health")
def health() -> dict:
    """Liveness stub.

    TODO(Phase 2): expand with dependency checks once retrieval/generation exist.
    """
    return {"status": "ok"}
