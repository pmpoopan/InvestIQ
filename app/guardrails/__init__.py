"""Decoupled safety layer: query policies, retrieval floor, then generation.

Swap the Phase 3 winner later by passing ``retrieve_fn`` / ``generate_fn`` into
``run_guarded`` — this package does not import hybrid retrieval or rerankers.
"""

from app.guardrails.pipeline import run_guarded
from app.guardrails.scoring import score_adversarial

__all__ = ["run_guarded", "score_adversarial"]
