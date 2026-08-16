"""Pydantic models for ingested documents and retrieval chunks.

Phase 1 will define Chunk, DocumentMetadata, and related request/response
shapes used across ingestion, retrieval, generation, and evaluation.
"""

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Source-document metadata.

    TODO(Phase 1): add fields (doc type, scheme, AMC, dates, pages, etc.).
    """

    pass


class Chunk(BaseModel):
    """A retrieval unit with text, metadata, and identifiers.

    TODO(Phase 1): add fields (text, chunk_id, source, section, etc.).
    """

    pass
