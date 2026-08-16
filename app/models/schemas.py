"""Pydantic models for ingested documents and retrieval chunks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal["sid", "factsheet", "drhp", "sebi_regulation"]
ChunkType = Literal["prose", "table", "clause"]
ChunkStrategy = Literal["section_aware", "fixed_size"]


class DocumentMetadata(BaseModel):
    """Source-document identity used to tag every chunk."""

    source_document: str
    document_type: DocumentType
    fund_name: str | None = None
    amc_name: str | None = None


class ParsedBlock(BaseModel):
    """A layout unit extracted from a PDF page (before chunking)."""

    text: str
    page_number: int
    block_type: ChunkType = "prose"
    section_heading: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class ParsedDocument(BaseModel):
    """Parser output: ordered blocks plus document-level identity."""

    metadata: DocumentMetadata
    blocks: list[ParsedBlock] = Field(default_factory=list)
    page_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    """Tunable chunking parameters for Phase 3 ablations."""

    strategy: ChunkStrategy = "section_aware"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 40
    max_section_chars: int = 3500
    keep_tables_intact: bool = True
    keep_clauses_intact: bool = True


class Chunk(BaseModel):
    """A retrieval unit with text and citation metadata."""

    chunk_id: str
    text: str
    source_document: str
    document_type: DocumentType
    fund_name: str | None = None
    amc_name: str | None = None
    page_number: str
    section_heading: str | None = None
    chunk_type: ChunkType
