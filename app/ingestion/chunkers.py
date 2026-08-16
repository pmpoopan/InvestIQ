"""Chunking strategies for ingested mutual-fund and IPO documents.

Call ``chunk_document`` with a ``ChunkingConfig`` so Phase 3 can ablate
``section_aware`` vs ``fixed_size`` and vary size/overlap without rewriting
parsers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.metadata import format_page_number, tag_metadata
from app.models.schemas import Chunk, ChunkingConfig, ParsedBlock, ParsedDocument


def chunk_document(
    parsed_document: ParsedDocument,
    config: ChunkingConfig | None = None,
    strategy: str | None = None,
) -> list[Chunk]:
    """Split a parsed document into retrieval chunks.

    ``strategy`` overrides ``config.strategy`` when provided (ablation convenience).
    """
    cfg = config or ChunkingConfig()
    if strategy:
        cfg = cfg.model_copy(update={"strategy": strategy})

    if not parsed_document.blocks:
        return []

    if cfg.strategy == "fixed_size":
        raw = _chunk_fixed_size(parsed_document, cfg)
    else:
        raw = _chunk_section_aware(parsed_document, cfg)

    numbered = _assign_chunk_ids(parsed_document, raw)
    return tag_metadata(parsed_document, numbered)


def write_chunks_jsonl(chunks: Sequence[Chunk], path: str | Path) -> None:
    """Serialize chunks to JSONL (one Chunk per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def _assign_chunk_ids(doc: ParsedDocument, chunks: list[Chunk]) -> list[Chunk]:
    stem = Path(doc.metadata.source_document).stem
    out: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        out.append(
            chunk.model_copy(
                update={"chunk_id": f"{doc.metadata.document_type}:{stem}:{i:04d}"}
            )
        )
    return out


def _chunk_section_aware(doc: ParsedDocument, cfg: ChunkingConfig) -> list[Chunk]:
    """Keep section/clause/table units together; window only when too long."""
    groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_key: tuple[str | None, str] | None = None

    for block in doc.blocks:
        key = (block.section_heading, block.block_type)
        atomic = (block.block_type == "table" and cfg.keep_tables_intact) or (
            block.block_type == "clause" and cfg.keep_clauses_intact
        )
        start_new = current and (atomic or key != current_key)
        if start_new:
            groups.append(current)
            current = []
        current.append(block)
        current_key = key

    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for group in groups:
        kind = group[0].block_type
        heading = group[0].section_heading
        if kind == "table" and cfg.keep_tables_intact:
            chunks.extend(_emit_atomic_blocks(doc, group, cfg))
        else:
            chunks.extend(_window_group(doc, group, cfg, heading, kind))
    return chunks


def _chunk_fixed_size(doc: ParsedDocument, cfg: ChunkingConfig) -> list[Chunk]:
    """Character windows over reading order; tables can stay intact."""
    chunks: list[Chunk] = []
    buffer: list[ParsedBlock] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            heading = buffer[0].section_heading
            chunks.extend(_window_group(doc, buffer, cfg, heading, "prose"))
            buffer = []

    for block in doc.blocks:
        if block.block_type == "table" and cfg.keep_tables_intact:
            flush()
            chunks.extend(_emit_atomic_blocks(doc, [block], cfg))
            continue
        buffer.append(block)
    flush()
    return chunks


def _emit_atomic_blocks(
    doc: ParsedDocument,
    blocks: list[ParsedBlock],
    cfg: ChunkingConfig,
) -> list[Chunk]:
    emitted: list[Chunk] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if len(text) < cfg.min_chunk_chars and block.block_type != "table":
            continue
        emitted.append(
            _make_chunk(
                doc,
                text=text,
                start=block.page_number,
                end=block.page_number,
                heading=block.section_heading,
                chunk_type=block.block_type,
            )
        )
    return emitted


def _window_group(
    doc: ParsedDocument,
    blocks: list[ParsedBlock],
    cfg: ChunkingConfig,
    heading: str | None,
    chunk_type: str,
) -> list[Chunk]:
    text = "\n".join(b.text.strip() for b in blocks if b.text.strip()).strip()
    if not text:
        return []
    pages = [b.page_number for b in blocks]
    start, end = min(pages), max(pages)

    intact_clause = (
        chunk_type == "clause"
        and cfg.keep_clauses_intact
        and len(text) <= cfg.max_section_chars
    )
    if intact_clause or len(text) <= cfg.chunk_size:
        if len(text) < cfg.min_chunk_chars:
            return []
        return [_make_chunk(doc, text, start, end, heading, chunk_type)]

    pieces = _split_with_overlap(text, cfg.chunk_size, cfg.chunk_overlap)
    out: list[Chunk] = []
    for piece in pieces:
        if len(piece.strip()) < cfg.min_chunk_chars:
            continue
        out.append(_make_chunk(doc, piece.strip(), start, end, heading, chunk_type))
    return out


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or len(text) <= size:
        return [text]
    overlap = max(0, min(overlap, size - 1))
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            break_at = text.rfind("\n", cursor + size // 2, end)
            if break_at == -1:
                break_at = text.rfind(" ", cursor + size // 2, end)
            if break_at > cursor:
                end = break_at
        pieces.append(text[cursor:end].strip())
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return [p for p in pieces if p]


def _make_chunk(
    doc: ParsedDocument,
    text: str,
    start: int,
    end: int,
    heading: str | None,
    chunk_type: str,
) -> Chunk:
    return Chunk(
        chunk_id="pending",
        text=text,
        source_document=doc.metadata.source_document,
        document_type=doc.metadata.document_type,
        fund_name=doc.metadata.fund_name,
        amc_name=doc.metadata.amc_name,
        page_number=format_page_number(start, end),
        section_heading=heading,
        chunk_type=chunk_type,  # type: ignore[arg-type]
    )
