"""Unit tests for baseline RAG JSON parsing and citation filtering."""

from __future__ import annotations

from app.generation.rag_chain import _NOT_FOUND, _parse_generation
from app.models.schemas import Chunk, RetrievedChunk
from app.retrieval.vector_store import chroma_metadata


def _hit(source: str, doc_type: str, page: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="x",
            text="Exit load is 1%.",
            source_document=source,
            document_type=doc_type,  # type: ignore[arg-type]
            page_number=page,
            chunk_type="prose",
        ),
        score=0.9,
    )


def test_chroma_metadata_has_no_nones() -> None:
    chunk = Chunk(
        chunk_id="sid:demo:0001",
        text="hello",
        source_document="demo.pdf",
        document_type="sid",
        fund_name=None,
        amc_name=None,
        page_number="4",
        section_heading=None,
        chunk_type="prose",
    )
    meta = chroma_metadata(chunk)
    assert None not in meta.values()
    assert meta["fund_name"] == ""
    assert meta["document_type"] == "sid"


def test_parse_generation_filters_invented_citations() -> None:
    hits = [_hit("SID - HDFC Flexi Cap Fund dated November 21, 2025_0.pdf", "sid", "5")]
    raw = """
    {
      "answer": "Exit load is 1% within 1 year.",
      "citations": [
        {"source_document": "SID - HDFC Flexi Cap Fund dated November 21, 2025_0.pdf", "document_type": "sid", "page_number": "5"},
        {"source_document": "made-up.pdf", "document_type": "sid", "page_number": "99"}
      ]
    }
    """
    parsed = _parse_generation(raw, hits)
    assert "1%" in parsed.answer
    assert len(parsed.citations) == 1
    assert parsed.citations[0].source_document.endswith("_0.pdf")


def test_parse_generation_handles_empty_json() -> None:
    parsed = _parse_generation("not json at all", [_hit("a.pdf", "sid", "1")])
    assert parsed.answer == _NOT_FOUND
    assert parsed.citations == []
