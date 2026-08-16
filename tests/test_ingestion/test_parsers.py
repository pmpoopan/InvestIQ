"""Ingestion tests: layout, tables, empty pages, and per-type metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunkers import chunk_document
from app.ingestion.parsers import (
    parse_drhp,
    parse_factsheet,
    parse_pdf,
    parse_sebi_regulation,
    parse_sid,
    _repair_clipped_text,
)
from app.models.schemas import ChunkingConfig
from tests.test_ingestion.pdf_fixtures import (
    write_drhp_pdf,
    write_drhp_toc_pdf,
    write_empty_pdf,
    write_factsheet_table_pdf,
    write_image_only_pdf,
    write_sebi_pdf,
    write_sid_pdf,
    write_two_column_pdf,
)


def test_factsheet_table_not_mangled_into_prose(tmp_path: Path) -> None:
    pdf = write_factsheet_table_pdf(tmp_path / "Fund Facts - HDFC Flexi Cap Fund_July 26.pdf")
    parsed = parse_factsheet(pdf)
    chunks = chunk_document(parsed, config=ChunkingConfig(min_chunk_chars=10))

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert table_chunks, f"expected table chunks, got types {[c.chunk_type for c in chunks]}"
    sample = table_chunks[0].text
    assert "ICICI Bank" in sample
    assert "9.18" in sample
    assert "Banks" in sample
    # Structured cells, not a single smashed token stream.
    assert "|" in sample or "\n" in sample
    smashed = "ICICI Bank Ltd.Banks9.18"
    assert smashed not in sample.replace(" ", "")


def test_multi_column_reading_order(tmp_path: Path) -> None:
    pdf = write_two_column_pdf(tmp_path / "two_col.pdf")
    parsed = parse_pdf(pdf, document_type="sid")
    text = "\n".join(b.text for b in parsed.blocks)
    assert "ALPHA_TOKEN" in text
    assert "BETA_TOKEN" in text
    assert text.index("ALPHA_TOKEN") < text.index("BETA_TOKEN")


def test_empty_and_image_only_pages_do_not_crash(tmp_path: Path) -> None:
    empty = write_empty_pdf(tmp_path / "empty.pdf")
    image_only = write_image_only_pdf(tmp_path / "image_only.pdf")

    empty_doc = parse_sid(empty)
    image_doc = parse_factsheet(image_only)

    assert empty_doc.blocks == []
    assert chunk_document(empty_doc) == []
    assert image_doc.blocks == []
    assert chunk_document(image_doc) == []
    assert empty_doc.warnings
    assert image_doc.warnings


@pytest.mark.parametrize(
    ("writer", "parser", "doc_type", "filename", "fund_name", "amc_name"),
    [
        (
            write_sid_pdf,
            parse_sid,
            "sid",
            "SID - HDFC Flexi Cap Fund dated November 21, 2025_0.pdf",
            "HDFC Flexi Cap Fund",
            "HDFC",
        ),
        (
            write_factsheet_table_pdf,
            parse_factsheet,
            "factsheet",
            "Fund Facts - HDFC Flexi Cap Fund_July 26.pdf",
            "HDFC Flexi Cap Fund",
            "HDFC",
        ),
        (
            write_drhp_pdf,
            parse_drhp,
            "drhp",
            "DRHP Arjun Jewellers.pdf",
            "Arjun Jewellers Limited",
            None,
        ),
        (
            write_sebi_pdf,
            parse_sebi_regulation,
            "sebi_regulation",
            "SEBI _ Securities and Exchange Board of India (Mutual Funds) Regulations, 2026.pdf",
            None,
            None,
        ),
    ],
)
def test_metadata_populated_per_document_type(
    tmp_path: Path,
    writer,
    parser,
    doc_type: str,
    filename: str,
    fund_name: str | None,
    amc_name: str | None,
) -> None:
    pdf = writer(tmp_path / filename)
    parsed = parser(pdf)
    chunks = chunk_document(parsed, config=ChunkingConfig(min_chunk_chars=10))
    assert chunks, f"no chunks produced for {doc_type}"
    sample = chunks[0]
    assert sample.source_document == filename
    assert sample.document_type == doc_type
    assert sample.fund_name == fund_name
    assert sample.amc_name == amc_name
    assert sample.page_number
    assert sample.chunk_type in {"prose", "table", "clause"}
    assert sample.chunk_id.startswith(f"{doc_type}:")


def test_toc_lines_are_not_treated_as_section_headings(tmp_path: Path) -> None:
    pdf = write_drhp_toc_pdf(tmp_path / "DRHP Arjun Jewellers.pdf")
    parsed = parse_drhp(pdf)
    chunks = chunk_document(parsed, config=ChunkingConfig(min_chunk_chars=10))
    risk = [c for c in chunks if "RISK_BODY_TOKEN" in c.text]
    assert risk, "expected the real Risk Factors body"
    heading = risk[0].section_heading or ""
    assert "SECTION II" in heading.upper()
    assert "SECTION VIII" not in heading.upper()
    assert "ARTICLES OF ASSOCIATION" not in heading.upper()
    poisoned = [
        c
        for c in chunks
        if c.section_heading
        and "SECTION VIII" in c.section_heading.upper()
        and "RISK_BODY_TOKEN" in c.text
    ]
    assert not poisoned


def test_clipped_table_first_letters_are_restored() -> None:
    vocab = [
        "Financial",
        "Automobile",
        "Fund",
        "Manager*",
        "Inception",
        "Investment",
        "Reinvestment",
        "JSW",
        "Bharat",
        "Exit",
        "Load",
    ]
    assert _repair_clipped_text("inancial Services", vocab) == "Financial Services"
    assert _repair_clipped_text("utomobile and Auto Components", vocab) == "Automobile and Auto Components"
    assert _repair_clipped_text("und Manager*", vocab) == "Fund Manager*"
    assert (
        _repair_clipped_text("und Manager*", ["Fund", "FUND", "Fund)", "fund"])
        == "Fund Manager*"
    )
    assert _repair_clipped_text("nception Date", vocab) == "Inception Date"
    assert _repair_clipped_text("nvestment Plans / Options", vocab) == "Investment Plans / Options"
    assert _repair_clipped_text("SW Infrastructure Ltd.", vocab) == "JSW Infrastructure Ltd."
    assert _repair_clipped_text("harat Electronics Ltd.", vocab) == "Bharat Electronics Ltd."
    assert _repair_clipped_text("$$ Exit Load", vocab) == "Exit Load"
    assert _repair_clipped_text("% to NAV", ["Auto", "to", "NAV"]) == "% to NAV"
    assert _repair_clipped_text("IT - Software", ["Exit", "IT", "Software"]) == "IT - Software"
