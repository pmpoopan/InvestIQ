"""Metadata tagging and filename -> fund/AMC mapping."""

from __future__ import annotations

import re
from pathlib import Path

from app.models.schemas import Chunk, DocumentMetadata, DocumentType, ParsedDocument

# Explicit mapping for this corpus. Unknown files fall back to filename heuristics.
DOCUMENT_CATALOG: dict[str, dict[str, str | None]] = {
    "SID - HDFC Flexi Cap Fund dated November 21, 2025_0.pdf": {
        "fund_name": "HDFC Flexi Cap Fund",
        "amc_name": "HDFC",
        "document_type": "sid",
    },
    "SID - HDFC Mid Cap Fund dated November 21, 2025_1.pdf": {
        "fund_name": "HDFC Mid Cap Fund",
        "amc_name": "HDFC",
        "document_type": "sid",
    },
    "SID - HDFC Small Cap Fund dated November 21, 2025_0.pdf": {
        "fund_name": "HDFC Small Cap Fund",
        "amc_name": "HDFC",
        "document_type": "sid",
    },
    "sid---sbi-focused-fund.pdf": {
        "fund_name": "SBI Focused Fund",
        "amc_name": "SBI",
        "document_type": "sid",
    },
    "sid---sbi-midcap-fund.pdf": {
        "fund_name": "SBI Midcap Fund",
        "amc_name": "SBI",
        "document_type": "sid",
    },
    "sid---sbi-small-cap-fund.pdf": {
        "fund_name": "SBI Small Cap Fund",
        "amc_name": "SBI",
        "document_type": "sid",
    },
    "Fund Facts - HDFC Flexi Cap Fund_July 26.pdf": {
        "fund_name": "HDFC Flexi Cap Fund",
        "amc_name": "HDFC",
        "document_type": "factsheet",
    },
    "Fund Facts - HDFC Mid-Cap Fund_July 26.pdf": {
        "fund_name": "HDFC Mid Cap Fund",
        "amc_name": "HDFC",
        "document_type": "factsheet",
    },
    "Fund Facts - HDFC Small Cap Fund_July 26.pdf": {
        "fund_name": "HDFC Small Cap Fund",
        "amc_name": "HDFC",
        "document_type": "factsheet",
    },
    "SBI-Focused-Fund-Factsheet-May-2026.pdf": {
        "fund_name": "SBI Focused Fund",
        "amc_name": "SBI",
        "document_type": "factsheet",
    },
    "SBI-Midcap-Fund-Factsheet-May-2026.pdf": {
        "fund_name": "SBI Midcap Fund",
        "amc_name": "SBI",
        "document_type": "factsheet",
    },
    "SBI-Small-Cap-Fund-Factsheet-May-2026.pdf": {
        "fund_name": "SBI Small Cap Fund",
        "amc_name": "SBI",
        "document_type": "factsheet",
    },
    "DRHP Arjun Jewellers.pdf": {
        "fund_name": "Arjun Jewellers Limited",
        "amc_name": None,
        "document_type": "drhp",
    },
    "SEBI _ Securities and Exchange Board of India (Mutual Funds) Regulations, 2026.pdf": {
        "fund_name": None,
        "amc_name": None,
        "document_type": "sebi_regulation",
    },
}

_AMC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"hdfc", re.I), "HDFC"),
    (re.compile(r"icici", re.I), "ICICI Prudential"),
    (re.compile(r"\bsbi\b", re.I), "SBI"),
    (re.compile(r"nippon", re.I), "Nippon India"),
]


def infer_document_type(path: str | Path, explicit: DocumentType | None = None) -> DocumentType:
    if explicit:
        return explicit
    p = Path(path)
    parent = p.parent.name.lower()
    mapping: dict[str, DocumentType] = {
        "sid": "sid",
        "factsheet": "factsheet",
        "drhp": "drhp",
        "sebi": "sebi_regulation",
    }
    if parent in mapping:
        return mapping[parent]
    name = p.name.lower()
    if "factsheet" in name or "fund facts" in name:
        return "factsheet"
    if "drhp" in name or "prospectus" in name:
        return "drhp"
    if "sebi" in name:
        return "sebi_regulation"
    if "sid" in name:
        return "sid"
    raise ValueError(f"Cannot infer document_type for {p}")


def metadata_for_path(
    path: str | Path,
    document_type: DocumentType | None = None,
) -> DocumentMetadata:
    """Resolve source filename, type, fund, and AMC for a PDF path."""
    p = Path(path)
    catalog = DOCUMENT_CATALOG.get(p.name, {})
    doc_type = document_type or catalog.get("document_type")  # type: ignore[assignment]
    if not doc_type:
        doc_type = infer_document_type(p)
    fund_name = catalog.get("fund_name")
    amc_name = catalog.get("amc_name")
    if fund_name is None:
        fund_name = _heuristic_fund_name(p.stem, doc_type)
    if amc_name is None and doc_type in {"sid", "factsheet"}:
        amc_name = _heuristic_amc(p.name)
    return DocumentMetadata(
        source_document=p.name,
        document_type=doc_type,
        fund_name=fund_name,
        amc_name=amc_name,
    )


def _heuristic_amc(filename: str) -> str | None:
    for pattern, amc in _AMC_PATTERNS:
        if pattern.search(filename):
            return amc
    return None


def _heuristic_fund_name(stem: str, document_type: DocumentType) -> str | None:
    if document_type == "sebi_regulation":
        return None
    cleaned = stem.replace("---", " ").replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"(?i)^(sid|fund facts|factsheet|drhp)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\s+(factsheet|dated.*|july \d+|may \d+|november.*)$",
        "",
        cleaned,
    )
    return cleaned.title() if cleaned else None


def format_page_number(start: int, end: int | None = None) -> str:
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def tag_metadata(document: ParsedDocument, chunks: list[Chunk]) -> list[Chunk]:
    """Ensure every chunk inherits document identity (fund/AMC/source/type)."""
    tagged: list[Chunk] = []
    for chunk in chunks:
        tagged.append(
            chunk.model_copy(
                update={
                    "source_document": document.metadata.source_document,
                    "document_type": document.metadata.document_type,
                    "fund_name": chunk.fund_name or document.metadata.fund_name,
                    "amc_name": chunk.amc_name or document.metadata.amc_name,
                }
            )
        )
    return tagged
