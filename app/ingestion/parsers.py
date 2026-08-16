"""PDF text extraction per document type (SID, factsheet, DRHP, SEBI).

PyMuPDF is used for page-level extraction. Tables are kept as delimited
blocks instead of being flattened into prose. Empty and image-only pages
yield no blocks and do not raise.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pymupdf

from app.ingestion.metadata import metadata_for_path
from app.models.schemas import DocumentType, ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

HeadingFn = Callable[[str], bool]

_SID_HEADING = re.compile(
    r"^(?:"
    r"SECTION\s+[IVXLC]+\b.*"
    r"|PART\s+[IVXLC]+\.\s+.+"
    r"|[A-H]\.\s+[A-Z].{6,}"
    r"|(?:XXV|XXIV|XXIII|XXII|XXI|XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\.\s+\S.+"
    r")$",
    re.IGNORECASE,
)

_DRHP_SECTION = re.compile(r"^SECTION\s+[IVXLC]+\b.*", re.IGNORECASE)
_DRHP_NAMED = re.compile(
    r"^(?:"
    r"DEFINITIONS AND ABBREVIATIONS"
    r"|FORWARD-LOOKING STATEMENTS"
    r"|RISK FACTORS"
    r"|OBJECTS OF THE (?:ISSUE|OFFER)"
    r"|OUR BUSINESS"
    r"|INDUSTRY OVERVIEW"
    r"|HISTORY AND CERTAIN CORPORATE MATTERS"
    r"|OUR MANAGEMENT"
    r"|FINANCIAL (?:INFORMATION|STATEMENTS)"
    r"|LEGAL AND OTHER INFORMATION"
    r"|ISSUE RELATED INFORMATION"
    r"|OTHER INFORMATION"
    r")\b.*$",
    re.IGNORECASE,
)

_SEBI_CHAPTER = re.compile(r"^CHAPTER\s+[IVXLC\d]+\b.*", re.IGNORECASE)
_SEBI_REG = re.compile(
    r"^(?:Regulation|Reg\.)\s+\d+\b.*",
    re.IGNORECASE,
)
_SEBI_CLAUSE = re.compile(r"^\d+\.\s+\S+.*")

_FACTSHEET_HEADING = re.compile(
    r"^(?:"
    r"Investment Objective"
    r"|Investment Strategy"
    r"|Fund Details"
    r"|Quantitative Data"
    r"|AUM\b.*"
    r"|Top 10 .+"
    r"|Portfolio Classification.+"
    r"|Asset Allocation.+"
    r"|Fund Manager.+"
    r"|Performance.+"
    r"|NAV\b.*"
    r")$",
    re.IGNORECASE,
)

_HEADERISH = re.compile(
    r"(?i)^(page\s+\d+|\d+\s*/\s*\d+|\d{1,3}|.*-SID|scheme information document)$"
)
_TOC_DOTS = re.compile(r"\.{3,}|(\.\s){4,}")
_TOC_LEADERS = re.compile(r"^[\.\s]{8,}\d{1,4}\s*$")
_TOC_TRAILING_PAGE = re.compile(r"\s{2,}\d{1,4}\s*$")
_COMMON_SHORT = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY",
    "CAN", "HAS", "WAS", "OUR", "ITS", "THIS", "THAT", "WITH", "FROM",
    "IT", "ID", "NAV", "AUM", "TER", "SIP", "STP", "SWP", "AMC",
}
_FUNCTION_WORDS = {
    "a", "an", "as", "at", "be", "by", "do", "for", "from", "if", "in",
    "is", "it", "no", "of", "on", "or", "so", "the", "to", "up", "us",
    "we", "and", "are", "but", "not", "you", "all", "any", "can", "has",
    "was", "our", "its", "with", "this", "that",
}


def parse_pdf(
    path: str | Path,
    document_type: DocumentType | None = None,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Dispatch to the parser for ``document_type`` (inferred from path if omitted)."""
    path = Path(path)
    meta = metadata_for_path(path, document_type)
    parsers = {
        "sid": parse_sid,
        "factsheet": parse_factsheet,
        "drhp": parse_drhp,
        "sebi_regulation": parse_sebi_regulation,
    }
    return parsers[meta.document_type](path, max_pages=max_pages)


def parse_sid(path: str | Path, max_pages: int | None = None) -> ParsedDocument:
    """Clause/section-heavy SID: group by numbered/lettered headings; keep tables."""
    return _parse_with_headings(
        path,
        document_type="sid",
        is_heading=_is_sid_heading,
        detect_tables=True,
        clause_mode=False,
        max_pages=max_pages,
    )


def parse_factsheet(path: str | Path, max_pages: int | None = None) -> ParsedDocument:
    """Dense tabular factsheet: tables as structured chunks, leftover KV/prose tagged."""
    return _parse_with_headings(
        path,
        document_type="factsheet",
        is_heading=_is_factsheet_heading,
        detect_tables=True,
        clause_mode=False,
        prefer_tables=True,
        max_pages=max_pages,
    )


def parse_drhp(path: str | Path, max_pages: int | None = None) -> ParsedDocument:
    """Long-form DRHP narrative with SECTION headers that may span many pages."""
    return _parse_with_headings(
        path,
        document_type="drhp",
        is_heading=_is_drhp_heading,
        detect_tables=True,
        clause_mode=False,
        max_pages=max_pages,
    )


def parse_sebi_regulation(path: str | Path, max_pages: int | None = None) -> ParsedDocument:
    """Clause-numbered SEBI text; falls back to paragraphs if numbering is absent."""
    return _parse_with_headings(
        path,
        document_type="sebi_regulation",
        is_heading=_is_sebi_heading,
        detect_tables=False,
        clause_mode=True,
        max_pages=max_pages,
    )


def _parse_with_headings(
    path: str | Path,
    document_type: DocumentType,
    is_heading: HeadingFn,
    detect_tables: bool,
    clause_mode: bool,
    prefer_tables: bool = False,
    max_pages: int | None = None,
) -> ParsedDocument:
    path = Path(path)
    meta = metadata_for_path(path, document_type)
    warnings: list[str] = []
    blocks: list[ParsedBlock] = []

    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        n_pages = page_count if max_pages is None else min(page_count, max_pages)
        heading: str | None = None
        text_pages = 0

        for i in range(n_pages):
            page = doc[i]
            page_no = i + 1
            page_units = _extract_page_units(page, detect_tables=detect_tables)
            if not page_units:
                continue
            text_pages += 1

            for unit in page_units:
                text = _clean_text(unit["text"])
                if not text or _HEADERISH.match(text.strip()):
                    continue
                utype = unit["type"]
                if utype == "table":
                    section_for_table = heading
                    if prefer_tables:
                        maybe = _first_line_heading(text, is_heading)
                        if maybe:
                            section_for_table = maybe
                    blocks.append(
                        ParsedBlock(
                            text=_format_structured(text),
                            page_number=page_no,
                            block_type="table",
                            section_heading=section_for_table,
                            bbox=unit.get("bbox"),
                        )
                    )
                    continue

                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                buffer: list[str] = []

                def flush_prose() -> None:
                    nonlocal buffer
                    joined = "\n".join(buffer).strip()
                    buffer = []
                    if not joined:
                        return
                    btype: str = "clause" if clause_mode and heading and _SEBI_REG.match(heading) else "prose"
                    if clause_mode and heading and (_SEBI_REG.match(heading) or _SEBI_CLAUSE.match(heading)):
                        btype = "clause"
                    blocks.append(
                        ParsedBlock(
                            text=joined,
                            page_number=page_no,
                            block_type=btype,  # type: ignore[arg-type]
                            section_heading=heading,
                            bbox=unit.get("bbox"),
                        )
                    )

                for idx, line in enumerate(lines):
                    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
                    if _is_toc_entry(line, nxt):
                        buffer.append(line)
                        continue
                    if is_heading(line):
                        flush_prose()
                        heading = _normalize_heading(line)
                        continue
                    if clause_mode and _SEBI_CLAUSE.match(line) and not _SEBI_REG.match(line):
                        flush_prose()
                        heading = _normalize_heading(line[:80])
                        buffer.append(line)
                        continue
                    buffer.append(line)
                flush_prose()

        if text_pages == 0:
            warnings.append(
                f"{path.name} produced no extractable text "
                "(empty or image-only PDF; OCR is not applied in Phase 1)."
            )

    if prefer_tables:
        blocks = _merge_and_promote_factsheet(blocks)

    return ParsedDocument(
        metadata=meta,
        blocks=_drop_tiny_noise(blocks),
        page_count=page_count,
        warnings=warnings,
    )


def _extract_page_units(page: pymupdf.Page, detect_tables: bool) -> list[dict]:
    """Return reading-order units: tables plus leftover text blocks."""
    text = page.get_text("text") or ""
    if not text.strip():
        return []

    tables: list[dict] = []
    table_rects: list[pymupdf.Rect] = []
    if detect_tables:
        try:
            found = page.find_tables()
            words = page.get_text("words")
            vocab = [w[4] for w in words if w[4].strip()]
            for table in found.tables:
                rendered = _table_to_text(table.extract(), vocab=vocab)
                if not rendered.strip():
                    continue
                bbox = tuple(table.bbox)
                tables.append({"text": rendered, "type": "table", "bbox": bbox})
                table_rects.append(pymupdf.Rect(bbox))
        except Exception as exc:  # layout can fail on odd pages; skip tables
            logger.debug("table detection skipped on page %s: %s", page.number, exc)

    raw_blocks = page.get_text("dict").get("blocks", [])
    text_units: list[dict] = []
    for block in raw_blocks:
        if block.get("type") != 0:
            continue
        rect = pymupdf.Rect(block["bbox"])
        if any(_overlap_ratio(rect, tr) > 0.45 for tr in table_rects):
            continue
        body = _dict_block_text(block).strip()
        if not body:
            continue
        text_units.append({"text": body, "type": "prose", "bbox": tuple(block["bbox"])})

    ordered_text = _reading_order(text_units, page.rect.width)
    combined = ordered_text + tables
    combined.sort(key=lambda u: (_band(u["bbox"][1]), u["bbox"][0] if u["bbox"] else 0))
    return combined


def _reading_order(units: list[dict], page_width: float) -> list[dict]:
    """Left-to-right columns, top-to-bottom within a column, when layout is multi-column."""
    if len(units) < 4:
        return sorted(units, key=lambda u: (_band(u["bbox"][1]), u["bbox"][0]))

    left_only: list[dict] = []
    right_only: list[dict] = []
    spanning: list[dict] = []
    mid = page_width * 0.5
    for unit in units:
        x0, _, x1, _ = unit["bbox"]
        if x1 < mid + page_width * 0.05:
            left_only.append(unit)
        elif x0 > mid - page_width * 0.05:
            right_only.append(unit)
        else:
            spanning.append(unit)

    if len(left_only) >= 2 and len(right_only) >= 2 and len(spanning) <= len(units) * 0.35:
        def by_y(u: dict) -> tuple[int, float]:
            return _band(u["bbox"][1]), u["bbox"][0]

        return sorted(spanning, key=by_y) + sorted(left_only, key=by_y) + sorted(right_only, key=by_y)

    return sorted(units, key=lambda u: (_band(u["bbox"][1]), u["bbox"][0]))


def _band(y: float, size: float = 8.0) -> int:
    return int(y / size)


def _overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = a & b
    area = abs(a)
    if inter.is_empty or area == 0:
        return 0.0
    return abs(inter) / area


def _dict_block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        parts = [span.get("text", "") for span in line.get("spans", [])]
        joined = "".join(parts).strip()
        if joined:
            lines.append(joined)
    return "\n".join(lines)


def _table_to_text(rows: list[list[str | None]], vocab: list[str] | None = None) -> str:
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [
            _repair_clipped_text(" ".join((cell or "").split()), vocab or []).strip()
            for cell in row
        ]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    normalized = [r + [""] * (width - len(r)) for r in cleaned]
    lines = [" | ".join(r) for r in normalized]
    return "[TABLE]\n" + "\n".join(lines)


def _format_structured(text: str) -> str:
    if text.startswith("[TABLE]"):
        return text
    return "[TABLE]\n" + text


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_toc_entry(line: str, next_line: str = "") -> bool:
    """True for table-of-contents rows (dotted leaders / page numbers), not real headings."""
    compact = line.strip()
    nxt = next_line.strip()
    if not compact:
        return False
    if _TOC_DOTS.search(compact) or _TOC_LEADERS.match(compact):
        return True
    if _TOC_LEADERS.match(nxt) or (_TOC_DOTS.search(nxt) and re.search(r"\d{1,4}\s*$", nxt)):
        return True
    if re.match(r"(?i)^section\s+[ivxlc]+\b", compact) and _TOC_TRAILING_PAGE.search(compact):
        return True
    return False


def _looks_clipped_token(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    if not letters:
        if re.search(r"\d", token):
            return False
        if re.fullmatch(r"[/\\\&+\-–,;:.()%₹`]+", token):
            return False
        return True
    if letters.lower() in _FUNCTION_WORDS:
        return False
    if letters.upper() in _COMMON_SHORT:
        return False
    if letters[0].islower():
        return True
    if letters.isupper() and 1 <= len(letters) <= 3:
        return True
    return False


def _complete_clipped_token(token: str, vocab: list[str]) -> str:
    letters = re.sub(r"[^A-Za-z]", "", token)
    if len(letters) < 2:
        stripped = re.sub(r"^[^A-Za-z]+", "", token).strip()
        return stripped
    scored: list[tuple[int, str]] = []
    for word in vocab:
        w_letters = re.sub(r"[^A-Za-z]", "", word)
        extra = len(w_letters) - len(letters)
        if extra < 1 or extra > 3:
            continue
        if w_letters.lower().endswith(letters.lower()) and w_letters.lower() != letters.lower():
            if not word[:1].isupper():
                continue
            if letters.isupper() and extra != 1:
                continue
            scored.append((extra, word))
    if not scored:
        return token
    best_extra = min(extra for extra, _ in scored)
    matches: dict[str, str] = {}
    for extra, word in scored:
        if extra != best_extra:
            continue
        cleaned = re.sub(r"[^A-Za-z]", "", word)
        key = cleaned.lower()
        current = matches.get(key)
        if current is None or (current.isupper() and not cleaned.isupper()):
            matches[key] = cleaned
    if len(matches) != 1:
        return token
    repaired = next(iter(matches.values()))
    suffix = token[len(letters) :] if token.startswith(letters) else re.sub(r"^.*?[A-Za-z]+", "", token)
    if suffix and not repaired.endswith(suffix):
        return repaired + suffix
    return repaired


def _repair_clipped_text(text: str, vocab: list[str]) -> str:
    """Restore first letters clipped by table cell bounds using full page words."""
    if not text or not vocab:
        return text
    lines_out: list[str] = []
    for line in text.splitlines() or [text]:
        tokens = line.split()
        if not tokens:
            lines_out.append(line)
            continue
        fixed = [
            _complete_clipped_token(tok, vocab) if _looks_clipped_token(tok) else tok
            for tok in tokens
        ]
        lines_out.append(" ".join(tok for tok in fixed if tok))
    return "\n".join(lines_out)


def _normalize_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" .:-")


def _first_line_heading(text: str, is_heading: HeadingFn) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[TABLE]"):
            continue
        if is_heading(line):
            return _normalize_heading(line)
    return None


def _is_sid_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) > 120 or _is_toc_entry(compact):
        return False
    return bool(_SID_HEADING.match(compact))


def _is_drhp_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) > 100 or _is_toc_entry(compact):
        return False
    if _DRHP_SECTION.match(compact) or _DRHP_NAMED.match(compact):
        return True
    if compact.isupper() and 16 <= len(compact) <= 80 and compact.replace(" ", "").isalpha():
        return True
    return False


def _is_sebi_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(_SEBI_CHAPTER.match(compact) or _SEBI_REG.match(compact))


def _is_factsheet_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) > 80:
        return False
    return bool(_FACTSHEET_HEADING.match(compact))


_PCT_OR_NUM = re.compile(r"^[₹`INR$]?\s*-?[\d,]+(?:\.\d+)?\s*%?$")


def _merge_and_promote_factsheet(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Join adjacent 3-line holdings blocks, then format them as tables."""
    merged: list[ParsedBlock] = []
    buf: list[ParsedBlock] = []

    def flush() -> None:
        if not buf:
            return
        if len(buf) == 1:
            merged.append(_promote_factsheet_grids(buf[0]))
        else:
            combined = buf[0].model_copy(
                update={"text": "\n".join(b.text for b in buf)}
            )
            merged.append(_promote_factsheet_grids(combined))
        buf.clear()

    for block in blocks:
        lines = [ln.strip() for ln in block.text.splitlines() if ln.strip()]
        holding_row = (
            block.block_type != "table"
            and len(lines) >= 3
            and bool(_PCT_OR_NUM.match(lines[2]))
            and not _PCT_OR_NUM.match(lines[0])
        )
        if holding_row:
            buf.append(block)
            continue
        flush()
        merged.append(block if block.block_type == "table" else _promote_factsheet_grids(block))
    flush()
    return merged


def _promote_factsheet_grids(block: ParsedBlock) -> ParsedBlock:
    """Turn stacked 'name / industry / 9.18' lines into a delimited table chunk."""
    if block.block_type == "table":
        return block
    lines = [ln.strip() for ln in block.text.splitlines() if ln.strip()]
    triples: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(lines):
        name, industry, value = lines[i], lines[i + 1], lines[i + 2]
        if (
            _PCT_OR_NUM.match(value)
            and not _PCT_OR_NUM.match(name)
            and len(name) < 80
            and len(industry) < 60
        ):
            triples.append((name, industry, value))
            i += 3
            continue
        break
    if len(triples) < 3:
        return block
    rows = ["Company | Industry | Value"]
    rows.extend(f"{a} | {b} | {c}" for a, b, c in triples)
    leftover = lines[i:]
    text = "[TABLE]\n" + "\n".join(rows)
    if leftover:
        text += "\n" + "\n".join(leftover)
    return block.model_copy(update={"text": text, "block_type": "table"})


def _drop_tiny_noise(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    kept: list[ParsedBlock] = []
    for block in blocks:
        if block.block_type == "table":
            kept.append(block)
            continue
        if len(block.text.strip()) < 8:
            continue
        kept.append(block)
    return kept
