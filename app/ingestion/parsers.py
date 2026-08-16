"""PDF parsers per document type (SID, factsheet, DRHP, SEBI regulation).

Phase 1 will implement type-specific extraction so clause-heavy SIDs, tabular
factsheets, long-form DRHPs, and numbered SEBI text are not forced through a
single generic PDF pipeline.
"""


def parse_pdf(path: str, document_type: str) -> None:
    """Parse a raw PDF into structured text/blocks.

    TODO(Phase 1): implement parsers per document type.
    """
    pass
