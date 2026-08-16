"""Chunking strategies for ingested mutual-fund and IPO documents.

Phase 1 will add document-type-aware chunking (section/clause vs table vs
narrative) so retrieval quality can be ablated later.
"""


def chunk_document(parsed_document: object, strategy: str) -> list:
    """Split a parsed document into retrieval chunks.

    TODO(Phase 1): implement chunking strategies.
    """
    pass
