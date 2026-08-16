"""Run ingestion over data/raw and write JSONL under data/processed."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from app.ingestion.chunkers import chunk_document, write_chunks_jsonl
from app.ingestion.parsers import parse_pdf
from app.models.schemas import Chunk, ChunkingConfig, DocumentType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

TYPE_DIRS: dict[DocumentType, str] = {
    "sid": "sid",
    "factsheet": "factsheet",
    "drhp": "drhp",
    "sebi_regulation": "sebi",
}


def ingest_corpus(
    raw_root: Path | None = None,
    processed_root: Path | None = None,
    config: ChunkingConfig | None = None,
) -> dict[DocumentType, list[Chunk]]:
    """Parse every PDF under data/raw and write one JSONL file per document type."""
    raw_root = raw_root or RAW_ROOT
    processed_root = processed_root or PROCESSED_ROOT
    config = config or ChunkingConfig()
    by_type: dict[DocumentType, list[Chunk]] = defaultdict(list)

    for doc_type, folder in TYPE_DIRS.items():
        directory = raw_root / folder
        if not directory.exists():
            continue
        for pdf in sorted(directory.glob("*.pdf")):
            logger.info("Parsing %s (%s)", pdf.name, doc_type)
            parsed = parse_pdf(pdf, document_type=doc_type)
            for warning in parsed.warnings:
                logger.warning(warning)
            chunks = chunk_document(parsed, config=config)
            logger.info("  %s -> %d chunks from %d pages", pdf.name, len(chunks), parsed.page_count)
            by_type[doc_type].extend(chunks)

    processed_root.mkdir(parents=True, exist_ok=True)
    for doc_type, chunks in by_type.items():
        out = processed_root / f"{doc_type}.jsonl"
        write_chunks_jsonl(chunks, out)
        logger.info("Wrote %s (%d chunks)", out, len(chunks))

    return dict(by_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="InvestIQ Phase 1 ingestion")
    parser.add_argument(
        "--strategy",
        choices=["section_aware", "fixed_size"],
        default="section_aware",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_corpus(
        config=ChunkingConfig(
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )


if __name__ == "__main__":
    main()
