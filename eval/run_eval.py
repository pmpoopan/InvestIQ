"""Golden-set eval harness: four retrieval configs + RAGAS-style metrics."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config.settings import REPO_ROOT, get_settings
from app.generation.rag_chain import RagUnavailableError, run_rag
from app.ingestion.pipeline import ingest_chunk_strategy, parse_all_documents
from app.models.schemas import Chunk, ChunkingConfig, RetrievedChunk
from app.retrieval.hybrid import BM25Index, hybrid_query
from app.retrieval.rerank import rerank
from app.retrieval.vector_store import get_collection, load_processed_chunks, query as dense_query, upsert_chunks
from eval.metrics import compute_metrics, is_placeholder_answer

logger = logging.getLogger(__name__)

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
FIXED_PROCESSED = REPO_ROOT / "data" / "processed" / "fixed_size"
SECTION_COLLECTION = "investiq_chunks"
FIXED_COLLECTION = "investiq_fixed_size"

CATEGORIES = (
    "factual_lookup",
    "definitional",
    "multi_document_synthesis",
    "adversarial_out_of_scope",
)
METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")

Retriever = Callable[[str], list[RetrievedChunk]]


def _config_specs() -> list[dict]:
    return [
        {
            "id": "a_baseline_fixed_dense",
            "label": "A — Baseline (fixed-size, dense-only, top_k=5)",
            "chunking": "fixed_size",
            "retrieval": "dense",
            "rerank": False,
            "top_k": 5,
        },
        {
            "id": "b_section_dense",
            "label": "B — Section-aware + dense-only, top_k=5",
            "chunking": "section_aware",
            "retrieval": "dense",
            "rerank": False,
            "top_k": 5,
        },
        {
            "id": "c_section_hybrid",
            "label": "C — Section-aware + hybrid (dense+BM25 RRF), top_k=5",
            "chunking": "section_aware",
            "retrieval": "hybrid",
            "rerank": False,
            "top_k": 5,
        },
        {
            "id": "d_section_hybrid_rerank",
            "label": "D — Section-aware + hybrid + cross-encoder (15→5)",
            "chunking": "section_aware",
            "retrieval": "hybrid",
            "rerank": True,
            "top_k": 5,
            "candidate_k": 15,
        },
    ]


def _collection_count(name: str) -> int:
    try:
        return int(get_collection(name=name).count())
    except Exception:
        return 0


def ensure_indexes(rebuild: bool = False) -> dict[str, int]:
    """Build fixed-size + section-aware Chroma collections as needed."""
    counts = {"section_aware": _collection_count(SECTION_COLLECTION), "fixed_size": _collection_count(FIXED_COLLECTION)}
    if rebuild or counts["section_aware"] == 0:
        section_chunks = load_processed_chunks()
        if not section_chunks:
            raise RuntimeError("No section-aware chunks in data/processed. Run ingestion first.")
        logger.info("Indexing %d section-aware chunks into %s", len(section_chunks), SECTION_COLLECTION)
        upsert_chunks(section_chunks, reset=True, collection_name=SECTION_COLLECTION)
        counts["section_aware"] = len(section_chunks)
    else:
        logger.info("Reusing Chroma collection %s (%d vectors)", SECTION_COLLECTION, counts["section_aware"])

    jsonl_ready = FIXED_PROCESSED.exists() and any(FIXED_PROCESSED.glob("*.jsonl"))
    if rebuild or counts["fixed_size"] == 0:
        if rebuild or not jsonl_ready:
            logger.info("Parsing corpus once for fixed-size chunking")
            parsed = parse_all_documents()
            ingest_chunk_strategy(
                parsed,
                ChunkingConfig(strategy="fixed_size", chunk_size=1200, chunk_overlap=150),
                FIXED_PROCESSED,
            )
        fixed_chunks = load_processed_chunks(FIXED_PROCESSED)
        logger.info("Indexing %d fixed-size chunks into %s", len(fixed_chunks), FIXED_COLLECTION)
        upsert_chunks(fixed_chunks, reset=True, collection_name=FIXED_COLLECTION)
        counts["fixed_size"] = len(fixed_chunks)
    else:
        logger.info("Reusing Chroma collection %s (%d vectors)", FIXED_COLLECTION, counts["fixed_size"])
    return counts


def _load_section_chunks() -> list[Chunk]:
    return load_processed_chunks()


def build_retriever(spec: dict, bm25: BM25Index | None) -> Retriever:
    top_k = int(spec["top_k"])
    collection = FIXED_COLLECTION if spec["chunking"] == "fixed_size" else SECTION_COLLECTION

    def dense(query: str) -> list[RetrievedChunk]:
        return dense_query(query, k=top_k, collection_name=collection)

    def hybrid(query: str) -> list[RetrievedChunk]:
        if bm25 is None:
            raise RuntimeError("BM25 index is required for hybrid retrieval")
        cand = int(spec.get("candidate_k") or 15)
        if spec.get("rerank"):
            fused = hybrid_query(
                query, bm25, collection_name=collection, top_k=cand, candidate_k=cand
            )
            return rerank(query, fused, top_k=top_k)
        return hybrid_query(
            query, bm25, collection_name=collection, top_k=top_k, candidate_k=cand
        )

    return hybrid if spec["retrieval"] == "hybrid" else dense


def _hits_payload(hits: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "score": hit.score,
            "source_document": hit.chunk.source_document,
            "document_type": hit.chunk.document_type,
            "page_number": hit.chunk.page_number,
            "section_heading": hit.chunk.section_heading,
            "text": hit.chunk.text[:4000],
        }
        for hit in hits
    ]


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def summarize_rows(rows: list[dict]) -> dict:
    overall = {m: _mean([r.get(m) for r in rows]) for m in METRICS}
    by_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        subset = [r for r in rows if r.get("category") == cat]
        by_category[cat] = {
            **{m: _mean([r.get(m) for r in subset]) for m in METRICS},
            "n": len(subset),
            "n_with_ground_truth": sum(1 for r in subset if not r.get("no_ground_truth_answer_yet")),
        }
    return {
        "overall": {**overall, "n": len(rows)},
        "by_category": by_category,
        "n_placeholder": sum(1 for r in rows if r.get("no_ground_truth_answer_yet")),
        "n_scored_recall": sum(1 for r in rows if r.get("context_recall") is not None),
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}"


def write_config_markdown(path: Path, spec: dict, summary: dict, rows: list[dict]) -> None:
    overall = summary["overall"]
    lines = [
        f"# {spec['label']}",
        "",
        f"- Config id: `{spec['id']}`",
        f"- Chunking: `{spec['chunking']}`",
        f"- Retrieval: `{spec['retrieval']}`"
        + (" + cross-encoder rerank (15→5)" if spec.get("rerank") else ""),
        f"- top_k: {spec['top_k']}",
        f"- Questions: {overall['n']} ({summary['n_placeholder']} flagged **no ground truth answer yet**)",
        f"- Context recall scored on {summary['n_scored_recall']} items with a real `expected_answer`",
        "",
        "## Overall",
        "",
        _md_table(
            ["faithfulness", "context precision", "context recall", "answer relevancy"],
            [[_fmt(overall[m]) for m in METRICS]],
        ),
        "",
        "## By category",
        "",
        _md_table(
            ["category", "n", "n with GT", "faithfulness", "context precision", "context recall", "answer relevancy"],
            [
                [
                    cat,
                    str(summary["by_category"][cat]["n"]),
                    str(summary["by_category"][cat]["n_with_ground_truth"]),
                    *[_fmt(summary["by_category"][cat][m]) for m in METRICS],
                ]
                for cat in CATEGORIES
            ],
        ),
        "",
        "## Per question",
        "",
        _md_table(
            ["id", "category", "ground truth", "faithfulness", "ctx precision", "ctx recall", "answer relevancy"],
            [
                [
                    str(r.get("id")),
                    str(r.get("category")),
                    "no ground truth answer yet" if r.get("no_ground_truth_answer_yet") else "yes",
                    _fmt(r.get("faithfulness")),
                    _fmt(r.get("context_precision")),
                    _fmt(r.get("context_recall")),
                    _fmt(r.get("answer_relevancy")),
                ]
                for r in rows
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_summary(path: Path, config_reports: list[tuple[dict, dict, list[dict]]], meta: dict) -> None:
    notes = _written_notes(config_reports)
    overall_rows = []
    for spec, summary, _rows in config_reports:
        o = summary["overall"]
        overall_rows.append([spec["label"], *[_fmt(o[m]) for m in METRICS], str(o["n"])])

    cat_sections: list[str] = []
    for cat in CATEGORIES:
        cat_rows = []
        for spec, summary, _rows in config_reports:
            c = summary["by_category"][cat]
            cat_rows.append(
                [
                    spec["id"].split("_", 1)[0].upper(),
                    spec["label"].split("—", 1)[-1].strip(),
                    *[_fmt(c[m]) for m in METRICS],
                    f"{c['n_with_ground_truth']}/{c['n']}",
                ]
            )
        cat_sections.extend(
            [
                f"### {cat}",
                "",
                _md_table(
                    ["cfg", "setup", "faithfulness", "context precision", "context recall", "answer relevancy", "GT / n"],
                    cat_rows,
                ),
                "",
            ]
        )

    placeholder_ids = sorted(
        {
            str(r.get("id"))
            for _spec, _s, rows in config_reports
            for r in rows
            if r.get("no_ground_truth_answer_yet")
        }
    )

    body = [
        "# InvestIQ ablation summary",
        "",
        "Four retrieval configurations on the same golden set. Metrics follow the RAGAS definitions "
        "(faithfulness, context precision, context recall, answer relevancy) implemented with Groq-as-judge "
        "and local BGE embeddings. The `ragas` Python package (0.4.x / 0.2.x) currently fails to import "
        "because it still references `langchain_community.chat_models.vertexai`, which was removed from "
        "langchain-community. DeepEval is the closest actively maintained packaged alternative.",
        "",
        f"- Generated (UTC): {meta.get('created_at')}",
        f"- Generator: `{meta.get('groq_model')}`",
        f"- Judge: same Groq model, temperature 0",
        f"- Embeddings: `{meta.get('embedding_model')}`",
        f"- Golden set: `{GOLDEN_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Overall (mean over the golden set)",
        "",
        "Context recall is averaged only over questions that have a real `expected_answer`. "
        "Placeholder items still contribute to faithfulness, context precision, and answer relevancy.",
        "",
        _md_table(
            ["configuration", "faithfulness", "context precision", "context recall", "answer relevancy", "n"],
            overall_rows,
        ),
        "",
        "## Per-category breakdown",
        "",
        "Different setups help different question types. Hybrid/BM25 is expected to matter more when "
        "the query has distinctive lexical keys (scheme names, clause numbers, TER). Dense-only may "
        "suffice for definitional paraphrases. Adversarial items test refusal, not fact recall.",
        "",
        *cat_sections,
        "## Placeholders still needing a manual `expected_answer`",
        "",
        "These IDs were flagged **no ground truth answer yet**. Automated scoring did **not** compare "
        "the model answer to `expected_answer`; context recall is omitted.",
        "",
        ", ".join(f"`{i}`" for i in placeholder_ids) if placeholder_ids else "_None._",
        "",
        "## Notes",
        "",
        notes,
        "",
        "### How to read the table",
        "",
        "- **Faithfulness**: fraction of answer claims entailed by retrieved context (hallucination check).",
        "- **Context precision**: RAGAS precision@k over retrieved chunks judged useful for the question.",
        "- **Context recall**: fraction of ground-truth statements supported by retrieved context (needs GT).",
        "- **Answer relevancy**: mean cosine similarity between the question and questions reverse-engineered from the answer.",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def _written_notes(config_reports: list[tuple[dict, dict, list[dict]]]) -> str:
    scored = []
    for spec, summary, _rows in config_reports:
        o = summary["overall"]
        present = [o[m] for m in METRICS if o[m] is not None]
        macro = sum(present) / len(present) if present else None
        scored.append((spec, summary, macro))
    scored.sort(key=lambda x: (x[2] is None, -(x[2] or 0)))
    best_spec, best_sum, best_macro = scored[0]
    lines = [
        f"**Best overall** on this run: **{best_spec['label']}** "
        f"(macro-average of available metrics = {_fmt(best_macro)}). "
        f"Headline scores: faithfulness {_fmt(best_sum['overall']['faithfulness'])}, "
        f"context precision {_fmt(best_sum['overall']['context_precision'])}, "
        f"context recall {_fmt(best_sum['overall']['context_recall'])}, "
        f"answer relevancy {_fmt(best_sum['overall']['answer_relevancy'])}.",
        "",
    ]

    by_id = {spec["id"]: (spec, summary) for spec, summary, _m in scored}
    a = by_id.get("a_baseline_fixed_dense")
    b = by_id.get("b_section_dense")
    c = by_id.get("c_section_hybrid")
    d = by_id.get("d_section_hybrid_rerank")

    def _delta(left, right, metric: str, category: str | None = None) -> str:
        if not left or not right:
            return "n/a"
        ls, rs = left[1], right[1]
        src = ls["by_category"][category] if category else ls["overall"]
        dst = rs["by_category"][category] if category else rs["overall"]
        lv, rv = src.get(metric), dst.get(metric)
        if lv is None or rv is None:
            return "n/a"
        return f"{rv - lv:+.3f}"

    if a and b:
        lines.append(
            f"Section-aware vs fixed-size (A→B) changed context precision by {_delta(a, b, 'context_precision')} "
            f"overall and faithfulness by {_delta(a, b, 'faithfulness')}."
        )
    if b and c:
        lines.append(
            "Hybrid vs dense-only (B→C), by category (context precision): "
            + "; ".join(f"{cat} {_delta(b, c, 'context_precision', cat)}" for cat in CATEGORIES)
            + "."
        )
        lines.append(
            "Hybrid vs dense-only (B→C), by category (faithfulness): "
            + "; ".join(f"{cat} {_delta(b, c, 'faithfulness', cat)}" for cat in CATEGORIES)
            + "."
        )
    if c and d:
        lines.append(
            f"Reranking (C→D) changed overall context precision by {_delta(c, d, 'context_precision')} "
            f"and faithfulness by {_delta(c, d, 'faithfulness')}."
        )

    lines.append("")
    lines.append("**Where each setup struggled** (lowest context precision in that config):")
    for spec, summary, _m in scored:
        cats = summary["by_category"]
        weakest = min(
            CATEGORIES,
            key=lambda cat: cats[cat]["context_precision"]
            if cats[cat]["context_precision"] is not None
            else 99,
        )
        lines.append(
            f"- {spec['label']}: weakest category **{weakest}** "
            f"(context precision {_fmt(cats[weakest]['context_precision'])}, "
            f"faithfulness {_fmt(cats[weakest]['faithfulness'])})."
        )

    lines.append("")
    lines.append(
        "Adversarial `expected_answer` values describe refusal *behavior*, not a fact to retrieve. "
        "Context recall on that category is therefore a noisy proxy. Factual and multi-document items "
        "with placeholder answers do not contribute to recall until you fill them from the SIDs."
    )
    return "\n".join(lines)


def run_one_config(
    spec: dict,
    items: list[dict],
    retriever: Retriever,
    *,
    score: bool,
    resume: dict[str, dict],
    sleep_s: float,
    checkpoint: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        qid = item.get("id")
        question = item.get("question") or ""
        expected = item.get("expected_answer") or ""
        placeholder = is_placeholder_answer(expected)
        prior = resume.get(qid)
        if prior and prior.get("answer") and not prior.get("error"):
            needs_metrics = score and prior.get("faithfulness") is None and not prior.get("metrics_error")
            if not needs_metrics:
                logger.info("Resume skip %s / %s", spec["id"], qid)
                rows.append(prior)
                continue
            logger.info("Resume scoring %s / %s", spec["id"], qid)
            contexts = [r.get("text") or "" for r in (prior.get("retrieved") or [])]
            prior.update(compute_metrics(question, prior["answer"], contexts, expected))
            time.sleep(sleep_s)
            rows.append(prior)
            if checkpoint:
                checkpoint(rows)
            continue
        logger.info("Running %s / %s", spec["id"], qid)
        hits = retriever(question)
        error = None
        answer = ""
        citations: list[dict] = []
        for attempt in range(4):
            try:
                response = run_rag(question, hits=hits)
                answer = response.answer
                citations = [c.model_dump() for c in response.citations]
                error = None
                break
            except RagUnavailableError as exc:
                error = str(exc)
                answer = str(exc)
                logger.warning("%s generation attempt %s failed: %s", qid, attempt + 1, exc)
                time.sleep(max(sleep_s, 8.0) * (attempt + 1))
        row: dict = {
            "id": qid,
            "category": item.get("category"),
            "question": question,
            "expected_answer": expected,
            "no_ground_truth_answer_yet": placeholder,
            "answer": answer,
            "citations": citations,
            "retrieved": _hits_payload(hits),
            "error": error,
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "answer_relevancy": None,
        }
        if score and not error:
            contexts = [h.chunk.text for h in hits]
            metrics = compute_metrics(question, answer, contexts, expected)
            row.update(metrics)
            time.sleep(sleep_s)
        elif not score:
            time.sleep(sleep_s)
        rows.append(row)
        if checkpoint:
            checkpoint(rows)
    return rows


def _dump_config(spec: dict, rows: list[dict], summary: dict, settings: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": spec,
            "embedding_model": settings["embedding_model"],
            "groq_model": settings["groq_model"],
            "metrics": "ragas-style (custom Groq judge; ragas package import-broken)",
        },
        "summary": summary,
        "results": rows,
    }
    json_path = RESULTS_DIR / f"{spec['id']}.json"
    md_path = RESULTS_DIR / f"{spec['id']}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_config_markdown(md_path, spec, summary, rows)
    logger.info("Wrote %s and %s", json_path, md_path)
    return json_path, md_path


def run_eval(
    *,
    configs: list[str] | None = None,
    rebuild_index: bool = False,
    skip_index: bool = False,
    score: bool = True,
    limit: int | None = None,
    sleep_s: float = 1.0,
) -> Path:
    settings = get_settings()
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = payload.get("golden_set") or []
    if limit:
        items = items[:limit]
    if not skip_index:
        ensure_indexes(rebuild=rebuild_index)

    wanted = set(configs) if configs else {s["id"] for s in _config_specs()}
    # allow short aliases a/b/c/d
    alias = {"a": "a_baseline_fixed_dense", "b": "b_section_dense", "c": "c_section_hybrid", "d": "d_section_hybrid_rerank"}
    wanted = {alias.get(x, x) for x in wanted}

    bm25 = None
    if any(s["id"] in wanted and s["retrieval"] == "hybrid" for s in _config_specs()):
        logger.info("Building BM25 over section-aware chunks")
        bm25 = BM25Index(_load_section_chunks())

    reports: list[tuple[dict, dict, list[dict]]] = []
    for spec in _config_specs():
        if spec["id"] not in wanted:
            continue
        json_path = RESULTS_DIR / f"{spec['id']}.json"
        resume: dict[str, dict] = {}
        if json_path.exists():
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            for row in prior.get("results") or []:
                if row.get("id"):
                    resume[row["id"]] = row
        retriever = build_retriever(spec, bm25)

        def checkpoint(current_rows: list[dict], _spec=spec) -> None:
            _dump_config(_spec, current_rows, summarize_rows(current_rows), settings)

        rows = run_one_config(
            spec,
            items,
            retriever,
            score=score,
            resume=resume,
            sleep_s=sleep_s,
            checkpoint=checkpoint,
        )
        summary = summarize_rows(rows)
        _dump_config(spec, rows, summary, settings)
        reports.append((spec, summary, rows))

    all_specs = {s["id"]: s for s in _config_specs()}
    for spec in _config_specs():
        json_path = RESULTS_DIR / f"{spec['id']}.json"
        if spec["id"] in {r[0]["id"] for r in reports}:
            continue
        if json_path.exists():
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            rows = prior.get("results") or []
            summary = prior.get("summary") or summarize_rows(rows)
            reports.append((all_specs[spec["id"]], summary, rows))

    reports.sort(key=lambda r: [s["id"] for s in _config_specs()].index(r[0]["id"]))
    summary_path = RESULTS_DIR / "ablation_summary.md"
    write_ablation_summary(
        summary_path,
        reports,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "groq_model": settings["groq_model"],
            "embedding_model": settings["embedding_model"],
        },
    )
    logger.info("Wrote %s", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="InvestIQ golden-set ablation")
    parser.add_argument("--configs", nargs="*", help="Config ids or a/b/c/d")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--no-score", action="store_true", help="Dump answers only, skip judge metrics")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eval(
        configs=args.configs,
        rebuild_index=args.rebuild_index,
        skip_index=args.skip_index,
        score=not args.no_score,
        limit=args.limit,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()
