"""Score golden-set adversarial items with the Phase 4 heuristic (no Groq judge)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import REPO_ROOT
from app.guardrails.pipeline import run_guarded
from app.guardrails.policies import classify_query
from app.guardrails.scoring import score_adversarial

logger = logging.getLogger(__name__)

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.json"
OUT_MD = REPO_ROOT / "eval" / "results" / "guardrail_adversarial.md"
OUT_JSON = REPO_ROOT / "eval" / "results" / "guardrail_adversarial.json"


def load_adversarial() -> list[dict]:
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return [item for item in payload.get("golden_set") or [] if item.get("category") == "adversarial_out_of_scope"]


def run_verification(*, skip_pipeline: bool = False) -> Path:
    """Evaluate the 8 adversarial questions.

    Default path calls ``run_guarded``. All eight golden adversarial prompts currently
    match a pre-generation policy, so this uses **zero Groq chat-completion calls**.
    Local retrieval is also skipped on those refusals. ``--skip-pipeline`` scores the
    canned policy messages only (still zero API calls, no Chroma).
    """
    items = load_adversarial()
    rows: list[dict] = []
    groq_calls_expected = 0
    for item in items:
        qid = str(item.get("id"))
        question = item.get("question") or ""
        decision = classify_query(question)
        if skip_pipeline:
            answer = decision.message or ""
            skipped_llm = True
        else:
            response = run_guarded(
                question,
                retrieve_fn=lambda _q: [],
                generate_fn=_fail_if_generate,
            )
            answer = response.answer
            skipped_llm = True
            if not decision.refuse:
                groq_calls_expected += 1
                skipped_llm = False
        score = score_adversarial(qid, question, answer)
        rows.append(
            {
                "id": qid,
                "question": question,
                "policy_code": decision.code,
                "pre_generation_refuse": decision.refuse,
                "skipped_llm": skipped_llm,
                "answer": answer,
                **score,
            }
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "groq_chat_completions_used": 0,
            "groq_chat_completions_if_unmatched_policies": groq_calls_expected,
            "judge": "rule-based (app.guardrails.scoring)",
        },
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(_markdown(rows), encoding="utf-8")
    logger.info("Wrote %s", OUT_MD)
    return OUT_MD


def _fail_if_generate(query: str, hits) -> None:
    raise AssertionError(
        f"Guardrail leaked to generation for {query!r} ({len(hits)} hits). "
        "Verification is supposed to stay Groq-free."
    )


def _markdown(rows: list[dict]) -> str:
    def cell(v: str) -> str:
        return str(v).replace("|", "/")

    table = [
        "| id | policy | verdict | confidence | skipped LLM | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        notes = "; ".join(row.get("reasons") or []) or "—"
        table.append(
            "| "
            + " | ".join(
                [
                    cell(row["id"]),
                    cell(row.get("policy_code") or "none"),
                    cell(row["verdict"]),
                    cell(row["confidence"]),
                    "yes" if row.get("skipped_llm") else "NO",
                    cell(notes),
                ]
            )
            + " |"
        )
    counts = {k: sum(1 for r in rows if r["verdict"] == k) for k in ("pass", "fail", "manual_review")}
    manual = [r["id"] for r in rows if r["verdict"] == "manual_review"]
    return "\n".join(
        [
            "# Phase 4 adversarial guardrail check",
            "",
            "Rule-based scoring only (no LLM judge). Pre-generation policies fire on all eight "
            "golden adversarial questions, so this run uses **0 Groq API calls**.",
            "",
            f"- pass: {counts['pass']}",
            f"- fail: {counts['fail']}",
            f"- manual_review: {counts['manual_review']}"
            + (f" — please inspect {', '.join(f'`{i}`' for i in manual)}" if manual else ""),
            "",
            *table,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score adversarial golden items without Groq")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Score canned policy text only (do not instantiate retrieve/generate)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_verification(skip_pipeline=args.skip_pipeline)


if __name__ == "__main__":
    main()
