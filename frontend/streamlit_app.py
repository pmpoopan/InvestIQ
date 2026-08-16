"""InvestIQ Streamlit chat — talks only to POST /api/research.

Internal: the API currently wraps the Phase 2 baseline retriever (dense, top_k=5)
plus Phase 4 guardrails. Retrieval/generation config is whatever the backend
exposes; this UI does not select or mention an ablation winner.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# streamlit run frontend/streamlit_app.py puts `frontend/` on sys.path, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.guardrails.messages import DISCLAIMER
from app.ingestion.metadata import DOCUMENT_CATALOG

DEFAULT_API_URL = os.getenv("INVESTIQ_API_URL", "http://127.0.0.1:8000")

_FOLLOWUP = re.compile(
    r"(?is)^("
    r"what\s+about|how\s+about|and\s+(the|its|that|this)|also|"
    r"same\s+(for|as)|for\s+that|that\s+fund|this\s+fund|"
    r"its\s+\w+|the\s+(ter|exit\s+load|benchmark|expense|aum|nav|sip)\b"
    r")"
)
_PRONOUN = re.compile(r"(?i)\b(it|its|this|that|they|them|those|these)\b")

_DOC_TYPE_LABEL = {
    "sid": "SID",
    "factsheet": "Factsheet",
    "drhp": "DRHP / IPO",
    "sebi_regulation": "SEBI",
}


def corpus_schemes() -> list[str]:
    names = {
        str(meta["fund_name"])
        for meta in DOCUMENT_CATALOG.values()
        if meta.get("document_type") in {"sid", "factsheet"} and meta.get("fund_name")
    }
    return sorted(names)


def corpus_ipos() -> list[str]:
    names = {
        str(meta["fund_name"])
        for meta in DOCUMENT_CATALOG.values()
        if meta.get("document_type") == "drhp" and meta.get("fund_name")
    }
    return sorted(names)


def looks_like_followup(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    if _FOLLOWUP.match(q):
        return True
    return len(q.split()) <= 12 and bool(_PRONOUN.search(q))


def last_named_scheme(history: list[dict[str, Any]], extra: str = "") -> str | None:
    blob = extra
    for msg in reversed(history):
        if msg.get("role") == "user":
            blob = f"{msg.get('content') or ''} {blob}"
            break
    blob_l = blob.lower()
    for name in sorted(corpus_schemes() + corpus_ipos(), key=len, reverse=True):
        if name.lower() in blob_l:
            return name
    fund_alias = re.search(r"\bfund\s+[a-f]\b", blob, re.I)
    if fund_alias:
        return fund_alias.group(0)
    return None


def standalone_query(history: list[dict[str, Any]], user_text: str) -> str:
    """Turn a likely follow-up into one self-contained string for the API.

    No extra LLM call. Does not send the raw transcript to the backend.
    """
    text = user_text.strip()
    if not looks_like_followup(text):
        return text
    last_user = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user = str(msg.get("content") or "").strip()
            break
    if not last_user:
        return text
    scheme = last_named_scheme(history, extra=text)
    if scheme and _PRONOUN.search(text):
        filled = _PRONOUN.sub(scheme, text, count=1)
        if scheme.lower() not in filled.lower():
            filled = f"{text} (regarding {scheme})"
        return filled
    if scheme:
        return f"{text} (regarding {scheme})"
    return f"{text} (follow-up to: {last_user})"


def call_research(api_url: str, query: str) -> dict[str, Any]:
    """POST /api/research once. Called only from chat submit, never on keystroke."""
    body = json.dumps({"query": query}).encode("utf-8")
    req = Request(
        f"{api_url.rstrip('/')}/api/research",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = parsed.get("detail") or detail
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"API {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach API at {api_url}. Start FastAPI first. ({exc.reason})"
        ) from exc


def ping_health(api_url: str) -> str:
    req = Request(f"{api_url.rstrip('/')}/health", method="GET")
    with urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    chunks = data.get("chunk_count")
    return f"ok · {chunks} chunks indexed" if chunks is not None else "ok"


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("disclaimer", DISCLAIMER)
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("health", None)


def _sidebar() -> str:
    st.sidebar.header("Corpus scope")
    st.sidebar.caption(
        "Ask about these ingested schemes and the IPO document. "
        "Names outside this list are out of scope."
    )
    funds = corpus_schemes()
    ipos = corpus_ipos()
    st.sidebar.subheader(f"Mutual funds ({len(funds)})")
    for name in funds:
        st.sidebar.markdown(f"- {name}")
    st.sidebar.subheader(f"IPO documents ({len(ipos)})")
    for name in ipos:
        st.sidebar.markdown(f"- {name}")
    st.sidebar.caption("Also indexed: SEBI Mutual Funds Regulations (scanned PDF may have no extractable text).")

    st.sidebar.divider()
    api_url = st.sidebar.text_input("API base URL", value=st.session_state.api_url)
    st.session_state.api_url = api_url
    if st.sidebar.button("Ping API (no LLM)"):
        try:
            st.session_state.health = ping_health(api_url)
        except Exception as exc:
            st.session_state.health = f"unreachable: {exc}"
    if st.session_state.health:
        st.sidebar.caption(st.session_state.health)
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    return api_url


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for cit in citations:
        doc = cit.get("source_document") or "unknown"
        dtype = _DOC_TYPE_LABEL.get(str(cit.get("document_type") or ""), cit.get("document_type") or "")
        page = cit.get("page_number") or "?"
        st.markdown(f"- {doc} · {dtype} · p. {page}")


def main() -> None:
    st.set_page_config(page_title="InvestIQ", page_icon="📑", layout="centered")
    _init_state()
    api_url = _sidebar()

    st.title("InvestIQ")
    st.caption("Research assistant over ingested SIDs, factsheets, a DRHP, and SEBI text.")
    st.info(st.session_state.disclaimer)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content") or "")
            if msg.get("role") == "assistant":
                _render_citations(msg.get("citations") or [])

    prompt = st.chat_input("Ask about a scheme, factsheet, or the IPO document")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    query = standalone_query(st.session_state.messages[:-1], prompt)
    with st.chat_message("assistant"):
        with st.spinner("Looking up documents…"):
            try:
                payload = call_research(api_url, query)
            except Exception as exc:
                err = str(exc)
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "citations": []}
                )
                return
        if payload.get("disclaimer"):
            st.session_state.disclaimer = payload["disclaimer"]
        answer = payload.get("answer") or ""
        citations = payload.get("citations") or []
        st.markdown(answer)
        _render_citations(citations)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )


if __name__ == "__main__":
    main()
