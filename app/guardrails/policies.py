"""Cheap query-side policies. No LLM classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.guardrails.messages import MESSAGES

_FLAGS = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class QueryDecision:
    refuse: bool
    code: str | None = None
    message: str | None = None


def _rule(code: str) -> QueryDecision:
    return QueryDecision(refuse=True, code=code, message=MESSAGES[code])


def classify_query(query: str) -> QueryDecision:
    """Return a refusal decision when a cheap pattern is enough; otherwise allow."""
    q = (query or "").strip()
    if not q:
        return QueryDecision(refuse=False)

    # Order matters: injection and transactions before generic "should I".
    if _INJECTION.search(q):
        return _rule("prompt_injection")
    if _TRANSACTIONAL.search(q):
        return _rule("transactional")
    if _ALLOCATION.search(q):
        return _rule("allocation_advice")
    if _FUTURE.search(q):
        return _rule("future_performance")
    if _MARKET.search(q):
        return _rule("market_speculation")
    if _ACCUSATION.search(q):
        return _rule("unverified_accusation")
    if _UNKNOWN_CORPUS.search(q):
        return _rule("unknown_corpus")
    if _ADVICE.search(q):
        return _rule("personalized_advice")
    return QueryDecision(refuse=False)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


_INJECTION = re.compile(
    r"(ignore\s+(your\s+)?(previous|prior|all)\s+instructions"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|tell\s+me\s+your\s+system\s+prompt"
    r"|system\s+prompt"
    r"|override\s+(your|the)\s+(rules|instructions)"
    r"|jailbreak)",
    _FLAGS,
)

_TRANSACTIONAL = re.compile(
    r"(auto-?debit"
    r"|set\s+up\s+a\s+sip"
    r"|(debit|access)\s+my\s+bank"
    r"|execute\s+(the\s+)?(investment|order|sip)"
    r"|place\s+(an?\s+)?order)",
    _FLAGS,
)

_ALLOCATION = re.compile(
    r"("
    r"how\s+should\s+i\s+(split|allocate|diversify|divide)"
    r"|split\s+(it|this|my\s+money|my\s+investment)"
    r"|i\s+have\s+.+\s+to\s+invest"
    r"|asset\s+allocation\s+for\s+me"
    r")",
    _FLAGS,
)

_FUTURE = re.compile(
    r"("
    r"best\s+returns"
    r"|will\s+(give\s+me\s+)?the\s+best\s+return"
    r"|will\s+(this|the|that)\s+fund\s+perform"
    r"|over\s+the\s+next\s+\d+\s+years"
    r"|future\s+(returns?|performance)"
    r"|predict\s+(the\s+)?(returns?|performance|nav)"
    r")",
    _FLAGS,
)

_MARKET = re.compile(
    r"("
    r"market\s+(going\s+to\s+)?crash"
    r"|stock\s+market\s+going\s+to"
    r"|will\s+the\s+(stock\s+)?market"
    r")",
    _FLAGS,
)

_ACCUSATION = re.compile(
    r"\b(scam|ponzi|fraudulent\s+scheme)\b",
    _FLAGS,
)

_UNKNOWN_CORPUS = re.compile(
    r"("
    r"isn'?t\s+in\s+your\s+document"
    r"|not\s+in\s+your\s+(document|corpus|data)"
    r"|don'?t\s+have\s+data"
    r"|fund\s+you\s+don'?t\s+have"
    r"|some\s+fund\s+you\s+don'?t"
    r")",
    _FLAGS,
)

_ADVICE = re.compile(
    r"("
    r"\bshould\s+i\s+(invest|buy|sell|redeem|hold|put)\b"
    r"|which\s+fund\s+should\s+i"
    r"|do\s+you\s+recommend"
    r"|what\s+do\s+you\s+recommend"
    r"|is\s+(it|this|that)\s+(a\s+)?good\s+(time|investment|idea)\s+to\s+invest"
    r"|good\s+time\s+to\s+(buy|invest)"
    r"|right\s+for\s+me"
    r")",
    _FLAGS,
)
