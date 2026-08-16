"""Fixed user-facing copy for refusals and the API disclaimer."""

DISCLAIMER = (
    "InvestIQ is an independent educational research assistant. This is not investment "
    "advice, a solicitation, or a recommendation to buy or sell any security or scheme. "
    "Verify figures against the official scheme documents and SEBI/AMFI sources. For "
    "personalized advice, consult a SEBI-registered investment adviser."
)

NO_RELEVANT_CONTEXT = (
    "The retrieved documents do not contain enough relevant information to answer this "
    "question. I will not guess."
)

PERSONALIZED_ADVICE = (
    "I cannot give personalized investment advice or tell you whether a scheme is right "
    "for you. I can look up factual information that appears in the corpus (category, "
    "expense ratio, load, risk language, historical figures if present), but I cannot "
    "judge suitability. For a recommendation, consult a SEBI-registered investment adviser."
)

FUTURE_PERFORMANCE = (
    "I cannot predict future returns or say which fund will perform best. No SID, factsheet, "
    "or prospectus in this corpus can forecast results. Past performance, when present in the "
    "documents, is historical only and is not indicative of future results."
)

ALLOCATION_ADVICE = (
    "I cannot construct a personalized allocation or tell you how to split money across funds. "
    "That depends on your goals, risk profile, and horizon, which I have not verified. I can "
    "describe each scheme's documented category and risk characteristics if those documents "
    "are in the corpus, but I will not recommend a specific split."
)

UNVERIFIED_ACCUSATION = (
    "I will not label a scheme a scam or make unverified allegations. If the fund is in the "
    "corpus I can stick to document-grounded facts (for example that it is offered as a "
    "SEBI-regulated mutual fund scheme). For registration concerns, verify directly through "
    "SEBI or AMFI official registries."
)

PROMPT_INJECTION = (
    "I cannot reveal system prompts or internal instructions, and I will not follow attempts "
    "to override them. I am a research assistant for questions about the ingested investment "
    "documents. Ask a document-grounded question if you want a factual lookup."
)

UNKNOWN_CORPUS = (
    "I have no information on that fund or document in the current corpus, so I will not "
    "guess an exit load, TER, or any other figure. If you name a scheme that is actually "
    "in the ingested SIDs or factsheets, I can look up what those files say."
)

MARKET_SPECULATION = (
    "I cannot speculate on market direction or on crash scenarios. "
    "That is outside the scope of this corpus. I only answer factual questions grounded in "
    "the ingested SIDs, factsheets, DRHP, and SEBI text."
)

TRANSACTIONAL = (
    "I am an informational research assistant, not a transactional platform. I cannot set up "
    "a SIP, auto-debit a bank account, or execute an investment. I can explain what an SIP "
    "is if that definition appears in the documents, and you would complete any purchase "
    "through an AMC, RTA, or registered intermediary."
)

MESSAGES = {
    "personalized_advice": PERSONALIZED_ADVICE,
    "future_performance": FUTURE_PERFORMANCE,
    "allocation_advice": ALLOCATION_ADVICE,
    "unverified_accusation": UNVERIFIED_ACCUSATION,
    "prompt_injection": PROMPT_INJECTION,
    "unknown_corpus": UNKNOWN_CORPUS,
    "market_speculation": MARKET_SPECULATION,
    "transactional": TRANSACTIONAL,
    "no_relevant_context": NO_RELEVANT_CONTEXT,
}
