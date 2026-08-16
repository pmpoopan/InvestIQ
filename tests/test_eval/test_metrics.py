from eval.metrics import _precision_at_k, is_placeholder_answer
from app.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from app.models.schemas import Chunk, RetrievedChunk


def _chunk(cid: str, text: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        source_document="x.pdf",
        document_type="sid",
        page_number="1",
        chunk_type="prose",
    )


def test_placeholder_detection():
    assert is_placeholder_answer("[FILL IN FROM ACTUAL SID...]")
    assert is_placeholder_answer("[FILL IN FROM ACTUAL SID ONCE COLLECTED — e.g. '1%']")
    assert not is_placeholder_answer("SIP is a systematic investment plan.")
    assert is_placeholder_answer("")


def test_precision_at_k_ragas_formula():
    # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2
    assert abs(_precision_at_k([True, False, True]) - (1.0 + 2 / 3) / 2) < 1e-9
    assert _precision_at_k([False, False]) == 0.0
    assert _precision_at_k([True, True]) == 1.0


def test_rrf_prefers_consensus():
    a = RetrievedChunk(chunk=_chunk("a"), score=0.9)
    b = RetrievedChunk(chunk=_chunk("b"), score=0.8)
    c = RetrievedChunk(chunk=_chunk("c"), score=0.7)
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]], top_k=2, rrf_k=60)
    assert [h.chunk.chunk_id for h in fused] == ["a", "b"] or fused[0].chunk.chunk_id in {"a", "b"}
    assert {h.chunk.chunk_id for h in fused} == {"a", "b"}


def test_bm25_ranks_lexical_hit():
    chunks = [
        _chunk("exit", "exit load of one percent if redeemed within 365 days"),
        _chunk("sip", "minimum sip amount is five hundred rupees"),
    ]
    index = BM25Index(chunks)
    hits = index.query("exit load 365 days", k=2)
    assert hits[0].chunk.chunk_id == "exit"
