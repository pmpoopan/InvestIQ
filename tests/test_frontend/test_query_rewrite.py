from frontend.streamlit_app import looks_like_followup, standalone_query


def test_standalone_leaves_first_question_alone():
    q = "What is the exit load on HDFC Flexi Cap Fund?"
    assert standalone_query([], q) == q
    assert looks_like_followup(q) is False


def test_followup_fills_scheme_from_prior_turn():
    history = [{"role": "user", "content": "What is the exit load on HDFC Flexi Cap Fund?"}]
    out = standalone_query(history, "What about its TER?")
    assert "HDFC Flexi Cap Fund" in out
    assert "Should I" not in out


def test_does_not_send_assistant_transcript():
    history = [
        {"role": "user", "content": "What is the SIP minimum for SBI Small Cap Fund?"},
        {"role": "assistant", "content": "Long answer with citations and numbers."},
    ]
    out = standalone_query(history, "and the benchmark?")
    assert "Long answer" not in out
    assert "SBI Small Cap Fund" in out
