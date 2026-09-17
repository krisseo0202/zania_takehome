"""Grounded generation: ordering, deduplication, abstention, logging."""

import logging

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.answering import FALLBACK, SYSTEM_PROMPT, answer_all
from tests.conftest import StubLLM, StubStore

# --- answering ---

def test_duplicate_questions_are_answered_once_but_returned_twice():
    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    llm = StubLLM()
    results = answer_all(store, ["Q1", "Q2", "Q1"], llm)
    assert [r["question"] for r in results] == ["Q1", "Q2", "Q1"]
    assert [r["answer"] for r in results] == ["A1", "A2", "A1"]
    assert len(llm.calls) == 2  # the repeat costs nothing


def test_empty_retrieval_abstains_without_a_model_call():
    llm = StubLLM()
    results = answer_all(StubStore([]), ["Q1"], llm)
    assert results == [{"question": "Q1", "answer": FALLBACK}]
    assert llm.calls == []


def test_logs_report_counts_and_timing(caplog):
    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    with caplog.at_level(logging.INFO, logger="app.answering"):
        answer_all(store, ["Q1", "Q2", "Q1"], StubLLM())
    summary = caplog.messages[-1]
    assert "answered 3 questions (2 distinct, 0 abstained) in" in summary


def test_failures_are_logged_with_the_question_and_re_raised(caplog):
    class BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("provider down")

    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    with caplog.at_level(logging.ERROR, logger="app.answering"), pytest.raises(RuntimeError):
        answer_all(store, ["Q1", "Q2"], BoomLLM())
    assert "q1/2: failed" in caplog.text
    assert "Q1" not in caplog.text  # position only: question text is not logged
    assert "provider down" in caplog.text  # traceback is kept


def test_abstention_is_logged_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="app.answering"):
        answer_all(StubStore([]), ["Q1"], StubLLM())
    assert "no passages retrieved" in caplog.text


def test_context_reaches_the_model_with_provenance():
    store = StubStore([
        Document("hosted on AWS", metadata={"chunk_id": "chunk-0", "source_id": "json:hosting"}),
        Document("AES-256 at rest", metadata={"chunk_id": "chunk-9", "source_id": "json:security"}),
    ])
    llm = StubLLM()
    answer_all(store, ["Where?"], llm)
    prompt = llm.calls[0]
    assert "[chunk-0 | json:hosting]\nhosted on AWS" in prompt
    assert "[chunk-9 | json:security]\nAES-256 at rest" in prompt
    assert "\n---\n" in prompt  # excerpts stay separable, not one blurred blob


def test_answer_survives_a_block_style_model_response():
    class BlockLLM:
        def invoke(self, messages):
            return AIMessage([{"type": "text", "text": "AWS"}])

    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    assert answer_all(store, ["Where?"], BlockLLM())[0]["answer"] == "AWS"


def test_empty_model_response_raises_instead_of_abstaining():
    class SilentLLM:
        def invoke(self, messages):
            return AIMessage("   ")

    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    with pytest.raises(RuntimeError, match="empty answer"):
        answer_all(store, ["Where?"], SilentLLM())


def test_prompt_and_code_abstain_with_the_same_literal():
    # A configured FALLBACK must reach the model, or answers stop matching.
    assert SYSTEM_PROMPT.count(FALLBACK) == 2
