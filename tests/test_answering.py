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
    results, _ = answer_all(store, ["Q1", "Q2", "Q1"], llm)
    assert [r["question"] for r in results] == ["Q1", "Q2", "Q1"]
    assert [r["answer"] for r in results] == ["A1", "A2", "A1"]
    assert len(llm.calls) == 2  # the repeat costs nothing


def test_empty_retrieval_abstains_without_a_model_call():
    llm = StubLLM()
    results, _ = answer_all(StubStore([]), ["Q1"], llm)
    assert results == [{"question": "Q1", "answer": FALLBACK, "confidence": "low", "sources": []}]
    assert llm.calls == []


def test_logs_report_counts_and_timing(caplog):
    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    with caplog.at_level(logging.INFO, logger="app.answering"):
        answer_all(store, ["Q1", "Q2", "Q1"], StubLLM())
    summary = caplog.messages[-1]
    assert "answered 3 questions (2 distinct, 0 abstained, 0+0 tokens) in" in summary


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
    assert answer_all(store, ["Where?"], BlockLLM())[0][0]["answer"] == "AWS"


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


@pytest.mark.parametrize("raw", ["Data Not Available", "Data Not Available.", "data not available"])
def test_whole_answer_abstentions_collapse_to_the_exact_literal(raw):
    class VariantLLM:
        def invoke(self, messages):
            return AIMessage(raw)

    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    assert answer_all(store, ["Q1"], VariantLLM())[0][0]["answer"] == FALLBACK


def test_partial_answers_keep_their_unavailable_parts_verbatim():
    # The guard must not swallow a real answer that marks one part unavailable.
    partial = "APM: performed. EUM: not performed. DEM: Data Not Available"

    class PartialLLM:
        def invoke(self, messages):
            return AIMessage(partial)

    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    assert answer_all(store, ["Q1"], PartialLLM())[0][0]["answer"] == partial


def test_prose_around_the_literal_is_not_silently_collapsed():
    # An explanation plus the literal is a prompt failure, not an abstention:
    # collapsing it would hide that the model ignored the instruction.
    prose = "The excerpts do not mention a Wireless Security Policy.\n\nData Not Available"

    class ProseLLM:
        def invoke(self, messages):
            return AIMessage(prose)

    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    assert answer_all(store, ["Q1"], ProseLLM())[0][0]["answer"] == prose


@pytest.mark.parametrize("raw,answer,confidence", [
    ("Yes. AWS [page:3]\nConfidence: high", "Yes. AWS [page:3]", "high"),
    ("Partially. No SLA stated.\n\nConfidence: Medium.", "Partially. No SLA stated.", "medium"),
    ("AWS", "AWS", "low"),  # no confidence line: the format was ignored, report low
    ("Data Not Available\nConfidence: high", FALLBACK, "low"),  # abstentions are always low
])
def test_confidence_line_is_split_off_the_answer(raw, answer, confidence):
    class FixedLLM:
        def invoke(self, messages):
            return AIMessage(raw)

    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    row = answer_all(store, ["Q1"], FixedLLM())[0][0]
    assert (row["answer"], row["confidence"]) == (answer, confidence)


def test_distinct_questions_are_answered_concurrently():
    import threading

    class BarrierLLM:
        """Each call waits for a second concurrent call; serial execution times out."""

        barrier = threading.Barrier(2, timeout=5)

        def invoke(self, messages):
            self.barrier.wait()
            return AIMessage("ok")

    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    results, _ = answer_all(store, ["Q1", "Q2"], BarrierLLM())
    assert [r["answer"] for r in results] == ["ok", "ok"]


def test_progress_counts_every_position_including_duplicates():
    store = StubStore([Document("x", metadata={"source_id": "json:a"})])
    seen = []
    answer_all(store, ["Q1", "Q2", "Q1"], StubLLM(),
               on_answer=lambda done, total, answer, spots: seen.append((done, total, spots)))
    assert seen[-1][:2] == (3, 3)
    assert [d for d, _, _ in seen] == sorted(d for d, _, _ in seen)  # monotone
    # The duplicate is answered once but marks both rows it occupies.
    assert sorted(spot for _, _, spots in seen for spot in spots) == [1, 2, 3]
