import json
import logging
from pathlib import Path

import pytest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from config import settings
from rag import (
    FALLBACK,
    SYSTEM_PROMPT,
    answer_all,
    chunk_documents,
    flatten_json,
    load_document,
    parse_questions,
)


class StubStore:
    """similarity_search with canned passages; records nothing itself."""

    def __init__(self, passages):
        self.passages = passages

    def similarity_search(self, question, k=5):
        return self.passages[:k]


class StubLLM:
    """Records every prompt it sees and answers `A<n>` in call order."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        return AIMessage(f"A{len(self.calls)}")

NAVE_PDF = "Nave-SOC2-Type-2-Report.pdf"  # public sample, not in the repo

TOY = {
    "hosting": {"provider": "Amazon Web Services (AWS)", "primary_region": "us-east-1"},
    "monitoring": {"APM": "enabled", "EUM": False, "incidents": 0, "DEM": None},
    "vendors": [{"name": "Acme"}],
}


# --- parse_questions ---

@pytest.mark.parametrize("payload", [
    b'[" Q1 ", "Q1"]',
    b'{"questions": ["Q1", " Q1"]}',
    b'[{"question": "Q1"}, {"question": "Q1 "}]',
])
def test_all_shapes_parse_the_same_and_keep_duplicates(payload):
    assert parse_questions(payload) == ["Q1", "Q1"]


@pytest.mark.parametrize("payload", [
    b"", b"[]", b"not json", b'["ok", ""]', b'["ok", 3]', b'{"nope": []}',
    b",id,question\n0,abc,Q1\n",  # CSV is not an accepted questions format
])
def test_bad_questions_raise(payload):
    with pytest.raises(ValueError):
        parse_questions(payload)


def test_real_sample_questions_parse():
    with open("samples/questions.json", "rb") as fh:
        questions = parse_questions(fh.read())
    assert questions[0] == "Where are your data centres located?"


# --- flatten_json: falsy values must survive ---

def test_flatten_preserves_paths_and_falsy_values():
    lines = flatten_json(TOY)
    assert "hosting.provider: \"Amazon Web Services (AWS)\"" in lines
    assert "monitoring.EUM: false" in lines
    assert "monitoring.incidents: 0" in lines
    assert "monitoring.DEM: null" in lines
    assert "vendors[0].name: \"Acme\"" in lines


# --- load_document ---

def test_json_becomes_one_document_per_top_level_key():
    docs = load_document("toy.json", json.dumps(TOY).encode())
    assert [d.metadata["source_id"] for d in docs] == [
        "json:hosting", "json:monitoring", "json:vendors"
    ]


def test_json_scalar_or_list_root_becomes_one_section():
    docs = load_document("toy.json", b'[{"a": 1}]')
    assert len(docs) == 1 and docs[0].metadata["source_id"] == "json:root"


@pytest.mark.parametrize("filename,data", [
    ("doc.txt", b"hello"),          # unsupported extension
    ("doc.json", b"{oops"),         # malformed JSON
    ("doc.pdf", b"not a pdf"),      # unreadable PDF
])
def test_bad_documents_raise(filename, data):
    with pytest.raises(ValueError):
        load_document(filename, data)


@pytest.mark.skipif(not Path(NAVE_PDF).exists(), reason=f"{NAVE_PDF} not checked in; see README")
def test_pdf_pages_keep_one_based_page_numbers():
    docs = load_document(NAVE_PDF, Path(NAVE_PDF).read_bytes())
    assert len(docs) > 1
    assert docs[0].metadata["page"] == 1
    assert [d.metadata["page"] for d in docs] == sorted(d.metadata["page"] for d in docs)


# --- chunking ---

def test_chunks_are_bounded_and_keep_provenance():
    docs = load_document("toy.json", json.dumps(TOY).encode())
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)
    assert {c.metadata["source_id"] for c in chunks} == {d.metadata["source_id"] for d in docs}
    assert [c.metadata["chunk_id"] for c in chunks] == [f"chunk-{i}" for i in range(len(chunks))]


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
    with caplog.at_level(logging.INFO, logger="rag"):
        answer_all(store, ["Q1", "Q2", "Q1"], StubLLM())
    summary = caplog.messages[-1]
    assert "answered 3 questions (2 distinct, 0 abstained) in" in summary


def test_failures_are_logged_with_the_question_and_re_raised(caplog):
    class BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("provider down")

    store = StubStore([Document("AWS", metadata={"source_id": "json:hosting"})])
    with caplog.at_level(logging.ERROR, logger="rag"), pytest.raises(RuntimeError):
        answer_all(store, ["Q1", "Q2"], BoomLLM())
    assert "q1/2: failed" in caplog.text
    assert "Q1" not in caplog.text  # position only: question text is not logged
    assert "provider down" in caplog.text  # traceback is kept


def test_abstention_is_logged_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="rag"):
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
