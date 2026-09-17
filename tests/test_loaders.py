"""Input parsing: questions, documents, chunking."""

import json

import pytest

from app.config import settings
from app.loaders import chunk_documents, flatten_json, load_document, parse_questions
from tests.conftest import NAVE_PDF, TOY, requires_nave_pdf

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


@requires_nave_pdf
def test_pdf_pages_keep_one_based_page_numbers():
    docs = load_document(NAVE_PDF, open(NAVE_PDF, "rb").read())
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
