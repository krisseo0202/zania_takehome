"""Index lifecycle. Fake embeddings test integration, not retrieval quality."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.answering import answer_all
from app.loaders import chunk_documents, load_document
from app.retrieval import build_index, open_index, retrieve
from tests.conftest import StubLLM


def test_retrieve_returns_k_chunks_with_provenance_intact(embeddings, toy_chunks):
    with open_index(toy_chunks, embeddings) as store:
        passages = retrieve(store, "Which cloud provider?", k=2)
    assert len(passages) == 2
    assert all(p.metadata["source_id"].startswith("json:") for p in passages)
    assert all(p.metadata["chunk_id"].startswith("chunk-") for p in passages)


def test_document_is_embedded_once_for_the_whole_question_list(embeddings, toy_chunks):
    with open_index(toy_chunks, embeddings) as store:
        answer_all(store, ["Q1", "Q2", "Q1"], StubLLM())

    assert sum(embeddings.doc_calls) == len(toy_chunks)  # indexed once, not per question
    assert embeddings.query_calls == 2                   # one per distinct question


def test_two_uploads_do_not_contaminate_each_other(embeddings):
    aws = chunk_documents(load_document("a.json", b'{"hosting": {"provider": "AWS"}}'))
    azure = chunk_documents(load_document("b.json", b'{"hosting": {"provider": "Azure"}}'))

    with open_index(aws, embeddings) as store_a, open_index(azure, embeddings) as store_b:
        text_a = " ".join(p.page_content for p in retrieve(store_a, "provider"))
        text_b = " ".join(p.page_content for p in retrieve(store_b, "provider"))

    assert "AWS" in text_a and "Azure" not in text_a
    assert "Azure" in text_b and "AWS" not in text_b


def test_collection_is_deleted_even_when_answering_raises(embeddings, toy_chunks):
    class BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        with open_index(toy_chunks, embeddings) as store:
            answer_all(store, ["Q1"], BoomLLM())

    # Gone, not merely out of scope: the store is unusable after cleanup.
    with pytest.raises(ValueError, match="not initialized"):
        retrieve(store, "Q1")


def test_cleanup_failure_does_not_mask_the_original_error(embeddings, toy_chunks, caplog):
    class BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):  # not the cleanup error
        with open_index(toy_chunks, embeddings) as store:
            store.delete_collection = lambda: (_ for _ in ()).throw(OSError("chroma busy"))
            answer_all(store, ["Q1"], BoomLLM())

    assert "could not delete collection" in caplog.text  # cleanup failure still visible


def test_partial_indexing_failure_drops_its_own_collection(toy_chunks):
    class BrokenEmbeddings(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            raise RuntimeError("embedding provider down")

    with pytest.raises(RuntimeError):
        build_index(toy_chunks, BrokenEmbeddings(size=64))


