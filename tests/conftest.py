"""Shared fixtures. Fake embeddings and stub models: no test touches the network."""

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from app.loaders import chunk_documents, load_document

NAVE_PDF = "Nave-SOC2-Type-2-Report.pdf"  # public sample, not in the repo

TOY = {
    "hosting": {"provider": "Amazon Web Services (AWS)", "primary_region": "us-east-1"},
    "monitoring": {"APM": "enabled", "EUM": False, "incidents": 0, "DEM": None},
    "vendors": [{"name": "Acme"}],
}

requires_nave_pdf = pytest.mark.skipif(
    not Path(NAVE_PDF).exists(), reason=f"{NAVE_PDF} not checked in; see README"
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


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Deterministic fake vectors that record how often each side was called."""

    doc_calls: list[int] = []
    query_calls: int = 0

    def embed_documents(self, texts):
        self.doc_calls.append(len(texts))
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return super().embed_query(text)




@pytest.fixture
def embeddings():
    return CountingEmbeddings(size=64, doc_calls=[], query_calls=0)


@pytest.fixture
def toy_chunks():
    return chunk_documents(load_document("toy.json", json.dumps(TOY).encode()))


@pytest.fixture
def api_key(monkeypatch):
    from pydantic import SecretStr
    from app.config import get_chat_model, get_embeddings, settings

    def use(value):
        monkeypatch.setattr(settings, "openai_api_key", SecretStr(value))
        get_embeddings.cache_clear()
        get_chat_model.cache_clear()
    yield use
    get_embeddings.cache_clear()
    get_chat_model.cache_clear()


