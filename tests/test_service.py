"""One upload end to end, and the production client wiring."""

import json

import pytest

from app.config import settings
from app.service import DocumentQAService
from tests.conftest import TOY, StubLLM

def test_service_runs_the_whole_pipeline(embeddings):
    service = DocumentQAService(embeddings, StubLLM())
    body = service.answer_document("toy.json", json.dumps(TOY).encode(), b'["Q1", "Q1"]')
    assert body == {"document": "toy.json", "results": [
        {"question": "Q1", "answer": "A1"}, {"question": "Q1", "answer": "A1"},
    ]}


def test_from_settings_fails_fast_without_an_api_key(api_key):
    api_key("")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        DocumentQAService.from_settings()


def test_from_settings_builds_bounded_clients(api_key):
    from app.config import settings

    api_key("sk-test")
    service = DocumentQAService.from_settings()  # constructs clients, no network
    assert service.llm.model_name == settings.chat_model
    assert service.llm.max_retries == settings.max_retries == service.embeddings.max_retries
    assert service.llm.request_timeout == settings.request_timeout == service.embeddings.request_timeout
