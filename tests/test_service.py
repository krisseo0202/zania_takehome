"""One upload end to end, and the production client wiring."""

import json

import pytest

from app.config import settings
from app.service import DocumentQAService
from tests.conftest import TOY, StubLLM

def test_service_runs_the_whole_pipeline(embeddings):
    service = DocumentQAService(embeddings, StubLLM())
    body = service.answer_document("toy.json", json.dumps(TOY).encode(), b'["Q1", "Q1"]')
    assert body["document"] == "toy.json"
    assert [(r["question"], r["answer"]) for r in body["results"]] == [("Q1", "A1"), ("Q1", "A1")]
    assert body["results"][0]["sources"] == body["results"][1]["sources"]  # cached, same evidence
    assert body["usage"]["embedding_tokens"] > 0


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


def test_estimate_cost_uses_configured_list_prices():
    from app.service import estimate_cost_usd

    # 1M input + 1M output + 1M embedding tokens = the three prices summed.
    expected = (
        settings.price_per_1m_input
        + settings.price_per_1m_output
        + settings.price_per_1m_embedding
    )
    assert estimate_cost_usd(1_000_000, 1_000_000, 1_000_000) == round(expected, 6)
    assert estimate_cost_usd(0, 0, 0) == 0.0


def test_sources_name_the_pages_or_sections_evidence_came_from(embeddings):
    body = DocumentQAService(embeddings, StubLLM()).answer_document(
        "toy.json", json.dumps(TOY).encode(), b'["Q1"]'
    )
    sources = body["results"][0]["sources"]
    assert sources, "an answered question must cite its evidence"
    assert all(s.startswith("json:") for s in sources)
    assert len(sources) == len(set(sources))  # deduplicated
