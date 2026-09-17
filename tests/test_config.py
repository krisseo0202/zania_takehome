"""Client construction. Nothing here touches the network."""

import pytest
from pydantic import SecretStr

from config import get_chat_model, get_embeddings, settings


@pytest.fixture(autouse=True)
def fresh_clients(monkeypatch):
    """A key for construction, and no client cached across tests."""
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-not-a-real-key"))
    get_embeddings.cache_clear()
    get_chat_model.cache_clear()
    yield
    get_embeddings.cache_clear()
    get_chat_model.cache_clear()


def test_missing_key_fails_fast_with_a_readable_message(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(""))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        get_embeddings()


def test_chat_client_carries_the_configured_limits():
    llm = get_chat_model()
    assert llm.model_name == settings.chat_model == "gpt-4o-mini"  # the challenge mandates it
    assert llm.temperature == settings.temperature
    assert llm.max_tokens == settings.max_answer_tokens
    assert llm.request_timeout == settings.request_timeout
    assert llm.max_retries == settings.max_retries


def test_embeddings_client_carries_timeout_and_bounded_retries():
    embeddings = get_embeddings()
    assert embeddings.model == settings.embedding_model
    assert embeddings.request_timeout == settings.request_timeout
    assert 0 < embeddings.max_retries <= 3  # bounded: retries must not burn the budget


def test_clients_are_reused_not_rebuilt_per_call():
    assert get_embeddings() is get_embeddings()
    assert get_chat_model() is get_chat_model()


def test_key_never_appears_in_a_repr():
    assert "sk-test" not in repr(settings) + repr(settings.openai_api_key)
