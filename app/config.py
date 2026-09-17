"""Tunables in one place, plus the OpenAI clients built from them."""

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")  # SecretStr: never prints itself
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # 500 rather than 1000: top_k fixes how many chunks reach the chat model,
    # not how big they are, so smaller chunks cut chat input (see examples/).
    chunk_size: int = 500  # characters, not tokens
    chunk_overlap: int = 150
    top_k: int = 5

    max_questions: int = 50
    max_concurrency: int = 5  # distinct questions answered at once; bounds the OpenAI rate
    log_level: str = "INFO"
    max_file_mb: int = 20  # per upload; both files are held in memory
    max_question_chars: int = 2000

    temperature: float = 0.0  # reproducible enough to audit run to run
    max_answer_tokens: int = 500
    request_timeout: float = 30.0  # seconds, per HTTP call
    max_retries: int = 2  # bounded: a retry loop on a big document burns budget

    fallback: str = "Data Not Available"

    # List prices per 1M tokens. They go stale, so they are configuration, and
    # every figure derived from them is reported as an estimate.
    price_per_1m_input: float = 0.15
    price_per_1m_output: float = 0.60
    price_per_1m_embedding: float = 0.02


settings = Settings()


# --- model clients -----------------------------------------------------------
# Built once and reused: each client owns an HTTP connection pool, and rebuilding
# one per request would throw that pool away. Cached, so tests can cache_clear().

def _require_key() -> SecretStr:
    if not settings.openai_api_key.get_secret_value():
        raise RuntimeError("OPENAI_API_KEY is not set: copy .env.example to .env")
    return settings.openai_api_key


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    logger.info("embeddings: %s (timeout %.0fs, retries %d)", settings.embedding_model,
                settings.request_timeout, settings.max_retries)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=_require_key(),
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    logger.info("chat: %s (temp %.1f, timeout %.0fs, retries %d)", settings.chat_model,
                settings.temperature, settings.request_timeout, settings.max_retries)
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=_require_key(),
        temperature=settings.temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
