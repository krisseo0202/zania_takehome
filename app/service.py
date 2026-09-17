"""One upload in, one response body out."""

import logging

import tiktoken

from app.answering import answer_all
from app.config import get_chat_model, get_embeddings, settings
from app.loaders import chunk_documents, load_document, parse_questions
from app.retrieval import open_index

logger = logging.getLogger(__name__)


def _embedding_tokens(chunks) -> int:
    """Count what indexing sent to the embedding model.

    The embeddings API returns no usage through LangChain, so this is counted
    locally with the model's own tokenizer rather than guessed at.
    """
    try:
        encoding = tiktoken.encoding_for_model(settings.embedding_model)
    except Exception:  # unknown model name: fall back to the current default
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(chunk.page_content)) for chunk in chunks)


def estimate_cost_usd(input_tokens: int, output_tokens: int, embedding_tokens: int) -> float:
    """List-price estimate, not a bill. Prices are configuration and go stale."""
    dollars = (
        input_tokens * settings.price_per_1m_input
        + output_tokens * settings.price_per_1m_output
        + embedding_tokens * settings.price_per_1m_embedding
    ) / 1_000_000
    return round(dollars, 6)


class DocumentQAService:
    """Holds the model clients and runs one upload end to end.

    Built once per process; `answer_document` keeps no state between calls, so
    concurrent requests are fine. Tests pass fake `embeddings` / `llm`.
    """

    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm

    @classmethod
    def from_settings(cls) -> "DocumentQAService":
        """Real OpenAI clients: key checked, timeouts and retries bounded (config.py)."""
        return cls(get_embeddings(), get_chat_model())

    def answer_document(self, filename: str, document_bytes: bytes, questions_bytes: bytes) -> dict:
        questions = parse_questions(questions_bytes)
        chunks = chunk_documents(load_document(filename, document_bytes))
        embedding_tokens = _embedding_tokens(chunks)

        with open_index(chunks, self.embeddings) as store:
            results, usage = answer_all(store, questions, self.llm)

        return {
            "document": filename,
            "results": results,
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "embedding_tokens": embedding_tokens,
                "estimated_cost_usd": estimate_cost_usd(
                    usage["input_tokens"], usage["output_tokens"], embedding_tokens
                ),
            },
        }
