"""One upload in, one response body out."""

import logging
from collections.abc import Callable

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
    texts = [chunk.page_content for chunk in chunks]
    try:
        try:
            encoding = tiktoken.encoding_for_model(settings.embedding_model)
        except KeyError:  # unknown model name: the current default tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        # tiktoken downloads its BPE file on first use; offline, the estimate
        # is the usual ~4 characters per token, and the cost line says so.
        logger.warning("tokenizer unavailable (%s); estimating embedding tokens", exc)
        return sum(len(text) // 4 for text in texts)
    return sum(len(encoding.encode(text)) for text in texts)


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

    def answer_document(
        self,
        filename: str,
        document_bytes: bytes,
        questions_bytes: bytes,
        on_progress: Callable[[str, dict], None] | None = None,
    ) -> dict:
        """Run one upload end to end.

        `on_progress(stage, detail)` is called as each stage completes, for
        callers that stream progress. It reports what actually happened, so a
        stage is announced after it finishes, never optimistically.
        """
        emit = on_progress or (lambda stage, detail: None)

        questions = parse_questions(questions_bytes)
        emit("questions", {"count": len(questions)})

        documents = load_document(filename, document_bytes)
        emit("load", {"sections": len(documents)})

        chunks = chunk_documents(documents)
        emit("chunk", {
            "chunks": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        })

        embedding_tokens = _embedding_tokens(chunks)

        with open_index(chunks, self.embeddings) as store:
            emit("index", {"vectors": len(chunks), "top_k": settings.top_k})
            results, usage = answer_all(
                store, questions, self.llm,
                on_generating=lambda spots: emit("generating", {"positions": spots}),
                on_answer=lambda done, total, answer, spots: emit("answer", {
                    "done": done,
                    "total": total,
                    "positions": spots,
                    "outcome": "abstain" if answer.text == settings.fallback else "done",
                    "confidence": answer.confidence,
                    "sources": answer.sources,
                    "ms": answer.ms,
                }),
            )

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
