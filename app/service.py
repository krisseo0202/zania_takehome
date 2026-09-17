"""One upload in, one response body out."""

from app.answering import answer_all
from app.config import get_chat_model, get_embeddings
from app.loaders import chunk_documents, load_document, parse_questions
from app.retrieval import open_index

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
        with open_index(chunks, self.embeddings) as store:
            results = answer_all(store, questions, self.llm)
        return {"document": filename, "results": results}
