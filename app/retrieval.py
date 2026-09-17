"""Per-upload vector index and similarity search."""

import logging
import time
from contextlib import contextmanager
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.util import elapsed_ms

logger = logging.getLogger(__name__)


# --- index -------------------------------------------------------------------

def build_index(chunks: list[Document], embeddings, collection_name: str = "") -> Chroma:
    """Embed the chunks once into a fresh, uniquely named in-process collection.

    The name is unique per upload so two concurrent requests cannot read each
    other's document. A partial insert deletes its own collection rather than
    leaving half a document behind.
    """
    start = time.perf_counter()
    name = collection_name or f"upload-{uuid4().hex}"
    store = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    try:
        store.add_documents(chunks)
    except Exception:
        logger.exception("indexing failed for %s, dropping partial collection", name)
        drop_index(store, name)
        raise

    logger.info("indexed %d chunks into %s in %d ms", len(chunks), name, elapsed_ms(start))
    return store


def drop_index(store, name: str) -> None:
    """Delete a collection without ever replacing an exception in flight."""
    try:
        store.delete_collection()
        logger.info("released index %s", name)
    except Exception:
        # Cleanup must not mask the original indexing or generation failure.
        logger.exception("could not delete collection %s; leaked in this process", name)


@contextmanager
def open_index(chunks: list[Document], embeddings):
    """Request-scoped index: built once, always deleted, errors included.

    In-process storage is not automatic cleanup, so the `finally` is the point:
    answer every question for this document inside the `with` block.
    """
    name = f"upload-{uuid4().hex}"
    store = build_index(chunks, embeddings, collection_name=name)
    try:
        yield store
    finally:
        drop_index(store, name)


def retrieve(store, question: str, k: int | None = None) -> list[Document]:
    """Top-k *chunks* (not pages) by cosine similarity, same embedding model."""
    # Resolved per call, not frozen into a default at import time.
    return store.similarity_search(question, k=k or settings.top_k)
