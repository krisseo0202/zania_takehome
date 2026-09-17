"""Uploads to LangChain Documents: parse, validate, chunk."""

import io
import json
import logging
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.util import elapsed_ms

logger = logging.getLogger(__name__)


# --- input 1: questions ------------------------------------------------------

def parse_questions(data: bytes) -> list[str]:
    """bytes -> ordered list of questions. JSON only, keeps duplicates.

    Accepted shapes: ["q"], {"questions": ["q"]}, [{"question": "q"}].
    """
    text = data.decode("utf-8-sig").strip()
    if not text:
        raise ValueError("questions file is empty")

    raw = _questions_from_json(text)
    questions = [q.strip() for q in raw if isinstance(q, str) and q.strip()]
    if len(questions) != len(raw):
        raise ValueError("every question must be a non-blank string")
    if not questions:
        raise ValueError("questions file contains no questions")
    if len(questions) > settings.max_questions:
        raise ValueError(f"too many questions (limit {settings.max_questions})")
    if any(len(q) > settings.max_question_chars for q in questions):
        raise ValueError(f"question exceeds {settings.max_question_chars} characters")

    logger.info("parsed %d questions (%d distinct)", len(questions), len(set(questions)))
    return questions


def _questions_from_json(text: str) -> list:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"questions file must be JSON (got {exc})") from exc
    if isinstance(obj, dict):
        obj = obj.get("questions")
    if not isinstance(obj, list):
        raise ValueError('expected ["q1", ...], {"questions": [...]}, or [{"question": ...}]')
    return [item.get("question") if isinstance(item, dict) else item for item in obj]


# --- input 2: document -------------------------------------------------------

def flatten_json(value, prefix: str = "") -> list[str]:
    """Nested JSON -> `full.path: value` lines. Keeps false, 0 and null."""
    if isinstance(value, dict) and value:
        return [
            line
            for key, item in value.items()
            for line in flatten_json(item, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list) and value:
        return [
            line
            for i, item in enumerate(value)
            for line in flatten_json(item, f"{prefix}[{i}]")
        ]
    return [f"{prefix or 'value'}: {json.dumps(value)}"]


def load_document(filename: str, data: bytes) -> list[Document]:
    """bytes -> one Document per PDF page or per top-level JSON section.

    Every Document carries `source`, `source_id` (`page:3` / `json:hosting`) so
    provenance survives chunking and shows up in retrieval debugging.
    """
    start = time.perf_counter()
    name = filename.lower()
    if name.endswith(".pdf"):
        docs = _load_pdf(filename, data)
    elif name.endswith(".json"):
        docs = _load_json(filename, data)
    else:
        raise ValueError(f"unsupported document type: {filename} (want .pdf or .json)")
    if not docs:
        raise ValueError(f"no extractable text in {filename} (scanned PDF? no OCR here)")

    logger.info(
        "loaded %s: %d sections, %d chars in %d ms",
        filename, len(docs), sum(len(d.page_content) for d in docs), elapsed_ms(start),
    )
    return docs


def _load_pdf(filename: str, data: bytes) -> list[Document]:
    docs = []
    # Parsing AND extraction sit inside the boundary: a corrupt page raises
    # from extract_text(), not from PdfReader(), and that is still bad input.
    try:
        pages = PdfReader(io.BytesIO(data)).pages
        for i, page in enumerate(pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(text, metadata={
                    "source": filename, "page": i, "source_id": f"page:{i}",
                }))
    except Exception as exc:
        raise ValueError(f"cannot read PDF {filename}: {exc}") from exc

    if len(docs) < len(pages):
        # Not an error, but the answers only cover the pages we could read.
        logger.warning("%s: %d of %d pages had no text layer",
                       filename, len(pages) - len(docs), len(pages))
    return docs


def _load_json(filename: str, data: bytes) -> list[Document]:
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read JSON {filename}: {exc}") from exc
    sections = obj.items() if isinstance(obj, dict) and obj else [("root", obj)]
    return [
        Document("\n".join(flatten_json(value, key)), metadata={
            "source": filename, "path": key, "source_id": f"json:{key}",
        })
        for key, value in sections
    ]


# --- chunking ----------------------------------------------------------------

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split into overlapping chunks, keeping page/path provenance."""
    start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{i}"

    logger.info(
        "chunked %d documents into %d chunks in %d ms",
        len(documents), len(chunks), elapsed_ms(start),
    )
    return chunks


