"""HTTP entrypoint: POST /answer (two uploads in, answers out), GET /health.

    uvicorn main:app --port 8000

Error contract: 400 bad questions/document, 413 upload too large, 422 missing
field (FastAPI's own), 502 OpenAI failed after the bounded retries.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.answering import EmptyAnswer
from app.config import settings
from app.service import DocumentQAService
from app.util import elapsed_ms

logger = logging.getLogger(__name__)
DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"  # React build, if any


def create_app(service: DocumentQAService | None = None) -> FastAPI:
    """`service=None` builds real clients at startup; tests pass one with fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fails the process at boot, not the first request, when the key is missing.
        app.state.service = service or DocumentQAService.from_settings()
        yield

    app = FastAPI(title="Zania QA Bot", lifespan=lifespan)
    app.state.service = service  # set now too, so TestClient works without `with`
    app.add_middleware(  # the Vite dev server; the built app is same-origin
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/answer")
    async def answer(questions_file: UploadFile, document_file: UploadFile):
        start = time.perf_counter()
        questions = await _read_limited(questions_file)
        document = await _read_limited(document_file)
        filename = document_file.filename or "document"
        try:
            # Embedding and Chroma are blocking; keep the event loop (and /health) free.
            body = await asyncio.to_thread(
                app.state.service.answer_document, filename, document, questions
            )
        except ValueError as exc:  # parse_questions / load_document reject the input
            raise HTTPException(400, str(exc)) from exc
        except (openai.APIError, EmptyAnswer) as exc:
            # APIError: rate limit, timeout, 5xx after the bounded retries.
            # EmptyAnswer: the model replied with nothing at all.
            logger.exception("%s: provider failed", filename)
            raise HTTPException(502, f"language model provider failed: {exc}") from exc

        # Filename and count only: never the questions or document text.
        logger.info("%s: %d answers in %d ms",
                    filename, len(body["results"]), elapsed_ms(start))
        return body

    if DIST_DIR.exists():
        # Mounted last so the SPA catch-all can never shadow /health or /answer.
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
    return app


async def _read_limited(upload: UploadFile) -> bytes:
    # ponytail: Starlette has already spooled the body; a Content-Length
    # middleware would reject oversize uploads before parsing.
    data = await upload.read()
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: exceeds {settings.max_file_mb} MB")
    return data


app = create_app()
