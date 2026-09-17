"""HTTP layer: status codes and response shape. No network, ever."""

import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from main import create_app
from tests.conftest import NAVE_PDF, TOY, requires_nave_pdf


def post(client, questions=b'["Q1", "Q2"]', document=None, filename="toy.json"):
    document = document if document is not None else json.dumps(TOY).encode()
    return client.post("/answer", files={
        "questions_file": ("questions.json", questions, "application/json"),
        "document_file": (filename, document, "application/octet-stream"),
    })


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "fallback": settings.fallback,
        "max_file_mb": settings.max_file_mb,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "temperature": settings.temperature,
    }


def test_health_publishes_the_configured_fallback(client, monkeypatch):
    # The front end filters answers on this literal; a config change must
    # reach it, or it counts abstentions against a stale string.
    monkeypatch.setattr(settings, "fallback", "No Evidence")
    assert client.get("/health").json()["fallback"] == "No Evidence"


def test_answer_returns_one_result_per_question_in_order(client):
    res = post(client)
    assert res.status_code == 200
    body = res.json()
    assert body["document"] == "toy.json"
    assert [(r["question"], r["answer"]) for r in body["results"]] == [("Q1", "A1"), ("Q2", "A2")]
    # Every answer carries the evidence it was drawn from.
    assert all(r["sources"] for r in body["results"])
    assert body["usage"]["embedding_tokens"] > 0


@requires_nave_pdf
def test_real_pdf_and_sample_questions_round_trip(client):
    with open(NAVE_PDF, "rb") as fh, open("samples/questions.json", "rb") as qs:
        pdf, questions = fh.read(), qs.read()
    res = post(client, questions=questions, document=pdf, filename=NAVE_PDF)
    assert res.status_code == 200
    assert len(res.json()["results"]) == len(json.loads(questions))


@pytest.mark.parametrize("kwargs", [
    dict(questions=b"[]"),                        # empty question list
    dict(questions=b"not json"),                  # unreadable questions
    dict(document=b"hello", filename="doc.txt"),  # unsupported document type
    dict(document=b"nope", filename="doc.pdf"),   # unreadable PDF
])
def test_bad_input_is_400(client, kwargs):
    res = post(client, **kwargs)
    assert res.status_code == 400
    assert res.json()["detail"]


def test_missing_file_is_422(client):
    res = client.post("/answer", files={"questions_file": ("q.json", b'["Q1"]')})
    assert res.status_code == 422


def test_oversize_upload_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_file_mb", 0)
    res = post(client)
    assert res.status_code == 413
    assert "exceeds 0 MB" in res.json()["detail"]


def test_provider_failure_is_502():
    class DownService:
        def answer_document(self, filename, document_bytes, questions_bytes):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    res = post(TestClient(create_app(service=DownService())))
    assert res.status_code == 502
    assert "provider failed" in res.json()["detail"]
