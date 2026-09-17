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
    }


def test_health_publishes_the_configured_fallback(client, monkeypatch):
    # The front end filters answers on this literal; a config change must
    # reach it, or it counts abstentions against a stale string.
    monkeypatch.setattr(settings, "fallback", "No Evidence")
    assert client.get("/health").json()["fallback"] == "No Evidence"


def test_answer_returns_one_result_per_question_in_order(client):
    res = post(client)
    assert res.status_code == 200
    assert res.json() == {"document": "toy.json", "results": [
        {"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"},
    ]}


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
