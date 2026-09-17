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


def _stream_events(client, questions=b'["Q1", "Q2"]', document=None, filename="toy.json"):
    document = document if document is not None else json.dumps(TOY).encode()
    files = {
        "questions_file": ("questions.json", questions, "application/json"),
        "document_file": (filename, document, "application/octet-stream"),
    }
    with client.stream("POST", "/answer/stream", files=files) as res:
        assert res.status_code == 200
        return [json.loads(line) for line in res.iter_lines() if line]


def test_stream_reports_each_stage_then_the_result(client):
    events = _stream_events(client)
    stages = [e["stage"] for e in events]
    assert stages[:4] == ["questions", "load", "chunk", "index"]
    assert stages[-1] == "done"
    # Answer progress counts up to the total, one event per question.
    answers = [e for e in events if e["stage"] == "answer"]
    assert [e["done"] for e in answers] == list(range(1, len(answers) + 1))
    assert all(e["total"] == 2 for e in answers)
    # The final event carries exactly what /answer would have returned.
    assert events[-1]["result"]["document"] == "toy.json"
    assert len(events[-1]["result"]["results"]) == 2


def test_stream_reports_a_bad_upload_as_an_error_event(client):
    # Right extension, unreadable contents: the failure can only be found once
    # the pipeline starts, so it arrives as an error event rather than a status.
    events = _stream_events(client, questions=b"not json")
    assert events[-1]["stage"] == "error"
    assert events[-1]["status"] == 400
    assert "done" not in [e["stage"] for e in events]


@pytest.mark.parametrize("filename", ["questions.py", "questions.json.py", "questions", "questions.txt"])
def test_questions_file_must_be_json_by_extension(client, filename):
    # Contents are a valid JSON question list; only the extension is wrong.
    res = client.post("/answer", files={
        "questions_file": (filename, b'["Q1"]', "application/json"),
        "document_file": ("toy.json", json.dumps(TOY).encode(), "application/octet-stream"),
    })
    assert res.status_code == 400
    assert "expected .json" in res.json()["detail"]


def test_document_file_must_be_pdf_or_json_by_extension(client):
    res = client.post("/answer", files={
        "questions_file": ("questions.json", b'["Q1"]', "application/json"),
        "document_file": ("report.py", json.dumps(TOY).encode(), "application/octet-stream"),
    })
    assert res.status_code == 400
    assert "expected .pdf or .json" in res.json()["detail"]


def test_stream_rejects_a_wrong_extension_before_doing_any_work(client):
    res = client.post("/answer/stream", files={
        "questions_file": ("questions.py", b'["Q1"]', "application/json"),
        "document_file": ("toy.json", json.dumps(TOY).encode(), "application/octet-stream"),
    })
    # A 400 status, not a 200 stream carrying an error event.
    assert res.status_code == 400
