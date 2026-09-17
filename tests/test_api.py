import pytest
pytest.importorskip("main", reason="HTTP layer not implemented yet")

"""HTTP layer: status codes and response shape. No network, ever.

The `client` fixture (tests/conftest.py) wires in the same fakes as
test_index.py through `create_app(service=...)`; only the 502 test below
builds its own app, since it needs a service that raises instead of answers.
"""

import io
import json

import httpx
import openai
from fastapi.testclient import TestClient

from main import MAX_FILE_MB, create_app

TOY_DOC = json.dumps({"hosting": {"provider": "AWS"}}).encode()
QUESTIONS = json.dumps(["Q1", "Q2"]).encode()


def _files(questions=QUESTIONS, doc_name="toy.json", doc_bytes=TOY_DOC):
    return {
        "questions_file": ("questions.json", io.BytesIO(questions), "application/json"),
        "document_file": (doc_name, io.BytesIO(doc_bytes), "application/octet-stream"),
    }


def test_health_is_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_answer_happy_path_preserves_order_and_count(client):
    resp = client.post("/answer", files=_files())
    assert resp.status_code == 200
    body = resp.json()
    assert body["document"] == "toy.json"
    assert [r["question"] for r in body["results"]] == ["Q1", "Q2"]
    assert len(body["results"]) == 2


def test_unsupported_document_extension_is_400(client):
    resp = client.post("/answer", files=_files(doc_name="doc.txt", doc_bytes=b"hello"))
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_malformed_questions_json_is_400(client):
    resp = client.post("/answer", files=_files(questions=b"not json"))
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_empty_questions_list_is_400(client):
    resp = client.post("/answer", files=_files(questions=b"[]"))
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_oversize_upload_is_413(client):
    oversize = b"0" * (MAX_FILE_MB * 1024 * 1024 + 1)
    resp = client.post("/answer", files=_files(doc_bytes=oversize))
    assert resp.status_code == 413
    assert "detail" in resp.json()


def test_provider_failure_is_502():
    class BoomService:
        def answer_document(self, filename, document_bytes, questions_bytes):
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            )

    client = TestClient(create_app(service=BoomService()))
    resp = client.post("/answer", files=_files())
    assert resp.status_code == 502
    assert "detail" in resp.json()
