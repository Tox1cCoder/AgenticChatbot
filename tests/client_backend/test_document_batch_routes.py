"""Sidecar batch upload proxy guards.

Pins that ``/documents/uploads`` forwards bytes for each incoming file in
order and preserves upstream status codes for the UI to render structured
per-file results.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import documents as documents_api


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.require_local_session] = lambda: object()
    return app


class _BatchUploadClientStub:
    def __init__(self, *, status_code: int = 201, payload=None):
        from client_backend.services.server_api import UploadProxyResponse

        self.status_code = status_code
        self.payload = payload or {"data": {"accepted_count": 1, "rejected_count": 0}}
        self.calls: list[dict] = []
        self._UploadProxyResponse = UploadProxyResponse

    async def upload_documents_bytes_with_status(self, *, conversation_id: str, files):
        self.calls.append({"conversation_id": conversation_id, "files": list(files)})
        return self._UploadProxyResponse(status_code=self.status_code, payload=self.payload)


def test_batch_upload_forwards_each_file_to_server(monkeypatch):
    stub = _BatchUploadClientStub()
    monkeypatch.setattr(documents_api, "get_server_client", lambda: stub)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/documents/uploads",
            data={"conversation_id": "conv-7"},
            files=[
                ("files", ("a.pdf", b"AAA", "application/pdf")),
                ("files", ("b.pdf", b"BBB", "application/pdf")),
            ],
        )

    assert response.status_code == 201
    assert len(stub.calls) == 1
    forwarded = stub.calls[0]
    assert forwarded["conversation_id"] == "conv-7"
    assert [item["filename"] for item in forwarded["files"]] == ["a.pdf", "b.pdf"]
    assert [item["content"] for item in forwarded["files"]] == [b"AAA", b"BBB"]
    assert [item["content_type"] for item in forwarded["files"]] == [
        "application/pdf",
        "application/pdf",
    ]


def test_batch_upload_preserves_207_status_code(monkeypatch):
    stub = _BatchUploadClientStub(
        status_code=207,
        payload={
            "data": {
                "accepted_count": 1,
                "rejected_count": 1,
                "files": [
                    {"filename": "a.pdf", "status": "rejected", "error_code": "DUPLICATE_FILENAME"},
                    {"filename": "b.pdf", "status": "accepted"},
                ],
            }
        },
    )
    monkeypatch.setattr(documents_api, "get_server_client", lambda: stub)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/documents/uploads",
            data={"conversation_id": "conv-7"},
            files=[
                ("files", ("a.pdf", b"a", "application/pdf")),
                ("files", ("b.pdf", b"b", "application/pdf")),
            ],
        )

    assert response.status_code == 207
    body = response.json()
    assert body["data"]["accepted_count"] == 1
    assert body["data"]["rejected_count"] == 1


def test_batch_upload_preserves_409_status_code(monkeypatch):
    stub = _BatchUploadClientStub(
        status_code=409,
        payload={
            "data": {
                "accepted_count": 0,
                "rejected_count": 2,
            }
        },
    )
    monkeypatch.setattr(documents_api, "get_server_client", lambda: stub)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/documents/uploads",
            data={"conversation_id": "conv-7"},
            files=[
                ("files", ("a.pdf", b"a", "application/pdf")),
                ("files", ("b.pdf", b"b", "application/pdf")),
            ],
        )

    assert response.status_code == 409
