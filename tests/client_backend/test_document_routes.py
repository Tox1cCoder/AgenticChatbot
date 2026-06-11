from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from client_backend.api import documents as documents_api


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.require_local_session] = lambda: object()
    return app


class _UploadClientStub:
    def __init__(self):
        self.upload_calls: list[dict] = []

    async def upload_document_bytes(
        self,
        *,
        conversation_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict:
        self.upload_calls.append(
            {
                "conversation_id": conversation_id,
                "filename": filename,
                "content": content,
                "content_type": content_type,
            }
        )
        return {"ok": True}


def test_document_routes_use_proxy_server_request(monkeypatch):
    calls: list[dict] = []

    async def _proxy(
        request: Request,
        *,
        upstream_path: str,
        params_override=None,
    ):
        calls.append(
            {
                "method": request.method,
                "path": upstream_path,
                "params": params_override,
            }
        )
        return JSONResponse({"path": upstream_path})

    monkeypatch.setattr(documents_api, "proxy_server_request", _proxy)

    with TestClient(_build_app()) as client:
        task_response = client.get("/documents/task/task-1")
        get_response = client.get("/documents/doc-1")
        list_response = client.get("/documents/conversation/conv-1")
        update_response = client.put("/documents/doc-1", json={"label": "updated"})
        delete_response = client.delete("/documents/doc-1")

    assert task_response.status_code == 200
    assert get_response.status_code == 200
    assert list_response.status_code == 200
    assert update_response.status_code == 200
    assert delete_response.status_code == 200
    assert calls == [
        {
            "method": "GET",
            "path": "/documents/task/task-1",
            "params": None,
        },
        {
            "method": "GET",
            "path": "/documents/doc-1",
            "params": None,
        },
        {
            "method": "GET",
            "path": "/documents/conversation/conv-1",
            "params": {"page": 1, "page_size": 20},
        },
        {
            "method": "PUT",
            "path": "/documents/doc-1",
            "params": None,
        },
        {
            "method": "DELETE",
            "path": "/documents/doc-1",
            "params": None,
        },
    ]


def test_upload_document_reads_file_and_forwards_bytes(monkeypatch):
    client_stub = _UploadClientStub()
    monkeypatch.setattr(documents_api, "get_server_client", lambda: client_stub)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("note.txt", b"hello", "text/plain")},
            data={"conversation_id": "conv-1"},
        )

    assert response.status_code == 201
    assert response.json() == {"ok": True}
    assert client_stub.upload_calls == [
        {
            "conversation_id": "conv-1",
            "filename": "note.txt",
            "content": b"hello",
            "content_type": "text/plain",
        }
    ]
