"""Document uploads are bounded by their declared length before the body is read.

Starlette parses the multipart form -- spooling every byte -- before the route
runs, and the route then reads each file into memory to relay it. A limit
checked anywhere after that is too late, so the route class checks the
declared ``Content-Length`` first.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings as ServerSettings
from client_backend.api import documents as documents_api
from client_backend.core.config import ClientSettings, client_settings
from client_backend.services.server_api import UploadProxyResponse

_LIMIT = 1_000
_UPLOAD_PATHS = [
    "/documents/upload",
    "/documents/uploads",
    "/api/documents/upload",
    "/api/documents/uploads",
]


class _CapturingServerClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def upload_document_bytes(self, **_kwargs) -> dict:
        self.calls.append("upload")
        return {"success": True}

    async def upload_documents_bytes_with_status(self, **_kwargs) -> UploadProxyResponse:
        self.calls.append("uploads")
        return UploadProxyResponse(status_code=201, payload={"success": True})


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setattr(client_settings, "document_upload_max_bytes", _LIMIT)
    server_client = _CapturingServerClient()
    monkeypatch.setattr(documents_api, "get_server_client", lambda: server_client)
    return server_client


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(documents_api.router)
    app.include_router(documents_api.router, prefix="/api")
    app.dependency_overrides[documents_api.require_local_session] = lambda: object()
    return app


def _file_field(path: str) -> str:
    return "files" if path.endswith("/uploads") else "file"


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_an_upload_declared_too_large_is_refused_and_never_relayed(upstream, path):
    oversized = b"%PDF" + b"\0" * (_LIMIT + 256 * 1024)

    with TestClient(_build_app()) as client:
        response = client.post(
            path,
            data={"conversation_id": "conv-1"},
            files={_file_field(path): ("big.pdf", oversized, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "DOCUMENT_UPLOAD_TOO_LARGE"
    assert upstream.calls == []


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_an_upload_within_the_limit_is_still_relayed(upstream, path):
    with TestClient(_build_app()) as client:
        response = client.post(
            path,
            data={"conversation_id": "conv-1"},
            files={_file_field(path): ("small.pdf", b"%PDF" + b"\0" * 100, "application/pdf")},
        )

    assert response.status_code == 201
    assert len(upstream.calls) == 1


async def _drive(app: FastAPI, path: str, headers: list[tuple[bytes, bytes]]) -> tuple[int, int]:
    """Call the ASGI app directly: returns (status, how often the body was read)."""
    reads = 0
    status: list[int] = []

    async def receive():
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"multipart/form-data; boundary=x"), *headers],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8100),
    }
    await app(scope, receive, send)
    return status[0], reads


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _UPLOAD_PATHS)
async def test_a_declared_oversized_body_is_never_received(upstream, path):
    status, reads = await _drive(_build_app(), path, [(b"content-length", b"1000000000")])

    assert status == 413
    assert reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _UPLOAD_PATHS)
async def test_an_upload_without_a_declared_length_is_refused(upstream, path):
    """Chunked transfer would otherwise bypass the length check."""
    status, reads = await _drive(_build_app(), path, [])

    assert status == 411
    assert reads == 0
    assert upstream.calls == []


def test_the_default_limit_matches_the_servers_upload_limit():
    server_limit = ServerSettings.model_fields["max_file_size_mb"].default * 1024 * 1024

    assert ClientSettings.model_fields["document_upload_max_bytes"].default == server_limit


@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_limit_is_refused_at_startup(value):
    with pytest.raises(ValidationError):
        ClientSettings(document_upload_max_bytes=value)
