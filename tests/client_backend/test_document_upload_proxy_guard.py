"""Phase 1 guards: client_backend remains a pure byte proxy for document uploads.

The canonical server owns document parsing, chunking, indexing, and
authorization. These tests pin that the sidecar:
  * Forwards file bytes + conversation_id unchanged to ``/documents/upload``.
  * Does not import any server-side parser/indexer module directly.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import documents as documents_api


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.require_local_session] = lambda: object()
    return app


class _CapturingServerClient:
    def __init__(self) -> None:
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
        return {"ok": True, "document_id": "doc-from-server"}


def test_upload_proxy_forwards_bytes_and_conversation_id(monkeypatch):
    client_stub = _CapturingServerClient()
    monkeypatch.setattr(documents_api, "get_server_client", lambda: client_stub)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/documents/upload",
            data={"conversation_id": "conv-123"},
            files={"file": ("paper.pdf", b"RAW-PDF-BYTES", "application/pdf")},
        )

    assert response.status_code == 201
    assert response.json() == {"ok": True, "document_id": "doc-from-server"}

    assert len(client_stub.upload_calls) == 1
    call = client_stub.upload_calls[0]
    assert call["conversation_id"] == "conv-123"
    assert call["filename"] == "paper.pdf"
    assert call["content"] == b"RAW-PDF-BYTES"
    assert call["content_type"] == "application/pdf"


def test_client_backend_does_not_import_server_parser_or_indexer():
    """The sidecar must not reach into server-side parsing/indexing code."""

    # Force a clean import so cached parents from other tests don't hide the issue.
    for mod_name in list(sys.modules):
        if mod_name.startswith("client_backend"):
            sys.modules.pop(mod_name, None)

    importlib.import_module("client_backend.api.documents")
    importlib.import_module("client_backend.services.server_api")

    forbidden_prefixes = (
        "app.services.document_processing_service",
        "app.services.document_index_service",
        "app.services.document_chunk_builder",
        "app.ai.agents.rag_agent",
        "app.ai.rag_tool_actions",
    )
    leaked: list[str] = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("client_backend"):
            continue
        module_attrs = getattr(module, "__dict__", {}) or {}
        for attr_value in module_attrs.values():
            if isinstance(attr_value, ModuleType):
                attr_name = getattr(attr_value, "__name__", "")
                if any(attr_name.startswith(prefix) for prefix in forbidden_prefixes):
                    leaked.append(f"{name} -> {attr_name}")
    assert not leaked, f"client_backend leaked server-side imports: {leaked}"
