"""The demo's document upload declares its length, which the sidecar requires.

The sidecar answers 411 to a document upload with no ``Content-Length``
(``client_backend/api/upload_limits.py``), because a chunked body could not be
bounded before it is parsed. This pins that the upload the demo actually sends,
through the session it actually uses, carries one that matches the body.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import requests
from requests.adapters import BaseAdapter

from app.ui.sidecar_session import SidecarSession


class _CacheDecorator:
    def __call__(self, *args: Any, **kwargs: Any):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda func: func

    def clear(self) -> None:
        return None


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()
        self.errors: list[str] = []

    def error(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.errors.append(message)

    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _CapturingAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[requests.PreparedRequest] = []

    def send(self, request, **_kwargs):
        self.sent.append(request)
        response = requests.Response()
        response.status_code = 201
        response._content = b'{"success": true, "data": {"accepted_count": 1}}'
        response.headers["content-type"] = "application/json"
        response.request = request
        return response

    def close(self) -> None:
        return None


class _UploadedFile:
    name = "paper.pdf"
    type = "application/pdf"

    def getvalue(self) -> bytes:
        return b"%PDF" + b"\0" * 2_000


def test_the_document_upload_declares_a_content_length_matching_its_body(monkeypatch):
    base_url = "http://127.0.0.1:8100"
    monkeypatch.setenv("CHATBOT_API_BASE_URL", base_url)
    streamlit = _StreamlitStub()
    streamlit.session_state.update(current_conversation_id="conv-1", auth_token="session")
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    sys.modules.pop("upload_support", None)
    upload_support = importlib.import_module("upload_support")

    adapter = _CapturingAdapter()
    session = SidecarSession(base_url, token_reader=lambda: None)
    session.mount("http://", adapter)
    monkeypatch.setattr(upload_support, "get_http_session", lambda: session)

    result = upload_support.upload_documents([_UploadedFile()])

    assert streamlit.errors == []
    assert result == {"success": True, "data": {"accepted_count": 1}}
    [sent] = adapter.sent
    assert sent.url == f"{base_url}/documents/uploads"
    assert sent.headers["Content-Length"] == str(len(sent.body))
