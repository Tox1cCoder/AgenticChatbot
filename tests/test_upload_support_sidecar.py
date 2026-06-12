from __future__ import annotations

import importlib
import sys
import types
from typing import Any


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
        self.session_state: dict[str, Any] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()

    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any):
            return None

        return _noop


class _Response:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"success": True, "data": {"documents": []}}


class _RecordingSession:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, Any]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: tuple[int, int]):
        self.get_calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _Response()


def _import_upload_support(monkeypatch, *, base_url: str):
    monkeypatch.setenv("CHATBOT_API_BASE_URL", base_url)
    monkeypatch.setitem(sys.modules, "streamlit", _StreamlitStub())
    sys.modules.pop("upload_support", None)
    return importlib.import_module("upload_support")


def test_conversation_document_list_uses_configured_sidecar_base_url(monkeypatch):
    upload_support = _import_upload_support(
        monkeypatch,
        base_url="http://127.0.0.1:8100",
    )
    session = _RecordingSession()
    monkeypatch.setattr(upload_support, "get_http_session", lambda: session)

    result = upload_support._cached_conversation_documents(
        conversation_id="conv-1",
        auth_token="local-session-token",
        cache_version=0,
    )

    assert result == {"success": True, "data": {"documents": []}}
    assert session.get_calls == [
        {
            "url": "http://127.0.0.1:8100/documents/conversation/conv-1",
            "headers": {"Authorization": "Bearer local-session-token"},
            "timeout": upload_support.REQUEST_TIMEOUT,
        }
    ]
