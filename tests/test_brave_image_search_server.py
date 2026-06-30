"""Contract tests for the Brave Image Search MCP server (image_search.md Phase 3).

The tool returns normalized JSON (never raw Brave output) with a bounded image
list. Network access is faked by patching ``httpx.Client`` so these tests are
deterministic and offline.
"""

from __future__ import annotations

import json

import httpx

from app.ai.mcp_servers import brave_image_search_server as srv


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, *, payload, exc, recorder, **kwargs):
        self._payload = payload
        self._exc = exc
        self._recorder = recorder
        self._init_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None, params=None):
        self._recorder.append(
            {
                "url": url,
                "headers": headers or {},
                "params": params or {},
                "timeout": self._init_kwargs.get("timeout"),
            }
        )
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._payload)


def _install_fake_httpx(monkeypatch, *, payload=None, exc=None):
    calls: list[dict] = []

    def factory(**kwargs):
        return _FakeClient(payload=payload or {}, exc=exc, recorder=calls, **kwargs)

    monkeypatch.setattr(srv.httpx, "Client", factory)
    return calls


def _representative_brave_payload() -> dict:
    return {
        "results": [
            {
                "title": "Sagrada Familia exterior",
                "url": "https://example.com/sagrada-page",
                "source": "example.com",
                "thumbnail": {"src": "https://img.test/thumb-1.jpg"},
                "properties": {
                    "url": "https://img.test/direct-1.jpg",
                    "width": 1200,
                    "height": 800,
                },
                "meta_url": {"hostname": "example.com"},
            },
            {
                # Missing properties.url — must be skipped (page URL is not renderable).
                "title": "No direct image",
                "url": "https://example.com/no-image-page",
                "thumbnail": {"src": "https://img.test/thumb-2.jpg"},
                "properties": {"width": 400, "height": 300},
            },
        ]
    }


def test_missing_api_key_returns_json_error(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(srv.settings, "brave_search_api_key", "", raising=False)
    calls = _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    result = json.loads(srv.brave_image_search("spain architecture"))

    assert "error" in result
    assert "BRAVE_SEARCH_API_KEY" in result["error"]
    assert calls == []  # never hit the network without a key


def test_request_includes_required_headers_and_default_safesearch(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    calls = _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    srv.brave_image_search("spain architecture")

    assert len(calls) == 1
    headers = calls[0]["headers"]
    assert headers.get("Accept") == "application/json"
    assert headers.get("Accept-Encoding") == "gzip"
    assert headers.get("X-Subscription-Token") == "test-key"
    # Default safesearch is strict and is forwarded to Brave.
    assert calls[0]["params"].get("safesearch") == "strict"


def test_count_is_clamped_to_config_maximum(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(srv.settings, "brave_image_search_max_count", 10, raising=False)
    calls = _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    srv.brave_image_search("spain architecture", count=50)

    assert calls[0]["params"]["count"] == 10


def test_only_supported_safesearch_values_are_sent(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    calls = _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    srv.brave_image_search("cats", safesearch="off")
    assert calls[-1]["params"]["safesearch"] == "off"

    # Invalid safesearch must not reach Brave; a structured argument error is returned.
    before = len(calls)
    result = json.loads(srv.brave_image_search("cats", safesearch="moderate"))
    assert "error" in result
    assert len(calls) == before  # no extra network call for the invalid request


def test_timeout_returns_retryable_json_error(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, exc=httpx.TimeoutException("slow"))

    result = json.loads(srv.brave_image_search("spain architecture"))

    assert "error" in result
    assert result.get("retryable") is True


def test_provider_exception_returns_json_error(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, exc=RuntimeError("boom"))

    result = json.loads(srv.brave_image_search("spain architecture"))

    assert "error" in result


def test_response_is_normalized_and_skips_items_without_direct_url(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, payload=_representative_brave_payload())

    result = json.loads(srv.brave_image_search("spain architecture"))

    assert result["provider"] == "brave_image_search"
    assert result["query"] == "spain architecture"
    images = result["images"]
    # Second result lacked properties.url and is dropped.
    assert len(images) == 1
    assert result["total_results"] == 1

    img = images[0]
    assert img["url"] == "https://img.test/direct-1.jpg"  # properties.url, not page url
    assert img["source_url"] == "https://example.com/sagrada-page"  # result.url
    assert img["thumbnail_url"] == "https://img.test/thumb-1.jpg"  # thumbnail.src
    assert img["width"] == 1200
    assert img["height"] == 800
    assert img["title"] == "Sagrada Familia exterior"
    assert img["provider"] == "brave_image_search"


def test_empty_results_returns_empty_images(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    _install_fake_httpx(monkeypatch, payload={"results": []})

    result = json.loads(srv.brave_image_search("nothing here"))

    assert result["images"] == []
    assert result["total_results"] == 0
