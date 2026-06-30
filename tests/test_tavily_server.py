from __future__ import annotations

import json

from app.ai.mcp_servers import tavily_server


def test_tavily_clamp_count_uses_default_and_cap(monkeypatch):
    monkeypatch.setattr(
        tavily_server.settings, "tavily_search_default_max_results", 5, raising=False
    )
    monkeypatch.setattr(tavily_server.settings, "tavily_search_max_results", 10, raising=False)

    assert tavily_server._clamp_int(None, default=5, minimum=1, maximum=10) == 5
    assert tavily_server._clamp_int(25, default=5, minimum=1, maximum=10) == 10
    assert tavily_server._clamp_int("bad", default=5, minimum=1, maximum=10) == 5


def test_tavily_error_payload_is_compact_json():
    payload = json.loads(tavily_server._error("missing key", operation="search", retryable=False))

    assert payload == {
        "error": "missing key",
        "provider": "tavily",
        "operation": "search",
        "retryable": False,
    }


class _FakeTavilyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_search_uses_configured_basic_depth_and_preserves_images(monkeypatch):
    client = _FakeTavilyClient(
        {
            "query": "openai news",
            "answer": "",
            "images": [{"url": "https://example.com/a.jpg", "description": "A"}],
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.9,
                    "raw_content": "Full text",
                }
            ],
            "usage": {"credits": 1},
            "request_id": "req-1",
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    monkeypatch.setattr(
        tavily_server.settings, "tavily_search_default_depth", "basic", raising=False
    )
    monkeypatch.setattr(
        tavily_server.settings, "tavily_search_auto_parameters", False, raising=False
    )

    payload = json.loads(tavily_server.tavily_search("openai news", max_results=25))

    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 10
    assert payload["provider"] == "tavily"
    assert payload["operation"] == "search"
    assert payload["images"][0]["url"] == "https://example.com/a.jpg"
    assert payload["results"][0]["raw_content"] == "Full text"
    assert payload["usage"] == {"credits": 1}


class _FakeExtractClient:
    def __init__(self):
        self.calls = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "url": "https://example.com/a",
                    "raw_content": "# Title\nBody",
                    "images": ["https://example.com/a.png"],
                    "favicon": "https://example.com/favicon.ico",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 1},
            "request_id": "req-extract",
        }


def test_extract_accepts_string_url_and_query_rerank(monkeypatch):
    client = _FakeExtractClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    monkeypatch.setattr(tavily_server.settings, "tavily_extract_max_urls", 5, raising=False)

    payload = json.loads(
        tavily_server.tavily_extract("https://example.com/a", query="pricing", include_images=True)
    )

    assert client.calls[0]["urls"] == ["https://example.com/a"]
    assert client.calls[0]["query"] == "pricing"
    assert client.calls[0]["extract_depth"] == "basic"
    assert client.calls[0]["format"] == "markdown"
    assert payload["operation"] == "extract"
    assert payload["results"][0]["raw_content"] == "# Title\nBody"
    assert payload["results"][0]["images"] == ["https://example.com/a.png"]
