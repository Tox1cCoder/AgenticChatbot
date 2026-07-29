from __future__ import annotations

import json

import pytest

from app.ai.mcp_servers import tavily_server
from app.ai.mcp_servers.tavily_server import (
    resolve_tavily_search_params,
    validate_tavily_query,
)


def test_explicit_depth_wins_and_disables_auto_parameters():
    resolved = resolve_tavily_search_params(
        search_depth="advanced",
        auto_parameters=True,
        default_depth="basic",
        default_auto=True,
    )
    assert resolved == {"search_depth": "advanced", "auto_parameters": False}


def test_auto_mode_omits_search_depth():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=True,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": None, "auto_parameters": True}


def test_no_depth_and_auto_disabled_sends_default_depth():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=None,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": "basic", "auto_parameters": False}


def test_omitted_auto_parameters_falls_back_to_configured_default():
    resolved = resolve_tavily_search_params(
        search_depth=None,
        auto_parameters=None,
        default_depth="basic",
        default_auto=True,
    )
    assert resolved == {"search_depth": None, "auto_parameters": True}


def test_unsupported_depth_falls_back_to_default_depth():
    resolved = resolve_tavily_search_params(
        search_depth="turbo",
        auto_parameters=None,
        default_depth="basic",
        default_auto=False,
    )
    assert resolved == {"search_depth": "basic", "auto_parameters": False}


def test_validate_query_strips_and_returns():
    assert validate_tavily_query("  red panda habitat  ") == "red panda habitat"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 401])
def test_validate_query_rejects_empty_and_overlong(bad):
    with pytest.raises(ValueError):
        validate_tavily_query(bad)


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
    assert client.calls[0]["include_images"] is True
    assert payload["provider"] == "tavily"
    assert payload["operation"] == "search"
    assert payload["images"][0]["url"] == "https://example.com/a.jpg"
    assert payload["results"][0]["raw_content"] == "Full text"
    assert payload["usage"] == {"credits": 1}


def test_search_explicit_false_suppresses_provider_images(monkeypatch):
    client = _FakeTavilyClient(
        {
            "images": [{"url": "https://example.com/unexpected.jpg"}],
            "results": [],
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_search("text research", include_images=False))

    assert client.calls[0]["include_images"] is False
    assert client.calls[0]["include_image_descriptions"] is False
    assert payload["images"] == []


def test_search_normalizes_result_bound_images_with_parent_provenance(monkeypatch):
    client = _FakeTavilyClient(
        {
            "images": [
                "https://cdn.example/query.jpg",
                {"url": "https://cdn.example/rover.jpg", "description": "duplicate"},
            ],
            "results": [
                {
                    "title": "Rover story",
                    "url": "https://publisher.example/rover",
                    "content": "Story",
                    "score": 0.91,
                    "images": [
                        {
                            "url": "https://cdn.example/rover.jpg",
                            "description": "Mars rover",
                        }
                    ],
                }
            ],
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_search("mars rover", include_images=True))

    assert [image["url"] for image in payload["images"]] == [
        "https://cdn.example/rover.jpg",
        "https://cdn.example/query.jpg",
    ]
    result_image = payload["images"][0]
    assert result_image["source_url"] == "https://publisher.example/rover"
    assert result_image["source_title"] == "Rover story"
    assert result_image["source_domain"] == "publisher.example"
    assert result_image["result_rank"] == 0
    assert result_image["result_score"] == 0.91
    assert result_image["provider"] == "tavily"
    assert payload["images"][1]["query_level"] is True


def test_tavily_search_omits_depth_in_auto_mode(monkeypatch):
    captured = {}

    class _FakeClient:
        def search(self, **params):
            captured.update(params)
            return {"results": [], "images": []}

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeClient())
    tavily_server.tavily_search(query="red panda", auto_parameters=True)
    assert "search_depth" not in captured
    assert captured["auto_parameters"] is True


def test_tavily_search_rejects_empty_query(monkeypatch):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: object())
    result = json.loads(tavily_server.tavily_search(query="   "))
    assert result["error"]
    assert result.get("retryable") is not True


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


class _FakeSiteClient:
    def __init__(self):
        self.map_calls = []
        self.crawl_calls = []

    def map(self, **kwargs):
        self.map_calls.append(kwargs)
        return {
            "base_url": kwargs["url"],
            "results": ["https://docs.example.com/a"],
            "usage": {"credits": 1},
        }

    def crawl(self, **kwargs):
        self.crawl_calls.append(kwargs)
        return {
            "base_url": kwargs["url"],
            "results": [{"url": "https://docs.example.com/a", "raw_content": "A"}],
            "usage": {"credits": 1},
        }


def test_map_clamps_site_traversal(monkeypatch):
    client = _FakeSiteClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(
        tavily_server.tavily_map("https://docs.example.com", max_depth=5, limit=999)
    )

    assert client.map_calls[0]["max_depth"] == 2
    assert client.map_calls[0]["limit"] == 50
    assert payload["operation"] == "map"
    assert payload["results"] == ["https://docs.example.com/a"]


def test_crawl_clamps_site_traversal_and_disables_external_by_default(monkeypatch):
    client = _FakeSiteClient()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(
        tavily_server.tavily_crawl(
            "https://docs.example.com", instructions="Find API pages", max_depth=5, limit=999
        )
    )

    assert client.crawl_calls[0]["max_depth"] == 1
    assert client.crawl_calls[0]["limit"] == 20
    assert client.crawl_calls[0]["allow_external"] is False
    assert payload["operation"] == "crawl"
    assert payload["results"][0]["raw_content"] == "A"
