from __future__ import annotations

import inspect
import json

import pytest
import requests
from tavily import errors as tavily_errors

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


def test_tavily_error_payload_bounds_and_redacts_untrusted_details():
    payload = json.loads(
        tavily_server._error(
            "request failed api_key=supersecret at "
            "https://private.example/path?token=supersecret "
            + ("x" * 600),
            operation="search",
        )
    )

    assert "supersecret" not in payload["error"]
    assert "private.example" not in payload["error"]
    assert len(payload["error"]) <= 200


class _FakeTavilyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_search_requests_ranked_sources_without_provider_answer(monkeypatch):
    client = _FakeTavilyClient(
        {
            "query": "openai news",
            "answer": "OpenAI shipped a model.",
            "images": [{"url": "https://example.com/a.jpg", "description": "A"}],
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.9,
                    "raw_content": "Full text",
                    "images": [{"url": "https://example.com/bound.jpg"}],
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

    payload = json.loads(tavily_server.tavily_search("openai news"))

    assert client.calls[0]["search_depth"] == "basic"
    assert client.calls[0]["max_results"] == 5
    assert client.calls[0]["include_answer"] is False
    assert client.calls[0]["include_raw_content"] is False
    assert client.calls[0]["include_usage"] is True
    assert client.calls[0]["timeout"] == 10
    assert "include_images" not in client.calls[0]
    assert "include_image_descriptions" not in client.calls[0]
    assert "images" not in payload
    assert payload["results"][0]["raw_content"] == "Full text"
    assert payload["usage"] == {"credits": 1}


def test_search_uses_only_sdk_timeout_control():
    source = inspect.getsource(tavily_server)

    assert "asyncio.timeout" not in source
    assert "asyncio.wait_for" not in source
    assert "tavily_search_timeout" not in source


def test_topic_and_time_range_are_forwarded(monkeypatch):
    client = _FakeTavilyClient({"results": []})
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    tavily_server.tavily_search("latest T1 results", topic="news", time_range="week")

    assert client.calls[0]["topic"] == "news"
    assert client.calls[0]["time_range"] == "week"


def test_news_publication_date_survives_normalization(monkeypatch):
    client = _FakeTavilyClient(
        {
            "results": [
                {
                    "title": "T1 wins",
                    "url": "https://news.example/t1",
                    "content": "Result",
                    "score": 0.9,
                    "published_date": "2026-08-09",
                }
            ]
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_search("T1", topic="news"))

    assert payload["results"][0]["published_date"] == "2026-08-09"


def test_duplicate_urls_merge_unique_chunks_and_keep_first_rank():
    payload = tavily_server._normalize_search_response(
        query="T1 roster",
        response={
            "results": [
                {
                    "title": "First",
                    "url": "https://EXAMPLE.com/team/?utm_source=x#roster",
                    "content": "A [...] B",
                    "score": 0.9,
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/team",
                    "content": "B [...] C",
                    "score": 0.8,
                },
            ]
        },
    )

    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "First"
    assert payload["results"][0]["url"].startswith("https://EXAMPLE.com")
    assert payload["results"][0]["content"] == "A [...] B [...] C"


def test_malformed_result_url_does_not_discard_later_valid_results():
    payload = tavily_server._normalize_search_response(
        query="T1 roster",
        response={
            "results": [
                {
                    "title": "Malformed",
                    "url": "https://[broken.example/path",
                    "content": "Do not keep this row.",
                    "score": 0.95,
                },
                {
                    "title": "Valid",
                    "url": "https://news.example/t1",
                    "content": "Keep this row.",
                    "score": 0.9,
                },
            ]
        },
    )

    assert [(item["title"], item["index"]) for item in payload["results"]] == [
        ("Valid", 2)
    ]


class _RaisingClient:
    def __init__(self, exc):
        self.exc = exc

    def search(self, **kwargs):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (tavily_errors.TimeoutError(10), True),
        (tavily_errors.UsageLimitExceededError("rate limited"), True),
        (tavily_errors.BadRequestError("bad query"), False),
        (tavily_errors.InvalidAPIKeyError("bad key"), False),
        (tavily_errors.ForbiddenError("plan"), False),
    ],
)
def test_search_errors_have_bounded_retryability(monkeypatch, exc, retryable):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _RaisingClient(exc))

    payload = json.loads(tavily_server.tavily_search("T1"))

    assert payload["retryable"] is retryable


def test_unclassified_http_5xx_error_is_retryable(monkeypatch):
    error = requests.HTTPError("gateway failure")
    error.response = type("Response", (), {"status_code": 502})()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _RaisingClient(error))

    payload = json.loads(tavily_server.tavily_search("T1"))

    assert payload["retryable"] is True


def test_raw_http_429_is_a_retryable_rate_limit(monkeypatch):
    error = requests.HTTPError("too many requests")
    error.response = type("Response", (), {"status_code": 429})()
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _RaisingClient(error))

    payload = json.loads(tavily_server.tavily_search("T1"))

    assert payload["error"] == "Tavily search rate_limit."
    assert payload["retryable"] is True


def test_unclassified_errors_are_not_retryable_and_do_not_leak_details(monkeypatch):
    monkeypatch.setattr(
        tavily_server,
        "_make_client",
        lambda: _RaisingClient(RuntimeError("secret-token-value")),
    )

    payload = json.loads(tavily_server.tavily_search("T1"))

    assert payload["retryable"] is False
    assert "secret-token-value" not in payload["error"]


def test_invalid_topic_returns_error_without_calling_tavily(monkeypatch):
    client = _FakeTavilyClient({"results": []})
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    payload = json.loads(tavily_server.tavily_search("T1", topic="sports"))

    assert payload["error"] == "Unsupported topic."
    assert client.calls == []


def test_invalid_control_error_does_not_echo_an_untrusted_value(monkeypatch):
    client = _FakeTavilyClient({"results": []})
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    untrusted = "api_key=supersecret-" + ("x" * 600)

    payload = json.loads(tavily_server.tavily_search("T1", topic=untrusted))

    assert "supersecret" not in payload["error"]
    assert len(payload["error"]) <= 200
    assert client.calls == []


def test_search_payload_orders_results_before_diagnostics(monkeypatch):
    client = _FakeTavilyClient(
        {
            "answer": "a",
            "results": [{"title": "T", "url": "https://e.example", "content": "c", "score": 1}],
            "usage": {"credits": 1},
            "request_id": "req-2",
            "response_time": 0.4,
        }
    )
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)

    keys = list(json.loads(tavily_server.tavily_search("ordering")).keys())

    assert keys[0] == "results"
    assert keys.index("results") < keys.index("usage")
    assert keys.index("results") < keys.index("request_id")
    assert keys.index("results") < keys.index("response_time")


def test_search_rejects_an_include_images_argument(monkeypatch):
    monkeypatch.setattr(tavily_server, "_make_client", lambda: _FakeTavilyClient({"results": []}))

    with pytest.raises(TypeError):
        tavily_server.tavily_search("apple park", include_images=True)


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


class _ExplodingNonSearchClient:
    def extract(self, **kwargs):
        raise RuntimeError(
            "api_key=supersecret at https://private.example/extract?token=supersecret"
        )

    def map(self, **kwargs):
        raise RuntimeError("token=supersecret at https://private.example/map")

    def crawl(self, **kwargs):
        raise RuntimeError("password=supersecret at https://private.example/crawl")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: tavily_server.tavily_extract("https://example.com"),
        lambda: tavily_server.tavily_map("https://example.com"),
        lambda: tavily_server.tavily_crawl("https://example.com"),
    ],
)
def test_nonsearch_provider_errors_do_not_expose_client_details(monkeypatch, invoke):
    monkeypatch.setattr(tavily_server, "_make_client", _ExplodingNonSearchClient)

    payload = json.loads(invoke())

    assert "supersecret" not in payload["error"]
    assert "private.example" not in payload["error"]
    assert len(payload["error"]) <= 200


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
