from __future__ import annotations

import json

from app.ai.mcp_servers import tavily_server


def test_tavily_clamp_count_uses_default_and_cap(monkeypatch):
    monkeypatch.setattr(tavily_server.settings, "tavily_search_default_max_results", 5, raising=False)
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
