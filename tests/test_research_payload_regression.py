"""Trace-shaped regression for the T1 research turn in example_run.txt.

The original failure: a Tavily payload led with a large scraped-image array, the
offload preview kept only that array, the model received no facts, and it
re-ran the search twice.
"""

from __future__ import annotations

import json

from app.ai.mcp_servers import tavily_server
from app.services.tool_result_preview import build_tool_result_preview

SOURCE_URL = "https://www.sheepesports.com/en/all/articles/lol-t1-completed-2026-lck-roster/en"


def _t1_provider_response() -> dict:
    return {
        "query": "T1 League of Legends Esports team news roster 2026",
        "answer": "T1 is a South Korean esports organization.",
        "images": [
            {"url": f"https://cdn.example/{i}.jpg", "description": "Moi"} for i in range(24)
        ],
        "results": [
            {
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": SOURCE_URL,
                "content": "T1 finalized its 2026 LCK roster. " * 120,
                "score": 0.887,
                "images": [{"url": "https://cdn.example/bound.jpg", "description": "Moi"}],
            },
            {
                "title": "T1 - Leaguepedia",
                "url": "https://lol.fandom.com/wiki/T1",
                "content": "T1 is a South Korean esports organization. " * 120,
                "score": 0.873,
            },
        ],
    }


def test_search_result_carries_facts_and_no_image_metadata(monkeypatch):
    class _Client:
        def search(self, **params):
            return _t1_provider_response()

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _Client())

    raw = tavily_server.tavily_search("T1 League of Legends Esports team news roster 2026")
    payload = json.loads(raw)

    assert "images" not in payload
    assert "cdn.example" not in raw
    assert payload["answer"].startswith("T1 is a South Korean")
    assert len(payload["results"]) == 2


def test_offload_preview_of_that_result_still_contains_both_sources(monkeypatch):
    class _Client:
        def search(self, **params):
            return _t1_provider_response()

    monkeypatch.setattr(tavily_server, "_make_client", lambda: _Client())
    raw = tavily_server.tavily_search("T1 League of Legends Esports team news roster 2026")

    preview = build_tool_result_preview(raw, budget_chars=4000)

    parsed = json.loads(preview.text)
    assert [entry["url"] for entry in parsed["results"]] == [
        SOURCE_URL,
        "https://lol.fandom.com/wiki/T1",
    ]
    assert "South Korean" in json.dumps(parsed)
    assert preview.omitted_arrays == ()
    assert preview.omitted_results == 0
