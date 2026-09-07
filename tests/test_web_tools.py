"""The three product web tools ordinary agents actually see.

``web_search`` discovers, ``web_open`` extracts an answer to one question from
a page, ``image_search`` finds a picture. The raw providers sit behind them, so
these tests own the argument mapping, the bounds, and the failure shapes that
used to be spread across the combined research tool.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.ai import web_tools
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_tools import (
    create_image_search_tool,
    create_web_open_tool,
    create_web_search_tool,
)

CONVERSATION_ID = "55555555-5555-5555-5555-555555555555"
NOW = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)


def _clock():
    return NOW


class _FakeTool:
    def __init__(self, name: str, payload: str, delay: float = 0.0):
        self.name = name
        self.payload = payload
        self.delay = delay
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_tools.settings, "remote_image_enrichment_enabled", True, raising=False
    )
    monkeypatch.setattr(web_tools.settings, "inline_rich_response_enabled", True, raising=False)
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


async def _run(tool, **kwargs):
    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink() as sink,
    ):
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


def _search_payload(results: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "results": results
            if results is not None
            else [
                {
                    "index": 1,
                    "title": "Aurora 4.2 release notes",
                    "url": "https://vendor.example/aurora/4.2",
                    "content": "Aurora 4.2 shipped on 14 March 2026.",
                    "score": 0.91,
                    "published_date": "2026-03-14",
                    "raw_content": "the entire page body " * 500,
                    "favicon": "https://vendor.example/favicon.ico",
                }
            ],
            "total_results": 1,
            "answer": "",
            "provider": "tavily",
            "operation": "search",
            "query": "aurora release",
        }
    )


def _extract_payload() -> str:
    return json.dumps(
        {
            "provider": "tavily",
            "operation": "extract",
            "urls": ["https://example.com/a"],
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "Aurora notes",
                    "raw_content": "Aurora 4.2 was released on 14 March 2026. "
                    "Unrelated filler about arctic terns. " * 5,
                }
            ],
            "failed_results": [
                {"url": "https://example.com/b", "error": "403 forbidden"},
            ],
            "usage": {"credits": 4},
            "request_id": "req-9",
        }
    )


def _brave_payload(*, count: int = 1, confidence: str = "high", slug: str = "team") -> str:
    return json.dumps(
        {
            "query": slug,
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://imgs.search.brave.com/{slug}-display-{index}.jpg",
                    "original_image_url": f"https://origin.example/{slug}-{index}.jpg",
                    "thumbnail_url": f"https://imgs.search.brave.com/{slug}-thumb-{index}.jpg",
                    "confidence": confidence,
                    "result_rank": index,
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": f"{slug} {index}",
                    "description": f"{slug} {index}",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://publisher.example/page",
                }
                for index in range(count)
            ],
            "total_results": count,
        }
    )


# --------------------------------------------------------------------------
# web_search
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_recent_search_reaches_the_provider_anchored_to_the_injected_clock():
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    await _run(
        tool,
        query="best local LLMs in 2024",
        objective="Find the currently strongest local models",
        freshness="recent",
        max_results=4,
    )

    assert tavily.calls == [
        {
            "query": "best local LLMs in 2026",
            "max_results": 4,
            "search_depth": "advanced",
            "include_raw_content": False,
            "topic": "news",
            "end_date": "2026-09-04",
        }
    ]


@pytest.mark.asyncio
async def test_an_as_of_search_forwards_the_historical_range_untouched():
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    await _run(
        tool,
        query="Python packaging guidance in 2024",
        objective="Report what the guidance said at the end of 2024",
        freshness="as_of",
        end_date="2024-12-31",
    )

    assert tavily.calls[0]["query"] == "Python packaging guidance in 2024"
    assert tavily.calls[0]["end_date"] == "2024-12-31"
    assert tavily.calls[0]["topic"] == "general"


@pytest.mark.asyncio
async def test_max_results_is_clamped_to_the_configured_ceiling(monkeypatch):
    monkeypatch.setattr(web_tools.settings, "web_search_max_results", 3, raising=False)
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    await _run(
        tool, query="aurora release", objective="Find the release date", max_results=19
    )

    assert tavily.calls[0]["max_results"] == 3


@pytest.mark.asyncio
async def test_results_are_projected_and_the_page_body_never_reaches_the_model():
    tool = create_web_search_tool(
        tavily_tool=_FakeTool("tavily_search", _search_payload()), clock=_clock
    )

    payload, _ = await _run(tool, query="aurora release", objective="Find the release date")

    assert payload["results"] == [
        {
            "title": "Aurora 4.2 release notes",
            "url": "https://vendor.example/aurora/4.2",
            "published_date": "2026-03-14",
            "snippet": "Aurora 4.2 shipped on 14 March 2026.",
            "score": 0.91,
        }
    ]
    assert "the entire page body" not in json.dumps(payload)
    assert "favicon" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_results_sharing_a_canonical_url_are_returned_once():
    tavily = _FakeTool(
        "tavily_search",
        _search_payload(
            [
                {
                    "title": "First",
                    "url": "https://EXAMPLE.com/team/?utm_source=x#roster",
                    "content": "release date 14 March 2026",
                    "score": 0.9,
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/team",
                    "content": "release date 14 March 2026",
                    "score": 0.8,
                },
            ]
        ),
    )
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    payload, _ = await _run(tool, query="aurora release", objective="Find the release date")

    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "First"


@pytest.mark.asyncio
async def test_the_projected_payload_honors_the_configured_character_cap(monkeypatch):
    monkeypatch.setattr(web_tools.settings, "web_search_result_max_chars", 700, raising=False)
    tavily = _FakeTool(
        "tavily_search",
        _search_payload(
            [
                {
                    "title": f"Source {index}",
                    "url": f"https://e.example/{index}",
                    "content": "release date evidence " * 60,
                    "score": 0.9,
                }
                for index in range(12)
            ]
        ),
    )
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ):
        raw = await tool.ainvoke(
            {"query": "aurora release", "objective": "Find the release date"}
        )

    assert len(raw) <= 700
    assert json.loads(raw)["omitted_results"] > 0


@pytest.mark.asyncio
async def test_a_provider_error_is_normalized_and_never_looks_like_evidence():
    tavily = _FakeTool(
        "tavily_search",
        json.dumps(
            {
                "provider": "tavily",
                "operation": "search",
                "error": "Tavily search rate_limit.",
                "retryable": True,
            }
        ),
    )
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    payload, _ = await _run(tool, query="aurora release", objective="Find the release date")

    assert payload["status"] == "error"
    assert payload["retryable"] is True
    assert "results" not in payload


@pytest.mark.asyncio
async def test_an_unusable_intent_is_refused_before_the_provider_is_called():
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    payload, _ = await _run(
        tool,
        query="projected 2030 grid capacity",
        objective="Find the 2030 projections",
        freshness="recent",
    )

    assert payload["status"] == "error"
    assert payload["error_type"] == "invalid_request"
    assert payload["retryable"] is False
    assert tavily.calls == []


@pytest.mark.asyncio
async def test_a_near_duplicate_query_reuses_the_turns_result():
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    await _run(
        tool,
        query="Aurora 4.2 release date and version history",
        objective="Find the release date",
    )
    payload, _ = await _run(
        tool,
        query="Aurora 4.2 version history and release date",
        objective="Find the release date",
    )

    assert len(tavily.calls) == 1
    assert payload["reused"] is True


@pytest.mark.asyncio
async def test_concurrent_matching_queries_share_one_reservation():
    tavily = _FakeTool("tavily_search", _search_payload(), delay=0.05)
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    await asyncio.gather(
        _run(tool, query="aurora release date", objective="Find the release date"),
        _run(tool, query="aurora release date", objective="Find the release date"),
    )

    assert len(tavily.calls) == 1


@pytest.mark.asyncio
async def test_web_search_never_extracts_the_pages_it_found():
    """Discovery is one decision and extraction is another. Auto-extracting
    every hit is what made a single question cost four page fetches."""
    extract = _FakeTool("tavily_extract", _extract_payload())
    tool = create_web_search_tool(
        tavily_tool=_FakeTool("tavily_search", _search_payload()),
        extract_tool=extract,
        clock=_clock,
    )

    await _run(tool, query="aurora release", objective="Find the release date")

    assert extract.calls == []


@pytest.mark.asyncio
async def test_cancelling_web_search_propagates_without_leaking_a_task():
    started = asyncio.Event()

    class _Blocking:
        name = "tavily_search"

        async def ainvoke(self, args):
            started.set()
            await asyncio.Event().wait()

    tool = create_web_search_tool(tavily_tool=_Blocking(), clock=_clock)
    tasks_before = asyncio.all_tasks()

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ):
        task = asyncio.create_task(
            tool.ainvoke({"query": "aurora release", "objective": "Find the release date"})
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    leftover = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)
    assert all(item.done() for item in leftover)


@pytest.mark.asyncio
async def test_web_search_is_refused_in_client_only_scope():
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=tavily, clock=_clock)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID,
        user_id="u1",
        agent_key="search",
        device_id="device-a",
        tool_scope="client_only",
    ):
        payload = json.loads(
            await tool.ainvoke({"query": "aurora", "objective": "Find the release date"})
        )

    assert payload["error_type"] == "permission_error"
    assert tavily.calls == []


def test_web_search_identity_is_internal():
    tool = create_web_search_tool(clock=_clock)

    assert tool.name == "web_search"
    assert tool.metadata["tool_origin"] == "internal"
    assert tool.metadata["qualified_tool_id"] == "internal::web_search"


# --------------------------------------------------------------------------
# web_open
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_open_always_sends_the_focused_question_to_the_provider():
    extract = _FakeTool("tavily_extract", _extract_payload())
    tool = create_web_open_tool(extract_tool=extract)

    await _run(
        tool,
        urls=["https://example.com/a"],
        question="Which release date and version are stated?",
    )

    assert extract.calls == [
        {
            "urls": ["https://example.com/a"],
            "query": "Which release date and version are stated?",
            "chunks_per_source": 3,
            "include_images": False,
        }
    ]


@pytest.mark.asyncio
async def test_web_open_clamps_the_url_count_to_the_configured_maximum(monkeypatch):
    monkeypatch.setattr(web_tools.settings, "web_open_max_urls", 2, raising=False)
    extract = _FakeTool("tavily_extract", _extract_payload())
    tool = create_web_open_tool(extract_tool=extract)

    await _run(
        tool,
        urls=[f"https://example.com/{index}" for index in range(9)],
        question="Which release date is stated?",
    )

    assert extract.calls[0]["urls"] == ["https://example.com/0", "https://example.com/1"]


def test_web_open_requires_a_question_at_the_schema_boundary():
    with pytest.raises(ValidationError):
        web_tools.WebOpenInput(urls=["https://example.com/a"])
    with pytest.raises(ValidationError):
        web_tools.WebOpenInput(urls=["https://example.com/a"], question="x")


def test_web_open_requires_at_least_one_url():
    with pytest.raises(ValidationError):
        web_tools.WebOpenInput(urls=[], question="Which release date is stated?")


@pytest.mark.asyncio
async def test_web_open_returns_focused_excerpts_not_the_whole_page():
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", _extract_payload()))

    payload, _ = await _run(
        tool,
        urls=["https://example.com/a"],
        question="Which release date is stated?",
    )

    assert payload["excerpts"]
    assert "14 March 2026" in " ".join(item["text"] for item in payload["excerpts"])
    assert payload["excerpts"][0]["url"] == "https://example.com/a"


@pytest.mark.asyncio
async def test_web_open_honors_the_configured_character_cap(monkeypatch):
    monkeypatch.setattr(web_tools.settings, "web_open_max_chars", 2_000, raising=False)
    payload_text = json.dumps(
        {
            "results": [
                {
                    "url": "https://example.com/a",
                    "raw_content": "release date evidence sentence. " * 400,
                }
            ],
            "failed_results": [],
        }
    )
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", payload_text))

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ):
        raw = await tool.ainvoke(
            {"urls": ["https://example.com/a"], "question": "Which release date is stated?"}
        )

    assert len(raw) <= 2_000


@pytest.mark.asyncio
async def test_an_overlong_query_is_corrective_and_spends_no_search_slot(monkeypatch):
    """A query the provider will refuse must not cost the turn its search.

    The coroutine is called directly here because the schema now rejects this
    query first; what is under test is the second boundary, for a caller that
    reached the tool without it.
    """
    monkeypatch.setattr(web_tools.settings, "research_budget_enabled", True, raising=False)
    search = _FakeTool("tavily_search", _search_payload())
    tool = create_web_search_tool(tavily_tool=search, clock=_clock)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ):
        rejected = json.loads(
            await tool.coroutine(query="q" * 450, objective="Find the published documentation")
        )
        accepted = json.loads(
            await tool.coroutine(
                query="aurora release notes", objective="Find the published documentation"
            )
        )

    assert rejected["error_type"] == "invalid_request"
    assert rejected["retryable"] is False
    assert [call["query"] for call in search.calls] == ["aurora release notes"]
    assert accepted["searches_used"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cap", "url_length"),
    [(3_000, 24), (2_000, 300)],
    ids=["short_urls", "long_urls"],
)
async def test_web_open_never_exceeds_the_cap_once_the_envelope_is_counted(
    monkeypatch, cap, url_length
):
    """The cap belongs to the string that reaches the model, not to a draft.

    Budgeting the excerpts alone leaves the URLs, the failure records and the
    envelope's own separators outside the accounting, so a configuration well
    inside its declared range returns more than it promised.
    """
    monkeypatch.setattr(web_tools.settings, "web_open_max_chars", cap, raising=False)
    monkeypatch.setattr(web_tools.settings, "web_open_max_excerpts", 8, raising=False)
    urls = [
        "https://example.com/" + str(index) * (url_length - len("https://example.com/"))
        for index in range(4)
    ]
    provider = json.dumps(
        {
            "results": [
                {
                    "url": url,
                    "title": "Release notes",
                    "raw_content": f"release {index} " + chr(65 + index) * 1_100,
                }
                for index, url in enumerate(urls)
            ],
            "failed_results": [],
        }
    )
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", provider))

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ):
        raw = await tool.ainvoke({"urls": urls, "question": "Which release is stated?"})

    assert len(raw) <= cap
    assert json.loads(raw)["question"] == "Which release is stated?"


@pytest.mark.asyncio
async def test_web_open_still_returns_evidence_at_the_default_configuration():
    """Bounding the envelope must not starve the excerpts it exists to carry."""
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", _extract_payload()))

    payload, _ = await _run(
        tool,
        urls=["https://example.com/a"],
        question="Which release date is stated?",
    )

    assert payload["excerpts"]


@pytest.mark.asyncio
async def test_web_open_reports_per_url_failures_without_provider_diagnostics():
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", _extract_payload()))

    payload, _ = await _run(
        tool,
        urls=["https://example.com/a", "https://example.com/b"],
        question="Which release date is stated?",
    )

    assert payload["failed"] == [{"url": "https://example.com/b", "error": "403 forbidden"}]
    assert "request_id" not in json.dumps(payload)
    assert "credits" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_web_open_normalizes_a_provider_error():
    extract = _FakeTool(
        "tavily_extract",
        json.dumps({"error": "Tavily extract timeout.", "retryable": True}),
    )
    tool = create_web_open_tool(extract_tool=extract)

    payload, _ = await _run(
        tool, urls=["https://example.com/a"], question="Which release date is stated?"
    )

    assert payload["status"] == "error"
    assert payload["retryable"] is True


@pytest.mark.asyncio
async def test_web_open_is_refused_in_client_only_scope():
    extract = _FakeTool("tavily_extract", _extract_payload())
    tool = create_web_open_tool(extract_tool=extract)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID,
        user_id="u1",
        agent_key="search",
        device_id="device-a",
        tool_scope="client_only",
    ):
        payload = json.loads(
            await tool.ainvoke(
                {"urls": ["https://example.com/a"], "question": "Which date is stated?"}
            )
        )

    assert payload["error_type"] == "permission_error"
    assert extract.calls == []


def test_web_open_identity_is_internal():
    tool = create_web_open_tool()

    assert tool.name == "web_open"
    assert tool.metadata["qualified_tool_id"] == "internal::web_open"


# --------------------------------------------------------------------------
# image_search
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_search_offers_a_provider_selected_image():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    payload, sink = await _run(tool, query="T1 team photo")

    assert brave.calls == [{"query": "T1 team photo"}]
    assert len(sink) == 1
    assert sink[0]["type"] == "image"
    assert payload["selected"] == 1


@pytest.mark.asyncio
async def test_image_search_never_returns_an_image_url_to_the_model():
    """Images reach the answer through the rich-item inventory alone. A URL in
    the tool result is a URL the model can paste into prose."""
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload())
    )

    payload, _ = await _run(tool, query="T1 team photo")

    assert "brave.com" not in json.dumps(payload)
    assert "origin.example" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_gallery_intent_returns_one_grid_item():
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload(count=4))
    )

    _, sink = await _run(tool, query="T1 team photo", intent="gallery")

    assert len(sink) == 1
    assert sink[0]["type"] == "image_group"
    assert len(sink[0]["payload"]["items"]) == 4


@pytest.mark.asyncio
async def test_a_gallery_honors_a_caller_supplied_count_bound():
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload(count=6))
    )

    _, sink = await _run(tool, query="T1 team photo", intent="gallery", max_images=3)

    assert len(sink[0]["payload"]["items"]) == 3


@pytest.mark.asyncio
async def test_figure_intent_offers_exactly_one_image():
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload(count=4))
    )

    _, sink = await _run(tool, query="T1 team photo")

    assert len(sink) == 1
    assert sink[0]["type"] == "image"


@pytest.mark.asyncio
async def test_two_distinct_subjects_each_get_their_own_search():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    await _run(tool, query="Pokemon Unite logo")
    await _run(tool, query="Pokemon Unite gameplay screenshot")

    assert [call["query"] for call in brave.calls] == [
        "Pokemon Unite logo",
        "Pokemon Unite gameplay screenshot",
    ]


@pytest.mark.asyncio
async def test_repeating_a_subject_does_not_buy_a_second_search():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    await _run(tool, query="Pokemon Unite logo")
    payload, sink = await _run(tool, query="Pokemon Unite logo")

    assert len(brave.calls) == 1
    assert payload["selected"] == 0
    assert sink == []


@pytest.mark.asyncio
async def test_concurrent_claims_on_one_subject_share_a_reservation():
    brave = _FakeTool("brave_image_search", _brave_payload(), delay=0.05)
    tool = create_image_search_tool(brave_tool=brave)

    await asyncio.gather(
        _run(tool, query="T1 team photo"),
        _run(tool, query="T1 team photo"),
    )

    assert len(brave.calls) == 1


@pytest.mark.asyncio
async def test_a_declared_recency_window_drops_a_stale_picture():
    stale_fresh = json.dumps(
        {
            "provider": "brave_image_search",
            "images": [
                {
                    "url": "https://imgs.search.brave.com/display-1.jpg",
                    "original_image_url": "https://origin.example/1.jpg",
                    "thumbnail_url": "https://imgs.search.brave.com/thumb-1.jpg",
                    "confidence": "high",
                    "result_rank": 1,
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": "stadium 1",
                    "description": "stadium 1",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://publisher.example/1",
                    "page_fetched": "2020-01-01T00:00:00+00:00",
                },
                {
                    "url": "https://imgs.search.brave.com/display-2.jpg",
                    "original_image_url": "https://origin.example/2.jpg",
                    "thumbnail_url": "https://imgs.search.brave.com/thumb-2.jpg",
                    "confidence": "high",
                    "result_rank": 2,
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": "stadium 2",
                    "description": "stadium 2",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://publisher.example/2",
                    "page_fetched": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }
    )
    tool = create_image_search_tool(brave_tool=_FakeTool("brave_image_search", stale_fresh))

    _, sink = await _run(tool, query="stadium photo", time_range="week")

    assert [item["payload"]["url"] for item in sink] == [
        "https://imgs.search.brave.com/thumb-2.jpg"
    ]


@pytest.mark.asyncio
async def test_a_brave_failure_yields_no_image_and_no_exception():
    class _Broken:
        name = "brave_image_search"

        async def ainvoke(self, args):
            raise RuntimeError("brave down")

    tool = create_image_search_tool(brave_tool=_Broken())

    payload, sink = await _run(tool, query="T1 team photo")

    assert sink == []
    assert payload["selected"] == 0


@pytest.mark.asyncio
async def test_the_remote_image_flag_closes_the_path_entirely(monkeypatch):
    monkeypatch.setattr(
        web_tools.settings, "remote_image_enrichment_enabled", False, raising=False
    )
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    payload, sink = await _run(tool, query="T1 team photo")

    assert brave.calls == []
    assert sink == []
    assert payload["selected"] == 0


@pytest.mark.asyncio
async def test_a_request_without_the_rich_capability_skips_the_provider():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    with (
        tool_execution_context(
            conversation_id=CONVERSATION_ID,
            user_id="u1",
            agent_key="search",
            rich_response_capable=False,
        ),
        selected_image_sink() as sink,
    ):
        await tool.ainvoke({"query": "T1 team photo"})

    assert brave.calls == []
    assert sink == []


@pytest.mark.asyncio
async def test_image_search_never_touches_the_text_provider():
    """Splitting the tools exists so a picture never waits on a web search."""
    tavily = _FakeTool("tavily_search", _search_payload())
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload()),
        tavily_tool=tavily,
    )

    await _run(tool, query="T1 team photo")

    assert tavily.calls == []


@pytest.mark.asyncio
async def test_image_search_is_refused_in_client_only_scope():
    brave = _FakeTool("brave_image_search", _brave_payload())
    tool = create_image_search_tool(brave_tool=brave)

    with tool_execution_context(
        conversation_id=CONVERSATION_ID,
        user_id="u1",
        agent_key="search",
        device_id="device-a",
        tool_scope="client_only",
    ):
        payload = json.loads(await tool.ainvoke({"query": "T1 team photo"}))

    assert payload["error_type"] == "permission_error"
    assert brave.calls == []


def test_image_search_identity_is_internal():
    tool = create_image_search_tool()

    assert tool.name == "image_search"
    assert tool.metadata["qualified_tool_id"] == "internal::image_search"


def test_the_descriptions_never_name_a_raw_provider_tool():
    """A description that names ``tavily_extract`` teaches the model to look for
    a tool ordinary discovery deliberately hides."""
    text = " ".join(
        tool.description
        for tool in (
            create_web_search_tool(clock=_clock),
            create_web_open_tool(),
            create_image_search_tool(),
        )
    ).lower()

    assert "tavily" not in text
    assert "brave" not in text


# --------------------------------------------------------------------------
# Rollout observations
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_observation_line_reports_the_char_pair_the_change_exists_to_move(caplog):
    """The provider payload may stay large; the model's view of it must not.
    That pair is the only number that shows whether the boundary is holding."""
    tool = create_web_search_tool(
        tavily_tool=_FakeTool("tavily_search", _search_payload()), clock=_clock
    )

    with caplog.at_level("INFO", logger="app.ai.web_tools"):
        await _run(tool, query="aurora release", objective="Find the release date")

    line = next(record.getMessage() for record in caplog.records if "web_tool_call" in
                record.getMessage())
    assert "operation=web_search" in line
    assert "outcome=completed" in line
    assert "freshness=timeless" in line
    provider_chars = int(line.split("provider_chars=")[1].split()[0])
    model_chars = int(line.split("model_chars=")[1].split()[0])
    assert provider_chars > model_chars > 0


@pytest.mark.asyncio
async def test_the_observation_line_never_carries_user_derived_text(caplog):
    """It is written on every call. One unredacted field would put user content
    into ordinary operational logs, and into an unbounded label space if the
    field were ever promoted to a metric."""
    secret_query = "zzsecretsubjectzz release notes"
    secret_objective = "zzsecretobjectivezz"
    tool = create_web_search_tool(
        tavily_tool=_FakeTool("tavily_search", _search_payload()), clock=_clock
    )

    with caplog.at_level("INFO", logger="app.ai.web_tools"):
        await _run(tool, query=secret_query, objective=secret_objective)

    for record in caplog.records:
        message = record.getMessage()
        assert "zzsecretsubjectzz" not in message
        assert "zzsecretobjectivezz" not in message
        assert "vendor.example" not in message


@pytest.mark.asyncio
async def test_a_repeated_query_is_reported_as_a_rejection(caplog, monkeypatch):
    monkeypatch.setattr(
        web_tools.settings, "research_max_search_calls_per_turn", 1, raising=False
    )
    tool = create_web_search_tool(
        tavily_tool=_FakeTool("tavily_search", _search_payload()), clock=_clock
    )

    await _run(tool, query="aurora release date", objective="Find the release date")
    with caplog.at_level("INFO", logger="app.ai.web_tools"):
        await _run(tool, query="quite unrelated ornithology digest", objective="Find birds")

    assert any(
        "outcome=repeated_query_rejected" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_repeated_image_subject_is_reported_as_a_rejection(caplog):
    tool = create_image_search_tool(
        brave_tool=_FakeTool("brave_image_search", _brave_payload())
    )

    await _run(tool, query="T1 team photo")
    with caplog.at_level("INFO", logger="app.ai.web_tools"):
        await _run(tool, query="T1 team photo")

    assert any(
        "outcome=repeated_subject_rejected" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_web_open_reports_urls_failures_and_excerpt_counts(caplog):
    tool = create_web_open_tool(extract_tool=_FakeTool("tavily_extract", _extract_payload()))

    with caplog.at_level("INFO", logger="app.ai.web_tools"):
        await _run(
            tool,
            urls=["https://example.com/a"],
            question="Which release date is stated?",
        )

    line = next(record.getMessage() for record in caplog.records if "web_tool_call" in
                record.getMessage())
    assert "operation=web_open" in line
    assert "urls=1" in line
    assert "failed=1" in line
    assert "excerpts=" in line
