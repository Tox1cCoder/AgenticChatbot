"""Recency handling for provider-native image discovery.

Brave's image endpoint has no ``freshness`` parameter, so a recency-scoped
answer cannot ask the provider for recent pictures. The only recency signal the
pipeline receives is ``page_fetched`` (when Brave last crawled the page hosting
the image), and it was being captured by the search server and then dropped at
the candidate boundary — so nothing downstream could act on it.

These tests pin the two halves: the signal survives into provenance, and a
declared recency window drops candidates known to be older than it. An unknown
crawl date is never treated as stale.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.ai import web_research_tool
from app.ai.image_discovery_flow import select_brave_candidates
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_execution import build_image_candidates_from_tool_result
from app.ai.web_research_tool import create_web_research_tool

CONVERSATION_ID = "33333333-3333-3333-3333-333333333333"


def _iso(days_ago: float) -> str:
    stamp = datetime.now(UTC) - timedelta(days=days_ago)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _image(rank: int, *, page_fetched: str | None = None, confidence: str = "high") -> dict:
    image = {
        "url": f"https://imgs.search.brave.com/display-{rank}.webp",
        "original_image_url": f"https://origin.example/photo-{rank}.webp",
        "thumbnail_url": f"https://imgs.search.brave.com/thumb-{rank}.webp",
        "confidence": confidence,
        "result_rank": rank,
        "provider": "brave_image_search",
        "mime_type": "image/webp",
        "title": f"Stadium photo {rank}",
        "description": f"Stadium photo {rank}",
        "width": 995,
        "height": 565,
        "source_url": f"https://publisher.example/story/{rank}",
    }
    if page_fetched is not None:
        image["page_fetched"] = page_fetched
    return image


def _payload(*images: dict) -> str:
    return json.dumps(
        {
            "query": "stadium photo",
            "provider": "brave_image_search",
            "images": list(images),
            "total_results": len(images),
        }
    )


def _urls(candidates: list[dict]) -> list[str]:
    return [candidate["payload"]["url"] for candidate in candidates]


# ---------------------------------------------------------------------------
# The signal reaching the pipeline at all
# ---------------------------------------------------------------------------
def test_page_fetched_survives_into_candidate_provenance():
    candidates = build_image_candidates_from_tool_result(
        _payload(_image(1, page_fetched="2026-08-10T00:00:00Z")),
        tool_call_id="c1",
        tool_name="brave_image_search",
        group_images=False,
    )

    assert candidates[0]["provenance"]["page_fetched"] == "2026-08-10T00:00:00Z"


# ---------------------------------------------------------------------------
# The recency window
# ---------------------------------------------------------------------------
def test_a_stale_image_is_dropped_inside_a_declared_recency_window():
    selected = select_brave_candidates(
        _payload(
            _image(1, page_fetched=_iso(90)),
            _image(2, page_fetched=_iso(1)),
        ),
        image_query="stadium photo",
        time_range="week",
    )

    assert _urls(selected) == ["https://imgs.search.brave.com/thumb-2.webp"]


def test_an_unknown_crawl_date_is_not_treated_as_stale():
    """Brave does not populate page_fetched on every result. Dropping those
    would silently disable image discovery for whole classes of query."""
    selected = select_brave_candidates(
        _payload(_image(1)),
        image_query="stadium photo",
        time_range="day",
    )

    assert _urls(selected) == ["https://imgs.search.brave.com/thumb-1.webp"]


def test_without_a_declared_window_an_old_image_is_still_eligible():
    """No window means no recency claim: a decades-old photograph is the right
    answer for most subjects, and page_fetched is a crawl time, not a subject
    date."""
    selected = select_brave_candidates(
        _payload(_image(1, page_fetched=_iso(900))),
        image_query="stadium photo",
    )

    assert _urls(selected) == ["https://imgs.search.brave.com/thumb-1.webp"]


def test_a_stale_high_confidence_result_yields_to_a_fresh_lower_tier_one():
    """Filtering has to run before the confidence tier, or an all-stale high
    tier shadows the fresh medium results that should have been shown."""
    selected = select_brave_candidates(
        _payload(
            _image(1, page_fetched=_iso(400), confidence="high"),
            _image(2, page_fetched=_iso(1), confidence="medium"),
        ),
        image_query="stadium photo",
        time_range="month",
    )

    assert _urls(selected) == ["https://imgs.search.brave.com/thumb-2.webp"]


# ---------------------------------------------------------------------------
# The scope actually reaching discovery
# ---------------------------------------------------------------------------
class _Tool:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.payload


TAVILY_PAYLOAD = json.dumps(
    {
        "results": [{"index": 1, "title": "Story", "url": "https://x.example", "content": "x"}],
        "total_results": 1,
        "provider": "tavily",
        "operation": "search",
        "query": "stadium renovation",
    }
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_research_tool.settings, "remote_image_enrichment_enabled", True, raising=False
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


async def _research(**call_args: object) -> list[dict]:
    brave = _Tool(
        _payload(
            _image(1, page_fetched=_iso(120)),
            _image(2, page_fetched=_iso(1)),
        )
    )
    tool = create_web_research_tool(tavily_tool=_Tool(TAVILY_PAYLOAD), brave_tool=brave)
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
        await tool.ainvoke(
            {"query": "stadium renovation", "image_query": "stadium photo", **call_args}
        )
    return list(sink)


@pytest.mark.asyncio
async def test_web_research_forwards_its_recency_scope_to_image_discovery():
    """The recency window the model declared for the facts is the same window
    the picture beside those facts has to satisfy."""
    selected = await _research(time_range="week")

    assert _urls(selected) == ["https://imgs.search.brave.com/thumb-2.webp"]


@pytest.mark.asyncio
async def test_a_news_topic_alone_declares_no_window():
    """``topic`` states what kind of source to search, not how recent the answer
    must be. Inventing a window from it guessed at the user's intent, and a
    guessed cutoff silently discards images nobody asked to exclude."""
    selected = await _research(topic="news")

    assert _urls(selected) == [
        "https://imgs.search.brave.com/thumb-1.webp",
        "https://imgs.search.brave.com/thumb-2.webp",
    ]
