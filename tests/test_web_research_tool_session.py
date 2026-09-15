from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai.tool_context import get_tool_context, tool_execution_context
from app.ai.web_research.contracts import ResearchRequest
from app.ai.web_tools import create_web_open_tool, create_web_search_tool


class _Session:
    mode = "quick"

    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.budget = SimpleNamespace(search_calls=1)
        self.requests: list[ResearchRequest] = []
        self.opens: list[tuple[list[str], str]] = []

    async def search(self, request: ResearchRequest):
        self.requests.append(request)
        return self.bundle

    async def open(self, urls, question):
        self.opens.append((list(urls), question))
        return self.bundle


@pytest.mark.asyncio
async def test_web_search_uses_the_exact_session_from_tool_context() -> None:
    bundle = SimpleNamespace(
        status="success",
        mode="quick",
        operation_index=1,
        sources=(
            SimpleNamespace(
                source_id="S1",
                title="Release notes",
                url="https://example.test/release",
                snippet="Released today.",
                published_at=None,
                status="search_result",
            ),
        ),
        images=(SimpleNamespace(candidate_id="I1", delivery_url="/web-images/private"),),
        failures=(),
        reused=False,
        omitted_source_count=0,
        omitted_image_count=0,
    )
    session = _Session(bundle)

    with tool_execution_context(
        conversation_id="conversation",
        user_id="user",
        agent_key="search",
        web_research_session=session,
    ):
        assert get_tool_context().web_research_session is session
        raw = await create_web_search_tool().ainvoke(
            {
                "query": "Aurora release",
                "objective": "Find the release date",
                "freshness": "recent",
                "visual_intent": "figure",
                "image_query": "Aurora interface",
            }
        )

    assert session.requests[0].visual_intent == "figure"
    assert session.requests[0].image_query == "Aurora interface"
    public = json.loads(raw)
    assert public["sources"][0]["source_id"] == "S1"
    assert "images" not in public
    assert "I1" not in raw
    assert "/web-images/" not in raw


def test_nested_tool_contexts_restore_their_own_research_session() -> None:
    first = object()
    second = object()

    with tool_execution_context(web_research_session=first):
        assert get_tool_context().web_research_session is first
        with tool_execution_context(web_research_session=second):
            assert get_tool_context().web_research_session is second
        assert get_tool_context().web_research_session is first


@pytest.mark.asyncio
async def test_web_open_uses_the_same_turn_session() -> None:
    bundle = SimpleNamespace(
        status="success",
        mode="quick",
        operation_index=2,
        sources=(),
        images=(),
        failures=(),
        reused=False,
        omitted_source_count=0,
        omitted_image_count=0,
    )
    session = _Session(bundle)

    with tool_execution_context(web_research_session=session):
        await create_web_open_tool().ainvoke(
            {"urls": ["https://example.test/release"], "question": "When was it released?"}
        )

    assert session.opens == [(["https://example.test/release"], "When was it released?")]
