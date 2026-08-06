"""Trace-shaped regression for the T1 turn in example_run.txt.

Original failure: the model was offered an author portrait labelled "Moi" and a
wiki asset labelled "research", while the one image that depicted the team was
rejected by a URL resize parameter. Relevance was inferred from the page title.
"""

from __future__ import annotations

import json
import re

import pytest

from app.ai.research_budget import reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.verified_image_sink import verified_image_sink
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.ai.web_research_tool import create_web_research_tool
from app.services.web_image_service import FetchedWebImage

CONVERSATION_ID = "22222222-2222-2222-2222-222222222222"
PORTRAIT_URL = "https://cdn.example/moi-1200x1600.jpg"
TEAM_URL = "https://cdn.example/t1-team-995x565.webp?w=3840&q=75"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
                "score": 0.887,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 League of Legends Esports team news roster 2026",
    }
)

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": PORTRAIT_URL,
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "Moi",
                "description": "Moi",
                "width": 1200,
                "height": 1600,
                "source_url": "https://sheepesports.example/t1",
            },
            {
                "url": TEAM_URL,
                "provider": "brave_image_search",
                "mime_type": "image/webp",
                "title": "T1 2026 roster",
                "description": "T1 2026 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            },
        ],
        "total_results": 2,
    }
)


class _Tool:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, args: dict) -> str:
        return self.payload


class _Service:
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        if url == PORTRAIT_URL:
            return FetchedWebImage(
                content=b"portrait", media_type="image/jpeg", width=1200, height=1600
            )
        return FetchedWebImage(
            content=b"team", media_type="image/webp", width=995, height=565
        )


class _Verifier:
    """Rejects the portrait on visible content, approves the team photo."""

    async def ainvoke(self, messages):
        text = messages[0].content[0]["text"]
        decisions = []
        for candidate_id, line in re.findall(r"^- (c\d+): (.*)$", text, flags=re.MULTILINE):
            portrait = "Moi" in line
            decisions.append(
                VisualCandidateDecision(
                    candidate_id=candidate_id,
                    depicts_requested_subject=not portrait,
                    materially_supports_answer=not portrait,
                    confidence=0.93 if not portrait else 0.91,
                    content_kind="portrait" if portrait else "photo",
                )
            )
        return VisualVerificationResult(decisions=decisions)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        "app.ai.web_research_tool.settings.vision_image_verification_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


@pytest.mark.asyncio
async def test_portrait_rejected_team_photo_approved():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(
            {
                "query": "T1 League of Legends Esports team news roster 2026",
                "image_query": "T1 League of Legends team photo",
            }
        )

    serialized_sink = json.dumps(sink)
    assert len(sink) == 1
    assert sink[0]["payload"]["url"] == TEAM_URL
    assert "Moi" not in serialized_sink
    assert PORTRAIT_URL not in serialized_sink
    assert PORTRAIT_URL not in raw
    assert "Moi" not in raw


@pytest.mark.asyncio
async def test_no_verifier_field_reaches_the_public_candidate():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )

    serialized = json.dumps(sink)
    for forbidden in (
        "confidence",
        "content_kind",
        "depicts_requested_subject",
        "materially_supports_answer",
        "candidate_id",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_approved_candidate_carries_decoded_dimensions():
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_Verifier(),
    )

    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )

    assert sink[0]["payload"]["width"] == 995
    assert sink[0]["payload"]["height"] == 565
    assert sink[0]["payload"]["mime_type"] == "image/webp"
