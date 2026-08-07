"""The full chain from an approved image to the marker the model can copy.

Every existing test stops at one seam: the regression test proves the verifier
approves the right image and stops at the sink; the prompt tests build an
inventory from hand-written candidates. Nothing joined them, so a break in
sink -> artifact -> turn context -> selection -> inventory would surface only as
"the assistant never shows an image", with every suite green.

That is exactly the failure mode being guarded here, because the answer to
"why is there no image" was a disabled flag once already, and the next time it
could as easily be a dropped candidate three layers down.
"""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.prompts import build_rich_response_guidance
from app.ai.research_budget import reset_research_budget
from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_execution import _attach_rich_candidates_to_artifact
from app.ai.verified_image_sink import verified_image_sink
from app.ai.visual_verifier import VisualCandidateDecision, VisualVerificationResult
from app.ai.web_research_tool import create_web_research_tool
from app.ai.workflow.tool_loop import ToolLoopMixin
from app.services.web_image_service import FetchedWebImage

CONVERSATION_ID = "77777777-7777-7777-7777-777777777777"
TEAM_URL = "https://cdn.example/t1-team.webp"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
                "score": 0.9,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
    }
)

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": TEAM_URL,
                "provider": "brave_image_search",
                "mime_type": "image/webp",
                "title": "T1 2026 roster",
                "description": "T1 2026 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            }
        ],
        "total_results": 1,
    }
)


class _Tool:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, args: dict) -> str:
        return self.payload


class _Service:
    async def fetch_url(self, url: str, *, provider: str = "other") -> FetchedWebImage:
        return FetchedWebImage(
            content=b"team", media_type="image/webp", width=995, height=565
        )


class _ApproveAll:
    async def ainvoke(self, messages):
        return VisualVerificationResult(
            decisions=[
                VisualCandidateDecision(
                    candidate_id="c0",
                    depicts_requested_subject=True,
                    materially_supports_answer=True,
                    confidence=0.95,
                    content_kind="photo",
                )
            ]
        )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_research_tool.settings,
        "vision_image_verification_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


async def _approved_candidates() -> list[dict]:
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
        web_image_service=_Service(),
        verifier_model=_ApproveAll(),
    )
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        await tool.ainvoke(
            {"query": "T1 roster 2026", "image_query": "T1 League of Legends team photo"}
        )
    return list(sink)


@pytest.mark.asyncio
async def test_an_approved_image_becomes_a_marker_the_model_can_copy():
    offered = await _approved_candidates()
    assert offered, "verification approved nothing — the chain starts empty"

    artifact: dict = {}
    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        verified_images=offered,
    )

    context: dict = {}
    ToolLoopMixin._lift_rich_candidates(context, [artifact])
    apply_rich_image_selection(context)

    candidates = context.get("rich_item_candidates") or []
    assert candidates, "the approved image was dropped between the sink and selection"

    guidance = build_rich_response_guidance(
        candidates=candidates, enabled=True, capability=True
    )

    item_id = candidates[0]["id"]
    assert f"<!--rich:{item_id}-->" in guidance
    assert "AVAILABLE RICH ITEMS" in guidance


@pytest.mark.asyncio
async def test_the_offered_marker_carries_the_decoded_dimensions():
    """Selection rejects on dimensions, so wrong ones silently empty the list."""
    offered = await _approved_candidates()

    payload = offered[0]["payload"]
    assert payload["width"] == 995
    assert payload["height"] == 565
    assert payload["mime_type"] == "image/webp"


@pytest.mark.asyncio
async def test_no_inventory_is_offered_when_the_flag_is_off(monkeypatch):
    """The current production state: the whole chain yields nothing by design."""
    monkeypatch.setattr(
        web_research_tool.settings,
        "vision_image_verification_enabled",
        False,
        raising=False,
    )

    assert await _approved_candidates() == []
    assert build_rich_response_guidance(candidates=[], enabled=True, capability=True) == ""
