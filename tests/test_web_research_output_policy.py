from __future__ import annotations

import pytest

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome, RoutingDecision
from app.ai.workflow.finalization import UNVERIFIED_WEB_RESPONSE, OutputValidator


def _outcome(
    content: str,
    *,
    sources: tuple[dict, ...] = (),
    rich_items: tuple[dict, ...] = (),
) -> ResponseOutcome:
    return ResponseOutcome(
        agent_id="search_agent",
        response=AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
            metadata={"_rich_item_candidates": list(rich_items)} if rich_items else {},
        ),
        provenance=OutcomeProvenance(web_sources=sources, rich_items=rich_items),
    )


@pytest.mark.asyncio
async def test_admitted_source_link_passes_web_policy() -> None:
    outcome = _outcome(
        "Verified [1](https://example.test/source).",
        sources=({"source_id": "S1", "url": "https://example.test/source"},),
    )

    validated = await OutputValidator().validate(outcome, {})

    assert validated.response.message.content.startswith("Verified")
    assert "web_evidence" in validated.provenance.output_policy_ids


@pytest.mark.asyncio
async def test_uncited_web_answer_is_replaced_not_published() -> None:
    image = {"id": "image:web:one", "type": "image", "payload": {"url": "/web-images/one"}}
    outcome = _outcome(
        "Unsupported current claim. <!--rich:image:web:one-->",
        sources=({"source_id": "S1", "url": "https://example.test/source"},),
        rich_items=(image,),
    )

    validated = await OutputValidator().validate(outcome, {})

    assert validated.response.message.content == UNVERIFIED_WEB_RESPONSE
    assert validated.response.metadata["web_verification"]["reason"] == "missing_valid_citation"
    assert "_rich_item_candidates" not in validated.response.metadata
    assert validated.provenance.rich_items == ()


@pytest.mark.asyncio
async def test_required_route_without_evidence_gets_server_owned_response() -> None:
    outcome = _outcome("The current price is 100.")
    state = {
        "routing_decision": RoutingDecision(
            agent_id="chat_agent",
            confidence=1,
            reason="current price",
            requires_web=True,
            research_mode="quick",
        )
    }

    validated = await OutputValidator().validate(outcome, state)

    assert validated.response.message.content == UNVERIFIED_WEB_RESPONSE
    assert validated.response.metadata["web_verification"]["reason"] == "no_admitted_sources"
