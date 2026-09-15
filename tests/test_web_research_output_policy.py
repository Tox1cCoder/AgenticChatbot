from __future__ import annotations

import pytest

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome, RoutingDecision
from app.ai.workflow.finalization import OutputValidationError, OutputValidator


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
async def test_uncited_web_answer_is_rejected_without_rewriting_model_content() -> None:
    image = {"id": "image:web:one", "type": "image", "payload": {"url": "/web-images/one"}}
    outcome = _outcome(
        "Unsupported current claim. <!--rich:image:web:one-->",
        sources=({"source_id": "S1", "url": "https://example.test/source"},),
        rich_items=(image,),
    )

    with pytest.raises(OutputValidationError) as excinfo:
        await OutputValidator().validate(outcome, {})

    assert excinfo.value.reason == "missing_web_citation"
    assert excinfo.value.retriable is True
    assert outcome.response.message.content == (
        "Unsupported current claim. <!--rich:image:web:one-->"
    )
    assert outcome.response.metadata["_rich_item_candidates"] == [image]
    assert outcome.provenance.rich_items == (image,)


@pytest.mark.asyncio
async def test_required_route_without_admitted_sources_keeps_the_model_answer() -> None:
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

    assert validated.response.message.content == "The current price is 100."
    assert "web_evidence" not in validated.provenance.output_policy_ids
