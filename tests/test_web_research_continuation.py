from __future__ import annotations

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.continuation import make_continuation_pause_node
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome


async def test_continue_carries_private_web_source_snapshot() -> None:
    source = {
        "source_id": "S1",
        "url": "https://example.test/source",
        "title": "Source",
        "snippet": "Evidence",
        "status": "search_result",
        "published_at": None,
        "provider": "test",
        "query_index": 1,
    }
    outcome = ResponseOutcome(
        agent_id="search_agent",
        response=AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="Partial"),
        ),
        provenance=OutcomeProvenance(web_sources=(source,)),
    )

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {
            "action": "continue",
            "expected_epoch": payload["execution_epoch"],
        }
    )
    command = await node(
        {
            "agent_outcome": outcome,
            "active_agent_id": "search_agent",
            "execution_epoch": 0,
            "generation_id": "generation",
            "logical_turn_id": "turn",
        }
    )

    assert command.update["carried_web_sources"] == [source]
    assert command.goto == "search_agent"
