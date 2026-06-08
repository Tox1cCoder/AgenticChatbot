from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_attach_planning_state_metadata_includes_planning_rubric():
    response = AgentResponse(
        agent_type=AgentType.PLANNING,
        agent_id="planning_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Plan ready"),
        metadata={},
    )
    state = {
        "planning_call_count": 1,
        "context": {
            "planning_rubric": {
                "status": "satisfied",
                "iterations": 1,
                "evaluations": [],
            }
        },
    }

    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, state)

    assert enriched.metadata["planning_rubric"]["status"] == "satisfied"
