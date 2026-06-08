import pytest

from app.ai.planning_runtime_adapter import PlanningRuntimeAdapter
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


@pytest.mark.asyncio
async def test_planning_runtime_adapter_preserves_planning_rubric_metadata():
    class _PlanningAgent:
        async def generate_plan(self, message, conversation_id=None):
            assert message.role == MessageRole.USER
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content="plan"),
                metadata={
                    "todos": [
                        {
                            "id": "t1",
                            "description": "Implement the native rubric evaluator",
                            "status": "pending",
                            "order": 0,
                        }
                    ],
                    "planning_rubric": {"status": "satisfied", "iterations": 1},
                },
            )

    from app.schemas.task_plan import PlanningRuntimeRequest

    result = await PlanningRuntimeAdapter(_PlanningAgent()).generate_plan(
        PlanningRuntimeRequest(
            user_message="Add rubric grading",
            conversation_id="conv-1",
            user_id="user-1",
        )
    )

    assert result.todos[0]["id"] == "t1"
    assert result.metadata["planning_rubric"]["status"] == "satisfied"
