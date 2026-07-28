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


@pytest.mark.asyncio
async def test_prepare_planning_context_carries_created_plan_rubric_metadata():
    from types import SimpleNamespace
    from uuid import uuid4

    from app.services.message_service import MessageService

    conversation_id = uuid4()
    user_id = uuid4()

    class _TaskPlanService:
        def __init__(self):
            self.created = False

        def get_conversation_tasks(self, *_args, **_kwargs):
            if self.created:
                return [
                    SimpleNamespace(
                        id=uuid4(),
                        description="Implement native planning rubric evaluator",
                        status=SimpleNamespace(value="pending"),
                        task_order=0,
                    )
                ]
            return []

        async def create_task_plan(self, *_args, **_kwargs):
            self.created = True
            return [
                SimpleNamespace(
                    id=uuid4(),
                    description="Implement native planning rubric evaluator",
                    status=SimpleNamespace(value="pending"),
                    task_order=0,
                )
            ]

        def get_active_or_next_task(self, *_args, **_kwargs):
            return None

        # The streaming path awaits the async twins; delegate so the double has
        # one source of truth per behavior.
        async def aget_conversation_tasks(self, *args, **kwargs):
            return self.get_conversation_tasks(*args, **kwargs)

        async def aget_active_or_next_task(self, *args, **kwargs):
            return self.get_active_or_next_task(*args, **kwargs)

        def consume_last_planning_runtime_metadata(self):
            return {"planning_rubric": {"status": "satisfied", "iterations": 1}}

    service = MessageService.__new__(MessageService)
    service.task_plan_service = _TaskPlanService()

    result = await MessageService._prepare_planning_context(
        service,
        conversation_id=conversation_id,
        user_id=user_id,
        message_content="Plan the rubric work",
        planning_mode_enabled=True,
        plan_lifecycle=None,
    )

    assert result.rubric_metadata == {"status": "satisfied", "iterations": 1}


def test_bot_metadata_preserves_planning_rubric():
    from app.core.response_constants import build_bot_metadata

    response = type(
        "Response",
        (),
        {
            "metadata": {
                "planning_rubric": {
                    "status": "satisfied",
                    "iterations": 1,
                    "evaluations": [],
                }
            },
            "tool_artifacts": None,
            "suggested_questions": None,
            "agent_id": "planning_agent",
        },
    )()

    metadata = build_bot_metadata(response, persona=None)

    assert metadata["planning_rubric"]["status"] == "satisfied"
