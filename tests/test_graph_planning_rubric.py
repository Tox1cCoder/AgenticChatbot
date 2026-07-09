import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.planning_rubric import PlanningRubricAttempt, PlanningRubricEvaluation
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


@pytest.mark.asyncio
async def test_planning_tools_node_stores_rubric_feedback_for_plan_mutation(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = object()

    async def fake_ensure_map(*_args, **_kwargs):
        return {}

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)

    async def fake_review(**_kwargs):
        return PlanningRubricAttempt(
            status="needs_revision",
            grading_run_id="run-1",
            iterations=1,
            source="planning_tools",
            rubric="- concrete_plan_scope",
            evaluations=[
                PlanningRubricEvaluation(
                    iteration=0,
                    result="needs_revision",
                    explanation="Task is vague.",
                    criteria=[
                        {
                            "name": "concrete_plan_scope",
                            "passed": False,
                            "gap": "Replace vague task with concrete behavior.",
                        }
                    ],
                )
            ],
            feedback="Replace vague task with concrete behavior.",
        )

    workflow._review_planning_todos_with_rubric = fake_review  # type: ignore[assignment]

    state = {
        "messages": [
            HumanMessage(content="make a plan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "todo-1",
                        "name": "write_todos",
                        "args": {
                            "action": "set_todos",
                            "todos": [
                                {
                                    "id": "t1",
                                    "description": "Fix backend",
                                    "status": "pending",
                                    "order": 0,
                                }
                            ],
                        },
                    }
                ],
            ),
        ],
        "todos": [],
        "current_task_index": None,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "planning",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
    }

    result = await workflow._planning_tools_node(state)

    assert result["context"]["planning_rubric"]["status"] == "needs_revision"
    assert "Replace vague task" in result["context"]["planning_rubric_feedback"]
    assert result["context"]["plan_just_modified"] is False


def test_should_continue_planning_routes_back_for_rubric_feedback():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"planning_agent": object()}
    state = {
        "planning_call_count": 1,
        "planning_mode_enabled": True,
        "planning_phase": "planning",
        "todos": [{"id": "t1", "description": "Fix backend", "status": "pending", "order": 0}],
        "messages": [HumanMessage(content="plan"), AIMessage(content="Plan updated")],
        "context": {"planning_rubric_feedback": "Make task concrete."},
    }

    assert workflow._should_continue_planning(state) == "planning_agent"
