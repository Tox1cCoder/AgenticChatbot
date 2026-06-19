import pytest
from langchain_core.messages import AIMessage

from app.ai.agents.planning_agent import PlanningAgent
from app.ai.schemas import AgentMessage, MessageRole, TodoStatus


def _agent():
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.agent_config_key = "planning"
    agent.model_name = "test-model"
    agent.runtime_model_resolver = None
    agent.gemini_client = None
    agent.langchain_model = None
    agent.mcp_manager = None
    agent.tools = []
    agent._tools_generation_seen = 0
    return agent


class _FakeToolModel:
    def bind_tools(self, *_args, **_kwargs):
        return self


def _todo_call(description, todo_id="t1"):
    return {
        "id": "call-1",
        "name": "write_todos",
        "args": {
            "action": "set_todos",
            "todos": [
                {
                    "id": todo_id,
                    "description": description,
                    "status": TodoStatus.PENDING.value,
                    "order": 0,
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_generate_plan_attaches_satisfied_rubric_metadata(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(
        agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime()
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (_FakeToolModel(), False),
    )

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Implement native planning rubric grading")]),
        AIMessage(
            content='{"rubric":"- request_fit: The tasks fit the user request.","rationale":"Small feature plan."}'
        ),
        AIMessage(
            content='{"result":"satisfied","explanation":"ok","criteria":[{"name":"request_fit","passed":true}]}'
        ),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert (
        response.metadata["todos"][0]["description"] == "Implement native planning rubric grading"
    )
    assert response.metadata["planning_rubric"]["status"] == "satisfied"
    assert response.metadata["planning_rubric"]["iterations"] == 1


@pytest.mark.asyncio
async def test_generate_plan_revises_after_needs_revision(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(
        agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime()
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (_FakeToolModel(), False),
    )

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Fix backend")]),
        AIMessage(
            content='{"rubric":"- concrete_backend_scope: Backend tasks name concrete behavior.","rationale":"The candidate is vague."}'
        ),
        AIMessage(
            content='{"result":"needs_revision","explanation":"vague","criteria":[{"name":"concrete_backend_scope","passed":false,"gap":"Name the concrete backend behavior."}]}'
        ),
        AIMessage(
            content="",
            tool_calls=[_todo_call("Add native Planning rubric evaluator and metadata contract")],
        ),
        AIMessage(
            content='{"result":"satisfied","explanation":"ok","criteria":[{"name":"concrete_backend_scope","passed":true}]}'
        ),
    ]

    async def fake_invoke(_model, messages, run_config=None):
        if len(responses) == 2:
            assert "Name the concrete backend behavior" in str(messages[-1].content)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert response.metadata["todos"][0]["description"] == (
        "Add native Planning rubric evaluator and metadata contract"
    )
    assert response.metadata["planning_rubric"]["status"] == "satisfied"
    assert response.metadata["planning_rubric"]["iterations"] == 2


@pytest.mark.asyncio
async def test_generate_plan_marks_max_iterations_reached(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "planning_rubric_max_iterations", 1)
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(
        agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime()
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (_FakeToolModel(), False),
    )

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Fix backend")]),
        AIMessage(
            content='{"rubric":"- concrete_scope: Tasks should name the concrete behavior for this request.","rationale":"The candidate is vague."}'
        ),
        AIMessage(
            content='{"result":"needs_revision","explanation":"vague","criteria":[{"name":"concrete_scope","passed":false,"gap":"Be concrete."}]}'
        ),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.metadata["planning_rubric"]["status"] == "max_iterations_reached"
    assert response.metadata["planning_rubric"]["feedback"]


@pytest.mark.asyncio
async def test_generate_plan_marks_grader_error(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(
        agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime()
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (_FakeToolModel(), False),
    )

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Implement native planning rubric grading")]),
        AIMessage(
            content='{"rubric":"- request_fit: Tasks fit this request.","rationale":"Generated from context."}'
        ),
        RuntimeError("grader unavailable"),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert response.metadata["planning_rubric"]["status"] == "grader_error"
    assert "grader unavailable" in response.metadata["planning_rubric"]["error"]
