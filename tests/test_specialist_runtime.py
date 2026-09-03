"""Per-invocation specialist subgraphs.

A specialist is compiled fresh for every invocation and carries the caller's
authenticated scope. Nothing about one user's or device's invocation may
survive into another's, and a worker invocation never produces a public
message.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.schemas import AgentType
from app.ai.workflow.contracts import ResponseOutcome, WorkerResult, WorkerTask
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
)


class RecordingAgent:
    """Stands in for a compiled ``create_agent`` subgraph."""

    instances: list[RecordingAgent] = []

    def __init__(self, *, messages=None, raises=None):
        self._messages = messages or [AIMessage(content="specialist answer", id="private-1")]
        self._raises = raises
        self.invoked_context = None
        RecordingAgent.instances.append(self)

    async def ainvoke(self, payload, context=None, config=None):
        self.invoked_context = context
        if self._raises is not None:
            raise self._raises
        return {"messages": [*payload["messages"], *self._messages]}


def _definition(agent_id="chat_agent", **overrides):
    payload = {
        "agent_id": agent_id,
        "agent_type": AgentType.CHAT,
        "model_config_key": "chat",
        "system_prompt_factory": lambda request: f"system prompt for {request.agent_id}",
        "tool_factory": lambda request: [],
        "output_policy_ids": ("public_content",),
    }
    payload.update(overrides)
    return SpecialistDefinition(**payload)


def _request(agent_id="chat_agent", *, user_id="user-1", device_id="device-1", **overrides):
    payload = {
        "agent_id": agent_id,
        "conversation_id": "conversation-1",
        "user_id": user_id,
        "device_id": device_id,
        "persona": None,
        "model_request": None,
        "messages": [HumanMessage(content="hello")],
        "history": [],
        "state": {},
    }
    payload.update(overrides)
    return SpecialistRequest(**payload)


def _factory(**overrides):
    RecordingAgent.instances = []
    built: list[dict] = []

    def build_agent(**kwargs):
        built.append(kwargs)
        return RecordingAgent()

    payload = {
        "definitions": {"chat_agent": _definition()},
        "runtime_model_resolver": SimpleNamespace(
            resolve_runtime_config=lambda *a, **k: SimpleNamespace(
                provider="gemini", model="m", temperature=1.0, api_key="k", capabilities={}
            )
        ),
        "model_factory": SimpleNamespace(create_model_from_runtime=lambda config, **kw: object()),
        "agent_builder": build_agent,
        "usage_recorder": None,
        "settings": SimpleNamespace(
            specialist_max_model_calls=8,
            specialist_max_tool_calls=16,
        ),
    }
    payload.update(overrides)
    factory = SpecialistFactory(**payload)
    factory.built = built
    return factory


async def test_standard_specialist_is_created_per_invocation():
    factory = _factory()

    first = await factory.invoke(_request(device_id="device-a"))
    second = await factory.invoke(_request(device_id="device-b"))

    assert isinstance(first, ResponseOutcome)
    assert isinstance(second, ResponseOutcome)
    # Two invocations, two compiled agents: nothing is cached across devices.
    assert len(RecordingAgent.instances) == 2
    assert RecordingAgent.instances[0] is not RecordingAgent.instances[1]


async def test_each_invocation_carries_its_own_authenticated_scope():
    factory = _factory()
    await factory.invoke(_request(user_id="user-a", device_id="device-a"))
    await factory.invoke(_request(user_id="user-b", device_id="device-b"))

    scopes = [
        (agent.invoked_context.user_id, agent.invoked_context.device_id)
        for agent in RecordingAgent.instances
    ]
    assert scopes == [("user-a", "device-a"), ("user-b", "device-b")]


async def test_compiled_specialists_are_never_cached_across_users():
    factory = _factory()
    await factory.invoke(_request(user_id="user-a"))
    await factory.invoke(_request(user_id="user-a"))
    assert len(RecordingAgent.instances) == 2


async def test_specialist_result_becomes_a_server_owned_response_outcome():
    factory = _factory()
    outcome = await factory.invoke(_request())

    assert isinstance(outcome, ResponseOutcome)
    assert outcome.agent_id == "chat_agent"
    assert outcome.response.message.content == "specialist answer"
    assert outcome.provenance.output_policy_ids == ("public_content",)


async def test_specialist_private_messages_do_not_become_public_content():
    factory = _factory(
        definitions={
            "chat_agent": _definition(),
        }
    )
    RecordingAgent.instances = []

    def build_agent(**kwargs):
        return RecordingAgent(
            messages=[
                AIMessage(content="", id="tool-turn", tool_calls=[]),
                AIMessage(content="final answer", id="private-final"),
            ]
        )

    factory._agent_builder = build_agent
    outcome = await factory.invoke(_request())

    assert outcome.response.message.content == "final answer"
    assert len(outcome.provenance.private_messages) >= 1


async def test_system_prompt_factory_receives_the_request():
    seen: list[str] = []

    def prompt_factory(request):
        seen.append(request.agent_id)
        return "prompt"

    factory = _factory(
        definitions={"chat_agent": _definition(system_prompt_factory=prompt_factory)}
    )
    await factory.invoke(_request())
    assert seen == ["chat_agent"]


async def test_async_prompt_and_tool_factories_are_awaited():
    async def prompt_factory(request):
        return "async prompt"

    async def tool_factory(request):
        return []

    factory = _factory(
        definitions={
            "chat_agent": _definition(
                system_prompt_factory=prompt_factory, tool_factory=tool_factory
            )
        }
    )
    await factory.invoke(_request())
    assert factory.built[0]["system_prompt"] == "async prompt"


async def test_unknown_specialist_is_rejected():
    factory = _factory()
    with pytest.raises(KeyError):
        await factory.invoke(_request(agent_id="nope_agent"))


def _worker_task(task_id: str = "t1", agent_id: str = "chat_agent", **overrides) -> WorkerTask:
    """The server-owned identity a dispatched worker carries."""
    payload = {
        "dispatch_id": "d1",
        "task_id": task_id,
        "position": 0,
        "objective": f"do {task_id}",
        "agent_id": agent_id,
    }
    payload.update(overrides)
    return WorkerTask(**payload)


async def test_worker_mode_never_appends_a_public_message():
    factory = _factory()
    result = await factory.invoke_worker(_request(), task=_worker_task())

    assert isinstance(result, WorkerResult)
    assert result.task_id == "t1"
    assert result.agent_id == "chat_agent"
    assert result.status == "completed"
    assert "public_messages" not in WorkerResult.model_fields


async def test_worker_failure_becomes_a_typed_failed_result():
    factory = _factory()

    def build_agent(**kwargs):
        return RecordingAgent(raises=RuntimeError("tool exploded"))

    factory._agent_builder = build_agent
    result = await factory.invoke_worker(_request(), task=_worker_task())

    assert result.status == "failed"
    assert result.error_code == "tool_execution_failed"
    assert result.content == ""


async def test_worker_execution_limit_maps_to_the_typed_code():
    from langchain.agents.middleware import ModelCallLimitMiddleware  # noqa: F401

    from app.ai.workflow.specialists import ModelCallLimitExceededError

    factory = _factory()

    def build_agent(**kwargs):
        return RecordingAgent(
            raises=ModelCallLimitExceededError(
                thread_count=9, run_count=9, thread_limit=8, run_limit=None
            )
        )

    factory._agent_builder = build_agent
    result = await factory.invoke_worker(_request(), task=_worker_task())

    assert result.status == "failed"
    assert result.error_code == "agent_execution_limit"


async def test_recursive_planning_worker_is_rejected():
    factory = _factory(
        definitions={
            "chat_agent": _definition(),
            "planning_agent": _definition("planning_agent", agent_type=AgentType.PLANNING),
        }
    )
    result = await factory.invoke_worker(
        _request(agent_id="planning_agent"), task=_worker_task(agent_id="planning_agent")
    )

    assert result.status == "failed"
    assert result.error_code == "recursive_planning"


def test_specialist_definition_is_frozen():
    import dataclasses

    definition = _definition()
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.agent_id = "other"
