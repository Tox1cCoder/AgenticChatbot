"""A custom agent's definition belongs to one turn, not to the shared factory.

The workflow builds one ``SpecialistFactory`` and every turn shares it. Custom
agents used to be registered on it and never removed, so:

* a Planning worker, which never registered one, either failed as
  ``agent_unavailable`` or ran whatever an earlier public turn had left there
  (that turn's prompt, tools, and handoff targets);
* the registry grew by one entry for every custom agent ever used.

These drive the real factory the workflow builds, through the real entry
points (``invoke_specialist_subgraph`` and ``PlanningWorkerRuntime``), with a
scripted model standing in for the provider.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import app.ai.graph as graph_module
from app.ai.agents.custom_agent import CustomAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.workflow.contracts import WorkerTask
from app.ai.workflow.inventory import CUSTOM_AGENT_NODE
from app.ai.workflow.planning_execution import PlanningLimits, PlanningWorkerRuntime

pytestmark = pytest.mark.usefixtures("disable_langsmith_tracing")


@pytest.fixture
def disable_langsmith_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class RecordingModel(BaseChatModel):
    """Answers every call and records the system prompt it was given."""

    system_prompts: list[str] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self

    def _record(self, messages) -> ChatResult:
        system = next((m for m in messages if getattr(m, "type", None) == "system"), None)
        self.system_prompts.append(str(getattr(system, "content", "")))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._record(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._record(messages)


def _runtime_config():
    return SimpleNamespace(
        agent_key="custom",
        provider="gemini",
        model="gemini-3-flash-preview",
        temperature=1.0,
        api_key="key",
        key_source="user",
        source="default",
        warnings=[],
        capabilities={},
        fallback_config=None,
    )


@pytest.fixture
def model():
    return RecordingModel(system_prompts=[])


@pytest.fixture
def bound_specs():
    """The runtime spec each custom-agent tool binding was built from."""
    return []


@pytest.fixture(autouse=True)
def offline_custom_agent(monkeypatch, model, bound_specs):
    """Keep the custom agent off MCP, the device runtime, and the tokenizer.

    The prompt is the spec's own instructions, so a test can read which
    definition a model call ran from what the model was told.
    """
    model_factory = SimpleNamespace(create_model_from_runtime=lambda config, **kwargs: model)
    monkeypatch.setattr(graph_module, "ModelFactory", model_factory)
    monkeypatch.setattr(CustomAgent, "_init_tools", AsyncMock(return_value=None))
    monkeypatch.setattr(CustomAgent, "_preflight_model_request", AsyncMock(return_value=None))
    monkeypatch.setattr(
        CustomAgent,
        "_build_system_prompt",
        lambda self, persona, has_tool_context, **kwargs: f"INSTRUCTIONS: {self.spec.prompt}",
    )

    def _tools(self, **kwargs):
        bound_specs.append(self.spec)
        return []

    monkeypatch.setattr(CustomAgent, "_get_tools_for_binding", _tools)


def _workflow() -> MultiAgentWorkflow:
    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    placeholder = SimpleNamespace(_convert_history_to_langchain_messages=lambda history: [])
    wf.chat_agent = placeholder
    wf.search_agent = placeholder
    wf.canvas_agent = placeholder
    wf.image_generator_agent = placeholder
    wf.agents = {
        "chat_agent": placeholder,
        "rag_agent": placeholder,
        "search_agent": placeholder,
        "image_generator_agent": placeholder,
        "planning_agent": placeholder,
        "canvas_agent": placeholder,
    }
    wf._runtime_model_resolver = SimpleNamespace(
        resolve_runtime_config=lambda *args, **kwargs: _runtime_config()
    )
    wf._model_usage_recorder = None
    wf._web_research_service = None
    wf._get_conversation_history = AsyncMock(return_value=[])
    wf._specialist_factory = wf._build_specialist_factory()
    wf.planning_worker_runtime = PlanningWorkerRuntime(
        specialist_factory=wf._specialist_factory,
        rag_execution_graph=object(),
        limits=PlanningLimits(
            max_tasks=8,
            max_concurrency=4,
            max_dispatch_waves=2,
            objective_max_chars=4000,
            parent_context_max_chars=12000,
        ),
    )
    return wf


def _entry(runtime_id: str, prompt: str) -> dict:
    return {
        "id": runtime_id.split(":", 1)[1],
        "runtime_agent_id": runtime_id,
        "name": "Data Analyst",
        "prompt": prompt,
        "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
        "tool_refs": [],
        "skill_refs": [],
    }


def _turn_state(roster: dict[str, dict], active_agent_id: str | None = None) -> dict:
    return {
        "active_agent_id": active_agent_id,
        "messages": [HumanMessage(content="analyze the sales data")],
        "conversation_id": None,
        "user_id": "user-1",
        "device_id": None,
        "persona": None,
        "model_request": None,
        "context": {},
        "custom_agents": roster,
    }


def _task(agent_id: str) -> WorkerTask:
    return WorkerTask(
        dispatch_id="dispatch-1",
        task_id="task-1",
        position=0,
        objective="analyze the sales data",
        agent_id=agent_id,
    )


async def test_a_planning_worker_resolves_a_custom_agent_nobody_registered(model):
    """The first custom-agent use in a conversation is often as a worker."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"

    result = await wf.planning_worker_runtime.run(
        _task(rid), _turn_state({rid: _entry(rid, "worker instructions")})
    )

    assert result.status == "completed", result.error_code
    assert result.content == "done"
    assert model.system_prompts == ["INSTRUCTIONS: worker instructions"]


async def test_each_turn_runs_its_own_definition_of_the_same_custom_agent(model, bound_specs):
    """A public turn must not leave its definition behind for a later worker.

    Turn one runs the agent with version-one instructions and a handoff target.
    The agent is then edited, and turn two dispatches it as a Planning worker:
    it must run version two, and without the parent-level ``hand_off``.
    """
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    other = f"custom_agent:{uuid4()}"
    first_roster = {rid: _entry(rid, "version one"), other: _entry(other, "other agent")}

    await wf.invoke_specialist_subgraph(CUSTOM_AGENT_NODE, _turn_state(first_roster, rid))
    result = await wf.planning_worker_runtime.run(
        _task(rid), _turn_state({rid: _entry(rid, "version two")})
    )

    assert result.status == "completed", result.error_code
    assert model.system_prompts == ["INSTRUCTIONS: version one", "INSTRUCTIONS: version two"]
    public_spec, worker_spec = bound_specs[0], bound_specs[-1]
    assert other in public_spec.allowed_handoff_targets
    assert worker_spec.prompt == "version two"
    assert worker_spec.allowed_handoff_targets == []


async def test_the_shared_factory_does_not_grow_across_custom_agent_turns():
    wf = _workflow()
    factory = wf._specialist_factory
    before = dict(factory._definitions)

    for _ in range(3):
        rid = f"custom_agent:{uuid4()}"
        await wf.invoke_specialist_subgraph(
            CUSTOM_AGENT_NODE, _turn_state({rid: _entry(rid, "instructions")}, rid)
        )

    assert factory._definitions == before


async def test_a_worker_for_a_custom_agent_missing_from_its_roster_is_unavailable():
    """Resolving per request must not fall back to some other turn's agent."""
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    await wf.invoke_specialist_subgraph(
        CUSTOM_AGENT_NODE, _turn_state({rid: _entry(rid, "instructions")}, rid)
    )

    result = await wf.planning_worker_runtime.run(_task(rid), _turn_state({}))

    assert result.status == "failed"
    assert result.error_code == "agent_unavailable"
