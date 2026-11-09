"""Answer visuals created before an approval interrupt survive the resume.

The resumed run re-enters the specialist with a fresh tool middleware, so a
descriptor that lived only in the old middleware's memory was silently lost:
the final answer placed its marker and finalization dropped it as unknown.
These drive the real parent graph over a real specialist subgraph with a real
checkpointer, the same harness the approval tests use.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.ai.answer_visual_tools import CREATE_ANSWER_VISUAL_TOOL_NAME, create_answer_visual_tool
from app.ai.workflow.contracts import RoutingDecision, TurnIdentity
from app.ai.workflow.graph_builder import build_workflow_graph
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.runtime_context import WorkflowRuntimeContext
from app.ai.workflow.state import build_checkpoint_thread_id
from tests.test_specialist_tool_pipeline import (
    GATED_POLICY,
    ScriptedChatModel,
    _factory,
    _gated_tool,
    _ParentGraphWorkflow,
)

pytestmark = pytest.mark.usefixtures("no_tracing")


@pytest.fixture
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


_STEPS = {
    "variant": "steps",
    "title": "Identify the shape",
    "steps": [{"title": "Trace the skyline"}, {"title": "Compare width and height"}],
}


class _MarkerAnsweringModel(ScriptedChatModel):
    """Scripted calls, then an answer that places the marker the tool returned."""

    def _next_for(self, messages) -> AIMessage:
        if self.call_count < len(self.responses):
            return self._next()
        self.call_count += 1
        markers = [
            match
            for message in messages
            if isinstance(message, ToolMessage)
            for match in re.findall(r"<!--rich:[^>]+-->", str(message.content))
        ]
        return AIMessage(content="Before\n\n" + "\n\n".join(markers) + "\n\nAfter")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=self._next_for(messages))])


class _RecordingWorkflow(_ParentGraphWorkflow):
    def __init__(self, factory, policy):
        super().__init__(factory, policy)
        self.outcomes: list = []

    async def invoke_specialist_subgraph(self, node_name, state):
        outcome = await super().invoke_specialist_subgraph(node_name, state)
        self.outcomes.append(outcome)
        return outcome


async def test_a_visual_created_before_an_approval_interrupt_reaches_the_resumed_answer():
    executed: list[str] = []
    model = _MarkerAnsweringModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-visual",
                        "name": CREATE_ANSWER_VISUAL_TOOL_NAME,
                        "args": {"visual": _STEPS},
                    }
                ],
            ),
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {}}]),
        ]
    )
    factory = _factory(model, [create_answer_visual_tool(), _gated_tool(executed)])
    workflow = _RecordingWorkflow(factory, GATED_POLICY)

    class _Routing:
        async def route(self, _context, _inventory, **_kwargs):
            return RoutingDecision(agent_id="chat_agent", confidence=1.0, reason="scripted")

    class _ContextBuilder:
        async def build(self, request):
            return {"message": request.message}

    graph = build_workflow_graph(
        workflow, checkpointer=InMemorySaver(), context_schema=WorkflowRuntimeContext
    )
    thread_id = build_checkpoint_thread_id("conversation-1", "message-1")
    state = {
        "turn_identity": TurnIdentity(
            request_id="request-1", turn_id="message-1", checkpoint_thread_id=thread_id
        ),
        "messages": [HumanMessage(content="explain with steps, then write it")],
        "assistant_message_id": "assistant-1",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "agent_history": [],
        "execution_phase": "routing",
    }
    context = WorkflowRuntimeContext(
        routing_service=_Routing(),
        inventory=build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={}),
        routing_context_builder=_ContextBuilder(),
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    await graph.ainvoke(state, config=config, context=context)
    assert (await graph.aget_state(config)).interrupts, "the gated call must pause the turn"

    resume = [{"task_id": "call-1", "tool_call_id": "call-1", "type": "approve", "args": None}]
    await graph.ainvoke(Command(resume=resume), config=config, context=context)

    assert executed == ["write_file"]
    outcome = workflow.outcomes[-1]
    placed = re.findall(r"<!--rich:([^>]+)-->", outcome.response.message.content)
    assert placed, "the resumed answer places the visual's marker"
    admitted = {item["id"]: item for item in outcome.provenance.rich_items}
    assert placed[0] in admitted
    assert admitted[placed[0]]["type"] == "answer_visual"
    assert admitted[placed[0]]["payload"]["variant"] == "steps"


class _RecordingSession:
    """Records how the specialist closes its web session; holds no evidence."""

    def __init__(self) -> None:
        self.closed_with: list[str] = []
        self.prepared_images: dict = {}
        self.answer_sources: list = []
        self.figure_ids = frozenset()
        from app.ai.web_research.source_registry import SourceRegistry

        self.source_registry = SourceRegistry(max_sources=8)
        self.latest_image_query = None
        self.latest_image_objective = None

    def model_evidence_blocks(self) -> list:
        return []

    async def suspend(self) -> None:
        self.closed_with.append("suspend")

    async def abort(self) -> None:
        self.closed_with.append("abort")

    async def finish(self, *_args, **_kwargs):
        self.closed_with.append("finish")

    def __getattr__(self, name):
        raise AttributeError(name)


async def test_an_approval_pause_suspends_the_web_session_instead_of_aborting_it():
    from types import SimpleNamespace

    from app.ai.workflow.specialists import SpecialistFactory

    session = _RecordingSession()
    executed: list[str] = []
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "write_file", "args": {}}])
        ]
    )
    base = _factory(model, [_gated_tool(executed)])
    factory = SpecialistFactory(
        definitions=base._definitions,
        runtime_model_resolver=base._runtime_model_resolver,
        model_factory=base._model_factory,
        settings=base._settings,
        web_research_service=SimpleNamespace(new_session=lambda *a, **k: session),
    )
    graph = build_workflow_graph(
        _ParentGraphWorkflow(factory, GATED_POLICY),
        checkpointer=InMemorySaver(),
        context_schema=WorkflowRuntimeContext,
    )

    class _Routing:
        async def route(self, _context, _inventory, **_kwargs):
            return RoutingDecision(agent_id="chat_agent", confidence=1.0, reason="scripted")

    class _ContextBuilder:
        async def build(self, request):
            return {"message": request.message}

    thread_id = build_checkpoint_thread_id("conversation-1", "message-1")
    await graph.ainvoke(
        {
            "turn_identity": TurnIdentity(
                request_id="request-1", turn_id="message-1", checkpoint_thread_id=thread_id
            ),
            "messages": [HumanMessage(content="write it")],
            "assistant_message_id": "assistant-1",
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "agent_history": [],
            "execution_phase": "routing",
        },
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 30},
        context=WorkflowRuntimeContext(
            routing_service=_Routing(),
            inventory=build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={}),
            routing_context_builder=_ContextBuilder(),
        ),
    )

    assert session.closed_with == ["suspend"]
