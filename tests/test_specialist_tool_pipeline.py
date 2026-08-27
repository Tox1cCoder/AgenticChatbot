"""What a standard specialist's tools get when the framework loop runs them.

Task 5 moved chat/search/canvas/image/custom off the bespoke tool loop and
onto ``create_agent``. The bespoke loop did more than call the function: it
established the tool execution context, ran the product's tool pipeline
(artifacts, images, offloading, rich items, deferred re-binding), and applied
the approval policy. These assert that the subgraph still does all of it,
because a tool that silently loses its device binding fails closed and looks
like a model mistake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.ai.schemas import AgentType
from app.ai.tool_context import get_tool_context
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
)

pytestmark = pytest.mark.usefixtures("no_tracing")


@pytest.fixture
def no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class ScriptedChatModel(BaseChatModel):
    responses: list = []
    call_count: int = 0
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self) -> AIMessage:
        message = self.responses[self.call_count]
        self.call_count += 1
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])


def _resolver():
    return SimpleNamespace(
        resolve_runtime_config=lambda *a, **k: SimpleNamespace(
            agent_key="chat",
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
    )


def _factory(model, tools):
    definition = SpecialistDefinition(
        agent_id="chat_agent",
        agent_type=AgentType.CHAT,
        model_config_key="chat",
        system_prompt_factory=lambda request: "You are a helpful assistant.",
        tool_factory=lambda request: list(tools),
        output_policy_ids=("public_content",),
    )
    return SpecialistFactory(
        definitions={"chat_agent": definition},
        runtime_model_resolver=_resolver(),
        model_factory=SimpleNamespace(create_model_from_runtime=lambda config, **kw: model),
        usage_recorder=None,
        settings=SimpleNamespace(specialist_max_model_calls=4, specialist_max_tool_calls=4),
    )


def _request(**overrides):
    payload = {
        "agent_id": "chat_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "persona": None,
        "model_request": None,
        "messages": [HumanMessage(content="go")],
        "history": [],
        "state": {},
    }
    payload.update(overrides)
    return SpecialistRequest(**payload)


def _calls_then_answers(tool_name: str) -> list[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": tool_name, "args": {}}]),
        AIMessage(content="done"),
    ]


async def test_a_tool_sees_the_authenticated_execution_scope():
    """A client tool refuses itself when the context carries no device."""
    seen: dict[str, object] = {}

    @tool
    def report_scope() -> str:
        """Report the ambient tool execution context."""
        context = get_tool_context()
        seen.update(
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            device_id=context.device_id,
            agent_key=context.agent_key,
        )
        return "ok"

    model = ScriptedChatModel(responses=_calls_then_answers("report_scope"))
    await _factory(model, [report_scope]).invoke(_request())

    assert seen == {
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "agent_key": "chat",
    }


async def test_a_tool_result_becomes_a_server_owned_artifact():
    """Provenance is what later validation and the UI trace both read.

    No product tool sets ``ToolMessage.artifact``; artifacts are built by the
    product's execution pipeline. A specialist that skips the pipeline reports
    an answer with no record of the tools that produced it.
    """

    @tool
    def lookup_price() -> str:
        """Look up a ticker price."""
        return "42"

    model = ScriptedChatModel(responses=_calls_then_answers("lookup_price"))
    outcome = await _factory(model, [lookup_price]).invoke(_request())

    assert [artifact.get("tool") for artifact in outcome.provenance.artifacts] == ["lookup_price"]


async def test_a_policy_gated_tool_does_not_run_without_approval():
    """The approval policy is origin- and mutation-scoped, not a name list.

    Matching only ``global_tools`` lets every editable client rule and the
    mutation floor through, which is the whole approval feature for MCP tools.
    """
    executed: list[str] = []

    @tool
    def delete_everything() -> str:
        """Delete the thing."""
        executed.append("ran")
        return "gone"

    delete_everything.metadata = {
        "tool_origin": "client_mcp",
        "server_name": "files",
        "qualified_tool_id": "files::delete_everything",
        "mutation": True,
    }

    policy = {
        "master_enabled": True,
        "global_tools": [],
        "client_rules": {
            "client_mcp": {"servers": {"files": True}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
    }

    model = ScriptedChatModel(responses=_calls_then_answers("delete_everything"))
    factory = _factory(model, [delete_everything])

    # Standalone there is no parent loop to pause, so the subgraph comes back
    # holding an interrupt. Inside the workflow graph the same interrupt
    # propagates and the turn waits — see the parent-graph test below.
    with pytest.raises(RuntimeError, match="paused for approval"):
        await factory.invoke(_request(hitl_policy=policy))

    assert executed == [], "a gated tool ran before anyone approved it"


async def test_an_ungated_tool_still_runs_when_another_call_is_gated():
    """One gated call in a batch must not stop the rest of the work."""
    executed: list[str] = []

    @tool
    def read_only() -> str:
        """Read the thing."""
        executed.append("read_only")
        return "value"

    read_only.metadata = {"tool_origin": "client_mcp", "server_name": "files"}

    policy = {
        "master_enabled": True,
        "global_tools": ["write_only"],
        "client_rules": {
            "client_mcp": {"servers": {}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
    }

    model = ScriptedChatModel(responses=_calls_then_answers("read_only"))
    outcome = await _factory(model, [read_only]).invoke(_request(hitl_policy=policy))

    assert executed == ["read_only"]
    assert outcome.response.message.content == "done"


# ----------------------------------------------------------------------
# approval through the real parent graph
# ----------------------------------------------------------------------


GATED_POLICY = {
    "master_enabled": True,
    "global_tools": [],
    "client_rules": {
        "client_mcp": {"servers": {"files": True}, "tools": {}},
        "client_skill": {"servers": {}, "tools": {}},
    },
}


def _gated_tool(executed: list[str]):
    @tool
    def write_file() -> str:
        """Write the file."""
        executed.append("write_file")
        return "written"

    write_file.metadata = {
        "tool_origin": "client_mcp",
        "server_name": "files",
        "qualified_tool_id": "files::write_file",
        "mutation": True,
    }
    return write_file


class _ParentGraphWorkflow:
    """The real parent graph over one real specialist subgraph."""

    def __init__(self, factory: SpecialistFactory, policy: dict):
        from app.ai.workflow.inventory import build_routing_inventory
        from app.ai.workflow.transitions import TransitionResolver

        self.agents = dict.fromkeys(["chat_agent"])
        self._factory = factory
        self._policy = policy
        self._inventory = build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={})
        self._resolver = TransitionResolver(inventory=self._inventory, max_delegation_depth=1)

        async def _unused(state):
            raise AssertionError("no pre-v2 node runs in this test")

        self._rag_node = _unused
        self._planning_node = _unused
        self._rag_tools_node = _unused
        self._planning_tools_node = _unused
        self._should_call_tools = lambda _state: "end"
        self._should_call_rag_tools = lambda _state: "end"
        self._should_call_planning_tools = lambda _state: "end"
        self._should_continue_rag = lambda _state: "end"
        self._should_continue_planning = lambda _state: "end"

    async def invoke_specialist_subgraph(self, _node_name, state):
        return await self._factory.invoke(_request(hitl_policy=self._policy))

    def build_transition_resolver(self):
        return self._resolver


async def _run_gated_turn(executed: list[str]):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from app.ai.workflow.contracts import RoutingDecision, TurnIdentity
    from app.ai.workflow.graph_builder import build_workflow_graph
    from app.ai.workflow.inventory import build_routing_inventory
    from app.ai.workflow.runtime_context import WorkflowRuntimeContext
    from app.ai.workflow.state import build_checkpoint_thread_id

    class _Routing:
        async def route(self, _context, _inventory, **_kwargs):
            return RoutingDecision(agent_id="chat_agent", confidence=1.0, reason="scripted")

    class _ContextBuilder:
        async def build(self, request):
            return {"message": request.message}

    model = ScriptedChatModel(responses=_calls_then_answers("write_file"))
    factory = _factory(model, [_gated_tool(executed)])
    graph = build_workflow_graph(
        _ParentGraphWorkflow(factory, GATED_POLICY),
        checkpointer=InMemorySaver(),
        context_schema=WorkflowRuntimeContext,
    )
    thread_id = build_checkpoint_thread_id("conversation-1", "message-1")
    state = {
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
    }
    context = WorkflowRuntimeContext(
        routing_service=_Routing(),
        inventory=build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={}),
        routing_context_builder=_ContextBuilder(),
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    await graph.ainvoke(state, config=config, context=context)
    return graph, config, context


async def test_a_gated_specialist_tool_pauses_the_whole_turn():
    executed: list[str] = []
    graph, config, _ = await _run_gated_turn(executed)

    snapshot = await graph.aget_state(config)

    assert snapshot.next == ("chat_agent",), "the turn must pause on the specialist node"
    assert executed == []
    payload = snapshot.interrupts[0].value
    assert [request["name"] for request in payload["action_requests"]] == ["write_file"]
    assert payload["action_requests"][0]["tool_call_id"] == "call-1"
    assert payload["metadata"]["tool_provenance"]["call-1"]["server_name"] == "files"


async def test_resume_is_recognised_as_an_approval_interrupt():
    """The service refuses to resume a node it does not consider an approval stop."""
    from app.ai.graph import _has_approval_interrupt

    executed: list[str] = []
    graph, config, _ = await _run_gated_turn(executed)
    snapshot = await graph.aget_state(config)

    assert _has_approval_interrupt(snapshot.next)


@pytest.mark.parametrize(
    ("decision", "expected_runs"),
    [("approve", ["write_file"]), ("reject", [])],
)
async def test_the_human_decision_decides_whether_the_tool_runs(decision, expected_runs):
    from langgraph.types import Command

    executed: list[str] = []
    graph, config, context = await _run_gated_turn(executed)

    resume = [{"task_id": "call-1", "tool_call_id": "call-1", "type": decision, "args": None}]
    final = await graph.ainvoke(Command(resume=resume), config=config, context=context)

    assert executed == expected_runs
    assert final["execution_phase"] == "completed"
