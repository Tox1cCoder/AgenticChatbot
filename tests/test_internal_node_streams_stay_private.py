"""Only a node that answers the user may stream text to the user.

The router runs *inside* the graph now, so its model call streams like any
other — and its model returns a `RoutingDecision`. Users saw raw control-plane
JSON as the assistant's reply:

    {"agent_id":"chat_agent","confidence":1.0,"reason":"The user is providing
     a general greeting ('xin chào'), ..."}

The `internal` tag mechanism for this already existed and was already active
(`settings.suppress_internal_stream_chunks`, `_is_internal_run`). The router
just never set it. So the immediate cause was one missing annotation — and
relying on every future author to remember that annotation is what this module
refuses to do.

`planning_actions` had the same latent leak: it grades a plan with a second
model call and passes no config either, so the rubric's structured output would
interleave into a Planning answer.

Enforcement is therefore structural and fails closed: a delta attributed to a
node outside `SPECIALIST_NODE_NAMES` is not public, whether or not anyone
remembered to tag it. An *unattributed* delta stays public, because
`iter_v3_events_from_graph` also serves scripted doubles and older runnables
that emit no node — suppressing those would break legitimate output to fix a
leak that cannot come from them.
"""

from __future__ import annotations

import pytest

from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)
from app.services.event_streaming.langchain_v3 import V3ProtocolTranslator

ROUTING_JSON = '{"agent_id":"chat_agent","confidence":1.0,"reason":"a general greeting"}'


def _messages_event(
    text: str,
    *,
    node: str | None,
    namespace: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """One v3 ``messages`` envelope carrying a text delta from ``node``."""
    metadata: dict = {}
    if node is not None:
        metadata["langgraph_node"] = node
    if tags is not None:
        metadata["tags"] = tags
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": list(namespace or []),
            "data": (
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": text},
                },
                metadata,
            ),
        },
    }


def _public_text(events) -> str:
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=True,
    )
    ctx = StreamProjectionContext()
    translator = V3ProtocolTranslator()
    text = []
    for raw in events:
        for canonical in translator.translate(raw):
            for public in projector.map_event(canonical, ctx):
                if public.type in {"message_delta", "content_delta"}:
                    text.append((public.data or {}).get("text") or "")
    return "".join(text)


# ----------------------------------------------------------------------
# the reported symptom
# ----------------------------------------------------------------------


def test_the_routers_decision_never_reaches_the_user():
    """The exact string a user saw, from the node that produced it."""
    assert _public_text([_messages_event(ROUTING_JSON, node="route")]) == ""


def test_the_router_is_suppressed_even_when_nobody_tagged_it_internal():
    """Structural, not discipline-based. The tag is absent here on purpose."""
    event = _messages_event(ROUTING_JSON, node="route")
    assert "tags" not in event["params"]["data"][1]
    assert _public_text([event]) == ""


def test_the_plan_grader_output_never_reaches_the_user():
    """`planning_actions` grades a plan with a second model call."""
    rubric = '{"status":"needs_revision","evaluations":[{"criterion":"scope"}]}'
    assert _public_text([_messages_event(rubric, node="planning_actions")]) == ""


@pytest.mark.parametrize(
    "node",
    ["planning_dispatch", "planning_collect", "planning_package", "validate_output", "finalize"],
)
def test_no_other_control_plane_node_can_publish_text(node):
    assert _public_text([_messages_event("internal chatter", node=node)]) == ""


# ----------------------------------------------------------------------
# and the answer still gets through
# ----------------------------------------------------------------------


def test_every_specialist_node_still_streams_its_answer():
    """The failure mode of a too-narrow allowlist is a silent empty reply.

    Assert the whole allowlist, so adding a specialist without adding it here
    shows up as a failing test rather than as a user getting nothing.
    """
    from app.ai.workflow.graph_builder import SPECIALIST_NODE_NAMES

    for node in sorted(SPECIALIST_NODE_NAMES):
        assert _public_text([_messages_event("hello", node=node)]) == "hello", (
            f"{node} is an answer-producing node and must reach the user"
        )


def test_nested_specialist_model_output_streams_under_its_public_namespace():
    event = _messages_event(
        "hello",
        node="model",
        namespace=["search_agent:run-123"],
    )

    assert _public_text([event]) == "hello"


@pytest.mark.parametrize("namespace", [["route:run-1"], ["planning_actions:run-2"]])
def test_nested_internal_model_output_stays_private(namespace):
    assert _public_text(
        [_messages_event("internal", node="model", namespace=namespace)]
    ) == ""


async def test_real_nested_agent_streams_each_model_chunk():
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langgraph.graph import END, START, MessagesState, StateGraph

    from app.services.event_streaming.langchain_v3 import iter_v3_events_from_graph

    class StreamingModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "streaming-test"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ABC"))])

        async def _agenerate(self, messages, **kwargs):
            return self._generate(messages)

        async def _astream(self, messages, **kwargs):
            for text in "ABC":
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    agent = create_agent(model=StreamingModel(), tools=[])

    async def search_agent(state, config):
        return await agent.ainvoke({"messages": state["messages"]}, config=config)

    builder = StateGraph(MessagesState)
    builder.add_node("search_agent", search_agent)
    builder.add_edge(START, "search_agent")
    builder.add_edge("search_agent", END)
    graph = builder.compile()
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=True,
    )
    context = StreamProjectionContext()
    chunks = []

    async for event in iter_v3_events_from_graph(
        graph, {"messages": [HumanMessage(content="go")]}, config=None
    ):
        for public in projector.map_event(event, context):
            if public.type == "message_delta":
                chunks.append(public.data["text"])

    assert chunks == ["A", "B", "C"]


def test_an_unattributed_delta_is_still_published():
    """Scripted doubles and older runnables emit no node; they must still work."""
    assert _public_text([_messages_event("hello", node=None)]) == "hello"


def test_the_allowlist_is_the_graphs_own_node_set_not_a_copy():
    """A hand-copied list would drift from the graph and silence a specialist."""
    from app.ai.workflow.graph_builder import (
        PLANNING_ENTRY_NODE,
        SPECIALIST_NODE_NAMES,
        SUBGRAPH_SPECIALIST_NODES,
    )
    from app.services.event_streaming.langchain_v3 import PUBLIC_ANSWER_NODES

    assert frozenset(SPECIALIST_NODE_NAMES) == PUBLIC_ANSWER_NODES
    assert set(SUBGRAPH_SPECIALIST_NODES) <= PUBLIC_ANSWER_NODES
    assert PLANNING_ENTRY_NODE in PUBLIC_ANSWER_NODES
    assert "route" not in PUBLIC_ANSWER_NODES


# ----------------------------------------------------------------------
# the annotation at the source, kept as well
# ----------------------------------------------------------------------


async def test_the_route_node_marks_its_model_run_internal():
    """Correct at the source too: the run is internal in traces, not just filtered."""
    from types import SimpleNamespace

    from app.ai.workflow.contracts import RoutingDecision
    from app.ai.workflow.graph_builder import make_route_node
    from app.ai.workflow.inventory import build_routing_inventory

    seen: dict = {}

    class _Router:
        context_builder = None

        async def route(self, context, inventory, **kwargs):
            seen.update(kwargs)
            return RoutingDecision(agent_id="chat_agent", confidence=1.0, reason="greeting")

    inventory = build_routing_inventory(base_agent_ids=["chat_agent"], custom_agents={})

    async def _build_routing_context(state, inv):
        return SimpleNamespace(message="xin chao", inventory_version=inv.version)

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            routing_service=_Router(),
            inventory=inventory,
            build_routing_context=_build_routing_context,
        )
    )

    await make_route_node()({"turn_identity": None, "messages": []}, runtime=runtime)

    run_config = seen.get("run_config") or {}
    assert "internal" in (run_config.get("tags") or []), (
        f"the routing model run must be tagged internal; got {run_config!r}"
    )


# ----------------------------------------------------------------------
# the tuple fallback path, which the v3 guard does not cover
# ----------------------------------------------------------------------


def _legacy_chunk_event(
    text: str,
    *,
    node: str | None,
    checkpoint_namespace: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """A ``state_snapshot`` carrier as ``_iter_tuple_fallback`` builds it."""
    from types import SimpleNamespace

    metadata: dict = {}
    if node is not None:
        metadata["langgraph_node"] = node
    if tags is not None:
        metadata["tags"] = tags
    if checkpoint_namespace is not None:
        metadata["langgraph_checkpoint_ns"] = checkpoint_namespace
    return {
        "kind": "messages_tuple",
        "chunk": SimpleNamespace(content=text, content_blocks=None),
        "metadata": metadata,
    }


def _public_text_from_legacy(payloads) -> str:
    from app.services.event_streaming.events import make_event

    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=True,
    )
    ctx = StreamProjectionContext()
    text = []
    for index, payload in enumerate(payloads, start=1):
        carrier = make_event("state_snapshot", sequence=index, data=payload)
        for public in projector.map_event(carrier, ctx):
            if public.type in {"message_delta", "content_delta"}:
                text.append((public.data or {}).get("text") or "")
    return "".join(text)


def test_the_router_is_also_silent_on_the_tuple_fallback_path():
    """The v3 guard lives in the translator, which the fallback does not use.

    Production speaks v3, so this path serves doubles and older runnables --
    which is exactly where a leak would hide from a v3-only test. The tag alone
    would cover the router here; it would not cover the untagged rubric grader.
    """
    assert _public_text_from_legacy([_legacy_chunk_event(ROUTING_JSON, node="route")]) == ""


def test_the_plan_grader_is_also_silent_on_the_tuple_fallback_path():
    rubric = '{"status":"needs_revision","evaluations":[]}'
    assert _public_text_from_legacy([_legacy_chunk_event(rubric, node="planning_actions")]) == ""


def test_a_specialist_still_streams_on_the_tuple_fallback_path():
    assert _public_text_from_legacy([_legacy_chunk_event("hello", node="chat_agent")]) == "hello"


def test_a_nested_specialist_still_streams_on_the_tuple_fallback_path():
    event = _legacy_chunk_event(
        "hello",
        node="model",
        checkpoint_namespace="search_agent:run-123|model:run-456",
    )

    assert _public_text_from_legacy([event]) == "hello"


@pytest.mark.parametrize("suppress_internal", [True, False])
def test_tuple_fallback_keeps_planning_worker_text_attributed(suppress_internal):
    from app.services.event_streaming.events import make_event

    payload = _legacy_chunk_event(
        "worker result",
        node="model",
        checkpoint_namespace="planning_model:run-1|model:run-2",
        tags=["internal", "planning_subagent"],
    )
    payload["metadata"].update(
        {
            "purpose": "planning_subagent",
            "subagent_dispatch_id": "dispatch-1",
            "subagent_task_id": "task-1",
            "subagent_agent": "search_agent",
        }
    )
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=suppress_internal,
    )

    events = list(
        projector.map_event(
            make_event("state_snapshot", sequence=1, data=payload),
            StreamProjectionContext(),
        )
    )

    assert [event.type for event in events] == ["subagent_message_delta"]
    assert events[0].subagent.id == "dispatch-1:task-1"
    assert events[0].data == {"text": "worker result", "channel": "text"}


def test_tuple_fallback_normalizes_cumulative_planning_worker_chunks():
    from app.services.event_streaming.events import make_event

    metadata = {
        "langgraph_node": "model",
        "langgraph_checkpoint_ns": "planning_model:run-1|model:run-2",
        "purpose": "planning_subagent",
        "subagent_dispatch_id": "dispatch-1",
        "subagent_task_id": "task-1",
        "subagent_agent": "search_agent",
    }
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=True,
    )
    context = StreamProjectionContext()
    deltas = []

    for sequence, text in enumerate(("A", "AB", "ABC"), start=1):
        payload = _legacy_chunk_event(text, node="model")
        payload["metadata"] = metadata
        for event in projector.map_event(
            make_event("state_snapshot", sequence=sequence, data=payload), context
        ):
            if event.type == "subagent_message_delta":
                deltas.append(event.data["text"])

    assert deltas == ["A", "B", "C"]


def test_tuple_fallback_does_not_emit_an_empty_reasoning_delta():
    from types import SimpleNamespace

    from app.services.event_streaming.events import make_event

    payload = {
        "kind": "messages_tuple",
        "chunk": SimpleNamespace(
            content="",
            content_blocks=[{"type": "reasoning", "reasoning": "", "text": ""}],
        ),
        "metadata": {"langgraph_node": "chat_agent"},
    }
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda *_args, **_kwargs: [],
        suppress_internal_stream_chunks=True,
    )

    events = list(
        projector.map_event(
            make_event("state_snapshot", sequence=1, data=payload),
            StreamProjectionContext(),
        )
    )

    assert events == []


def test_an_unattributed_chunk_still_streams_on_the_tuple_fallback_path():
    assert _public_text_from_legacy([_legacy_chunk_event("hello", node=None)]) == "hello"
