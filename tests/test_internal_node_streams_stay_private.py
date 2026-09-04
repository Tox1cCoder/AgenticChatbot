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


def _messages_event(text: str, *, node: str | None, tags: list[str] | None = None) -> dict:
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
            "namespace": [],
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
