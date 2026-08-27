"""Whole turns through the real routing-v2 graph, start to finish.

Every other suite pins one component. This one compiles the actual parent
graph against a real checkpointer and runs turns through it, because the
invariants that matter most are the ones no single component owns: that a turn
routes once, that only the finalizer reaches END, that a handoff moves the
active agent without touching the initial decision, that a failed turn
publishes nothing, and that two turns never see each other's state.

Only the model and the specialist loops are faked. Routing, transitions,
validation, finalization, persistence of graph state, and the topology itself
are the real implementations.

Scope note, so a green run is not read as more than it is: HITL resume across
a fresh process, planning worker fan-out, and streamed/non-streamed parity are
NOT covered here — they need the Task 7/8 cutover first, and are listed in the
plan's open items rather than silently omitted.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    HandoffOutcome,
    OutcomeProvenance,
    PendingTransition,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkflowError,
    WorkflowRoutingException,
)
from app.ai.workflow.graph_builder import build_workflow_graph
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.runtime_context import WorkflowRuntimeContext
from app.ai.workflow.state import build_checkpoint_thread_id

BASE_AGENT_IDS = [
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
]

SUBGRAPH_AGENT_IDS = [
    "chat_agent",
    "search_agent",
    "image_generator_agent",
    "canvas_agent",
]


# ----------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------


def _response(agent_id: str, content: str) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=agent_id,
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


def _answer(agent_id: str, content: str = "answer", **provenance) -> ResponseOutcome:
    return ResponseOutcome(
        agent_id=agent_id,
        response=_response(agent_id, content),
        provenance=OutcomeProvenance(**provenance),
    )


def _handoff(from_agent: str, to_agent: str, call_id: str = "call-1") -> HandoffOutcome:
    return HandoffOutcome(
        agent_id=from_agent,
        handoff=AgentTransition(
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            source="handoff",
            tool_call_id=call_id,
        ),
    )


class ScriptedRoutingService:
    """Returns one decision per turn, or raises a typed routing failure."""

    def __init__(self, agent_id: str | None = None, *, error: WorkflowError | None = None):
        self._agent_id = agent_id
        self._error = error
        self.calls = 0

    async def route(self, _context, _inventory, **_kwargs) -> RoutingDecision:
        self.calls += 1
        if self._error is not None:
            raise WorkflowRoutingException(self._error)
        return RoutingDecision(
            agent_id=self._agent_id, confidence=0.9, reason="scripted for the test"
        )


class ScriptedContextBuilder:
    async def build(self, request):
        return {"message": request.message}


class ScriptedWorkflow:
    """Real graph, scripted specialists.

    ``outcomes`` maps an agent id to the outcomes it produces in order, so a
    turn can hand off once and answer on the second visit.
    """

    def __init__(self, outcomes: dict[str, list], *, on_invoke=None):
        self.agents = dict.fromkeys(BASE_AGENT_IDS)
        self._outcomes = {key: list(value) for key, value in outcomes.items()}
        self.visited: list[str] = []
        self.states_seen: list[dict] = []
        self._on_invoke = on_invoke

        async def _legacy_node(state):
            # A pre-v2 node hands back a state dict carrying ``response``; the
            # parent wrapper is what turns that into a server-owned outcome.
            outcome = self._next(state.get("active_agent_id"), state)
            return {**state, "response": outcome.response}

        self._rag_node = _legacy_node
        self._planning_node = _legacy_node
        self._rag_tools_node = _legacy_node
        self._planning_tools_node = _legacy_node
        self._should_call_tools = lambda _state: "end"
        self._should_call_rag_tools = lambda _state: "end"
        self._should_call_planning_tools = lambda _state: "end"
        self._should_continue_rag = lambda _state: "end"
        self._should_continue_planning = lambda _state: "end"

    def _next(self, agent_id, state):
        self.visited.append(str(agent_id))
        self.states_seen.append(dict(state))
        if self._on_invoke is not None:
            self._on_invoke(agent_id, state)
        queue = self._outcomes.get(str(agent_id))
        if not queue:
            return _answer(str(agent_id))
        return queue.pop(0)

    async def invoke_specialist_subgraph(self, _node_name, state):
        return self._next(state.get("active_agent_id"), state)

    def build_transition_resolver(self):
        from app.ai.workflow.transitions import TransitionResolver

        return TransitionResolver(
            inventory=build_routing_inventory(
                base_agent_ids=BASE_AGENT_IDS, custom_agents={}
            ),
            max_delegation_depth=2,
        )


def _initial_state(
    *,
    conversation_id: str = "conversation-1",
    turn_id: str = "message-1",
    user_id: str = "user-1",
    device_id: str = "device-1",
) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "turn_identity": TurnIdentity(
            request_id=f"request-{turn_id}",
            turn_id=turn_id,
            checkpoint_thread_id=build_checkpoint_thread_id(conversation_id, turn_id),
        ),
        "messages": [HumanMessage(content="hello")],
        "assistant_message_id": f"assistant-{turn_id}",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "device_id": device_id,
        "agent_history": [],
        "execution_phase": "routing",
    }


async def _run_turn(
    workflow: ScriptedWorkflow,
    routing_service: ScriptedRoutingService,
    *,
    checkpointer=None,
    state: dict | None = None,
    custom_agents: dict | None = None,
):
    """Compile the real graph and run one whole turn through it."""
    saver = checkpointer or InMemorySaver()
    graph = build_workflow_graph(
        workflow, checkpointer=saver, context_schema=WorkflowRuntimeContext
    )
    turn_state = state or _initial_state()
    context = WorkflowRuntimeContext(
        routing_service=routing_service,
        inventory=build_routing_inventory(
            base_agent_ids=BASE_AGENT_IDS, custom_agents=custom_agents or {}
        ),
        routing_context_builder=ScriptedContextBuilder(),
    )
    config = {
        "configurable": {"thread_id": turn_state["turn_identity"].checkpoint_thread_id},
        "recursion_limit": 30,
    }
    final = await graph.ainvoke(turn_state, config=config, context=context)
    return final, saver


# ----------------------------------------------------------------------
# every specialist completes a whole turn
# ----------------------------------------------------------------------


@pytest.mark.parametrize("agent_id", BASE_AGENT_IDS)
async def test_a_turn_completes_through_every_base_specialist(agent_id):
    workflow = ScriptedWorkflow({agent_id: [_answer(agent_id, "the answer")]})
    routing = ScriptedRoutingService(agent_id)

    final, _ = await _run_turn(workflow, routing)

    assert final["execution_phase"] == "completed"
    assert final["final_agent_id"] == agent_id
    assert final["response"].message.content == "the answer"


async def test_a_turn_completes_through_a_dynamic_custom_specialist():
    custom_id = "custom_agent:writer"
    custom_agents = {
        custom_id: {"name": "Writer", "description": "writes", "enabled": True}
    }
    workflow = ScriptedWorkflow({custom_id: [_answer(custom_id, "custom answer")]})
    routing = ScriptedRoutingService(custom_id)

    final, _ = await _run_turn(workflow, routing, custom_agents=custom_agents)

    assert final["execution_phase"] == "completed"
    assert final["final_agent_id"] == custom_id


async def test_the_router_runs_exactly_once_per_turn():
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent")]})
    routing = ScriptedRoutingService("chat_agent")

    await _run_turn(workflow, routing)

    assert routing.calls == 1


# ----------------------------------------------------------------------
# handoffs
# ----------------------------------------------------------------------


def _pending(to_agent: str, *, from_agent: str = "chat_agent", call_id: str = "call-1"):
    """The state the real hand_off tool leaves behind.

    The tool returns ``Command(graph=PARENT, goto="resolve_transition")`` with
    this update, so it reaches the resolver without passing through the
    specialist wrapper at all. Seeding it is what makes these end-to-end rather
    than a re-test of the tool, which ``test_agent_transitions`` already owns.
    """
    return PendingTransition(
        from_agent_id=from_agent,
        to_agent_id=to_agent,
        tool_call_id=call_id,
        tool_message_id=f"handoff:{call_id}",
        reason="better suited",
    )


async def _run_handoff(workflow, *, pending, active="chat_agent", history=None):
    """Route to ``active``, which hands off, and let the resolver decide.

    Production reaches the resolver from the hand_off tool's parent Command;
    this reaches it through the wrapper's HandoffOutcome branch. Both arrive
    with ``pending_transition`` already in state, which is the resolver's
    actual contract, so what is under test here is identical either way.
    """
    state = _initial_state()
    state["pending_transition"] = pending
    if history is not None:
        state["agent_history"] = history

    workflow._outcomes.setdefault(
        active, [_handoff(active, pending.to_agent_id, pending.tool_call_id)]
    )
    final, _ = await _run_turn(workflow, ScriptedRoutingService(active), state=state)
    return final


async def test_an_accepted_handoff_moves_the_active_agent_and_keeps_the_decision():
    workflow = ScriptedWorkflow({"search_agent": [_answer("search_agent", "searched")]})

    final = await _run_handoff(workflow, pending=_pending("search_agent"))

    assert final["routing_decision"].agent_id == "chat_agent", (
        "the initial decision is immutable; a handoff records a transition"
    )
    assert final["final_agent_id"] == "search_agent"
    assert workflow.visited == ["chat_agent", "search_agent"]


async def test_an_accepted_handoff_appends_one_ordered_transition():
    workflow = ScriptedWorkflow({"search_agent": [_answer("search_agent")]})

    final = await _run_handoff(workflow, pending=_pending("search_agent"))

    assert [(t.from_agent_id, t.to_agent_id) for t in final["agent_history"]] == [
        (None, "chat_agent"),
        ("chat_agent", "search_agent"),
    ]
    assert final["agent_history"][0].source == "router"
    assert final["agent_history"][1].source == "handoff"


async def test_a_self_handoff_is_refused():
    """Delegating to yourself is a control-plane error, not a retry."""
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent", "carried on")]})

    final = await _run_handoff(workflow, pending=_pending("chat_agent"))

    assert final["final_agent_id"] == "chat_agent"
    assert [t.to_agent_id for t in final["agent_history"]] == ["chat_agent"], (
        "a refused handoff must not append a transition"
    )


async def test_a_handoff_to_an_agent_that_already_answered_is_refused():
    """One agent per turn: a cycle is how a turn fails to terminate."""
    workflow = ScriptedWorkflow({"search_agent": [_answer("search_agent")]})
    history = [
        AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
        AgentTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            source="handoff",
            tool_call_id="c1",
        ),
    ]

    final = await _run_handoff(
        workflow,
        pending=_pending("chat_agent", from_agent="search_agent", call_id="c2"),
        active="search_agent",
        history=history,
    )

    assert final["final_agent_id"] == "search_agent"
    assert len([t for t in final["agent_history"] if t.source == "handoff"]) == 1


async def test_handoff_depth_is_bounded():
    """Past the configured depth the turn answers rather than delegating on."""
    workflow = ScriptedWorkflow({"canvas_agent": [_answer("canvas_agent")]})
    history = [
        AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
        AgentTransition(
            from_agent_id="chat_agent",
            to_agent_id="rag_agent",
            source="handoff",
            tool_call_id="a",
        ),
        AgentTransition(
            from_agent_id="rag_agent",
            to_agent_id="search_agent",
            source="handoff",
            tool_call_id="b",
        ),
    ]

    final = await _run_handoff(
        workflow,
        pending=_pending("canvas_agent", from_agent="search_agent", call_id="c"),
        active="search_agent",
        history=history,
    )

    accepted = [t for t in final["agent_history"] if t.source == "handoff"]
    assert len(accepted) == 2, "the depth limit did not bound the delegation chain"
    assert final["final_agent_id"] == "search_agent"


def test_the_handoff_outcome_branch_is_unreachable_in_production():
    """Recorded, not excused: nothing constructs a HandoffOutcome.

    The specialist wrapper has an ``isinstance(outcome, HandoffOutcome)``
    branch routing to the resolver, but a real handoff never gets there — the
    hand_off tool returns ``Command(graph=PARENT, goto=resolve_transition)``
    from inside the subgraph, so the wrapper is bypassed entirely. The branch
    also cannot work on its own: it updates ``agent_outcome`` but never
    ``pending_transition``, and the resolver fails closed without one. The
    tests above reach it only because they seed the pending transition.

    Delete the branch or give it the pending update — do not loosen this.
    """
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    constructors = [
        path.name
        for path in sorted((repo_root / "app").rglob("*.py"))
        if re.search(r"(?<!class )HandoffOutcome\s*\(", path.read_text(encoding="utf-8"))
    ]
    assert constructors == [], (
        f"HandoffOutcome is constructed now ({constructors}) — the wrapper branch "
        "is live, so verify it also sets pending_transition"
    )


# ----------------------------------------------------------------------
# failure paths publish nothing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "routing_timeout",
        "routing_provider_unavailable",
        "routing_invalid_output",
        "routing_target_unavailable",
    ],
)
async def test_a_router_failure_ends_the_turn_with_a_typed_error(code):
    error = WorkflowError(code=code, retriable=True, request_id="request-message-1")
    workflow = ScriptedWorkflow({})
    routing = ScriptedRoutingService(error=error)

    final, _ = await _run_turn(workflow, routing)

    assert final["execution_phase"] == "failed"
    assert final["workflow_error"].code == code
    assert workflow.visited == [], "a failed route must not execute a specialist"


async def test_a_failed_turn_publishes_no_assistant_text():
    error = WorkflowError(
        code="routing_provider_unavailable", retriable=True, request_id="request-message-1"
    )
    final, _ = await _run_turn(ScriptedWorkflow({}), ScriptedRoutingService(error=error))

    assert final["response"].message.content == ""
    assert not final.get("validated_public_content")


async def test_a_router_failure_never_substitutes_chat():
    """The one substitution this refactor exists to remove."""
    error = WorkflowError(
        code="routing_timeout", retriable=True, request_id="request-message-1"
    )
    final, _ = await _run_turn(ScriptedWorkflow({}), ScriptedRoutingService(error=error))

    assert final.get("routing_decision") is None
    assert final.get("final_agent_id") is None


async def test_an_unknown_routing_target_fails_instead_of_guessing():
    workflow = ScriptedWorkflow({})
    routing = ScriptedRoutingService("no_such_agent")

    final, _ = await _run_turn(workflow, routing)

    assert final["execution_phase"] == "failed"
    assert final["workflow_error"].code == "routing_target_unavailable"
    assert workflow.visited == []


# ----------------------------------------------------------------------
# validation and finalization are mandatory
# ----------------------------------------------------------------------


async def test_an_empty_answer_fails_validation_rather_than_publishing():
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent", "")]})
    final, _ = await _run_turn(workflow, ScriptedRoutingService("chat_agent"))

    assert final["execution_phase"] == "failed"
    assert final["workflow_error"].code == "response_validation_failed"


async def test_a_completed_turn_appends_exactly_one_assistant_message():
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent", "just one")]})
    final, _ = await _run_turn(workflow, ScriptedRoutingService("chat_agent"))

    assistant = [
        message
        for message in final["messages"]
        if getattr(message, "type", None) == "ai"
    ]
    assert len(assistant) == 1
    assert assistant[0].content == "just one"
    assert assistant[0].id == "assistant-message-1"


async def test_the_finalizer_records_all_three_agent_identities():
    """Initial, active, and final are separate facts after a handoff."""
    workflow = ScriptedWorkflow({"search_agent": [_answer("search_agent")]})

    final = await _run_handoff(workflow, pending=_pending("search_agent"))

    workflow_metadata = final["response"].metadata["workflow"]
    assert workflow_metadata["initial_agent_id"] == "chat_agent"
    assert workflow_metadata["active_agent_id"] == "search_agent"
    assert workflow_metadata["final_agent_id"] == "search_agent"


async def test_validated_content_matches_the_published_message():
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent", "exact text")]})
    final, _ = await _run_turn(workflow, ScriptedRoutingService("chat_agent"))

    assert final["validated_public_content"] == "exact text"
    assert final["response"].message.content == "exact text"


# ----------------------------------------------------------------------
# turn isolation
# ----------------------------------------------------------------------


async def test_each_turn_gets_its_own_checkpoint_thread():
    saver = InMemorySaver()
    for turn_id in ("message-1", "message-2"):
        await _run_turn(
            ScriptedWorkflow({"chat_agent": [_answer("chat_agent")]}),
            ScriptedRoutingService("chat_agent"),
            checkpointer=saver,
            state=_initial_state(turn_id=turn_id),
        )

    threads = {
        checkpoint.config["configurable"]["thread_id"]
        async for checkpoint in saver.alist(None)
    }
    assert threads == {
        "routing-v2:conversation-1:message-1",
        "routing-v2:conversation-1:message-2",
    }


async def test_a_new_turn_does_not_inherit_the_previous_turns_agent():
    """Per-turn threads exist so an append reducer cannot read an older turn."""
    saver = InMemorySaver()
    await _run_turn(
        ScriptedWorkflow({"canvas_agent": [_answer("canvas_agent")]}),
        ScriptedRoutingService("canvas_agent"),
        checkpointer=saver,
        state=_initial_state(turn_id="message-1"),
    )
    final, _ = await _run_turn(
        ScriptedWorkflow({"chat_agent": [_answer("chat_agent")]}),
        ScriptedRoutingService("chat_agent"),
        checkpointer=saver,
        state=_initial_state(turn_id="message-2"),
    )

    assert final["final_agent_id"] == "chat_agent"
    assert len(final["agent_history"]) == 1


async def test_concurrent_turns_do_not_leak_user_or_device_context():
    seen: list[tuple[str, str, str]] = []

    def record(agent_id, state):
        seen.append((str(agent_id), str(state.get("user_id")), str(state.get("device_id"))))

    async def one(turn_id, user_id, device_id, agent_id):
        workflow = ScriptedWorkflow({agent_id: [_answer(agent_id)]}, on_invoke=record)
        return await _run_turn(
            workflow,
            ScriptedRoutingService(agent_id),
            state=_initial_state(
                conversation_id=f"conversation-{turn_id}",
                turn_id=turn_id,
                user_id=user_id,
                device_id=device_id,
            ),
        )

    left, right = await asyncio.gather(
        one("message-a", "user-a", "device-a", "search_agent"),
        one("message-b", "user-b", "device-b", "canvas_agent"),
    )

    assert left[0]["final_agent_id"] == "search_agent"
    assert right[0]["final_agent_id"] == "canvas_agent"
    assert ("search_agent", "user-a", "device-a") in seen
    assert ("canvas_agent", "user-b", "device-b") in seen
    assert not [entry for entry in seen if entry[1:] == ("user-a", "device-b")]
    assert not [entry for entry in seen if entry[1:] == ("user-b", "device-a")]


async def test_a_specialist_never_sees_another_turns_evidence():
    """Evidence is turn-scoped; a second turn starts from an empty pack."""
    saver = InMemorySaver()
    first = ScriptedWorkflow(
        {
            "rag_agent": [
                _answer("rag_agent", "grounded", evidence=({"evidence_id": "E1"},))
            ]
        }
    )
    await _run_turn(
        first,
        ScriptedRoutingService("rag_agent"),
        checkpointer=saver,
        state=_initial_state(turn_id="message-1"),
    )

    second = ScriptedWorkflow({"rag_agent": [_answer("rag_agent", "second")]})
    await _run_turn(
        second,
        ScriptedRoutingService("rag_agent"),
        checkpointer=saver,
        state=_initial_state(turn_id="message-2"),
    )

    carried = second.states_seen[0].get("agent_outcome")
    assert carried is None, "the second turn started with the first turn's outcome"


# ----------------------------------------------------------------------
# grounding reaches the graph boundary
# ----------------------------------------------------------------------


async def test_an_answer_citing_unretrieved_evidence_fails_validation():
    """The graph-level backstop behind the citation rule.

    The renderer and the stream filter both drop an unresolvable marker, so one
    should never arrive here. If it does, publishing is the wrong answer.
    """
    workflow = ScriptedWorkflow(
        {
            "chat_agent": [
                _answer(
                    "chat_agent",
                    "Revenue rose [E9].",
                    evidence=({"evidence_id": "E1"},),
                )
            ]
        }
    )
    final, _ = await _run_turn(workflow, ScriptedRoutingService("chat_agent"))

    assert final["execution_phase"] == "failed"
    assert final["workflow_error"].code == "response_validation_failed"


async def test_an_answer_citing_retrieved_evidence_publishes():
    workflow = ScriptedWorkflow(
        {
            "chat_agent": [
                _answer(
                    "chat_agent",
                    "Revenue rose [E1].",
                    evidence=({"evidence_id": "E1"},),
                )
            ]
        }
    )
    final, _ = await _run_turn(workflow, ScriptedRoutingService("chat_agent"))

    assert final["execution_phase"] == "completed"
    assert "[E1]" in final["response"].message.content


async def test_a_real_rag_answer_with_citations_is_publishable():
    """The live RAG path renders ``[E1]`` into its own answer text.

    Now that a *claim* selects ``rag_grounding``, the pre-v2 wrapper has to
    carry the evidence the runtime recorded — otherwise every legitimate
    citation would be read as invented and every grounded answer would fail.
    """

    class GroundedRagWorkflow(ScriptedWorkflow):
        def __init__(self):
            super().__init__({})

            async def _rag(state):
                self._next(state.get("active_agent_id"), state)
                context = dict(state.get("context") or {})
                context["tool_artifacts"] = [
                    {
                        "tool_call_id": "search-1",
                        "rag_evidence": {"records": [{"evidence_id": "E1"}]},
                    }
                ]
                return {
                    **state,
                    "context": context,
                    "response": _response("rag_agent", "Revenue rose [E1]."),
                }

            self._rag_node = _rag

    final, _ = await _run_turn(GroundedRagWorkflow(), ScriptedRoutingService("rag_agent"))

    assert final["execution_phase"] == "completed", (
        "a grounded citation was rejected as invented — the wrapper is not "
        "carrying the evidence the runtime recorded"
    )
    assert "[E1]" in final["response"].message.content


async def test_an_artifact_the_runtime_never_recorded_fails_validation():
    response = _response("canvas_agent", "made a thing")
    response.tool_artifacts = [{"artifact_id": "forged"}]
    outcome = ResponseOutcome(
        agent_id="canvas_agent", response=response, provenance=OutcomeProvenance()
    )
    workflow = ScriptedWorkflow({"canvas_agent": [outcome]})

    final, _ = await _run_turn(workflow, ScriptedRoutingService("canvas_agent"))

    assert final["execution_phase"] == "failed"
    assert final["workflow_error"].code == "response_validation_failed"


# ----------------------------------------------------------------------
# the pending transition contract
# ----------------------------------------------------------------------


async def test_a_handoff_to_an_unknown_target_returns_to_the_originator():
    """An invalid target is refused, and the turn continues where it was."""
    workflow = ScriptedWorkflow({"chat_agent": [_answer("chat_agent", "carried on")]})

    final = await _run_handoff(workflow, pending=_pending("no_such_agent"))

    assert final["execution_phase"] == "completed"
    assert final["final_agent_id"] == "chat_agent"
    assert [t.to_agent_id for t in final["agent_history"]] == ["chat_agent"]


async def test_the_pending_transition_is_cleared_after_resolution():
    workflow = ScriptedWorkflow({"search_agent": [_answer("search_agent")]})

    final = await _run_handoff(workflow, pending=_pending("search_agent"))

    assert final.get("pending_transition") is None


def test_pending_transition_requires_its_paired_tool_message_id():
    """The marker id is what lets a rejection replace the request in place."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            tool_call_id="call-1",
            reason="because",
        )
