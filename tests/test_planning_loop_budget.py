"""A Planning model that never stops proposing calls still ends the turn.

Planning's model node loops through the parent graph: a call it cannot run is
answered with ``unsupported_planning_tool`` and the model is asked again. Until
Planning counted its calls, nothing but LangGraph's ``recursion_limit`` bounded
that loop, and the turn ended as a ``GraphRecursionError`` -- a generic failure
that threw away everything the turn had done.

Now each Planning call is counted on the same execution budget the specialists
use, and the call that would leave the run unable to reach ``finalize`` is the
reserved, tool-free answer. The turn ends through the existing budget pause.

These compile the real parent graph; only the Planning model is scripted.
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.ai.workflow.continuation import make_continuation_pause_node
from app.ai.workflow.contracts import ResponseOutcome, RoutingDecision, TurnIdentity
from app.ai.workflow.execution_budget import ExecutionBudgetLimits
from app.ai.workflow.graph_builder import build_workflow_graph
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.runtime_context import WorkflowRuntimeContext
from app.ai.workflow.specialists import HARD_LIMIT_PARTIAL_TEXT
from app.ai.workflow.state import build_checkpoint_thread_id
from app.core.config import settings
from tests.planning_graph_support import DEFAULT_BASE_AGENT_IDS, scripted_planning_node_factory


class StubbornPlanningModel:
    """Proposes a tool Planning has no node for, on every call, forever.

    It does not even stop when tools are withheld, so the node has to cope with
    a reserved answer that still carries a tool call and no text.
    """

    def __init__(self) -> None:
        self.forced: list[bool] = []

    async def __call__(self, state):
        context = state.get("context") or {}
        self.forced.append(bool(context.get("force_final_response")))
        call_id = f"call-{len(self.forced)}"
        return SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=[{"name": "tool_search", "id": call_id, "args": {"query": "x"}}],
            ),
            metadata={},
        )


class PlanningOnlyRouter:
    async def route(self, _context, _inventory, **_kwargs) -> RoutingDecision:
        return RoutingDecision(agent_id="planning_agent", confidence=0.9, reason="scripted")


class ScriptedContextBuilder:
    async def build(self, request):
        return {"message": request.message}


class PlanningWorkflow:
    def __init__(self, model, budget_limits: ExecutionBudgetLimits | None = None):
        self.agents = dict.fromkeys(DEFAULT_BASE_AGENT_IDS)
        overrides = {} if budget_limits is None else {"budget_limits": budget_limits}
        self.planning_node_factory = scripted_planning_node_factory(model, **overrides)

    async def invoke_specialist_subgraph(self, node_name, state):  # pragma: no cover
        raise AssertionError(f"only Planning runs in this test, not {node_name}")

    def build_transition_resolver(self):
        from app.ai.workflow.transitions import TransitionResolver

        return TransitionResolver(
            inventory=build_routing_inventory(
                base_agent_ids=list(DEFAULT_BASE_AGENT_IDS), custom_agents={}
            ),
            max_delegation_depth=2,
        )


def _limits(soft_model_calls: int) -> ExecutionBudgetLimits:
    return ExecutionBudgetLimits(
        soft_model_calls=soft_model_calls,
        hard_model_calls=soft_model_calls + 1,
        soft_tool_calls=10,
        hard_tool_calls=11,
        total_epochs_per_turn=5,
    )


def _state() -> dict:
    return {
        "turn_identity": TurnIdentity(
            request_id="request-1",
            turn_id="turn-1",
            checkpoint_thread_id=build_checkpoint_thread_id("conversation-1", "turn-1"),
        ),
        "messages": [HumanMessage(content="research this in depth")],
        "assistant_message_id": "assistant-1",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "agent_history": [],
        "execution_phase": "routing",
    }


async def _run(workflow, *, recursion_limit: int, resume=None, saver=None):
    saver = saver or InMemorySaver()
    graph = build_workflow_graph(
        workflow, checkpointer=saver, context_schema=WorkflowRuntimeContext
    )
    context = WorkflowRuntimeContext(
        routing_service=PlanningOnlyRouter(),
        inventory=build_routing_inventory(
            base_agent_ids=list(DEFAULT_BASE_AGENT_IDS), custom_agents={}
        ),
        routing_context_builder=ScriptedContextBuilder(),
    )
    config = {
        "configurable": {"thread_id": build_checkpoint_thread_id("conversation-1", "turn-1")},
        "recursion_limit": recursion_limit,
    }
    payload = _state() if resume is None else Command(resume=resume)
    return await graph.ainvoke(payload, config=config, context=context), saver


def _pause(final: dict) -> dict:
    interrupts = final.get("__interrupt__") or ()
    assert interrupts, f"the turn did not pause; phase={final.get('execution_phase')!r}"
    return interrupts[0].value


async def test_a_runaway_planning_loop_pauses_instead_of_raising_a_recursion_error(
    monkeypatch,
):
    """With auto-continue on and the default budget, the step ceiling ends it."""
    monkeypatch.setattr(settings, "generation_auto_continue", True)
    model = StubbornPlanningModel()

    final, _ = await _run(PlanningWorkflow(model), recursion_limit=30)

    pause = _pause(final)
    assert pause["type"] == "execution_budget_exhausted"
    assert pause["budget"]["exhausted_by"] == "hard_limit"
    assert pause["validated_content"] == HARD_LIMIT_PARTIAL_TEXT
    # Every call before the last one was offered tools; only the last was not.
    assert model.forced[-1] is True
    assert not any(model.forced[:-1])
    assert len(model.forced) > 3
    assert final["agent_outcome"].response.metadata["planning_budget_reached"] is True
    assert final["agent_outcome"].response.metadata["planning_call_count"] == len(model.forced)


async def test_stopping_the_paused_runaway_turn_publishes_the_partial(monkeypatch):
    monkeypatch.setattr(settings, "generation_auto_continue", True)
    model = StubbornPlanningModel()
    _, saver = await _run(PlanningWorkflow(model), recursion_limit=30)

    final, _ = await _run(
        PlanningWorkflow(model),
        recursion_limit=30,
        saver=saver,
        resume={"action": "stop", "continuation_id": "c-1", "expected_epoch": 0},
    )

    assert final["execution_phase"] == "completed"
    assert final["response"].message.content == HARD_LIMIT_PARTIAL_TEXT
    assert final.get("workflow_error") is None


async def test_planning_model_calls_count_against_the_epoch_budget(monkeypatch):
    """The soft model-call rung reserves the answer, as it does for specialists."""
    monkeypatch.setattr(settings, "generation_auto_continue", False)
    model = StubbornPlanningModel()

    final, _ = await _run(PlanningWorkflow(model, _limits(3)), recursion_limit=100)

    pause = _pause(final)
    assert pause["budget"]["exhausted_by"] == "model_calls"
    assert pause["budget"]["model_calls"] == 3
    assert model.forced == [False, False, True]
    # No call is left without its result in the transcript.
    assert isinstance(final["agent_outcome"], ResponseOutcome)


async def test_budget_pause_publishes_actual_planning_call_count_and_reached_metadata(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow

    monkeypatch.setattr(settings, "generation_auto_continue", False)
    final, _ = await _run(
        PlanningWorkflow(StubbornPlanningModel(), _limits(3)), recursion_limit=100,
    )

    response = final["agent_outcome"].response
    assert response.metadata["planning_call_count"] == 3
    assert response.metadata["planning_budget_reached"] is True
    assert final["planning_call_count"] == 3
    # The same values must reach stream completion from the checkpoint.
    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, final)
    assert enriched.metadata["planning_call_count"] == 3
    assert enriched.metadata["planning_budget_reached"] is True
    # Finalization must not erase why this already-packaged partial was produced.
    final["execution_budget"]["forced_synthesis"] = False
    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, final)
    assert enriched.metadata["planning_budget_reached"] is True


async def test_continued_planning_keeps_turn_count_but_clears_previous_epoch_reached(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow

    monkeypatch.setattr(settings, "generation_auto_continue", False)
    _, saver = await _run(
        PlanningWorkflow(StubbornPlanningModel(), _limits(3)), recursion_limit=100,
    )

    async def answer(_state):
        return SimpleNamespace(message=SimpleNamespace(content="Finished the plan", tool_calls=[]))

    final, _ = await _run(
        PlanningWorkflow(answer, _limits(3)), recursion_limit=100, saver=saver,
        resume={"action": "continue", "continuation_id": "c-1", "expected_epoch": 0},
    )
    response = final["response"]
    assert response.metadata["planning_call_count"] == 4
    assert response.metadata["planning_budget_reached"] is False
    assert final["planning_call_count"] == 4
    response.metadata["planning_budget_reached"] = True  # stale prior-epoch metadata
    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, final)
    assert enriched.metadata["planning_call_count"] == 4
    assert enriched.metadata["planning_budget_reached"] is False


async def test_old_planning_checkpoint_preserves_recorded_call_count_on_resume():
    async def answer(_state):
        return SimpleNamespace(message=SimpleNamespace(content="Finished", tool_calls=[]))

    factory = PlanningWorkflow(answer, _limits(20)).planning_node_factory
    command = await factory.planning_model({
        "planning_call_count": 7,
        "execution_budget": {"model_calls": 10, "turn_model_calls": 12},
    })
    assert command.update["planning_call_count"] == 8
    assert command.update["execution_budget"]["turn_model_calls"] == 13


async def test_auto_continued_planning_epochs_still_end_at_the_step_ceiling(monkeypatch):
    """Rolling epochs in one run must not walk the turn into the recursion limit."""
    monkeypatch.setattr(settings, "generation_auto_continue", True)
    model = StubbornPlanningModel()

    final, _ = await _run(PlanningWorkflow(model, _limits(3)), recursion_limit=40)

    pause = _pause(final)
    assert pause["budget"]["exhausted_by"] == "hard_limit"
    # Each epoch ends in one reserved answer: at least one epoch rolled over by
    # itself before the step ceiling stopped the run.
    assert model.forced.count(True) >= 2
    assert model.forced[:3] == [False, False, True]
    assert final["agent_outcome"].response.metadata["planning_call_count"] == len(model.forced)
    assert final["agent_outcome"].response.metadata["planning_budget_reached"] is True


def test_the_reserved_call_reaches_the_production_model_call_tool_free():
    """The flag the node sets is the one the workflow's Planning call honours."""
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.workflow.planning_execution import _forced_synthesis_state

    state = {"context": {"planning_rubric_feedback": "keep"}}
    kwargs = MultiAgentWorkflow._final_response_kwargs(_forced_synthesis_state(state))

    assert kwargs["disable_tools"] is True
    assert "Do not call any more tools" in kwargs["tool_budget_notice"]
    assert state["context"] == {"planning_rubric_feedback": "keep"}, "checkpointed state untouched"


async def test_a_hard_limit_pause_asks_instead_of_rolling_the_epoch():
    """A run out of supersteps cannot host another epoch; a fresh run can."""
    asked: list[dict] = []

    def interrupt_fn(payload):
        asked.append(payload)
        return {"action": "stop", "continuation_id": "c-1", "expected_epoch": 0}

    node = make_continuation_pause_node(interrupt_fn=interrupt_fn, auto_continue=True)
    command = await node(
        {
            "active_agent_id": "planning_agent",
            "execution_epoch": 0,
            "execution_budget": {"exhausted_by": "hard_limit", "forced_synthesis": True},
        }
    )

    assert [payload["budget"]["exhausted_by"] for payload in asked] == ["hard_limit"]
    assert command.goto == "finalize"
