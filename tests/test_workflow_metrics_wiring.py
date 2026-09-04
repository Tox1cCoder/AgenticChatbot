"""Every counter the runbook tells an operator to watch must actually move.

`RoutingMetricsRecorder` defined a full workflow surface but only
`RoutingService` ever called it. Eight methods had no production caller at all,
so `grounding.abstained`, `finalization.failed.*` and `transition.rejected.*`
sat at zero no matter what happened — and a counter that never moves is
indistinguishable from one reporting good news. That is worse than having no
metric, because the runbook pointed at them during a canary.

There was no export path either: routing had no `/metrics` endpoint, while
conversation-compaction, model-usage, rich-images and RAG all had one. Wiring
the call sites without that would have left the numbers unreachable.

These tests drive the *real* call sites rather than calling the recorder
directly. A test that asserts `recorder.worker_completed(...)` increments a
counter proves nothing about whether anything invokes it.
"""

from __future__ import annotations

import pytest

from app.observability.routing import get_routing_metrics_recorder


@pytest.fixture
def recorder():
    instance = get_routing_metrics_recorder()
    instance.reset()
    yield instance
    instance.reset()


# ----------------------------------------------------------------------
# finalization: "any non-zero value is a turn that produced no answer"
# ----------------------------------------------------------------------


def _finalizer():
    from app.ai.workflow.finalization import PublicResponseFinalizer

    return PublicResponseFinalizer.__new__(PublicResponseFinalizer)


def test_a_failed_turn_increments_finalization_failed_and_terminal_error(recorder):
    from app.ai.workflow.contracts import WorkflowError

    _finalizer()._finalize_failure(
        {},
        WorkflowError(
            code="routing_provider_unavailable",
            retriable=True,
            request_id="request-1",
            details={"reason": "missing_credentials"},
        ),
    )

    assert recorder.counters["finalization.failed.routing_provider_unavailable"] == 1
    assert recorder.counters["workflow.error.routing_provider_unavailable.retriable"] == 1


def test_a_terminal_error_records_whether_retrying_helps(recorder):
    from app.ai.workflow.contracts import WorkflowError

    _finalizer()._finalize_failure(
        {},
        WorkflowError(code="response_validation_failed", retriable=False, request_id="request-1"),
    )

    assert recorder.counters["workflow.error.response_validation_failed.terminal"] == 1


# ----------------------------------------------------------------------
# transitions: "agents are ping-ponging"
# ----------------------------------------------------------------------


def _transition_state(**overrides):
    state = {
        "active_agent_id": "chat_agent",
        "agent_history": [],
        "turn_identity": None,
    }
    state.update(overrides)
    return state


async def test_an_accepted_handoff_is_counted(recorder):
    from app.ai.workflow.contracts import PendingTransition
    from app.ai.workflow.inventory import build_routing_inventory
    from app.ai.workflow.transitions import TransitionResolver

    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent", "search_agent"], custom_agents={}
    )
    resolver = TransitionResolver(inventory=inventory, max_delegation_depth=3)
    state = _transition_state(
        pending_transition=PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            reason="needs the web",
            tool_call_id="call-1",
            tool_message_id="msg-1",
        )
    )

    await resolver(state)

    assert recorder.counters["transition.accepted.chat_agent.search_agent"] == 1


async def test_a_refused_handoff_is_counted_by_reason(recorder):
    from app.ai.workflow.contracts import PendingTransition
    from app.ai.workflow.inventory import build_routing_inventory
    from app.ai.workflow.transitions import TransitionResolver

    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent", "search_agent"], custom_agents={}
    )
    resolver = TransitionResolver(inventory=inventory, max_delegation_depth=3)
    state = _transition_state(
        pending_transition=PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="missing_agent",
            reason="nowhere",
            tool_call_id="call-1",
            tool_message_id="msg-1",
        )
    )

    await resolver(state)

    rejected = [key for key in recorder.counters if key.startswith("transition.rejected.")]
    assert rejected, f"no rejection counter moved; saw {dict(recorder.counters)}"


# ----------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------


def test_a_finished_worker_is_counted_by_status_and_agent(recorder):
    from app.ai.workflow.contracts import WorkerResult
    from app.ai.workflow.planning_execution import PlanningWorkerRuntime

    PlanningWorkerRuntime._finish(
        None,
        WorkerResult(
            dispatch_id="d1",
            task_id="w1",
            position=0,
            agent_id="search_agent",
            status="completed",
            content="done",
        ),
        "Search",
    )

    assert recorder.counters["worker.completed.search_agent"] == 1


def test_a_failed_worker_is_counted_separately(recorder):
    from app.ai.workflow.contracts import WorkerResult
    from app.ai.workflow.planning_execution import PlanningWorkerRuntime

    PlanningWorkerRuntime._finish(
        None,
        WorkerResult(
            dispatch_id="d1",
            task_id="w1",
            position=0,
            agent_id="rag_agent",
            status="failed",
            error_code="agent_execution_limit",
        ),
    )

    assert recorder.counters["worker.failed.rag_agent"] == 1


def test_a_custom_worker_collapses_to_one_metric_label(recorder):
    """A per-instance custom-agent id must never become a metric label."""
    from app.ai.workflow.contracts import WorkerResult
    from app.ai.workflow.planning_execution import PlanningWorkerRuntime

    PlanningWorkerRuntime._finish(
        None,
        WorkerResult(
            dispatch_id="d1",
            task_id="w1",
            position=0,
            agent_id="custom_agent:9f2c1a7e-0000-4000-8000-000000000000",
            status="completed",
            content="done",
        ),
    )

    assert recorder.counters["worker.completed.custom_agent"] == 1
    assert not any("9f2c1a7e" in key for key in recorder.counters)


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


def test_the_recorder_renders_a_scrapeable_payload(recorder):
    from app.ai.workflow.contracts import WorkflowError

    _finalizer()._finalize_failure(
        {}, WorkflowError(code="finalization_failed", retriable=False, request_id="r")
    )
    rendered = recorder.render()

    assert isinstance(rendered, str)
    assert "finalization_failed" in rendered


def test_routing_metrics_are_reachable_over_http(recorder):
    """The other four observability surfaces have an endpoint; routing had none.

    Wiring counters nothing can scrape would have moved the problem rather
    than fixed it.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.ai.workflow.contracts import WorkflowError
    from app.api.health import create_health_router

    _finalizer()._finalize_failure(
        {}, WorkflowError(code="routing_timeout", retriable=True, request_id="r")
    )

    app = FastAPI()
    app.include_router(create_health_router())
    response = TestClient(app).get("/metrics/routing")

    assert response.status_code == 200
    assert "routing_timeout" in response.text


def test_no_metric_label_carries_an_unbounded_identifier(recorder):
    """Request, conversation, user and message IDs belong in traces, not labels."""
    from app.ai.workflow.contracts import WorkflowError

    _finalizer()._finalize_failure(
        {"active_agent_id": "chat_agent"},
        WorkflowError(
            code="routing_timeout",
            retriable=True,
            request_id="11111111-1111-1111-1111-111111111111",
        ),
    )

    assert not any("1111" in key for key in recorder.counters)
