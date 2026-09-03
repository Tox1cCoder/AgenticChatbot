"""A Planning node factory for tests that only need the graph's shape.

``build_workflow_graph`` registers Planning's six nodes from
``workflow.planning_node_factory.descriptors()``, so any test that compiles the
parent graph has to supply one. Most of those tests never route into Planning —
they are checking topology, or scripting a different specialist — so the
collaborators here raise rather than returning plausible values: a stub that
quietly answered would let a test pass while exercising nothing.

Tests that *do* drive Planning build their own factory with real scripted
collaborators; see ``tests/test_planning_worker_fanout.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.planning_execution import PlanningLimits, PlanningNodeFactory

DEFAULT_BASE_AGENT_IDS = (
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
)


def stub_planning_limits() -> PlanningLimits:
    """The production defaults, stated explicitly so a test can see them."""
    return PlanningLimits(
        max_tasks=8,
        max_concurrency=4,
        max_dispatch_waves=2,
        objective_max_chars=4000,
        parent_context_max_chars=12000,
    )


def stub_planning_node_factory(
    *, base_agent_ids: tuple[str, ...] = DEFAULT_BASE_AGENT_IDS
) -> PlanningNodeFactory:
    """A factory whose nodes exist but whose collaborators refuse to run."""

    async def _never_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "this test supplied a stub Planning factory but routed into Planning; "
            "build a real factory with scripted collaborators instead"
        )

    return PlanningNodeFactory(
        call_model=_never_called,
        worker_runtime=SimpleNamespace(run=_never_called),
        limits=stub_planning_limits(),
        inventory_for=lambda state: build_routing_inventory(
            base_agent_ids=list(base_agent_ids), custom_agents={}
        ),
        resolve_allowed_tools=lambda agent_id, state: (),
        apply_todo_actions=_never_called,
    )


def scripted_planning_node_factory(call_model, **overrides: Any) -> PlanningNodeFactory:
    """A factory that drives Planning's real nodes from a scripted model.

    For tests that route *into* Planning and want the production node
    behaviour: validation, fan-out, collection, and packaging all run, and only
    the model turn is scripted. The worker runtime and todo applier still
    refuse unless a test supplies them, so a test that dispatches has to say so.
    """

    async def _never_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "this scripted Planning factory has no worker runtime or todo applier; "
            "pass one if the test dispatches subagents or writes todos"
        )

    payload: dict[str, Any] = {
        "call_model": call_model,
        "worker_runtime": SimpleNamespace(run=_never_called),
        "limits": stub_planning_limits(),
        "inventory_for": lambda state: build_routing_inventory(
            base_agent_ids=list(DEFAULT_BASE_AGENT_IDS), custom_agents={}
        ),
        "resolve_allowed_tools": lambda agent_id, state: (),
        "apply_todo_actions": _never_called,
    }
    payload.update(overrides)
    return PlanningNodeFactory(**payload)
