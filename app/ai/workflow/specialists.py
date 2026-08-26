"""Parent-level specialist wrappers for the routing-v2 graph.

A wrapper does three things and nothing else:

1. verifies the node it is running in matches ``state["active_agent_id"]``;
2. runs the specialist;
3. converts the result into an ``AgentOutcome`` and a dynamic ``Command``.

Wrappers have no static outgoing edges, so the parent graph's dynamic routing
is the only thing that decides what runs next. No wrapper appends the terminal
public ``AIMessage`` and no wrapper reaches ``END`` — ``finalize`` owns both.

The execution *internals* behind these wrappers are the pre-v2 agent loops.
Task 5 of the cutover replaces those internals with per-invocation
``create_agent`` subgraphs without changing this contract.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import Command

from app.ai.schemas import AgentResponse
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome, WorkflowError
from app.ai.workflow.inventory import CUSTOM_AGENT_NODE, CUSTOM_AGENT_PREFIX

logger = logging.getLogger(__name__)

__all__ = [
    "FINALIZE_OWNS_TERMINAL_MESSAGE",
    "make_specialist_wrapper",
    "make_tool_stage_wrapper",
    "resolve_node_for_agent_id",
]

# Turn-scoped context flag telling the pre-v2 ``_finalize_agent_response`` path
# that the parent finalizer owns the terminal public message.
FINALIZE_OWNS_TERMINAL_MESSAGE = "v2_finalize_owns_terminal_message"

SpecialistCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
StageRouter = Callable[[dict[str, Any]], Any]


def resolve_node_for_agent_id(agent_id: str | None) -> str | None:
    """Map an agent ID onto its graph node without consulting the inventory."""
    if not agent_id:
        return None
    return CUSTOM_AGENT_NODE if agent_id.startswith(CUSTOM_AGENT_PREFIX) else agent_id


def _with_finalizer_ownership(state: dict[str, Any]) -> dict[str, Any]:
    working = dict(state)
    context = dict(working.get("context") or {})
    context[FINALIZE_OWNS_TERMINAL_MESSAGE] = True
    working["context"] = context
    return working


def _response_outcome(agent_id: str, response: AgentResponse) -> ResponseOutcome:
    """Build a server-owned outcome from a specialist response.

    Provenance is assembled from runtime records only. Model text can never
    declare its own evidence, artifacts, images, or validation authority.
    """
    artifacts = tuple(
        artifact for artifact in (response.tool_artifacts or []) if isinstance(artifact, dict)
    )
    metadata = response.metadata or {}
    images = tuple(image for image in (metadata.get("images") or []) if isinstance(image, dict))
    return ResponseOutcome(
        agent_id=agent_id,
        response=response,
        provenance=OutcomeProvenance(artifacts=artifacts, images=images),
    )


def _carry_forward(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Project a legacy node's mutated state back onto the v2 update.

    Only keys the v2 state schema declares survive; everything else the legacy
    path scribbled on the dict is discarded rather than silently persisted.
    """
    update: dict[str, Any] = {}
    for key in (
        "context",
        "todos",
        "current_task_index",
        "planning_call_count",
        "planning_phase",
        "plan_lifecycle",
        "task_plan_id",
        "current_task",
        "all_tasks",
        "custom_agents",
    ):
        if key in result and result[key] is not state.get(key):
            update[key] = result[key]

    appended = _appended_messages(state.get("messages") or [], result.get("messages") or [])
    if appended:
        update["messages"] = appended
    return update


def _appended_messages(before: list[Any], after: list[Any]) -> list[Any]:
    if len(after) <= len(before):
        return []
    return list(after[len(before) :])


def make_specialist_wrapper(
    node_name: str,
    specialist: SpecialistCallable,
    *,
    stage_router: StageRouter,
    stage_targets: dict[str, str],
) -> Callable[..., Awaitable[Command]]:
    """Build the parent wrapper node for one specialist.

    ``stage_router`` is the pre-v2 decision function ("tools", "approval",
    "end", ...). ``stage_targets`` maps those decisions onto v2 node names;
    ``"end"`` always maps to ``validate_output``, never to ``END``.
    """

    async def wrapper(state: dict[str, Any], runtime: Any = None) -> Command:
        active_agent_id = state.get("active_agent_id")
        expected_node = resolve_node_for_agent_id(active_agent_id)
        if expected_node != node_name:
            # Reaching a specialist that is not the active agent means the
            # control plane and the topology disagree. Fail closed.
            logger.error(
                "Specialist node %s reached while active agent is %s",
                node_name,
                active_agent_id,
            )
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="response_validation_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "active_agent_node_mismatch", "node": node_name},
                    ),
                },
                goto="finalize",
            )

        working = _with_finalizer_ownership(state)
        result = await specialist(working)
        result = result if isinstance(result, dict) else working

        update = _carry_forward(state, result)
        decision = stage_router(result)
        if _is_awaitable(decision):
            decision = await decision

        target = stage_targets.get(str(decision), "validate_output")
        if target == "validate_output":
            response = result.get("response")
            if not isinstance(response, AgentResponse):
                return Command(
                    update={
                        **update,
                        "execution_phase": "failed",
                        "workflow_error": WorkflowError(
                            code="response_validation_failed",
                            retriable=False,
                            request_id=_request_id(state),
                            details={"reason": "specialist_produced_no_response"},
                        ),
                    },
                    goto="finalize",
                )
            update["agent_outcome"] = _response_outcome(
                active_agent_id or response.agent_id, response
            )
            update["execution_phase"] = "validating"

        return Command(update=update, goto=target)

    wrapper.__name__ = f"{node_name}_wrapper"
    return wrapper


def make_tool_stage_wrapper(
    node_name: str,
    stage: SpecialistCallable,
    *,
    stage_router: StageRouter,
) -> Callable[..., Awaitable[Command]]:
    """Wrap a pre-v2 tool/approval stage so it routes dynamically.

    These stages hand control back to a specialist or, when the turn is done,
    to ``validate_output``. They never reach ``END``.
    """

    async def wrapper(state: dict[str, Any], runtime: Any = None) -> Command:
        result = await stage(dict(state))
        result = result if isinstance(result, dict) else state
        update = _carry_forward(state, result)

        decision = stage_router(result)
        if _is_awaitable(decision):
            decision = await decision
        decision = str(decision)

        if decision == "end":
            response = result.get("response")
            if isinstance(response, AgentResponse):
                update["agent_outcome"] = _response_outcome(
                    str(result.get("active_agent_id") or response.agent_id), response
                )
                update["execution_phase"] = "validating"
                return Command(update=update, goto="validate_output")
            return Command(
                update={
                    **update,
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="tool_execution_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "stage_ended_without_response", "stage": node_name},
                    ),
                },
                goto="finalize",
            )

        if decision == "approval":
            return Command(update=update, goto="approval")
        if decision == "tools":
            return Command(update=update, goto="tools")

        target = resolve_node_for_agent_id(decision) or "validate_output"
        if decision != (result.get("active_agent_id") or decision):
            update["active_agent_id"] = decision
        return Command(update=update, goto=target)

    wrapper.__name__ = f"{node_name}_wrapper"
    return wrapper


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def _request_id(state: dict[str, Any]) -> str:
    identity = state.get("turn_identity")
    request_id = getattr(identity, "request_id", None)
    return str(request_id) if request_id else "unknown"
