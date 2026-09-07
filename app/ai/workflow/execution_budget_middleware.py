"""Adapter that lets a ``create_agent`` specialist share the budget accountant.

Thin on purpose. The counting and the decisions belong to
:mod:`app.ai.workflow.execution_budget`, because RAG and Planning do not run
through this middleware at all and must reach the same answers.

What this adds is the framework-shaped half of the contract:

* refusing a tool call **and pairing a ``ToolMessage`` with it**, because a
  provider rejects a transcript containing an unanswered tool call — refusing
  without answering would convert a budget stop into a provider error one call
  later;
* translating the framework's own limit exception into a budget outcome, so a
  ladder that failed to reserve the answer is recorded as ``hard_limit`` rather
  than surfacing as a generic execution error.

What it deliberately does **not** do is strip ``request.tools``.
``ToolExecutionMiddleware.awrap_model_call`` is the innermost model-call
wrapper in ``build_specialist_middleware`` and re-offers the live tool factory's
tools unconditionally, so any override here is overwritten before the model
sees it. Suppression happens one level up, at the request: ``disable_tools`` in
``request.extras``, which every agent's ``tool_factory`` already honours — and
because it takes effect inside the factory, it survives provider retries and
fallbacks too. This middleware only reports that synthesis is forced; the graph
sets that flag on the next request.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from app.ai.workflow.execution_budget import ExecutionBudgetAccountant

logger = logging.getLogger(__name__)

__all__ = ["SoftExecutionBudgetMiddleware"]

#: Returned in place of the tool result for a call the budget refused. Phrased
#: as a state of the world rather than a failure: an error-shaped answer invites
#: the model to retry the same call.
_REFUSAL_TEXT = (
    "Evidence gathering has ended for this execution epoch, so this tool was "
    "not run. Answer now from the evidence already gathered, and say plainly "
    "what remains unknown."
)


class SoftExecutionBudgetMiddleware(AgentMiddleware):
    """Report budget decisions for one specialist invocation."""

    def __init__(self, *, accountant: ExecutionBudgetAccountant) -> None:
        super().__init__()
        self._accountant = accountant

    @property
    def accountant(self) -> ExecutionBudgetAccountant:
        return self._accountant

    def snapshot(self) -> dict[str, Any]:
        """The budget as the specialist wrapper stores it on workflow state."""
        return self._accountant.state.model_dump(mode="json")

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        decision = self._accountant.note_tool_call()
        if decision.allowed:
            return await handler(request)

        call = getattr(request, "tool_call", None) or {}
        logger.info(
            "Refused tool call '%s' after %s tool call(s) in epoch %s",
            call.get("name"),
            self._accountant.state.tool_calls,
            self._accountant.state.execution_epoch,
        )
        return ToolMessage(
            content=_REFUSAL_TEXT,
            tool_call_id=str(call.get("id") or ""),
            name=str(call.get("name") or "tool"),
            status="success",
        )

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self._accountant.note_model_call()
        try:
            return await handler(request)
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            # The framework ceiling fired, which means the soft rung below it
            # did not reserve the answer. Recorded as a budget outcome so the
            # graph can build a deterministic partial instead of reporting
            # ``agent_execution_limit`` to a client.
            self._accountant.note_hard_limit()
            raise
