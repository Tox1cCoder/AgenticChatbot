"""The middleware is an adapter, and its placement is the contract.

Two things here are easy to get wrong and expensive to get wrong:

* **Tool suppression cannot live in this middleware.**
  ``ToolExecutionMiddleware.awrap_model_call`` is the innermost model-call
  wrapper in ``build_specialist_middleware`` and unconditionally re-offers the
  live factory's tools, so a ``tools=[]`` set here is overwritten. The
  repository already suppresses at the request boundary — ``disable_tools`` in
  ``request.extras``, which every agent's ``tool_factory`` honours — and
  because that happens *inside* the factory it also survives provider retries
  and fallbacks. This middleware therefore only reports that synthesis is
  forced; it does not try to win an override race it would lose.

* **The refused tool call must still be answered.** Providers reject a
  transcript with an unanswered tool call, so refusing one without pairing a
  ``ToolMessage`` turns a budget stop into a provider error on the very next
  call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.ai.workflow.execution_budget import (
    ExecutionBudgetAccountant,
    ExecutionBudgetLimits,
)
from app.ai.workflow.execution_budget_middleware import SoftExecutionBudgetMiddleware


def _accountant(**overrides) -> ExecutionBudgetAccountant:
    values = {
        "soft_model_calls": 3,
        "hard_model_calls": 5,
        "soft_tool_calls": 2,
        "hard_tool_calls": 4,
        "total_epochs_per_turn": 3,
    }
    values.update(overrides)
    return ExecutionBudgetAccountant(limits=ExecutionBudgetLimits(**values))


def _middleware(accountant: ExecutionBudgetAccountant) -> SoftExecutionBudgetMiddleware:
    return SoftExecutionBudgetMiddleware(accountant=accountant)


def _tool_request(name: str = "web_search", call_id: str = "call-1"):
    return SimpleNamespace(
        tool_call={"id": call_id, "name": name, "args": {}},
        tool=SimpleNamespace(name=name, metadata={}),
        state={},
        runtime=SimpleNamespace(context=None, config=None),
    )


def _model_request():
    return SimpleNamespace(messages=[], tools=[SimpleNamespace(name="web_search")])


async def _ran(_request):
    return ToolMessage(content="ran", tool_call_id="call-1", name="web_search")


async def _answered(_request):
    return AIMessage(content="answer")


# ----------------------------------------------------------------------
# tool calls
# ----------------------------------------------------------------------


async def test_a_tool_call_within_budget_reaches_the_handler():
    accountant = _accountant()
    middleware = _middleware(accountant)
    calls: list = []

    async def handler(request):
        calls.append(request)
        return await _ran(request)

    message = await middleware.awrap_tool_call(_tool_request(), handler)

    assert len(calls) == 1
    assert message.content == "ran"
    assert accountant.state.tool_calls == 1


async def test_the_refused_call_never_reaches_the_handler():
    accountant = _accountant(soft_tool_calls=1)
    middleware = _middleware(accountant)
    await middleware.awrap_tool_call(_tool_request(), _ran)
    reached = False

    async def handler(request):
        nonlocal reached
        reached = True
        return await _ran(request)

    await middleware.awrap_tool_call(_tool_request(call_id="call-2"), handler)

    assert reached is False


async def test_the_refused_call_is_answered_with_its_own_tool_message():
    """An unanswered tool call is a provider error on the next request."""
    accountant = _accountant(soft_tool_calls=1)
    middleware = _middleware(accountant)
    await middleware.awrap_tool_call(_tool_request(), _ran)

    message = await middleware.awrap_tool_call(_tool_request(call_id="call-2"), _ran)

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-2"
    assert message.name == "web_search"


async def test_the_refusal_tells_the_model_what_to_do_next():
    accountant = _accountant(soft_tool_calls=1)
    middleware = _middleware(accountant)
    await middleware.awrap_tool_call(_tool_request(), _ran)

    message = await middleware.awrap_tool_call(_tool_request(call_id="call-2"), _ran)

    assert "evidence" in str(message.content).lower()


async def test_the_refusal_is_not_reported_as_a_tool_error():
    """It is a budget decision. An error status invites a retry."""
    accountant = _accountant(soft_tool_calls=1)
    middleware = _middleware(accountant)
    await middleware.awrap_tool_call(_tool_request(), _ran)

    message = await middleware.awrap_tool_call(_tool_request(call_id="call-2"), _ran)

    assert message.status != "error"


async def test_a_refusal_records_the_reason_on_the_budget():
    accountant = _accountant(soft_tool_calls=1)
    middleware = _middleware(accountant)
    await middleware.awrap_tool_call(_tool_request(), _ran)

    await middleware.awrap_tool_call(_tool_request(call_id="call-2"), _ran)

    assert accountant.state.exhausted_by == "tool_calls"
    assert accountant.state.forced_synthesis is True


# ----------------------------------------------------------------------
# model calls
# ----------------------------------------------------------------------


async def test_a_model_call_within_budget_passes_through_untouched():
    accountant = _accountant()
    middleware = _middleware(accountant)
    seen: list = []

    async def handler(request):
        seen.append(request)
        return await _answered(request)

    await middleware.awrap_model_call(_model_request(), handler)

    assert len(seen) == 1
    assert accountant.state.model_calls == 1


async def test_the_middleware_does_not_try_to_strip_the_tool_list():
    """R3. It would lose: ToolExecutionMiddleware re-offers tools after it.

    Suppression belongs to ``disable_tools`` on the request, which the tool
    factory reads — so this asserts the middleware leaves ``tools`` alone
    rather than pretending to control it.
    """
    accountant = _accountant(soft_model_calls=1)
    middleware = _middleware(accountant)
    request = _model_request()
    observed: list = []

    async def handler(inner):
        observed.append(list(getattr(inner, "tools", []) or []))
        return await _answered(inner)

    await middleware.awrap_model_call(request, handler)

    assert observed == [list(request.tools)]


async def test_reaching_the_soft_model_limit_marks_forced_synthesis():
    accountant = _accountant(soft_model_calls=2)
    middleware = _middleware(accountant)

    await middleware.awrap_model_call(_model_request(), _answered)
    await middleware.awrap_model_call(_model_request(), _answered)

    assert accountant.state.forced_synthesis is True
    assert accountant.state.exhausted_by == "model_calls"


async def test_the_budget_snapshot_is_readable_after_the_run():
    """The specialist wrapper copies this onto the workflow state."""
    accountant = _accountant()
    middleware = _middleware(accountant)
    await middleware.awrap_model_call(_model_request(), _answered)

    snapshot = middleware.snapshot()

    assert snapshot["model_calls"] == 1
    assert snapshot["exhausted_by"] is None


# ----------------------------------------------------------------------
# hard limit
# ----------------------------------------------------------------------


async def test_the_frameworks_own_limit_becomes_a_budget_outcome():
    """Its typed exception must not reach the client as an execution error."""
    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

    accountant = _accountant()
    middleware = _middleware(accountant)

    async def handler(_request):
        raise ModelCallLimitExceededError(
            thread_count=9, run_count=9, thread_limit=9, run_limit=None
        )

    with pytest.raises(ModelCallLimitExceededError):
        await middleware.awrap_model_call(_model_request(), handler)

    assert accountant.state.exhausted_by == "hard_limit"
    assert accountant.state.forced_synthesis is True


async def test_an_ordinary_provider_failure_is_not_a_budget_outcome():
    """Only the limit exception means the ladder failed."""
    accountant = _accountant()
    middleware = _middleware(accountant)

    async def handler(_request):
        raise RuntimeError("provider is down")

    with pytest.raises(RuntimeError):
        await middleware.awrap_model_call(_model_request(), handler)

    assert accountant.state.exhausted_by is None
