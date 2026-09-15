"""The configured routing retry must actually get to make a provider call.

One deadline covered every attempt, so a first attempt slower than the whole
budget left the second with nothing: the router reported two attempts having
made one call, and failed ``routing_timeout`` with empty details. The retry
exists for exactly the transient failure that is most likely to be slow, so
sharing the deadline disabled it when it was needed.

The total deadline is kept as the outer bound -- a turn cannot wait forever --
and each attempt now gets its own, smaller one underneath it.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.inventory import RoutingInventory
from app.ai.workflow.routing import RoutingService


class _Settings:
    router_model = "gemini-3-flash-preview"
    router_provider = "gemini"
    routing_timeout_seconds = 1.0
    routing_attempt_timeout_seconds = 0.3
    routing_max_attempts = 2
    router_thinking_level = "low"


class _Resolver:
    def __init__(self, effort=None):
        self.effort = effort
        self.seen: list[dict] = []

    def resolve_runtime_config(self, user_id, agent_key, model_request, **kwargs):
        from app.core.runtime_modeling import ResolvedRuntimeModelConfig

        self.seen.append({"agent_key": agent_key, **kwargs})
        return ResolvedRuntimeModelConfig(
            agent_key=agent_key,
            provider="gemini",
            model="gemini-3-flash-preview",
            temperature=1.0,
            api_key="k",
            key_source="user",
            source="default",
            capabilities={"supports_structured_output": True, "supports_reasoning": True},
            reasoning_effort=self.effort,
        )


class _SlowModel:
    """Every attempt is slower than one attempt's share of the deadline."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return {"parsed": None, "parsing_error": None}


class _Factory:
    def __init__(self, model):
        self.model = model
        self.configs: list = []

    def create_model_from_runtime(self, config, **kwargs):
        self.configs.append(config)
        return self.model


def _service(settings, resolver, factory) -> RoutingService:
    return RoutingService(
        runtime_model_resolver=resolver,
        model_factory=factory,
        settings=settings,
    )


def _context():
    from app.ai.workflow.routing import RoutingContext

    return RoutingContext(message="hi", serialized_json='{"turn": "hi"}')


@pytest.mark.asyncio
async def test_a_slow_first_attempt_still_leaves_the_retry_a_provider_call():
    model = _SlowModel(delay=1.5)
    factory = _Factory(model)
    service = _service(_Settings(), _Resolver(), factory)

    with pytest.raises(WorkflowRoutingException):
        await service.route(
            _context(),
            RoutingInventory(agents=(), version="v1"),
            user_id=None,
            model_request=None,
            request_id="req-1",
        )

    assert model.calls == 2, "the retry never reached the provider"


@pytest.mark.asyncio
async def test_a_routing_timeout_reports_what_it_waited_for():
    """``details={}`` gave an operator nothing to act on."""
    model = _SlowModel(delay=1.5)
    service = _service(_Settings(), _Resolver(), _Factory(model))

    with pytest.raises(WorkflowRoutingException) as excinfo:
        await service.route(
            _context(),
            RoutingInventory(agents=(), version="v1"),
            user_id=None,
            model_request=None,
            request_id="req-2",
        )

    error = excinfo.value.error
    assert error.code == "routing_timeout"
    assert error.details.get("attempts") == 2
    assert error.details.get("attempt_timeout_seconds") == pytest.approx(0.3)
    assert error.details.get("timeout_seconds") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_the_router_runs_below_the_providers_default_thinking_level():
    """Routing is classification; the model's own default is 'high'.

    Nothing set an effort for the router, so gemini-3-flash-preview reasoned at
    its default level behind an eight-second deadline.
    """
    model = _SlowModel(delay=0.0)
    factory = _Factory(model)
    service = _service(_Settings(), _Resolver(effort=None), factory)

    with pytest.raises(WorkflowRoutingException):
        await service.route(
            _context(),
            RoutingInventory(agents=(), version="v1"),
            user_id=None,
            model_request=None,
            request_id="req-3",
        )

    assert factory.configs, "the router never built a model"
    assert factory.configs[0].reasoning_effort == "low"


@pytest.mark.asyncio
async def test_an_explicitly_configured_router_effort_is_not_overridden():
    """A level the account pinned for the router outranks the default."""
    model = _SlowModel(delay=0.0)
    factory = _Factory(model)
    service = _service(_Settings(), _Resolver(effort="high"), factory)

    with pytest.raises(WorkflowRoutingException):
        await service.route(
            _context(),
            RoutingInventory(agents=(), version="v1"),
            user_id=None,
            model_request=None,
            request_id="req-4",
        )

    assert factory.configs[0].reasoning_effort == "high"
