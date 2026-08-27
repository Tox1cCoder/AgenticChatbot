"""Focused middleware contracts for routing-v2 specialists.

Each middleware here is small and single-purpose on purpose. What the stack
must guarantee, together, is the order: authorization decides before HITL asks,
HITL asks before the tool implementation runs, and usage is recorded once per
provider attempt whether or not that attempt succeeded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.workflow.middleware import (
    RequestBudgetMiddleware,
    RuntimeModelMiddleware,
    SpecialistToolScope,
    ToolExecutionMiddleware,
    UsageRecordingMiddleware,
    build_specialist_middleware,
)
from app.core.runtime_modeling import (
    ResolvedRuntimeModelConfig,
    RuntimeFallbackConfig,
)


def _runtime_config(provider="gemini", model="gemini-3-flash-preview", **overrides):
    payload = {
        "agent_key": "chat",
        "provider": provider,
        "model": model,
        "temperature": 1.0,
        "api_key": "key",
        "key_source": "user",
        "source": "default",
        "capabilities": {"supports_tool_calling": True},
    }
    payload.update(overrides)
    return ResolvedRuntimeModelConfig(**payload)


class FakeModel:
    def __init__(self, name="primary"):
        self.name = name

    def bind_tools(self, tools, **kwargs):
        return self


def _model_request(**overrides):
    payload = {
        "model": FakeModel(),
        "messages": [HumanMessage(content="hi")],
        "system_message": None,
        "tool_choice": None,
        "tools": [],
        "response_format": None,
        "state": {},
        "runtime": SimpleNamespace(context=None),
        "model_settings": {},
    }
    payload.update(overrides)
    return SimpleNamespace(**payload, override=lambda **kw: _model_request(**{**payload, **kw}))


# ----------------------------------------------------------------------
# runtime model resolution
# ----------------------------------------------------------------------


class FakeResolver:
    def __init__(self, resolved=None):
        self.resolved = resolved or _runtime_config()
        self.calls: list[tuple] = []

    def resolve_runtime_config(self, user_id, agent_key, request_override=None, **kwargs):
        self.calls.append((user_id, agent_key, request_override))
        return self.resolved


class FakeFactory:
    def __init__(self):
        self.built: list[str] = []

    def create_model_from_runtime(self, config, **kwargs):
        self.built.append(f"{config.provider}:{config.model}")
        return FakeModel(name=f"{config.provider}:{config.model}")


async def test_runtime_model_middleware_resolves_per_invocation():
    resolver = FakeResolver()
    factory = FakeFactory()
    middleware = RuntimeModelMiddleware(
        runtime_model_resolver=resolver,
        model_factory=factory,
        agent_key="chat",
        user_id="user-1",
        model_request={"provider": "gemini"},
    )

    seen: list[str] = []

    async def handler(request):
        seen.append(request.model.name)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(_model_request(), handler)

    assert seen == ["gemini:gemini-3-flash-preview"]
    assert resolver.calls == [("user-1", "chat", {"provider": "gemini"})]


async def test_runtime_model_middleware_falls_back_after_a_provider_error():
    resolved = _runtime_config(
        fallback_config=RuntimeFallbackConfig(
            provider="openai", model="gpt-5", temperature=1.0, api_key="k2", key_source="env"
        )
    )
    factory = FakeFactory()
    middleware = RuntimeModelMiddleware(
        runtime_model_resolver=FakeResolver(resolved),
        model_factory=factory,
        agent_key="chat",
        user_id="user-1",
        model_request=None,
    )

    attempts: list[str] = []

    async def handler(request):
        attempts.append(request.model.name)
        if len(attempts) == 1:
            raise ConnectionError("primary provider down")
        return AIMessage(content="recovered")

    result = await middleware.awrap_model_call(_model_request(), handler)

    assert result.content == "recovered"
    assert attempts == ["gemini:gemini-3-flash-preview", "openai:gpt-5"]


async def test_runtime_model_middleware_reraises_without_a_fallback_candidate():
    middleware = RuntimeModelMiddleware(
        runtime_model_resolver=FakeResolver(_runtime_config(fallback_config=None)),
        model_factory=FakeFactory(),
        agent_key="chat",
        user_id="user-1",
        model_request=None,
    )

    async def handler(request):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await middleware.awrap_model_call(_model_request(), handler)


# ----------------------------------------------------------------------
# usage recording
# ----------------------------------------------------------------------


class SpyUsageRecorder:
    def __init__(self):
        self.attempts: list[tuple[str, str, str]] = []

    async def record_one_async_attempt(self, *, call, provider, model, operation):
        try:
            result = await call()
        except BaseException:
            self.attempts.append((provider, model, "error"))
            raise
        self.attempts.append((provider, model, "success"))
        return result


async def test_usage_is_recorded_once_per_successful_provider_attempt():
    recorder = SpyUsageRecorder()
    middleware = UsageRecordingMiddleware(
        usage_recorder=recorder, agent_id="chat_agent", runtime_config_provider=_runtime_config
    )

    async def handler(request):
        return AIMessage(content="ok")

    await middleware.awrap_model_call(_model_request(), handler)

    assert recorder.attempts == [("gemini", "gemini-3-flash-preview", "success")]


async def test_usage_is_recorded_for_a_failed_provider_attempt():
    recorder = SpyUsageRecorder()
    middleware = UsageRecordingMiddleware(
        usage_recorder=recorder, agent_id="chat_agent", runtime_config_provider=_runtime_config
    )

    async def handler(request):
        raise ConnectionError("provider down")

    with pytest.raises(ConnectionError):
        await middleware.awrap_model_call(_model_request(), handler)

    assert recorder.attempts == [("gemini", "gemini-3-flash-preview", "error")]


async def test_usage_recording_is_a_noop_without_a_recorder():
    middleware = UsageRecordingMiddleware(
        usage_recorder=None, agent_id="chat_agent", runtime_config_provider=_runtime_config
    )

    async def handler(request):
        return AIMessage(content="ok")

    assert (await middleware.awrap_model_call(_model_request(), handler)).content == "ok"


# ----------------------------------------------------------------------
# tool authorization
# ----------------------------------------------------------------------


def _tool_request(name="do_thing", args=None, call_id="call-1"):
    return SimpleNamespace(
        tool_call={"id": call_id, "name": name, "args": args or {}},
        tool=SimpleNamespace(name=name, metadata={}),
        state={},
        runtime=SimpleNamespace(context=None),
    )


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def _scope(agent=None, **overrides):
    payload = {
        "agent": agent,
        "agent_key": "chat",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
    }
    payload.update(overrides)
    return SpecialistToolScope(**payload)


async def test_the_execution_map_contains_the_tools_the_model_was_offered():
    """Approval and execution must judge the objects the model actually saw."""
    scope = _scope()
    offered = FakeTool("do_thing")
    scope.offer([offered])

    assert (await scope.tool_map())["do_thing"] is offered


async def test_a_tool_loaded_mid_turn_joins_the_same_execution_map():
    scope = _scope()
    scope.offer([FakeTool("do_thing")])
    resolved = await scope.tool_map()
    resolved["late_arrival"] = FakeTool("late_arrival")

    assert "late_arrival" in await scope.tool_map(), "the map must not be rebuilt per call"


async def test_tool_execution_runs_inside_the_authenticated_scope(monkeypatch):
    from app.ai import tool_context
    from app.ai.workflow import middleware as middleware_module

    seen: dict = {}

    async def fake_execute(**kwargs):
        context = tool_context.get_tool_context()
        seen.update(
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            device_id=context.device_id,
            agent_key=context.agent_key,
        )
        return (
            [{"tool_call_id": "call-1", "name": "do_thing", "content": "ran"}],
            [{"tool_call_id": "call-1", "tool": "do_thing", "status": "success"}],
            [],
        )

    monkeypatch.setattr(middleware_module, "execute_tool_calls", fake_execute)

    scope = _scope()
    scope.offer([FakeTool("do_thing")])
    middleware = ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools)

    async def handler(request):
        raise AssertionError("the framework must not execute the tool itself")

    message = await middleware.awrap_tool_call(_tool_request(), handler)

    assert seen == {
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "agent_key": "chat",
    }
    assert message.content == "ran"
    assert middleware.artifacts == [
        {"tool_call_id": "call-1", "tool": "do_thing", "status": "success"}
    ]


async def test_a_failed_tool_call_is_reported_as_an_error_result(monkeypatch):
    from app.ai.workflow import middleware as middleware_module

    async def fake_execute(**kwargs):
        return (
            [{"tool_call_id": "call-1", "name": "do_thing", "content": "boom"}],
            [{"tool_call_id": "call-1", "tool": "do_thing", "status": "error"}],
            [],
        )

    monkeypatch.setattr(middleware_module, "execute_tool_calls", fake_execute)

    scope = _scope()
    scope.offer([FakeTool("do_thing")])
    middleware = ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools)

    message = await middleware.awrap_tool_call(_tool_request(), _unused_handler)
    assert message.status == "error"


async def test_images_are_collected_separately_from_artifacts(monkeypatch):
    from app.ai.workflow import middleware as middleware_module

    async def fake_execute(**kwargs):
        return (
            [{"tool_call_id": "call-1", "name": "do_thing", "content": "ok"}],
            [{"tool_call_id": "call-1", "tool": "do_thing", "status": "success"}],
            [{"image_id": "img-1"}],
        )

    monkeypatch.setattr(middleware_module, "execute_tool_calls", fake_execute)

    scope = _scope()
    scope.offer([FakeTool("do_thing")])
    middleware = ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools)

    await middleware.awrap_tool_call(_tool_request(), _unused_handler)
    assert middleware.images == [{"image_id": "img-1"}]


async def _no_tools():
    return []


async def _unused_handler(request):
    raise AssertionError("the framework must not execute the tool itself")


# ----------------------------------------------------------------------
# request budget
# ----------------------------------------------------------------------


async def test_request_budget_preflight_runs_before_every_model_attempt():
    preflights: list[int] = []

    async def preflight(request, runtime_config):
        preflights.append(len(request.messages))
        return None

    middleware = RequestBudgetMiddleware(
        preflight=preflight, runtime_config_provider=_runtime_config
    )

    async def handler(request):
        return AIMessage(content="ok")

    await middleware.awrap_model_call(_model_request(), handler)
    await middleware.awrap_model_call(_model_request(), handler)

    assert preflights == [1, 1]


async def test_request_budget_preflight_can_replace_the_messages():
    async def preflight(request, runtime_config):
        return [HumanMessage(content="compacted")]

    middleware = RequestBudgetMiddleware(
        preflight=preflight, runtime_config_provider=_runtime_config
    )

    seen: list[str] = []

    async def handler(request):
        seen.append(request.messages[0].content)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(_model_request(), handler)
    assert seen == ["compacted"]


# ----------------------------------------------------------------------
# stack composition
# ----------------------------------------------------------------------


def test_specialist_stack_installs_the_approval_gate_when_a_policy_is_active():
    stack = _stack(hitl_policy={"master_enabled": True, "global_tools": ["do_thing"]})
    names = [type(item).__name__ for item in stack]

    assert "ToolApprovalMiddleware" in names
    assert "ToolExecutionMiddleware" in names


def test_a_disabled_policy_installs_no_approval_gate():
    """Master-off is the one way approval is skipped, and it is explicit."""
    stack = _stack(hitl_policy={"master_enabled": False})
    assert "ToolApprovalMiddleware" not in [type(item).__name__ for item in stack]


def test_specialist_stack_enforces_framework_call_limits_with_error_exit():
    names = [type(item).__name__ for item in _stack(hitl_policy=None)]
    assert "ModelCallLimitMiddleware" in names
    assert "ToolCallLimitMiddleware" in names
    assert "ToolApprovalMiddleware" not in names


def test_specialist_stack_records_usage_inside_the_fallback_loop():
    """Each provider attempt must be recorded, including a fallback attempt."""
    names = [type(item).__name__ for item in _stack(usage_recorder=SpyUsageRecorder())]
    assert names.index("RuntimeModelMiddleware") < names.index("UsageRecordingMiddleware")


def _stack(**overrides):
    from app.ai.workflow.middleware import ToolApprovalMiddleware

    scope = _scope()
    payload = {
        "runtime_model_resolver": FakeResolver(),
        "model_factory": FakeFactory(),
        "agent_key": "chat",
        "agent_id": "chat_agent",
        "user_id": "user-1",
        "model_request": None,
        "usage_recorder": None,
        "hitl_policy": None,
        "max_model_calls": 8,
        "max_tool_calls": 16,
        "tool_execution": ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools),
        "approval": ToolApprovalMiddleware(scope=scope, hitl_policy={}),
    }
    payload.update(overrides)
    return build_specialist_middleware(**payload)
