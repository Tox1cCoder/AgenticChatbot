"""Focused middleware contracts for routing-v2 specialists.

Each middleware here is small and single-purpose on purpose. What the stack
must guarantee, together, is the order: authorization decides before HITL asks,
HITL asks before the tool implementation runs, and usage is recorded once per
provider attempt whether or not that attempt succeeded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.workflow.middleware import (
    ArtifactCaptureMiddleware,
    RequestBudgetMiddleware,
    RuntimeModelMiddleware,
    ToolAuthorizationDenied,
    ToolAuthorizationMiddleware,
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


async def test_authorization_runs_before_the_tool_implementation():
    executed: list[str] = []

    def authorize(tool_name, args, *, user_id, device_id):
        return tool_name != "forbidden_tool"

    middleware = ToolAuthorizationMiddleware(
        authorize=authorize, user_id="user-1", device_id="device-1"
    )

    async def handler(request):
        executed.append(request.tool_call["name"])
        return ToolMessage(content="ran", tool_call_id="call-1")

    result = await middleware.awrap_tool_call(_tool_request("forbidden_tool"), handler)

    assert executed == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "not authorized" in result.content


async def test_authorized_tools_reach_the_implementation():
    executed: list[str] = []

    middleware = ToolAuthorizationMiddleware(
        authorize=lambda *a, **k: True, user_id="user-1", device_id="device-1"
    )

    async def handler(request):
        executed.append(request.tool_call["name"])
        return ToolMessage(content="ran", tool_call_id="call-1")

    await middleware.awrap_tool_call(_tool_request(), handler)
    assert executed == ["do_thing"]


async def test_authorization_denial_is_model_visible_not_an_exception():
    middleware = ToolAuthorizationMiddleware(
        authorize=lambda *a, **k: False, user_id=None, device_id=None
    )

    async def handler(request):
        raise AssertionError("implementation must not run")

    message = await middleware.awrap_tool_call(_tool_request(), handler)
    assert message.tool_call_id == "call-1"


def test_tool_authorization_denied_carries_the_tool_name():
    error = ToolAuthorizationDenied("secret_tool", "device scope")
    assert error.tool_name == "secret_tool"
    assert "device scope" in str(error)


# ----------------------------------------------------------------------
# artifact capture
# ----------------------------------------------------------------------


async def test_artifacts_are_captured_into_private_state_not_public_content():
    middleware = ArtifactCaptureMiddleware()

    async def handler(request):
        message = ToolMessage(content="done", tool_call_id="call-1")
        message.artifact = {"artifact_id": "artifact-1", "kind": "table"}
        return message

    await middleware.awrap_tool_call(_tool_request(), handler)

    assert middleware.artifacts == [{"artifact_id": "artifact-1", "kind": "table"}]


async def test_artifact_capture_ignores_tools_without_artifacts():
    middleware = ArtifactCaptureMiddleware()

    async def handler(request):
        return ToolMessage(content="done", tool_call_id="call-1")

    await middleware.awrap_tool_call(_tool_request(), handler)
    assert middleware.artifacts == []


async def test_artifact_capture_collects_images_separately():
    middleware = ArtifactCaptureMiddleware()

    async def handler(request):
        message = ToolMessage(content="done", tool_call_id="call-1")
        message.artifact = {"artifact_id": "a-1", "kind": "image", "image_id": "img-1"}
        return message

    await middleware.awrap_tool_call(_tool_request(), handler)

    assert middleware.images == [{"artifact_id": "a-1", "kind": "image", "image_id": "img-1"}]


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


def test_specialist_stack_orders_authorization_before_approval():
    stack = build_specialist_middleware(
        runtime_model_resolver=FakeResolver(),
        model_factory=FakeFactory(),
        agent_key="chat",
        agent_id="chat_agent",
        user_id="user-1",
        device_id="device-1",
        model_request=None,
        usage_recorder=None,
        authorize=lambda *a, **k: True,
        hitl_policy={"master_enabled": True, "global_tools": ["do_thing"]},
        max_model_calls=8,
        max_tool_calls=16,
    )
    names = [type(item).__name__ for item in stack]

    authorization_index = names.index("ToolAuthorizationMiddleware")
    assert "HumanInTheLoopMiddleware" in names
    assert authorization_index < names.index("HumanInTheLoopMiddleware")


def test_specialist_stack_enforces_framework_call_limits_with_error_exit():
    stack = build_specialist_middleware(
        runtime_model_resolver=FakeResolver(),
        model_factory=FakeFactory(),
        agent_key="chat",
        agent_id="chat_agent",
        user_id="user-1",
        device_id="device-1",
        model_request=None,
        usage_recorder=None,
        authorize=lambda *a, **k: True,
        hitl_policy=None,
        max_model_calls=8,
        max_tool_calls=16,
    )
    names = [type(item).__name__ for item in stack]
    assert "ModelCallLimitMiddleware" in names
    assert "ToolCallLimitMiddleware" in names
    # No HITL middleware is installed when the turn has no approval policy.
    assert "HumanInTheLoopMiddleware" not in names


def test_specialist_stack_records_usage_inside_the_fallback_loop():
    """Each provider attempt must be recorded, including a fallback attempt."""
    stack = build_specialist_middleware(
        runtime_model_resolver=FakeResolver(),
        model_factory=FakeFactory(),
        agent_key="chat",
        agent_id="chat_agent",
        user_id="user-1",
        device_id="device-1",
        model_request=None,
        usage_recorder=SpyUsageRecorder(),
        authorize=lambda *a, **k: True,
        hitl_policy=None,
        max_model_calls=8,
        max_tool_calls=16,
    )
    names = [type(item).__name__ for item in stack]
    assert names.index("RuntimeModelMiddleware") < names.index("UsageRecordingMiddleware")
