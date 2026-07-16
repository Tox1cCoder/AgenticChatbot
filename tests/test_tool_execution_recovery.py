from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from app.ai.tool_execution import execute_tool_calls, invoke_tool_attempt
from app.ai.tool_execution_policy import ToolExecutionPolicy, ToolIdentity
from app.core.config import settings


def _attempt_policy(
    *,
    cancellation: str = "cooperative",
    timeout_seconds: float = 0.05,
    hard_timeout_seconds: float = 0.15,
    total_timeout_seconds: float = 0.3,
    outer_timeout_disabled: bool = False,
) -> ToolExecutionPolicy:
    return ToolExecutionPolicy(
        identity=ToolIdentity(
            tool_origin="internal",
            qualified_tool_id="internal::deadline_test",
            exposed_tool_name="deadline_test",
            source_tool_name="deadline_test",
            server_name=None,
        ),
        timeout_seconds=timeout_seconds,
        hard_timeout_seconds=hard_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
        max_attempts=1,
        retry_safe=False,
        idempotent=False,
        metadata_trusted=False,
        outer_timeout_disabled=outer_timeout_disabled,
        cancellation=cancellation,
        client_execution_timeout_seconds=None,
        client_response_timeout_seconds=None,
        policy_source="test",
        policy_config_keys=(),
        timeout_hint="",
    )


@pytest.mark.asyncio
async def test_soft_timeout_cancels_cooperative_async_tool():
    cancellation_observed = asyncio.Event()

    class _CooperativeTool:
        async def ainvoke(self, args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_observed.set()
                raise

    outcome = await invoke_tool_attempt(
        _CooperativeTool(),
        {},
        policy=_attempt_policy(),
        remaining_total_seconds=0.3,
    )

    assert cancellation_observed.is_set()
    assert isinstance(outcome.exception, TimeoutError)
    assert outcome.timeout_phase == "soft_timeout"
    assert outcome.cancellation_attempted is True
    assert outcome.cancellation_completed is True


@pytest.mark.asyncio
async def test_hard_timeout_stops_waiting_for_cancellation_suppressing_tool():
    release = asyncio.Event()
    finished = asyncio.Event()

    class _CancellationSuppressingTool:
        async def ainvoke(self, args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            finally:
                finished.set()

    outcome = await invoke_tool_attempt(
        _CancellationSuppressingTool(),
        {},
        policy=_attempt_policy(),
        remaining_total_seconds=0.3,
    )

    try:
        assert isinstance(outcome.exception, TimeoutError)
        assert outcome.timeout_phase == "hard_timeout"
        assert outcome.cancellation_attempted is True
        assert outcome.cancellation_completed is False
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)


@pytest.mark.asyncio
async def test_sync_thread_timeout_is_recorded_as_abandon_only():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_results: list[str] = []

    class _BlockingSyncTool:
        def invoke(self, args):
            started.set()
            release.wait(timeout=2)
            worker_results.append("thread-result")
            finished.set()
            return worker_results[-1]

    runner = asyncio.create_task(
        invoke_tool_attempt(
            _BlockingSyncTool(),
            {},
            policy=_attempt_policy(
                cancellation="abandon_only",
                timeout_seconds=0.3,
                hard_timeout_seconds=0.5,
                total_timeout_seconds=0.6,
            ),
            remaining_total_seconds=0.6,
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 1)
        outcome = await runner
        assert isinstance(outcome.exception, TimeoutError)
        assert outcome.timeout_phase == "soft_timeout"
        assert outcome.cancellation_attempted is True
        assert outcome.cancellation_completed is False
        assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)

    assert worker_results == ["thread-result"]


@pytest.mark.asyncio
async def test_parent_cancellation_during_soft_wait_cancels_and_consumes_child():
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = asyncio.Event()
    child_cancelled = asyncio.Event()
    finished = asyncio.Event()
    exception_contexts: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: exception_contexts.append(context))

    class _LateFailingCancellationTool:
        async def ainvoke(self, args):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                child_cancelled.set()
                raise RuntimeError("child cleanup failed") from None
            finally:
                finished.set()

    runner = asyncio.create_task(
        invoke_tool_attempt(
            _LateFailingCancellationTool(),
            {},
            policy=_attempt_policy(
                timeout_seconds=1,
                hard_timeout_seconds=1.2,
                total_timeout_seconds=1.5,
            ),
            remaining_total_seconds=1.5,
        )
    )

    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        await asyncio.sleep(0)

        assert child_cancelled.is_set()
        assert finished.is_set()
        assert not any(
            "Task exception was never retrieved" in str(context.get("message", ""))
            for context in exception_contexts
        )
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_parent_cancellation_during_hard_cleanup_cancels_and_consumes_child():
    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    soft_cancellation_observed = asyncio.Event()
    parent_cancellation_observed = asyncio.Event()
    finished = asyncio.Event()
    exception_contexts: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: exception_contexts.append(context))

    class _TwiceCancelledTool:
        async def ainvoke(self, args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                soft_cancellation_observed.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    parent_cancellation_observed.set()
                    raise RuntimeError("second cancellation cleanup failed") from None
            finally:
                finished.set()

    runner = asyncio.create_task(
        invoke_tool_attempt(
            _TwiceCancelledTool(),
            {},
            policy=_attempt_policy(
                timeout_seconds=0.05,
                hard_timeout_seconds=1,
                total_timeout_seconds=1.2,
            ),
            remaining_total_seconds=1.2,
        )
    )

    try:
        await asyncio.wait_for(soft_cancellation_observed.wait(), timeout=1)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        await asyncio.sleep(0)

        assert parent_cancellation_observed.is_set()
        assert finished.is_set()
        assert not any(
            "Task exception was never retrieved" in str(context.get("message", ""))
            for context in exception_contexts
        )
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remaining_total_seconds",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(0.0, id="zero"),
        pytest.param(-0.1, id="negative-finite"),
    ],
)
async def test_exhausted_cumulative_deadline_prevents_attempt_start(remaining_total_seconds):
    invocations: list[dict] = []

    class _ImmediateTool:
        async def ainvoke(self, args):
            invocations.append(args)
            return "side effect"

    outcome = await invoke_tool_attempt(
        _ImmediateTool(),
        {"value": 1},
        policy=_attempt_policy(outer_timeout_disabled=True),
        remaining_total_seconds=remaining_total_seconds,
    )

    assert isinstance(outcome.exception, TimeoutError)
    assert outcome.timeout_phase == "hard_timeout"
    assert outcome.cancellation_attempted is False
    assert outcome.cancellation_completed is False
    assert invocations == []


@pytest.mark.asyncio
async def test_abandoned_task_exception_is_consumed():
    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    finished = asyncio.Event()
    exception_contexts: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: exception_contexts.append(context))

    class _LateFailingTool:
        async def ainvoke(self, args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                raise RuntimeError("late failure") from None
            finally:
                finished.set()

    try:
        outcome = await invoke_tool_attempt(
            _LateFailingTool(),
            {},
            policy=_attempt_policy(),
            remaining_total_seconds=0.3,
        )
        assert outcome.timeout_phase == "hard_timeout"

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)

        assert not any(
            context.get("message") == "Task exception was never retrieved"
            for context in exception_contexts
        )
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_execute_tool_calls_recovers_missing_client_tool_from_active_catalog(monkeypatch):
    invoked_args: list[dict] = []

    class _ClientTool:
        name = "client__desktop_commander__start_process"

        async def ainvoke(self, args: dict):
            invoked_args.append(args)
            return "started"

    async def _noop_refresh(**kwargs):
        return None

    monkeypatch.setattr(
        "app.ai.tool_execution._refresh_tool_map_after_search",
        _noop_refresh,
    )
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_runtime_tools",
        lambda **kwargs: [_ClientTool()],
    )
    monkeypatch.setattr("app.ai.tool_execution.is_client_tool", lambda _tool: True)
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_tool_device_id",
        lambda _tool: "device-123",
    )
    monkeypatch.setattr(
        "app.ai.tool_execution._mark_tool_used_if_deferred",
        lambda _tool_name: None,
    )

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "tool-1",
                "name": "client__desktop_commander__start_process",
                "args": {"command": "notepad.exe"},
            }
        ],
        tool_map={},
        device_id="device-123",
        user_id="user-123",
        conversation_id="conversation-1",
        agent=SimpleNamespace(agent_config_key="chat"),
    )

    assert invoked_args == [{"command": "notepad.exe"}]
    assert outputs == [
        {
            "tool_call_id": "tool-1",
            "name": "client__desktop_commander__start_process",
            "content": "started",
            "render": artifacts[0]["render"],
        }
    ]
    assert artifacts[0]["status"] == "success"
    assert images == []


@pytest.mark.asyncio
async def test_execute_tool_calls_recovers_aliased_server_tool(monkeypatch):
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    invoked = []

    class _ServerTool:
        name = "search"

        async def ainvoke(self, args):
            invoked.append(args)
            return "searched"

    brave_tool = _ServerTool()

    class _Manager:
        async def get_tools(self):
            return [brave_tool]

        def get_server_for_tool(self, tool):
            return "brave"

    async def _manager():
        return _Manager()

    async def _noop_refresh(**kwargs):
        return None

    reset_deferred_tool_state()
    try:
        get_deferred_tool_state().autoload(
            conversation_id="conv-1",
            agent_key="chat",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )
        monkeypatch.setattr("app.ai.tool_execution._refresh_tool_map_after_search", _noop_refresh)
        monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

        outputs, artifacts, _ = await execute_tool_calls(
            tool_calls=[{"id": "call-1", "name": "brave__search", "args": {"query": "x"}}],
            tool_map={},
            conversation_id="conv-1",
            user_id="user-1",
            agent=SimpleNamespace(agent_config_key="chat", tool_state_key="chat"),
        )

        assert invoked == [{"query": "x"}]
        assert outputs[0]["name"] == "brave__search"
        assert artifacts[0]["status"] == "success"
    finally:
        reset_deferred_tool_state()


@pytest.mark.asyncio
async def test_recover_missing_tool_does_not_bind_unloaded_server_tool(monkeypatch):
    from app.ai.deferred_tool_state import reset_deferred_tool_state
    from app.ai.tool_execution import _recover_missing_tool
    from app.core.config import settings

    reset_deferred_tool_state()
    monkeypatch.setattr(settings, "mcp_tool_search_enabled", True)

    dangerous_tool = SimpleNamespace(
        name="delete_everything",
        description="Unselected server tool",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "admin",
            "qualified_tool_id": "admin::delete_everything",
        },
    )

    class _Manager:
        async def get_tools(self):
            return [dangerous_tool]

        def get_server_for_tool(self, tool):
            return "admin"

    async def _manager():
        return _Manager()

    reset_deferred_tool_state()
    try:
        monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)
        tool_map: dict[str, object] = {}

        recovered = await _recover_missing_tool(
            tool_name="delete_everything",
            tool_map=tool_map,
            agent=SimpleNamespace(
                agent_config_key="custom",
                tool_state_key="custom_agent:abc",
            ),
            conversation_id="conv-1",
            user_id="user-1",
            device_id=None,
        )

        assert recovered is None
        assert "delete_everything" not in tool_map
    finally:
        reset_deferred_tool_state()


@pytest.mark.asyncio
async def test_execute_tool_calls_times_out_slow_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.01)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _SlowTool:
        name = "slow_tool"
        metadata = {}

        async def ainvoke(self, args):
            await asyncio.sleep(1)
            return "too late"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "slow_tool", "args": {}}],
        tool_map={"slow_tool": _SlowTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "timeout"
    assert payload["retryable"] is True
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_execute_tool_calls_metadata_none_disables_timeout(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.01)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _LongRunningTool:
        name = "dispatch_like_tool"
        metadata = {"execution_timeout_seconds": None}

        async def ainvoke(self, args):
            await asyncio.sleep(0.05)
            return "done"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "dispatch_like_tool", "args": {}}],
        tool_map={"dispatch_like_tool": _LongRunningTool()},
    )

    assert outputs[0]["content"] == "done"
    assert artifacts[0].get("status") != "error"


def test_resolve_tool_timeout_seconds_bridges_dispatch_subagents_policy_metadata(monkeypatch):
    """The new ``application_execution_policy`` metadata shape (Task 2) must
    keep disabling the outer timeout for the one allowlisted identity, until
    the runner is migrated onto ``resolve_tool_execution_policy`` directly."""
    from app.ai.tool_execution import _resolve_tool_timeout_seconds

    monkeypatch.setattr(settings, "tool_execution_timeout", 30)

    dispatch_tool = SimpleNamespace(
        name="dispatch_subagents",
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::dispatch_subagents",
            "application_execution_policy": {"disable_outer_timeout": True},
        },
    )
    assert _resolve_tool_timeout_seconds(dispatch_tool) is None


def test_resolve_tool_timeout_seconds_ignores_disable_for_other_internal_tools(monkeypatch):
    """Only the exact dispatch_subagents identity may disable the outer
    timeout — a different internal tool claiming the same metadata key must
    still get the ordinary bounded timeout."""
    from app.ai.tool_execution import _resolve_tool_timeout_seconds

    monkeypatch.setattr(settings, "tool_execution_timeout", 30)

    other_tool = SimpleNamespace(
        name="some_other_tool",
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::some_other_tool",
            "application_execution_policy": {"disable_outer_timeout": True},
        },
    )
    assert _resolve_tool_timeout_seconds(other_tool) == 30.0


@pytest.mark.asyncio
async def test_execute_tool_calls_metadata_overrides_timeout(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 10)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 0)

    class _SlowOverrideTool:
        name = "slow_override_tool"
        metadata = {"execution_timeout_seconds": 0.01}

        async def ainvoke(self, args):
            await asyncio.sleep(1)
            return "too late"

    outputs, _, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "slow_override_tool", "args": {}}],
        tool_map={"slow_override_tool": _SlowOverrideTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_execute_tool_calls_retries_retry_safe_transient_failure(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 1)
    calls = 0

    class _RetrySafeTool:
        name = "safe_reader"
        metadata = {"retry_safe": True}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("connection reset")
            return "ok"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "safe_reader", "args": {}}],
        tool_map={"safe_reader": _RetrySafeTool()},
    )

    assert calls == 2
    assert outputs[0]["content"] == "ok"
    assert artifacts[0]["status"] == "success"


@pytest.mark.asyncio
async def test_execute_tool_calls_does_not_auto_retry_unknown_side_effect_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
    monkeypatch.setattr(settings, "tool_execution_max_retries", 2)
    calls = 0

    class _MaybeSideEffectTool:
        name = "send_message"
        metadata = {}

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            raise ConnectionError("connection reset")

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "send_message", "args": {"body": "hi"}}],
        tool_map={"send_message": _MaybeSideEffectTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert calls == 1
    assert payload["error_type"] == "network"
    assert payload["retryable"] is True
    assert artifacts[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_execute_tool_calls_empty_tool_map_returns_error_for_each_call(monkeypatch):
    monkeypatch.setattr(settings, "mcp_tool_search_enabled", False)

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[
            {"id": "call-1", "name": "missing_one", "args": {}},
            {"id": "call-2", "name": "missing_two", "args": {}},
        ],
        tool_map={},
    )

    assert [output["tool_call_id"] for output in outputs] == ["call-1", "call-2"]
    assert all(json.loads(output["content"])["status"] == "error" for output in outputs)
    assert all(artifact["status"] == "error" for artifact in artifacts)


@pytest.mark.asyncio
async def test_execute_tool_calls_missing_tool_name_returns_compact_argument_error():
    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "args": {"query": "x"}}],
        tool_map={},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "argument"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "argument"


@pytest.mark.asyncio
async def test_execute_tool_calls_client_device_mismatch_returns_compact_error(monkeypatch):
    class _ClientTool:
        name = "client__desktop__read_file"
        metadata = {}

        async def ainvoke(self, args):
            return "never"

    monkeypatch.setattr("app.ai.tool_execution.is_client_tool", lambda _tool: True)
    monkeypatch.setattr(
        "app.ai.tool_execution.get_client_tool_device_id",
        lambda _tool: "device-a",
    )

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "client__desktop__read_file", "args": {}}],
        tool_map={"client__desktop__read_file": _ClientTool()},
        device_id="device-b",
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "permission"
    assert payload["retryable"] is False
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "permission"
