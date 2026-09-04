from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest
from anyio import BrokenResourceError, ClosedResourceError

from app.ai.client_runtime_errors import ClientRuntimeToolError
from app.ai.tool_execution import (
    execute_tool_calls,
    invoke_tool_attempt,
    invoke_tool_with_policy,
)
from app.ai.tool_execution_policy import ToolExecutionPolicy, ToolIdentity
from app.core.config import ToolExecutionPolicyOverride, settings
from app.schemas.runtime_protocol import RuntimeErrorContext


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
    assert payload["retryable"] is False
    assert artifacts[0]["failure_retryable"] is True
    assert artifacts[0]["policy_retry_allowed"] is False
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_a_tool_claiming_the_outer_timeout_exemption_fails_closed(monkeypatch):
    """This used to assert the opposite, and the reason it did is gone.

    ``dispatch_subagents`` ran a whole fan-out inside one interactive call, so
    it was allowlisted to opt out of the outer timeout. Fan-out is parent-graph
    topology now and the schema is never executed, so the allowlist is empty —
    and a tool that still claims the exemption must be *refused*, not quietly
    run unbounded. Failing closed is the point: an unbounded interactive call
    that nobody reviewed is worse than a timeout.
    """
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.02)
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 0.01)

    invoked = False

    class _DispatchSubagentsTool:
        name = "dispatch_subagents"
        metadata = {
            "tool_origin": "internal",
            "qualified_tool_id": "internal::dispatch_subagents",
            "application_execution_policy": {"disable_outer_timeout": True},
        }

        async def ainvoke(self, args):  # pragma: no cover - must not run
            nonlocal invoked
            invoked = True
            return "done"

    result, error_detail, error_content, _diagnostics = await invoke_tool_with_policy(
        _DispatchSubagentsTool(),
        {},
        tool_name="dispatch_subagents",
    )

    assert result is None
    assert invoked is False, "policy resolution must reject before the tool runs"
    assert error_detail is not None


@pytest.mark.asyncio
async def test_execute_tool_calls_ignores_legacy_numeric_timeout_metadata(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 0.01)
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 0.005)

    class _SlowOverrideTool:
        name = "slow_override_tool"
        metadata = {"execution_timeout_seconds": 10.0}

        async def ainvoke(self, args):
            await asyncio.sleep(0.05)
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
    calls = 0

    class _RetrySafeTool:
        name = "safe_reader"
        metadata = {
            "application_execution_policy": {
                "max_attempts": 2,
                "retry_safe": True,
            }
        }

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
async def test_retry_success_preserves_attempt_history_policy_diagnostics_and_logs(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(settings, "tool_execution_timeout", 30.0)
    calls = 0

    class _RetryThenSuccessTool:
        name = "diagnostic_reader"
        metadata = {
            "application_execution_policy": {
                "max_attempts": 2,
                "retry_safe": True,
            }
        }

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary network failure")
            return "ok"

    with caplog.at_level(logging.INFO, logger="app.ai.tool_execution"):
        outputs, artifacts, _ = await execute_tool_calls(
            tool_calls=[
                {
                    "id": "call-diagnostics-success",
                    "name": "diagnostic_reader",
                    "args": {"token": "raw-secret-argument"},
                }
            ],
            tool_map={"diagnostic_reader": _RetryThenSuccessTool()},
        )

    artifact = artifacts[0]
    assert outputs[0]["content"] == "ok"
    assert artifact["policy"]["tool_origin"] == "internal"
    assert artifact["policy"]["timeout_seconds"] == 30.0
    assert len(artifact["attempt_history"]) == 2
    assert artifact["attempt_history"][0]["error_type"] == "network"
    assert artifact["attempt_history"][1]["error_type"] is None
    assert artifact["attempt_history"][0]["metadata_trusted"] is True

    attempt_logs = [
        record.tool_execution
        for record in caplog.records
        if record.getMessage() == "tool_execution_attempt"
    ]
    assert len(attempt_logs) == 2
    assert "raw-secret-argument" not in json.dumps(attempt_logs)


@pytest.mark.asyncio
async def test_terminal_failure_has_policy_diagnostics_and_sanitized_attempt_log(
    caplog,
):
    class _StructuredFailureTool:
        name = "structured_failure"
        metadata = {}

        async def ainvoke(self, args):
            raise ClientRuntimeToolError(
                RuntimeErrorContext(
                    message="local secret path was denied",
                    code="PERMISSION_DENIED",
                    detail={"path": "C:/private/secret.txt"},
                )
            )

    with caplog.at_level(logging.INFO, logger="app.ai.tool_execution"):
        outputs, artifacts, _ = await execute_tool_calls(
            tool_calls=[
                {
                    "id": "call-diagnostics-failure",
                    "name": "structured_failure",
                    "args": {"path": "C:/private/secret.txt"},
                }
            ],
            tool_map={"structured_failure": _StructuredFailureTool()},
        )

    payload = json.loads(outputs[0]["content"])
    artifact = artifacts[0]
    assert "attempt_history" not in payload
    assert "policy" not in payload
    assert artifact["policy"]["tool_origin"] == "internal"
    assert artifact["policy"]["timeout_seconds"] == 30.0
    assert len(artifact["attempt_history"]) == 1
    assert artifact["attempt_history"][0]["error_type"] == "permission"
    assert artifact["runtime_error_context"]["detail"]["path"].endswith("secret.txt")

    attempt_logs = [
        record.tool_execution
        for record in caplog.records
        if record.getMessage() == "tool_execution_attempt"
    ]
    assert len(attempt_logs) == 1
    serialized_logs = json.dumps(attempt_logs)
    assert "secret.txt" not in serialized_logs
    assert "raw-secret-argument" not in serialized_logs


@pytest.mark.asyncio
async def test_execute_tool_calls_does_not_auto_retry_unknown_side_effect_tool(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_timeout", 1)
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
    assert payload["retryable"] is False
    assert artifacts[0]["failure_retryable"] is True
    assert artifacts[0]["policy_retry_allowed"] is False
    assert artifacts[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_retries_share_one_total_deadline(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 0.01)
    remaining_by_attempt: list[float | None] = []
    calls = 0

    class _SlowTransientTool:
        name = "slow_reader"
        metadata = {
            "application_execution_policy": {
                "timeout_seconds": 0.08,
                "hard_timeout_seconds": 0.09,
                "total_timeout_seconds": 0.1,
                "max_attempts": 2,
                "retry_safe": True,
            }
        }

        async def ainvoke(self, args):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.06)
            raise ConnectionError("temporarily unavailable")

    from app.ai import tool_execution as tool_execution_module

    real_invoke_tool_attempt = tool_execution_module.invoke_tool_attempt

    async def _record_remaining(tool, tool_args, *, policy, remaining_total_seconds):
        remaining_by_attempt.append(remaining_total_seconds)
        return await real_invoke_tool_attempt(
            tool,
            tool_args,
            policy=policy,
            remaining_total_seconds=remaining_total_seconds,
        )

    monkeypatch.setattr(tool_execution_module, "invoke_tool_attempt", _record_remaining)

    started_at = time.monotonic()
    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "slow_reader", "args": {}}],
        tool_map={"slow_reader": _SlowTransientTool()},
    )
    elapsed = time.monotonic() - started_at

    assert calls == 2
    assert len(remaining_by_attempt) == 2
    assert remaining_by_attempt[0] is not None
    assert remaining_by_attempt[1] is not None
    assert 0 < remaining_by_attempt[1] < remaining_by_attempt[0]
    assert remaining_by_attempt[1] < 0.06
    assert elapsed <= 0.2
    assert json.loads(outputs[0]["content"])["error_type"] == "timeout"
    assert artifacts[0]["attempts"] == 2


@pytest.mark.asyncio
async def test_closed_session_before_mcp_write_reconnects_unsafe_widget_create(monkeypatch):
    invocations = 0
    reconnect_calls = 0

    class _WidgetCreateTool:
        name = "widget_create"
        metadata = {
            "tool_origin": "server_mcp",
            "server_name": "widgets",
            "source_tool_name": "widget_create",
            "qualified_tool_id": "widgets::widget_create",
        }

        async def ainvoke(self, args):
            nonlocal invocations
            invocations += 1
            raise ClosedResourceError

    class _FreshWidgetCreateTool(_WidgetCreateTool):
        async def ainvoke(self, args):
            nonlocal invocations
            invocations += 1
            return "created"

    class _Manager:
        async def reconnect_and_get_tool(self, tool_name):
            nonlocal reconnect_calls
            reconnect_calls += 1
            assert tool_name == "widget_create"
            return _FreshWidgetCreateTool()

    async def _manager():
        return _Manager()

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "widget_create", "args": {}}],
        tool_map={"widget_create": _WidgetCreateTool()},
    )

    assert invocations == 2
    assert reconnect_calls == 1
    assert outputs[0]["content"] == "created"
    assert artifacts[0]["attempts"] == 2


@pytest.mark.asyncio
async def test_unsafe_broken_session_does_not_reconnect_and_repeat(monkeypatch):
    invocations = 0
    reconnect_calls = 0

    class _UnsafeServerTool:
        name = "mutate_remote"
        metadata = {
            "tool_origin": "server_mcp",
            "server_name": "remote",
            "source_tool_name": "mutate_remote",
            "qualified_tool_id": "remote::mutate_remote",
        }

        async def ainvoke(self, args):
            nonlocal invocations
            invocations += 1
            raise BrokenResourceError

    class _Manager:
        async def reconnect_and_get_tool(self, tool_name):
            nonlocal reconnect_calls
            reconnect_calls += 1
            raise AssertionError("unsafe tool must not reconnect")

    async def _manager():
        return _Manager()

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "mutate_remote", "args": {}}],
        tool_map={"mutate_remote": _UnsafeServerTool()},
    )

    assert invocations == 1
    assert reconnect_calls == 0
    assert json.loads(outputs[0]["content"])["retryable"] is False
    assert artifacts[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_safe_session_reconnect_counts_as_next_attempt(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 0.01)
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "safe-session": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "remote::read_remote",
                },
                timeout_seconds=0.08,
                hard_timeout_seconds=0.09,
                total_timeout_seconds=0.12,
                max_attempts=2,
                retry_safe=True,
            )
        },
    )
    invocations = 0
    reconnect_calls = 0

    class _ServerTool:
        name = "read_remote"
        metadata = {
            "tool_origin": "server_mcp",
            "server_name": "remote",
            "source_tool_name": "read_remote",
            "qualified_tool_id": "remote::read_remote",
        }

        async def ainvoke(self, args):
            nonlocal invocations
            invocations += 1
            await asyncio.sleep(0.02)
            raise ClosedResourceError

    class _FreshServerTool(_ServerTool):
        async def ainvoke(self, args):
            nonlocal invocations
            invocations += 1
            await asyncio.sleep(0.01)
            return "recovered"

    class _Manager:
        async def reconnect_and_get_tool(self, tool_name):
            nonlocal reconnect_calls
            reconnect_calls += 1
            await asyncio.sleep(0.02)
            return _FreshServerTool()

    async def _manager():
        return _Manager()

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

    started_at = time.monotonic()
    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "read_remote", "args": {}}],
        tool_map={"read_remote": _ServerTool()},
    )
    elapsed = time.monotonic() - started_at

    assert invocations == 2
    assert reconnect_calls == 1
    assert outputs[0]["content"] == "recovered"
    assert artifacts[0]["attempts"] == 2
    assert len(artifacts[0]["attempt_history"]) == 2
    first, second = artifacts[0]["attempt_history"]
    assert first["qualified_tool_id"] == second["qualified_tool_id"]
    assert first["policy_source"] == second["policy_source"]
    assert first["policy_config_keys"] == second["policy_config_keys"] == ["safe-session"]
    assert elapsed <= 0.2


@pytest.mark.asyncio
async def test_reconnect_failure_records_terminal_attempt(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 0.01)
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "safe-session": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "remote::read_remote",
                },
                timeout_seconds=0.08,
                hard_timeout_seconds=0.09,
                total_timeout_seconds=0.2,
                max_attempts=2,
                retry_safe=True,
            )
        },
    )

    class _ServerTool:
        name = "read_remote"
        metadata = {
            "tool_origin": "server_mcp",
            "server_name": "remote",
            "source_tool_name": "read_remote",
            "qualified_tool_id": "remote::read_remote",
        }

        async def ainvoke(self, args):
            raise ClosedResourceError

    class _Manager:
        async def reconnect_and_get_tool(self, tool_name):
            raise TimeoutError("reconnect timed out")

    async def _manager():
        return _Manager()

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _manager)

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "read_remote", "args": {}}],
        tool_map={"read_remote": _ServerTool()},
    )

    assert json.loads(outputs[0]["content"])["error_type"] == "timeout"
    assert artifacts[0]["attempts"] == 2
    assert artifacts[0]["attempt_history"][-1]["error_type"] == "timeout"
    assert artifacts[0]["attempt_history"][-1]["auto_retry_allowed"] is False


@pytest.mark.asyncio
async def test_policy_resolution_failure_is_sanitized_for_model(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "policy-a": ToolExecutionPolicyOverride(
                match={"tool_origin": "internal"},
                timeout_seconds=1,
            ),
            "policy-b": ToolExecutionPolicyOverride(
                match={"tool_origin": "internal"},
                timeout_seconds=2,
            ),
        },
    )

    class _Tool:
        name = "read"
        metadata = {}

        async def ainvoke(self, args):
            return "unreachable"

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "read", "args": {}}],
        tool_map={"read": _Tool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["error_type"] == "configuration"
    assert payload["retryable"] is False
    assert "policy-a" not in outputs[0]["content"]
    assert "policy-b" not in outputs[0]["content"]
    assert "policy-a" in artifacts[0]["diagnostic"]


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


@pytest.mark.asyncio
async def test_skill_terminal_error_output_is_flagged_to_reach_model_uncut():
    terminal = "skill command exited with code 1: " + "x" * 6000

    class _SkillFailureTool:
        name = "client__demo__run_skill_command"
        metadata = {}

        async def ainvoke(self, args):
            raise ClientRuntimeToolError(
                RuntimeErrorContext(
                    message=terminal,
                    code="RUNTIME_ERROR",
                    detail={"qualified_tool_id": "skill::demo::run_skill_command"},
                )
            )

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[
            {
                "id": "call-skill",
                "name": "client__demo__run_skill_command",
                "args": {"argv": ["demo-cli"]},
            }
        ],
        tool_map={"client__demo__run_skill_command": _SkillFailureTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["untrusted_terminal_output"] == terminal
    assert outputs[0]["preserve_full_content"] is True
    assert artifacts[0]["skill_terminal_error"] is True


def test_skill_terminal_error_output_is_not_offloaded():
    from app.ai.tool_execution import _apply_offload_to_outputs_and_artifacts

    class _FakeOffloadService:
        threshold_chars = 10

        def offload_if_large(self, **kwargs):
            return {
                "blob_id": "blob-1",
                "size_bytes": len(kwargs["output_text"].encode("utf-8")),
                "output": "preview only\n\n[Output offloaded]",
            }

    content = json.dumps(
        {"status": "error", "untrusted_terminal_output": "x" * 200},
        separators=(",", ":"),
    )
    outputs = [
        {
            "tool_call_id": "call-1",
            "name": "client__demo__run_skill_command",
            "content": content,
            "preserve_full_content": True,
        }
    ]
    artifacts = [
        {
            "tool_call_id": "call-1",
            "tool": "client__demo__run_skill_command",
            "output": content,
        }
    ]

    _apply_offload_to_outputs_and_artifacts(
        outputs=outputs,
        artifacts=artifacts,
        conversation_id="00000000-0000-0000-0000-000000000001",
        user_id="00000000-0000-0000-0000-000000000002",
        offload_service=_FakeOffloadService(),
    )

    assert outputs[0]["content"] == content
    assert "blob_id" not in artifacts[0]


@pytest.mark.asyncio
async def test_execute_tool_calls_offloads_off_the_event_loop_thread(monkeypatch):
    """offload_if_large commits a multi-MB payload synchronously to Postgres.

    Running that commit on the event loop thread would stall every other
    concurrent request/stream for its duration. Recording the thread identity
    the fake service observes proves the offload actually ran elsewhere,
    rather than merely proving the call happened at all.
    """
    caller_thread_id = threading.get_ident()
    offload_thread_ids: list[int] = []

    class _ThreadRecordingOffloadService:
        threshold_chars = 1

        def offload_if_large(self, **kwargs):
            offload_thread_ids.append(threading.get_ident())
            return {
                "blob_id": "blob-thread-check",
                "size_bytes": len(kwargs["output_text"].encode("utf-8")),
                "output": "preview only\n\n[Output offloaded]",
            }

    monkeypatch.setattr(
        "app.ai.tool_execution._resolve_offload_service",
        lambda: _ThreadRecordingOffloadService(),
    )

    class _BigOutputTool:
        name = "big_output_tool"
        metadata = {}

        async def ainvoke(self, args):
            return "y" * 50

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "big_output_tool", "args": {}}],
        tool_map={"big_output_tool": _BigOutputTool()},
        conversation_id="00000000-0000-0000-0000-000000000003",
        user_id="00000000-0000-0000-0000-000000000004",
    )

    assert artifacts[0]["blob_id"] == "blob-thread-check"
    assert outputs[0]["content"].startswith("preview only")
    assert offload_thread_ids, "offload_if_large was never called"
    assert offload_thread_ids[0] != caller_thread_id, (
        "offload ran on the event loop thread, blocking concurrent requests"
    )
