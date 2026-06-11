"""
Per-client tool invocation isolation (client_invocation.md).

Covers:
- T010: a turn without a device_id clears the checkpointed device_id, so a
  conversation started on machine A cannot keep dispatching to A.
- T011: with two devices connected, a turn from device B binds only B's
  catalog and dispatch targets B.
- T002: an invalid/foreign/inactive device_id on the request is replaced with
  None when the workflow request is built.
- T003: loaded-client-tool lookups without a full (device_id, session_id)
  scope return nothing instead of scanning across devices.
- T012: the dispatch guard rejects missing/mismatched context devices and
  disconnected clients with a graceful tool error result (FR-4), never a
  silent dispatch and never a crashed turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.ai.client_runtime_tools import _CLIENT_TOOL_CACHE, get_client_runtime_tools
from app.ai.client_tool_catalog import reset_all_client_catalogs
from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import GraphState, WorkflowExecutionRequest
from app.ai.tool_context import tool_execution_context
from app.core.config import settings
from app.services import client_runtime_store as runtime_store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    InMemoryClientRuntimeStore,
    reset_client_runtime_store,
)
from app.services.message_service import MessageService


def _catalog(server_name: str, tool_name: str) -> dict:
    return {
        "tools": [
            {
                "name": tool_name,
                "origin": "mcp",
                "server_name": server_name,
                "qualified_id": f"{server_name}::{tool_name}",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    }


def _reset_runtime_state() -> None:
    reset_client_runtime_store()
    reset_deferred_tool_state()
    _CLIENT_TOOL_CACHE.clear()
    reset_all_client_catalogs()


async def _connect_device(store, *, user_id, device_id, session_id, server, tool):
    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id=session_id,
            user_id=user_id,
            tool_catalog=_catalog(server, tool),
            tool_catalog_version=1,
        )
    )


def _compile_checkpointed_state_graph():
    builder = StateGraph(GraphState)
    builder.add_node("noop", lambda state: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_turn_without_device_id_clears_checkpointed_device():
    """T010: device_id from turn 1 must not leak into a turn that has none."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    graph = _compile_checkpointed_state_graph()
    config = {"configurable": {"thread_id": "thread-isolation"}}
    device_a = str(uuid4())

    turn_one = workflow._build_initial_state_from_request(
        WorkflowExecutionRequest(message="hello from A", device_id=device_a)
    )
    await graph.ainvoke(turn_one, config)
    assert graph.get_state(config).values.get("device_id") == device_a

    turn_two = workflow._build_initial_state_from_request(
        WorkflowExecutionRequest(message="hello with no device", device_id=None)
    )
    await graph.ainvoke(turn_two, config)

    assert graph.get_state(config).values.get("device_id") is None
    # A turn without a device binds zero client tools.
    assert get_client_runtime_tools(user_id=str(uuid4()), device_id=None) == []


@pytest.mark.asyncio
async def test_turn_from_device_b_binds_and_dispatches_only_b(monkeypatch):
    """T011: same user, devices A+B connected; a turn from B only sees B."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_a = uuid4()
    device_b = uuid4()

    await _connect_device(
        store,
        user_id=user_id,
        device_id=device_a,
        session_id="session-a",
        server="desktop_commander",
        tool="start_process",
    )
    await _connect_device(
        store,
        user_id=user_id,
        device_id=device_b,
        session_id="session-b",
        server="time_server",
        tool="get_current_time",
    )

    dispatches: list[dict] = []

    async def _capture_dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", _capture_dispatch)

    try:
        tools_b = get_client_runtime_tools(user_id=str(user_id), device_id=str(device_b))

        assert [tool.name for tool in tools_b] == ["client__time_server__get_current_time"]
        assert all(tool.metadata["device_id"] == str(device_b) for tool in tools_b)

        with tool_execution_context(
            conversation_id="conversation-1",
            user_id=str(user_id),
            agent_key="chat",
            device_id=str(device_b),
        ):
            result = await tools_b[0].coroutine()

        assert result == "ok"
        assert [d["device_id"] for d in dispatches] == [str(device_b)]
    finally:
        _reset_runtime_state()


def _build_message_service() -> MessageService:
    return MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
    )


def _message_create_data(conversation_id, device_id) -> SimpleNamespace:
    return SimpleNamespace(
        conversation_id=conversation_id,
        content="hello",
        device_id=device_id,
        attachments=None,
        model_config_field=None,
        inline_rich_response_v1=False,
    )


@pytest.mark.asyncio
async def test_inactive_device_id_dropped_when_building_workflow_request(monkeypatch):
    """T002: a device_id without an active runtime session becomes None."""
    _reset_runtime_state()
    monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    service = _build_message_service()
    user_id = uuid4()
    stale_device_id = uuid4()

    try:
        _, _, request = await service._build_user_message_workflow_request(
            message_create_data=_message_create_data(uuid4(), stale_device_id),
            user_id=user_id,
            conversation=None,
        )

        assert request.device_id is None
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_foreign_user_device_id_dropped_when_building_workflow_request(monkeypatch):
    """T002: a device registered to another user is treated as no device."""
    _reset_runtime_state()
    monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    requesting_user = uuid4()
    other_user = uuid4()
    other_users_device = uuid4()
    await _connect_device(
        store,
        user_id=other_user,
        device_id=other_users_device,
        session_id="session-foreign",
        server="desktop_commander",
        tool="start_process",
    )

    service = _build_message_service()

    try:
        _, _, request = await service._build_user_message_workflow_request(
            message_create_data=_message_create_data(uuid4(), other_users_device),
            user_id=requesting_user,
            conversation=None,
        )

        assert request.device_id is None
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_active_device_id_preserved_when_building_workflow_request(monkeypatch):
    """T002: a device with an active session for this user passes through."""
    _reset_runtime_state()
    monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()
    await _connect_device(
        store,
        user_id=user_id,
        device_id=device_id,
        session_id="session-active",
        server="time_server",
        tool="get_current_time",
    )

    service = _build_message_service()

    try:
        _, _, request = await service._build_user_message_workflow_request(
            message_create_data=_message_create_data(uuid4(), device_id),
            user_id=user_id,
            conversation=None,
        )

        assert request.device_id == str(device_id)
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_loaded_client_tools_require_device_and_session_scope():
    """T003: lookups without full scope return [] instead of scanning devices."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()
    await _connect_device(
        store,
        user_id=user_id,
        device_id=device_id,
        session_id="session-a",
        server="desktop_commander",
        tool="start_process",
    )

    try:
        deferred_state = get_deferred_tool_state()
        deferred_state.autoload_client_tools(
            conversation_id="conversation-1",
            agent_key="chat",
            references=[
                SimpleNamespace(
                    tool_name="client__desktop_commander__start_process",
                    server_name="desktop_commander",
                    device_id=str(device_id),
                    session_id="session-a",
                    catalog_version=1,
                    tool_instance_id="instance-1",
                )
            ],
            device_id=str(device_id),
            session_id="session-a",
            user_id=str(user_id),
        )

        fully_scoped = deferred_state.get_loaded_client_tools(
            "conversation-1",
            "chat",
            device_id=str(device_id),
            session_id="session-a",
        )
        assert [tool.tool_name for tool in fully_scoped] == [
            "client__desktop_commander__start_process"
        ]

        # No scope at all: must not leak any device's tools.
        assert deferred_state.get_loaded_client_tools("conversation-1", "chat") == []

        # Device without any resolvable active session: nothing.
        await store.delete_session(device_id)
        assert (
            deferred_state.get_loaded_client_tools(
                "conversation-1",
                "chat",
                device_id=str(device_id),
            )
            == []
        )
    finally:
        _reset_runtime_state()


async def _bound_tool_for_device(store, *, user_id, device_id, session_id):
    await _connect_device(
        store,
        user_id=user_id,
        device_id=device_id,
        session_id=session_id,
        server="time_server",
        tool="get_current_time",
    )
    tools = get_client_runtime_tools(user_id=str(user_id), device_id=str(device_id))
    assert len(tools) == 1
    return tools[0]


def _assert_graceful_tool_error(result) -> None:
    """FR-4 + FR-2: a plain error string the model can read, mentioning only
    this chat session's client — never another device."""
    assert isinstance(result, str)
    assert "not connected" in result or "reconnected" in result
    assert "chat session" in result


@pytest.mark.asyncio
async def test_dispatch_guard_rejects_context_device_mismatch(monkeypatch):
    """T012: ctx.device != bound device -> tool error result, no dispatch."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_a = uuid4()
    device_b = uuid4()
    dispatches: list[dict] = []

    async def _capture_dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", _capture_dispatch)

    try:
        tool_b = await _bound_tool_for_device(
            store, user_id=user_id, device_id=device_b, session_id="session-b"
        )

        with tool_execution_context(
            conversation_id="conversation-1",
            user_id=str(user_id),
            agent_key="chat",
            device_id=str(device_a),
        ):
            result = await tool_b.coroutine()

        _assert_graceful_tool_error(result)
        assert dispatches == []
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_dispatch_guard_rejects_missing_context_device(monkeypatch):
    """T012/D3: missing ctx.device_id must NOT fall back to the bound device."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_b = uuid4()
    dispatches: list[dict] = []

    async def _capture_dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", _capture_dispatch)

    try:
        tool_b = await _bound_tool_for_device(
            store, user_id=user_id, device_id=device_b, session_id="session-b"
        )

        # No tool_execution_context at all: the stale-state case D3 hid.
        result = await tool_b.coroutine()

        _assert_graceful_tool_error(result)
        assert dispatches == []
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_disconnected_client_returns_graceful_tool_error(monkeypatch):
    """T012/FR-4: device disconnects after binding -> error result, turn lives."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_b = uuid4()
    dispatches: list[dict] = []

    async def _capture_dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "ok"}

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", _capture_dispatch)

    try:
        tool_b = await _bound_tool_for_device(
            store, user_id=user_id, device_id=device_b, session_id="session-b"
        )
        await store.delete_session(device_b)

        with tool_execution_context(
            conversation_id="conversation-1",
            user_id=str(user_id),
            agent_key="chat",
            device_id=str(device_b),
        ):
            result = await tool_b.coroutine()

        _assert_graceful_tool_error(result)
        assert dispatches == []
    finally:
        _reset_runtime_state()


@pytest.mark.asyncio
async def test_dispatch_race_disconnect_returns_graceful_tool_error(monkeypatch):
    """T005: client vanishes between the guard and the dispatch call."""
    _reset_runtime_state()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_b = uuid4()

    async def _disconnect_race(**kwargs):
        raise RuntimeError("Client device is not connected for this user.")

    monkeypatch.setattr(ClientDeviceService, "dispatch_tool_call", _disconnect_race)

    try:
        tool_b = await _bound_tool_for_device(
            store, user_id=user_id, device_id=device_b, session_id="session-b"
        )

        with tool_execution_context(
            conversation_id="conversation-1",
            user_id=str(user_id),
            agent_key="chat",
            device_id=str(device_b),
        ):
            result = await tool_b.coroutine()

        _assert_graceful_tool_error(result)
    finally:
        _reset_runtime_state()
