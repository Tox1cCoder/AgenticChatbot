from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.agents.base_agent import BaseAgent
from app.ai.client_runtime_tools import _CLIENT_TOOL_CACHE, get_client_runtime_tools
from app.ai.client_tool_catalog import get_client_tool_catalog, reset_all_client_catalogs
from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
from app.ai.graph import MultiAgentWorkflow
from app.ai.hitl_config import requires_human_approval
from app.ai.mcp_tool_catalog import ToolReference
from app.ai.schemas import AgentType
from app.core.config import settings
from app.core.exceptions.http import CustomHTTPException
from app.models.hitl_interrupt import HITLInterruptStatus
from app.schemas.runtime_protocol import ToolDispatchRequest
from app.services import client_runtime_store as runtime_store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    InMemoryClientRuntimeStore,
    reset_client_runtime_store,
)
from app.services.message_service import MessageService


class _BindingTestAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "binding-test"

    def _get_base_system_prompt(self) -> str:
        return "binding-test"


@pytest.mark.asyncio
async def test_client_runtime_tools_are_scoped_per_user_and_device():
    reset_client_runtime_store()
    _CLIENT_TOOL_CACHE.clear()
    reset_all_client_catalogs()

    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_a = uuid4()
    user_b = uuid4()
    device_a = uuid4()
    device_b = uuid4()

    await store.put_session(
        DeviceSessionRecord(
            device_id=device_a,
            session_id="session-a",
            user_id=user_a,
            tool_catalog={
                "tools": [
                    {
                        "name": "start_process",
                        "origin": "mcp",
                        "server_name": "desktop_commander",
                        "qualified_id": "desktop_commander::start_process",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            tool_catalog_version=1,
        )
    )
    await store.put_session(
        DeviceSessionRecord(
            device_id=device_b,
            session_id="session-b",
            user_id=user_b,
            tool_catalog={
                "tools": [
                    {
                        "name": "get_current_time",
                        "origin": "mcp",
                        "server_name": "time_server",
                        "qualified_id": "time_server::get_current_time",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            tool_catalog_version=1,
        )
    )

    try:
        tools_a = get_client_runtime_tools(user_id=str(user_a), device_id=str(device_a))
        tools_b = get_client_runtime_tools(user_id=str(user_b), device_id=str(device_b))
        wrong_user_tools = get_client_runtime_tools(user_id=str(user_a), device_id=str(device_b))

        assert [tool.name for tool in tools_a] == ["client__desktop_commander__start_process"]
        assert [tool.name for tool in tools_b] == ["client__time_server__get_current_time"]
        assert wrong_user_tools == []
        assert tools_a[0].metadata["session_id"] == "session-a"
        assert tools_a[0].metadata["catalog_version"] == 1
        assert tools_a[0].metadata["tool_instance_id"]
    finally:
        _CLIENT_TOOL_CACHE.clear()
        reset_all_client_catalogs()
        reset_client_runtime_store()


@pytest.mark.asyncio
async def test_client_runtime_tools_ignore_non_mcp_catalog_entries():
    reset_client_runtime_store()
    _CLIENT_TOOL_CACHE.clear()
    reset_all_client_catalogs()

    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()

    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id="session-mixed",
            user_id=user_id,
            tool_catalog={
                "tools": [
                    {
                        "name": "start_process",
                        "origin": "mcp",
                        "server_name": "desktop_commander",
                        "qualified_id": "desktop_commander::start_process",
                        "input_schema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "filesystem_read_text",
                        "origin": "native",
                        "qualified_id": "native::filesystem_read_text",
                        "input_schema": {"type": "object", "properties": {}},
                    },
                ]
            },
            tool_catalog_version=2,
        )
    )

    try:
        runtime_tools = get_client_runtime_tools(user_id=str(user_id), device_id=str(device_id))
        search_catalog = get_client_tool_catalog(str(device_id), str(user_id))

        assert [tool.name for tool in runtime_tools] == ["client__desktop_commander__start_process"]
        assert [tool.tool_name for tool in search_catalog.list_all()] == [
            "client__desktop_commander__start_process"
        ]
    finally:
        _CLIENT_TOOL_CACHE.clear()
        reset_all_client_catalogs()
        reset_client_runtime_store()


def test_deferred_binding_only_includes_loaded_client_tools(monkeypatch):
    agent = _BindingTestAgent(agent_config_key="chat")

    server_tool = SimpleNamespace(name="tool_search")
    loaded_client_tool = SimpleNamespace(name="client__time_server__get_current_time")
    unloaded_client_tool = SimpleNamespace(name="client__desktop_commander__start_process")

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.build_deferred_tool_list",
        lambda **kwargs: [server_tool],
    )
    monkeypatch.setattr(
        agent,
        "_get_client_runtime_tools",
        lambda **kwargs: [loaded_client_tool, unloaded_client_tool],
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_active_client_runtime_session",
        lambda **kwargs: SimpleNamespace(session_id="session-a"),
    )

    class _DeferredStateStub:
        def get_loaded_client_tools(
            self, conversation_id: str, agent_key: str, device_id=None, session_id=None
        ):
            assert session_id == "session-a"
            all_tools = [
                SimpleNamespace(
                    tool_name="client__time_server__get_current_time",
                    device_id="device-a",
                ),
                SimpleNamespace(
                    tool_name="client__desktop_commander__start_process",
                    device_id="device-b",
                ),
            ]
            if device_id:
                return [t for t in all_tools if t.device_id == str(device_id)]
            return all_tools

    monkeypatch.setattr(
        "app.ai.deferred_tool_state.get_deferred_tool_state",
        lambda: _DeferredStateStub(),
    )

    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        user_id="user-1",
        device_id="device-a",
    )

    assert [tool.name for tool in tools] == [
        "tool_search",
        "client__time_server__get_current_time",
    ]


def test_client_tools_follow_explicit_hitl_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "enable_human_in_the_loop", True)
    monkeypatch.setattr(settings, "hitl_tools_require_approval", [])

    assert requires_human_approval(["client__desktop_commander__start_process"]) is False
    assert requires_human_approval(["server_only_tool"]) is False

    monkeypatch.setattr(
        settings,
        "hitl_tools_require_approval",
        ["client__desktop_commander__start_process"],
    )

    assert requires_human_approval(["client__desktop_commander__start_process"]) is True


def test_interrupt_resume_rejects_device_mismatch():
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_device_id = uuid4()

    interrupt_record = SimpleNamespace(
        conversation_id=conversation_id,
        thread_id="thread-1",
        device_id=interrupt_device_id,
        status=HITLInterruptStatus.PENDING,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(
            validate_conversation_access=lambda *_args, **_kwargs: None
        ),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
        hitl_interrupt_repository=SimpleNamespace(
            get_by_id=lambda _interrupt_id: interrupt_record,
            try_transition_to_resolving=lambda **_kwargs: True,
        ),
    )

    with pytest.raises(CustomHTTPException) as exc_info:
        service._validate_and_claim_interrupt_resume(
            thread_id="thread-1",
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id="interrupt-1",
            device_id=uuid4(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "INTERRUPT_DEVICE_MISMATCH"


@pytest.mark.asyncio
async def test_refresh_tool_map_after_search_uses_active_session_scope(monkeypatch):
    from app.ai.tool_execution import _refresh_tool_map_after_search

    state_calls = []
    device_id = str(uuid4())

    class _DeferredStateStub:
        def get_loaded_client_tools(
            self, conversation_id, agent_key, device_id=None, session_id=None
        ):
            state_calls.append(
                {
                    "conversation_id": conversation_id,
                    "agent_key": agent_key,
                    "device_id": device_id,
                    "session_id": session_id,
                }
            )
            return [SimpleNamespace(tool_name="client__time_server__get_current_time")]

    async def _get_global_mcp_manager():
        return None

    monkeypatch.setattr(
        "app.ai.mcp_registry.get_global_mcp_manager",
        _get_global_mcp_manager,
    )
    monkeypatch.setattr(
        "app.ai.deferred_tool_binding.get_deferred_tools_for_binding",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "app.ai.deferred_tool_state.get_deferred_tool_state",
        lambda: _DeferredStateStub(),
    )
    monkeypatch.setattr(
        "app.ai.tool_execution.get_active_client_runtime_session",
        lambda **kwargs: SimpleNamespace(session_id="session-a"),
    )
    monkeypatch.setattr(
        "app.ai.client_runtime_tools.get_client_runtime_tools",
        lambda **kwargs: [SimpleNamespace(name="client__time_server__get_current_time")],
    )

    tool_map = {}
    await _refresh_tool_map_after_search(
        tool_map=tool_map,
        agent=SimpleNamespace(agent_config_key="chat"),
        conversation_id="conversation-1",
        user_id="user-1",
        device_id=device_id,
    )

    assert state_calls == [
        {
            "conversation_id": "conversation-1",
            "agent_key": "chat",
            "device_id": device_id,
            "session_id": "session-a",
        }
    ]
    assert "client__time_server__get_current_time" in tool_map


@pytest.mark.asyncio
async def test_cleanup_stale_sessions_fails_pending_requests_without_waiting_for_timeout():
    store = InMemoryClientRuntimeStore()
    stale_session = DeviceSessionRecord(
        device_id=uuid4(),
        session_id="stale-session",
        user_id=uuid4(),
        last_heartbeat=datetime.now(timezone.utc)
        - timedelta(seconds=settings.client_runtime_heartbeat_interval_seconds * 3),
    )
    await store.put_session(stale_session)

    dispatch_task = asyncio.create_task(
        store.dispatch_request(
            stale_session,
            ToolDispatchRequest(
                request_id="req-1",
                tool_name="start_process",
                qualified_tool_id="desktop_commander::start_process",
                arguments={"command": "echo hi"},
                timeout_seconds=5,
            ),
            timeout_seconds=5,
        )
    )

    await asyncio.sleep(0)
    cleaned_count = await store.cleanup_stale_sessions()
    result = await dispatch_task

    assert cleaned_count == 1
    assert result["success"] is False
    assert result["error_context"]["code"] == "DEVICE_DISCONNECTED"


@pytest.mark.asyncio
async def test_device_cleanup_still_updates_db_when_runtime_store_fails(monkeypatch):
    class _ExplodingStore:
        async def cleanup_stale_sessions(self) -> int:
            raise RuntimeError("MISCONF")

    monkeypatch.setattr(
        "app.services.client_device_service.get_client_runtime_store",
        lambda: _ExplodingStore(),
    )

    service = ClientDeviceService.__new__(ClientDeviceService)
    service.session = None
    service.repository = SimpleNamespace(mark_stale_devices_offline=lambda _timeout_seconds: 2)

    cleaned_count = await service.cleanup_stale_sessions()

    assert cleaned_count == 2


@pytest.mark.asyncio
async def test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope():
    reset_client_runtime_store()
    reset_deferred_tool_state()
    _CLIENT_TOOL_CACHE.clear()
    reset_all_client_catalogs()

    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()

    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id="session-a",
            user_id=user_id,
            tool_catalog={
                "tools": [
                    {
                        "name": "start_process",
                        "origin": "mcp",
                        "server_name": "desktop_commander",
                        "qualified_id": "desktop_commander::start_process",
                        "tool_instance_id": "instance-123",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            tool_catalog_version=2,
        )
    )

    try:
        deferred_state = get_deferred_tool_state()
        deferred_state.autoload(
            conversation_id="conversation-1",
            agent_key="chat",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )
        deferred_state.autoload_client_tools(
            conversation_id="conversation-1",
            agent_key="chat",
            references=[
                SimpleNamespace(
                    tool_name="client__desktop_commander__start_process",
                    server_name="desktop_commander",
                    device_id=str(device_id),
                    session_id="session-a",
                    catalog_version=2,
                    tool_instance_id="instance-123",
                )
            ],
            device_id=str(device_id),
            session_id="session-a",
            user_id=str(user_id),
        )

        snapshot = deferred_state.snapshot(
            conversation_id="conversation-1",
            agent_key="chat",
            device_id=str(device_id),
            session_id="session-a",
        )

        reset_deferred_tool_state()

        restored = get_deferred_tool_state().restore(
            conversation_id="conversation-1",
            agent_key="chat",
            snapshot=snapshot,
            device_id=str(device_id),
            session_id="session-a",
            user_id=str(user_id),
        )

        assert restored == {"server_tools": 1, "client_tools": 1}
        loaded_server_refs = get_deferred_tool_state().get_loaded("conversation-1", "chat")
        assert [ref.call_name for ref in loaded_server_refs] == ["brave__search"]
        assert (
            get_deferred_tool_state().get_server_for_loaded_tool(
                "conversation-1",
                "chat",
                "brave__search",
            )
            == "brave"
        )
        assert get_deferred_tool_state().get_all_loaded_tool_names(
            "conversation-1",
            "chat",
            device_id=str(device_id),
            session_id="session-a",
        ) == ["brave__search", "client__desktop_commander__start_process"]
    finally:
        reset_deferred_tool_state()
        _CLIENT_TOOL_CACHE.clear()
        reset_all_client_catalogs()
        reset_client_runtime_store()


@pytest.mark.asyncio
async def test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context(monkeypatch):
    reset_client_runtime_store()
    reset_deferred_tool_state()
    _CLIENT_TOOL_CACHE.clear()
    reset_all_client_catalogs()

    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()

    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id="session-a",
            user_id=user_id,
            tool_catalog={
                "tools": [
                    {
                        "name": "start_process",
                        "origin": "mcp",
                        "server_name": "desktop_commander",
                        "qualified_id": "desktop_commander::start_process",
                        "tool_instance_id": "instance-123",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            tool_catalog_version=2,
        )
    )

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    agent = SimpleNamespace(agent_config_key="chat", agent_id="chat_agent")
    state = {
        "conversation_id": "conversation-1",
        "user_id": str(user_id),
        "device_id": str(device_id),
        "context": {},
    }

    async def _fake_execute_tool_calls(**kwargs):
        deferred_state = get_deferred_tool_state()
        deferred_state.autoload(
            conversation_id="conversation-1",
            agent_key="chat",
            references=[
                ToolReference(
                    tool_name="search",
                    server_name="brave",
                    call_name="brave__search",
                )
            ],
        )
        deferred_state.autoload_client_tools(
            conversation_id="conversation-1",
            agent_key="chat",
            references=[
                SimpleNamespace(
                    tool_name="client__desktop_commander__start_process",
                    server_name="desktop_commander",
                    device_id=str(device_id),
                    session_id="session-a",
                    catalog_version=2,
                    tool_instance_id="instance-123",
                )
            ],
            device_id=str(device_id),
            session_id="session-a",
            user_id=str(user_id),
        )
        return ([], [], [])

    monkeypatch.setattr("app.ai.graph.execute_tool_calls", _fake_execute_tool_calls)

    await workflow._execute_agent_tool_calls(
        state=state,
        agent=agent,
        tool_calls=[{"id": "tool-1", "name": "tool_search", "args": {"query": "search"}}],
        tool_map={"tool_search": SimpleNamespace(name="tool_search")},
    )

    snapshot = state["context"]["deferred_tool_snapshot"]
    assert [tool["call_name"] for tool in snapshot["server_tools"]] == ["brave__search"]
    assert [tool["tool_name"] for tool in snapshot["client_tools"]] == [
        "client__desktop_commander__start_process"
    ]

    reset_deferred_tool_state()

    restored = workflow._hydrate_deferred_tool_snapshot_from_state(
        state,
        agent=agent,
    )

    assert restored is True
    loaded_server_refs = get_deferred_tool_state().get_loaded("conversation-1", "chat")
    assert [ref.call_name for ref in loaded_server_refs] == ["brave__search"]
    assert get_deferred_tool_state().get_all_loaded_tool_names(
        "conversation-1",
        "chat",
        device_id=str(device_id),
        session_id="session-a",
    ) == ["brave__search", "client__desktop_commander__start_process"]
