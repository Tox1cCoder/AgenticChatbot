from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.agents.base_agent import BaseAgent
from app.ai.client_runtime_tools import _CLIENT_TOOL_CACHE, get_client_runtime_tools
from app.ai.hitl_config import requires_human_approval
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
                        "name": "shell_execute",
                        "origin": "native",
                        "qualified_id": "native::shell_execute",
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
                        "name": "filesystem_read_text",
                        "origin": "native",
                        "qualified_id": "native::filesystem_read_text",
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

        assert [tool.name for tool in tools_a] == ["client__shell_execute"]
        assert [tool.name for tool in tools_b] == ["client__filesystem_read_text"]
        assert wrong_user_tools == []
    finally:
        _CLIENT_TOOL_CACHE.clear()
        reset_client_runtime_store()


def test_deferred_binding_only_includes_loaded_client_tools(monkeypatch):
    agent = _BindingTestAgent(agent_config_key="chat")

    server_tool = SimpleNamespace(name="tool_search")
    loaded_client_tool = SimpleNamespace(name="client__filesystem_read_text")
    unloaded_client_tool = SimpleNamespace(name="client__shell_execute")

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

    class _DeferredStateStub:
        def get_loaded_client_tools(self, conversation_id: str, agent_key: str):
            return [
                SimpleNamespace(tool_name="client__filesystem_read_text", device_id="device-a"),
                SimpleNamespace(tool_name="client__shell_execute", device_id="device-b"),
            ]

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
        "client__filesystem_read_text",
    ]


def test_client_tools_require_human_approval_by_default(monkeypatch):
    monkeypatch.setattr(settings, "enable_human_in_the_loop", True)
    monkeypatch.setattr(settings, "hitl_tools_require_approval", [])

    assert requires_human_approval(["client__shell_execute"]) is True
    assert requires_human_approval(["server_only_tool"]) is False


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
                tool_name="shell_execute",
                qualified_tool_id="native::shell_execute",
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
