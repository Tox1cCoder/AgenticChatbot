"""
Regression tests for multi-sidecar hardening.

Covers:
- Same conversation used from two sidecars with overlapping client tool names
- Reconnect with new session invalidating stale tool_instance_id
- HITL resume rejection when session or catalog version changes
- Deferred autoload isolation across two devices in one conversation
- device_id is None path binds zero client tools when bridge is enabled
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.client_runtime_tools import make_tool_instance_id
from app.ai.deferred_tool_state import (
    ClientToolScope,
    DeferredToolState,
)
from app.ai.schemas import AgentType
from app.core.config import settings
from app.core.exceptions.http import CustomHTTPException
from app.models.hitl_interrupt import HITLInterruptStatus
from app.schemas.runtime_protocol import ToolDispatchRequest
from app.schemas.workflow import InterruptDecision, InterruptDecisionType
from app.services.message_service import MessageService


def _make_instance_id(device_id: str, session_id: str, qid: str, version: int) -> str:
    composite = f"{device_id}:{session_id}:{qid}:{version}"
    return hashlib.sha256(composite.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Same conversation, two sidecars, overlapping tool names
# ---------------------------------------------------------------------------


class TestOverlappingToolNames:
    """Two sidecars publish the same tool name in the same conversation.
    Each device must keep its own pool without cross-contamination."""

    def test_client_tool_scope_isolation(self):
        state = DeferredToolState()
        conv_id = "conv-1"
        agent_key = "chat"
        device_a = "device-aaa"
        device_b = "device-bbb"
        session_a = "session-aaa"
        session_b = "session-bbb"

        ref_a = SimpleNamespace(
            tool_name="client__shell_execute",
            server_name="native",
            device_id=device_a,
            tool_instance_id=_make_instance_id(device_a, session_a, "native::shell_execute", 1),
        )
        ref_b = SimpleNamespace(
            tool_name="client__shell_execute",
            server_name="native",
            device_id=device_b,
            tool_instance_id=_make_instance_id(device_b, session_b, "native::shell_execute", 1),
        )

        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[ref_a],
            device_id=device_a,
            session_id=session_a,
        )
        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[ref_b],
            device_id=device_b,
            session_id=session_b,
        )

        tools_a = state.get_loaded_client_tools(
            conv_id, agent_key, device_id=device_a, session_id=session_a
        )
        tools_b = state.get_loaded_client_tools(
            conv_id, agent_key, device_id=device_b, session_id=session_b
        )

        assert len(tools_a) == 1
        assert len(tools_b) == 1
        assert tools_a[0].device_id == device_a
        assert tools_b[0].device_id == device_b
        assert tools_a[0].tool_instance_id != tools_b[0].tool_instance_id

    def test_broad_lookup_returns_both_devices(self):
        """When no device_id filter is applied, both scopes are returned."""
        state = DeferredToolState()
        conv_id = "conv-2"

        for dev, sess in [("dev-x", "sess-x"), ("dev-y", "sess-y")]:
            ref = SimpleNamespace(
                tool_name="client__shell_execute",
                server_name="native",
                device_id=dev,
                tool_instance_id="",
            )
            state.autoload_client_tools(
                conversation_id=conv_id,
                agent_key="chat",
                references=[ref],
                device_id=dev,
                session_id=sess,
            )

        all_tools = state.get_loaded_client_tools(conv_id, "chat")
        assert len(all_tools) == 2
        device_ids = {t.device_id for t in all_tools}
        assert device_ids == {"dev-x", "dev-y"}


# ---------------------------------------------------------------------------
# 2. Reconnect with new session invalidates stale tool_instance_id
# ---------------------------------------------------------------------------


class TestReconnectInvalidatesInstanceId:
    def test_tool_instance_id_changes_on_new_session(self):
        """Same device+tool but new session produces a different tool_instance_id."""
        device = "device-reconnect"
        qid = "native::shell_execute"
        old_id = make_tool_instance_id(device, "session-old", qid, 1)
        new_id = make_tool_instance_id(device, "session-new", qid, 1)
        assert old_id != new_id

    def test_tool_instance_id_changes_on_catalog_version_bump(self):
        """Same device+session+tool but catalog version bump produces new ID."""
        device = "device-bump"
        session = "session-bump"
        qid = "native::shell_execute"
        v1 = make_tool_instance_id(device, session, qid, 1)
        v2 = make_tool_instance_id(device, session, qid, 2)
        assert v1 != v2

    def test_sidecar_rejects_stale_session_request(self):
        """_validate_tool_request rejects when expected_session_id mismatches."""
        from client_backend.services.runtime_bridge import RuntimeBridgeService

        bridge = RuntimeBridgeService.__new__(RuntimeBridgeService)
        bridge._session_id = "session-new"
        bridge._tool_catalog_version = 2
        bridge._current_tool_catalog = {
            "native::shell_execute": {
                "name": "shell_execute",
                "qualified_id": "native::shell_execute",
                "tool_instance_id": "abc123",
            }
        }

        request = ToolDispatchRequest(
            request_id="req-1",
            tool_name="shell_execute",
            qualified_tool_id="native::shell_execute",
            arguments={},
            expected_session_id="session-old",
            expected_catalog_version=2,
            tool_instance_id="abc123",
        )

        error = bridge._validate_tool_request(request)
        assert error is not None
        assert "Session mismatch" in error

    def test_sidecar_rejects_stale_catalog_version(self):
        """_validate_tool_request rejects when catalog version is stale."""
        from client_backend.services.runtime_bridge import RuntimeBridgeService

        bridge = RuntimeBridgeService.__new__(RuntimeBridgeService)
        bridge._session_id = "session-current"
        bridge._tool_catalog_version = 3
        bridge._current_tool_catalog = {
            "native::shell_execute": {
                "name": "shell_execute",
                "qualified_id": "native::shell_execute",
                "tool_instance_id": "def456",
            }
        }

        request = ToolDispatchRequest(
            request_id="req-2",
            tool_name="shell_execute",
            qualified_tool_id="native::shell_execute",
            arguments={},
            expected_session_id="session-current",
            expected_catalog_version=2,
        )

        error = bridge._validate_tool_request(request)
        assert error is not None
        assert "Catalog version mismatch" in error

    def test_sidecar_rejects_unknown_tool(self):
        """_validate_tool_request rejects a tool not in the current catalog."""
        from client_backend.services.runtime_bridge import RuntimeBridgeService

        bridge = RuntimeBridgeService.__new__(RuntimeBridgeService)
        bridge._session_id = "session-1"
        bridge._tool_catalog_version = 1
        bridge._current_tool_catalog = {}

        request = ToolDispatchRequest(
            request_id="req-3",
            tool_name="ghost_tool",
            qualified_tool_id="native::ghost_tool",
            arguments={},
            expected_session_id="session-1",
            expected_catalog_version=1,
        )

        error = bridge._validate_tool_request(request)
        assert error is not None
        assert "Unknown tool" in error

    def test_sidecar_rejects_mismatched_tool_instance_id(self):
        """_validate_tool_request rejects when tool_instance_id doesn't match."""
        from client_backend.services.runtime_bridge import RuntimeBridgeService

        bridge = RuntimeBridgeService.__new__(RuntimeBridgeService)
        bridge._session_id = "session-1"
        bridge._tool_catalog_version = 1
        bridge._current_tool_catalog = {
            "native::shell_execute": {
                "name": "shell_execute",
                "qualified_id": "native::shell_execute",
                "tool_instance_id": "current_id_abc",
            }
        }

        request = ToolDispatchRequest(
            request_id="req-4",
            tool_name="shell_execute",
            qualified_tool_id="native::shell_execute",
            arguments={},
            expected_session_id="session-1",
            expected_catalog_version=1,
            tool_instance_id="stale_id_xyz",
        )

        error = bridge._validate_tool_request(request)
        assert error is not None
        assert "tool_instance_id mismatch" in error

    def test_sidecar_accepts_valid_request(self):
        """_validate_tool_request returns None for a fully valid request."""
        from client_backend.services.runtime_bridge import RuntimeBridgeService

        instance_id = _make_instance_id("dev-1", "sess-1", "native::shell_execute", 1)
        bridge = RuntimeBridgeService.__new__(RuntimeBridgeService)
        bridge._session_id = "sess-1"
        bridge._tool_catalog_version = 1
        bridge._current_tool_catalog = {
            "native::shell_execute": {
                "name": "shell_execute",
                "qualified_id": "native::shell_execute",
                "tool_instance_id": instance_id,
            }
        }

        request = ToolDispatchRequest(
            request_id="req-5",
            tool_name="shell_execute",
            qualified_tool_id="native::shell_execute",
            arguments={},
            expected_session_id="sess-1",
            expected_catalog_version=1,
            tool_instance_id=instance_id,
        )

        error = bridge._validate_tool_request(request)
        assert error is None

    def test_device_runtime_connect_expires_stale_interrupts_without_warning(
        self, monkeypatch, caplog
    ):
        import importlib.util

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from pathlib import Path

        device_id = uuid4()
        session_id = "session-new"
        fake_db = object()
        expired_calls = []

        module_spec = importlib.util.spec_from_file_location(
            "device_runtime_api_under_test",
            Path(__file__).resolve().parents[1] / "app" / "api" / "device_runtime.py",
        )
        assert module_spec is not None and module_spec.loader is not None
        device_runtime = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(device_runtime)

        class _StubClientDeviceService:
            def __init__(self, db):
                assert db is fake_db
                self.repository = SimpleNamespace(
                    get_by_id=lambda _id: SimpleNamespace(id=_id, user_id=uuid4())
                )

            async def start_session(self, device_id, session_id):
                return SimpleNamespace(
                    device_id=device_id,
                    session_id=session_id,
                    user_id=uuid4(),
                )

        class _StubInterruptRepository:
            def __init__(self, session_factory):
                self._session_factory = session_factory

            def expire_stale_client_tool_interrupts(self, *, device_id, current_session_id):
                expired_calls.append(
                    {
                        "device_id": device_id,
                        "current_session_id": current_session_id,
                        "session_factory": self._session_factory,
                    }
                )
                return 1

        async def _fake_handle_connection(self):
            await self.websocket.send_json({"type": "ack"})
            await self.websocket.close()

        monkeypatch.setattr(device_runtime, "ClientDeviceService", _StubClientDeviceService)
        monkeypatch.setattr(
            "app.repositories.hitl_interrupt.HITLInterruptRepository",
            _StubInterruptRepository,
        )
        monkeypatch.setattr(
            device_runtime.DeviceRuntimeGateway,
            "handle_connection",
            _fake_handle_connection,
        )
        monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)

        app = FastAPI()
        app.include_router(device_runtime.router)
        app.dependency_overrides[device_runtime.get_db] = lambda: fake_db

        with caplog.at_level("WARNING", logger="app.api.device_runtime"):
            with TestClient(app) as client:
                with client.websocket_connect(
                    f"/device-runtime/{device_id}/connect?session_id={session_id}"
                ) as websocket:
                    assert websocket.receive_json() == {"type": "ack"}

        assert len(expired_calls) == 1
        assert expired_calls[0]["device_id"] == device_id
        assert expired_calls[0]["current_session_id"] == session_id
        assert callable(expired_calls[0]["session_factory"])
        assert not any(
            "Interrupt invalidation failed on connect" in record.getMessage()
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# 3. HITL resume rejection when device changes
# ---------------------------------------------------------------------------


class TestHITLResumeSessionValidation:
    def test_interrupt_resume_rejects_when_device_changed(self):
        """If the interrupt was created for device-A but resume comes from
        device-B, it must be rejected with INTERRUPT_DEVICE_MISMATCH."""
        conversation_id = uuid4()
        user_id = uuid4()
        device_id = uuid4()

        interrupt_record = SimpleNamespace(
            conversation_id=conversation_id,
            thread_id="thread-1",
            device_id=device_id,
            status=HITLInterruptStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            session_id="session-old",
            catalog_version=1,
            tool_instance_id="old_instance_id",
        )

        service = MessageService(
            message_repository=SimpleNamespace(),
            conversation_validation_utils=SimpleNamespace(
                validate_conversation_access=lambda *_a, **_kw: None
            ),
            message_validation_utils=SimpleNamespace(),
            ai_service=SimpleNamespace(),
            hitl_interrupt_repository=SimpleNamespace(
                get_by_id=lambda _id: interrupt_record,
                try_transition_to_resolving=lambda **_kw: True,
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

    def test_interrupt_resume_rejects_when_session_changed(self, monkeypatch):
        conversation_id = uuid4()
        user_id = uuid4()
        device_id = uuid4()

        interrupt_record = SimpleNamespace(
            conversation_id=conversation_id,
            thread_id="thread-1",
            device_id=device_id,
            user_id=user_id,
            status=HITLInterruptStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            session_id="session-old",
            catalog_version=1,
            tool_instance_id="instance-old",
            interrupt_metadata_json={
                "tool_provenance": {
                    "tool-call-1": {
                        "device_id": str(device_id),
                        "tool_origin": "client_native",
                        "qualified_tool_id": "native::shell_execute",
                        "session_id": "session-old",
                        "catalog_version": 1,
                        "tool_instance_id": "instance-old",
                    }
                }
            },
        )

        service = MessageService(
            message_repository=SimpleNamespace(),
            conversation_validation_utils=SimpleNamespace(
                validate_conversation_access=lambda *_a, **_kw: None
            ),
            message_validation_utils=SimpleNamespace(),
            ai_service=SimpleNamespace(),
            hitl_interrupt_repository=SimpleNamespace(
                get_by_id=lambda _id: interrupt_record,
                try_transition_to_resolving=lambda **_kw: True,
            ),
        )

        monkeypatch.setattr(
            "app.services.client_device_service.ClientDeviceService.lookup_active_session",
            lambda _device_id: SimpleNamespace(
                user_id=user_id,
                session_id="session-new",
                tool_catalog_version=1,
                tool_catalog={
                    "tools": [
                        {
                            "qualified_id": "native::shell_execute",
                            "tool_instance_id": "instance-new",
                        }
                    ]
                },
            ),
        )

        with pytest.raises(CustomHTTPException) as exc_info:
            service._validate_and_claim_interrupt_resume(
                thread_id="thread-1",
                conversation_id=conversation_id,
                user_id=user_id,
                interrupt_id="interrupt-1",
                device_id=device_id,
                decisions=[
                    InterruptDecision(
                        type=InterruptDecisionType.APPROVE,
                        task_id="tool-call-1",
                        action="client__shell_execute",
                    )
                ],
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.error_code == "INTERRUPT_SESSION_MISMATCH"

    def test_interrupt_resume_rejects_when_catalog_changes(self, monkeypatch):
        conversation_id = uuid4()
        user_id = uuid4()
        device_id = uuid4()

        interrupt_record = SimpleNamespace(
            conversation_id=conversation_id,
            thread_id="thread-1",
            device_id=device_id,
            user_id=user_id,
            status=HITLInterruptStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            session_id="session-1",
            catalog_version=1,
            tool_instance_id="instance-v1",
            interrupt_metadata_json={
                "tool_provenance": {
                    "tool-call-1": {
                        "device_id": str(device_id),
                        "tool_origin": "client_native",
                        "qualified_tool_id": "native::shell_execute",
                        "session_id": "session-1",
                        "catalog_version": 1,
                        "tool_instance_id": "instance-v1",
                    }
                }
            },
        )

        service = MessageService(
            message_repository=SimpleNamespace(),
            conversation_validation_utils=SimpleNamespace(
                validate_conversation_access=lambda *_a, **_kw: None
            ),
            message_validation_utils=SimpleNamespace(),
            ai_service=SimpleNamespace(),
            hitl_interrupt_repository=SimpleNamespace(
                get_by_id=lambda _id: interrupt_record,
                try_transition_to_resolving=lambda **_kw: True,
            ),
        )

        monkeypatch.setattr(
            "app.services.client_device_service.ClientDeviceService.lookup_active_session",
            lambda _device_id: SimpleNamespace(
                user_id=user_id,
                session_id="session-1",
                tool_catalog_version=2,
                tool_catalog={
                    "tools": [
                        {
                            "qualified_id": "native::shell_execute",
                            "tool_instance_id": "instance-v2",
                        }
                    ]
                },
            ),
        )

        with pytest.raises(CustomHTTPException) as exc_info:
            service._validate_and_claim_interrupt_resume(
                thread_id="thread-1",
                conversation_id=conversation_id,
                user_id=user_id,
                interrupt_id="interrupt-1",
                device_id=device_id,
                decisions=[
                    InterruptDecision(
                        type=InterruptDecisionType.APPROVE,
                        task_id="tool-call-1",
                        action="client__shell_execute",
                    )
                ],
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.error_code == "INTERRUPT_CATALOG_MISMATCH"


class TestInterruptScopePersistence:
    def test_derive_interrupt_execution_scope_ignores_non_client_provenance_entries(self):
        scope = MessageService._derive_interrupt_execution_scope(
            {
                "device_id": "device-1",
                "tool_provenance": {
                    "server-call": {
                        "device_id": "device-1",
                    },
                    "client-call": {
                        "device_id": "device-1",
                        "tool_origin": "client_native",
                        "qualified_tool_id": "native::shell_execute",
                        "session_id": "session-1",
                        "catalog_version": 4,
                        "tool_instance_id": "inst-4",
                    },
                },
            }
        )

        assert scope == {
            "session_id": "session-1",
            "catalog_version": 4,
            "tool_instance_id": "inst-4",
        }


# ---------------------------------------------------------------------------
# 4. Deferred autoload isolation across two devices in one conversation
# ---------------------------------------------------------------------------


class TestDeferredAutoloadIsolation:
    def test_two_devices_independent_lru_pools(self):
        """Each device gets its own LRU pool; filling one does not evict
        from the other."""
        state = DeferredToolState()
        conv_id = "conv-lru"
        agent_key = "chat"
        max_tools = 2

        for i in range(2):
            ref = SimpleNamespace(
                tool_name=f"client__tool_a{i}",
                server_name="native",
                device_id="dev-a",
                tool_instance_id=f"inst-a{i}",
            )
            state.autoload_client_tools(
                conversation_id=conv_id,
                agent_key=agent_key,
                references=[ref],
                device_id="dev-a",
                session_id="sess-a",
                max_tools=max_tools,
            )

        for i in range(2):
            ref = SimpleNamespace(
                tool_name=f"client__tool_b{i}",
                server_name="native",
                device_id="dev-b",
                tool_instance_id=f"inst-b{i}",
            )
            state.autoload_client_tools(
                conversation_id=conv_id,
                agent_key=agent_key,
                references=[ref],
                device_id="dev-b",
                session_id="sess-b",
                max_tools=max_tools,
            )

        tools_a = state.get_loaded_client_tools(
            conv_id, agent_key, device_id="dev-a", session_id="sess-a"
        )
        tools_b = state.get_loaded_client_tools(
            conv_id, agent_key, device_id="dev-b", session_id="sess-b"
        )

        assert len(tools_a) == 2
        assert len(tools_b) == 2
        assert all(t.device_id == "dev-a" for t in tools_a)
        assert all(t.device_id == "dev-b" for t in tools_b)

    def test_overflow_device_a_does_not_evict_device_b(self):
        """Overflowing device A pool evicts from A, not B."""
        state = DeferredToolState()
        conv_id = "conv-evict"
        agent_key = "chat"
        max_tools = 1

        ref_a1 = SimpleNamespace(
            tool_name="client__old_tool",
            server_name="native",
            device_id="dev-a",
            tool_instance_id="inst-old",
        )
        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[ref_a1],
            device_id="dev-a",
            session_id="sess-a",
            max_tools=max_tools,
        )

        ref_b = SimpleNamespace(
            tool_name="client__b_tool",
            server_name="native",
            device_id="dev-b",
            tool_instance_id="inst-b",
        )
        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[ref_b],
            device_id="dev-b",
            session_id="sess-b",
            max_tools=max_tools,
        )

        # Overflow device A
        ref_a2 = SimpleNamespace(
            tool_name="client__new_tool",
            server_name="native",
            device_id="dev-a",
            tool_instance_id="inst-new",
        )
        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[ref_a2],
            device_id="dev-a",
            session_id="sess-a",
            max_tools=max_tools,
        )

        tools_a = state.get_loaded_client_tools(
            conv_id, agent_key, device_id="dev-a", session_id="sess-a"
        )
        tools_b = state.get_loaded_client_tools(
            conv_id, agent_key, device_id="dev-b", session_id="sess-b"
        )

        assert len(tools_a) == 1
        assert tools_a[0].tool_name == "client__new_tool"
        assert len(tools_b) == 1
        assert tools_b[0].tool_name == "client__b_tool"

    def test_device_lookup_without_session_id_uses_active_session(self, monkeypatch):
        state = DeferredToolState()
        conv_id = "conv-session-scope"
        agent_key = "chat"
        device_id = str(uuid4())

        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[
                SimpleNamespace(
                    tool_name="client__old_tool",
                    server_name="native",
                    device_id=device_id,
                    session_id="sess-old",
                    catalog_version=1,
                    tool_instance_id="inst-old",
                )
            ],
            device_id=device_id,
            session_id="sess-old",
        )
        state.autoload_client_tools(
            conversation_id=conv_id,
            agent_key=agent_key,
            references=[
                SimpleNamespace(
                    tool_name="client__new_tool",
                    server_name="native",
                    device_id=device_id,
                    session_id="sess-new",
                    catalog_version=2,
                    tool_instance_id="inst-new",
                )
            ],
            device_id=device_id,
            session_id="sess-new",
        )

        monkeypatch.setattr(
            "app.services.client_device_service.ClientDeviceService.lookup_active_session",
            lambda _device_id: SimpleNamespace(session_id="sess-new"),
        )

        tools = state.get_loaded_client_tools(
            conv_id,
            agent_key,
            device_id=device_id,
        )

        assert [tool.tool_name for tool in tools] == ["client__new_tool"]


# ---------------------------------------------------------------------------
# 5. device_id is None binds zero client tools when bridge is enabled
# ---------------------------------------------------------------------------


class TestNoDeviceBindsZeroTools:
    def test_get_tools_for_binding_no_device_id(self, monkeypatch):
        """When device_id is None and bridge is enabled, zero client tools
        are bound."""
        from app.ai.agents.base_agent import BaseAgent

        class _TestAgent(BaseAgent):
            def _init_gemini(self) -> None:
                self.gemini_client = None
                self.langchain_model = None

            @property
            def agent_type(self) -> AgentType:
                return AgentType.CHAT

            @property
            def agent_id(self) -> str:
                return "no-device-test"

            def _get_base_system_prompt(self) -> str:
                return "test"

        agent = _TestAgent(agent_config_key="chat")

        call_log = []

        def tracking_get(**kwargs):
            call_log.append(kwargs)
            return [SimpleNamespace(name="client__should_not_appear")]

        monkeypatch.setattr(agent, "_get_client_runtime_tools", tracking_get)
        monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
        monkeypatch.setattr(
            "app.ai.agents.base_agent.should_use_deferred_loading",
            lambda _: False,
        )

        tools = agent._get_tools_for_binding(
            conversation_id="conv-1",
            user_id="user-1",
            device_id=None,
        )

        assert len(call_log) == 0
        assert not any(getattr(t, "name", "").startswith("client__") for t in tools)

    def test_deferred_state_autoload_returns_empty_without_device(self):
        """autoload_client_tools returns [] when device_id is None."""
        state = DeferredToolState()
        ref = SimpleNamespace(
            tool_name="client__shell_execute",
            server_name="native",
            device_id="dev-1",
            tool_instance_id="inst-1",
        )
        result = state.autoload_client_tools(
            conversation_id="conv-1",
            agent_key="chat",
            references=[ref],
            device_id=None,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 6. ClientToolScope unit tests
# ---------------------------------------------------------------------------


class TestClientToolScope:
    def test_add_and_retrieve(self):
        scope = ClientToolScope()
        scope.add(
            "client__t1",
            "native",
            "dev-1",
            1,
            max_tools=5,
            tool_instance_id="abc",
        )
        tool = scope.get("client__t1")
        assert tool is not None
        assert tool.tool_instance_id == "abc"
        assert tool.device_id == "dev-1"

    def test_lru_eviction_within_scope(self):
        scope = ClientToolScope()
        scope.add("client__old", "native", "dev-1", 1, max_tools=1)
        scope.add("client__new", "native", "dev-1", 1, max_tools=1)
        assert "client__old" not in scope
        assert "client__new" in scope

    def test_update_existing_tool(self):
        scope = ClientToolScope()
        scope.add(
            "client__t",
            "native",
            "dev-1",
            1,
            max_tools=5,
            tool_instance_id="v1",
        )
        scope.add(
            "client__t",
            "native",
            "dev-1",
            2,
            max_tools=5,
            tool_instance_id="v2",
        )
        assert len(scope) == 1
        assert scope.get("client__t").tool_instance_id == "v2"
        assert scope.get("client__t").catalog_version == 2


# ---------------------------------------------------------------------------
# 7. make_tool_instance_id consistency between server and sidecar
# ---------------------------------------------------------------------------


class TestToolInstanceIdConsistency:
    def test_server_and_sidecar_produce_same_id(self):
        """The formula in client_runtime_tools.make_tool_instance_id must
        match the sidecar _make_tool_instance_id."""
        from client_backend.services.runtime_bridge import (
            _make_tool_instance_id,
        )

        device = "device-abc"
        session = "session-xyz"
        qid = "mcp::code_review::lint"
        version = 3

        server_id = make_tool_instance_id(device, session, qid, version)
        sidecar_id = _make_tool_instance_id(device, session, qid, version)
        assert server_id == sidecar_id

    def test_different_inputs_produce_different_ids(self):
        base = make_tool_instance_id("d", "s", "q", 1)
        assert base != make_tool_instance_id("d2", "s", "q", 1)
        assert base != make_tool_instance_id("d", "s2", "q", 1)
        assert base != make_tool_instance_id("d", "s", "q2", 1)
        assert base != make_tool_instance_id("d", "s", "q", 2)
