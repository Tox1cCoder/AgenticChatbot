from types import SimpleNamespace

import pytest

from client_backend.schemas.runtime import ToolDispatchRequest
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


class _ServerClientStub:
    def __init__(self):
        self.base_url = "http://server.test"

    def is_authenticated(self) -> bool:
        return True


class _StubSkillExecutionEngine:
    """Records dispatch calls and returns a preconfigured envelope."""

    def __init__(self, envelope: dict):
        self._envelope = envelope
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self):
        return self

    async def execute(self, qualified_tool_id, arguments, context):
        self.calls.append((qualified_tool_id, arguments, context))
        return self._envelope


def _make_bridge() -> RuntimeBridgeService:
    return RuntimeBridgeService(server_client=_ServerClientStub())


@pytest.mark.asyncio
async def test_execute_tool_request_routes_skill_capability_to_engine(monkeypatch):
    bridge = _make_bridge()
    bridge._device_id = "device-123"
    bridge._session_id = "session-123"

    stub_engine = _StubSkillExecutionEngine({"ok": True, "result": {"items": []}})
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", stub_engine)

    result = await bridge._execute_tool_request(
        ToolDispatchRequest(
            request_id="r1",
            tool_name="list_things",
            qualified_tool_id="skill::demo::list_things",
            arguments={"filter": "all"},
            timeout_seconds=10,
        )
    )

    assert result == {"items": []}
    assert len(stub_engine.calls) == 1
    qualified_tool_id, arguments, context = stub_engine.calls[0]
    assert qualified_tool_id == "skill::demo::list_things"
    assert arguments == {"filter": "all"}
    assert context["device_id"] == "device-123"
    assert context["session_id"] == "session-123"
    assert context["timeout_seconds"] == 10


@pytest.mark.asyncio
async def test_skill_failure_routes_through_error_path(monkeypatch):
    bridge = _make_bridge()
    bridge._current_tool_catalog = {
        "skill::demo::list_things": {
            "qualified_id": "skill::demo::list_things",
            "name": "list_things",
        }
    }

    failure_envelope = {
        "ok": False,
        "error": {
            "code": "MISSING_SECRET",
            "message": "required secret 'api_key' is not configured",
            "repair": {"type": "configure_secret", "secret": "api_key"},
        },
    }
    stub_engine = _StubSkillExecutionEngine(failure_envelope)
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", stub_engine)

    sent_payloads: list = []

    async def _fake_send_runtime_message(payload) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr(bridge, "_send_runtime_message", _fake_send_runtime_message)

    await bridge._handle_tool_request(
        ToolDispatchRequest(
            request_id="r2",
            tool_name="list_things",
            qualified_tool_id="skill::demo::list_things",
            arguments={},
            timeout_seconds=10,
        )
    )

    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert payload.success is False
    assert payload.error_context.code == "MISSING_SECRET"
    assert payload.error_context.detail is not None
    assert payload.error_context.detail["repair"] == {
        "type": "configure_secret",
        "secret": "api_key",
    }


@pytest.mark.asyncio
async def test_stale_session_rejects_before_engine_dispatch(monkeypatch):
    bridge = _make_bridge()
    bridge._current_tool_catalog = {
        "skill::demo::list_things": {
            "qualified_id": "skill::demo::list_things",
            "name": "list_things",
        }
    }
    bridge._session_id = "current"

    stub_engine = _StubSkillExecutionEngine({"ok": True, "result": {}})
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", stub_engine)

    sent_payloads: list = []

    async def _fake_send_runtime_message(payload) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr(bridge, "_send_runtime_message", _fake_send_runtime_message)

    error = bridge._validate_tool_request(
        ToolDispatchRequest(
            request_id="r3",
            tool_name="list_things",
            qualified_tool_id="skill::demo::list_things",
            arguments={},
            timeout_seconds=10,
            expected_session_id="OLD",
        )
    )
    assert error is not None
    assert "session" in error.lower()

    await bridge._handle_tool_request(
        ToolDispatchRequest(
            request_id="r3",
            tool_name="list_things",
            qualified_tool_id="skill::demo::list_things",
            arguments={},
            timeout_seconds=10,
            expected_session_id="OLD",
        )
    )

    assert stub_engine.calls == []
    assert sent_payloads[0].success is False


@pytest.mark.asyncio
async def test_catalog_version_mismatch_rejects_before_engine_dispatch(monkeypatch):
    bridge = _make_bridge()
    bridge._current_tool_catalog = {
        "skill::demo::list_things": {
            "qualified_id": "skill::demo::list_things",
            "name": "list_things",
        }
    }
    bridge._tool_catalog_version = 3

    request = ToolDispatchRequest(
        request_id="r4",
        tool_name="list_things",
        qualified_tool_id="skill::demo::list_things",
        arguments={},
        timeout_seconds=10,
        expected_catalog_version=2,
    )

    error = bridge._validate_tool_request(request)
    assert error is not None
    assert "version" in error.lower()

    stub_engine = _StubSkillExecutionEngine({"ok": True, "result": {}})
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", stub_engine)
    sent_payloads: list = []

    async def _capture(payload) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr(bridge, "_send_runtime_message", _capture)

    await bridge._handle_tool_request(request)

    assert stub_engine.calls == []
    assert sent_payloads[0].success is False


@pytest.mark.asyncio
async def test_missing_capability_rejects_before_engine_dispatch(monkeypatch):
    bridge = _make_bridge()
    bridge._current_tool_catalog = {}

    stub_engine = _StubSkillExecutionEngine({"ok": True, "result": {}})
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", stub_engine)

    request = ToolDispatchRequest(
        request_id="r5",
        tool_name="unknown",
        qualified_tool_id="skill::demo::unknown",
        arguments={},
        timeout_seconds=10,
    )

    error = bridge._validate_tool_request(request)
    assert error is not None
    assert "unknown tool" in error.lower()

    # Drive the full dispatch path to prove the engine is never invoked when
    # the capability is absent from the catalog (the earlier direct-validate
    # assertion alone would pass even if routing skipped the guard).
    sent_payloads: list = []

    async def _capture(payload) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr(bridge, "_send_runtime_message", _capture)

    await bridge._handle_tool_request(request)

    assert stub_engine.calls == []
    assert sent_payloads[0].success is False


@pytest.mark.asyncio
async def test_activate_skill_still_works_unchanged(monkeypatch):
    bridge = _make_bridge()

    monkeypatch.setattr(
        runtime_bridge_module,
        "get_skills_registry",
        lambda: SimpleNamespace(
            get_skill=lambda name: (
                SimpleNamespace(name="demo", enabled=True, content="Follow the demo instructions.")
                if name == "demo"
                else None
            )
        ),
    )

    result = await bridge._execute_tool_request(
        ToolDispatchRequest(
            request_id="req-skill",
            tool_name="activate_skill",
            qualified_tool_id="client_skill::activate",
            arguments={"skill_name": "demo"},
            timeout_seconds=10,
        )
    )

    assert "Skill: demo" in result
    assert "Follow the demo instructions." in result
