import pytest

from client_backend.schemas.runtime import ToolDispatchRequest
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


class _ServerClient:
    base_url = "http://server.test"

    def is_authenticated(self):
        return True


class _Engine:
    instances = []

    def __init__(self, **kwargs):
        self.calls = []
        _Engine.instances.append(self)

    async def execute(self, qualified_tool_id, arguments, context):
        self.calls.append((qualified_tool_id, arguments, context))
        return {"ok": True, "result": {"ran": True}}


def _bridge():
    bridge = RuntimeBridgeService(server_client=_ServerClient())
    bridge._device_id = "device-a"
    bridge._session_id = "session-a"
    return bridge


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_command_dispatch_passes_approval_and_device_context(monkeypatch, approved):
    _Engine.instances = []
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", _Engine)
    bridge = _bridge()

    result = await bridge._execute_tool_request(
        ToolDispatchRequest(
            request_id="request-1",
            tool_name="run_skill_command",
            qualified_tool_id="skill::demo::run_skill_command",
            arguments={"argv": ["demo-cli"]},
            timeout_seconds=12,
            mutation_approved=approved,
        )
    )

    assert result == {"ran": True}
    call = _Engine.instances[0].calls[0]
    assert call[0] == "skill::demo::run_skill_command"
    assert call[1] == {"argv": ["demo-cli"]}
    assert call[2] == {
        "device_id": "device-a",
        "session_id": "session-a",
        "timeout_seconds": 12,
        "mutation_approved": approved,
    }


@pytest.mark.asyncio
async def test_stale_session_is_rejected_before_command_engine(monkeypatch):
    _Engine.instances = []
    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", _Engine)
    bridge = _bridge()
    bridge._current_tool_catalog = {
        "skill::demo::run_skill_command": {
            "qualified_id": "skill::demo::run_skill_command",
            "name": "run_skill_command",
            "tool_instance_id": "current-instance",
        }
    }
    request = ToolDispatchRequest(
        request_id="request-2",
        tool_name="run_skill_command",
        qualified_tool_id="skill::demo::run_skill_command",
        arguments={"argv": ["demo-cli"]},
        expected_session_id="old-session",
        mutation_approved=True,
    )

    error = bridge._validate_tool_request(request)

    assert error is not None
    assert "session" in error.lower()
    assert _Engine.instances == []


def test_skill_command_requires_complete_device_session_catalog_binding():
    bridge = _bridge()
    bridge._tool_catalog_version = 3
    bridge._current_tool_catalog = {
        "skill::demo::run_skill_command": {
            "qualified_id": "skill::demo::run_skill_command",
            "name": "run_skill_command",
            "tool_instance_id": "current-instance",
        }
    }
    request = ToolDispatchRequest(
        request_id="request-unbound",
        tool_name="run_skill_command",
        qualified_tool_id="skill::demo::run_skill_command",
        arguments={"argv": ["demo-cli"]},
        mutation_approved=True,
    )

    error = bridge._validate_tool_request(request)

    assert error is not None
    assert "binding" in error.lower()


@pytest.mark.asyncio
async def test_command_failure_uses_structured_runtime_error(monkeypatch):
    class _FailureEngine:
        async def execute(self, qualified_tool_id, arguments, context):
            return {
                "ok": False,
                "error": {
                    "code": "COMMAND_NOT_FOUND",
                    "message": "not owned",
                    "repair": {"type": "inspect_skill_commands"},
                },
            }

    monkeypatch.setattr(runtime_bridge_module, "SkillExecutionEngine", _FailureEngine)
    bridge = _bridge()

    with pytest.raises(Exception) as exc_info:
        await bridge._execute_skill_capability(
            "skill::demo::run_skill_command",
            {"argv": ["missing"]},
            10,
            True,
        )

    assert getattr(exc_info.value, "code", None) == "COMMAND_NOT_FOUND"
    assert getattr(exc_info.value, "repair", None) == {"type": "inspect_skill_commands"}
