"""Device and provenance isolation for editable HITL settings."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.hitl import _to_response, clear_hitl_setting, get_hitl_settings, update_hitl_settings
from app.core.exceptions import CustomHTTPException
from app.schemas.hitl import HitlSettingsUpdate
from app.services import hitl_settings_service as service_module
from app.services.hitl_settings_service import HitlSettingsService


class _MemoryRepository:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.session_factory = None

    def list_by_device(self, user_id, device_id):
        return [row for row in self.rows if row.user_id == user_id and row.device_id == device_id]

    def bulk_set(self, user_id, device_id, items):
        for item in items:
            self.rows.append(SimpleNamespace(user_id=user_id, device_id=device_id, **item))

    def delete(self, user_id, device_id, tool_origin, scope_type, scope_value):
        before = len(self.rows)
        self.rows = [
            row
            for row in self.rows
            if not (
                row.user_id == user_id
                and row.device_id == device_id
                and row.tool_origin == tool_origin
                and row.scope_type == scope_type
                and row.scope_value == scope_value
            )
        ]
        return len(self.rows) != before


def _row(user_id, device_id, origin, scope_type, value, required=False):
    return SimpleNamespace(
        user_id=user_id,
        device_id=device_id,
        tool_origin=origin,
        scope_type=scope_type,
        scope_value=value,
        require_approval=required,
    )


@pytest.fixture
def owned_devices(monkeypatch):
    user_id = uuid4()
    device_a = uuid4()
    device_b = uuid4()
    monkeypatch.setattr(
        service_module,
        "_device_belongs_to_user",
        lambda requested_user, requested_device, _session_factory: (
            requested_user == user_id and requested_device in {device_a, device_b}
        ),
        raising=False,
    )
    return user_id, device_a, device_b


def test_get_settings_returns_only_the_requested_devices_rows(owned_devices):
    user_id, device_a, device_b = owned_devices
    repo = _MemoryRepository(
        [
            _row(
                user_id,
                device_a,
                "client_skill",
                "tool",
                "skill::kobo-library::run_skill_command",
            ),
            _row(
                user_id,
                device_b,
                "client_mcp",
                "server",
                "desktop-commander",
            ),
        ]
    )
    service = HitlSettingsService(repo)

    machine_b = service.get_settings(user_id, device_id=str(device_b))

    assert machine_b["device_id"] == device_b
    assert machine_b["tools"] == []
    assert machine_b["servers"] == [
        {
            "scope_type": "server",
            "scope_value": "desktop-commander",
            "tool_origin": "client_mcp",
            "require_approval": False,
        }
    ]


def test_response_serializes_device_and_skill_tool_origin(owned_devices):
    user_id, device_a, _device_b = owned_devices
    repo = _MemoryRepository(
        [
            _row(
                user_id,
                device_a,
                "client_skill",
                "tool",
                "skill::kobo-library::run_skill_command",
                required=True,
            )
        ]
    )

    response = _to_response(HitlSettingsService(repo).get_settings(user_id, str(device_a)))
    payload = response.model_dump(mode="json", by_alias=True)

    assert payload["deviceId"] == str(device_a)
    assert payload["tools"][0]["toolOrigin"] == "client_skill"


def test_missing_device_context_fails_closed(owned_devices):
    user_id, _device_a, _device_b = owned_devices

    with pytest.raises(CustomHTTPException) as exc_info:
        HitlSettingsService(_MemoryRepository()).get_settings(user_id, device_id=None)

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "HITL_DEVICE_REQUIRED"


def test_apply_rejects_server_owned_origin_with_stable_error(owned_devices, monkeypatch):
    user_id, device_a, _device_b = owned_devices
    monkeypatch.setattr(
        service_module,
        "_lookup_device_session",
        lambda _user_id, _device_id: SimpleNamespace(tool_catalog={"tools": []}),
    )

    with pytest.raises(CustomHTTPException) as exc_info:
        HitlSettingsService(_MemoryRepository()).apply(
            user_id,
            str(device_a),
            [
                {
                    "tool_origin": "server_mcp",
                    "scope_type": "tool",
                    "scope_value": "search::query",
                    "require_approval": True,
                }
            ],
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "HITL_TOOL_ORIGIN_INVALID"


@pytest.mark.asyncio
async def test_api_endpoints_pass_device_and_origin_through_service(owned_devices, monkeypatch):
    user_id, device_a, _device_b = owned_devices
    qualified_id = "skill::kobo-library::run_skill_command"
    repo = _MemoryRepository()
    monkeypatch.setattr(
        service_module,
        "_lookup_device_session",
        lambda _user_id, _device_id: SimpleNamespace(
            tool_catalog={
                "tools": [
                    {
                        "origin": "client_skill",
                        "server_name": "skill_kobo_library",
                        "qualified_id": qualified_id,
                        "name": "run_skill_command",
                    }
                ]
            }
        ),
    )
    service = HitlSettingsService(repo)
    payload = HitlSettingsUpdate.model_validate(
        {
            "items": [
                {
                    "scopeType": "tool",
                    "scopeValue": qualified_id,
                    "toolOrigin": "client_skill",
                    "requireApproval": False,
                }
            ]
        }
    )

    updated = await update_hitl_settings(
        payload,
        service,
        user_id,
        device_id=str(device_a),
        device_id_snake=None,
    )
    fetched = await get_hitl_settings(
        service,
        user_id,
        device_id=str(device_a),
        device_id_snake=None,
    )
    cleared = await clear_hitl_setting(
        service,
        user_id,
        device_id=str(device_a),
        device_id_snake=None,
        tool_origin="client_skill",
        scope_type="tool",
        scope_value=qualified_id,
        scope_type_snake=None,
        scope_value_snake=None,
        tool_origin_snake=None,
    )

    assert updated.data.tools[0].tool_origin == "client_skill"
    assert fetched.data.device_id == device_a
    assert cleared.data.tools == []
