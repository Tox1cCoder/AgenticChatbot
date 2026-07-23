from app.services.custom_agent_capability_resolver import (
    client_tool_binding_key,
    client_tool_logical_key,
    resolve_custom_agent_capabilities,
    skill_refs_match,
)

SAVED_TOOL = {
    "type": "client",
    "device_id": "device-a",
    "session_id": "session-a",
    "catalog_version": "1",
    "tool_instance_id": "instance-a",
    "server_name": "desktop-commander",
    "qualified_tool_id": "desktop-commander::read_file",
    "tool_name": "read_file",
}


def _live_tool(**overrides):
    value = {
        "type": "client",
        "device_id": "device-b",
        "session_id": "session-b",
        "catalog_version": "4",
        "tool_instance_id": "instance-b",
        "server_name": "desktop-commander",
        "qualified_tool_id": "desktop-commander::read_file",
        "tool_name": "read_file",
    }
    value.update(overrides)
    return value


def test_same_logical_mcp_rebinds_to_current_device_identity():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[_live_tool()],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert result.status == "ready"
    assert result.missing_tools == []
    assert result.resolved_client_tool_refs[0]["device_id"] == "device-b"
    assert result.resolved_client_tool_refs[0]["session_id"] == "session-b"
    assert result.resolved_client_tool_refs[0]["tool_instance_id"] == "instance-b"


def test_same_tool_name_under_different_server_is_missing():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[
            _live_tool(
                server_name="other-server",
                qualified_tool_id="other-server::read_file",
            )
        ],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert result.status == "degraded"
    assert result.resolved_client_tool_refs == []
    assert result.missing_tools[0]["qualified_tool_id"] == ("desktop-commander::read_file")


def test_missing_skill_degrades_and_legacy_server_source_matches_client_once():
    missing = resolve_custom_agent_capabilities(
        selected_tool_refs=[],
        selected_skill_refs=[
            {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )
    compatible = resolve_custom_agent_capabilities(
        selected_tool_refs=[],
        selected_skill_refs=[
            {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        live_tool_refs=[],
        live_skill_refs=[
            {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
        ],
        request_device_id="device-b",
        device_available=True,
    )

    assert missing.status == "degraded"
    assert missing.missing_skills[0]["lookup_name"] == "kobo-library"
    assert compatible.status == "ready"
    assert missing.warnings == ["Selected skill 'kobo-library' is not available on this device."]


def test_local_dependencies_without_session_are_device_unavailable():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=False,
    )

    assert result.status == "device_unavailable"
    assert result.resolved_client_tool_refs == []
    assert len(result.warnings) == 1


def test_same_logical_tool_on_non_request_device_is_not_a_candidate():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL],
        selected_skill_refs=[],
        live_tool_refs=[_live_tool(device_id="device-b")],
        live_skill_refs=[],
        request_device_id="device-a",
        device_available=True,
    )

    assert result.status == "degraded"
    assert result.resolved_client_tool_refs == []


def test_client_tool_logical_key_rejects_incomplete_or_server_refs():
    assert client_tool_logical_key(SAVED_TOOL) == (
        "desktop-commander",
        "desktop-commander::read_file",
    )
    assert client_tool_logical_key({"type": "client", "qualified_tool_id": "x::y"}) is None
    assert (
        client_tool_logical_key(
            {"type": "server_mcp", "server_name": "x", "qualified_tool_id": "x::y"}
        )
        is None
    )


def test_binding_key_covers_every_submitted_live_identity_field():
    assert client_tool_binding_key(SAVED_TOOL) == (
        "desktop-commander",
        "desktop-commander::read_file",
        "device-a",
        "session-a",
        "1",
        "instance-a",
        "read_file",
    )


def test_duplicate_missing_refs_emit_one_missing_entry_and_warning():
    result = resolve_custom_agent_capabilities(
        selected_tool_refs=[SAVED_TOOL, dict(SAVED_TOOL)],
        selected_skill_refs=[],
        live_tool_refs=[],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )

    assert len(result.missing_tools) == 1
    assert len(result.warnings) == 1


def test_legacy_skill_ref_matches_current_client_ref_but_not_another_name():
    legacy = {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
    current = {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
    other = {"source": "client", "lookup_name": "calendar", "name": "calendar"}

    assert skill_refs_match(legacy, current) is True
    assert skill_refs_match(legacy, other) is False
