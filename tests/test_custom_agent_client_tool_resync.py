"""Custom-agent client tools must survive a sidecar reconnect/catalog resync.

A custom agent persists its selected client tools with the session-scoped
identity (session_id / catalog_version / tool_instance_id) captured at save
time. Those rotate on every sidecar reconnect/resync, so the strict
authorization match would otherwise strand the agent's own tools on the
follow-up turn (not bound + not findable via tool_search). Re-resolution
rebinds the persisted refs onto the device's current live tools by stable
identity (device + qualified_tool_id), while cross-device isolation and the
strict matcher itself stay intact.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.ai.custom_agent_runtime import (
    build_custom_agent_runtime_spec,
    filter_tools_for_custom_agent,
    rebase_client_tool_refs,
)

_QID = "client__csv__profile"

# Selected at agent-creation time, in session-1.
PERSISTED_CLIENT_REF = {
    "type": "client",
    "device_id": "desktop-1",
    "session_id": "session-1",
    "catalog_version": "1",
    "tool_instance_id": "csv-profile-instance-1",
    "server_name": "csv",
    "qualified_tool_id": _QID,
    "tool_name": "profile",
}


class _FakeTool:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata


def _spec(tool_refs):
    return build_custom_agent_runtime_spec(
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "runtime_agent_id": "custom_agent:11111111-1111-1111-1111-111111111111",
            "model_agent_key": "custom",
            "name": "Analyst",
            "prompt": "p",
            "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
            "tool_refs": tool_refs,
            "skill_refs": [],
        }
    )


def _live_tool(*, device_id="desktop-1", session_id="session-2", instance="csv-profile-instance-2"):
    """A live client tool from the CURRENT (reconnected) session."""
    return _FakeTool(
        "client__csv__profile",
        {
            "tool_origin": "client_mcp",
            "is_client_tool": True,
            "device_id": device_id,
            "session_id": session_id,
            "catalog_version": 2,
            "tool_instance_id": instance,
            "qualified_tool_id": _QID,
        },
    )


def test_strict_match_rejects_stale_ref_before_rebase():
    """Baseline: the persisted (session-1) ref does NOT match the live
    (session-2) tool — this is the bug the user hit."""
    spec = _spec([PERSISTED_CLIENT_REF])
    allowed, warnings = filter_tools_for_custom_agent(
        [_live_tool()], spec, request_device_id="desktop-1"
    )
    assert allowed == []
    assert warnings


def test_rebased_ref_authorizes_current_session_tool():
    """After rebasing onto the live tool, the agent's own selected tool is
    authorized again."""
    spec = _spec([PERSISTED_CLIENT_REF])
    live = _live_tool()

    rebased = rebase_client_tool_refs(
        spec.allowed_client_tool_refs, [live], request_device_id="desktop-1"
    )
    rebased_spec = spec.model_copy(update={"allowed_client_tool_refs": rebased})

    allowed, warnings = filter_tools_for_custom_agent(
        [live], rebased_spec, request_device_id="desktop-1"
    )
    assert [t.name for t in allowed] == ["client__csv__profile"]
    assert warnings == []


def test_rebased_allowlist_tracks_current_instance_id():
    """tool_search discovery must follow the current instance id, not the stale
    one, or the agent can never re-find its own tool."""
    spec = _spec([PERSISTED_CLIENT_REF])
    rebased = rebase_client_tool_refs(
        spec.allowed_client_tool_refs, [_live_tool()], request_device_id="desktop-1"
    )
    rebased_spec = spec.model_copy(update={"allowed_client_tool_refs": rebased})

    allowlist = set(rebased_spec.client_tool_search_allowlist())
    assert "csv-profile-instance-2" in allowlist
    assert "csv-profile-instance-1" not in allowlist


def test_rebase_does_not_authorize_foreign_device_tool():
    """A live tool with the same qualified id but on a DIFFERENT device must
    never be used to rebase — cross-device isolation is preserved."""
    spec = _spec([PERSISTED_CLIENT_REF])  # selected on desktop-1
    foreign = _live_tool(device_id="desktop-2", session_id="session-9")

    rebased = rebase_client_tool_refs(
        spec.allowed_client_tool_refs, [foreign], request_device_id="desktop-1"
    )
    # The stale ref is left untouched (no same-device live tool to rebase onto).
    assert rebased is spec.allowed_client_tool_refs

    rebased_spec = spec.model_copy(update={"allowed_client_tool_refs": rebased})
    allowed, warnings = filter_tools_for_custom_agent(
        [foreign], rebased_spec, request_device_id="desktop-1"
    )
    assert allowed == []
    assert warnings


def test_rebase_is_noop_without_device_or_refs():
    spec = _spec([PERSISTED_CLIENT_REF])
    # No device → unchanged (same object).
    refs = spec.allowed_client_tool_refs
    assert rebase_client_tool_refs(refs, [_live_tool()], request_device_id=None) is refs
    # No refs → unchanged.
    assert rebase_client_tool_refs([], [_live_tool()], request_device_id="desktop-1") == []


def test_rebase_leaves_ref_when_tool_absent_from_live_catalog():
    """A genuinely-removed tool stays unmatched (reported unavailable), not
    silently rebased onto something else."""
    spec = _spec([PERSISTED_CLIENT_REF])
    other = _FakeTool(
        "client__other__thing",
        {
            "tool_origin": "client_mcp",
            "is_client_tool": True,
            "device_id": "desktop-1",
            "session_id": "session-2",
            "catalog_version": 2,
            "tool_instance_id": "other-instance",
            "qualified_tool_id": "client__other__thing",
        },
    )
    rebased = rebase_client_tool_refs(
        spec.allowed_client_tool_refs, [other], request_device_id="desktop-1"
    )
    assert rebased is spec.allowed_client_tool_refs


def test_catalog_rebuilds_on_session_change_with_same_version():
    """A reconnect can reset catalog_version to a colliding number; the catalog
    must still rebuild when the session_id changes, or rotated instance ids stay
    unfindable via tool_search."""
    from app.ai.client_tool_catalog import ClientToolCatalog

    catalog = ClientToolCatalog("dev-1", "user-1")
    tool_catalog = {
        "tools": [
            {
                "name": "profile",
                "qualified_id": _QID,
                "origin": "mcp",
                "server_name": "csv",
                "tool_instance_id": "inst-A",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    }

    session_a = SimpleNamespace(
        user_id="user-1",
        session_id="sess-A",
        tool_catalog_version=0,
        tool_catalog=tool_catalog,
    )
    assert catalog.refresh_from_session(session_a) is True
    assert catalog.session_id == "sess-A"

    # New session, SAME version number (0), different session id.
    session_b = SimpleNamespace(
        user_id="user-1",
        session_id="sess-B",
        tool_catalog_version=0,
        tool_catalog=tool_catalog,
    )
    assert catalog.refresh_from_session(session_b) is True
    assert catalog.session_id == "sess-B"
