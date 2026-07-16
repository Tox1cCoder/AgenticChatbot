"""Skill visibility is scoped to the originating client device (no server skills)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai import skill_resolver, skills_tool
from app.ai.tool_execution_policy import (
    resolve_tool_execution_policy,
    tool_policy_context,
)
from app.core.config import settings
from app.services import client_runtime_store as runtime_store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    InMemoryClientRuntimeStore,
    reset_client_runtime_store,
)


def _client_session(
    skills,
    *,
    user_id="user-1",
    session_id="session-1",
    device_id=None,
    tool_catalog=None,
):
    return SimpleNamespace(
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        skill_catalog={"skills": skills},
        tool_catalog=tool_catalog or {"tools": []},
    )


def test_summaries_list_only_the_bound_clients_skills(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: _client_session(
            [{"name": "client-skill", "description": "client description", "enabled": True}]
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_id)

    assert [(entry["name"], entry["source"]) for entry in summaries] == [("client-skill", "client")]


def test_summaries_are_empty_without_a_device():
    assert skills_tool.get_available_skill_summaries(user_id="user-1", device_id=None) == []


def test_summaries_scoped_to_the_originating_device(monkeypatch):
    device_a, device_b = str(uuid4()), str(uuid4())
    catalogs = {
        device_a: [{"name": "skill-a", "description": "a", "enabled": True}],
        device_b: [{"name": "skill-b", "description": "b", "enabled": True}],
    }

    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda device_uuid: (
            _client_session(catalogs[str(device_uuid)]) if str(device_uuid) in catalogs else None
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_b)

    assert [entry["name"] for entry in summaries] == ["skill-b"]


@pytest.mark.asyncio
async def test_activate_skill_unknown_name_returns_graceful_error(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: _client_session(
            [{"name": "client-skill", "description": "d", "enabled": True}],
            device_id=device_id,
        ),
    )

    dispatched = []

    async def _record_dispatch(**kwargs):
        dispatched.append(kwargs)
        return {"success": True, "result": ""}

    monkeypatch.setattr(skills_tool.ClientDeviceService, "dispatch_tool_call", _record_dispatch)

    tool = skills_tool.create_activate_skill_tool(user_id="user-1", device_id=device_id)
    # "find-skills" exists in this repo's skills/ folder; it must NOT resolve.
    result = await tool.ainvoke({"skill_name": "find-skills"})

    assert "not found" in result.lower()
    assert "client-skill" in result  # only this session's skills are offered
    assert dispatched == []


@pytest.mark.asyncio
async def test_activation_appends_exact_model_callable_skill_tool_name(monkeypatch):
    device_id = str(uuid4())
    session = _client_session(
        [{"name": "client-skill", "description": "d", "enabled": True}],
        device_id=device_id,
        tool_catalog={
            "tools": [
                {
                    "name": "run_skill_command",
                    "origin": "mcp",
                    "server_name": "skill_client_skill",
                    "qualified_id": "mcp::collision",
                },
                {
                    "name": "run_skill_command",
                    "origin": "skill",
                    "server_name": "skill_client_skill",
                    "qualified_id": "skill::client-skill::run_skill_command",
                    "mutation": True,
                },
            ]
        },
    )
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: session,
    )

    dispatches = []

    async def _dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "skill body"}

    monkeypatch.setattr(skills_tool.ClientDeviceService, "dispatch_tool_call", _dispatch)
    tool = skills_tool.create_activate_skill_tool(user_id="user-1", device_id=device_id)

    result = await tool.ainvoke({"skill_name": "client-skill"})

    assert "skill body" in result
    assert "client__skill_client_skill__run_skill_command_2" in result
    assert "exact model-callable tool" in result.lower()
    assert dispatches[0]["execution_timeout_seconds"] == settings.client_runtime_ws_timeout_seconds
    assert dispatches[0]["response_timeout_seconds"] == settings.client_runtime_ws_timeout_seconds


@pytest.mark.asyncio
async def test_activate_skill_uses_scoped_policy_deadlines_and_client_identity(monkeypatch):
    device_id = str(uuid4())
    session = _client_session(
        [{"name": "client-skill", "description": "d", "enabled": True}],
        device_id=device_id,
    )
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: session,
    )
    dispatches = []

    async def _dispatch(**kwargs):
        dispatches.append(kwargs)
        return {"success": True, "result": "skill body"}

    monkeypatch.setattr(skills_tool.ClientDeviceService, "dispatch_tool_call", _dispatch)
    tool = skills_tool.create_activate_skill_tool(user_id="user-1", device_id=device_id)

    assert tool.metadata["tool_origin"] == "client_skill"
    assert tool.metadata["qualified_tool_id"] == "client_skill::activate"

    policy = resolve_tool_execution_policy(
        tool,
        exposed_tool_name=tool.name,
        invocation_kind="client_runtime",
    )
    with tool_policy_context(policy):
        result = await tool.ainvoke({"skill_name": "client-skill"})

    assert result == "skill body"
    assert dispatches[0]["execution_timeout_seconds"] == 28.0
    assert dispatches[0]["response_timeout_seconds"] == 29.0


@pytest.mark.asyncio
async def test_client_skill_activation_dispatch_is_not_rejected_by_mcp_tool_catalog():
    reset_client_runtime_store()
    store = InMemoryClientRuntimeStore()
    runtime_store_module._store = store

    user_id = uuid4()
    device_id = uuid4()
    await store.put_session(
        DeviceSessionRecord(
            device_id=device_id,
            session_id="session-1",
            user_id=user_id,
            tool_catalog={
                "tools": [
                    {
                        "name": "get_current_time",
                        "origin": "mcp",
                        "server_name": "time",
                        "qualified_id": "time::get_current_time",
                    }
                ]
            },
            tool_catalog_version=4,
            skill_catalog={
                "skills": [{"name": "client-skill", "description": "d", "enabled": True}]
            },
        )
    )

    dispatched = []

    async def _dispatch_request(session, request, timeout_seconds):
        dispatched.append((session, request, timeout_seconds))
        return {"success": True, "result": "skill body"}

    store.dispatch_request = _dispatch_request

    try:
        result = await ClientDeviceService.dispatch_tool_call(
            user_id=str(user_id),
            device_id=str(device_id),
            tool_name="activate_skill",
            qualified_tool_id="client_skill::activate",
            arguments={"skill_name": "client-skill"},
            execution_timeout_seconds=4.0,
            response_timeout_seconds=5.0,
        )
    finally:
        reset_client_runtime_store()

    assert result == {"success": True, "result": "skill body"}
    assert len(dispatched) == 1
    assert dispatched[0][1].qualified_tool_id == "client_skill::activate"
    assert dispatched[0][1].timeout_seconds == 4.0
    assert dispatched[0][2] == 5.0
