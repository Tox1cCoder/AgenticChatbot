"""Skill visibility is scoped to the originating client device (no server skills)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai import skill_resolver, skills_tool


def _client_session(skills, *, user_id="user-1", session_id="session-1", device_id=None):
    return SimpleNamespace(
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        skill_catalog={"skills": skills},
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

    assert [(entry["name"], entry["source"]) for entry in summaries] == [
        ("client-skill", "client")
    ]


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
