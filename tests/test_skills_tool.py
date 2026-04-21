from types import SimpleNamespace
from uuid import uuid4

from app.ai import skill_resolver, skills_tool


def test_available_skill_summaries_include_server_and_client_sources(monkeypatch):
    device_id = str(uuid4())

    monkeypatch.setattr(
        skill_resolver,
        "get_server_skills_registry",
        lambda: SimpleNamespace(
            get_active_skills=lambda: [
                SimpleNamespace(name="server-skill", description="server description", content="x")
            ]
        ),
    )
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: SimpleNamespace(
            user_id="user-1",
            session_id="session-1",
            skill_catalog={
                "skills": [
                    {
                        "name": "client-skill",
                        "description": "client description",
                        "enabled": True,
                        "tags": ["tag-1"],
                    }
                ]
            },
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_id)

    assert [(entry["name"], entry["source"]) for entry in summaries] == [
        ("client-skill", "client"),
        ("server-skill", "server"),
    ]


def test_available_skill_summaries_disambiguate_duplicate_names(monkeypatch):
    device_id = str(uuid4())

    monkeypatch.setattr(
        skill_resolver,
        "get_server_skills_registry",
        lambda: SimpleNamespace(
            get_active_skills=lambda: [
                SimpleNamespace(name="shared-skill", description="server description", content="x")
            ]
        ),
    )
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: SimpleNamespace(
            user_id="user-1",
            session_id="session-1",
            skill_catalog={
                "skills": [
                    {
                        "name": "shared-skill",
                        "description": "client description",
                        "enabled": True,
                    }
                ]
            },
        ),
    )

    summaries = skills_tool.get_available_skill_summaries(user_id="user-1", device_id=device_id)

    assert [(entry["lookup_name"], entry["source"]) for entry in summaries] == [
        ("client:shared-skill", "client"),
        ("server:shared-skill", "server"),
    ]
