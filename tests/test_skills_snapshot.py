from types import SimpleNamespace

from app.ai import skills_snapshot


def test_list_repo_skills_for_demo_serializes_registry_entries(monkeypatch):
    monkeypatch.setattr(
        skills_snapshot,
        "get_server_skills_registry",
        lambda: SimpleNamespace(
            get_all_skills=lambda: [
                SimpleNamespace(
                    name="server-skill",
                    description="description",
                    enabled=True,
                    folder_path="C:/repo/skills/server-skill",
                    content="body",
                )
            ]
        ),
    )

    payload = skills_snapshot.list_repo_skills_for_demo()

    assert payload == {
        "skills": [
            {
                "name": "server-skill",
                "description": "description",
                "enabled": True,
                "folderPath": "C:/repo/skills/server-skill",
            }
        ],
        "totalCount": 1,
        "enabledCount": 1,
    }


def test_get_repo_skill_detail_for_demo_returns_content(monkeypatch):
    monkeypatch.setattr(
        skills_snapshot,
        "get_server_skills_registry",
        lambda: SimpleNamespace(
            get_skill=lambda _name: SimpleNamespace(
                name="server-skill",
                description="description",
                enabled=True,
                folder_path="C:/repo/skills/server-skill",
                content="# Instructions",
            )
        ),
    )

    payload = skills_snapshot.get_repo_skill_detail_for_demo("server-skill")

    assert payload == {
        "name": "server-skill",
        "description": "description",
        "enabled": True,
        "folderPath": "C:/repo/skills/server-skill",
        "content": "# Instructions",
    }


def test_reload_repo_skills_for_demo_reloads_registry(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        skills_snapshot,
        "get_server_skills_registry",
        lambda: SimpleNamespace(
            reload=lambda: calls.append("reload"),
            get_all_skills=lambda: [
                SimpleNamespace(enabled=True),
                SimpleNamespace(enabled=False),
            ],
        ),
    )

    payload = skills_snapshot.reload_repo_skills_for_demo()

    assert calls == ["reload"]
    assert payload == {"message": "Repo skills reloaded: 2 found (1 enabled)"}
