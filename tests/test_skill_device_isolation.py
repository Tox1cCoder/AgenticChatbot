from types import SimpleNamespace
from uuid import uuid4

from app.ai.skill_resolver import list_resolved_skills


def test_client_skills_are_read_only_from_requested_device(monkeypatch):
    user_id = uuid4()
    device_a = uuid4()
    device_b = uuid4()
    sessions = {
        device_a: SimpleNamespace(
            user_id=user_id,
            session_id="session-a",
            skill_catalog={
                "skills": [
                    {"name": "cli-anything-google-calendar", "enabled": True},
                    {"name": "kobo-library", "enabled": True},
                ]
            },
        ),
        device_b: SimpleNamespace(
            user_id=user_id,
            session_id="session-b",
            skill_catalog={"skills": []},
        ),
    }
    monkeypatch.setattr(
        "app.services.client_device_service.ClientDeviceService.lookup_active_session",
        lambda device_id: sessions.get(device_id),
    )

    skills_a = list_resolved_skills(user_id=str(user_id), device_id=str(device_a))
    skills_b = list_resolved_skills(user_id=str(user_id), device_id=str(device_b))

    assert {skill.name for skill in skills_a} == {
        "cli-anything-google-calendar",
        "kobo-library",
    }
    assert skills_b == []


def test_foreign_user_cannot_read_device_skills(monkeypatch):
    owner_id = uuid4()
    other_id = uuid4()
    device_id = uuid4()
    monkeypatch.setattr(
        "app.services.client_device_service.ClientDeviceService.lookup_active_session",
        lambda _device_id: SimpleNamespace(
            user_id=owner_id,
            session_id="session-a",
            skill_catalog={"skills": [{"name": "kobo-library", "enabled": True}]},
        ),
    )

    assert list_resolved_skills(user_id=str(other_id), device_id=str(device_id)) == []
