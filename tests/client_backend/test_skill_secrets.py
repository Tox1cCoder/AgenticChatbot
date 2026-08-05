import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services.skill_runtime import secrets as secrets_module
from client_backend.services.skill_runtime.secrets import SkillSecretStore

USER_ID = "user-a"


def _configure_profile(tmp_path, monkeypatch):
    original = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        secrets_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: USER_ID),
    )
    return original


def test_secret_bindings_are_encrypted_and_namespaced_by_skill(tmp_path, monkeypatch):
    original = _configure_profile(tmp_path, monkeypatch)
    try:
        store = SkillSecretStore()
        store.set_for_skill("calendar", "ACCESS_TOKEN", "calendar-secret")
        store.set_for_skill("mail", "ACCESS_TOKEN", "mail-secret")

        assert store.get_for_skill("calendar") == {"ACCESS_TOKEN": "calendar-secret"}
        assert store.get_for_skill("mail") == {"ACCESS_TOKEN": "mail-secret"}
        assert store.list_for_skill("calendar") == ["ACCESS_TOKEN"]
        secret_file = get_profile_subdir(USER_ID, "skills") / "secrets.json"
        raw = secret_file.read_text(encoding="utf-8")
        assert "calendar-secret" not in raw
        assert "mail-secret" not in raw
        assert json.loads(raw)["encryption"]
    finally:
        client_settings.profile_root = original


def test_set_for_skill_strips_pasted_whitespace(tmp_path, monkeypatch):
    original = _configure_profile(tmp_path, monkeypatch)
    try:
        store = SkillSecretStore()
        store.set_for_skill("calendar", "ACCESS_TOKEN", "  ya29.token-value\n")

        assert store.get_for_skill("calendar") == {"ACCESS_TOKEN": "ya29.token-value"}
    finally:
        client_settings.profile_root = original


def test_set_for_skill_rejects_blank_value(tmp_path, monkeypatch):
    original = _configure_profile(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="secret value"):
            SkillSecretStore().set_for_skill("calendar", "ACCESS_TOKEN", "   \n")
    finally:
        client_settings.profile_root = original


def test_delete_and_remove_skill_never_affect_other_skill(tmp_path, monkeypatch):
    original = _configure_profile(tmp_path, monkeypatch)
    try:
        store = SkillSecretStore()
        store.set_for_skill("calendar", "A", "one")
        store.set_for_skill("calendar", "B", "two")
        store.set_for_skill("mail", "A", "mail")

        assert store.delete_for_skill("calendar", "A") is True
        assert store.get_for_skill("calendar") == {"B": "two"}
        store.remove_skill("calendar")

        assert store.get_for_skill("calendar") == {}
        assert store.get_for_skill("mail") == {"A": "mail"}
    finally:
        client_settings.profile_root = original


def test_two_machine_profile_roots_do_not_share_secret_bindings(tmp_path, monkeypatch):
    original = _configure_profile(tmp_path, monkeypatch)
    try:
        machine_a = tmp_path / "machine-a"
        machine_b = tmp_path / "machine-b"
        client_settings.profile_root = str(machine_a)
        SkillSecretStore().set_for_skill("calendar", "TOKEN", "machine-a-token")

        client_settings.profile_root = str(machine_b)
        assert SkillSecretStore().get_for_skill("calendar") == {}
        SkillSecretStore().set_for_skill("calendar", "TOKEN", "machine-b-token")

        client_settings.profile_root = str(machine_a)
        assert SkillSecretStore().get_for_skill("calendar") == {"TOKEN": "machine-a-token"}
    finally:
        client_settings.profile_root = original


class _Skill:
    name = "calendar"
    declared_secrets = ["CALENDAR_TOKEN", "CALENDAR_ID"]


class _Registry:
    async def initialize(self):
        return None

    def get_skill(self, name):
        return _Skill() if name == "calendar" else None


class _Store:
    def __init__(self):
        self.values = {}

    def set_for_skill(self, skill, name, value):
        self.values.setdefault(skill, {})[name] = value

    def list_for_skill(self, skill):
        return sorted(self.values.get(skill, {}))

    def delete_for_skill(self, skill, name):
        return self.values.get(skill, {}).pop(name, None) is not None


def _build_app():
    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    return app


def test_per_skill_secret_api_never_returns_values(monkeypatch):
    store = _Store()
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: store)
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: _Registry())

    with TestClient(_build_app()) as client:
        saved = client.post(
            "/skills/calendar/secrets",
            json={"name": "ACCESS_TOKEN", "value": "never-echo"},
        )
        listed = client.get("/skills/calendar/secrets")
        removed = client.delete("/skills/calendar/secrets/ACCESS_TOKEN")

    assert saved.status_code == 200
    assert listed.status_code == 200
    # Declared names first, in the author's order, then configured extras.
    assert listed.json()["data"] == {
        "secrets": [
            {"name": "CALENDAR_TOKEN", "declared": True, "configured": False},
            {"name": "CALENDAR_ID", "declared": True, "configured": False},
            {"name": "ACCESS_TOKEN", "declared": False, "configured": True},
        ]
    }
    assert "never-echo" not in saved.text
    assert "never-echo" not in listed.text
    assert removed.status_code == 200


def test_declared_secret_reports_configured_once_bound(monkeypatch):
    store = _Store()
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: store)
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: _Registry())

    with TestClient(_build_app()) as client:
        client.post(
            "/skills/calendar/secrets",
            json={"name": "CALENDAR_TOKEN", "value": "bound"},
        )
        listed = client.get("/skills/calendar/secrets")

    assert listed.json()["data"]["secrets"] == [
        {"name": "CALENDAR_TOKEN", "declared": True, "configured": True},
        {"name": "CALENDAR_ID", "declared": True, "configured": False},
    ]


def test_secret_api_rejects_unknown_skill(monkeypatch):
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: _Store())
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: _Registry())

    with TestClient(_build_app()) as client:
        response = client.post(
            "/skills/missing/secrets",
            json={"name": "TOKEN", "value": "secret"},
        )

    assert response.status_code == 404
