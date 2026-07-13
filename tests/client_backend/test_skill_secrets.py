"""Tests for client_backend.services.skill_runtime.secrets and the secret API routes."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services.skill_runtime import secrets as secrets_module
from client_backend.services.skill_runtime.secrets import SkillSecretStore, redact_secret_values
from shared.skills.manifest import SkillManifest, load_manifest


@pytest.fixture
def no_user(monkeypatch):
    """Fake an environment with no active upstream user session."""
    monkeypatch.setattr(
        secrets_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: None),
    )


@pytest.fixture
def fake_profile(tmp_path, monkeypatch):
    """Point the profile root at a tmp dir and fake an active user id.

    Mirrors the profile-root + fake-user pattern in
    tests/client_backend/test_skills_registry.py::
    test_skill_enabled_state_is_isolated_per_user_profile. Yields a mutable
    ``SimpleNamespace`` so a test can flip to "no active user" mid-test by
    setting ``current_user_id = None``.
    """
    profile_root = tmp_path / "profiles"
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(profile_root)

    auth_state = SimpleNamespace(current_user_id="user-a")
    monkeypatch.setattr(
        secrets_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: auth_state.current_user_id),
    )

    try:
        yield auth_state
    finally:
        client_settings.profile_root = original_profile_root


# -- env lookup --------------------------------------------------------


def test_env_lookup_returns_value_and_unknown_is_none(no_user):
    store = SkillSecretStore(environ={"API_TOKEN": "v"})
    assert store.get("API_TOKEN") == "v"
    assert store.get("UNKNOWN") is None
    assert store.has("API_TOKEN") is True
    assert store.has("UNKNOWN") is False


# -- stored lookup + persistence ----------------------------------------


def test_stored_secret_persists_across_fresh_instances(fake_profile):
    store = SkillSecretStore()
    store.set("TOK", "secretval")

    assert store.get("TOK") == "secretval"

    fresh_store = SkillSecretStore()
    assert fresh_store.get("TOK") == "secretval"
    assert "TOK" in fresh_store.list_stored_names()


def test_profile_secret_takes_precedence_over_env(fake_profile):
    store = SkillSecretStore(environ={"TOK": "env-val", "ONLY_ENV": "env-only"})
    store.set("TOK", "profile-val")

    assert store.get("TOK") == "profile-val"
    assert store.get("ONLY_ENV") == "env-only"


def test_secret_value_is_encrypted_at_rest(fake_profile):
    store = SkillSecretStore()
    store.set("TOK", "supersecret")

    secrets_path = get_profile_subdir("user-a", "skills") / "secrets.json"
    raw = secrets_path.read_bytes()
    assert b"supersecret" not in raw


def test_corrupt_or_malformed_store_reads_as_empty_never_crashes(fake_profile):
    store = SkillSecretStore()
    store.set("TOK", "v")
    secrets_path = get_profile_subdir("user-a", "skills") / "secrets.json"

    # Garbage that is not even JSON.
    secrets_path.write_text("not json at all", encoding="utf-8")
    assert SkillSecretStore().get("TOK") is None
    assert SkillSecretStore().list_stored_names() == set()

    # Valid JSON envelope shape but an undecryptable ciphertext.
    secrets_path.write_text(
        '{"version": 1, "encryption": "fernet", "ciphertext": "bogus"}', encoding="utf-8"
    )
    assert SkillSecretStore().get("TOK") is None

    # Valid JSON but not the expected envelope object.
    secrets_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert SkillSecretStore().get("TOK") is None


def test_missing_secret_has_is_false(fake_profile):
    store = SkillSecretStore()
    assert store.has("NOPE") is False


def test_no_active_user_get_is_env_only_and_writes_raise(fake_profile):
    fake_profile.current_user_id = None
    store = SkillSecretStore(environ={"TOK": "env-val"})

    assert store.get("TOK") == "env-val"
    assert store.list_stored_names() == set()

    with pytest.raises(RuntimeError):
        store.set("TOK", "x")
    with pytest.raises(RuntimeError):
        store.delete("TOK")


def test_delete_secret(fake_profile):
    store = SkillSecretStore()
    store.set("TOK", "v")

    assert store.delete("TOK") is True
    assert store.get("TOK") is None
    assert store.delete("TOK") is False


# -- redaction -----------------------------------------------------------


def test_redact_secret_values_longest_first_and_skips_empty():
    result = redact_secret_values("a=abcdef b=abc", {"abc", "abcdef", ""})
    assert result == "a=<redacted> b=<redacted>"
    assert "abc" not in result
    assert "abcdef" not in result


# -- API -------------------------------------------------------------------


def _manifest_with_secret() -> SkillManifest:
    return load_manifest(
        {
            "schema_version": "1.0",
            "name": "demo",
            "description": "Demo skill with a declared secret.",
            "runtime": {"type": "python_module", "module": "skills.demo.cli"},
            "secrets": [{"name": "API_TOKEN", "required": True, "description": "API token"}],
            "capabilities": [
                {
                    "name": "do_thing",
                    "description": "Does a thing.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": []},
                }
            ],
        }
    )


class _Skill:
    def __init__(self, name: str, *, manifest: SkillManifest | None = None):
        self.name = name
        self.manifest = manifest


class _RegistryStub:
    def __init__(self, skills: dict):
        self.skills = skills

    async def initialize(self) -> None:
        pass

    def get_skill(self, name: str):
        return self.skills.get(name)


class _SecretStoreStub:
    def __init__(self, configured: set[str] | None = None):
        self.configured = set(configured) if configured else set()
        self.set_calls: list[tuple[str, str]] = []

    def has(self, name: str) -> bool:
        return name in self.configured

    def set(self, name: str, value: str) -> None:
        self.set_calls.append((name, value))
        self.configured.add(name)


class _RaisingSecretStoreStub:
    def set(self, name: str, value: str) -> None:
        raise RuntimeError("no active user profile for secret storage")


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    return app


def test_get_skill_secrets_lists_names_without_values(monkeypatch):
    registry = _RegistryStub({"demo": _Skill("demo", manifest=_manifest_with_secret())})
    store = _SecretStoreStub(configured={"API_TOKEN"})
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: registry)
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: store)

    with TestClient(_build_app()) as client:
        response = client.get("/skills/demo/secrets")
        missing_response = client.get("/skills/missing/secrets")

    assert response.status_code == 200
    assert response.json()["data"]["secrets"] == [
        {
            "name": "API_TOKEN",
            "required": True,
            "description": "API token",
            "configured": True,
        }
    ]
    assert missing_response.status_code == 404


def test_get_skill_secrets_returns_empty_list_for_manifestless_skill(monkeypatch):
    registry = _RegistryStub({"demo": _Skill("demo", manifest=None)})
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: registry)
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: _SecretStoreStub())

    with TestClient(_build_app()) as client:
        response = client.get("/skills/demo/secrets")

    assert response.status_code == 200
    assert response.json()["data"]["secrets"] == []


def test_set_skill_secret_calls_store_and_never_echoes_value(monkeypatch):
    store = _SecretStoreStub()
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: store)

    with TestClient(_build_app()) as client:
        response = client.post("/skills/secrets", json={"name": "API_TOKEN", "value": "shh"})

    assert response.status_code == 200
    assert response.json()["data"] == {"name": "API_TOKEN", "configured": True}
    assert "shh" not in response.text
    assert store.set_calls == [("API_TOKEN", "shh")]


def test_set_skill_secret_maps_no_profile_error_to_400(monkeypatch):
    monkeypatch.setattr(skills_api, "get_secret_store", lambda: _RaisingSecretStoreStub())

    with TestClient(_build_app()) as client:
        response = client.post("/skills/secrets", json={"name": "X", "value": "y"})

    assert response.status_code == 400
