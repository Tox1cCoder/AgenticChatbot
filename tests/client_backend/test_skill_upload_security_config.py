"""Security bounds for the sidecar skill upload foundations.

Covers the three places a skill ZIP workflow can widen the sidecar's attack
surface before any archive byte is read: restored-bearer authorization, profile
path construction from a caller-supplied user id, and the process-wide CORS and
binding configuration.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import jwt
import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials

from client_backend.core import auth as auth_module
from client_backend.core.config import (
    ClientSettings,
    get_client_settings,
    initialize_client_environment,
)
from client_backend.core.paths import (
    get_skill_catalog_state_path,
    get_skill_locks_root,
    get_skill_operations_root,
    get_skill_uploads_root,
)
from client_backend.main import create_app


@pytest.fixture()
def isolated_settings_env(monkeypatch, tmp_path):
    """Construct settings without the tracked .env.client or ambient CLIENT_ vars.

    ``ClientSettings(_env_file=None)`` is not enough: the class overrides
    ``settings_customise_sources`` and always builds its own dotenv source from
    ``CLIENT_ENV_FILE`` (defaulting to the tracked ``.env.client``), and real
    environment variables are still read.

    ``CLIENT_PROFILE_ROOT`` is deliberately preserved: ``tests/conftest.py`` sets
    it so no test ever writes into the developer's live LOCALAPPDATA profile.
    """
    import os

    preserved = {"CLIENT_PROFILE_ROOT"}
    for name in list(os.environ):
        if name.startswith("CLIENT_") and name not in preserved:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLIENT_ENV_FILE", str(tmp_path / "absent.env"))
    get_client_settings.cache_clear()
    yield tmp_path
    get_client_settings.cache_clear()


def _restorable_auth_service(*, user_id: str, access_token: str | None):
    """An upstream auth service that starts cold and restores exactly one user."""
    state = {"authenticated": False, "user_id": None}

    async def restore_session(requested_user_id: str) -> bool:
        if requested_user_id != user_id:
            return False
        state["authenticated"] = True
        state["user_id"] = user_id
        return True

    return SimpleNamespace(
        restore_session=restore_session,
        is_authenticated=lambda: state["authenticated"],
        get_current_user_id=lambda: state["user_id"],
        get_current_access_token=lambda: access_token,
    )


def test_forged_bearer_subject_cannot_authorize_restored_session(monkeypatch):
    forged = jwt.encode({"sub": "user-a"}, "attacker-key-" + "a" * 32, algorithm="HS256")
    auth = _restorable_auth_service(user_id="user-a", access_token="real-token")
    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: auth)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=forged)

    with pytest.raises(auth_module.HTTPException) as exc_info:
        asyncio.run(auth_module.require_local_session(credentials))

    assert exc_info.value.status_code == 401


def test_exact_restored_access_token_remains_compatible(monkeypatch):
    token = jwt.encode({"sub": "user-a"}, "upstream-key-" + "b" * 32, algorithm="HS256")
    auth = _restorable_auth_service(user_id="user-a", access_token=token)
    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "get_runtime_bridge",
        lambda: SimpleNamespace(
            get_registered_device_id=lambda: "device-1",
            get_device_identifier=lambda: "install-1",
        ),
    )

    payload = asyncio.run(
        auth_module.require_local_session(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        )
    )

    assert payload.user_id == "user-a"


@pytest.mark.parametrize(
    "helper",
    [
        get_skill_uploads_root,
        get_skill_operations_root,
        get_skill_locks_root,
        get_skill_catalog_state_path,
    ],
)
@pytest.mark.parametrize("hostile", ["../other-user", "a/b", "a\\b", "", ".", ".."])
def test_skill_profile_paths_reject_traversal(helper, hostile):
    with pytest.raises(ValueError):
        helper(hostile)


def test_skill_profile_paths_stay_under_the_user_skill_root(isolated_settings_env):
    from client_backend.core.paths import profile_subdir_path

    root = profile_subdir_path("user-a", "skills").resolve()
    for path in (
        get_skill_uploads_root("user-a"),
        get_skill_operations_root("user-a"),
        get_skill_locks_root("user-a"),
        get_skill_catalog_state_path("user-a"),
    ):
        assert path.resolve().is_relative_to(root)


def test_skill_upload_defaults_are_production_bounded(isolated_settings_env):
    settings = ClientSettings()

    assert settings.skill_upload_max_bytes == 25 * 1024 * 1024
    assert settings.skill_upload_max_expanded_bytes == 100 * 1024 * 1024
    assert settings.skill_upload_max_file_bytes == 50 * 1024 * 1024
    assert settings.skill_upload_max_entries == 2000
    assert settings.skill_upload_max_compression_ratio == 200
    assert settings.skill_upload_max_path_depth == 20
    assert settings.skill_upload_max_path_chars == 240
    assert settings.skill_upload_ttl_seconds == 1800
    assert settings.skill_operation_receipt_ttl_seconds == 3600
    assert settings.skill_upload_max_outstanding == 5
    assert settings.skill_upload_quota_bytes == 250 * 1024 * 1024
    assert settings.skill_upload_rate_limit_count == 10
    assert settings.skill_upload_rate_limit_window_seconds == 60
    assert settings.skill_install_lock_timeout_seconds == 10
    assert settings.allow_non_loopback_backend is False


@pytest.mark.parametrize(
    "field",
    [
        "skill_upload_max_bytes",
        "skill_upload_max_expanded_bytes",
        "skill_upload_max_file_bytes",
        "skill_upload_max_entries",
        "skill_upload_max_compression_ratio",
        "skill_upload_max_path_depth",
        "skill_upload_max_path_chars",
        "skill_upload_ttl_seconds",
        "skill_operation_receipt_ttl_seconds",
        "skill_upload_max_outstanding",
        "skill_upload_quota_bytes",
        "skill_upload_rate_limit_count",
        "skill_upload_rate_limit_window_seconds",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_limits_are_rejected(isolated_settings_env, field, value):
    with pytest.raises(ValueError):
        ClientSettings(**{field: value})


def test_allowed_origins_default_to_explicit_loopback_origins(isolated_settings_env):
    settings = ClientSettings()

    assert "*" not in settings.allowed_origins
    assert "http://127.0.0.1:3000" in settings.allowed_origins
    assert "http://localhost:8501" in settings.allowed_origins


def test_allowed_origins_parse_from_comma_separated_environment(isolated_settings_env):
    settings = ClientSettings(allowed_origins=" http://localhost:4000 , http://127.0.0.1:4000 ")

    assert settings.allowed_origins == ["http://localhost:4000", "http://127.0.0.1:4000"]


@pytest.mark.parametrize(
    "value",
    ["*", "http://localhost:3000,*", "localhost:3000", "not a url", "http://", ""],
)
def test_wildcard_and_malformed_origins_are_rejected(isolated_settings_env, value):
    with pytest.raises(ValueError):
        ClientSettings(allowed_origins=value)


def test_create_app_uses_configured_origins_and_never_wildcard(isolated_settings_env, monkeypatch):
    monkeypatch.setenv("CLIENT_ALLOWED_ORIGINS", "http://localhost:4321")
    get_client_settings.cache_clear()

    app = create_app()
    cors = [middleware for middleware in app.user_middleware if middleware.cls is CORSMiddleware]

    assert len(cors) == 1
    options = cors[0].kwargs
    assert options["allow_origins"] == ["http://localhost:4321"]
    assert options["allow_credentials"] is False
    # Preserved: a browser on a public origin cannot reach a loopback sidecar
    # without the Private Network Access opt-in.
    assert options["allow_private_network"] is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_non_loopback_binding_requires_explicit_opt_in(isolated_settings_env, tmp_path, host):
    settings = ClientSettings(
        backend_host=host,
        profile_root=str(tmp_path / "profile"),
    )

    with pytest.raises(ValueError) as exc_info:
        initialize_client_environment(settings)

    assert "allow_non_loopback_backend" in str(exc_info.value)


def test_non_loopback_binding_is_allowed_with_explicit_opt_in(isolated_settings_env, tmp_path):
    settings = ClientSettings(
        backend_host="0.0.0.0",
        allow_non_loopback_backend=True,
        profile_root=str(tmp_path / "profile"),
    )

    assert initialize_client_environment(settings) is settings


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_binding_needs_no_opt_in(isolated_settings_env, tmp_path, host):
    settings = ClientSettings(
        backend_host=host,
        profile_root=str(tmp_path / "profile"),
    )

    assert initialize_client_environment(settings) is settings
