"""Only this machine's user may drive the sidecar (audit Task 5).

``POST /auth/restore?user_id=`` hands out access, refresh and local-session tokens
with no credential, and ``GET /auth/users`` lists the ids it accepts. Any local
process, a command run as the sandbox account, or a DNS-rebinding page could chain
the two and take over the session. These tests pin the transport checks that close
that: a per-launch token only the user can read, a Host allowlist, and an Origin
check on mutations.

The suite runs with ``CLIENT_TRUST_CHECKS_ENABLED=false`` (tests/conftest.py), so
every test here turns the checks on explicitly.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from client_backend import main as main_module
from client_backend.core import launch_token
from client_backend.core.config import client_settings

SIDECAR = "http://127.0.0.1:8100"
BROWSER_ORIGIN = "http://localhost:3000"


@pytest.fixture
def trusted(monkeypatch, tmp_path):
    """Checks on, a private profile root, and a token issued for this 'launch'."""

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    monkeypatch.setattr(client_settings, "trust_checks_enabled", True)
    token = launch_token.issue_launch_token()
    yield token
    launch_token.revoke_launch_token()


@pytest.fixture
def client():
    return TestClient(main_module.create_app(), base_url=SIDECAR)


def test_restore_without_the_launch_token_is_refused(trusted, client):
    response = client.post("/auth/restore", params={"user_id": "victim"})

    assert response.status_code == 401
    assert response.json() == {"detail": "This route requires the sidecar launch token"}


def test_listing_users_without_the_launch_token_is_refused(trusted, client):
    assert client.get("/auth/users").status_code == 401
    assert client.get("/api/auth/users").status_code == 401
    assert client.get("/auth/users/").status_code == 401


def test_session_minting_routes_ignore_a_browser_origin(trusted, client):
    """An Origin header is forgeable by any local process; it never admits restore."""

    response = client.post(
        "/auth/restore", params={"user_id": "victim"}, headers={"Origin": BROWSER_ORIGIN}
    )

    assert response.status_code == 401
    assert client.get("/auth/users", headers={"Origin": BROWSER_ORIGIN}).status_code == 401


def test_the_launch_token_admits_the_session_minting_routes(trusted, client):
    headers = {"X-Kani-Client": trusted}

    listed = client.get("/auth/users", headers=headers)
    restored = client.post("/auth/restore", params={"user_id": "nobody"}, headers=headers)

    assert listed.status_code == 200
    assert listed.json() == {"users": []}
    # The route itself answered: there is no stored session for this user.
    assert restored.json()["detail"] == "Failed to restore session - credentials may be expired"


def test_a_wrong_launch_token_is_refused(trusted, client):
    for presented in ("wrong", trusted + "x", "ü-not-ascii".encode("latin-1")):
        response = client.get("/auth/users", headers={"X-Kani-Client": presented})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid launch token"}


def test_a_wrong_launch_token_is_refused_even_from_an_allowed_origin(trusted, client):
    response = client.get(
        "/mcp/sandbox", headers={"X-Kani-Client": "wrong", "Origin": BROWSER_ORIGIN}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid launch token"}


def test_other_routes_need_the_token_or_an_allowed_browser_origin(trusted, client):
    assert client.get("/mcp/sandbox").status_code == 401

    browser = client.get("/mcp/sandbox", headers={"Origin": BROWSER_ORIGIN})
    # Admitted past the transport check, then refused by the route's own session check.
    assert browser.status_code == 401
    assert browser.json() == {"detail": "Missing local session bearer token"}


def test_a_foreign_host_is_refused(trusted, client):
    """A DNS-rebinding page reaches 127.0.0.1 under its own hostname."""

    response = client.get(
        "/auth/users", headers={"Host": "evil.example", "X-Kani-Client": trusted}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Host is not allowed"}
    assert client.get("/health", headers={"Host": "evil.example:8100"}).status_code == 400


@pytest.mark.parametrize("host", ["localhost:8100", "127.0.0.1", "[::1]:8100", "LOCALHOST"])
def test_loopback_hosts_are_accepted(trusted, client, host):
    response = client.get("/auth/users", headers={"Host": host, "X-Kani-Client": trusted})

    assert response.status_code == 200


def test_the_configured_backend_host_is_accepted(trusted, client, monkeypatch):
    monkeypatch.setattr(client_settings, "backend_host", "sidecar.internal")

    response = client.get(
        "/auth/users", headers={"Host": "sidecar.internal:8100", "X-Kani-Client": trusted}
    )

    assert response.status_code == 200


def test_the_test_client_host_is_not_trusted(trusted):
    """``testserver`` is TestClient's default Host; production must not accept it."""

    response = TestClient(main_module.create_app()).get(
        "/auth/users", headers={"X-Kani-Client": trusted}
    )

    assert response.status_code == 400


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_mutation_from_a_foreign_origin_is_refused(trusted, client, method):
    response = client.request(
        method,
        "/auth/logout",
        headers={"Origin": "https://evil.example", "X-Kani-Client": trusted},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Origin is not allowed"}


def test_an_opaque_origin_mutation_is_refused(trusted, client):
    response = client.post("/auth/logout", headers={"Origin": "null"})

    assert response.status_code == 403


def test_a_request_without_origin_passes_the_origin_check(trusted, client):
    response = client.post("/auth/logout", headers={"X-Kani-Client": trusted})

    # The route itself answered: logout needs the bearer session it ends.
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing local session bearer token"}


def test_health_stays_open(trusted, client):
    for path in ("/health", "/api/health", "/health/live", "/health/ready"):
        assert client.get(path).status_code == 200, path


def test_without_an_issued_token_guarded_routes_fail_closed(trusted, client):
    launch_token.revoke_launch_token()

    assert client.get("/auth/users", headers={"X-Kani-Client": trusted}).status_code == 503
    assert client.get("/mcp/sandbox", headers={"Origin": BROWSER_ORIGIN}).status_code == 503
    assert client.get("/health").status_code == 200


def test_disabled_checks_leave_the_sidecar_as_it_was(monkeypatch, tmp_path):
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    monkeypatch.setattr(client_settings, "trust_checks_enabled", False)

    response = TestClient(main_module.create_app()).get("/auth/users")

    assert response.status_code == 200


def test_a_rotated_token_replaces_the_previous_one(trusted, client):
    path = launch_token.launch_token_path()
    assert path.read_text(encoding="utf-8") == trusted

    rotated = launch_token.issue_launch_token()

    assert rotated != trusted
    assert path.read_text(encoding="utf-8") == rotated
    assert client.get("/auth/users", headers={"X-Kani-Client": trusted}).status_code == 401
    assert client.get("/auth/users", headers={"X-Kani-Client": rotated}).status_code == 200
    assert not list(path.parent.glob("*.tmp"))


def test_the_token_file_is_created_exclusively_and_owner_only(monkeypatch, tmp_path):
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    opened: list[tuple[int, int]] = []
    real_open = os.open

    def _recording_open(path, flags, mode=0o777, *args, **kwargs):
        opened.append((flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(launch_token.os, "open", _recording_open)
    try:
        launch_token.issue_launch_token()
    finally:
        launch_token.revoke_launch_token()

    assert len(opened) == 1
    flags, mode = opened[0]
    assert mode == 0o600
    assert flags & os.O_CREAT and flags & os.O_EXCL


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
def test_the_token_file_grants_only_the_current_user(trusted):
    listing = subprocess.run(
        ["icacls", str(launch_token.launch_token_path())], capture_output=True, text=True
    ).stdout

    entries = [line for line in listing.splitlines()[:-1] if ":(" in line]
    assert len(entries) == 1, listing
    assert os.environ["USERNAME"].lower() in entries[0].lower()
    assert "(I)" not in entries[0]


def test_a_failed_private_write_leaves_no_token_and_no_staging_file(monkeypatch, tmp_path):
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))

    def _fail(_path):
        raise launch_token.LaunchTokenError("icacls refused")

    monkeypatch.setattr(launch_token.sys, "platform", "win32")
    monkeypatch.setattr(launch_token, "_restrict_to_current_user", _fail)

    with pytest.raises(launch_token.LaunchTokenError):
        launch_token.issue_launch_token()

    assert launch_token.current_launch_token() is None
    assert list(tmp_path.iterdir()) == []


# ── Lifespan: one token per launch ─────────────────────────────────────────


class _Bridge:
    async def stop(self) -> None:
        return None


class _InstallService:
    async def shutdown(self) -> None:
        return None


async def _noop() -> None:
    return None


@pytest.fixture
def quiet_lifespan(monkeypatch, tmp_path):
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    monkeypatch.setattr(main_module, "initialize_client_environment", lambda: None)
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(main_module, "initialize_skills_registry", _noop)
    monkeypatch.setattr(main_module, "get_runtime_bridge", lambda: _Bridge())
    monkeypatch.setattr(main_module, "get_skill_installation_service", lambda: _InstallService())
    monkeypatch.setattr(main_module, "shutdown_mcp_manager", _noop)
    monkeypatch.setattr(main_module, "close_server_client", _noop)
    yield tmp_path
    launch_token.revoke_launch_token()


@pytest.mark.asyncio
async def test_each_launch_issues_a_new_token_and_shutdown_removes_it(
    quiet_lifespan, monkeypatch
):
    monkeypatch.setattr(client_settings, "trust_checks_enabled", True)
    path = launch_token.launch_token_path()

    async with main_module.lifespan(main_module.app):
        first = path.read_text(encoding="utf-8")
        assert launch_token.launch_token_matches(first)
    assert not path.exists()
    assert launch_token.current_launch_token() is None

    async with main_module.lifespan(main_module.app):
        second = path.read_text(encoding="utf-8")

    assert first != second


@pytest.mark.asyncio
async def test_a_launch_that_cannot_write_the_token_serves_503(quiet_lifespan, monkeypatch):
    monkeypatch.setattr(client_settings, "trust_checks_enabled", True)

    def _fail():
        raise launch_token.LaunchTokenError("icacls refused")

    monkeypatch.setattr(main_module, "issue_launch_token", _fail)

    async with main_module.lifespan(main_module.app):
        client = TestClient(main_module.create_app(), base_url=SIDECAR)
        assert client.get("/auth/users", headers={"X-Kani-Client": "x"}).status_code == 503
        assert client.get("/health").status_code == 200


@pytest.mark.asyncio
async def test_a_launch_with_checks_disabled_warns_and_issues_nothing(
    quiet_lifespan, monkeypatch, caplog
):
    monkeypatch.setattr(client_settings, "trust_checks_enabled", False)

    with caplog.at_level(logging.WARNING, logger="client_backend"):
        async with main_module.lifespan(main_module.app):
            assert not launch_token.launch_token_path().exists()

    assert any(
        record.levelno == logging.WARNING and "CLIENT_TRUST_CHECKS_ENABLED" in record.message
        for record in caplog.records
    )
