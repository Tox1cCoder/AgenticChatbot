"""The sidecar's CORS surface is an explicit allowlist, in every environment.

This file used to assert the opposite -- that any origin, including an external
one, was allowed in both development and production. That is unsafe for this
process specifically: the sidecar executes local shell commands, reads and writes
the filesystem, installs skill bundles, and holds an upstream session, so a
wildcard let any page the user visited drive all of it from the browser.

The replacement contract keeps every origin the product actually ships (Next.js
dev server, Streamlit, and the Tauri desktop shell's custom-protocol and
Windows/dev origins) and refuses everything else. Environment no longer changes
the answer -- a permissive development mode is what leaks into production
installs.
"""

import os

import pytest
from fastapi.testclient import TestClient

from client_backend.core.config import get_client_settings


def _preflight(client: TestClient, path: str, *, origin: str, method: str):
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


def _create_app_with_env(environment: str):
    """Create a fresh app with the given environment, bypassing lru_cache."""
    os.environ["CLIENT_ENVIRONMENT"] = environment
    get_client_settings.cache_clear()
    from client_backend.main import create_app

    return create_app()


@pytest.fixture()
def restore_environment():
    yield
    os.environ["CLIENT_ENVIRONMENT"] = "development"
    get_client_settings.cache_clear()


@pytest.mark.parametrize("environment", ["development", "production"])
@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
        "http://localhost:8501",
        "tauri://localhost",
        "http://tauri.localhost",
        "http://tauri.localhost:1420",
    ],
)
def test_shipped_frontend_origins_are_allowed(restore_environment, environment, origin):
    """Every origin the product ships from must preflight successfully."""
    app = _create_app_with_env(environment)
    with TestClient(app) as client:
        response = _preflight(client, "/ai/conversations/123", origin=origin, method="PATCH")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "PATCH" in response.headers["access-control-allow-methods"]
    assert "authorization" in response.headers.get("access-control-allow-headers", "").lower()


@pytest.mark.parametrize("environment", ["development", "production"])
@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example.com",
        "http://localhost:9999",
        "http://tauri.localhost.evil.example",
    ],
)
def test_unconfigured_origins_receive_no_allow_origin_header(
    restore_environment, environment, origin
):
    """An origin outside the allowlist must not be granted access, in any environment."""
    app = _create_app_with_env(environment)
    with TestClient(app) as client:
        response = _preflight(client, "/ai/conversations/123", origin=origin, method="PATCH")

    assert "access-control-allow-origin" not in response.headers


def test_private_network_access_stays_enabled(restore_environment):
    """A browser on a public origin cannot reach loopback without this opt-in."""
    app = _create_app_with_env("production")
    with TestClient(app) as client:
        response = client.options(
            "/ai/conversations/123",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Private-Network": "true",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-private-network") == "true"
