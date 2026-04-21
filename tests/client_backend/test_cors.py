import os

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


def test_cors_preflight_allows_any_origin_in_development():
    """In development mode (the default), CORS is wide-open."""
    app = _create_app_with_env("development")
    with TestClient(app) as client:
        response = _preflight(
            client,
            "/ai/conversations/123",
            origin="http://[::1]:3000",
            method="PATCH",
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_preflight_allows_localhost_in_production():
    """In production mode, CORS remains permissive."""
    app = _create_app_with_env("production")
    try:
        with TestClient(app) as client:
            response = _preflight(
                client,
                "/ai/conversations/123",
                origin="http://localhost:3000",
                method="PATCH",
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
    finally:
        # Reset to development so other tests aren't affected
        os.environ["CLIENT_ENVIRONMENT"] = "development"
        get_client_settings.cache_clear()


def test_cors_preflight_allows_tauri_localhost_http_in_production():
    """In production mode, Tauri dev origins are allowed."""
    app = _create_app_with_env("production")
    try:
        with TestClient(app) as client:
            response = _preflight(
                client,
                "/ai/conversations/123",
                origin="http://tauri.localhost:1420",
                method="PATCH",
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
    finally:
        os.environ["CLIENT_ENVIRONMENT"] = "development"
        get_client_settings.cache_clear()


def test_cors_preflight_allows_tauri_scheme_in_production():
    """In production mode, Tauri custom protocol origins are allowed."""
    app = _create_app_with_env("production")
    try:
        with TestClient(app) as client:
            response = _preflight(
                client,
                "/ai/conversations/123",
                origin="tauri://localhost",
                method="PATCH",
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
    finally:
        os.environ["CLIENT_ENVIRONMENT"] = "development"
        get_client_settings.cache_clear()


def test_cors_preflight_allows_non_local_origin_in_production():
    """In production mode, external origins are allowed."""
    app = _create_app_with_env("production")
    try:
        with TestClient(app) as client:
            response = _preflight(
                client,
                "/ai/conversations/123",
                origin="https://evil.example.com",
                method="PATCH",
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
    finally:
        os.environ["CLIENT_ENVIRONMENT"] = "development"
        get_client_settings.cache_clear()
