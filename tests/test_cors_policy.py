"""The API's CORS policy follows ``settings.cors_origins``.

``*`` means any origin without credentials; an explicit list means exactly
those origins, with credentials; an empty list is the local-development
default and keeps any origin without credentials.
"""

from fastapi.testclient import TestClient

import app.main as main


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_wildcard_never_combines_any_origin_with_credentials(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", ["*"])
    response = _preflight(TestClient(main.create_app()), "https://evil.example")
    assert response.headers.get("access-control-allow-credentials") != "true"


def test_explicit_origins_are_the_only_ones_allowed(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", ["http://localhost:8501"])
    client = TestClient(main.create_app())
    allowed = _preflight(client, "http://localhost:8501")
    refused = _preflight(client, "https://evil.example")
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert allowed.headers.get("access-control-allow-credentials") == "true"
    assert "access-control-allow-origin" not in refused.headers


def test_an_empty_list_keeps_any_origin_without_credentials(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", [])
    response = _preflight(TestClient(main.create_app()), "http://localhost:8501")
    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") != "true"


def test_a_wildcard_mixed_into_a_list_is_still_a_wildcard(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", ["http://localhost:8501", "*"])
    response = _preflight(TestClient(main.create_app()), "https://evil.example")
    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") != "true"
