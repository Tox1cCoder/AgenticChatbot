"""Authenticated media-route behavior for selected web images."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.web_images import _get_repository, _get_service, router
from app.core.auth import get_current_user_id
from app.services.web_image_service import (
    FetchedWebImage,
    WebImageRejected,
    WebImageUpstreamFailure,
)


@pytest.fixture
def client_and_state():
    app = FastAPI()
    app.include_router(router)
    user_id = uuid4()
    image_id = uuid4()
    state = {
        "record": SimpleNamespace(id=image_id, user_id=user_id),
        "result": FetchedWebImage(
            content=b"PNGBYTES",
            media_type="image/png",
            width=20,
            height=10,
        ),
    }

    class Repository:
        async def aget_for_user(self, requested_image_id, requested_user_id):
            if requested_image_id != image_id or requested_user_id != user_id:
                return None
            return state["record"]

    class Service:
        async def fetch(self, _record):
            result = state["result"]
            if isinstance(result, Exception):
                raise result
            return result

    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[_get_repository] = Repository
    app.dependency_overrides[_get_service] = Service
    return TestClient(app), image_id, state


def test_web_image_requires_authentication():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app, raise_server_exceptions=False).get(f"/web-images/{uuid4()}")

    assert response.status_code in {401, 403}


def test_web_image_hides_missing_or_other_user_reference(client_and_state):
    client, _image_id, _state = client_and_state

    response = client.get(f"/web-images/{uuid4()}")

    assert response.status_code == 404


def test_web_image_success_returns_safe_private_response(client_and_state):
    client, image_id, _state = client_and_state

    response = client.get(f"/web-images/{image_id}")

    assert response.status_code == 200
    assert response.content == b"PNGBYTES"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, max-age=300"
    assert response.headers["content-security-policy"] == "default-src 'none'"


@pytest.mark.parametrize(
    ("error", "expected_status"),
    (
        (WebImageUpstreamFailure("timeout"), 504),
        (WebImageUpstreamFailure("status"), 502),
        (WebImageRejected("mime"), 502),
    ),
)
def test_web_image_failure_is_generic_and_bounded(client_and_state, error, expected_status):
    client, image_id, state = client_and_state
    state["result"] = error

    response = client.get(f"/web-images/{image_id}")

    assert response.status_code == expected_status
    assert response.json() == {"detail": "Visual unavailable"}
    assert "upstream" not in response.text.lower()


def test_main_application_registers_web_image_route():
    from app.main import app

    assert "/web-images/{image_id}" in app.openapi()["paths"]
