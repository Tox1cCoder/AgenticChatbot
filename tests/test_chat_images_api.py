from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_images import _get_repository, _get_service, router
from app.core.auth import get_current_user_id


@pytest.fixture
def client_and_state():
    app = FastAPI()
    app.include_router(router)

    user_id = uuid4()
    image_id = uuid4()
    record = SimpleNamespace(
        id=image_id, user_id=user_id, content_type="image/png", storage_path="ab/x.png"
    )

    class _Repo:
        def get_for_user(self, iid, uid):
            return record if (iid == image_id and uid == user_id) else None

    class _Svc:
        def read_bytes(self, rec):
            return b"PNGBYTES"

    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[_get_repository] = lambda: _Repo()
    app.dependency_overrides[_get_service] = lambda: _Svc()
    return TestClient(app), image_id


def test_get_image_returns_bytes(client_and_state):
    client, image_id = client_and_state
    resp = client.get(f"/chat-images/{image_id}")
    assert resp.status_code == 200
    assert resp.content == b"PNGBYTES"
    assert resp.headers["content-type"].startswith("image/png")


def test_get_unknown_image_404(client_and_state):
    client, _ = client_and_state
    resp = client.get(f"/chat-images/{uuid4()}")
    assert resp.status_code == 404
