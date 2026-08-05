from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.tool_result_blobs import _get_repository, _get_service, router
from app.core.auth import get_current_user_id


@pytest.fixture
def client_and_state():
    app = FastAPI()
    app.include_router(router)

    user_id = uuid4()
    blob_id = uuid4()
    record = SimpleNamespace(id=blob_id, user_id=user_id, content_type="text/plain")

    class _Repo:
        def get_for_user(self, bid, uid):
            return record if (bid == blob_id and uid == user_id) else None

    state = {"read_text": lambda rec: "full text"}

    class _Svc:
        def read_text(self, rec):
            return state["read_text"](rec)

    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[_get_repository] = lambda: _Repo()
    app.dependency_overrides[_get_service] = lambda: _Svc()
    return TestClient(app), blob_id, state


def test_get_blob_returns_text(client_and_state):
    client, blob_id, _ = client_and_state
    resp = client.get(f"/tool-results/{blob_id}")
    assert resp.status_code == 200
    assert resp.text == "full text"


def test_get_unknown_blob_404(client_and_state):
    client, _, _ = client_and_state
    resp = client.get(f"/tool-results/{uuid4()}")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Tool result not found"


def test_get_corrupt_blob_returns_404_not_500(client_and_state):
    """A record with neither content nor storage_path raises ValueError from
    ``read_text``. That must not leak as an unhandled 500 with a stack trace —
    the tool path for the same failure already returns a clean not-found.
    """
    client, blob_id, state = client_and_state

    def _raise_value_error(rec):
        raise ValueError(f"Tool result blob {rec.id} has neither content nor storage_path")

    state["read_text"] = _raise_value_error

    resp = client.get(f"/tool-results/{blob_id}")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Tool result not found"


def test_get_blob_with_missing_legacy_file_returns_404_not_500(client_and_state):
    """A legacy on-disk blob whose file vanished raises OSError from
    ``read_text``. That must also surface as a clean not-found, not a 500.
    """
    client, blob_id, state = client_and_state

    def _raise_os_error(rec):
        raise OSError("legacy blob file is missing")

    state["read_text"] = _raise_os_error

    resp = client.get(f"/tool-results/{blob_id}")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Tool result not found"
