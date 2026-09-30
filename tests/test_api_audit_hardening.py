"""Boundary behaviour closed by the 2026-09 API audit, asserted over real HTTP.

Each test here failed before its fix:

* a refresh token or a widget WebSocket token authenticated ordinary routes;
* the feedback routes let user B read, and write, feedback on user A's message
  (the stats and list routes did not authenticate at all);
* a malformed device id, a zero ``page_size`` and a negative ``latestMessages``
  were 500s instead of client errors.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from dependency_injector import providers
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.client_devices import get_device_service
from app.api.client_devices import router as client_devices_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.core.auth import get_current_user, get_current_user_id, get_refresh_token_user_id
from app.core.container import Container, setup_auto_injection
from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.schemas.feedback import FeedbackRead
from app.services.jwt_service import JwtService
from app.services.widget_runtime import WidgetTokenService
from app.utils.exception_handler import register_exception_handlers
from tests.token_state_stub import stub_token_states


@pytest.fixture(autouse=True)
def _restore_wiring():
    setup_auto_injection(Container)
    yield
    setup_auto_injection(Container)


@pytest.fixture(autouse=True)
def _live_token_users(monkeypatch):
    """Every made-up user here is live at token version 0; no users table."""
    stub_token_states(monkeypatch)


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------


def _token_app() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/whoami")
    async def whoami(user_id: UUID = Depends(get_current_user_id)):
        return {"user_id": str(user_id)}

    @app.post("/refresh")
    async def refresh(user_id: UUID = Depends(get_refresh_token_user_id)):
        return {"user_id": str(user_id)}

    return TestClient(app)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_an_access_token_authenticates():
    user_id = uuid4()
    token = JwtService().create_access_token({"sub": str(user_id)})

    response = _token_app().get("/whoami", headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


def test_a_refresh_token_is_not_an_access_token():
    token = JwtService().create_refresh_token({"sub": str(uuid4())})

    response = _token_app().get("/whoami", headers=_bearer(token))

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_TOKEN_TYPE"


def test_a_refresh_token_still_refreshes():
    user_id = uuid4()
    token = JwtService().create_refresh_token({"sub": str(user_id)})

    response = _token_app().post("/refresh", headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


def test_a_widget_token_is_not_an_access_token():
    token, _expires_at = WidgetTokenService().mint(
        widget_id="w-1", session_id=str(uuid4()), user_id=str(uuid4())
    )

    response = _token_app().get("/whoami", headers=_bearer(token))

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_TOKEN_TYPE"


# ---------------------------------------------------------------------------
# Feedback ownership
# ---------------------------------------------------------------------------

OWNER = uuid4()
INTRUDER = uuid4()
MESSAGE_ID = uuid4()


class _Messages:
    """Message access as the real service enforces it: 404 missing, 403 foreign."""

    def get_by_id(self, message_id, user_id):
        if message_id != MESSAGE_ID:
            raise ResourceNotFoundException(detail="Message not found")
        if user_id != OWNER:
            raise AuthorizationException(detail="Access denied to this conversation")
        return SimpleNamespace(id=message_id)


class _Feedback:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.record = FeedbackRead(
            id=uuid4(),
            created_at=now,
            updated_at=now,
            deleted_at=None,
            message_id=MESSAGE_ID,
            user_id=OWNER,
            rating=2,
            comment="private note",
        )
        self.calls: list[str] = []

    def create_feedback(self, data, user_id):
        self.calls.append("create")
        return self.record

    def get_by_message(self, message_id):
        self.calls.append("list")
        return self.record

    def get_message_rating_stats(self, message_id):
        self.calls.append("stats")
        return {"messageId": str(message_id), "rating": 2}

    def get_user_feedback_for_message(self, message_id, user_id):
        self.calls.append("user")
        return self.record if user_id == OWNER else None


@pytest.fixture
def feedback_api():
    feedback = _Feedback()

    def client_for(user_id: UUID | None) -> TestClient:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(feedback_router)
        if user_id is not None:
            app.dependency_overrides[get_current_user_id] = lambda: user_id
        return TestClient(app)

    with (
        Container.message_service.override(providers.Object(_Messages())),
        Container.feedback_service.override(providers.Object(feedback)),
    ):
        yield client_for, feedback


def test_the_owner_reads_feedback_on_their_message(feedback_api):
    client_for, _feedback = feedback_api

    response = client_for(OWNER).get(f"/messages/{MESSAGE_ID}/feedbacks")

    assert response.status_code == 200
    assert response.json()["data"][0]["comment"] == "private note"


@pytest.mark.parametrize(
    "path", ["/feedbacks", "/feedbacks/stats", "/feedbacks/user"], ids=["list", "stats", "user"]
)
def test_another_user_cannot_read_feedback_on_a_message(feedback_api, path):
    client_for, feedback = feedback_api

    response = client_for(INTRUDER).get(f"/messages/{MESSAGE_ID}{path}")

    assert response.status_code == 403
    assert "private note" not in response.text
    assert feedback.calls == []


def test_another_user_cannot_rate_a_message(feedback_api):
    client_for, feedback = feedback_api

    response = client_for(INTRUDER).post(
        f"/messages/{MESSAGE_ID}/feedbacks",
        json={"messageId": str(MESSAGE_ID), "rating": 1},
    )

    assert response.status_code == 403
    assert feedback.calls == []


@pytest.mark.parametrize("path", ["/feedbacks", "/feedbacks/stats"], ids=["list", "stats"])
def test_feedback_reads_require_authentication(feedback_api, path):
    client_for, feedback = feedback_api

    response = client_for(None).get(f"/messages/{MESSAGE_ID}{path}")

    assert response.status_code in {401, 403}
    assert feedback.calls == []


# ---------------------------------------------------------------------------
# Client errors that used to be 500s
# ---------------------------------------------------------------------------


def _devices_client(owner: UUID) -> TestClient:
    device_id = uuid4()
    device = SimpleNamespace(
        id=device_id,
        user_id=owner,
        device_identifier="dev",
        display_name="Laptop",
        platform="windows",
        app_version="1",
        runtime_version="1",
        status=SimpleNamespace(value="online"),
        last_seen_at=None,
        created_at=datetime.now(timezone.utc),
    )
    service = SimpleNamespace(
        repository=SimpleNamespace(get_by_id=lambda _id: device if _id == device_id else None)
    )
    app = FastAPI()
    app.include_router(client_devices_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=owner)
    app.dependency_overrides[get_device_service] = lambda: service
    client = TestClient(app, raise_server_exceptions=False)
    client.device_id = device_id
    return client


def test_a_malformed_device_id_is_a_client_error():
    client = _devices_client(uuid4())

    assert client.get("/client-devices/not-a-uuid").status_code == 422
    assert (
        client.put(
            "/client-devices/not-a-uuid/tool-catalog",
            json={"device_id": str(uuid4()), "catalog": {}},
        ).status_code
        == 422
    )


def test_a_device_id_path_still_resolves_the_owned_device():
    client = _devices_client(uuid4())

    response = client.get(f"/client-devices/{client.device_id}")

    assert response.status_code == 200
    assert response.json()["display_name"] == "Laptop"


def test_document_listing_page_bounds_are_client_errors():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(documents_router)
    app.dependency_overrides[get_current_user_id] = lambda: uuid4()
    conversation_id = uuid4()

    with Container.document_service.override(providers.Object(SimpleNamespace())):
        client = TestClient(app, raise_server_exceptions=False)
        zero_size = client.get(
            f"/documents/conversation/{conversation_id}", params={"page_size": 0}
        )
        zero_page = client.get(f"/documents/conversation/{conversation_id}", params={"page": 0})

    assert zero_size.status_code == 422
    assert zero_page.status_code == 422


def test_a_task_with_no_linked_document_is_not_found(monkeypatch):
    """Such a task has no owner to check, so its status used to go to any caller."""
    import app.api.documents as documents_api

    status_calls: list[str] = []

    class _Processing:
        async def get_processing_status(self, task_id):
            status_calls.append(task_id)
            return {"task_id": task_id, "state": "SUCCESS"}

    class _NoDocuments:
        def __init__(self, _session_factory):
            pass

        def get_by_processing_task_id(self, _task_id):
            return None

    monkeypatch.setattr(documents_api, "DocumentRepository", _NoDocuments)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(documents_router)
    app.dependency_overrides[get_current_user_id] = lambda: uuid4()
    document_service = SimpleNamespace(repository=SimpleNamespace(session_factory=None))

    with (
        Container.document_service.override(providers.Object(document_service)),
        Container.document_processing_service.override(providers.Object(_Processing())),
    ):
        response = TestClient(app, raise_server_exceptions=False).get("/documents/task/t-1")

    assert response.status_code == 404
    assert "SUCCESS" not in response.text
    assert status_calls == []


@pytest.mark.parametrize(
    "path", ["/conversations/{id}", "/ai/conversations/{id}"], ids=["canonical", "ai_sdk"]
)
def test_deleting_a_conversation_runs_the_sync_service_off_the_event_loop(path):
    """An ``async def`` route calling sync DB code blocked every other request."""
    from app.api.ai_sdk import router as ai_sdk_router

    service_threads: list[int] = []

    class _Conversations:
        def delete_conversation(self, conversation_id, user_id):
            service_threads.append(threading.get_ident())

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(conversations_router)
    app.include_router(ai_sdk_router)
    app.dependency_overrides[get_current_user_id] = lambda: uuid4()

    @app.get("/loop-thread")
    async def loop_thread():
        return {"ident": threading.get_ident()}

    with (
        Container.conversation_service.override(providers.Object(_Conversations())),
        TestClient(app) as client,
    ):
        loop_ident = client.get("/loop-thread").json()["ident"]
        response = client.delete(path.format(id=uuid4()))

    assert response.status_code == 200
    assert response.json()["message"] == "Conversation deleted successfully"
    assert len(service_threads) == 1
    assert service_threads[0] != loop_ident


@pytest.mark.parametrize("latest", [-1, 101])
def test_latest_messages_is_bounded(latest):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(conversations_router)
    app.dependency_overrides[get_current_user_id] = lambda: uuid4()

    with (
        Container.conversation_service.override(providers.Object(SimpleNamespace())),
        Container.project_service.override(providers.Object(SimpleNamespace())),
    ):
        response = TestClient(app, raise_server_exceptions=False).get(
            "/conversations/", params={"latestMessages": latest}
        )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Widget lookup
# ---------------------------------------------------------------------------


def test_widget_lookup_treats_like_wildcards_in_the_id_as_literals():
    """An id of ``%`` used to match every message of the caller that had metadata."""
    from sqlalchemy.dialects import postgresql

    from app.api.widgets import _widget_messages_statement

    compiled = _widget_messages_statement(uuid4(), "w_1%").compile(
        dialect=postgresql.dialect()
    )

    assert "w/_1/%" in compiled.params.values()
    assert "ESCAPE '/'" in str(compiled)
