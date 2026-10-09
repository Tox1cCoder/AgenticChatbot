"""HTTP contract of ``POST /conversations/{id}/fork`` on the real application.

The route is driven through ``app.main.app`` so the registered exception
handlers, the real JWT dependency and the real ``AppAutoInjector`` wiring are
all exercised. Only the container's fork service provider is overridden, and
the service itself is the real one over a fake repository and coordinator, so
the error envelopes come from the real translation code.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error
from app.core.container import Container, container, setup_auto_injection
from app.main import app
from app.repositories.conversation_fork import (
    ConversationForkRepository,
    ForkError,
    ForkErrorCode,
)
from app.services.conversation_fork_service import ConversationForkService
from tests.token_state_stub import stub_token_states

# Importing app.main points the injector maps at the container instance; routes
# already captured their providers, so restore the declarative baseline.
setup_auto_injection(Container)

_OWNER = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_SOURCE = UUID("827faf55-1041-4357-9e1a-0f7d031fa546")
_MESSAGE = UUID("11111111-1111-4111-8111-111111111111")
_REQUEST = UUID("22222222-2222-4222-8222-222222222222")
_URL = f"/conversations/{_SOURCE}/fork"


@pytest.fixture(autouse=True)
def _isolate_injector_wiring(monkeypatch) -> Iterator[None]:
    setup_auto_injection(Container)
    stub_token_states(monkeypatch)
    try:
        yield
    finally:
        setup_auto_injection(Container)


def _headers(user_id: UUID = _OWNER) -> dict[str, str]:
    token = container.jwt_service().create_access_token({"sub": str(user_id)})
    return {"Authorization": f"Bearer {token}"}


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"messageId": str(_MESSAGE), "requestId": str(_REQUEST)}
    body.update(overrides)
    return body


def _child() -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        created_at=now,
        updated_at=now,
        deleted_at=None,
        owner_id=_OWNER,
        title="Source (fork)",
        persona_prompt=None,
        planning_mode_enabled=False,
        plan_lifecycle=None,
        project_id=None,
        forked_from_conversation_id=_SOURCE,
        forked_from_message_id=_MESSAGE,
        forked_through_sequence=4,
    )


class _Repository:
    def __init__(self, *, replay: Any = None, result: Any = None, error: Exception | None = None):
        self.replay = replay
        self.result = result
        self.error = error
        self.plans: list[Any] = []

    async def apreflight(self, *, plan):
        self.plans.append(plan)
        return self.replay

    async def afork_snapshot(self, *, plan):
        if self.error is not None:
            raise self.error
        return self.result, True


class _Coordinator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    @contextlib.asynccontextmanager
    async def hold(self, conversation_id, *, request_id):
        if self.error is not None:
            raise self.error
        yield


def _service(repository: Any, coordinator: Any = None) -> ConversationForkService:
    return ConversationForkService(
        repository=repository,
        turn_coordinator=coordinator or _Coordinator(),
        compaction_publisher=None,
        lock_timeout_seconds=1.0,
        max_messages=10,
        max_bytes=1000,
    )


@contextlib.contextmanager
def _serving(service: ConversationForkService) -> Iterator[TestClient]:
    with Container.conversation_fork_service.override(providers.Object(service)):
        yield TestClient(app)


def _error_envelope(response) -> dict[str, Any]:
    body = response.json()
    assert body["success"] is False
    return body


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_the_real_app_registers_the_fork_route_and_generates_openapi():
    schema = app.openapi()

    operation = schema["paths"]["/conversations/{conversation_id}/fork"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]
    request_schema = schema["components"]["schemas"]["ConversationForkRequest"]
    assert {"messageId", "requestId"} <= set(request_schema["required"])
    assert {"200", "201"} <= operation["responses"].keys()


def test_the_injected_service_is_resolved_through_the_route():
    child = _child()
    repository = _Repository(result=child)

    with _serving(_service(repository)) as client:
        response = client.post(_URL, json=_body(), headers=_headers())

    assert response.status_code == 201, response.text
    assert repository.plans[0].source_conversation_id == _SOURCE
    assert repository.plans[0].owner_id == _OWNER


# ---------------------------------------------------------------------------
# success and replay
# ---------------------------------------------------------------------------


def test_a_new_fork_returns_201_with_the_envelope_and_lineage():
    child = _child()

    with _serving(_service(_Repository(result=child))) as client:
        response = client.post(_URL, json=_body(title="Alt"), headers=_headers())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == str(child.id)
    assert body["data"]["forkedFromConversationId"] == str(_SOURCE)
    assert body["data"]["forkedFromMessageId"] == str(_MESSAGE)
    assert body["data"]["forkedThroughSequence"] == 4


def test_an_identical_retry_returns_the_same_child_with_200():
    child = _child()

    with _serving(_service(_Repository(replay=child))) as client:
        response = client.post(_URL, json=_body(), headers=_headers())

    assert response.status_code == 200, response.text
    assert response.json()["data"]["id"] == str(child.id)


def test_snake_case_field_names_are_accepted_too():
    repository = _Repository(result=_child())

    with _serving(_service(repository)) as client:
        response = client.post(
            _URL,
            json={"message_id": str(_MESSAGE), "request_id": str(_REQUEST)},
            headers=_headers(),
        )

    assert response.status_code == 201, response.text


# ---------------------------------------------------------------------------
# authentication and malformed input
# ---------------------------------------------------------------------------


def test_fork_requires_authentication():
    repository = _Repository(result=_child())

    with _serving(_service(repository)) as client:
        response = client.post(_URL, json=_body())

    assert response.status_code == 401
    assert repository.plans == []


def test_the_owner_comes_from_the_token_never_from_the_body():
    other = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    repository = _Repository(result=_child())

    with _serving(_service(repository)) as client:
        client.post(
            _URL, json=_body(ownerId=str(other), owner_id=str(other)), headers=_headers(_OWNER)
        )

    assert repository.plans[0].owner_id == _OWNER


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"requestId": str(_REQUEST)},
        {"messageId": str(_MESSAGE)},
        _body(messageId="not-a-uuid"),
        _body(requestId="not-a-uuid"),
        _body(title=""),
        _body(title="x" * 256),
    ],
)
def test_a_malformed_body_is_rejected_with_422_before_the_service_runs(body):
    repository = _Repository(result=_child())

    with _serving(_service(repository)) as client:
        response = client.post(_URL, json=body, headers=_headers())

    assert response.status_code == 422
    assert repository.plans == []


def test_a_malformed_conversation_id_is_rejected_with_422():
    repository = _Repository(result=_child())

    with _serving(_service(repository)) as client:
        response = client.post("/conversations/not-a-uuid/fork", json=_body(), headers=_headers())

    assert response.status_code == 422
    assert repository.plans == []


# ---------------------------------------------------------------------------
# error envelopes through the registered exception handlers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ForkErrorCode.SOURCE_NOT_FOUND, 404),
        (ForkErrorCode.SOURCE_ACCESS_DENIED, 403),
        (ForkErrorCode.MESSAGE_NOT_FOUND, 404),
        (ForkErrorCode.POINT_NOT_COMPLETED, 409),
        (ForkErrorCode.REQUEST_CONFLICT, 409),
        (ForkErrorCode.RESULT_DELETED, 410),
        (ForkErrorCode.DOCUMENT_CONTEXT_UNSUPPORTED, 409),
        (ForkErrorCode.HISTORY_TOO_LARGE, 413),
        (ForkErrorCode.TURN_CONFLICT, 409),
    ],
)
def test_every_fork_refusal_is_a_typed_envelope(code, status):
    service = _service(_Repository(error=ForkError(code)))

    with _serving(service) as client:
        response = client.post(_URL, json=_body(), headers=_headers())

    assert response.status_code == status
    body = _error_envelope(response)
    assert body["code"] == code.value
    assert body["message"]


def test_a_coordinator_turn_conflict_is_translated_to_a_409_envelope():
    conflict = WorkflowRoutingException(
        workflow_error("conversation_turn_conflict", request_id="r-1")
    )
    service = _service(_Repository(result=_child()), _Coordinator(error=conflict))

    with _serving(service) as client:
        response = client.post(_URL, json=_body(), headers=_headers())

    assert response.status_code == 409
    assert _error_envelope(response)["code"] == "conversation_turn_conflict"


class _PgError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


def _repository_failing_with(sqlstate: str) -> ConversationForkRepository:
    """The real repository whose transaction always fails with ``sqlstate``."""
    repository = ConversationForkRepository(
        session_factory=lambda: None, async_session_factory=lambda: None
    )

    async def _always_fail(_work):
        raise DBAPIError("fork", {}, _PgError(sqlstate))

    repository._arun_refusable = _always_fail  # noqa: SLF001 - the DB boundary

    async def _no_replay(*, plan):
        return None

    repository.apreflight = _no_replay
    return repository


@pytest.mark.parametrize("sqlstate", ["55P03", "40001"])
def test_row_lock_timeout_and_exhausted_serialization_retries_are_409_turn_conflicts(sqlstate):
    service = _service(_repository_failing_with(sqlstate))

    with _serving(service) as client:
        response = client.post(_URL, json=_body(), headers=_headers())

    assert response.status_code == 409
    assert _error_envelope(response)["code"] == "conversation_turn_conflict"


# ---------------------------------------------------------------------------
# neighbours are unchanged
# ---------------------------------------------------------------------------


def test_the_conversation_routes_next_to_fork_are_still_registered():
    paths = app.openapi()["paths"]

    assert "get" in paths["/conversations/{conversation_id}"]
    assert "patch" in paths["/conversations/{conversation_id}"]
    assert "delete" in paths["/conversations/{conversation_id}"]
    assert "post" in paths["/conversations/"]
    assert "get" in paths["/conversations/{conversation_id}/messages"]
