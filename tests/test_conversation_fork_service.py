"""Unit coverage for the fork service boundary and the pure fork rules.

The transaction itself is exercised against real PostgreSQL in
``tests/integration/test_conversation_fork_postgres.py``. These tests pin what
does not need a database: request identity, the default title, which source
rows are a completed fork point, how every failure becomes an HTTP error, and
that the service is resolvable from the application's container.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error
from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.core.exceptions.http import CustomHTTPException
from app.models.enums import MessageRole
from app.models.generation import GenerationStatus
from app.repositories.conversation_fork import (
    ForkError,
    ForkErrorCode,
    compute_fork_eligibility,
    default_fork_title,
)
from app.schemas.conversation import ConversationForkRequest, ConversationRead
from app.services.conversation_fork_service import (
    ConversationForkService,
    fork_request_fingerprint,
)

SOURCE_ID = UUID("827faf55-1041-4357-9e1a-0f7d031fa546")
MESSAGE_ID = UUID("11111111-1111-4111-8111-111111111111")


# ---------------------------------------------------------------------------
# request identity and title
# ---------------------------------------------------------------------------


def test_fingerprint_is_sha256_of_canonical_json_including_a_null_title():
    expected_json = (
        '{"message_id":"11111111-1111-4111-8111-111111111111",'
        '"source_conversation_id":"827faf55-1041-4357-9e1a-0f7d031fa546",'
        '"title":null}'
    )

    fingerprint = fork_request_fingerprint(
        source_conversation_id=SOURCE_ID, message_id=MESSAGE_ID, title=None
    )

    assert fingerprint == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert len(fingerprint) == 64


def test_fingerprint_distinguishes_every_requested_field():
    base = {"source_conversation_id": SOURCE_ID, "message_id": MESSAGE_ID, "title": None}
    variants = [
        base,
        {**base, "title": "Alt"},
        {**base, "message_id": uuid4()},
        {**base, "source_conversation_id": uuid4()},
    ]

    assert len({fork_request_fingerprint(**variant) for variant in variants}) == 4


def test_fingerprint_hashes_non_ascii_titles_deterministically():
    first = fork_request_fingerprint(
        source_conversation_id=SOURCE_ID, message_id=MESSAGE_ID, title="Nhánh thử"
    )
    again = fork_request_fingerprint(
        source_conversation_id=SOURCE_ID, message_id=MESSAGE_ID, title="Nhánh thử"
    )

    assert first == again


def test_default_title_appends_fork_suffix():
    assert default_fork_title("Trip planning") == "Trip planning (fork)"


def test_default_title_trims_source_to_248_characters_so_it_fits_255():
    title = default_fork_title("x" * 255)

    assert title == "x" * 248 + " (fork)"
    assert len(title) == 255


# ---------------------------------------------------------------------------
# fork-point eligibility
# ---------------------------------------------------------------------------


def _row(
    *,
    sender: int = MessageRole.assistant.value,
    content: str = "answer",
    metadata: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(), sender=sender, content=content, message_metadata=metadata or {}
    )


def _generation(
    *,
    status: GenerationStatus,
    assistant_message_id: UUID | None,
    continuation_available: bool = False,
    logical_turn_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        status=status,
        assistant_message_id=assistant_message_id,
        continuation_available=continuation_available,
        logical_turn_id=logical_turn_id or str(uuid4()),
    )


def test_completed_generation_makes_its_terminal_reply_eligible():
    reply = _row()
    generation = _generation(status=GenerationStatus.COMPLETED, assistant_message_id=reply.id)

    assert compute_fork_eligibility([reply], [generation]) == {reply.id: True}


def test_user_rows_are_never_fork_points():
    question = _row(sender=MessageRole.user.value)

    assert compute_fork_eligibility([question], []) == {}


@pytest.mark.parametrize(
    ("status", "continuation_available", "eligible"),
    [
        (GenerationStatus.COMPLETED, False, True),
        (GenerationStatus.COMPLETED_PARTIAL, False, True),
        (GenerationStatus.COMPLETED_PARTIAL, True, False),
        (GenerationStatus.CONTINUABLE, True, False),
        (GenerationStatus.CONTINUABLE, False, False),
        (GenerationStatus.STOPPED, True, False),
        (GenerationStatus.FAILED, False, False),
        (GenerationStatus.RUNNING, False, False),
    ],
)
def test_only_finished_generations_without_a_live_continuation_are_eligible(
    status, continuation_available, eligible
):
    reply = _row(metadata={"partial": True})
    generation = _generation(
        status=status,
        assistant_message_id=reply.id,
        continuation_available=continuation_available,
    )

    assert compute_fork_eligibility([reply], [generation])[reply.id] is eligible


def test_continuation_chain_accepts_only_the_final_epoch():
    turn = str(uuid4())
    first = _row(
        metadata={"partial": True, "continuable": True, "generation_id": None},
    )
    final = _row(metadata={"partial": True, "continued": True})
    generation = _generation(
        status=GenerationStatus.COMPLETED_PARTIAL,
        assistant_message_id=final.id,
        logical_turn_id=turn,
    )
    first.message_metadata["generation_id"] = str(generation.id)
    first.message_metadata["logical_turn_id"] = turn
    final.message_metadata["generation_id"] = str(generation.id)

    result = compute_fork_eligibility([first, final], [generation])

    assert result == {first.id: False, final.id: True}


def test_a_reply_matched_only_by_its_turn_is_not_the_terminal_message():
    question = _row(sender=MessageRole.user.value)
    stale = _row(metadata={"reply_to_user_message_id": str(question.id)})
    generation = _generation(
        status=GenerationStatus.COMPLETED,
        assistant_message_id=uuid4(),
        logical_turn_id=str(question.id),
    )

    assert compute_fork_eligibility([question, stale], [generation]) == {stale.id: False}


@pytest.mark.parametrize(
    ("metadata", "eligible"),
    [
        ({}, True),
        ({"partial": True}, True),
        ({"partial": True, "continued": True}, True),
        ({"paused": True}, False),
        ({"interrupt": {"interrupt_id": "i-1"}}, False),
        ({"stopped": True, "partial": True}, False),
        ({"continuable": True}, False),
        ({"generation_id": str(uuid4())}, False),
        ({"continuation_id": str(uuid4())}, False),
        ({"logical_turn_id": "turn-1"}, False),
    ],
)
def test_legacy_reply_without_a_generation_is_judged_by_its_metadata(metadata, eligible):
    reply = _row(metadata=metadata)

    assert compute_fork_eligibility([reply], [])[reply.id] is eligible


def test_hidden_or_empty_replies_are_never_eligible():
    hidden = _row(content="  ", metadata={"paused": True})
    empty = _row(content="")

    result = compute_fork_eligibility([hidden, empty], [])

    assert result == {hidden.id: False, empty.id: False}


@pytest.mark.parametrize(
    ("metadata", "eligible"),
    [
        ({"fork_history": True, "fork_cutoff_eligible": True}, True),
        ({"fork_history": True, "fork_cutoff_eligible": False, "partial": True}, False),
        ({"fork_history": True}, False),
    ],
)
def test_copied_rows_use_their_recorded_eligibility(metadata, eligible):
    reply = _row(metadata=metadata)

    assert compute_fork_eligibility([reply], [])[reply.id] is eligible


# ---------------------------------------------------------------------------
# service boundary
# ---------------------------------------------------------------------------


def _child(**overrides: Any) -> SimpleNamespace:
    now = datetime.now(UTC)
    values = {
        "id": uuid4(),
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "owner_id": uuid4(),
        "title": "Source (fork)",
        "persona_prompt": None,
        "planning_mode_enabled": False,
        "plan_lifecycle": None,
        "project_id": None,
        "forked_from_conversation_id": SOURCE_ID,
        "forked_from_message_id": MESSAGE_ID,
        "forked_through_sequence": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Repository:
    def __init__(self, *, replay: Any = None, result: Any = None, error: Exception | None = None):
        self.replay = replay
        self.result = result
        self.error = error
        self.plans: list[Any] = []

    async def apreflight(self, *, plan):
        self.plans.append(("preflight", plan))
        return self.replay

    async def afork_snapshot(self, *, plan):
        self.plans.append(("fork", plan))
        if self.error is not None:
            raise self.error
        return self.result, True


class _Coordinator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.held: list[tuple[str, str]] = []

    @contextlib.asynccontextmanager
    async def hold(self, conversation_id, *, request_id):
        self.held.append((conversation_id, request_id))
        if self.error is not None:
            raise self.error
        yield


def _request(title: str | None = None) -> ConversationForkRequest:
    return ConversationForkRequest(message_id=MESSAGE_ID, request_id=uuid4(), title=title)


def _service(repository, coordinator=None, publisher=None) -> ConversationForkService:
    return ConversationForkService(
        repository=repository,
        turn_coordinator=coordinator or _Coordinator(),
        compaction_publisher=publisher,
        lock_timeout_seconds=7.5,
        max_messages=10,
        max_bytes=1000,
    )


async def test_new_fork_runs_under_the_source_turn_lock_and_publishes_compaction():
    child = _child()
    repository = _Repository(result=child)
    coordinator = _Coordinator()
    published: list[UUID] = []
    request = _request("Alt")
    owner = uuid4()

    read, created = await _service(repository, coordinator, published.append).fork_conversation(
        source_conversation_id=SOURCE_ID, owner_id=owner, request=request
    )

    assert created is True
    assert isinstance(read, ConversationRead)
    assert read.id == child.id and read.forked_through_sequence == 4
    assert coordinator.held == [(str(SOURCE_ID), str(request.request_id))]
    assert published == [child.id]
    plan = repository.plans[-1][1]
    assert (plan.owner_id, plan.message_id, plan.title) == (owner, MESSAGE_ID, "Alt")
    assert plan.fingerprint == fork_request_fingerprint(
        source_conversation_id=SOURCE_ID, message_id=MESSAGE_ID, title="Alt"
    )
    assert (plan.lock_timeout_seconds, plan.max_messages, plan.max_bytes) == (7.5, 10, 1000)


async def test_replay_returns_the_existing_child_without_taking_the_lock_or_publishing():
    child = _child()
    repository = _Repository(replay=child)
    coordinator = _Coordinator()
    published: list[UUID] = []

    read, created = await _service(repository, coordinator, published.append).fork_conversation(
        source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
    )

    assert (read.id, created) == (child.id, False)
    assert coordinator.held == []
    assert published == []
    assert [step for step, _ in repository.plans] == ["preflight"]


async def test_publisher_failure_does_not_fail_a_committed_fork(caplog):
    child = _child()

    def broken_publisher(_conversation_id):
        raise RuntimeError("broker down")

    with caplog.at_level(logging.WARNING):
        read, created = await _service(
            _Repository(result=child), publisher=broken_publisher
        ).fork_conversation(source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request())

    assert (read.id, created) == (child.id, True)
    assert "broker_publish_failed" in caplog.text


async def test_turn_coordinator_conflict_becomes_a_409_with_the_lowercase_code():
    conflict = WorkflowRoutingException(
        workflow_error("conversation_turn_conflict", request_id="r-1")
    )
    service = _service(_Repository(result=_child()), _Coordinator(error=conflict))

    with pytest.raises(CustomHTTPException) as raised:
        await service.fork_conversation(
            source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
        )

    assert raised.value.status_code == 409
    assert raised.value.error_code == "conversation_turn_conflict"


async def test_other_workflow_errors_are_not_disguised_as_turn_conflicts():
    other = WorkflowRoutingException(workflow_error("finalization_failed", request_id="r-1"))
    service = _service(_Repository(result=_child()), _Coordinator(error=other))

    with pytest.raises(WorkflowRoutingException):
        await service.fork_conversation(
            source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
        )


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ForkErrorCode.MESSAGE_NOT_FOUND, 404),
        (ForkErrorCode.POINT_NOT_COMPLETED, 409),
        (ForkErrorCode.REQUEST_CONFLICT, 409),
        (ForkErrorCode.RESULT_DELETED, 410),
        (ForkErrorCode.DOCUMENT_CONTEXT_UNSUPPORTED, 409),
        (ForkErrorCode.HISTORY_TOO_LARGE, 413),
        (ForkErrorCode.TURN_CONFLICT, 409),
    ],
)
async def test_fork_errors_map_to_their_http_status_and_code(code, status):
    service = _service(_Repository(error=ForkError(code)))

    with pytest.raises(CustomHTTPException) as raised:
        await service.fork_conversation(
            source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
        )

    assert raised.value.status_code == status
    assert raised.value.error_code == code.value


def test_turn_conflict_keeps_the_existing_lowercase_workflow_code():
    assert ForkErrorCode.TURN_CONFLICT.value == "conversation_turn_conflict"


@pytest.mark.parametrize(
    ("code", "exception_type", "status", "error_code"),
    [
        (
            ForkErrorCode.SOURCE_NOT_FOUND,
            ResourceNotFoundException,
            404,
            "CONVERSATION_NOT_FOUND",
        ),
        (
            ForkErrorCode.SOURCE_ACCESS_DENIED,
            AuthorizationException,
            403,
            "CONVERSATION_ACCESS_DENIED",
        ),
    ],
)
async def test_source_errors_reuse_the_existing_conversation_conventions(
    code, exception_type, status, error_code
):
    service = _service(_Repository(error=ForkError(code)))

    with pytest.raises(exception_type) as raised:
        await service.fork_conversation(
            source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
        )

    assert (raised.value.status_code, raised.value.error_code) == (status, error_code)


async def test_unrelated_errors_propagate_untranslated():
    service = _service(_Repository(error=RuntimeError("disk on fire")))

    with pytest.raises(RuntimeError, match="disk on fire"):
        await service.fork_conversation(
            source_conversation_id=SOURCE_ID, owner_id=uuid4(), request=_request()
        )


# ---------------------------------------------------------------------------
# dependency injection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def container():
    from app.core.container import Container

    return Container()


def test_container_resolves_the_fork_service_with_async_transport(container):
    from app.repositories.conversation_fork import ConversationForkRepository

    service = container.conversation_fork_service()

    assert isinstance(service, ConversationForkService)
    assert isinstance(service.repository, ConversationForkRepository)
    assert service.repository.async_session_factory is not None
    assert service.turn_coordinator is container.conversation_turn_coordinator()
    assert service.compaction_publisher is not None


def test_auto_injector_maps_the_fork_service_to_its_container_provider():
    from app.core.container import Container
    from app.core.dependency_injection import AppAutoInjector

    provider = AppAutoInjector.wiring_map[ConversationForkService]

    assert provider is Container.conversation_fork_service
    assert isinstance(provider(), ConversationForkService)


def test_auto_injected_route_parameter_resolves_the_fork_service():
    import inspect

    import app.core.container  # noqa: F401 - builds the wiring map
    from app.core.dependency_injection import AppAutoInjector

    async def route(fork_service):
        return fork_service

    # Set explicitly: this module's postponed annotations would be strings,
    # while route modules annotate with the class itself.
    route.__annotations__ = {"fork_service": ConversationForkService}
    injected = AppAutoInjector.auto_inject()(route)

    dependency = inspect.signature(injected).parameters["fork_service"].default.dependency

    assert isinstance(dependency(), ConversationForkService)


def test_fingerprint_json_matches_the_documented_canonical_form():
    payload = {
        "source_conversation_id": str(SOURCE_ID),
        "message_id": str(MESSAGE_ID),
        "title": "t",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert (
        fork_request_fingerprint(source_conversation_id=SOURCE_ID, message_id=MESSAGE_ID, title="t")
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
