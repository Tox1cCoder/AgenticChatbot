"""One control surface, reachable through every transport.

The properties that matter here are boundary properties, so they are asserted
over real HTTP against the real routers rather than by calling the service:

* **Owner scoping is a 404, not a 403.** "It exists but is not yours" is
  information about another user's conversation, so a wrong owner and a wrong
  id must be indistinguishable.
* **A pending Stop is a 202.** A client has to be able to tell "settled" from
  "accepted" without parsing the body, because the worker may be
  mid-provider-call in another process and nothing has confirmed yet.
* **Continue is not a new turn.** No user message, no route. The endpoint that
  quietly created one would re-route the continuation and could land it on a
  different specialist than the one holding the evidence.
* **The pause's internals stop at the boundary.** The validated text already
  reached the client as deltas and lives in the message row; the checkpoint
  thread is a resume handle. Neither belongs in a payload a client reads.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.messages import router as messages_router
from app.core.auth import get_current_user_id
from app.core.container import Container, setup_auto_injection
from app.models.generation import GenerationStatus
from app.schemas.generation import GenerationSnapshot
from app.services.event_streaming.events import make_event
from app.utils.exception_handler import register_exception_handlers

CONVERSATION_ID = UUID("11111111-1111-1111-1111-111111111111")
GENERATION_ID = UUID("33333333-3333-3333-3333-333333333333")
CONTINUATION_ID = UUID("44444444-4444-4444-4444-444444444444")
ASSISTANT_MESSAGE_ID = UUID("55555555-5555-5555-5555-555555555555")


@pytest.fixture(autouse=True)
def _restore_wiring():
    setup_auto_injection(Container)
    yield
    setup_auto_injection(Container)


def _snapshot(
    *,
    status: GenerationStatus = GenerationStatus.RUNNING,
    version: int = 2,
    epoch: int = 0,
    continuation_available: bool = False,
    continuation_id: UUID | None = None,
    block_reason: str | None = None,
) -> GenerationSnapshot:
    return GenerationSnapshot(
        generation_id=GENERATION_ID,
        logical_turn_id="turn-1",
        conversation_id=CONVERSATION_ID,
        status=status,
        version=version,
        execution_epoch=epoch,
        continuation_id=continuation_id,
        continuation_available=continuation_available,
        continuation_block_reason=block_reason,
        assistant_message_id=ASSISTANT_MESSAGE_ID,
        terminal_reason=None,
    )


class _Service:
    """The control surface, recording what each route asked of it."""

    def __init__(self, *, snapshot: GenerationSnapshot | None = None, owner: UUID) -> None:
        self._snapshot = snapshot
        self._owner = owner
        self.stop_calls: list[dict[str, Any]] = []
        self.continue_calls: list[dict[str, Any]] = []
        self.turn_scoped_calls: list[dict[str, Any]] = []
        self.continue_events: list[Any] = [
            make_event("complete", sequence=1, data={"message": {"id": "m1"}})
        ]

    async def aget_generation(self, *, generation_id, conversation_id, user_id):
        if user_id != self._owner or conversation_id != CONVERSATION_ID:
            return None
        if generation_id != GENERATION_ID:
            return None
        return self._snapshot

    async def stop_generation(
        self, *, generation_id, conversation_id, user_id, idempotency_key, expected_version
    ):
        self.stop_calls.append(
            {
                "generation_id": generation_id,
                "idempotency_key": idempotency_key,
                "expected_version": expected_version,
            }
        )
        return self._snapshot

    async def stop_message_generation(self, *, conversation_id, user_id, user_message_id):
        self.turn_scoped_calls.append({"user_message_id": user_message_id})
        return {"status": "not_inflight", "message": None}

    def _legacy_stop_result(self, snapshot, *, user_id):
        from app.services.message_service import MessageService

        return MessageService._legacy_stop_result(self, snapshot, user_id=user_id)

    def _generation_status_data(self, snapshot):
        from app.services.message_service import MessageService

        return MessageService._generation_status_data(snapshot)

    def get_by_id(self, message_id, user_id):
        raise RuntimeError("no message store in this harness")

    async def continue_message_generation_stream(self, **kwargs):
        self.continue_calls.append(kwargs)
        for event in self.continue_events:
            yield event


def _client(service: _Service, user_id: UUID) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(messages_router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    return TestClient(app)


def _parse_sse(text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in text.splitlines():
        if line.startswith("data: "):
            body = line[len("data: ") :].strip()
            if body and body != "[DONE]":
                payloads.append(json.loads(body))
    return payloads


# ----------------------------------------------------------------------
# GET /messages/generations/{id}
# ----------------------------------------------------------------------


def test_the_snapshot_endpoint_returns_the_authoritative_state():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["generationId"] == str(GENERATION_ID)
    assert data["status"] == "running"
    assert data["version"] == 2


def test_the_snapshot_carries_the_version_a_fenced_command_needs():
    """R5. A client with no version cannot issue a fenced Stop at all."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(version=7), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )

    assert response.json()["data"]["version"] == 7


def test_the_snapshot_publishes_no_checkpoint_thread_or_budget():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )

    body = response.text
    for leaked in ("checkpointThread", "checkpoint_thread", "researchAccounting", "userId"):
        assert leaked not in body, f"{leaked} crossed the boundary"


def test_another_users_generation_is_a_404_not_a_403():
    """Existence is itself information about another user's conversation."""
    service = _Service(snapshot=_snapshot(), owner=uuid4())

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, uuid4()).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )

    assert response.status_code == 404


def test_an_unknown_generation_answers_exactly_as_a_foreign_one():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        client = _client(service, owner)
        unknown = client.get(
            f"/messages/generations/{uuid4()}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )
        foreign = _client(service, uuid4()).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(CONVERSATION_ID)},
        )

    # The whole body, not one field: any difference at all — a code, a
    # message, a hint — tells the caller whether the generation exists.
    assert unknown.status_code == foreign.status_code == 404
    assert unknown.json() == foreign.json()


def test_a_generation_read_through_the_wrong_conversation_is_refused():
    """The id is public; the conversation predicate is what makes it safe."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).get(
            f"/messages/generations/{GENERATION_ID}",
            params={"conversation_id": str(uuid4())},
        )

    assert response.status_code == 404


# ----------------------------------------------------------------------
# POST /messages/stop
# ----------------------------------------------------------------------


def test_a_pending_stop_is_202_so_a_client_can_tell_it_is_unsettled():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(status=GenerationStatus.STOP_REQUESTED), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "idempotencyKey": "stop-key-0001",
                "expectedVersion": 2,
            },
        )

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "stop_requested"


def test_a_settled_stop_is_200():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(status=GenerationStatus.STOPPED), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "idempotencyKey": "stop-key-0001",
                "expectedVersion": 2,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "cancelled"


def test_the_stop_response_carries_the_durable_snapshot():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(status=GenerationStatus.STOPPED), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "idempotencyKey": "stop-key-0001",
                "expectedVersion": 2,
            },
        )

    generation = response.json()["data"]["generation"]
    assert generation["status"] == "stopped"
    assert generation["version"] == 2


def test_the_clients_fence_and_key_reach_the_command_unchanged():
    """A fence the endpoint substituted would defeat the point of having one."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "idempotencyKey": "stop-key-abcdef",
                "expectedVersion": 2,
            },
        )

    assert service.stop_calls == [
        {
            "generation_id": GENERATION_ID,
            "idempotency_key": "stop-key-abcdef",
            "expected_version": 2,
        }
    ]


def test_a_stop_without_a_version_reads_one_rather_than_inventing_it():
    """A client predating run_start has no fence, so the row supplies it."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(version=5), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
            },
        )

    assert service.stop_calls[0]["expected_version"] == 5


def test_a_stop_naming_only_the_user_message_uses_the_turn_scoped_path():
    owner = uuid4()
    user_message_id = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/stop",
            json={
                "conversationId": str(CONVERSATION_ID),
                "userMessageId": str(user_message_id),
            },
        )

    assert response.status_code == 200
    assert service.turn_scoped_calls == [{"user_message_id": user_message_id}]
    assert service.stop_calls == []


def test_a_stop_naming_no_turn_at_all_is_rejected():
    """Without an identity this would have to guess which turn to stop."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/stop", json={"conversationId": str(CONVERSATION_ID)}
        )

    assert response.status_code == 422


# ----------------------------------------------------------------------
# POST /messages/continue
# ----------------------------------------------------------------------


def test_continue_streams_internal_sse_and_forwards_the_lease_identity():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    call = service.continue_calls[0]
    assert call["generation_id"] == GENERATION_ID
    assert call["continuation_id"] == CONTINUATION_ID
    assert call["expected_version"] == 3
    assert call["idempotency_key"] == "continue-key-0001"


def test_continue_creates_no_user_message_event():
    """Continue is more of the same answer, not another question."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)
    service.continue_events = [
        make_event("run_start", sequence=1, data={"generation_id": str(GENERATION_ID)}),
        make_event("message_delta", sequence=2, data={"text": "more"}),
        make_event("complete", sequence=3, data={"message": {"id": "m1"}}),
    ]

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    types = [payload.get("type") for payload in _parse_sse(response.text)]
    assert "user_message_created" not in types
    assert "agent_selected" not in types
    assert "token" in types


def test_continue_requires_a_continuation_id():
    """A Continue without one has nothing single-use to consume."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    assert response.status_code == 422


def test_continue_requires_a_fence():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
            },
        )

    assert response.status_code == 422


def test_a_short_idempotency_key_is_refused():
    """A guessable key lets one client replay another's command."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "short",
                "expectedVersion": 3,
            },
        )

    assert response.status_code == 422


# ----------------------------------------------------------------------
# what the internal SSE adapter publishes
# ----------------------------------------------------------------------


def test_the_pause_reaches_the_client_as_continuation_available():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)
    service.continue_events = [
        make_event(
            "continuation_available",
            sequence=1,
            data={
                "generation_id": str(GENERATION_ID),
                "continuation_id": str(CONTINUATION_ID),
                "status": "continuable",
                "version": 4,
                "execution_epoch": 1,
                "continuation_available": True,
                "assistant_message_id": str(ASSISTANT_MESSAGE_ID),
            },
        )
    ]

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    payloads = _parse_sse(response.text)
    offer = next(p for p in payloads if p.get("type") == "continuation_available")
    assert offer["continuation_id"] == str(CONTINUATION_ID)
    assert offer["version"] == 4
    assert offer["continuation_available"] is True


def test_run_start_reaches_the_client_as_generation_start():
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)
    service.continue_events = [
        make_event(
            "run_start",
            sequence=1,
            data={
                "generation_id": str(GENERATION_ID),
                "status": "continuing",
                "version": 3,
                "execution_epoch": 1,
            },
        )
    ]

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    payloads = _parse_sse(response.text)
    start = next(p for p in payloads if p.get("type") == "generation_start")
    assert start["generation_id"] == str(GENERATION_ID)
    assert start["version"] == 3


def test_the_validated_text_and_resume_handle_do_not_cross_the_boundary():
    """Both are internal: the text arrived as deltas, the thread is a handle."""
    owner = uuid4()
    service = _Service(snapshot=_snapshot(), owner=owner)
    service.continue_events = [
        make_event(
            "continuation_available",
            sequence=1,
            data={
                "type": "execution_budget_exhausted",
                "generation_id": str(GENERATION_ID),
                "validated_content": "SECRET-PARTIAL-TEXT",
                "budget": {"exhausted_by": "tool_calls"},
                "thread_id": "wf2:conv:turn-1",
                "active_agent_id": "search_agent",
                "status": "continuable",
                "version": 4,
            },
        )
    ]

    with Container.message_service.override(providers.Object(service)):
        response = _client(service, owner).post(
            "/messages/continue",
            json={
                "conversationId": str(CONVERSATION_ID),
                "generationId": str(GENERATION_ID),
                "continuationId": str(CONTINUATION_ID),
                "idempotencyKey": "continue-key-0001",
                "expectedVersion": 3,
            },
        )

    assert "SECRET-PARTIAL-TEXT" not in response.text
    assert "wf2:conv:turn-1" not in response.text
    assert "exhausted_by" not in response.text


# ----------------------------------------------------------------------
# what the AI SDK adapter publishes
# ----------------------------------------------------------------------


def _ai_sdk_parts(events: list[Any]) -> list[dict[str, Any]]:
    """Run the real AI SDK adapter over a scripted canonical stream."""
    import asyncio

    from app.services.event_streaming.ai_sdk_v6 import (
        AISDKV6StreamAdapter,
        AISDKV6StreamState,
    )

    async def source():
        for event in events:
            yield event

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m1", text_id="t1", reasoning_id="r1"),
        heartbeat_interval_seconds=30.0,
    )

    async def collect():
        return [chunk async for chunk in adapter.iter_sse()]

    return [
        json.loads(chunk[len("data: ") :].strip())
        for chunk in asyncio.run(collect())
        if chunk.startswith("data: ") and chunk[len("data: ") :].strip() != "[DONE]"
    ]


def test_the_ai_sdk_publishes_the_same_three_lifecycle_events():
    """Parity: both adapters project the same canonical events."""
    parts = _ai_sdk_parts(
        [
            make_event(
                "run_start",
                sequence=1,
                data={"generation_id": str(GENERATION_ID), "version": 2, "status": "running"},
            ),
            make_event(
                "generation_status",
                sequence=2,
                data={"generation_id": str(GENERATION_ID), "version": 3, "status": "stopped"},
            ),
            make_event(
                "continuation_available",
                sequence=3,
                data={
                    "generation_id": str(GENERATION_ID),
                    "continuation_id": str(CONTINUATION_ID),
                    "version": 4,
                    "status": "continuable",
                },
            ),
        ]
    )

    generation_parts = [part for part in parts if part.get("type") == "data-generation"]
    assert [part["data"]["phase"] for part in generation_parts] == [
        "start",
        "status",
        "continuation_available",
    ]


def test_the_ai_sdk_generation_part_is_not_transient():
    """A reconnecting client still needs the identity and the fence.

    Marking it transient would drop exactly the fields that make a Stop or a
    Continue addressable.
    """
    parts = _ai_sdk_parts(
        [
            make_event(
                "run_start",
                sequence=1,
                data={"generation_id": str(GENERATION_ID), "version": 2, "status": "running"},
            )
        ]
    )

    part = next(p for p in parts if p.get("type") == "data-generation")
    assert part.get("transient") is not True
    assert part["data"]["version"] == 2


def test_the_ai_sdk_strips_the_same_private_pause_fields():
    parts = _ai_sdk_parts(
        [
            make_event(
                "continuation_available",
                sequence=1,
                data={
                    "type": "execution_budget_exhausted",
                    "generation_id": str(GENERATION_ID),
                    "validated_content": "SECRET-PARTIAL-TEXT",
                    "thread_id": "wf2:conv:turn-1",
                    "budget": {"exhausted_by": "tool_calls"},
                    "status": "continuable",
                    "version": 4,
                },
            )
        ]
    )

    serialized = json.dumps(parts)
    assert "SECRET-PARTIAL-TEXT" not in serialized
    assert "wf2:conv:turn-1" not in serialized
    assert "exhausted_by" not in serialized


def test_neither_adapter_derives_a_status_from_the_stream_simply_ending():
    """A closed socket is transport recovery, never a lifecycle transition."""
    parts = _ai_sdk_parts(
        [make_event("message_delta", sequence=1, data={"text": "partial answer"})]
    )

    assert all(part.get("type") != "data-generation" for part in parts)

    legacy = [
        payload
        for payload in (
            _legacy(make_event("message_delta", sequence=1, data={"text": "partial answer"})),
        )
        if payload
    ]
    assert all(payload.get("type") != "generation_status" for payload in legacy)


def _legacy(event: Any) -> dict[str, Any] | None:
    from app.services.event_streaming.internal_sse import legacy_event_from_v3

    return legacy_event_from_v3(event)


def test_the_two_adapters_agree_on_the_lifecycle_fields_they_publish():
    """Parity asserted as a set comparison, not by reading both by eye."""
    event = make_event(
        "continuation_available",
        sequence=1,
        data={
            "type": "execution_budget_exhausted",
            "generation_id": str(GENERATION_ID),
            "continuation_id": str(CONTINUATION_ID),
            "status": "continuable",
            "version": 4,
            "execution_epoch": 1,
            "continuation_available": True,
            "continuation_block_reason": None,
            "assistant_message_id": str(ASSISTANT_MESSAGE_ID),
            "terminal_reason": None,
            "logical_turn_id": "turn-1",
            "conversation_id": str(CONVERSATION_ID),
            "validated_content": "SECRET",
            "budget": {"exhausted_by": "tool_calls"},
            "thread_id": "wf2:conv:turn-1",
            "active_agent_id": "search_agent",
        },
    )

    internal = _legacy(event)
    ai_sdk = next(
        part for part in _ai_sdk_parts([event]) if part.get("type") == "data-generation"
    )

    internal_fields = set(internal) - {"type"}
    ai_sdk_fields = set(ai_sdk["data"]) - {"phase"}
    assert internal_fields == ai_sdk_fields
