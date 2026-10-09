"""Fork-conversation request contract, lineage columns, and settings (no database)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.orm import configure_mappers

from app.core.config import Settings
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import (
    ConversationForkRequest,
    ConversationInDB,
    ConversationRead,
)
from app.services.conversation_service import ConversationService


def _request_payload(**overrides):
    payload = {"messageId": str(uuid4()), "requestId": str(uuid4())}
    payload.update(overrides)
    return payload


def test_fork_request_accepts_camel_case_and_omitted_title() -> None:
    request = ConversationForkRequest.model_validate(_request_payload())

    assert request.title is None
    assert request.message_id is not None
    assert request.request_id is not None


def test_fork_request_accepts_snake_case_population() -> None:
    message_id, request_id = uuid4(), uuid4()

    request = ConversationForkRequest(message_id=message_id, request_id=request_id, title="Alt")

    assert (request.message_id, request.request_id, request.title) == (
        message_id,
        request_id,
        "Alt",
    )


@pytest.mark.parametrize("field", ["messageId", "requestId"])
def test_fork_request_rejects_missing_or_malformed_uuid(field: str) -> None:
    payload = _request_payload()
    payload[field] = "not-a-uuid"
    with pytest.raises(ValidationError):
        ConversationForkRequest.model_validate(payload)

    del payload[field]
    with pytest.raises(ValidationError):
        ConversationForkRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("title", "valid"), [("", False), ("a", True), ("a" * 255, True), ("a" * 256, False)]
)
def test_fork_request_title_length_boundaries(title: str, valid: bool) -> None:
    payload = _request_payload(title=title)
    if valid:
        assert ConversationForkRequest.model_validate(payload).title == title
    else:
        with pytest.raises(ValidationError):
            ConversationForkRequest.model_validate(payload)


def _conversation_attrs(**overrides):
    now = datetime.now(UTC)
    attrs = {
        "id": uuid4(),
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "owner_id": uuid4(),
        "title": "Source",
        "persona_prompt": None,
        "planning_mode_enabled": False,
        "plan_lifecycle": None,
        "project_id": None,
    }
    attrs.update(overrides)
    return attrs


@pytest.mark.parametrize("schema", [ConversationRead, ConversationInDB])
def test_existing_conversation_has_absent_lineage(schema) -> None:
    read = schema.model_validate(SimpleNamespace(**_conversation_attrs()))

    assert read.forked_from_conversation_id is None
    assert read.forked_from_message_id is None
    assert read.forked_through_sequence is None


@pytest.mark.parametrize("schema", [ConversationRead, ConversationInDB])
def test_lineage_serializes_camel_case_without_internal_fields(schema) -> None:
    source, message = uuid4(), uuid4()
    read = schema.model_validate(
        SimpleNamespace(
            **_conversation_attrs(
                forked_from_conversation_id=source,
                forked_from_message_id=message,
                forked_through_sequence=4,
                fork_request_id=uuid4(),
                fork_request_fingerprint="f" * 64,
            )
        )
    )

    dumped = read.model_dump(by_alias=True)

    assert dumped["forkedFromConversationId"] == source
    assert dumped["forkedFromMessageId"] == message
    assert dumped["forkedThroughSequence"] == 4
    assert not {"forkRequestId", "forkRequestFingerprint"} & set(dumped)


def test_service_read_conversion_exposes_lineage() -> None:
    source, message = uuid4(), uuid4()
    entity = SimpleNamespace(
        **_conversation_attrs(
            forked_from_conversation_id=source,
            forked_from_message_id=message,
            forked_through_sequence=2,
        )
    )

    service = object.__new__(ConversationService)
    read = service._convert_to_read_schema(entity)

    assert read.forked_from_conversation_id == source
    assert read.forked_from_message_id == message
    assert read.forked_through_sequence == 2


def test_mappers_configure_with_second_conversation_message_join_path() -> None:
    configure_mappers()

    messages = Conversation.messages.property
    parent = Message.conversation.property
    assert [c.name for c in messages.local_remote_pairs[0]] == ["id", "conversation_id"]
    assert [c.name for c in parent.local_remote_pairs[0]] == ["conversation_id", "id"]


def test_lineage_constraints_are_declared() -> None:
    table = Conversation.__table__
    assert {
        "forked_from_conversation_id",
        "forked_from_message_id",
        "forked_through_sequence",
        "fork_request_id",
        "fork_request_fingerprint",
    } <= set(table.c.keys())
    assert all(
        table.c[name].nullable
        for name in (
            "forked_from_conversation_id",
            "forked_from_message_id",
            "forked_through_sequence",
            "fork_request_id",
            "fork_request_fingerprint",
        )
    )

    uniques = {c.name: c for c in table.constraints if isinstance(c, UniqueConstraint)}
    unique = uniques["uq_conversations_owner_fork_request"]
    assert [c.name for c in unique.columns] == ["owner_id", "fork_request_id"]

    checks = {c.name for c in table.constraints if isinstance(c, CheckConstraint)}
    assert {
        "ck_conversations_forked_through_sequence_positive",
        "ck_conversations_fork_request_fields_together",
    } <= checks

    fks = {fk.name: fk for fk in table.foreign_keys if fk.name}
    for name in (
        "fk_conversations_forked_from_conversation_id",
        "fk_conversations_forked_from_message_id",
    ):
        assert fks[name].ondelete == "SET NULL"


def test_fork_limit_settings_defaults() -> None:
    fields = Settings.model_fields

    assert fields["conversation_fork_max_messages"].default == 5000
    assert fields["conversation_fork_max_bytes"].default == 32 * 1024 * 1024
