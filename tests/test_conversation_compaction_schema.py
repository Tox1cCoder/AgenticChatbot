from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.conversation import Conversation
from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.conversation_summary_job import ConversationSummaryJob, SummaryJobStatus
from app.models.message import Message


def _named_constraint(table, constraint_type, name):
    return next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, constraint_type) and constraint.name == name
    )


def test_conversation_owns_positive_next_message_sequence():
    table = Conversation.__table__
    column = table.c.next_message_sequence

    assert isinstance(column.type, BigInteger)
    assert column.nullable is False
    assert str(column.server_default.arg) == "1"
    check = _named_constraint(
        table,
        CheckConstraint,
        "ck_conversations_next_message_sequence_positive",
    )
    assert str(check.sqltext) == "next_message_sequence > 0"


def test_message_sequence_is_unique_per_conversation_and_prompt_indexed():
    table = Message.__table__
    column = table.c.sequence

    assert isinstance(column.type, BigInteger)
    assert column.nullable is False
    unique = _named_constraint(table, UniqueConstraint, "uq_messages_conversation_sequence")
    assert [item.name for item in unique.columns] == ["conversation_id", "sequence"]

    prompt_index = next(
        index
        for index in table.indexes
        if isinstance(index, Index) and index.name == "ix_messages_prompt_history"
    )
    assert [item.name for item in prompt_index.columns] == ["conversation_id", "sequence"]
    assert str(prompt_index.dialect_options["postgresql"]["where"]) == "deleted_at IS NULL"


def test_memory_summary_is_one_to_one_structured_and_sequence_scoped():
    table = ConversationMemorySummary.__table__

    assert set(table.c) == {
        table.c.conversation_id,
        table.c.summary_payload,
        table.c.summary_schema_version,
        table.c.last_summarized_sequence,
        table.c.summary_version,
        table.c.source_message_count,
        table.c.source_token_count,
        table.c.summary_token_count,
        table.c.provider,
        table.c.model,
        table.c.tokenizer,
        table.c.prompt_version,
        table.c.is_valid,
        table.c.created_at,
        table.c.updated_at,
    }
    assert table.c.conversation_id.primary_key is True
    assert table.primary_key.name == "pk_conversation_memory_summaries"
    assert isinstance(table.c.conversation_id.type, UUID)
    assert isinstance(table.c.summary_payload.type, JSONB)
    assert table.c.summary_payload.nullable is False
    assert str(table.c.summary_payload.server_default.arg) == "'{}'::jsonb"

    cursor_fk = _named_constraint(
        table,
        ForeignKeyConstraint,
        "fk_memory_summary_conversation_sequence",
    )
    assert [element.parent.name for element in cursor_fk.elements] == [
        "conversation_id",
        "last_summarized_sequence",
    ]
    assert [element.target_fullname for element in cursor_fk.elements] == [
        "messages.conversation_id",
        "messages.sequence",
    ]


def test_memory_summary_numeric_checks_are_database_enforced():
    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in ConversationMemorySummary.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert checks == {
        "ck_memory_summary_schema_version_positive": "summary_schema_version > 0",
        "ck_memory_summary_version_positive": "summary_version > 0",
        "ck_memory_summary_source_message_count_nonnegative": "source_message_count >= 0",
        "ck_memory_summary_source_token_count_nonnegative": "source_token_count >= 0",
        "ck_memory_summary_token_count_nonnegative": "summary_token_count >= 0",
    }


def test_summary_job_is_one_per_conversation_with_same_conversation_target():
    table = ConversationSummaryJob.__table__

    assert table.c.conversation_id.primary_key is True
    assert table.primary_key.name == "pk_conversation_summary_jobs"
    assert isinstance(table.c.requested_through_sequence.type, BigInteger)
    assert table.c.requested_through_sequence.nullable is False
    assert table.c.status.type.length == 16
    assert table.c.lease_token.nullable is True
    assert isinstance(table.c.lease_token.type, UUID)

    target_fk = _named_constraint(
        table,
        ForeignKeyConstraint,
        "fk_summary_job_conversation_sequence",
    )
    assert [element.parent.name for element in target_fk.elements] == [
        "conversation_id",
        "requested_through_sequence",
    ]
    assert [element.target_fullname for element in target_fk.elements] == [
        "messages.conversation_id",
        "messages.sequence",
    ]

    due_index = next(index for index in table.indexes if index.name == "ix_summary_jobs_due")
    assert [column.name for column in due_index.columns] == ["status", "available_at"]


def test_summary_job_status_and_attempt_count_checks_are_database_enforced():
    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in ConversationSummaryJob.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert set(SummaryJobStatus) == {
        SummaryJobStatus.IDLE,
        SummaryJobStatus.PENDING,
        SummaryJobStatus.PROCESSING,
        SummaryJobStatus.RETRY,
        SummaryJobStatus.DEAD,
    }
    assert checks == {
        "ck_summary_jobs_status": ("status IN ('idle', 'pending', 'processing', 'retry', 'dead')"),
        "ck_summary_jobs_attempt_count_nonnegative": "attempt_count >= 0",
    }


def test_conversation_owned_summary_rows_cascade_on_delete():
    summary_fk = next(
        foreign_key
        for foreign_key in ConversationMemorySummary.__table__.c.conversation_id.foreign_keys
        if foreign_key.target_fullname == "conversations.id"
    )
    job_fk = next(
        foreign_key
        for foreign_key in ConversationSummaryJob.__table__.c.conversation_id.foreign_keys
        if foreign_key.target_fullname == "conversations.id"
    )

    assert summary_fk.ondelete == "CASCADE"
    assert job_fk.ondelete == "CASCADE"
