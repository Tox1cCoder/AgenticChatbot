from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import make_transient_to_detached

from app.models.enums import MessageRole
from app.repositories.conversation_compaction import (
    ConversationCompactionRepository,
    SummaryJobClaim,
)
from app.schemas.message import MessageRead


def _sql(statement) -> str:
    return " ".join(str(statement.compile(dialect=postgresql.dialect())).lower().split())


def _params(statement) -> list[object]:
    compiled = statement.compile(dialect=postgresql.dialect())
    return list(compiled.params.values())


class _AllocatedSequenceResult:
    def scalar_one_or_none(self) -> int:
        return 1


class _DetachingPersistenceSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, _statement):
        return _AllocatedSequenceResult()

    def add(self, message) -> None:
        self.message = message

    def flush(self) -> None:
        now = datetime.now(timezone.utc)
        self.message.created_at = now
        self.message.updated_at = now
        self.message.deleted_at = None

    def commit(self) -> None:
        pass

    def expunge(self, message) -> None:
        make_transient_to_detached(message)


def test_persisted_message_serializes_feedback_after_session_detaches_it() -> None:
    session = _DetachingPersistenceSession()
    repository = ConversationCompactionRepository(lambda: session)

    message = repository.persist_message(
        {
            "id": uuid4(),
            "conversation_id": uuid4(),
            "sender": MessageRole.user.value,
            "content": "hello",
            "message_metadata": {},
        }
    )

    assert inspect(message).detached
    assert MessageRead.model_validate(message).feedback is None


def test_job_upsert_coalesces_targets_and_preserves_a_live_lease() -> None:
    statement = ConversationCompactionRepository._job_upsert_statement(
        conversation_id=uuid4(),
        requested_through_sequence=7,
        now=datetime.now(timezone.utc),
    )

    sql = _sql(statement)

    assert "on conflict (conversation_id) do update" in sql
    assert "greatest(" in sql
    assert "status = case" in sql
    assert "lease_expires_at" in sql
    assert "processing" in _params(statement)


def test_generic_claim_uses_skip_locked_and_due_filter() -> None:
    statement = ConversationCompactionRepository._claim_select_statement(
        conversation_id=None,
        now=datetime.now(timezone.utc),
    )

    sql = _sql(statement)

    assert "available_at <=" in sql
    assert "for update skip locked" in sql


def test_notification_claim_ignores_debounce_but_still_locks() -> None:
    conversation_id = uuid4()
    statement = ConversationCompactionRepository._claim_select_statement(
        conversation_id=conversation_id,
        now=datetime.now(timezone.utc),
    )

    sql = _sql(statement)

    assert "conversation_id =" in sql
    assert "available_at <=" not in sql
    assert "for update skip locked" in sql


def test_memory_cas_checks_version_cursor_and_lease() -> None:
    claim = SummaryJobClaim(
        conversation_id=uuid4(),
        requested_through_sequence=12,
        lease_token=uuid4(),
    )
    statement = ConversationCompactionRepository._memory_cas_update_statement(
        claim=claim,
        base_summary_version=3,
        base_cursor=8,
        summary_payload={"facts": ["stable"]},
        summary_schema_version=1,
        source_message_count=4,
        source_token_count=50,
        summary_token_count=4,
        provider="gemini",
        model="gemini-2.5-flash",
        tokenizer="local_bytes_v1",
        prompt_version="v1",
    )

    sql = _sql(statement)

    assert "summary_version =" in sql
    assert "last_summarized_sequence is not distinct from" in sql
    assert "conversation_summary_jobs.lease_token" in sql
    assert "conversation_summary_jobs.status" in sql
    assert "processing" in _params(statement)
