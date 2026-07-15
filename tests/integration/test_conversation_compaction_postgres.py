from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.conversation_summary_job import ConversationSummaryJob
from app.models.enums import MessageRole
from app.models.feedback import Feedback
from app.models.message import Message
from app.models.user import User
from app.repositories.conversation_compaction import ConversationCompactionRepository
from app.repositories.message import MessageRepository
from app.schemas.message import MessageUpdate


@pytest.fixture(scope="module")
def session_factory():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Conversation.__table__,
            Message.__table__,
            Feedback.__table__,
            ConversationMemorySummary.__table__,
            ConversationSummaryJob.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture()
def seeded_repository(
    session_factory,
) -> Iterator[tuple[ConversationCompactionRepository, UUID, UUID]]:
    owner_id = uuid4()
    conversation_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"owner-{owner_id}",
                email=f"{owner_id}@example.test",
                password_hash="test",
            )
        )
        session.add(
            Conversation(
                id=conversation_id,
                owner_id=owner_id,
                title="compaction test",
            )
        )
    repository = ConversationCompactionRepository(session_factory)
    yield repository, owner_id, conversation_id
    with session_factory.begin() as session:
        session.execute(
            delete(ConversationSummaryJob).where(
                ConversationSummaryJob.conversation_id == conversation_id
            )
        )
        session.execute(
            delete(ConversationMemorySummary).where(
                ConversationMemorySummary.conversation_id == conversation_id
            )
        )
        message_ids = select(Message.id).where(Message.conversation_id == conversation_id)
        session.execute(delete(Feedback).where(Feedback.message_id.in_(message_ids)))
        session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        session.execute(delete(Conversation).where(Conversation.id == conversation_id))
        session.execute(delete(User).where(User.id == owner_id))


def _message(conversation_id: UUID, role: MessageRole, content: str) -> dict:
    return {
        "id": uuid4(),
        "conversation_id": conversation_id,
        "sender": role.value,
        "content": content,
        "message_metadata": {},
    }


def test_atomic_sequences_and_assistant_job_coalescing(seeded_repository, session_factory) -> None:
    repository, _, conversation_id = seeded_repository
    message_repository = MessageRepository(
        session_factory,
        compaction_repository=repository,
    )

    user = message_repository.create(_message(conversation_id, MessageRole.user, "one"))
    first = repository.persist_message(_message(conversation_id, MessageRole.assistant, "two"))
    second = repository.persist_message(_message(conversation_id, MessageRole.assistant, "three"))

    assert [user.sequence, first.sequence, second.sequence] == [1, 2, 3]
    with session_factory() as session:
        conversation = session.get(Conversation, conversation_id)
        job = session.get(ConversationSummaryJob, conversation_id)
        assert conversation.next_message_sequence == 4
        assert job.requested_through_sequence == 3
        assert job.status == "pending"


def test_target_advances_without_stealing_live_lease(seeded_repository, session_factory) -> None:
    repository, _, conversation_id = seeded_repository
    first = repository.persist_message(_message(conversation_id, MessageRole.assistant, "first"))
    claim = repository.claim_job(conversation_id, lease_seconds=120)
    assert claim is not None

    second = repository.persist_message(_message(conversation_id, MessageRole.assistant, "second"))

    with session_factory() as session:
        job = session.get(ConversationSummaryJob, conversation_id)
        assert job.status == "processing"
        assert job.lease_token == claim.lease_token
        assert job.requested_through_sequence == second.sequence
    assert claim.requested_through_sequence == first.sequence
    assert repository.complete_claim(claim) == "pending"


def test_claim_token_retry_dead_and_expired_lease_reconciliation(
    seeded_repository, session_factory
) -> None:
    repository, _, conversation_id = seeded_repository
    repository.persist_message(_message(conversation_id, MessageRole.assistant, "first"))
    claim = repository.claim_job(conversation_id, lease_seconds=120)
    assert claim is not None
    wrong = claim._replace(lease_token=uuid4())
    assert repository.complete_claim(wrong) is None
    assert not repository.fail_claim(wrong, error_code="timeout", permanent=False)

    retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert repository.fail_claim(
        claim,
        error_code="provider_timeout",
        permanent=False,
        retry_at=retry_at,
    )
    retry_claim = repository.claim_job(conversation_id, lease_seconds=1)
    assert retry_claim is not None
    assert retry_claim.attempt_count == 1
    with session_factory.begin() as session:
        job = session.get(ConversationSummaryJob, conversation_id)
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    due = repository.reconcile_due_jobs(dispatch_debounce_seconds=30)
    assert conversation_id in due
    with session_factory() as session:
        job = session.get(ConversationSummaryJob, conversation_id)
        assert job.status == "retry"
        assert job.lease_token is None
        assert job.last_error_code == "lease_expired"

    dead_claim = repository.claim_job(conversation_id, lease_seconds=120)
    assert dead_claim is not None
    assert repository.fail_claim(dead_claim, error_code="invalid_output", permanent=True)
    with session_factory() as session:
        assert session.get(ConversationSummaryJob, conversation_id).status == "dead"


def test_owned_memory_input_cas_and_mutation_invalidation(
    seeded_repository, session_factory
) -> None:
    repository, owner_id, conversation_id = seeded_repository
    message_repository = MessageRepository(
        session_factory,
        compaction_repository=repository,
    )
    covered = repository.persist_message(_message(conversation_id, MessageRole.user, "old fact"))
    target = repository.persist_message(_message(conversation_id, MessageRole.assistant, "answer"))
    claim = repository.claim_job(conversation_id, lease_seconds=120)
    assert claim is not None
    compaction_input = repository.load_compaction_input(claim, owner_id=owner_id)
    assert compaction_input is not None
    assert [message.sequence for message in compaction_input.messages] == [1, 2]
    assert repository.load_compaction_input(claim, owner_id=uuid4()) is None

    assert repository.persist_memory_cas(
        claim,
        base_summary_version=0,
        base_cursor=None,
        summary_payload={"facts": ["old fact"]},
        summary_schema_version=1,
        last_summarized_sequence=target.sequence,
        source_message_count=2,
        source_token_count=10,
        summary_token_count=3,
        provider="gemini",
        model="gemini-2.5-flash",
        tokenizer="local_bytes_v1",
        prompt_version="v1",
    )
    assert not repository.persist_memory_cas(
        claim,
        base_summary_version=0,
        base_cursor=None,
        summary_payload={"facts": ["stale"]},
        summary_schema_version=1,
        last_summarized_sequence=target.sequence,
        source_message_count=2,
        source_token_count=10,
        summary_token_count=3,
        provider="gemini",
        model="gemini-2.5-flash",
        tokenizer="local_bytes_v1",
        prompt_version="v1",
    )
    memory = repository.get_owned_valid_memory(conversation_id, owner_id)
    assert memory is not None
    assert repository.get_owned_valid_memory(conversation_id, uuid4()) is None

    updated = message_repository.update(
        covered.id,
        MessageUpdate(content="corrected fact"),
    )
    assert updated is not None
    assert updated.content == "corrected fact"
    with session_factory() as session:
        memory = session.get(ConversationMemorySummary, conversation_id)
        job = session.get(ConversationSummaryJob, conversation_id)
        assert memory.summary_payload == {}
        assert memory.last_summarized_sequence is None
        assert memory.is_valid is False
        assert memory.summary_version == 2
        assert job.requested_through_sequence == target.sequence
        assert job.status == "pending"

    assert message_repository.delete(covered.id)
    with session_factory() as session:
        message = session.get(Message, covered.id)
        assert message.deleted_at is not None


def test_completion_moves_caught_up_job_to_idle(seeded_repository) -> None:
    repository, _, conversation_id = seeded_repository
    repository.persist_message(_message(conversation_id, MessageRole.assistant, "answer"))
    claim = repository.claim_job(conversation_id, lease_seconds=120)

    assert claim is not None
    assert repository.complete_claim(claim) == "idle"


def test_expired_lease_cannot_load_persist_or_complete(seeded_repository, session_factory) -> None:
    repository, owner_id, conversation_id = seeded_repository
    target = repository.persist_message(_message(conversation_id, MessageRole.assistant, "answer"))
    claim = repository.claim_job(conversation_id, lease_seconds=120)
    assert claim is not None
    with session_factory.begin() as session:
        job = session.get(ConversationSummaryJob, conversation_id)
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    assert repository.load_compaction_input(claim, owner_id=owner_id) is None
    assert not repository.persist_memory_cas(
        claim,
        base_summary_version=0,
        base_cursor=None,
        summary_payload={"facts": ["stale"]},
        summary_schema_version=1,
        last_summarized_sequence=target.sequence,
        source_message_count=1,
        source_token_count=2,
        summary_token_count=1,
        provider="gemini",
        model="gemini-2.5-flash",
        tokenizer="local_bytes_v1",
        prompt_version="v1",
    )
    assert repository.complete_claim(claim) is None


def test_concurrent_persistence_never_duplicates_sequence_or_regresses_target(
    seeded_repository, session_factory
) -> None:
    repository, _, conversation_id = seeded_repository

    def persist(index: int) -> Message:
        return repository.persist_message(
            _message(conversation_id, MessageRole.assistant, f"answer-{index}")
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        messages = list(executor.map(persist, range(12)))

    assert sorted(message.sequence for message in messages) == list(range(1, 13))
    with session_factory() as session:
        job = session.get(ConversationSummaryJob, conversation_id)
        conversation = session.get(Conversation, conversation_id)
        assert job.requested_through_sequence == 12
        assert conversation.next_message_sequence == 13


def test_generic_claim_skips_a_locked_job(seeded_repository, session_factory) -> None:
    repository, owner_id, locked_conversation_id = seeded_repository
    other_conversation_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            Conversation(
                id=other_conversation_id,
                owner_id=owner_id,
                title="other compaction test",
            )
        )
    repository.persist_message(_message(locked_conversation_id, MessageRole.assistant, "locked"))
    repository.persist_message(_message(other_conversation_id, MessageRole.assistant, "claimable"))
    try:
        with session_factory() as locking_session:
            locking_session.execute(
                select(ConversationSummaryJob)
                .where(ConversationSummaryJob.conversation_id == locked_conversation_id)
                .with_for_update()
            ).scalar_one()
            claim = repository.claim_job(None, lease_seconds=120)
            assert claim is not None
            assert claim.conversation_id == other_conversation_id
            locking_session.rollback()
    finally:
        with session_factory.begin() as session:
            session.execute(
                delete(ConversationSummaryJob).where(
                    ConversationSummaryJob.conversation_id == other_conversation_id
                )
            )
            session.execute(delete(Message).where(Message.conversation_id == other_conversation_id))
            session.execute(delete(Conversation).where(Conversation.id == other_conversation_id))


def test_database_rejects_cross_conversation_memory_cursor(
    seeded_repository, session_factory
) -> None:
    repository, owner_id, conversation_id = seeded_repository
    other_conversation_id = uuid4()
    with session_factory.begin() as session:
        session.add(
            Conversation(
                id=other_conversation_id,
                owner_id=owner_id,
                title="cursor source",
            )
        )
    other_message = repository.persist_message(
        _message(other_conversation_id, MessageRole.assistant, "other")
    )
    try:
        with session_factory() as session:
            session.add(
                ConversationMemorySummary(
                    conversation_id=conversation_id,
                    summary_payload={},
                    last_summarized_sequence=other_message.sequence,
                    summary_version=1,
                    is_valid=False,
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
    finally:
        with session_factory.begin() as session:
            session.execute(
                delete(ConversationSummaryJob).where(
                    ConversationSummaryJob.conversation_id == other_conversation_id
                )
            )
            session.execute(delete(Message).where(Message.conversation_id == other_conversation_id))
            session.execute(delete(Conversation).where(Conversation.id == other_conversation_id))


def test_historical_backfill_listing_and_request_are_idempotent(
    seeded_repository, session_factory
) -> None:
    repository, _, conversation_id = seeded_repository
    message = repository.persist_message(
        _message(conversation_id, MessageRole.assistant, "historical")
    )
    with session_factory.begin() as session:
        session.execute(
            delete(ConversationSummaryJob).where(
                ConversationSummaryJob.conversation_id == conversation_id
            )
        )

    assert (conversation_id, message.sequence) in repository.list_backfill_candidates(limit=100)
    assert repository.request_backfill(conversation_id, message.sequence)
    assert not repository.request_backfill(conversation_id, message.sequence)
    assert (conversation_id, message.sequence) not in repository.list_backfill_candidates(limit=100)
