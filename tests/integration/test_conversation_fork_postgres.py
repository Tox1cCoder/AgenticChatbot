"""The fork snapshot against real PostgreSQL: atomicity, isolation and races.

Every test drives :class:`ConversationForkService` over a real async session
transport. Concurrency tests interleave a second, real session at a precise
point of the fork transaction by wrapping one of the repository's private
steps; each such test names the mechanism it proves and fails when that
mechanism is removed (see the task report's mutation checks).

Skipped without ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, delete, event, func, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.exceptions.http import CustomHTTPException
from app.models.base import Base
from app.models.chat_image import ChatImage
from app.models.conversation import Conversation
from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.conversation_summary_job import ConversationSummaryJob
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.document import Document
from app.models.enums import MessageRole
from app.models.feedback import Feedback
from app.models.generation import Generation, GenerationCommand, GenerationStatus
from app.models.message import Message
from app.models.project import Project
from app.models.user import User
from app.models.web_image_reference import WebImageReference
from app.repositories.conversation_compaction import ConversationCompactionRepository
from app.repositories.conversation_fork import ConversationForkRepository
from app.schemas.conversation import ConversationForkRequest
from app.services.conversation_fork_service import ConversationForkService
from app.services.conversation_turn_coordinator import (
    ConversationTurnCoordinator,
    InProcessTurnLockBackend,
    PostgresAdvisoryLockBackend,
)

pytestmark = pytest.mark.selector_event_loop

USER = MessageRole.user.value
ASSISTANT = MessageRole.assistant.value

_TABLES = [
    User.__table__,
    Project.__table__,
    Conversation.__table__,
    Message.__table__,
    Feedback.__table__,
    Generation.__table__,
    GenerationCommand.__table__,
    ConversationMemorySummary.__table__,
    ConversationSummaryJob.__table__,
    CustomAgent.__table__,
    ConversationCustomAgent.__table__,
    Document.__table__,
    ChatImage.__table__,
    WebImageReference.__table__,
]


def _async_url(database_url: str) -> str:
    for prefix in ("postgresql+psycopg2://", "postgresql://"):
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql+psycopg://", 1)
    return database_url


@dataclass
class Pg:
    sync: sessionmaker
    async_factory: Any
    async_engine: Any
    owners: list[UUID] = field(default_factory=list)


@pytest.fixture(scope="module")
def pg_engines():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(engine, tables=_TABLES)
    # NullPool: every test runs on its own event loop, and a pooled async
    # connection is bound to the loop that opened it.
    async_engine = create_async_engine(_async_url(database_url), poolclass=NullPool)
    try:
        yield sessionmaker(bind=engine, expire_on_commit=False), async_engine
    finally:
        engine.dispose()


@pytest.fixture()
def pg(pg_engines) -> Iterator[Pg]:
    sync, async_engine = pg_engines
    state = Pg(
        sync=sync,
        async_factory=async_sessionmaker(bind=async_engine, expire_on_commit=False),
        async_engine=async_engine,
    )
    yield state
    _purge(sync, state.owners)


def _purge(sync: sessionmaker, owners: list[UUID]) -> None:
    if not owners:
        return
    with sync.begin() as session:
        conversations = select(Conversation.id).where(Conversation.owner_id.in_(owners))
        for model in (ConversationSummaryJob, ConversationMemorySummary):
            session.execute(delete(model).where(model.conversation_id.in_(conversations)))
        generations = select(Generation.id).where(Generation.conversation_id.in_(conversations))
        session.execute(
            delete(GenerationCommand).where(GenerationCommand.generation_id.in_(generations))
        )
        for model in (Generation, ConversationCustomAgent, Document, ChatImage, WebImageReference):
            session.execute(delete(model).where(model.conversation_id.in_(conversations)))
        session.execute(delete(CustomAgent).where(CustomAgent.owner_id.in_(owners)))
        session.execute(delete(Message).where(Message.conversation_id.in_(conversations)))
        session.execute(delete(Conversation).where(Conversation.owner_id.in_(owners)))
        session.execute(delete(Project).where(Project.owner_id.in_(owners)))
        session.execute(delete(User).where(User.id.in_(owners)))


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


@dataclass
class Source:
    owner_id: UUID
    conversation_id: UUID
    ids: dict[int, UUID]

    def id_at(self, sequence: int) -> UUID:
        return self.ids[sequence]


def _user(pg: Pg) -> UUID:
    owner_id = uuid4()
    pg.owners.append(owner_id)
    with pg.sync.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"fork-{owner_id.hex[:20]}",
                email=f"{owner_id}@example.test",
                password_hash="test",
            )
        )
    return owner_id


def _conversation(pg: Pg, owner_id: UUID, *, title: str = "Source", **values: Any) -> UUID:
    conversation_id = uuid4()
    with pg.sync.begin() as session:
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title=title, **values))
    return conversation_id


def _message(
    pg: Pg,
    conversation_id: UUID,
    sequence: int,
    sender: int,
    content: str,
    metadata: dict | None = None,
    **values: Any,
) -> UUID:
    message_id = uuid4()
    created = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=sequence)
    with pg.sync.begin() as session:
        session.add(
            Message(
                id=message_id,
                conversation_id=conversation_id,
                sender=sender,
                content=content,
                message_metadata=metadata or {},
                sequence=sequence,
                created_at=created,
                updated_at=created,
                **values,
            )
        )
        session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                next_message_sequence=func.greatest(
                    Conversation.next_message_sequence, sequence + 1
                )
            )
        )
    return message_id


def _generation(
    pg: Pg,
    source: Source,
    *,
    turn: UUID,
    status: GenerationStatus,
    assistant_message_id: UUID | None,
    continuation_available: bool = False,
) -> UUID:
    generation_id = uuid4()
    with pg.sync.begin() as session:
        session.add(
            Generation(
                id=generation_id,
                logical_turn_id=str(turn),
                checkpoint_thread_id=f"{source.conversation_id}:{turn}",
                user_id=source.owner_id,
                conversation_id=source.conversation_id,
                status=status,
                assistant_message_id=assistant_message_id,
                continuation_available=continuation_available,
            )
        )
    return generation_id


def _completed_turn(pg: Pg, source: Source, sequence: int, question: str, answer: str) -> None:
    user_id = _message(pg, source.conversation_id, sequence, USER, question)
    reply_id = _message(
        pg,
        source.conversation_id,
        sequence + 1,
        ASSISTANT,
        answer,
        {"reply_to_user_message_id": str(user_id), "model": "m-1"},
    )
    source.ids[sequence] = user_id
    source.ids[sequence + 1] = reply_id
    _generation(
        pg, source, turn=user_id, status=GenerationStatus.COMPLETED, assistant_message_id=reply_id
    )


def _six_message_source(pg: Pg, owner_id: UUID | None = None, **values: Any) -> Source:
    """Turns at sequences 1-2, 3-4 and 5-6, each with a completed generation."""
    owner_id = owner_id or _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id, **values), {})
    for sequence in (1, 3, 5):
        _completed_turn(pg, source, sequence, f"question {sequence}", f"answer {sequence + 1}")
    return source


# ---------------------------------------------------------------------------
# the service under test
# ---------------------------------------------------------------------------


def _service(
    pg: Pg,
    *,
    coordinator: Any = None,
    publisher: Any = None,
    lock_timeout_seconds: float = 5.0,
    max_messages: int = 5000,
    max_bytes: int = 32 * 1024 * 1024,
    session_factory: Any = None,
) -> ConversationForkService:
    repository = ConversationForkRepository(
        pg.sync, async_session_factory=session_factory or pg.async_factory
    )
    return ConversationForkService(
        repository=repository,
        turn_coordinator=coordinator
        or ConversationTurnCoordinator(
            backend=InProcessTurnLockBackend(), timeout_seconds=lock_timeout_seconds
        ),
        compaction_publisher=publisher,
        lock_timeout_seconds=lock_timeout_seconds,
        max_messages=max_messages,
        max_bytes=max_bytes,
    )


def _request(message_id: UUID, *, request_id: UUID | None = None, title: str | None = None):
    return ConversationForkRequest(
        message_id=message_id, request_id=request_id or uuid4(), title=title
    )


async def _fork(service: ConversationForkService, source: Source, request, owner_id=None):
    return await service.fork_conversation(
        source_conversation_id=source.conversation_id,
        owner_id=owner_id or source.owner_id,
        request=request,
    )


async def _refused(coroutine) -> CustomHTTPException:
    with pytest.raises(CustomHTTPException) as raised:
        await coroutine
    return raised.value


def _messages(pg: Pg, conversation_id: UUID) -> list[Message]:
    with pg.sync() as session:
        return list(
            session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence)
            ).scalars()
        )


def _children(pg: Pg, owner_id: UUID) -> list[Conversation]:
    with pg.sync() as session:
        return list(
            session.execute(
                select(Conversation).where(
                    Conversation.owner_id == owner_id,
                    Conversation.forked_from_conversation_id.is_not(None),
                )
            ).scalars()
        )


def _commit(pg: Pg, statement) -> None:
    with pg.sync.begin() as session:
        session.execute(statement)


# ---------------------------------------------------------------------------
# what is copied
# ---------------------------------------------------------------------------


async def test_fork_copies_only_the_prefix_with_new_ids_and_mapped_reply_links(pg):
    source = _six_message_source(pg, title="Trip planning", persona_prompt="Be terse")
    published: list[UUID] = []
    service = _service(pg, publisher=published.append)

    read, created = await _fork(service, source, _request(source.id_at(4)))

    assert created is True
    assert read.title == "Trip planning (fork)"
    assert read.forked_from_conversation_id == source.conversation_id
    assert read.forked_from_message_id == source.id_at(4)
    assert read.forked_through_sequence == 4
    assert published == [read.id]

    copied = _messages(pg, read.id)
    originals = _messages(pg, source.conversation_id)
    assert [row.sequence for row in copied] == [1, 2, 3, 4]
    assert {row.id for row in copied}.isdisjoint({row.id for row in originals})
    assert [row.content for row in copied] == [row.content for row in originals[:4]]
    assert [row.created_at for row in copied] == [row.created_at for row in originals[:4]]
    by_sequence = {row.sequence: row for row in copied}
    assert by_sequence[4].message_metadata["reply_to_user_message_id"] == str(by_sequence[3].id)
    assert by_sequence[2].message_metadata["reply_to_user_message_id"] == str(by_sequence[1].id)
    assert all(row.message_metadata["fork_history"] is True for row in copied)
    assert by_sequence[4].message_metadata["fork_cutoff_eligible"] is True
    assert "fork_cutoff_eligible" not in by_sequence[3].message_metadata

    with pg.sync() as session:
        child = session.get(Conversation, read.id)
        assert child.next_message_sequence == 5
        assert child.persona_prompt == "Be terse"
        assert child.plan_lifecycle is None
        job = session.get(ConversationSummaryJob, read.id)
        assert (job.status, job.requested_through_sequence) == ("pending", 4)
        assert session.get(ConversationMemorySummary, read.id) is None
        assert (
            session.execute(
                select(func.count())
                .select_from(Generation)
                .where(Generation.conversation_id == read.id)
            ).scalar_one()
            == 0
        )
    assert len(originals) == 6


async def test_deleted_rows_and_hidden_artifacts_leave_sequence_gaps(pg):
    owner_id = _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id), {})
    _completed_turn(pg, source, 1, "q1", "a1")
    _commit(pg, update(Message).where(Message.id == source.id_at(2)).values(deleted_at=func.now()))
    hidden = _message(pg, source.conversation_id, 3, ASSISTANT, "", {"paused": True})
    _completed_turn(pg, source, 4, "q4", "a5")

    read, _ = await _fork(_service(pg), source, _request(source.id_at(5)))

    assert [row.sequence for row in _messages(pg, read.id)] == [1, 4, 5]
    assert hidden not in {row.id for row in _messages(pg, read.id)}
    with pg.sync() as session:
        assert session.get(Conversation, read.id).next_message_sequence == 6


async def test_custom_agent_links_and_project_are_copied_from_live_attachments(pg):
    owner_id = _user(pg)
    project_id = uuid4()
    with pg.sync.begin() as session:
        session.add(Project(id=project_id, owner_id=owner_id, name="P"))
    source = _six_message_source(pg, owner_id, project_id=project_id, planning_mode_enabled=True)
    agents = []
    with pg.sync.begin() as session:
        for index, deleted in enumerate((False, True, False)):
            agent = CustomAgent(
                owner_id=owner_id,
                name=f"A{index}",
                slug=f"a{index}-{uuid4().hex[:6]}",
                prompt="p",
                provider_type="gemini",
                model="m",
                deleted_at=datetime.now(UTC) if deleted else None,
            )
            session.add(agent)
            session.flush()
            agents.append(agent.id)
            session.add(
                ConversationCustomAgent(
                    owner_id=owner_id,
                    conversation_id=source.conversation_id,
                    custom_agent_id=agent.id,
                    agent_order=index,
                )
            )

    read, _ = await _fork(_service(pg), source, _request(source.id_at(2)))

    assert read.project_id == project_id
    assert read.planning_mode_enabled is True
    with pg.sync() as session:
        links = session.execute(
            select(ConversationCustomAgent.custom_agent_id, ConversationCustomAgent.agent_order)
            .where(ConversationCustomAgent.conversation_id == read.id)
            .order_by(ConversationCustomAgent.agent_order)
        ).all()
    assert [tuple(link) for link in links] == [(agents[0], 0), (agents[2], 2)]


async def test_media_references_are_validated_against_the_source_owner(pg):
    owner_id = _user(pg)
    stranger = _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id), {})
    strangers_conversation = _conversation(pg, stranger)
    with pg.sync.begin() as session:
        own_chat = ChatImage(
            conversation_id=source.conversation_id,
            user_id=owner_id,
            sha256="a" * 64,
            size_bytes=1,
            storage_path="x",
        )
        foreign_chat = ChatImage(
            conversation_id=strangers_conversation,
            user_id=stranger,
            sha256="b" * 64,
            size_bytes=1,
            storage_path="y",
        )
        selected = WebImageReference(
            conversation_id=source.conversation_id,
            user_id=owner_id,
            upstream_url="https://example.com/a.png",
            provider="brave",
            lifecycle_state="selected",
        )
        pending = WebImageReference(
            conversation_id=source.conversation_id,
            user_id=owner_id,
            upstream_url="https://example.com/b.png",
            provider="brave",
            lifecycle_state="pending",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        session.add_all([own_chat, foreign_chat, selected, pending])
    question = _message(pg, source.conversation_id, 1, USER, "show me")
    metadata = {
        "images": [
            {"image_id": str(own_chat.id), "url": f"/chat-images/{own_chat.id}"},
            {"image_id": str(foreign_chat.id), "url": f"/chat-images/{foreign_chat.id}"},
        ],
        "rich_items": [
            {
                "id": "kept",
                "type": "image",
                "payload": {"url": f"/web-images/{selected.id}", "mime_type": "image/png"},
            },
            {
                "id": "pending",
                "type": "image",
                "payload": {"url": f"/web-images/{pending.id}", "mime_type": "image/png"},
            },
        ],
    }
    reply = _message(
        pg,
        source.conversation_id,
        2,
        ASSISTANT,
        "A <!--rich:kept--> B <!--rich:pending-->",
        metadata,
    )
    _generation(
        pg, source, turn=question, status=GenerationStatus.COMPLETED, assistant_message_id=reply
    )

    read, _ = await _fork(_service(pg), source, _request(reply))

    copied = _messages(pg, read.id)[1]
    assert copied.content == "A <!--rich:kept--> B <!--rich:pending-->"
    assert [image["image_id"] for image in copied.message_metadata["images"]] == [str(own_chat.id)]
    assert [item["id"] for item in copied.message_metadata["rich_items"]] == ["kept"]
    assert copied.message_metadata["rich_reference_warnings"] == [
        {"code": "unknown_rich_item", "id": "pending"}
    ]
    with pg.sync() as session:
        # No reference rows were copied, and the pending one keeps its own expiry.
        assert (
            session.execute(
                select(func.count())
                .select_from(WebImageReference)
                .where(WebImageReference.conversation_id == read.id)
            ).scalar_one()
            == 0
        )
        assert session.get(WebImageReference, pending.id).lifecycle_state == "pending"


# ---------------------------------------------------------------------------
# fork-point eligibility
# ---------------------------------------------------------------------------


def _continued_source(pg: Pg) -> tuple[Source, UUID, UUID]:
    """1 user question, 2 a paused partial epoch, 3 the final continued reply."""
    owner_id = _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id), {})
    question = _message(pg, source.conversation_id, 1, USER, "long task")
    generation_id = uuid4()
    first = _message(
        pg,
        source.conversation_id,
        2,
        ASSISTANT,
        "first half",
        {
            "partial": True,
            "continuable": True,
            "generation_id": str(generation_id),
            "logical_turn_id": str(question),
            "execution_epoch": 0,
        },
    )
    final = _message(
        pg,
        source.conversation_id,
        3,
        ASSISTANT,
        "second half",
        {
            "partial": True,
            "continued": True,
            "generation_id": str(generation_id),
            "logical_turn_id": str(question),
            "execution_epoch": 1,
        },
    )
    with pg.sync.begin() as session:
        session.add(
            Generation(
                id=generation_id,
                logical_turn_id=str(question),
                checkpoint_thread_id=f"{source.conversation_id}:{question}",
                user_id=owner_id,
                conversation_id=source.conversation_id,
                status=GenerationStatus.COMPLETED_PARTIAL,
                assistant_message_id=final,
                continuation_available=False,
            )
        )
    return source, first, final


async def test_substantive_partial_history_is_copied_but_is_not_a_fork_point(pg):
    source, first, final = _continued_source(pg)
    service = _service(pg)

    refused = await _refused(_fork(service, source, _request(first)))
    assert (refused.status_code, refused.error_code) == (409, "FORK_POINT_NOT_COMPLETED")

    read, _ = await _fork(service, source, _request(final))
    copied = {row.sequence: row for row in _messages(pg, read.id)}
    assert copied[2].content == "first half"
    assert copied[2].message_metadata["fork_cutoff_eligible"] is False
    assert copied[3].message_metadata["fork_cutoff_eligible"] is True
    for row in copied.values():
        assert not {"generation_id", "logical_turn_id", "execution_epoch", "continuable"} & set(
            row.message_metadata
        )


async def test_nested_fork_rejects_copied_partials_and_accepts_copied_final_replies(pg):
    source, _, final = _continued_source(pg)
    service = _service(pg)
    child, _ = await _fork(service, source, _request(final))
    copied = {row.sequence: row.id for row in _messages(pg, child.id)}
    child_source = Source(source.owner_id, child.id, copied)

    refused = await _refused(_fork(service, child_source, _request(copied[2])))
    grandchild, created = await _fork(service, child_source, _request(copied[3]))

    assert (refused.status_code, refused.error_code) == (409, "FORK_POINT_NOT_COMPLETED")
    assert created is True
    assert grandchild.title == "Source (fork) (fork)"
    nested = {row.sequence: row for row in _messages(pg, grandchild.id)}
    assert nested[2].message_metadata["fork_cutoff_eligible"] is False
    assert nested[3].message_metadata["fork_cutoff_eligible"] is True
    assert grandchild.forked_from_conversation_id == child.id


async def test_cutoff_must_be_a_live_assistant_message_of_the_source(pg):
    source = _six_message_source(pg)
    other = _six_message_source(pg, source.owner_id)
    hidden = _message(pg, source.conversation_id, 7, ASSISTANT, "", {"paused": True})
    _commit(pg, update(Message).where(Message.id == source.id_at(6)).values(deleted_at=func.now()))
    service = _service(pg)

    cases = [
        (uuid4(), 404, "FORK_MESSAGE_NOT_FOUND"),
        (other.id_at(2), 404, "FORK_MESSAGE_NOT_FOUND"),
        (source.id_at(6), 404, "FORK_MESSAGE_NOT_FOUND"),
        (source.id_at(3), 409, "FORK_POINT_NOT_COMPLETED"),
        (hidden, 409, "FORK_POINT_NOT_COMPLETED"),
    ]
    for message_id, status, code in cases:
        refused = await _refused(_fork(service, source, _request(message_id)))
        assert (refused.status_code, refused.error_code) == (status, code), message_id
    assert _children(pg, source.owner_id) == []


async def test_live_or_abandoned_active_generation_refuses_the_fork(pg):
    source = _six_message_source(pg)
    later_turn = _message(pg, source.conversation_id, 7, USER, "still running")
    _generation(
        pg, source, turn=later_turn, status=GenerationStatus.STARTING, assistant_message_id=None
    )

    refused = await _refused(_fork(_service(pg), source, _request(source.id_at(4))))

    assert (refused.status_code, refused.error_code) == (409, "conversation_turn_conflict")


async def test_earlier_reply_forks_while_a_later_turn_awaits_approval(pg):
    source = _six_message_source(pg)
    paused_question = _message(pg, source.conversation_id, 7, USER, "delete the files")
    paused_reply = _message(
        pg,
        source.conversation_id,
        8,
        ASSISTANT,
        "I need approval to continue.",
        {"paused": True, "interrupt": {"interrupt_id": "i-1"}, "thread_id": "t"},
    )
    _generation(
        pg,
        source,
        turn=paused_question,
        status=GenerationStatus.CONTINUABLE,
        assistant_message_id=paused_reply,
    )
    service = _service(pg)

    read, created = await _fork(service, source, _request(source.id_at(6)))
    refused = await _refused(_fork(service, source, _request(paused_reply)))

    assert created is True and read.forked_through_sequence == 6
    assert (refused.status_code, refused.error_code) == (409, "FORK_POINT_NOT_COMPLETED")


# ---------------------------------------------------------------------------
# ownership, guards and replay
# ---------------------------------------------------------------------------


async def test_owner_isolation_for_sources_and_request_ids(pg):
    source = _six_message_source(pg)
    stranger_source = _six_message_source(pg)
    service = _service(pg)
    request_id = uuid4()

    denied = await _refused(
        _fork(service, source, _request(source.id_at(2)), owner_id=stranger_source.owner_id)
    )
    missing = await _refused(
        service.fork_conversation(
            source_conversation_id=uuid4(),
            owner_id=source.owner_id,
            request=_request(source.id_at(2)),
        )
    )
    mine, mine_created = await _fork(
        service, source, _request(source.id_at(2), request_id=request_id)
    )
    theirs, theirs_created = await _fork(
        service, stranger_source, _request(stranger_source.id_at(2), request_id=request_id)
    )

    assert (denied.status_code, denied.error_code) == (403, "CONVERSATION_ACCESS_DENIED")
    assert (missing.status_code, missing.error_code) == (404, "CONVERSATION_NOT_FOUND")
    assert mine_created and theirs_created and mine.id != theirs.id
    assert theirs.owner_id == stranger_source.owner_id


async def test_deleted_source_is_not_found(pg):
    source = _six_message_source(pg)
    _commit(
        pg,
        update(Conversation)
        .where(Conversation.id == source.conversation_id)
        .values(deleted_at=func.now()),
    )

    refused = await _refused(_fork(_service(pg), source, _request(source.id_at(2))))

    assert (refused.status_code, refused.error_code) == (404, "CONVERSATION_NOT_FOUND")


async def test_documents_and_oversized_prefixes_are_refused_before_copying(pg):
    source = _six_message_source(pg)
    sized = _six_message_source(pg, source.owner_id)
    with pg.sync.begin() as session:
        session.add(
            Document(
                conversation_id=source.conversation_id,
                filename="a.pdf",
                filename_key="a.pdf",
                file_type="application/pdf",
            )
        )

    documents = await _refused(_fork(_service(pg), source, _request(source.id_at(2))))
    too_many = await _refused(_fork(_service(pg, max_messages=3), sized, _request(sized.id_at(4))))
    too_big = await _refused(_fork(_service(pg, max_bytes=40), sized, _request(sized.id_at(4))))
    exact, _ = await _fork(_service(pg, max_messages=4), sized, _request(sized.id_at(4)))

    assert (documents.status_code, documents.error_code) == (
        409,
        "FORK_DOCUMENT_CONTEXT_UNSUPPORTED",
    )
    assert (too_many.status_code, too_many.error_code) == (413, "FORK_HISTORY_TOO_LARGE")
    assert (too_big.status_code, too_big.error_code) == (413, "FORK_HISTORY_TOO_LARGE")
    assert len(_messages(pg, exact.id)) == 4
    assert len(_children(pg, source.owner_id)) == 1


async def test_replay_conflict_and_deleted_result(pg):
    source = _six_message_source(pg)
    published: list[UUID] = []
    service = _service(pg, publisher=published.append)
    request_id = uuid4()

    first, first_created = await _fork(
        service, source, _request(source.id_at(4), request_id=request_id, title="Alt")
    )
    again, again_created = await _fork(
        service, source, _request(source.id_at(4), request_id=request_id, title="Alt")
    )
    conflict = await _refused(
        _fork(service, source, _request(source.id_at(4), request_id=request_id, title="Other"))
    )
    _commit(
        pg,
        update(Conversation)
        .where(Conversation.id == source.conversation_id)
        .values(deleted_at=func.now()),
    )
    after_source_delete, replayed = await _fork(
        service, source, _request(source.id_at(4), request_id=request_id, title="Alt")
    )
    _commit(
        pg, update(Conversation).where(Conversation.id == first.id).values(deleted_at=func.now())
    )
    gone = await _refused(
        _fork(service, source, _request(source.id_at(4), request_id=request_id, title="Alt"))
    )

    assert (first_created, again_created, replayed) == (True, False, False)
    assert first.id == again.id == after_source_delete.id
    assert first.title == "Alt"
    assert (conflict.status_code, conflict.error_code) == (409, "FORK_REQUEST_CONFLICT")
    assert (gone.status_code, gone.error_code) == (410, "FORK_RESULT_DELETED")
    assert published == [first.id]
    assert len(_children(pg, source.owner_id)) == 1


async def test_failure_after_message_insert_leaves_no_rows(pg, monkeypatch):
    source = _six_message_source(pg)
    with pg.sync.begin() as session:
        agent = CustomAgent(
            owner_id=source.owner_id, name="A", slug="a", prompt="p", provider_type="g", model="m"
        )
        session.add(agent)
        session.flush()
        session.add(
            ConversationCustomAgent(
                owner_id=source.owner_id,
                conversation_id=source.conversation_id,
                custom_agent_id=agent.id,
            )
        )
    inserted: list[int] = []
    original = ConversationForkRepository._insert_messages

    def insert_then_fail(session, child_id, rows, eligibility, available):
        original(session, child_id, rows, eligibility, available)
        inserted.append(
            session.execute(
                select(func.count()).select_from(Message).where(Message.conversation_id == child_id)
            ).scalar_one()
        )
        raise RuntimeError("injected failure after message insert")

    monkeypatch.setattr(
        ConversationForkRepository, "_insert_messages", staticmethod(insert_then_fail)
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        await _fork(_service(pg), source, _request(source.id_at(4)))

    assert inserted == [4]
    assert _children(pg, source.owner_id) == []
    with pg.sync() as session:
        owned = select(Conversation.id).where(Conversation.owner_id == source.owner_id)
        assert (
            session.execute(
                select(func.count()).select_from(Message).where(Message.conversation_id.in_(owned))
            ).scalar_one()
            == 6
        )
        assert (
            session.execute(
                select(func.count())
                .select_from(ConversationCustomAgent)
                .where(ConversationCustomAgent.conversation_id.in_(owned))
            ).scalar_one()
            == 1
        )
        assert (
            session.execute(
                select(func.count())
                .select_from(ConversationSummaryJob)
                .where(ConversationSummaryJob.conversation_id.in_(owned))
            ).scalar_one()
            == 0
        )


async def test_refusals_do_not_escape_the_session_as_errors(pg):
    """The app's session context logs any escaping exception as a rollback failure."""
    source = _six_message_source(pg)
    escaped: list[BaseException] = []

    @contextlib.asynccontextmanager
    async def watched_session():
        async with pg.async_factory() as session:
            try:
                yield session
            except BaseException as exc:
                escaped.append(exc)
                raise

    service = _service(pg, session_factory=watched_session)

    not_completed = await _refused(_fork(service, source, _request(source.id_at(3))))
    denied = await _refused(_fork(service, source, _request(source.id_at(2)), owner_id=_user(pg)))

    assert not_completed.error_code == "FORK_POINT_NOT_COMPLETED"
    assert denied.error_code == "CONVERSATION_ACCESS_DENIED"
    assert escaped == []
    assert _children(pg, source.owner_id) == []


async def test_publish_happens_after_commit_and_a_failed_publish_is_reconciled(pg):
    source = _six_message_source(pg)
    seen_at_publish: list[bool] = []

    def publisher(child_id: UUID) -> None:
        with pg.sync() as session:
            seen_at_publish.append(session.get(Conversation, child_id) is not None)
        raise RuntimeError("broker down")

    read, created = await _fork(
        _service(pg, publisher=publisher), source, _request(source.id_at(4))
    )

    assert created is True
    assert seen_at_publish == [True]
    due = ConversationCompactionRepository(pg.sync).reconcile_due_jobs(
        now=datetime.now(UTC) + timedelta(seconds=1), limit=1000
    )
    assert read.id in due


async def test_generation_rows_are_loaded_in_a_bounded_number_of_queries(pg):
    source = _six_message_source(pg)
    for sequence in range(7, 47, 2):
        _completed_turn(pg, source, sequence, f"q{sequence}", f"a{sequence + 1}")
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(pg.async_engine.sync_engine, "before_cursor_execute", record)
    try:
        await _fork(_service(pg), source, _request(source.id_at(46)))
    finally:
        event.remove(pg.async_engine.sync_engine, "before_cursor_execute", record)

    assert sum("FROM generations" in statement for statement in statements) == 2


# ---------------------------------------------------------------------------
# isolation, locking and retries
# ---------------------------------------------------------------------------


async def test_fork_transaction_runs_at_repeatable_read(pg, monkeypatch):
    source = _six_message_source(pg)
    observed: list[str] = []
    original = ConversationForkRepository._load_prefix

    def observe(session, plan, cutoff_sequence):
        observed.append(session.execute(text("SHOW transaction_isolation")).scalar_one())
        return original(session, plan, cutoff_sequence)

    monkeypatch.setattr(ConversationForkRepository, "_load_prefix", staticmethod(observe))

    await _fork(_service(pg), source, _request(source.id_at(4)))

    assert observed == ["repeatable read"]


async def test_concurrent_edit_and_cutoff_delete_do_not_tear_the_snapshot(pg, monkeypatch):
    """Message CRUD takes no turn lock; one snapshot keeps checks and copy consistent."""
    source = _six_message_source(pg)
    original = ConversationForkRepository._load_prefix

    def edit_then_load(session, plan, cutoff_sequence):
        _commit(pg, update(Message).where(Message.id == source.id_at(1)).values(content="edited"))
        _commit(
            pg, update(Message).where(Message.id == source.id_at(4)).values(deleted_at=func.now())
        )
        return original(session, plan, cutoff_sequence)

    monkeypatch.setattr(ConversationForkRepository, "_load_prefix", staticmethod(edit_then_load))

    read, _ = await _fork(_service(pg), source, _request(source.id_at(4)))

    copied = _messages(pg, read.id)
    assert [row.sequence for row in copied] == [1, 2, 3, 4]
    assert copied[0].content == "question 1"


async def test_source_delete_waits_for_the_fork_row_lock(pg, monkeypatch):
    source = _six_message_source(pg)
    original = ConversationForkRepository._insert_messages
    deleter: dict[str, Any] = {}

    def delete_source() -> None:
        _commit(
            pg,
            update(Conversation)
            .where(Conversation.id == source.conversation_id)
            .values(deleted_at=func.now()),
        )

    def insert_while_delete_waits(session, child_id, rows, eligibility, available):
        thread = threading.Thread(target=delete_source, daemon=True)
        thread.start()
        time.sleep(0.5)
        deleter["blocked"] = thread.is_alive()
        deleter["thread"] = thread
        original(session, child_id, rows, eligibility, available)

    monkeypatch.setattr(
        ConversationForkRepository, "_insert_messages", staticmethod(insert_while_delete_waits)
    )

    read, created = await _fork(_service(pg), source, _request(source.id_at(4)))
    deleter["thread"].join(timeout=10)

    assert deleter["blocked"] is True
    assert created is True and len(_messages(pg, read.id)) == 4
    with pg.sync() as session:
        assert session.get(Conversation, source.conversation_id).deleted_at is not None
        assert session.get(Conversation, read.id).deleted_at is None


async def test_serialization_failure_retries_on_a_fresh_snapshot(pg, monkeypatch):
    source = _six_message_source(pg, title="Before")
    original = ConversationForkRepository._lock_source
    attempts: list[int] = []

    def rename_then_lock(self, session, plan):
        attempts.append(1)
        if len(attempts) == 1:
            _commit(
                pg,
                update(Conversation)
                .where(Conversation.id == source.conversation_id)
                .values(title="After"),
            )
        return original(self, session, plan)

    monkeypatch.setattr(ConversationForkRepository, "_lock_source", rename_then_lock)

    read, created = await _fork(_service(pg), source, _request(source.id_at(4)))

    assert len(attempts) == 2
    assert created is True and read.title == "After (fork)"


async def test_exhausted_serialization_retries_become_a_turn_conflict(pg, monkeypatch):
    source = _six_message_source(pg)
    original = ConversationForkRepository._lock_source
    attempts: list[int] = []

    def always_conflict(self, session, plan):
        attempts.append(1)
        _commit(
            pg,
            update(Conversation)
            .where(Conversation.id == source.conversation_id)
            .values(title=f"t{len(attempts)}"),
        )
        return original(self, session, plan)

    monkeypatch.setattr(ConversationForkRepository, "_lock_source", always_conflict)

    refused = await _refused(_fork(_service(pg), source, _request(source.id_at(4))))

    assert (refused.status_code, refused.error_code) == (409, "conversation_turn_conflict")
    assert len(attempts) == 3
    assert _children(pg, source.owner_id) == []


async def test_turn_append_after_the_snapshot_retries_without_touching_the_prefix(pg, monkeypatch):
    """A concurrent turn persisting its reply bumps the source's sequence allocator."""
    source = _six_message_source(pg)
    original = ConversationForkRepository._lock_source
    appended: list[int] = []

    def append_then_lock(self, session, plan):
        if not appended:
            message = ConversationCompactionRepository(pg.sync).persist_message(
                {
                    "id": uuid4(),
                    "conversation_id": source.conversation_id,
                    "sender": ASSISTANT,
                    "content": "late reply",
                    "message_metadata": {},
                }
            )
            appended.append(message.sequence)
        return original(self, session, plan)

    monkeypatch.setattr(ConversationForkRepository, "_lock_source", append_then_lock)

    read, _ = await _fork(_service(pg), source, _request(source.id_at(4)))

    assert appended == [7]
    assert [row.sequence for row in _messages(pg, read.id)] == [1, 2, 3, 4]
    assert len(_messages(pg, source.conversation_id)) == 7


async def test_held_source_row_lock_yields_a_bounded_turn_conflict(pg):
    source = _six_message_source(pg)
    holder = pg.sync()
    holder.execute(
        select(Conversation.id).where(Conversation.id == source.conversation_id).with_for_update()
    )
    started = time.monotonic()
    try:
        refused = await _refused(
            asyncio.wait_for(
                _fork(_service(pg, lock_timeout_seconds=1.0), source, _request(source.id_at(4))),
                timeout=20,
            )
        )
    finally:
        holder.rollback()
        holder.close()

    assert (refused.status_code, refused.error_code) == (409, "conversation_turn_conflict")
    assert time.monotonic() - started < 10
    assert _children(pg, source.owner_id) == []


async def test_running_turn_holding_the_advisory_lock_refuses_the_fork(pg):
    source = _six_message_source(pg)
    coordinator = ConversationTurnCoordinator(
        backend=PostgresAdvisoryLockBackend(pg.sync, poll_interval_seconds=0.02),
        timeout_seconds=0.3,
    )
    service = _service(pg, coordinator=coordinator)

    async with coordinator.hold(str(source.conversation_id), request_id="continue-1"):
        refused = await _refused(_fork(service, source, _request(source.id_at(4))))
    after, created = await _fork(service, source, _request(source.id_at(4)))

    assert (refused.status_code, refused.error_code) == (409, "conversation_turn_conflict")
    assert created is True and after.forked_through_sequence == 4


async def test_preflight_snapshot_is_closed_before_the_turn_lock(pg):
    source = _six_message_source(pg, title="Before lock")
    open_sessions: list[int] = []
    sessions_at_lock: list[int] = []

    class Tracked:
        def __init__(self) -> None:
            self.session = pg.async_factory()

        async def __aenter__(self):
            open_sessions.append(1)
            return await self.session.__aenter__()

        async def __aexit__(self, *exc_info):
            open_sessions.pop()
            return await self.session.__aexit__(*exc_info)

    class RenamingBackend(InProcessTurnLockBackend):
        async def acquire(self, key, *, timeout_seconds):
            sessions_at_lock.append(len(open_sessions))
            _commit(
                pg,
                update(Conversation)
                .where(Conversation.id == source.conversation_id)
                .values(title="After lock"),
            )
            return await super().acquire(key, timeout_seconds=timeout_seconds)

    service = _service(
        pg,
        coordinator=ConversationTurnCoordinator(backend=RenamingBackend(), timeout_seconds=5),
        session_factory=Tracked,
    )

    read, _ = await _fork(service, source, _request(source.id_at(4)))

    assert sessions_at_lock == [0]
    assert read.title == "After lock (fork)"


# ---------------------------------------------------------------------------
# request-ID contention across concurrent forks
# ---------------------------------------------------------------------------


async def _wait_for_lock_waiters(pg: Pg, expected: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with pg.sync() as session:
            waiting = session.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
        if waiting >= expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"expected {expected} sessions waiting on a lock")


async def _race(pg: Pg, held: list[UUID], forks: list[Any]) -> list[Any]:
    """Start every fork while ``held`` rows are locked, then release them together.

    Both transactions take their snapshot and pass the in-transaction replay
    check before either can insert, which is the window the unique constraint
    exists for.
    """
    holder = pg.sync()
    holder.execute(select(Conversation.id).where(Conversation.id.in_(held)).with_for_update())
    tasks = [asyncio.ensure_future(fork) for fork in forks]
    try:
        await _wait_for_lock_waiters(pg, len(forks))
    finally:
        holder.rollback()
        holder.close()
    return await asyncio.gather(*tasks, return_exceptions=True)


def _independent_service(pg: Pg) -> ConversationForkService:
    """Its own in-process lock: models two workers the turn lock does not serialize."""
    return _service(pg, lock_timeout_seconds=15)


async def test_same_request_id_on_two_sources_creates_exactly_one_child(pg):
    first = _six_message_source(pg)
    second = _six_message_source(pg, first.owner_id)
    request_id = uuid4()

    results = await _race(
        pg,
        [first.conversation_id, second.conversation_id],
        [
            _fork(_independent_service(pg), first, _request(first.id_at(4), request_id=request_id)),
            _fork(
                _independent_service(pg), second, _request(second.id_at(4), request_id=request_id)
            ),
        ],
    )

    created = [result for result in results if isinstance(result, tuple)]
    refused = [result for result in results if isinstance(result, CustomHTTPException)]
    assert len(created) == 1 and created[0][1] is True, results
    assert [(error.status_code, error.error_code) for error in refused] == [
        (409, "FORK_REQUEST_CONFLICT")
    ]
    assert len(_children(pg, first.owner_id)) == 1


async def test_identical_concurrent_requests_converge_on_one_child(pg):
    source = _six_message_source(pg)
    request_id = uuid4()

    results = await _race(
        pg,
        [source.conversation_id],
        [
            _fork(
                _independent_service(pg), source, _request(source.id_at(4), request_id=request_id)
            )
            for _ in range(2)
        ],
    )

    assert all(isinstance(result, tuple) for result in results), results
    assert {read.id for read, _ in results} == {_children(pg, source.owner_id)[0].id}
    assert sorted(created for _, created in results) == [False, True]
    assert len(_children(pg, source.owner_id)) == 1
