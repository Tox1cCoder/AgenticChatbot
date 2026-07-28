from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from app.api.conversations import get_conversations
from app.repositories import conversation as conversation_repository
from app.repositories.conversation import ConversationRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.pagination import ConversationPaginationParams
from app.services.conversation_service import ConversationService


class _RecordingRepository:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def get_by_owner_id(self, owner_id, **kwargs):
        self.kwargs = {"owner_id": owner_id, **kwargs}
        return Paginator.create([], 0, kwargs["page"], kwargs["limit"])


class _UserValidation:
    def validate_user_exists(self, _owner_id) -> None:
        return None


class _ConversationValidation:
    pass


@pytest.mark.asyncio
async def test_conversation_route_forwards_search_to_service() -> None:
    owner_id = uuid4()

    class Service:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def get_by_user_id(self, user_id, **kwargs):
            self.kwargs = {"user_id": user_id, **kwargs}
            return Paginator.create([], 0, kwargs["page"], kwargs["limit"])

    service = Service()

    await get_conversations(
        conversation_service=service,
        user_id=owner_id,
        pagination=ConversationPaginationParams(page=2, limit=25),
        include=["messages"],
        latest_messages=3,
        search="  Roadmap  ",
    )

    assert service.kwargs["search"] == "  Roadmap  "


def test_conversation_service_normalizes_and_forwards_search() -> None:
    owner_id = uuid4()
    repository = _RecordingRepository()
    service = ConversationService(
        conversation_repository=repository,
        user_validation_utils=_UserValidation(),
        conversation_validation_utils=_ConversationValidation(),
    )

    service.get_by_user_id(owner_id, page=1, limit=10, search="  RoadMap  ")

    assert repository.kwargs["search"] == "roadmap"


def test_conversation_service_treats_blank_search_as_unfiltered() -> None:
    repository = _RecordingRepository()
    service = ConversationService(
        conversation_repository=repository,
        user_validation_utils=_UserValidation(),
        conversation_validation_utils=_ConversationValidation(),
    )

    service.get_by_user_id(uuid4(), search="   ")

    assert repository.kwargs["search"] is None


def _postgres_sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).lower()


def test_search_query_covers_titles_and_all_live_messages_with_stable_rank() -> None:
    count_statement, page_statement = conversation_repository._build_owned_conversation_queries(
        owner_id=uuid4(),
        page=2,
        limit=10,
        order_by="updated_at",
        order_direction="desc",
        search="roadmap",
    )

    count_sql = _postgres_sql(count_statement)
    page_sql = _postgres_sql(page_statement)

    assert "conversations.owner_id" in page_sql
    assert "conversations.deleted_at is null" in page_sql
    assert "exists (select messages.id" in page_sql
    assert "messages.deleted_at is null" in page_sql
    assert "lower(messages.content)" in page_sql
    assert "case when" in page_sql
    assert "lower(conversations.title) = 'roadmap'" in page_sql
    assert "conversations.updated_at desc" in page_sql
    assert "conversations.id asc" in page_sql
    assert "offset 10" in page_sql
    assert "limit 10" in page_sql
    assert "count(conversations.id)" in count_sql


def test_blank_search_query_preserves_requested_sorting() -> None:
    _, page_statement = conversation_repository._build_owned_conversation_queries(
        owner_id=uuid4(),
        page=1,
        limit=20,
        order_by="created_at",
        order_direction="asc",
        search=None,
    )

    page_sql = _postgres_sql(page_statement)

    assert "case when" not in page_sql
    assert "conversations.created_at asc" in page_sql


def test_repository_uses_filtered_count_and_keeps_distinct_page_order() -> None:
    conversations = [
        SimpleNamespace(id=uuid4(), title="Exact"),
        SimpleNamespace(id=uuid4(), title="Prefix"),
        SimpleNamespace(id=uuid4(), title="Message"),
    ]

    class Result:
        def __init__(self, items):
            self.items = items

        def scalar(self):
            return 3

        def scalars(self):
            return self

        def all(self):
            return self.items

    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _statement):
            self.calls += 1
            if self.calls == 1:
                return Result([1, 2, 3])
            return Result(conversations)

    session = Session()

    @contextmanager
    def session_factory():
        yield session

    result = ConversationRepository(session_factory).get_by_owner_id(
        uuid4(), page=2, limit=3, search="roadmap"
    )

    assert result.items == conversations
    assert result.meta.total == 3
    assert result.meta.current_page == 2
