from __future__ import annotations

from uuid import uuid4

import pytest

from app.api.conversations import get_conversations
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
