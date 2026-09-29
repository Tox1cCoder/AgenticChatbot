"""Deleting a conversation schedules its checkpoint cleanup safely."""

from __future__ import annotations

import asyncio
import gc
import warnings
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.services.conversation_service as module
from app.services.conversation_service import ConversationService


class _Checkpoints:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.release = asyncio.Event()

    async def delete_thread(self, thread_id: str) -> None:
        await self.release.wait()
        self.deleted.append(thread_id)


def _service(checkpoints) -> ConversationService:
    return ConversationService(
        conversation_repository=SimpleNamespace(delete=lambda _id: True),
        user_validation_utils=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(
            validate_conversation_access=lambda *_args: None
        ),
        checkpoint_manager=checkpoints,
    )


@pytest.mark.asyncio
async def test_the_cleanup_task_is_held_until_it_finishes():
    checkpoints = _Checkpoints()
    conversation_id = uuid4()

    assert _service(checkpoints).delete_conversation(conversation_id, uuid4()) is True

    # Only a strong reference keeps a pending task alive through a collection.
    gc.collect()
    assert len(module._BACKGROUND_TASKS) == 1
    checkpoints.release.set()
    await asyncio.gather(*module._BACKGROUND_TASKS)

    assert checkpoints.deleted == [str(conversation_id)]
    assert not module._BACKGROUND_TASKS


def test_without_an_event_loop_nothing_is_left_unawaited():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        deleted = _service(SimpleNamespace(delete_thread=None)).delete_conversation(
            uuid4(), uuid4()
        )
        gc.collect()

    assert deleted is True
    assert not module._BACKGROUND_TASKS
