"""No sync database call may run on the event loop before the first token.

Phase 1 removed the three blocking calls in ``create_message_stream`` itself.
Assembling the workflow request still blocked, deeper down: the task-plan and
custom-agent services each performed their own synchronous access checks and
lookups.

This test drives the real container-built ``MessageService`` and asserts the
count of sync-engine checkouts taken while the loop is running.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.message import MessageCreate

pytestmark = pytest.mark.selector_event_loop


@pytest.fixture
def message_service(require_async_db):
    from app.core.container import get_container

    return get_container().message_service()


async def test_workflow_request_assembly_makes_no_sync_db_calls(
    message_service, seeded_conversation_id, forbid_sync_db_on_event_loop
):
    conversation = (
        await message_service.conversation_validation_utils.conversation_repository.aget_by_id(
            seeded_conversation_id
        )
    )
    assert conversation is not None

    # Seeding and the conversation load above happen before the guard matters;
    # clear anything recorded so far so the assertion covers only the path
    # under test.
    forbid_sync_db_on_event_loop.clear()

    await message_service._build_user_message_workflow_request(
        message_create_data=MessageCreate(conversation_id=seeded_conversation_id, content="hi"),
        user_id=conversation.owner_id,
        conversation=conversation,
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
    )

    assert forbid_sync_db_on_event_loop == [], (
        "workflow-request assembly still blocks the event loop on the sync "
        "engine at:\n  " + "\n  ".join(dict.fromkeys(forbid_sync_db_on_event_loop))
    )


async def test_guard_detects_a_deliberate_sync_call(forbid_sync_db_on_event_loop, require_async_db):
    """The guard must actually fire, or the test above is vacuous."""
    from sqlalchemy import text

    from app.database.session import SessionLocal

    with SessionLocal() as session:
        session.execute(text("select 1"))

    assert forbid_sync_db_on_event_loop, "guard failed to notice a sync session on the loop"
