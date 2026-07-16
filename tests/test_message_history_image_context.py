from types import SimpleNamespace

from app.ai.history import db_message_to_agent_message
from app.ai.schemas import MessageRole
from app.models.enums import MessageRole as DBMessageRole

ATTACHMENTS = [{"name": "screen.png", "mime": "image/png", "data": "abc123"}]


def _db_message(sender=DBMessageRole.user.value):
    return SimpleNamespace(
        id="msg-1",
        sender=sender,
        content="look at this",
        message_metadata={"attachments": ATTACHMENTS},
        created_at=None,
        deleted_at=None,
    )


def test_history_provider_preserves_user_attachments():
    msg = db_message_to_agent_message(_db_message())

    assert msg.role == MessageRole.USER
    assert msg.attachments == ATTACHMENTS


def test_history_provider_keeps_assistant_history_text_only():
    msg = db_message_to_agent_message(_db_message(sender=DBMessageRole.assistant.value))

    assert msg.role == MessageRole.ASSISTANT
    assert msg.attachments is None
