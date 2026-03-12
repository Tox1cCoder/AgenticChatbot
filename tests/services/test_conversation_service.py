import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.models.enums import PlanLifecycle


def _make_conversation():
    conversation = MagicMock()
    conversation.id = uuid.uuid4()
    conversation.created_at = datetime.now(timezone.utc)
    conversation.updated_at = datetime.now(timezone.utc)
    conversation.deleted_at = None
    conversation.owner_id = uuid.uuid4()
    conversation.title = "Test conversation"
    conversation.persona_prompt = "Be concise"
    conversation.planning_mode_enabled = True
    conversation.plan_lifecycle = PlanLifecycle.executing
    return conversation


class TestConversationSchemas:
    def test_public_update_schema_hides_plan_lifecycle(self):
        from app.schemas.conversation import ConversationUpdate

        assert "plan_lifecycle" not in ConversationUpdate.model_fields


class TestConversationReadSerialization:
    def test_convert_to_read_schema_includes_planning_fields(self):
        from app.services.conversation_service import ConversationService

        service = ConversationService(
            conversation_repository=MagicMock(),
            user_validation_utils=MagicMock(),
            conversation_validation_utils=MagicMock(),
        )

        result = service._convert_to_read_schema(_make_conversation(), include=[])

        assert result.planning_mode_enabled is True
        assert result.plan_lifecycle == PlanLifecycle.executing
