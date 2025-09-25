from __future__ import annotations
from typing import List, Optional
from uuid import UUID

from google import genai

from app.core.config import settings
from app.repositories.message import MessageRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole
from app.factories.message_factory import MessageFactory
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params
from app.interfaces.message_service_interface import IMessageService
from app.services.ai_service import AIService
import logging


class MessageService(IMessageService):
    """Service layer for Message operations"""

    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_validation_utils: ConversationValidationUtils,
        message_validation_utils: MessageValidationUtils,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils

    def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_exists(
            message_create_data.conversation_id
        )

        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )

        created_message = self.repository.create(message_entity)

        if message_create_data.role == MessageRole.user:
            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=self._generate_bot_response(message_create_data.content),
            )
            self.repository.create(bot_response_entity)

        return MessageRead.model_validate(created_message)

    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        return MessageRead.model_validate(message_entity)

    def get_conversation_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
    ) -> Paginator[MessageRead]:
        # Validate pagination parameters at service layer
        validate_pagination_params(page, limit)

        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )
        paginated_messages = self.repository.get_by_conversation_id(
            conversation_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
        )
        # Convert items to MessageRead schemas
        message_reads = [
            MessageRead.model_validate(msg) for msg in paginated_messages.items
        ]
        # Return new Paginator with converted items
        return Paginator.create(
            message_reads, paginated_messages.meta.total, page, limit
        )

    def get_user_messages(
        self,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> Paginator[MessageRead]:
        # Validate pagination parameters at service layer
        validate_pagination_params(page, limit)

        paginated_messages = self.repository.get_by_user_id(
            user_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
        )
        # Convert items to MessageRead schemas
        message_reads = [
            MessageRead.model_validate(msg) for msg in paginated_messages.items
        ]
        # Return new Paginator with converted items
        return Paginator.create(
            message_reads, paginated_messages.meta.total, page, limit
        )

    def update_message(
        self,
        message_id: UUID,
        user_id: UUID,
        message_update_data: MessageUpdate,
    ) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        updated_message = self.repository.update(message_entity.id, message_update_data)
        return MessageRead.model_validate(updated_message)

    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        return self.repository.delete(message_id)

    def _generate_bot_response(self, user_message: str) -> str:
        """Generate bot response using the multi-agent system."""
        try:
            ai_service = AIService()
            return ai_service.get_bot_response_sync(user_message)

        except ImportError as e:
            logger = logging.getLogger(__name__)
            return self._generate_bot_response_fallback(user_message)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"AI service error: {str(e)}")
            return self._generate_bot_response_fallback(user_message)

    def _generate_bot_response_fallback(self, user_message: str) -> str:
        """Fallback bot response generation using original Gemini implementation."""
        api_key = settings.gemini_api_key
        if not api_key:
            return "[Error: Gemini API key not configured]"
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        system_prompt = "You are a helpful assistant."
        prompt = f"{system_prompt}\nUser: {user_message}"

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"[Gemini API error: {str(e)}]"
