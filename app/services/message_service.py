from __future__ import annotations
from typing import List, Optional
from uuid import UUID

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


logger = logging.getLogger(__name__)


class MessageService(IMessageService):
    """Service layer for Message operations"""

    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_validation_utils: ConversationValidationUtils,
        message_validation_utils: MessageValidationUtils,
        ai_service: Optional[AIService] = None,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        # Initialize AI service for bot response generation
        self.ai_service = ai_service or AIService()

    async def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_exists(
            message_create_data.conversation_id
        )

        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )

        created_message = self.repository.create(message_entity)

        if message_create_data.role == MessageRole.user:
            # Check if there are documents being processed for this conversation
            processing_message = self._check_processing_documents(
                message_create_data.conversation_id
            )

            if processing_message:
                bot_response_content = processing_message
            else:
                # Delegate bot response generation to AIService
                bot_response_content = await self.ai_service.generate_bot_response(
                    user_message=message_create_data.content,
                    conversation_id=message_create_data.conversation_id,
                    user_id=None,  # Can be extracted from conversation if needed
                )

            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=bot_response_content,
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
        # Validate pagination parameters
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
        # Validate pagination parameters
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

    def _check_processing_documents(self, conversation_id: UUID) -> Optional[str]:
        """Check if there are documents being processed for this conversation."""
        from app.database.session import get_db
        from app.models.document import Document
        from app.schemas.document import DocumentStatus
        from sqlalchemy.orm import Session

        db: Session = next(get_db())
        try:
            # Check for documents in processing state for this conversation
            processing_docs = (
                db.query(Document)
                .filter(
                    Document.conversation_id == conversation_id,
                    Document.status == DocumentStatus.PROCESSING.value,
                )
                .all()
            )

            if processing_docs:
                doc_names = [doc.filename for doc in processing_docs]
                if len(doc_names) == 1:
                    return f"Your document '{doc_names[0]}' is still being processed. Please wait for processing to complete before asking questions about it."
                else:
                    return f"Your documents {', '.join(doc_names)} are still being processed. Please wait for processing to complete before asking questions about them."

            return None
        finally:
            db.close()
