from typing import List, Optional, TYPE_CHECKING
from uuid import UUID
from fastapi import HTTPException, status

from app.repositories.message import MessageRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole
from app.factories.message_factory import MessageFactory

if TYPE_CHECKING:
    from app.core.container import DIContainer


class MessageService:
    """Service layer for Message operations"""

    def __init__(
        self,
        container: "DIContainer",
        message_repository: MessageRepository,
        conversation_repository: ConversationRepository,
        user_repository: UserRepository,
    ):
        """
        Initialize MessageService with injected dependencies.

        Args:
            container: DI container for additional dependency resolution
            message_repository: Injected message repository
            conversation_repository: Injected conversation repository
            user_repository: Injected user repository
        """
        self.container = container
        self.repository = message_repository
        self.conversation_repository = conversation_repository
        self.user_repository = user_repository

    def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        """Create a new message with validation"""
        # Validate conversation exists
        if not self.conversation_repository.exists(message_create_data.conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Validate parent message exists if provided
        if message_create_data.parent_message_id:
            parent_message_entity = self.repository.get_by_id(
                message_create_data.parent_message_id
            )
            if not parent_message_entity:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Parent message not found",
                )

            # Ensure parent message is in the same conversation
            if (
                parent_message_entity.conversation_id
                != message_create_data.conversation_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message must be in the same conversation",
                )

        # Create message entity using factory
        message_entity = MessageFactory.create_from_schema(message_create_data)

        # Save to repository
        created_message = self.repository.create(message_entity)

        # If this is a user message, generate a simple bot response
        if message_create_data.role == MessageRole.user:
            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=self._generate_bot_response(message_create_data.content),
                parent_message_id=created_message.id,  # Reply to the user message
            )
            self.repository.create(bot_response_entity)

        return MessageRead.model_validate(created_message)

    def get_message_by_id(self, message_id: UUID) -> MessageRead:
        """Get message by ID"""
        message_entity = self.repository.get_by_id(message_id)
        if not message_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )
        return MessageRead.model_validate(message_entity)

    def get_conversation_messages(
        self, conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[MessageRead]:
        """Get messages for a conversation with access validation"""
        # Validate user has access to conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        message_entities = self.repository.get_by_conversation_id(
            conversation_id, skip=skip, limit=limit
        )
        return [MessageRead.model_validate(msg) for msg in message_entities]

    def get_conversation_thread(
        self, conversation_id: UUID, user_id: UUID
    ) -> List[MessageRead]:
        """Get conversation thread ordered by timestamp"""
        # Validate user has access to conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        message_entities = self.repository.get_conversation_thread(conversation_id)
        return [MessageRead.model_validate(msg) for msg in message_entities]

    def get_message_replies(
        self, parent_message_id: UUID, user_id: UUID
    ) -> List[MessageRead]:
        """Get all replies to a specific message"""
        # Validate parent message exists and user has access
        parent_message_entity = self.repository.get_by_id(parent_message_id)
        if not parent_message_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Parent message not found"
            )

        if not self.conversation_repository.user_owns_conversation(
            user_id, parent_message_entity.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        reply_entities = self.repository.get_message_replies(parent_message_id)
        return [MessageRead.model_validate(message) for message in reply_entities]

    def update_message(
        self, message_id: UUID, user_id: UUID, message_update_data: MessageUpdate
    ) -> MessageRead:
        """Update message with ownership validation"""
        message_entity = self.repository.get_by_id(message_id)
        if not message_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Validate user has access to the conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, message_entity.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        updated_message = self.repository.update(message_entity, message_update_data)
        return MessageRead.model_validate(updated_message)

    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        """Delete message with ownership validation"""
        message_entity = self.repository.get_by_id(message_id)
        if not message_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Validate user has access to the conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, message_entity.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        return self.repository.delete(message_id)

    def _generate_bot_response(self, user_message: str) -> str:
        """Generate a simple bot response"""
        user_message = user_message.lower()

        if "hello" in user_message or "hi" in user_message:
            return "Hello! How can I help you today?"
        elif "how are you" in user_message:
            return "I'm doing great, thank you for asking! How are you?"
        elif "bye" in user_message or "goodbye" in user_message:
            return "Goodbye! Have a great day!"
        else:
            return f"I received your message: '{user_message}'. Thanks for chatting with me!"
