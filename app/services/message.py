from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.message import MessageRepository
from app.repositories.conversation import ConversationRepository
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole


class MessageService:
    """Service layer for Message operations"""

    def __init__(self, db: Session):
        self.repository = MessageRepository(db)
        self.conversation_repository = ConversationRepository(db)

    def create_message(self, message_data: MessageCreate) -> MessageRead:
        """Create a new message with validation"""
        # Validate conversation exists
        if not self.conversation_repository.exists(message_data.conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Validate parent message exists if provided
        if message_data.parent_message_id:
            parent_message = self.repository.get_by_id(message_data.parent_message_id)
            if not parent_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Parent message not found",
                )

            # Ensure parent message is in the same conversation
            if parent_message.conversation_id != message_data.conversation_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message must be in the same conversation",
                )

        # Create the message
        message = self.repository.create(message_data)

        # If this is a user message, generate a simple bot response
        if message_data.role == MessageRole.user:
            bot_response_data = MessageCreate(
                conversation_id=message_data.conversation_id,
                content=self._generate_bot_response(message_data.content),
                role=MessageRole.assistant,
                parent_message_id=message.id,  # Reply to the user message
            )
            self.repository.create(bot_response_data)

        return MessageRead.model_validate(message)

    def get_message_by_id(self, message_id: UUID) -> MessageRead:
        """Get message by ID"""
        message = self.repository.get_by_id(message_id)
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )
        return MessageRead.model_validate(message)

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

        messages = self.repository.get_by_conversation_id(
            conversation_id, skip=skip, limit=limit
        )
        return [MessageRead.model_validate(msg) for msg in messages]

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

        messages = self.repository.get_conversation_thread(conversation_id)
        return [MessageRead.model_validate(msg) for msg in messages]

    def get_message_replies(
        self, parent_message_id: UUID, user_id: UUID
    ) -> List[MessageRead]:
        """Get all replies to a specific message"""
        # Validate parent message exists and user has access
        parent_message = self.repository.get_by_id(parent_message_id)
        if not parent_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Parent message not found"
            )

        if not self.conversation_repository.user_owns_conversation(
            user_id, parent_message.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        replies = self.repository.get_message_replies(parent_message_id)
        return [MessageRead.model_validate(message) for message in replies]

    def update_message(
        self, message_id: UUID, user_id: UUID, message_data: MessageUpdate
    ) -> MessageRead:
        """Update message with ownership validation"""
        message = self.repository.get_by_id(message_id)
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Validate user has access to the conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, message.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        updated_message = self.repository.update(message, message_data)
        return MessageRead.model_validate(updated_message)

    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        """Delete message with ownership validation"""
        message = self.repository.get_by_id(message_id)
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Validate user has access to the conversation
        if not self.conversation_repository.user_owns_conversation(
            user_id, message.conversation_id
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
