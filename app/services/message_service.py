from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from uuid import UUID
from fastapi import HTTPException, status

from pathlib import Path
import os
import google.genai
from dotenv import load_dotenv

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
        container: DIContainer,
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

        # Create message entity using factory
        message_entity = MessageFactory.create_from_schema(message_create_data)

        # Save to repository
        created_message = self.repository.create(message_entity)

        # If this is a user message, generate a simple bot response
        if message_create_data.sender == MessageRole.user.value:
            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=self._generate_bot_response(message_create_data.content),
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
        """Generate a bot response"""

        BASE_DIR = Path(__file__).resolve().parent.parent  # app/
        ENV_PATH = BASE_DIR / "core" / ".env"

        load_dotenv(dotenv_path=ENV_PATH)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "[Error: Gemini API key not configured]"

        system_prompt = (
            "You are a helpful chatbot. Please answer in a short, concise sentence."
        )
        prompt = f"{system_prompt}\nUser: {user_message}"

        try:
            client = google.genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"[Gemini API error: {str(e)}]"
