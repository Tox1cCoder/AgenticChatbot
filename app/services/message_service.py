from __future__ import annotations
from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status

import google.generativeai as genai

from app.core.config import settings
from app.repositories.message import MessageRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole
from app.factories.message_factory import MessageFactory
from app.services.validation_service import (
    ConversationValidationService,
    MessageValidationService,
)


class MessageService:
    """Service layer for Message operations"""

    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_repository: ConversationRepository,
        user_repository: UserRepository,
        conversation_validation_service: ConversationValidationService,
        message_validation_service: MessageValidationService,
    ):
        """
        Initialize MessageService with injected dependencies.

        Args:
            message_repository: Injected message repository
            conversation_repository: Injected conversation repository
            user_repository: Injected user repository
            conversation_validation_service: Injected conversation validation service
            message_validation_service: Injected message validation service
        """
        self.repository = message_repository
        self.conversation_repository = conversation_repository
        self.user_repository = user_repository
        self.conversation_validation_service = conversation_validation_service
        self.message_validation_service = message_validation_service

    def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        """Create a new message with role from request data"""
        # Validate conversation exists
        if not self.conversation_validation_service.validate_conversation_exists(
            message_create_data.conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Create message entity using factory with role from schema
        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )

        # Save to repository
        created_message = self.repository.create(message_entity)

        # Auto-generate bot response only for user messages
        if message_create_data.role == MessageRole.user:
            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=self._generate_bot_response(message_create_data.content),
            )
            self.repository.create(bot_response_entity)

        return MessageRead.model_validate(created_message)

    def get_message_by_id(self, message_id: UUID) -> MessageRead:
        """Get message by ID"""
        if not self.message_validation_service.validate_message_exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        message_entity = self.repository.get_by_id(message_id)
        return MessageRead.model_validate(message_entity)

    def get_conversation_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[MessageRead]:
        """Get messages for a conversation with access validation"""
        # Validate user has access to conversation
        is_valid, validation_errors = (
            self.conversation_validation_service.validate_conversation_access(
                user_id, conversation_id
            )
        )

        if not is_valid:
            if "Conversation not found" in validation_errors:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation",
                )

        message_entities = self.repository.get_by_conversation_id(
            conversation_id, skip=skip, limit=limit
        )
        return [MessageRead.model_validate(msg) for msg in message_entities]

    def get_conversation_thread(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> List[MessageRead]:
        """Get conversation thread ordered by timestamp"""
        # Validate user has access to conversation
        is_valid, validation_errors = (
            self.conversation_validation_service.validate_conversation_access(
                user_id, conversation_id
            )
        )

        if not is_valid:
            if "Conversation not found" in validation_errors:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation",
                )

        message_entities = self.repository.get_conversation_thread(conversation_id)
        return [MessageRead.model_validate(msg) for msg in message_entities]

    def update_message(
        self,
        message_id: UUID,
        user_id: UUID,
        message_update_data: MessageUpdate,
    ) -> MessageRead:
        """Update message with ownership validation"""
        # Validate message access
        is_valid, validation_errors = (
            self.message_validation_service.validate_message_access(user_id, message_id)
        )

        if not is_valid:
            if "Message not found" in validation_errors:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation",
                )

        message_entity = self.repository.get_by_id(message_id)
        updated_message = self.repository.update(message_entity, message_update_data)
        return MessageRead.model_validate(updated_message)

    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        """Delete message with ownership validation"""
        # Validate message access
        is_valid, validation_errors = (
            self.message_validation_service.validate_message_access(user_id, message_id)
        )

        if not is_valid:
            if "Message not found" in validation_errors:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation",
                )

        return self.repository.delete(message_id)

    def _generate_bot_response(self, user_message: str) -> str:
        """Generate a bot response using Gemini API"""
        api_key = settings.gemini_api_key
        if not api_key:
            return "[Error: Gemini API key not configured]"
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        system_prompt = (
            "You are a helpful chatbot. Please answer in a short, concise sentence."
        )
        prompt = f"{system_prompt}\nUser: {user_message}"

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"[Gemini API error: {str(e)}]"
