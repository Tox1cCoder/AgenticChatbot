from __future__ import annotations
from typing import Optional, List
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
from app.utils.text_processing import sanitize_persona
import logging


logger = logging.getLogger(__name__)


class MessageService(IMessageService):
    """Service layer for Message operations"""

    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_validation_utils: ConversationValidationUtils,
        message_validation_utils: MessageValidationUtils,
        ai_service: AIService,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        self.ai_service = ai_service

    async def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_exists(
            message_create_data.conversation_id
        )

        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )

        created_message = self.repository.create(message_entity)

        if message_create_data.role == MessageRole.user:
            # Get the user_id and persona from the conversation
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            user_id = conversation.owner_id if conversation else None
            persona = conversation.persona_prompt if conversation else None
            sanitized_persona = sanitize_persona(persona)

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            bot_response = await self.ai_service.generate_bot_response(
                user_message=message_create_data.content,
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                attachments=attachments,
            )

            bot_response_content = (
                bot_response.message.content
                if bot_response and bot_response.message
                else "Error: No response generated"
            )

            # Create metadata for bot response
            bot_metadata = dict(bot_response.metadata) if bot_response else {}
            if sanitized_persona:
                bot_metadata.setdefault("persona_used", sanitized_persona)

            if bot_response and bot_response.tool_artifacts:
                bot_metadata.setdefault("tool_artifacts", bot_response.tool_artifacts)

            # Extract images from bot response metadata
            if (
                bot_response
                and bot_response.metadata
                and "images" in bot_response.metadata
            ):
                bot_metadata["images"] = bot_response.metadata["images"]

            bot_response_entity = MessageFactory.create_bot_response(
                conversation_id=message_create_data.conversation_id,
                content=bot_response_content,
                message_metadata=bot_metadata,
            )
            self.repository.create(bot_response_entity)

        return MessageRead.model_validate(created_message)

    async def create_message_stream(self, message_create_data: MessageCreate):
        """
        Create a message and stream the bot response.
        Yields chunks as they arrive from the AI service.
        """
        self.conversation_validation_utils.validate_conversation_exists(
            message_create_data.conversation_id
        )

        # Create and persist the user message
        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )
        created_message = self.repository.create(message_entity)

        # Yield user message creation event
        yield {
            "type": "user_message_created",
            "message": MessageRead.model_validate(created_message).model_dump(
                mode="json"
            ),
        }

        if message_create_data.role == MessageRole.user:
            # Get the user_id and persona from the conversation
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            user_id = conversation.owner_id if conversation else None
            persona = conversation.persona_prompt if conversation else None
            sanitized_persona = sanitize_persona(persona)

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            # Stream bot response generation
            bot_response_content = "Error: No response generated"
            bot_response = None

            try:
                async for event in self.ai_service.generate_bot_response_stream(
                    user_message=message_create_data.content,
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    attachments=attachments,
                ):
                    event_type = event.get("type")

                    if event_type == "token":
                        # Yield token to client
                        yield {"type": "token", "content": event.get("content", "")}

                    elif event_type == "tool":
                        # Yield tool execution event
                        yield {
                            "type": "tool",
                            "name": event.get("name"),
                            "status": event.get("status"),
                        }

                    elif event_type == "interrupt":
                        # Yield interrupt event - workflow paused for human approval
                        yield {
                            "type": "interrupt",
                            "thread_id": str(message_create_data.conversation_id),
                            "next": event.get("next"),
                            "pending_tool_calls": event.get("pending_tool_calls"),
                            "message": "Workflow paused - awaiting human approval for tool execution",
                        }
                        # Workflow is paused - don't create a bot message yet
                        # The resume endpoint will handle that
                        return

                    elif event_type == "complete":
                        # Store final response
                        bot_response = event.get("response")
                        if bot_response and bot_response.message:
                            content = bot_response.message.content
                            # Ensure content is not empty
                            bot_response_content = (
                                content
                                if content and content.strip()
                                else "No response generated"
                            )
                        else:
                            bot_response_content = "Error: No response generated"

                        break

                    elif event_type == "error":
                        # Handle error
                        bot_response = event.get("response")
                        error_msg = event.get("error", "Unknown error")
                        if bot_response and bot_response.message:
                            content = bot_response.message.content
                            bot_response_content = (
                                content
                                if content and content.strip()
                                else f"Error: {error_msg}"
                            )
                        else:
                            bot_response_content = f"Error: {error_msg}"

                        break

                # Ensure content is valid (not empty)
                if not bot_response_content or not bot_response_content.strip():
                    bot_response_content = "No response generated"

                # Create metadata for bot response
                bot_metadata = dict(bot_response.metadata) if bot_response else {}
                if sanitized_persona:
                    bot_metadata.setdefault("persona_used", sanitized_persona)

                if bot_response and bot_response.tool_artifacts:
                    bot_metadata.setdefault(
                        "tool_artifacts", bot_response.tool_artifacts
                    )

                # Extract images from bot response metadata (from Search or Image Generator agents)
                if (
                    bot_response
                    and bot_response.metadata
                    and "images" in bot_response.metadata
                ):
                    bot_metadata["images"] = bot_response.metadata["images"]

                # Create and persist bot response message
                bot_response_entity = MessageFactory.create_bot_response(
                    conversation_id=message_create_data.conversation_id,
                    content=bot_response_content,
                    message_metadata=bot_metadata,
                )
                bot_message = self.repository.create(bot_response_entity)

                # Yield final completion event with full message
                yield {
                    "type": "complete",
                    "message": MessageRead.model_validate(bot_message).model_dump(
                        mode="json"
                    ),
                }

            except Exception as exc:
                logger.error(
                    f"Error in streaming message creation: {exc}", exc_info=True
                )

                # Create error bot response
                error_content = f"Error generating response: {str(exc)}"
                error_metadata = {"error": str(exc)}

                error_response_entity = MessageFactory.create_bot_response(
                    conversation_id=message_create_data.conversation_id,
                    content=error_content,
                    message_metadata=error_metadata,
                )
                error_message = self.repository.create(error_response_entity)

                yield {
                    "type": "error",
                    "error": str(exc),
                    "message": MessageRead.model_validate(error_message).model_dump(
                        mode="json"
                    ),
                }

    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        if hasattr(message_entity, "content") and (
            not message_entity.content or not message_entity.content.strip()
        ):
            message_entity.content = "[Empty message]"
        return MessageRead.model_validate(message_entity)

    def get_conversation_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
        include_feedback: bool = False,
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
            include_feedback=include_feedback,
        )
        message_reads = []
        for msg in paginated_messages.items:
            if hasattr(msg, "content") and (not msg.content or not msg.content.strip()):
                msg.content = "[Empty message]"
            message_reads.append(MessageRead.model_validate(msg))

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
        include_feedback: bool = False,
    ) -> Paginator[MessageRead]:
        # Validate pagination parameters
        validate_pagination_params(page, limit)

        paginated_messages = self.repository.get_by_user_id(
            user_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
            include_feedback=include_feedback,
        )
        message_reads = []
        for msg in paginated_messages.items:
            if hasattr(msg, "content") and (not msg.content or not msg.content.strip()):
                msg.content = "[Empty message]"
            message_reads.append(MessageRead.model_validate(msg))

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

    async def resume_workflow(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_input: Optional[str] = None,
        rejection_messages: Optional[List] = None,
    ) -> MessageRead:
        """
        Resume a paused workflow and return the bot's response message.

        Args:
            conversation_id: The conversation ID
            user_id: The user ID
            user_input: Optional user input (not currently used)
            rejection_messages: Optional list of ToolMessages indicating tool rejection
        """
        # Validate conversation access
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        # Resume the workflow through AI service
        bot_response = await self.ai_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=user_input,
            rejection_messages=rejection_messages,
        )

        # Extract response content
        bot_response_content = (
            bot_response.message.content
            if bot_response and bot_response.message
            else "Error: No response generated after resume"
        )

        # Create metadata for bot response
        bot_metadata = dict(bot_response.metadata) if bot_response else {}
        if bot_response and bot_response.tool_artifacts:
            bot_metadata.setdefault("tool_artifacts", bot_response.tool_artifacts)

        # Extract images from bot response metadata
        if bot_response and bot_response.metadata and "images" in bot_response.metadata:
            bot_metadata["images"] = bot_response.metadata["images"]

        # Save the bot response as a message in the database
        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=bot_response_content,
            message_metadata=bot_metadata,
        )

        # Create the message
        bot_message = self.repository.create(bot_response_entity)

        return MessageRead.model_validate(bot_message)
