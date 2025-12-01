from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID, uuid4
import redis

from app.repositories.message import MessageRepository
from app.repositories.tool_approval import ToolApprovalRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole
from app.models.tool_approval import DecisionType
from app.factories.message_factory import MessageFactory
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params
from app.interfaces.message_service_interface import IMessageService
from app.services.ai_service import AIService
from app.ai.schemas import InterruptDecision, InterruptResponse, InterruptDecisionType
from app.utils.text_processing import sanitize_persona
from app.core.config import settings
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
        tool_approval_repository: Optional[ToolApprovalRepository] = None,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        self.ai_service = ai_service
        self.tool_approval_repository = tool_approval_repository

        # Initialize Redis connection for timeout tracking
        self.redis_client = self._init_redis_client()

    def _init_redis_client(self):
        """Create a Redis client if configuration is provided."""
        redis_url = getattr(settings, "redis_url", "") or ""
        if not redis_url.strip():
            logger.debug("Redis URL not configured; HITL timeout tracking disabled.")
            return None

        try:
            return redis.from_url(redis_url)
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis at {redis_url}: {e}. Timeout tracking disabled."
            )
            return None

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

            # Check if response contains interrupt information
            if (
                bot_response
                and bot_response.metadata
                and "interrupt" in bot_response.metadata
            ):
                user_message_read = MessageRead.model_validate(created_message)
                interrupt_payload = bot_response.metadata["interrupt"]
                if isinstance(interrupt_payload, dict):
                    interrupt_payload = InterruptResponse.model_validate(
                        interrupt_payload
                    )
                user_message_read.interrupt = interrupt_payload
                return user_message_read

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

                    if event_type == "agent_selected":
                        # Yield agent selection notification to client
                        yield {"type": "agent_selected", "agent": event.get("agent")}

                    elif event_type == "token":
                        # Yield token to client
                        yield {"type": "token", "content": event.get("content", "")}

                    elif event_type == "thinking":
                        # Yield thinking/reasoning content to client
                        yield {"type": "thinking", "content": event.get("content", "")}

                    elif event_type == "tool":
                        # Yield tool execution event
                        yield {
                            "type": "tool",
                            "name": event.get("name"),
                            "status": event.get("status"),
                        }

                    elif event_type == "interrupt":
                        # Yield interrupt event - workflow paused for human approval
                        interrupt_response = event.get("interrupt")

                        if self.redis_client and interrupt_response:
                            interrupt_id = interrupt_response.get("interrupt_id")
                            if interrupt_id:
                                key = f"interrupt:{message_create_data.conversation_id}:{interrupt_id}"
                                timeout_seconds = (
                                    settings.hitl_approval_timeout_minutes * 60
                                )
                                try:
                                    self.redis_client.setex(
                                        key,
                                        timeout_seconds,
                                        datetime.utcnow().isoformat(),
                                    )
                                    deadline = datetime.utcnow() + timedelta(
                                        minutes=settings.hitl_approval_timeout_minutes
                                    )
                                    if "metadata" not in interrupt_response:
                                        interrupt_response["metadata"] = {}
                                    interrupt_response["metadata"][
                                        "timeout_deadline"
                                    ] = deadline.isoformat()
                                except Exception:
                                    pass

                        yield {
                            "type": "interrupt",
                            "thread_id": str(message_create_data.conversation_id),
                            "next": event.get("next"),
                            "pending_tool_calls": event.get("pending_tool_calls"),
                            "interrupt": interrupt_response,
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

    async def resume_message_creation(
        self,
        thread_id: str,
        conversation_id: UUID,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
    ) -> MessageRead:
        """
        Resume message creation after handling interrupts.

        Args:
            thread_id: Thread ID from the interrupt response
            conversation_id: Conversation ID
            decisions: List of approval/rejection/edit decisions
            interrupt_id: LangGraph interrupt identifier for targeted resume

        Returns:
            MessageRead of the created bot response message
        """
        self.conversation_validation_utils.validate_conversation_exists(conversation_id)

        if self.redis_client and interrupt_id:
            key = f"interrupt:{conversation_id}:{interrupt_id}"
            try:
                stored_timestamp = self.redis_client.get(key)
                if stored_timestamp:
                    stored_time = datetime.fromisoformat(
                        stored_timestamp.decode("utf-8")
                    )
                    elapsed_minutes = (
                        datetime.utcnow() - stored_time
                    ).total_seconds() / 60
                    if elapsed_minutes > settings.hitl_approval_timeout_minutes:
                        raise TimeoutError(
                            f"Interrupt approval timeout exceeded: {elapsed_minutes:.1f} minutes elapsed, "
                            f"limit is {settings.hitl_approval_timeout_minutes} minutes"
                        )
            except TimeoutError:
                raise
            except Exception:
                pass

        # Get the conversation to retrieve user_id and persona
        conversation = (
            self.conversation_validation_utils.conversation_repository.get_by_id(
                conversation_id
            )
        )
        user_id = conversation.owner_id if conversation else None
        persona = conversation.persona_prompt if conversation else None
        sanitized_persona = sanitize_persona(persona)

        if self.tool_approval_repository and user_id:
            try:
                for decision in decisions:
                    # Map decision type to enum
                    decision_type_map = {
                        InterruptDecisionType.ACCEPT: DecisionType.ACCEPT,
                        InterruptDecisionType.APPROVE: DecisionType.ACCEPT,
                        InterruptDecisionType.EDIT: DecisionType.EDIT,
                        InterruptDecisionType.REJECT: DecisionType.REJECT,
                        InterruptDecisionType.RESPOND: DecisionType.REJECT,
                    }

                    approval_data = {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "interrupt_id": interrupt_id or "unknown",
                        "tool_name": decision.action or "unknown",
                        "tool_call_id": decision.task_id or "unknown",
                        "original_args": decision.original_args or {},
                        "modified_args": (
                            decision.modified_args
                            if decision.decision in [InterruptDecisionType.EDIT]
                            else None
                        ),
                        "decision": decision_type_map.get(
                            decision.decision, DecisionType.REJECT
                        ),
                    }
                    self.tool_approval_repository.create(approval_data)
            except Exception:
                pass

        bot_response = await self.ai_service.resume_interrupted_execution(
            thread_id=thread_id,
            conversation_id=conversation_id,
            decisions=decisions,
            interrupt_id=interrupt_id,
        )

        if self.redis_client and interrupt_id:
            key = f"interrupt:{conversation_id}:{interrupt_id}"
            try:
                self.redis_client.delete(key)
            except Exception:
                pass

        if (
            bot_response
            and bot_response.metadata
            and "interrupt" in bot_response.metadata
        ):
            latest_message = self.repository.get_latest_by_conversation(conversation_id)
            if latest_message:
                message_read = MessageRead.model_validate(latest_message)
            else:
                message_read = MessageRead(
                    id=uuid4(),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    deleted_at=None,
                    conversation_id=conversation_id,
                    sender=MessageRole.assistant.value,
                    content="Tool execution requires approval",
                    message_metadata={},
                )
            interrupt_payload = bot_response.metadata["interrupt"]
            if isinstance(interrupt_payload, dict):
                # Add interrupt counter for better UX
                interrupt_count = (
                    interrupt_payload.get("metadata", {}).get("interrupt_count", 0) + 1
                )
                if "metadata" not in interrupt_payload:
                    interrupt_payload["metadata"] = {}
                interrupt_payload["metadata"]["interrupt_count"] = interrupt_count

                if interrupt_count > 1:
                    interrupt_payload["metadata"][
                        "message"
                    ] = f"The assistant needs approval for additional tools (request {interrupt_count})"
                else:
                    interrupt_payload["metadata"][
                        "message"
                    ] = "The assistant wants to use tools that require approval"

                MAX_INTERRUPT_DEPTH = 5
                if interrupt_count > MAX_INTERRUPT_DEPTH:
                    pass  # Could auto-reject or provide fallback

                interrupt_payload = InterruptResponse.model_validate(interrupt_payload)
            message_read.interrupt = interrupt_payload
            return message_read

        bot_response_content = (
            bot_response.message.content
            if bot_response and bot_response.message
            else "Error: No response after resuming"
        )

        # Create metadata for bot response
        bot_metadata = dict(bot_response.metadata) if bot_response else {}
        if sanitized_persona:
            bot_metadata.setdefault("persona_used", sanitized_persona)

        if bot_response and bot_response.tool_artifacts:
            bot_metadata.setdefault("tool_artifacts", bot_response.tool_artifacts)

        # Extract images from bot response metadata
        if bot_response and bot_response.metadata and "images" in bot_response.metadata:
            bot_metadata["images"] = bot_response.metadata["images"]

        # Create and persist bot response message
        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=bot_response_content,
            message_metadata=bot_metadata,
        )
        bot_message = self.repository.create(bot_response_entity)

        return MessageRead.model_validate(bot_message)

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
