from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Any, Dict, TYPE_CHECKING
from uuid import UUID
import redis


from app.repositories.message import MessageRepository
from app.repositories.tool_approval import ToolApprovalRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole, TaskStatus
from app.models.tool_approval import DecisionType
from app.factories.message_factory import MessageFactory
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params
from app.interfaces.message_service_interface import IMessageService
from app.services.ai_service import AIService
from app.schemas.task_plan import TaskPlanCreate, TaskPlanUpdate
from app.ai.schemas import (
    AgentResponse,
    InterruptDecision,
    InterruptResponse,
    InterruptDecisionType,
)
from app.utils.text_processing import sanitize_persona, fix_markdown_code_blocks
from app.core.config import settings
from app.core.exceptions import PauseReason
from app.ai.suggestion_generator import generate_follow_up_suggestions
from app.core.response_constants import (
    extract_response_content,
    build_bot_metadata,
    normalize_message_content,
    NO_RESPONSE_GENERATED,
    ERROR_NO_RESPONSE,
    ERROR_RESPONSE_AFTER_RESUME,
    UNKNOWN_ERROR,
)

if TYPE_CHECKING:
    from app.interfaces.task_plan_service_interface import ITaskPlanService


class MessageService(IMessageService):
    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_validation_utils: ConversationValidationUtils,
        message_validation_utils: MessageValidationUtils,
        ai_service: AIService,
        tool_approval_repository: Optional[ToolApprovalRepository] = None,
        task_plan_service: Optional["ITaskPlanService"] = None,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        self.ai_service = ai_service
        self.tool_approval_repository = tool_approval_repository
        self.task_plan_service = task_plan_service
        self.redis_client = self._init_redis_client()

    def _init_redis_client(self):
        redis_url = getattr(settings, "redis_url", "") or ""
        if not redis_url.strip():
            return None
        return redis.from_url(redis_url)

    def _get_conversation_context(
        self, conversation_id: UUID, user_id: Optional[UUID] = None
    ) -> Tuple[Optional[UUID], Optional[str]]:
        """Get user_id and persona from conversation."""
        conversation = (
            self.conversation_validation_utils.conversation_repository.get_by_id(
                conversation_id
            )
        )
        resolved_user_id = user_id or (conversation.owner_id if conversation else None)
        persona = conversation.persona_prompt if conversation else None
        return resolved_user_id, persona

    def _create_bot_response_message(
        self, conversation_id: UUID, content: str, metadata: Dict[str, Any]
    ) -> MessageRead:
        """Create and persist a bot response message."""
        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=content,
            message_metadata=metadata,
        )
        bot_message = self.repository.create(bot_response_entity)
        try:
            if getattr(self.ai_service, "workflow", None):
                self.ai_service.workflow.invalidate_history_cache(str(conversation_id))
        except Exception:
            pass
        return MessageRead.model_validate(bot_message)

    async def _generate_and_add_suggestions(
        self, user_query: str, response_content: str, metadata: Dict[str, Any]
    ) -> None:
        """Generate follow-up suggestions and add to metadata."""
        try:
            suggestions = await generate_follow_up_suggestions(
                user_query=user_query,
                response_content=response_content,
            )
            if suggestions:
                metadata["suggested_questions"] = suggestions
        except Exception:
            pass

    def _handle_task_completion(
        self,
        conversation_id: UUID,
        user_id: UUID,
        current_task: Any,
        plan_saved: bool,
        metadata: Dict[str, Any],
    ) -> None:
        """Mark current task as completed and add next task to metadata."""
        if not self.task_plan_service or not current_task or not user_id or plan_saved:
            return

        try:
            # Re-fetch task to verify status is still pending before marking complete
            fresh_task = self.task_plan_service.get_by_id(current_task.id, user_id)
            if fresh_task and fresh_task.status == TaskStatus.pending:
                self.task_plan_service.mark_task_completed(current_task.id, user_id)
                # Get next task for metadata
                next_task = self.task_plan_service.get_next_task(
                    conversation_id, user_id
                )
                if next_task:
                    metadata["next_task"] = {
                        "id": str(next_task.id),
                        "description": next_task.description,
                        "order": next_task.task_order,
                    }
        except Exception:
            pass

    def _is_first_user_message(self, conversation_id: UUID) -> bool:
        """Check if this is the first user message in the conversation."""
        try:
            # Check if conversation has a default title (needs generation)
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    conversation_id
                )
            )
            if not conversation:
                return False
            # Check for default/placeholder titles that need generation
            default_titles = {"New Conversation", "Untitled", ""}
            return conversation.title in default_titles or conversation.title is None
        except Exception:
            return False

    async def _generate_title_async(
        self,
        conversation_id: UUID,
        user_message: str,
    ) -> Optional[str]:
        """
        Generate and update conversation title asynchronously.

        Returns:
            The generated title if successful, None otherwise.
        """
        try:
            title = await self.ai_service.generate_conversation_title(user_message)
            if title:
                # Update conversation with generated title
                from app.schemas.conversation import ConversationUpdate

                self.conversation_validation_utils.conversation_repository.update(
                    conversation_id, ConversationUpdate(title=title)
                )
                return title
        except Exception:
            pass
        return None

    def _handle_redis_interrupt_storage(
        self,
        conversation_id: UUID,
        interrupt_id: Optional[str],
        interrupt_response: Dict[str, Any],
    ) -> None:
        """Store interrupt information in Redis with timeout."""
        if not self.redis_client or not interrupt_response or not interrupt_id:
            return

        key = f"interrupt:{conversation_id}:{interrupt_id}"
        timeout_seconds = settings.hitl_approval_timeout_minutes * 60
        try:
            self.redis_client.setex(
                key, timeout_seconds, datetime.now(timezone.utc).isoformat()
            )
            deadline = datetime.now(timezone.utc) + timedelta(
                minutes=settings.hitl_approval_timeout_minutes
            )
            if "metadata" not in interrupt_response:
                interrupt_response["metadata"] = {}
            interrupt_response["metadata"]["timeout_deadline"] = deadline.isoformat()
        except Exception:
            pass

    def _clear_redis_interrupt(
        self, conversation_id: UUID, interrupt_id: Optional[str]
    ) -> None:
        """Clear interrupt information from Redis."""
        if not self.redis_client or not interrupt_id:
            return

        key = f"interrupt:{conversation_id}:{interrupt_id}"
        try:
            self.redis_client.delete(key)
        except Exception:
            pass

    def _persist_interrupt_bot_message(
        self,
        conversation_id: UUID,
        interrupt_payload: Any,
        sanitized_persona: Optional[str] = None,
        pending_tool_calls: Optional[Any] = None,
        thread_id: Optional[str] = None,
        next_nodes: Optional[Any] = None,
    ) -> MessageRead:
        """
        Persist an assistant message that represents a paused workflow awaiting HITL approval.

        This makes pending approvals recoverable via normal message history APIs (DB-backed),
        rather than only via the live SSE stream.
        """
        if isinstance(interrupt_payload, InterruptResponse):
            interrupt_dict = interrupt_payload.model_dump(mode="json")
        elif isinstance(interrupt_payload, dict):
            interrupt_dict = interrupt_payload
        else:
            interrupt_dict = {"raw": str(interrupt_payload)}

        metadata: Dict[str, Any] = {
            "interrupt": interrupt_dict,
            "paused": True,
            "pause_reason": "tool_approval_required",
        }
        if sanitized_persona:
            metadata["persona_used"] = sanitized_persona
        if thread_id:
            metadata["thread_id"] = thread_id
        if next_nodes is not None:
            metadata["next"] = next_nodes
        if pending_tool_calls is not None:
            metadata["pending_tool_calls"] = pending_tool_calls

        return self._create_bot_response_message(
            conversation_id=conversation_id,
            content="Tool execution requires approval",
            metadata=metadata,
        )

    async def create_message(
        self, message_create_data: MessageCreate, user_id: UUID
    ) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, message_create_data.conversation_id
        )

        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )

        created_message = self.repository.create(message_entity)

        if message_create_data.role == MessageRole.user:
            # Get the user_id and persona from the conversation
            user_id, persona = self._get_conversation_context(
                message_create_data.conversation_id, user_id
            )
            sanitized_persona = sanitize_persona(persona)

            # Get conversation for planning mode check
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )

            # Use shared helper to prepare planning context
            planning_ctx = await self._prepare_planning_context(
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                message_content=message_create_data.content,
                planning_mode_enabled=planning_mode_enabled,
            )
            planning_mode_enabled = planning_ctx["planning_mode_enabled"]
            has_existing_plan = planning_ctx["has_existing_plan"]
            current_task = planning_ctx["current_task"]
            current_task_context = planning_ctx["current_task_context"]
            existing_tasks_dict = planning_ctx["existing_tasks_dict"]

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            auto_execute_plan = (
                planning_mode_enabled
                and has_existing_plan
                and current_task_context is not None
            )

            (
                bot_response_content,
                bot_metadata,
                bot_response,
                execution_count,
                interrupt_payload,
            ) = await self._run_plan_execution_loop(
                message_content=message_create_data.content,
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                sanitized_persona=sanitized_persona,
                planning_mode_enabled=planning_mode_enabled,
                has_existing_plan=has_existing_plan,
                current_task=current_task,
                current_task_context=current_task_context,
                existing_tasks_dict=existing_tasks_dict,
                attachments=attachments,
                auto_execute_plan=auto_execute_plan,
            )

            if interrupt_payload:
                user_message_read = MessageRead.model_validate(created_message)
                if isinstance(interrupt_payload, dict):
                    interrupt_payload = InterruptResponse.model_validate(
                        interrupt_payload
                    )
                user_message_read.interrupt = interrupt_payload

                # Persist an assistant "approval required" message so clients can
                # recover pending approvals from message history (not just SSE).
                try:
                    self._persist_interrupt_bot_message(
                        conversation_id=message_create_data.conversation_id,
                        interrupt_payload=interrupt_payload,
                        sanitized_persona=sanitized_persona,
                        thread_id=getattr(interrupt_payload, "thread_id", None),
                        pending_tool_calls=(
                            [
                                r.model_dump(mode="json")
                                for r in getattr(
                                    interrupt_payload, "action_requests", []
                                )
                            ]
                            if isinstance(interrupt_payload, InterruptResponse)
                            else None
                        ),
                    )
                except Exception:
                    pass

                return user_message_read

            self._create_bot_response_message(
                conversation_id=message_create_data.conversation_id,
                content=bot_response_content,
                metadata=bot_metadata,
            )

        return MessageRead.model_validate(created_message)

    async def create_message_stream(
        self, message_create_data: MessageCreate, user_id: UUID
    ):
        """
        Create a message and stream the bot response.
        Yields chunks as they arrive from the AI service.
        """
        self.conversation_validation_utils.validate_conversation_access(
            user_id, message_create_data.conversation_id
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

        # Start async title generation only if this is a user message and the first one
        title_task = None
        if message_create_data.role == MessageRole.user:
            needs_title = self._is_first_user_message(
                message_create_data.conversation_id
            )
            if needs_title:
                title_task = asyncio.create_task(
                    self._generate_title_async(
                        message_create_data.conversation_id, message_create_data.content
                    )
                )

        def _cancel_title_task():
            """Cancel title task if running to prevent resource leaks."""
            if title_task and not title_task.done():
                title_task.cancel()

        if message_create_data.role == MessageRole.user:
            # Get the user_id and persona from the conversation
            user_id, persona = self._get_conversation_context(
                message_create_data.conversation_id, user_id
            )
            sanitized_persona = sanitize_persona(persona)

            # Get conversation for planning mode check
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )

            # Use shared helper to prepare planning context
            planning_ctx = await self._prepare_planning_context(
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                message_content=message_create_data.content,
                planning_mode_enabled=planning_mode_enabled,
            )
            planning_mode_enabled = planning_ctx["planning_mode_enabled"]
            has_existing_plan = planning_ctx["has_existing_plan"]
            current_task = planning_ctx["current_task"]
            current_task_context = planning_ctx["current_task_context"]
            existing_tasks_dict = planning_ctx["existing_tasks_dict"]

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            # Stream bot response generation
            bot_response_content = ERROR_NO_RESPONSE
            bot_response = None

            try:
                async for event in self.ai_service.generate_bot_response_stream(
                    user_message=message_create_data.content,
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    attachments=attachments,
                    current_task=current_task_context,
                    planning_mode_enabled=planning_mode_enabled,
                    has_existing_plan=has_existing_plan,
                    existing_tasks=existing_tasks_dict,
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
                            "tool_call_id": event.get("tool_call_id"),
                            "args": event.get("args"),
                            "result": event.get("result"),
                        }

                    elif event_type == "interrupt":
                        # Yield interrupt event - workflow paused for human approval
                        interrupt_response = event.get("interrupt")
                        interrupt_id = (
                            interrupt_response.get("interrupt_id")
                            if interrupt_response
                            else None
                        )
                        self._handle_redis_interrupt_storage(
                            message_create_data.conversation_id,
                            interrupt_id,
                            interrupt_response,
                        )

                        yield {
                            "type": "interrupt",
                            "thread_id": event.get("thread_id")
                            or str(message_create_data.conversation_id),
                            "next": event.get("next"),
                            "pending_tool_calls": event.get("pending_tool_calls"),
                            "interrupt": interrupt_response,
                            "message": self._persist_interrupt_bot_message(
                                conversation_id=message_create_data.conversation_id,
                                interrupt_payload=interrupt_response,
                                sanitized_persona=sanitized_persona,
                                pending_tool_calls=event.get("pending_tool_calls"),
                                thread_id=event.get("thread_id")
                                or str(message_create_data.conversation_id),
                                next_nodes=event.get("next"),
                            ).model_dump(mode="json"),
                        }
                        # Workflow is paused - don't create a bot message yet
                        # The resume endpoint will handle that
                        _cancel_title_task()
                        return

                    elif event_type == "complete":
                        # Store final response
                        bot_response = event.get("response")
                        bot_response_content = extract_response_content(
                            bot_response, NO_RESPONSE_GENERATED
                        )
                        break

                    elif event_type == "error":
                        # Handle error
                        bot_response = event.get("response")
                        error_msg = event.get("error", UNKNOWN_ERROR)
                        bot_response_content = extract_response_content(
                            bot_response, f"Error: {error_msg}"
                        )
                        break

                # Ensure content is valid (not empty)
                if not bot_response_content or not bot_response_content.strip():
                    bot_response_content = NO_RESPONSE_GENERATED

                # Fix markdown code blocks that may be missing newlines
                bot_response_content = fix_markdown_code_blocks(bot_response_content)

                # Create metadata for bot response
                bot_metadata = build_bot_metadata(bot_response, sanitized_persona)

                plan_saved, has_existing_plan = self._sync_plan_from_response_metadata(
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    bot_response=bot_response,
                    has_existing_plan=has_existing_plan,
                )
                if plan_saved:
                    bot_metadata["plan_saved"] = True

                # Sync todos from graph state if present (for planning agent)
                if (
                    bot_response
                    and bot_response.metadata
                    and bot_response.metadata.get("todos")
                ):
                    self._sync_todos_to_database(
                        conversation_id=message_create_data.conversation_id,
                        user_id=user_id,
                        todos=bot_response.metadata["todos"],
                    )
                    bot_metadata["todos_synced"] = True

                # Mark current task as completed if planning mode is active
                # Skip if plan was just replaced (old task IDs are invalid)
                if planning_mode_enabled:
                    self._handle_task_completion(
                        message_create_data.conversation_id,
                        user_id,
                        current_task,
                        plan_saved,
                        bot_metadata,
                    )

                # Generate follow-up question suggestions
                await self._generate_and_add_suggestions(
                    message_create_data.content, bot_response_content, bot_metadata
                )

                # Create and persist bot response message
                bot_message = self._create_bot_response_message(
                    conversation_id=message_create_data.conversation_id,
                    content=bot_response_content,
                    metadata=bot_metadata,
                )

                # Yield final completion event with full message
                yield {
                    "type": "complete",
                    "message": bot_message.model_dump(mode="json"),
                }

                # Yield title update event if title was generated in parallel
                if title_task:
                    generated_title = await title_task
                    if generated_title:
                        yield {
                            "type": "title_updated",
                            "title": generated_title,
                            "conversation_id": str(message_create_data.conversation_id),
                        }

            except Exception as exc:
                error_content = f"Error generating response: {str(exc)}"
                error_metadata = {"error": str(exc)}

                error_message = self._create_bot_response_message(
                    conversation_id=message_create_data.conversation_id,
                    content=error_content,
                    metadata=error_metadata,
                )

                yield {
                    "type": "error",
                    "error": str(exc),
                    "message": error_message.model_dump(mode="json"),
                }
                _cancel_title_task()

    async def resume_message_creation(
        self,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
    ) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        if self.redis_client and interrupt_id:
            key = f"interrupt:{conversation_id}:{interrupt_id}"
            try:
                stored_timestamp = self.redis_client.get(key)
                if stored_timestamp:
                    stored_time = datetime.fromisoformat(
                        stored_timestamp.decode("utf-8")
                    )
                    elapsed_minutes = (
                        datetime.now(timezone.utc) - stored_time
                    ).total_seconds() / 60
                    if elapsed_minutes > settings.hitl_approval_timeout_minutes:
                        # Clean up the expired key
                        self.redis_client.delete(key)
                        raise TimeoutError(
                            f"This approval request has expired ({elapsed_minutes:.0f} minutes elapsed, "
                            f"limit is {settings.hitl_approval_timeout_minutes} minutes). "
                            "Please send a new message to try again."
                        )
            except TimeoutError:
                raise
            except Exception:
                pass

        # Get the conversation to retrieve user_id and persona
        user_id, persona = self._get_conversation_context(conversation_id, user_id)
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
            decisions=decisions,
            interrupt_id=interrupt_id,
        )

        self._clear_redis_interrupt(conversation_id, interrupt_id)
        if (
            bot_response
            and bot_response.metadata
            and "interrupt" in bot_response.metadata
        ):
            interrupt_payload = bot_response.metadata["interrupt"]
            if isinstance(interrupt_payload, dict):
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

            persisted = self._persist_interrupt_bot_message(
                conversation_id=conversation_id,
                interrupt_payload=interrupt_payload,
                sanitized_persona=sanitized_persona,
                thread_id=(
                    interrupt_payload.thread_id
                    if isinstance(interrupt_payload, InterruptResponse)
                    else None
                ),
            )
            return persisted

        bot_response_content = extract_response_content(
            bot_response, ERROR_RESPONSE_AFTER_RESUME
        )

        # Create metadata for bot response
        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)

        # Create and persist bot response message
        return self._create_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=bot_metadata,
        )

    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        if hasattr(message_entity, "content"):
            message_entity.content = normalize_message_content(message_entity.content)
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
            if hasattr(msg, "content"):
                msg.content = normalize_message_content(msg.content)
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
            if hasattr(msg, "content"):
                msg.content = normalize_message_content(msg.content)
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
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        bot_response = await self.ai_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=user_input,
            rejection_messages=rejection_messages,
        )

        bot_response_content = extract_response_content(
            bot_response, NO_RESPONSE_GENERATED
        )

        bot_metadata = build_bot_metadata(bot_response)

        return self._create_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=bot_metadata,
        )

    def _sync_plan_from_response_metadata(
        self,
        conversation_id: UUID,
        user_id: Optional[UUID],
        bot_response: Optional[AgentResponse],
        has_existing_plan: bool,
    ) -> Tuple[bool, bool]:
        if (
            not self.task_plan_service
            or not bot_response
            or not bot_response.metadata
            or not user_id
        ):
            return False, has_existing_plan

        plan_payload = bot_response.metadata.get("plan")
        if not plan_payload:
            return False, has_existing_plan

        plan_modified = bool(bot_response.metadata.get("plan_modified"))
        if has_existing_plan and not plan_modified:
            return False, has_existing_plan

        try:
            self.task_plan_service.sync_plan_from_agent(
                conversation_id=conversation_id,
                plan_payload=plan_payload,
                user_id=user_id,
                replace_existing=True,
            )
            return True, True
        except Exception:
            return False, has_existing_plan

    @staticmethod
    def _build_task_context_dict(task: Optional[Any]) -> Optional[Dict[str, Any]]:
        if not task:
            return None
        return {
            "id": str(task.id),
            "description": task.description,
            "order": getattr(task, "task_order", getattr(task, "order", 0)),
            "status": (
                task.status.value if hasattr(task.status, "value") else str(task.status)
            ),
        }

    async def _prepare_planning_context(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message_content: str,
        planning_mode_enabled: bool,
    ) -> Dict[str, Any]:
        """
        Prepare task planning context for message creation.

        Returns a dict with keys:
        - planning_mode_enabled: bool
        - has_existing_plan: bool
        - current_task: Optional[task entity]
        - current_task_context: Optional[dict]
        - existing_tasks_dict: Optional[List[dict]]
        """
        result = {
            "planning_mode_enabled": planning_mode_enabled,
            "has_existing_plan": False,
            "current_task": None,
            "current_task_context": None,
            "existing_tasks_dict": None,
        }

        if not self.task_plan_service or not user_id:
            return result

        try:
            existing_tasks = self.task_plan_service.get_conversation_tasks(
                conversation_id, user_id, include_completed=True
            )
            result["has_existing_plan"] = len(existing_tasks) > 0

            # Create plan if planning mode enabled but no plan exists
            if planning_mode_enabled and not result["has_existing_plan"]:
                created_tasks = await self.task_plan_service.create_task_plan(
                    conversation_id, message_content, user_id
                )
                result["has_existing_plan"] = len(created_tasks) > 0
                if result["has_existing_plan"]:
                    existing_tasks = self.task_plan_service.get_conversation_tasks(
                        conversation_id, user_id, include_completed=True
                    )

            if result["has_existing_plan"]:
                result["planning_mode_enabled"] = True

            # Get current task for execution context
            result["current_task"] = self.task_plan_service.get_next_task(
                conversation_id, user_id
            )
            result["current_task_context"] = self._build_task_context_dict(
                result["current_task"]
            )

            # Convert existing tasks to dict for planning agent
            if existing_tasks:
                result["existing_tasks_dict"] = [
                    {
                        "id": str(task.id),
                        "description": task.description,
                        "status": (
                            task.status.value
                            if hasattr(task.status, "value")
                            else str(task.status)
                        ),
                        "task_order": task.task_order,
                    }
                    for task in existing_tasks
                ]
        except Exception:
            pass

        return result

    async def _run_plan_execution_loop(
        self,
        *,
        message_content: str,
        conversation_id: UUID,
        user_id: Optional[UUID],
        sanitized_persona: Optional[str],
        planning_mode_enabled: bool,
        has_existing_plan: bool,
        current_task: Optional[Any],
        current_task_context: Optional[Dict[str, Any]],
        existing_tasks_dict: Optional[List[Dict[str, Any]]],
        attachments: Optional[list],
        auto_execute_plan: bool,
    ) -> Tuple[
        Optional[str],
        Dict[str, Any],
        Optional[AgentResponse],
        int,
        Optional[Dict[str, Any]],
    ]:
        """
        Execute planning workflow with graph-driven ReAct loop.

        The graph now handles iteration internally via the planning_tools node.
        This function makes a single call and syncs todo state afterward.
        """
        bot_response: Optional[AgentResponse] = None
        bot_metadata: Dict[str, Any] = {}
        has_plan = has_existing_plan

        # Single call to AI service - graph handles ReAct loop internally
        bot_response = await self.ai_service.generate_bot_response(
            user_message=message_content,
            conversation_id=conversation_id,
            user_id=user_id,
            attachments=attachments,
            current_task=current_task_context,
            all_tasks=existing_tasks_dict,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_plan,
            existing_tasks=existing_tasks_dict,
        )

        # Handle interrupts (HITL)
        if (
            bot_response
            and bot_response.metadata
            and "interrupt" in bot_response.metadata
        ):
            return (
                None,
                {},
                bot_response,
                0,  # execution_count not relevant with graph-driven execution
                bot_response.metadata["interrupt"],
            )

        bot_response_content = extract_response_content(bot_response)

        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)

        # Sync plan from response metadata (legacy and new format)
        plan_saved, has_plan = self._sync_plan_from_response_metadata(
            conversation_id=conversation_id,
            user_id=user_id,
            bot_response=bot_response,
            has_existing_plan=has_plan,
        )
        if plan_saved:
            bot_metadata["plan_saved"] = True

        # Sync todos from graph state if present
        if bot_response and bot_response.metadata.get("todos"):
            self._sync_todos_to_database(
                conversation_id=conversation_id,
                user_id=user_id,
                todos=bot_response.metadata["todos"],
            )
            bot_metadata["todos_synced"] = True

        # Check if planning budget was reached
        if bot_response and bot_response.metadata.get("planning_budget_reached"):
            bot_metadata["execution_paused"] = True
            bot_metadata["execution_pause_reason"] = PauseReason.MAX_TASKS_REACHED.value
            bot_metadata["execution_pause_message"] = (
                "Completed a planning iteration. Send a message to continue."
            )

        # Refresh task data for next_task info
        if planning_mode_enabled and self.task_plan_service and user_id:
            try:
                next_task = self.task_plan_service.get_next_task(
                    conversation_id, user_id
                )
                if next_task:
                    bot_metadata["next_task"] = {
                        "id": str(next_task.id),
                        "description": next_task.description,
                        "order": next_task.task_order,
                    }
            except Exception:
                pass

        # Fix markdown code blocks
        bot_response_content = fix_markdown_code_blocks(bot_response_content)

        return bot_response_content, bot_metadata, bot_response, 1, None

    def _sync_todos_to_database(
        self,
        conversation_id: UUID,
        user_id: Optional[UUID],
        todos: List[Dict[str, Any]],
    ) -> None:
        if not self.task_plan_service or not todos or not user_id:
            return

        existing_tasks = self.task_plan_service.get_conversation_tasks(
            conversation_id, user_id, include_completed=True
        )

        # No existing tasks - create new plan from todos
        if not existing_tasks:
            plan_payload = {
                "tasks": [
                    {
                        "description": todo.get("description", ""),
                    }
                    for todo in todos
                ]
            }
            self.task_plan_service.sync_plan_from_agent(
                conversation_id=conversation_id,
                user_id=user_id,
                plan_payload=plan_payload,
            )
            return

        # Build lookup maps for existing tasks
        task_by_id = {str(task.id): task for task in existing_tasks}
        existing_task_ids = set(task_by_id.keys())

        # Track which todos are new vs updates
        todo_ids_in_response = set()

        for i, todo in enumerate(todos):
            todo_id = str(todo.get("id", ""))
            todo_status = todo.get("status", "pending")
            todo_description = todo.get("description", "")

            todo_ids_in_response.add(todo_id)

            # Try to find matching existing task
            task = task_by_id.get(todo_id)

            if task:
                # Update existing task
                task_status_str = (
                    task.status.value
                    if hasattr(task.status, "value")
                    else str(task.status)
                )
                needs_update = False
                update_data = {}

                # Check if description changed
                if todo_description and todo_description != task.description:
                    update_data["description"] = todo_description
                    needs_update = True

                # Check if status changed
                if (
                    todo_status in ("completed", "COMPLETED")
                    and task_status_str != "completed"
                ):
                    self.task_plan_service.mark_task_completed(task.id, user_id)
                elif (
                    todo_status in ("in_progress", "IN_PROGRESS")
                    and task_status_str == "pending"
                ):
                    if hasattr(self.task_plan_service, "mark_task_in_progress"):
                        self.task_plan_service.mark_task_in_progress(task.id, user_id)

                # Apply description update if needed
                if needs_update and update_data:
                    self.task_plan_service.update_task(
                        task.id, user_id, TaskPlanUpdate(**update_data)
                    )
            else:
                # This is a new task - add it
                self.task_plan_service.task_plan_repository.create(
                    TaskPlanCreate(
                        conversation_id=conversation_id,
                        description=todo_description,
                        task_order=i,
                    )
                )

        # Handle removed tasks - delete tasks that are in DB but not in todos
        if todo_ids_in_response and all(
            todo_id not in ("", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
            for todo_id in todo_ids_in_response
        ):
            for task_id in existing_task_ids:
                if task_id not in todo_ids_in_response:
                    self.task_plan_service.delete_task(UUID(task_id), user_id)
