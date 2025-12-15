from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any, Dict, TYPE_CHECKING
from uuid import UUID, uuid4
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
from app.ai.schemas import (
    AgentResponse,
    InterruptDecision,
    InterruptResponse,
    InterruptDecisionType,
)
from app.utils.text_processing import sanitize_persona, fix_markdown_code_blocks
from app.core.config import settings
from app.core.exceptions import PauseReason

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

        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content="Tool execution requires approval",
            message_metadata=metadata,
        )
        bot_message = self.repository.create(bot_response_entity)
        return MessageRead.model_validate(bot_message)

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

            # Check if planning mode is enabled and get current task
            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )
            current_task = None
            current_task_context = None
            has_existing_plan = False
            existing_tasks_dict = None
            if self.task_plan_service and user_id:
                try:
                    existing_tasks = self.task_plan_service.get_conversation_tasks(
                        message_create_data.conversation_id,
                        user_id,
                        include_completed=True,
                    )
                    has_existing_plan = len(existing_tasks) > 0

                    if planning_mode_enabled and not has_existing_plan:
                        created_tasks = await self.task_plan_service.create_task_plan(
                            message_create_data.conversation_id,
                            message_create_data.content,
                            user_id,
                        )
                        has_existing_plan = len(created_tasks) > 0
                        if has_existing_plan:
                            existing_tasks = (
                                self.task_plan_service.get_conversation_tasks(
                                    message_create_data.conversation_id,
                                    user_id,
                                    include_completed=True,
                                )
                            )

                    if has_existing_plan:
                        planning_mode_enabled = True

                    # Get current task for execution context
                    current_task = self.task_plan_service.get_next_task(
                        message_create_data.conversation_id, user_id
                    )
                    current_task_context = self._build_task_context_dict(current_task)

                    # Convert existing tasks to dict for planning agent
                    if existing_tasks:
                        existing_tasks_dict = [
                            {
                                "id": str(task.id),
                                "description": task.description,
                                "status": (
                                    task.status.value
                                    if hasattr(task.status, "value")
                                    else str(task.status)
                                ),
                                "task_order": task.task_order,
                                "dependencies": [
                                    str(d) for d in (task.dependencies or [])
                                ],
                            }
                            for task in existing_tasks
                        ]
                except Exception:
                    pass

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
                        pending_tool_calls=[
                            r.model_dump(mode="json")
                            for r in getattr(interrupt_payload, "action_requests", [])
                        ]
                        if isinstance(interrupt_payload, InterruptResponse)
                        else None,
                    )
                except Exception:
                    pass

                return user_message_read

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

            # Check if planning mode is enabled and get current task
            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )
            current_task = None
            current_task_context = None
            has_existing_plan = False
            existing_tasks_dict = None
            if self.task_plan_service and user_id:
                try:
                    existing_tasks = self.task_plan_service.get_conversation_tasks(
                        message_create_data.conversation_id,
                        user_id,
                        include_completed=True,
                    )
                    has_existing_plan = len(existing_tasks) > 0

                    if planning_mode_enabled and not has_existing_plan:
                        created_tasks = await self.task_plan_service.create_task_plan(
                            message_create_data.conversation_id,
                            message_create_data.content,
                            user_id,
                        )
                        has_existing_plan = len(created_tasks) > 0
                        if has_existing_plan:
                            existing_tasks = (
                                self.task_plan_service.get_conversation_tasks(
                                    message_create_data.conversation_id,
                                    user_id,
                                    include_completed=True,
                                )
                            )

                    if has_existing_plan:
                        planning_mode_enabled = True

                    # Get current task for execution context
                    current_task = self.task_plan_service.get_next_task(
                        message_create_data.conversation_id, user_id
                    )
                    current_task_context = self._build_task_context_dict(current_task)

                    # Convert existing tasks to dict for planning agent
                    if existing_tasks:
                        existing_tasks_dict = [
                            {
                                "id": str(task.id),
                                "description": task.description,
                                "status": (
                                    task.status.value
                                    if hasattr(task.status, "value")
                                    else str(task.status)
                                ),
                                "task_order": task.task_order,
                                "dependencies": [
                                    str(d) for d in (task.dependencies or [])
                                ],
                            }
                            for task in existing_tasks
                        ]

                except Exception:
                    pass

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

            # Stream bot response generation
            bot_response_content = "Error: No response generated"
            bot_response = None

            if auto_execute_plan:
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
                    auto_execute_plan=True,
                )

                if interrupt_payload:
                    if isinstance(interrupt_payload, dict):
                        interrupt_payload = InterruptResponse.model_validate(
                            interrupt_payload
                        )

                    persisted = self._persist_interrupt_bot_message(
                        conversation_id=message_create_data.conversation_id,
                        interrupt_payload=interrupt_payload,
                        sanitized_persona=sanitized_persona,
                        thread_id=interrupt_payload.thread_id,
                        next_nodes=None,
                    )
                    yield {
                        "type": "interrupt",
                        "thread_id": interrupt_payload.thread_id,
                        "next": None,
                        "pending_tool_calls": None,
                        "interrupt": interrupt_payload,
                        "message": persisted.model_dump(mode="json"),
                    }
                    return

                bot_response_entity = MessageFactory.create_bot_response(
                    conversation_id=message_create_data.conversation_id,
                    content=bot_response_content,
                    message_metadata=bot_metadata,
                )
                bot_message = self.repository.create(bot_response_entity)

                # Emit simplified events (no token streaming) for auto-execution
                agent_name = bot_response.agent_id if bot_response else "chat_agent"
                yield {"type": "agent_selected", "agent": agent_name}
                yield {"type": "token", "content": bot_response_content}
                yield {
                    "type": "complete",
                    "message": MessageRead.model_validate(bot_message).model_dump(
                        mode="json"
                    ),
                }
                return

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

                # Fix markdown code blocks that may be missing newlines
                bot_response_content = fix_markdown_code_blocks(bot_response_content)

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

                plan_saved, has_existing_plan = self._sync_plan_from_response_metadata(
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    bot_response=bot_response,
                    has_existing_plan=has_existing_plan,
                )
                if plan_saved:
                    bot_metadata["plan_saved"] = True

                # Mark current task as completed if planning mode is active
                # Skip if plan was just replaced (old task IDs are invalid)
                if (
                    planning_mode_enabled
                    and current_task
                    and self.task_plan_service
                    and user_id
                    and not plan_saved
                ):
                    try:
                        # Re-fetch task to verify status is still pending before marking complete
                        fresh_task = self.task_plan_service.get_by_id(
                            current_task.id, user_id
                        )
                        if fresh_task and fresh_task.status == TaskStatus.pending:
                            self.task_plan_service.mark_task_completed(
                                current_task.id, user_id
                            )
                            # Get next task for metadata
                            next_task = self.task_plan_service.get_next_task(
                                message_create_data.conversation_id, user_id
                            )
                            if next_task:
                                bot_metadata["next_task"] = {
                                    "id": str(next_task.id),
                                    "description": next_task.description,
                                    "order": next_task.task_order,
                                }
                    except Exception:
                        pass

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
                pass
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
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        bot_response = await self.ai_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=user_input,
            rejection_messages=rejection_messages,
        )

        bot_response_content = (
            bot_response.message.content
            if bot_response and bot_response.message
            else "No response generated after resume"
        )

        bot_metadata = dict(bot_response.metadata) if bot_response else {}
        if bot_response and bot_response.tool_artifacts:
            bot_metadata.setdefault("tool_artifacts", bot_response.tool_artifacts)

        if bot_response and bot_response.metadata and "images" in bot_response.metadata:
            bot_metadata["images"] = bot_response.metadata["images"]

        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=bot_response_content,
            message_metadata=bot_metadata,
        )

        bot_message = self.repository.create(bot_response_entity)
        return MessageRead.model_validate(bot_message)

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
            "dependencies": [str(dep) for dep in (task.dependencies or [])],
            "status": (
                task.status.value if hasattr(task.status, "value") else str(task.status)
            ),
        }

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
        max_tasks = getattr(settings, "max_auto_plan_tasks", 3)
        attachments_for_iteration = attachments
        combined_contents: List[str] = []
        execution_count = 0
        bot_response: Optional[AgentResponse] = None
        bot_metadata: Dict[str, Any] = {}
        has_plan = has_existing_plan

        while True:
            # Exit early if no task to execute (plan complete)
            if auto_execute_plan and execution_count > 0 and not current_task_context:
                break

            bot_response = await self.ai_service.generate_bot_response(
                user_message=message_content,
                conversation_id=conversation_id,
                user_id=user_id,
                attachments=attachments_for_iteration,
                current_task=current_task_context,
                all_tasks=existing_tasks_dict,
                planning_mode_enabled=planning_mode_enabled,
                has_existing_plan=has_plan,
                existing_tasks=existing_tasks_dict,
            )

            if (
                bot_response
                and bot_response.metadata
                and "interrupt" in bot_response.metadata
            ):
                return (
                    None,
                    {},
                    bot_response,
                    execution_count,
                    bot_response.metadata["interrupt"],
                )

            bot_response_content = (
                bot_response.message.content
                if bot_response and bot_response.message
                else "No response generated"
            )
            combined_contents.append(bot_response_content)

            bot_metadata = dict(bot_response.metadata) if bot_response else {}
            if sanitized_persona:
                bot_metadata.setdefault("persona_used", sanitized_persona)

            if bot_response and bot_response.tool_artifacts:
                bot_metadata.setdefault("tool_artifacts", bot_response.tool_artifacts)

            if (
                bot_response
                and bot_response.metadata
                and "images" in bot_response.metadata
            ):
                bot_metadata["images"] = bot_response.metadata["images"]

            plan_saved, has_plan = self._sync_plan_from_response_metadata(
                conversation_id=conversation_id,
                user_id=user_id,
                bot_response=bot_response,
                has_existing_plan=has_plan,
            )
            if plan_saved:
                bot_metadata["plan_saved"] = True

            next_task = None
            # Mark current task as completed if planning mode is active
            # Skip if plan was just replaced (old task IDs are invalid)
            if (
                planning_mode_enabled
                and current_task
                and self.task_plan_service
                and user_id
                and not plan_saved
            ):
                fresh_task = self.task_plan_service.get_by_id(current_task.id, user_id)
                if fresh_task and fresh_task.status == TaskStatus.pending:
                    self.task_plan_service.mark_task_completed(current_task.id, user_id)

            # Always refresh task data after processing (whether plan was saved or task completed)
            if planning_mode_enabled and self.task_plan_service and user_id:
                # Refresh existing_tasks_dict with updated status
                refreshed_tasks = self.task_plan_service.get_conversation_tasks(
                    conversation_id, user_id, include_completed=True
                )
                existing_tasks_dict = [
                    {
                        "id": str(t.id),
                        "description": t.description,
                        "status": (
                            t.status.value
                            if hasattr(t.status, "value")
                            else str(t.status)
                        ),
                        "task_order": t.task_order,
                        "dependencies": [str(d) for d in (t.dependencies or [])],
                    }
                    for t in refreshed_tasks
                ]

                next_task = self.task_plan_service.get_next_task(
                    conversation_id, user_id
                )
                if next_task:
                    bot_metadata["next_task"] = {
                        "id": str(next_task.id),
                        "description": next_task.description,
                        "order": next_task.task_order,
                    }

            execution_count += 1
            attachments_for_iteration = None

            # Stop if: not auto-executing, no more tasks, or hit max tasks limit
            if not auto_execute_plan or not next_task or execution_count >= max_tasks:
                if auto_execute_plan and next_task and execution_count >= max_tasks:
                    bot_metadata["execution_paused"] = True
                    bot_metadata["execution_pause_reason"] = (
                        PauseReason.MAX_TASKS_REACHED.value
                    )
                    bot_metadata["execution_pause_message"] = (
                        f"Executed {execution_count} tasks. Send a message to continue with remaining tasks."
                    )
                break

            current_task = next_task
            current_task_context = self._build_task_context_dict(next_task)

        # If no content was generated but we have a pause message, use that as content
        if not combined_contents and bot_metadata.get("execution_pause_message"):
            combined_content = bot_metadata["execution_pause_message"]
        else:
            combined_content = (
                "\n\n---\n\n".join(combined_contents)
                if combined_contents
                else "No response generated"
            )

        # Fix markdown code blocks that may be missing newlines before opening fences
        combined_content = fix_markdown_code_blocks(combined_content)

        if auto_execute_plan and execution_count > 1:
            bot_metadata["auto_executed_tasks"] = execution_count

        return combined_content, bot_metadata, bot_response, execution_count, None
