from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Any, Dict, TYPE_CHECKING
from uuid import UUID
import redis


from app.repositories.message import MessageRepository
from app.services.generation_registry import get_generation_registry
from app.repositories.tool_approval import ToolApprovalRepository
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead
from app.models.enums import MessageRole, PlanLifecycle
from app.models.tool_approval import DecisionType
from app.factories.message_factory import MessageFactory
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params
from app.interfaces.message_service_interface import IMessageService
from app.services.ai_service import AIService
from app.services.model_config_service import ModelConfigService
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
        model_config_service: Optional[ModelConfigService] = None,
        tool_approval_repository: Optional[ToolApprovalRepository] = None,
        hitl_interrupt_repository: Optional[HITLInterruptRepository] = None,
        task_plan_service: Optional["ITaskPlanService"] = None,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        self.ai_service = ai_service
        self.model_config_service = model_config_service
        self.tool_approval_repository = tool_approval_repository
        self.hitl_interrupt_repository = hitl_interrupt_repository
        self.task_plan_service = task_plan_service
        self.redis_client = self._init_redis_client()

    def _resolve_persistent_model_request(
        self, user_id: Optional[UUID]
    ) -> Optional[Dict[str, Any]]:
        if not self.model_config_service or not user_id:
            return None

        try:
            model_request = self.model_config_service.get_effective_model_request(
                user_id
            )
        except Exception as exc:
            logging.warning(
                "Failed to load persistent model config for user %s: %s",
                user_id,
                type(exc).__name__,
                exc_info=True,
            )
            return None

        return model_request if isinstance(model_request, dict) else None

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
        self,
        conversation_id: UUID,
        content: str,
        metadata: Dict[str, Any],
        message_id: UUID | None = None,
    ) -> MessageRead:
        """Create and persist a bot response message."""
        bot_response_entity = MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=content,
            message_metadata=metadata,
            id=message_id,
        )
        bot_message = self.repository.create(bot_response_entity)
        try:
            if getattr(self.ai_service, "workflow", None):
                self.ai_service.workflow.invalidate_history_cache(str(conversation_id))
        except Exception:
            pass
        return MessageRead.model_validate(bot_message)

    @staticmethod
    def _coerce_plan_lifecycle(
        raw_lifecycle: Optional[str | PlanLifecycle],
    ) -> Optional[PlanLifecycle]:
        if isinstance(raw_lifecycle, PlanLifecycle):
            return raw_lifecycle
        if isinstance(raw_lifecycle, str):
            try:
                return PlanLifecycle(raw_lifecycle.strip().lower())
            except ValueError:
                return None
        return None

    @staticmethod
    def _infer_lifecycle_from_todos(
        todos: Optional[List[Dict[str, Any]]],
    ) -> Optional[PlanLifecycle]:
        if not isinstance(todos, list) or not todos:
            return None

        statuses = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            raw_status = todo.get("status")
            if hasattr(raw_status, "value"):
                raw_status = raw_status.value
            statuses.append(str(raw_status or "").strip().lower())

        if not statuses:
            return None

        if all(status in {"completed", "skipped"} for status in statuses):
            return PlanLifecycle.completed

        if any(
            status in {"in_progress", "completed", "skipped"} for status in statuses
        ):
            return PlanLifecycle.executing

        return PlanLifecycle.draft

    def _infer_plan_lifecycle(
        self,
        *,
        response: Optional[AgentResponse],
        current_lifecycle: Optional[str | PlanLifecycle],
    ) -> Optional[PlanLifecycle]:
        if not response or not isinstance(getattr(response, "metadata", None), dict):
            return None

        metadata = response.metadata
        if metadata.get("interrupt") is not None:
            return PlanLifecycle.paused

        if metadata.get("all_tasks_completed"):
            return PlanLifecycle.completed

        if metadata.get("pause_reason") or metadata.get("planning_budget_reached"):
            return PlanLifecycle.paused

        inferred_from_todos = self._infer_lifecycle_from_todos(metadata.get("todos"))
        if inferred_from_todos is not None:
            return inferred_from_todos

        current = self._coerce_plan_lifecycle(current_lifecycle)
        if current == PlanLifecycle.executing:
            return PlanLifecycle.executing
        return None

    def _set_plan_lifecycle(
        self,
        conversation_id: UUID,
        user_id: Optional[UUID],
        lifecycle: Optional[PlanLifecycle],
    ) -> None:
        if not self.task_plan_service or user_id is None or lifecycle is None:
            return

        try:
            self.task_plan_service.set_plan_lifecycle(
                conversation_id=conversation_id,
                user_id=user_id,
                lifecycle=lifecycle,
            )
        except Exception as exc:
            logging.warning(
                "Failed to set plan lifecycle for conversation %s: %s",
                conversation_id,
                type(exc).__name__,
                exc_info=True,
            )

    def _sync_response_plan_state(
        self,
        *,
        conversation_id: UUID,
        user_id: Optional[UUID],
        bot_response: Optional[AgentResponse],
        current_lifecycle: Optional[str | PlanLifecycle],
    ) -> bool:
        lifecycle = self._infer_plan_lifecycle(
            response=bot_response,
            current_lifecycle=current_lifecycle,
        )
        metadata = getattr(bot_response, "metadata", None) or {}
        todos = metadata.get("todos")

        if isinstance(todos, list):
            self._sync_todos_to_database(
                conversation_id=conversation_id,
                user_id=user_id,
                todos=todos,
                lifecycle=lifecycle,
            )
            return True

        self._set_plan_lifecycle(conversation_id, user_id, lifecycle)
        return False

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
            if not interrupt_response.get("metadata"):
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
        user_id: Optional[UUID] = None,
        message_id: UUID | None = None,
    ) -> MessageRead:
        """
        Persist an assistant message that represents a paused workflow awaiting HITL approval.

        Also creates a durable HITLInterrupt lifecycle record so pending approvals
        are recoverable via normal message history APIs (DB-backed) and survive
        process restarts.
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

        bot_message = self._create_bot_response_message(
            conversation_id=conversation_id,
            content="",
            metadata=metadata,
            message_id=message_id,
        )

        # Create durable lifecycle record
        if self.hitl_interrupt_repository and user_id:
            interrupt_id = interrupt_dict.get("interrupt_id")
            _thread_id = thread_id or str(conversation_id)
            if interrupt_id:
                try:
                    expires_at = datetime.now(timezone.utc) + timedelta(
                        minutes=settings.hitl_approval_timeout_minutes
                    )
                    action_requests = interrupt_dict.get("action_requests") or []
                    self.hitl_interrupt_repository.create(
                        interrupt_id=interrupt_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        thread_id=_thread_id,
                        expires_at=expires_at,
                        action_requests_json=action_requests,
                        assistant_message_id=bot_message.id,
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to create durable interrupt record for interrupt_id=%s: %s",
                        interrupt_id,
                        exc,
                        exc_info=True,
                    )

        return bot_message

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
            # Load conversation once — reused for context and planning mode
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            user_id = user_id or (conversation.owner_id if conversation else None)
            persona = conversation.persona_prompt if conversation else None
            sanitized_persona = sanitize_persona(persona)

            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )
            _lifecycle = getattr(conversation, "plan_lifecycle", None)
            plan_lifecycle_value = (
                _lifecycle.value
                if hasattr(_lifecycle, "value")
                else str(_lifecycle)
                if _lifecycle is not None
                else None
            )

            # Use shared helper to prepare planning context (non-stream path)
            planning_ctx = await self._prepare_planning_context(
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                message_content=message_create_data.content,
                planning_mode_enabled=planning_mode_enabled,
                plan_lifecycle=plan_lifecycle_value,
            )
            planning_mode_enabled = planning_ctx["planning_mode_enabled"]
            has_existing_plan = planning_ctx["has_existing_plan"]
            current_task_context = planning_ctx["current_task_context"]
            existing_tasks_dict = planning_ctx["existing_tasks_dict"]
            plan_lifecycle_value = planning_ctx.get("plan_lifecycle")

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            model_request = (
                message_create_data.model_config_field
                if (
                    hasattr(message_create_data, "model_config_field")
                    and isinstance(message_create_data.model_config_field, dict)
                    and message_create_data.model_config_field
                )
                else self._resolve_persistent_model_request(user_id)
            )

            (
                bot_response_content,
                bot_metadata,
                interrupt_payload,
            ) = await self._run_plan_execution_loop(
                message_content=message_create_data.content,
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                sanitized_persona=sanitized_persona,
                planning_mode_enabled=planning_mode_enabled,
                has_existing_plan=has_existing_plan,
                current_task_context=current_task_context,
                existing_tasks_dict=existing_tasks_dict,
                attachments=attachments,
                model_request=model_request,
                persona=sanitized_persona,
                plan_lifecycle=plan_lifecycle_value,
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
                        user_id=user_id,
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
        self,
        message_create_data: MessageCreate,
        user_id: UUID,
        bot_message_id: UUID | None = None,
    ):
        """
        Create a message and stream the bot response.
        Yields chunks as they arrive from the AI service.

        Integrates with GenerationRegistry so that in-flight streams can be
        cancelled via ``POST /messages/stop`` or HTTP disconnect without
        persisting cancellation/disconnect artifacts as error messages.
        """
        self.conversation_validation_utils.validate_conversation_access(
            user_id, message_create_data.conversation_id
        )

        # Create and persist the user message
        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )
        created_message = self.repository.create(message_entity)
        user_message_id = created_message.id  # stable key for registry

        # Register in-flight entry
        registry = get_generation_registry()
        inflight = registry.register(
            user_message_id=user_message_id,
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
        )

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
            # Load conversation once — reused for title check, context, and planning mode
            conversation = (
                self.conversation_validation_utils.conversation_repository.get_by_id(
                    message_create_data.conversation_id
                )
            )
            default_titles = {"New Conversation", "Untitled", ""}
            needs_title = conversation is not None and (
                conversation.title in default_titles or conversation.title is None
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
            # Extract context from the already-loaded conversation
            user_id = user_id or (conversation.owner_id if conversation else None)
            persona = conversation.persona_prompt if conversation else None
            sanitized_persona = sanitize_persona(persona)

            planning_mode_enabled = (
                conversation.planning_mode_enabled if conversation else False
            )
            _lc = getattr(conversation, "plan_lifecycle", None)
            plan_lifecycle_value = (
                _lc.value
                if hasattr(_lc, "value")
                else str(_lc)
                if _lc is not None
                else None
            )

            # Use shared helper to prepare planning context (stream path)
            planning_ctx = await self._prepare_planning_context(
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                message_content=message_create_data.content,
                planning_mode_enabled=planning_mode_enabled,
                plan_lifecycle=plan_lifecycle_value,
            )
            planning_mode_enabled = planning_ctx["planning_mode_enabled"]
            has_existing_plan = planning_ctx["has_existing_plan"]
            current_task_context = planning_ctx["current_task_context"]
            existing_tasks_dict = planning_ctx["existing_tasks_dict"]
            plan_lifecycle_value = planning_ctx.get("plan_lifecycle")

            # Extract attachments from message_create_data if present
            attachments = (
                message_create_data.attachments
                if hasattr(message_create_data, "attachments")
                else None
            )

            model_request = (
                message_create_data.model_config_field
                if (
                    hasattr(message_create_data, "model_config_field")
                    and isinstance(message_create_data.model_config_field, dict)
                    and message_create_data.model_config_field
                )
                else self._resolve_persistent_model_request(user_id)
            )

            # Stream bot response generation
            bot_response_content = ERROR_NO_RESPONSE
            bot_response = None
            bot_message_persisted = False

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
                    model_request=model_request,
                    persona=sanitized_persona,
                    plan_lifecycle=plan_lifecycle_value,
                ):
                    # ---- Check cancellation before processing each event ----
                    if inflight.is_cancelled:
                        logging.info(
                            "Stream cancelled for user_message_id=%s",
                            user_message_id,
                        )
                        break

                    event_type = event.get("type")

                    if event_type == "agent_selected":
                        inflight.selected_agent = event.get("agent")
                        inflight.touch()
                        yield {"type": "agent_selected", "agent": event.get("agent")}

                    elif event_type == "token":
                        token_content = event.get("content", "")
                        inflight.partial_text += token_content
                        inflight.touch()
                        yield {"type": "token", "content": token_content}

                    elif event_type == "thinking":
                        thinking_content = event.get("content", "")
                        inflight.partial_thinking += thinking_content
                        inflight.touch()
                        yield {"type": "thinking", "content": thinking_content}

                    elif event_type == "tool":
                        inflight.touch()
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
                        self._set_plan_lifecycle(
                            message_create_data.conversation_id,
                            user_id,
                            PlanLifecycle.paused,
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
                                user_id=user_id,
                                message_id=bot_message_id,
                            ).model_dump(mode="json"),
                        }
                        # Workflow is paused - don't create a bot message yet
                        _cancel_title_task()
                        inflight.resolve()
                        registry.remove(user_message_id)
                        return

                    elif event_type == "continuation_start":
                        # Auto-continue round marker — pass through without breaking
                        inflight.touch()
                        yield event

                    elif event_type == "node_complete":
                        # Node completion event — pass through without breaking
                        inflight.touch()
                        yield event

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

                # ---- Handle cancellation after the loop exits ----
                if inflight.is_cancelled:
                    _cancel_title_task()
                    partial = inflight.partial_text.strip()
                    if partial:
                        partial = fix_markdown_code_blocks(partial)
                        bot_message = self._create_bot_response_message(
                            conversation_id=message_create_data.conversation_id,
                            content=partial,
                            metadata={
                                "stopped": True,
                                "partial": True,
                                "stop_reason": "user_requested",
                                "persona_used": sanitized_persona,
                                "reply_to_user_message_id": str(user_message_id),
                            },
                            message_id=bot_message_id,
                        )
                        inflight.resolve(bot_message.model_dump(mode="json"))
                    else:
                        inflight.resolve(None)
                    registry.remove(user_message_id)
                    return

                bot_response_content = fix_markdown_code_blocks(bot_response_content)

                # Create metadata for bot response
                bot_metadata = build_bot_metadata(bot_response, sanitized_persona)
                bot_metadata["reply_to_user_message_id"] = str(user_message_id)

                # Sync todos and persist the derived lifecycle from graph state.
                if self._sync_response_plan_state(
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    bot_response=bot_response,
                    current_lifecycle=plan_lifecycle_value,
                ):
                    bot_metadata["todos_synced"] = True

                # Generate follow-up question suggestions
                await self._generate_and_add_suggestions(
                    message_create_data.content, bot_response_content, bot_metadata
                )

                # Create and persist bot response message
                bot_message = self._create_bot_response_message(
                    conversation_id=message_create_data.conversation_id,
                    content=bot_response_content,
                    metadata=bot_metadata,
                    message_id=bot_message_id,
                )
                bot_message_persisted = True

                # Resolve the inflight future with the final message
                inflight.resolve(bot_message.model_dump(mode="json"))
                registry.remove(user_message_id)

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

            except (asyncio.CancelledError, GeneratorExit):
                # Cancellation / disconnect: persist partial text if available,
                # do NOT create an error message.
                _cancel_title_task()
                if not bot_message_persisted:
                    partial = inflight.partial_text.strip()
                    if partial:
                        partial = fix_markdown_code_blocks(partial)
                        bot_msg = self._create_bot_response_message(
                            conversation_id=message_create_data.conversation_id,
                            content=partial,
                            metadata={
                                "stopped": True,
                                "partial": True,
                                "stop_reason": "disconnect",
                                "persona_used": sanitized_persona,
                                "reply_to_user_message_id": str(user_message_id),
                            },
                            message_id=bot_message_id,
                        )
                        inflight.resolve(bot_msg.model_dump(mode="json"))
                    else:
                        inflight.resolve(None)
                    registry.remove(user_message_id)
                return

            except Exception as exc:
                _cancel_title_task()
                if bot_message_persisted:
                    return
                error_content = f"Error generating response: {str(exc)}"
                error_metadata = {"error": str(exc)}

                error_message = self._create_bot_response_message(
                    conversation_id=message_create_data.conversation_id,
                    content=error_content,
                    metadata=error_metadata,
                    message_id=bot_message_id,
                )

                inflight.resolve(error_message.model_dump(mode="json"))
                registry.remove(user_message_id)

                yield {
                    "type": "error",
                    "error": str(exc),
                    "message": error_message.model_dump(mode="json"),
                }
                _cancel_title_task()

    def _validate_and_claim_interrupt_resume(
        self,
        *,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        interrupt_id: Optional[str],
    ) -> Any:
        from app.core.exceptions import CustomHTTPException
        from fastapi import status as http_status
        from app.models.hitl_interrupt import HITLInterruptStatus

        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        fetched_interrupt_record = None
        if self.hitl_interrupt_repository and interrupt_id:
            record = self.hitl_interrupt_repository.get_by_id(interrupt_id)
            fetched_interrupt_record = record
            if record is None:
                raise CustomHTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail=f"Interrupt '{interrupt_id}' not found.",
                    error_code="INTERRUPT_NOT_FOUND",
                )
            if record.conversation_id != conversation_id:
                raise CustomHTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Interrupt does not belong to this conversation.",
                    error_code="INTERRUPT_NOT_FOUND",
                )
            if record.thread_id != thread_id:
                raise CustomHTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail="Thread ID does not match the pending interrupt state.",
                    error_code="INTERRUPT_THREAD_MISMATCH",
                )
            now = datetime.now(timezone.utc)
            if record.status == HITLInterruptStatus.EXPIRED or (
                record.status == HITLInterruptStatus.PENDING
                and record.expires_at <= now
            ):
                if record.status == HITLInterruptStatus.PENDING:
                    try:
                        self.hitl_interrupt_repository.mark_expired(interrupt_id)
                    except Exception:
                        pass
                raise CustomHTTPException(
                    status_code=http_status.HTTP_410_GONE,
                    detail=(
                        "This approval request has expired. "
                        "Please send a new message to try again."
                    ),
                    error_code="INTERRUPT_EXPIRED",
                )
            if record.status in (
                HITLInterruptStatus.RESOLVED,
                HITLInterruptStatus.RESOLVING,
            ):
                raise CustomHTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail="This interrupt has already been resolved.",
                    error_code="INTERRUPT_ALREADY_RESOLVED",
                )

            won_race = self.hitl_interrupt_repository.try_transition_to_resolving(
                interrupt_id=interrupt_id,
                conversation_id=conversation_id,
                resolved_by_user_id=user_id,
            )
            if not won_race:
                raise CustomHTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail="This interrupt was claimed by a concurrent request.",
                    error_code="INTERRUPT_CONFLICT",
                )

        elif self.redis_client and interrupt_id:
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
                        self.redis_client.delete(key)
                        raise CustomHTTPException(
                            status_code=http_status.HTTP_410_GONE,
                            detail=(
                                f"This approval request has expired "
                                f"({elapsed_minutes:.0f} min elapsed, "
                                f"limit is {settings.hitl_approval_timeout_minutes} min). "
                                "Please send a new message to try again."
                            ),
                            error_code="INTERRUPT_EXPIRED",
                        )
            except CustomHTTPException:
                raise
            except Exception:
                pass

        return fetched_interrupt_record

    def _audit_interrupt_resume_decisions(
        self,
        *,
        conversation_id: UUID,
        user_id: Optional[UUID],
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str],
        fetched_interrupt_record: Any = None,
    ) -> None:
        if not self.tool_approval_repository or not user_id:
            return

        stored_original_args: Dict[str, Any] = {}
        if interrupt_id:
            try:
                if (
                    fetched_interrupt_record
                    and fetched_interrupt_record.action_requests_json
                ):
                    for req in fetched_interrupt_record.action_requests_json:
                        if isinstance(req, dict):
                            key = req.get("tool_call_id") or req.get("task_id")
                            if key:
                                stored_original_args[key] = req.get("args") or {}
                            action = req.get("action")
                            if action and action not in stored_original_args:
                                stored_original_args[action] = req.get("args") or {}
            except Exception:
                pass

        decision_type_map = {
            InterruptDecisionType.APPROVE: DecisionType.ACCEPT,
            InterruptDecisionType.EDIT: DecisionType.EDIT,
            InterruptDecisionType.REJECT: DecisionType.REJECT,
        }
        for decision in decisions:
            try:
                is_edit = decision.type == InterruptDecisionType.EDIT
                orig_key = decision.task_id or decision.action or ""
                original_args = (
                    stored_original_args.get(orig_key)
                    or stored_original_args.get(decision.action or "")
                    or {}
                )
                approval_data = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "interrupt_id": interrupt_id or "unknown",
                    "tool_name": decision.action or "unknown",
                    "tool_call_id": decision.task_id or "unknown",
                    "original_args": original_args,
                    "modified_args": decision.args if is_edit else None,
                    "decision": decision_type_map.get(
                        decision.type, DecisionType.REJECT
                    ),
                }
                self.tool_approval_repository.create(approval_data)
            except Exception as audit_exc:
                logging.warning(
                    "Audit write failed for interrupt_id=%s decision=%s: %s",
                    interrupt_id,
                    decision.type,
                    audit_exc,
                    exc_info=True,
                )

    def _normalize_nested_interrupt_payload(self, interrupt_payload: Any) -> Any:
        if isinstance(interrupt_payload, dict):
            interrupt_count = (interrupt_payload.get("metadata") or {}).get(
                "interrupt_count", 0
            ) + 1
            if not interrupt_payload.get("metadata"):
                interrupt_payload["metadata"] = {}
            interrupt_payload["metadata"]["interrupt_count"] = interrupt_count

            MAX_INTERRUPT_DEPTH = 5
            if interrupt_count > MAX_INTERRUPT_DEPTH:
                pass

            return InterruptResponse.model_validate(interrupt_payload)

        return interrupt_payload

    async def resume_message_creation_stream(
        self,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
        bot_message_id: UUID | None = None,
    ):
        fetched_interrupt_record = self._validate_and_claim_interrupt_resume(
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id=interrupt_id,
        )

        user_id, persona = self._get_conversation_context(conversation_id, user_id)
        sanitized_persona = sanitize_persona(persona)

        self._audit_interrupt_resume_decisions(
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=decisions,
            interrupt_id=interrupt_id,
            fetched_interrupt_record=fetched_interrupt_record,
        )

        partial_text = ""
        bot_message_persisted = False

        try:
            async for event in self.ai_service.resume_interrupted_execution_stream(
                thread_id=thread_id,
                decisions=decisions,
            ):
                event_type = event.get("type")

                if event_type == "agent_selected":
                    yield {"type": "agent_selected", "agent": event.get("agent")}

                elif event_type == "token":
                    token_content = event.get("content", "")
                    partial_text += token_content
                    yield {"type": "token", "content": token_content}

                elif event_type == "thinking":
                    yield {"type": "thinking", "content": event.get("content", "")}

                elif event_type == "tool":
                    yield {
                        "type": "tool",
                        "name": event.get("name"),
                        "status": event.get("status"),
                        "tool_call_id": event.get("tool_call_id"),
                        "args": event.get("args"),
                        "result": event.get("result"),
                    }

                elif event_type == "continuation_start":
                    yield event

                elif event_type == "node_complete":
                    yield event

                elif event_type == "interrupt":
                    self._clear_redis_interrupt(conversation_id, interrupt_id)

                    if self.hitl_interrupt_repository and interrupt_id:
                        try:
                            self.hitl_interrupt_repository.mark_resolved(interrupt_id)
                        except Exception:
                            pass

                    interrupt_response = event.get("interrupt")
                    normalized_interrupt = self._normalize_nested_interrupt_payload(
                        interrupt_response
                    )
                    next_interrupt_id = (
                        interrupt_response.get("interrupt_id")
                        if isinstance(interrupt_response, dict)
                        else None
                    )
                    self._handle_redis_interrupt_storage(
                        conversation_id,
                        next_interrupt_id,
                        interrupt_response,
                    )

                    persisted = self._persist_interrupt_bot_message(
                        conversation_id=conversation_id,
                        interrupt_payload=normalized_interrupt,
                        sanitized_persona=sanitized_persona,
                        pending_tool_calls=event.get("pending_tool_calls"),
                        thread_id=event.get("thread_id") or thread_id,
                        next_nodes=event.get("next"),
                        user_id=user_id,
                        message_id=bot_message_id,
                    )
                    self._set_plan_lifecycle(
                        conversation_id,
                        user_id,
                        PlanLifecycle.paused,
                    )
                    bot_message_persisted = True

                    yield {
                        "type": "interrupt",
                        "thread_id": event.get("thread_id") or thread_id,
                        "next": event.get("next"),
                        "pending_tool_calls": event.get("pending_tool_calls"),
                        "interrupt": (
                            normalized_interrupt.model_dump(mode="json")
                            if isinstance(normalized_interrupt, InterruptResponse)
                            else normalized_interrupt
                        ),
                        "message": persisted.model_dump(mode="json"),
                    }
                    return

                elif event_type == "complete":
                    bot_response = event.get("response")

                    self._clear_redis_interrupt(conversation_id, interrupt_id)
                    if self.hitl_interrupt_repository and interrupt_id:
                        try:
                            self.hitl_interrupt_repository.mark_resolved(interrupt_id)
                        except Exception:
                            pass

                    bot_response_content = extract_response_content(
                        bot_response, ERROR_RESPONSE_AFTER_RESUME
                    )
                    bot_response_content = fix_markdown_code_blocks(
                        bot_response_content
                    )

                    bot_metadata = build_bot_metadata(bot_response, sanitized_persona)
                    if self._sync_response_plan_state(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        bot_response=bot_response,
                        current_lifecycle=None,
                    ):
                        bot_metadata["todos_synced"] = True
                    bot_message = self._create_bot_response_message(
                        conversation_id=conversation_id,
                        content=bot_response_content,
                        metadata=bot_metadata,
                        message_id=bot_message_id,
                    )
                    bot_message_persisted = True

                    yield {
                        "type": "complete",
                        "message": bot_message.model_dump(mode="json"),
                    }
                    return

                elif event_type == "error":
                    self._clear_redis_interrupt(conversation_id, interrupt_id)

                    error_msg = event.get("error", UNKNOWN_ERROR)
                    error_message = self._create_bot_response_message(
                        conversation_id=conversation_id,
                        content=f"Error generating response: {error_msg}",
                        metadata={"error": error_msg},
                        message_id=bot_message_id,
                    )
                    bot_message_persisted = True

                    yield {
                        "type": "error",
                        "error": error_msg,
                        "message": error_message.model_dump(mode="json"),
                    }
                    return

            self._clear_redis_interrupt(conversation_id, interrupt_id)

            if not bot_message_persisted:
                fallback_message = self._create_bot_response_message(
                    conversation_id=conversation_id,
                    content=ERROR_RESPONSE_AFTER_RESUME,
                    metadata={"error": ERROR_RESPONSE_AFTER_RESUME},
                    message_id=bot_message_id,
                )
                yield {
                    "type": "error",
                    "error": ERROR_RESPONSE_AFTER_RESUME,
                    "message": fallback_message.model_dump(mode="json"),
                }

        except (asyncio.CancelledError, GeneratorExit):
            if bot_message_persisted:
                return

            partial = partial_text.strip()
            if partial:
                partial = fix_markdown_code_blocks(partial)
                self._create_bot_response_message(
                    conversation_id=conversation_id,
                    content=partial,
                    metadata={
                        "stopped": True,
                        "partial": True,
                        "stop_reason": "disconnect",
                        "persona_used": sanitized_persona,
                    },
                    message_id=bot_message_id,
                )
            return

        except Exception as exc:
            self._clear_redis_interrupt(conversation_id, interrupt_id)
            if bot_message_persisted:
                return

            error_message = self._create_bot_response_message(
                conversation_id=conversation_id,
                content=f"Error generating response: {str(exc)}",
                metadata={"error": str(exc)},
                message_id=bot_message_id,
            )

            yield {
                "type": "error",
                "error": str(exc),
                "message": error_message.model_dump(mode="json"),
            }

    async def stop_message_generation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_message_id: UUID,
    ) -> dict:
        """
        Request cancellation of an in-flight streaming generation.

        Returns a dict with:
          - ``status``: ``"cancelled"`` | ``"not_inflight"``
          - ``message``: optional MessageRead dict (the persisted partial/final message)
        """
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        registry = get_generation_registry()
        entry = registry.get(user_message_id)

        if entry is None:
            # Not in flight – generation already completed or never started.
            return {"status": "not_inflight", "message": None}

        # Verify the caller owns this entry
        if entry.conversation_id != conversation_id or entry.user_id != user_id:
            return {"status": "not_inflight", "message": None}

        # Signal cancellation
        entry.request_cancel()

        # Wait for the producer to finish (with a timeout)
        try:
            result = await asyncio.wait_for(asyncio.shield(entry.done), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            result = None

        # Clean up
        registry.remove(user_message_id)

        return {"status": "cancelled", "message": result}

    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        if hasattr(message_entity, "content"):
            message_entity.content = normalize_message_content(
                message_entity.content,
                getattr(message_entity, "message_metadata", None),
            )
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
                msg.content = normalize_message_content(
                    msg.content,
                    getattr(msg, "message_metadata", None),
                )
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
                msg.content = normalize_message_content(
                    msg.content,
                    getattr(msg, "message_metadata", None),
                )
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
    ) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_access(
            user_id, conversation_id
        )

        bot_response = await self.ai_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=user_input,
        )

        bot_response_content = extract_response_content(
            bot_response, NO_RESPONSE_GENERATED
        )

        bot_metadata = build_bot_metadata(bot_response)
        if self._sync_response_plan_state(
            conversation_id=conversation_id,
            user_id=user_id,
            bot_response=bot_response,
            current_lifecycle=None,
        ):
            bot_metadata["todos_synced"] = True

        return self._create_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=bot_metadata,
        )

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
        plan_lifecycle: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare task planning context for message creation.

        Returns a dict with keys:
        - planning_mode_enabled: bool
        - has_existing_plan: bool
        - current_task_context: Optional[dict]
        - existing_tasks_dict: Optional[List[dict]]
        - plan_lifecycle: Optional[str]
        """
        result = {
            "planning_mode_enabled": planning_mode_enabled,
            "has_existing_plan": False,
            "current_task_context": None,
            "existing_tasks_dict": None,
            "plan_lifecycle": plan_lifecycle,
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
            current_task = self.task_plan_service.get_active_or_next_task(
                conversation_id, user_id
            )
            result["current_task_context"] = self._build_task_context_dict(current_task)

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
        except Exception as exc:
            logging.warning(
                "Failed to prepare planning context for conversation %s: %s",
                conversation_id,
                type(exc).__name__,
                exc_info=True,
            )

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
        current_task_context: Optional[Dict[str, Any]],
        existing_tasks_dict: Optional[List[Dict[str, Any]]],
        attachments: Optional[list],
        model_request: Optional[Dict[str, Any]] = None,
        persona: Optional[str] = None,
        plan_lifecycle: Optional[str] = None,
    ) -> Tuple[
        Optional[str],
        Dict[str, Any],
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
            model_request=model_request,
            persona=persona,
            plan_lifecycle=plan_lifecycle,
        )

        # Handle interrupts (HITL)
        if (
            bot_response
            and bot_response.metadata
            and "interrupt" in bot_response.metadata
        ):
            self._set_plan_lifecycle(
                conversation_id,
                user_id,
                PlanLifecycle.paused,
            )
            return (
                None,
                {},
                bot_response.metadata["interrupt"],
            )

        bot_response_content = extract_response_content(bot_response)

        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)

        # Sync todos and persist the derived lifecycle from graph state if present.
        if self._sync_response_plan_state(
            conversation_id=conversation_id,
            user_id=user_id,
            bot_response=bot_response,
            current_lifecycle=plan_lifecycle,
        ):
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
                next_task = self.task_plan_service.get_active_or_next_task(
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

        bot_response_content = fix_markdown_code_blocks(bot_response_content)

        return bot_response_content, bot_metadata, None

    def _sync_todos_to_database(
        self,
        conversation_id: UUID,
        user_id: Optional[UUID],
        todos: List[Dict[str, Any]],
        lifecycle: Optional[PlanLifecycle] = None,
    ) -> None:
        if not self.task_plan_service or user_id is None or todos is None:
            return

        try:
            self.task_plan_service.sync_todos_from_agent(
                conversation_id=conversation_id,
                user_id=user_id,
                todos=todos,
                preserve_existing_status=False,
                lifecycle=lifecycle,
            )
        except Exception as exc:
            logging.warning(
                "Failed to sync todos for conversation %s: %s",
                conversation_id,
                type(exc).__name__,
                exc_info=True,
            )
