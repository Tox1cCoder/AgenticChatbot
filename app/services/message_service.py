from __future__ import annotations

import asyncio
import contextlib
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

try:
    import redis
except ImportError:  # pragma: no cover - exercised in environments without redis installed
    redis = None

from fastapi import status as http_status

from app.ai.suggestion_generator import generate_follow_up_suggestions
from app.ai.utils import resolve_interrupt_decision_id
from app.core.config import settings
from app.core.exceptions import CustomHTTPException, PauseReason
from app.core.response_constants import (
    ERROR_RESPONSE_AFTER_RESUME,
    NO_RESPONSE_GENERATED,
    UNKNOWN_ERROR,
    build_bot_metadata,
    externalize_metadata_images,
    extract_response_content,
    normalize_message_content,
)
from app.core.rich_placement import finalize_article_content
from app.core.rich_response import (
    PROTECTED_IMAGE_URL_PREFIXES,
    RichItemType,
    provenance_provider,
    remove_inline_rich_reference,
    validate_rich_references,
)
from app.factories.message_factory import MessageFactory
from app.interfaces.message_service_interface import IMessageService
from app.models.enums import MessageRole, PlanLifecycle
from app.models.hitl_interrupt import HITLInterruptStatus
from app.models.tool_approval import DecisionType
from app.observability.rich_images import rich_image_metrics
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.repositories.message import MessageRepository
from app.repositories.tool_approval import ToolApprovalRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageRead, MessageUpdate
from app.schemas.workflow import (
    InterruptDecision,
    InterruptDecisionType,
    InterruptResponse,
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
)
from app.services.ai_service import AIService
from app.services.client_device_service import ClientDeviceService
from app.services.event_streaming.events import V3StreamEvent, make_event
from app.services.generation_registry import get_generation_registry
from app.services.verified_image_bytes import take_verified_bytes
from app.usage import UsageContext
from app.utils.text_processing import fix_markdown_code_blocks, sanitize_persona
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params

if TYPE_CHECKING:
    from app.interfaces.task_plan_service_interface import ITaskPlanService


def _merge_stream_tool_artifacts_into_response(
    response: WorkflowResponse | None,
    stream_tool_artifacts: list[dict[str, Any]] | None,
) -> None:
    """Preserve structured streaming tool artifacts on the final response."""
    if response is None or not stream_tool_artifacts:
        return

    existing = list(response.tool_artifacts or [])
    existing_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in existing:
        if not isinstance(artifact, dict):
            continue
        key = (str(artifact.get("tool_call_id") or ""), str(artifact.get("tool") or ""))
        existing_by_key[key] = artifact

    for artifact in stream_tool_artifacts:
        if not isinstance(artifact, dict):
            continue
        key = (str(artifact.get("tool_call_id") or ""), str(artifact.get("tool") or ""))
        existing_artifact = existing_by_key.get(key)
        if existing_artifact is not None:
            for field in ("args", "output", "error", "status", "render"):
                if artifact.get(field) not in (None, "", [], {}) and existing_artifact.get(
                    field
                ) in (
                    None,
                    "",
                    [],
                    {},
                ):
                    existing_artifact[field] = artifact[field]
            continue
        existing.append(artifact)
        existing_by_key[key] = artifact

    response.tool_artifacts = existing or None


def _service_event_from_ai_event(
    event: V3StreamEvent,
    *,
    sequence: int,
) -> V3StreamEvent:
    """Re-stamp a canonical AI-service event with the service-level sequence.

    The service interleaves its own events (``user_message_created``,
    ``title_updated``, terminal ``complete``/``error``/``interrupt``) with
    forwarded AI events, so sequence numbers are re-assigned here to keep the
    public stream strictly monotonic.
    """
    return event.model_copy(update={"sequence": sequence})


class MessageService(IMessageService):
    def __init__(
        self,
        message_repository: MessageRepository,
        conversation_validation_utils: ConversationValidationUtils,
        message_validation_utils: MessageValidationUtils,
        ai_service: AIService,
        tool_approval_repository: ToolApprovalRepository | None = None,
        hitl_interrupt_repository: HITLInterruptRepository | None = None,
        task_plan_service: ITaskPlanService | None = None,
        custom_agent_service: Any | None = None,
        tool_approval_setting_repository=None,
        chat_image_service=None,
        web_image_service=None,
    ):
        self.repository = message_repository
        self.conversation_validation_utils = conversation_validation_utils
        self.message_validation_utils = message_validation_utils
        self.ai_service = ai_service
        self.tool_approval_repository = tool_approval_repository
        self.hitl_interrupt_repository = hitl_interrupt_repository
        self.task_plan_service = task_plan_service
        # Resolves attached custom agents into workflow state each user turn.
        self.custom_agent_service = custom_agent_service
        # Externalizes inline image base64 out of persisted message metadata.
        self.chat_image_service = chat_image_service
        # Converts selected third-party URLs to authenticated opaque references.
        self.web_image_service = web_image_service
        # Resolves the per-user HITL approval policy into workflow state each turn.
        self.tool_approval_setting_repository = tool_approval_setting_repository
        self.redis_client = self._init_redis_client()

    def _init_redis_client(self):
        redis_url = getattr(settings, "redis_url", "") or ""
        if redis is None or not redis_url.strip():
            return None
        return redis.from_url(redis_url)

    @staticmethod
    def _normalize_tool_provenance_map(
        interrupt_metadata: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(interrupt_metadata, dict):
            return {}

        raw_provenance = interrupt_metadata.get("tool_provenance")
        if not isinstance(raw_provenance, dict):
            return {}

        return {str(key): value for key, value in raw_provenance.items() if isinstance(value, dict)}

    @staticmethod
    def _is_client_runtime_provenance_entry(provenance: dict[str, Any]) -> bool:
        if not isinstance(provenance, dict):
            return False

        tool_origin = str(provenance.get("tool_origin") or "").strip().lower()
        if tool_origin:
            return tool_origin.startswith("client_")

        # Qualified IDs identify both server and client MCP tools.  Only the
        # runtime binding fields are a safe legacy fallback when origin is
        # absent.
        return (
            provenance.get("session_id") not in (None, "")
            or provenance.get("catalog_version") is not None
            or provenance.get("tool_instance_id") not in (None, "")
        )

    @classmethod
    def _derive_interrupt_execution_scope(
        cls,
        interrupt_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        client_entries = [
            provenance
            for provenance in cls._normalize_tool_provenance_map(interrupt_metadata).values()
            if cls._is_client_runtime_provenance_entry(provenance)
        ]
        if not client_entries:
            return {
                "session_id": None,
                "catalog_version": None,
                "tool_instance_id": None,
            }

        session_ids = {
            str(provenance["session_id"])
            for provenance in client_entries
            if provenance.get("session_id") not in (None, "")
        }
        catalog_versions = {
            int(provenance["catalog_version"])
            for provenance in client_entries
            if provenance.get("catalog_version") is not None
        }
        tool_instance_ids = {
            str(provenance["tool_instance_id"])
            for provenance in client_entries
            if provenance.get("tool_instance_id") not in (None, "")
        }

        return {
            "session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
            "catalog_version": (
                next(iter(catalog_versions)) if len(catalog_versions) == 1 else None
            ),
            "tool_instance_id": (
                next(iter(tool_instance_ids)) if len(tool_instance_ids) == 1 else None
            ),
        }

    @classmethod
    def _record_execution_scope_provenance(cls, record: Any) -> dict[str, Any]:
        provenance: dict[str, Any] = {}
        if getattr(record, "device_id", None) is not None:
            provenance["device_id"] = str(record.device_id)
        if getattr(record, "session_id", None) not in (None, ""):
            provenance["session_id"] = str(record.session_id)
        if getattr(record, "catalog_version", None) is not None:
            provenance["catalog_version"] = int(record.catalog_version)
        if getattr(record, "tool_instance_id", None) not in (None, ""):
            provenance["tool_instance_id"] = str(record.tool_instance_id)
        return provenance

    @classmethod
    def _get_runtime_validation_provenance(
        cls,
        record: Any,
        decisions: list[InterruptDecision] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        provenance_map = cls._normalize_tool_provenance_map(
            getattr(record, "interrupt_metadata_json", None)
        )

        if decisions is not None:
            actionable_decisions = [
                decision
                for decision in decisions
                if decision.type
                in (
                    InterruptDecisionType.APPROVE,
                    InterruptDecisionType.EDIT,
                )
            ]
            if not actionable_decisions:
                return []

            matched_entries: list[tuple[str, dict[str, Any]]] = []
            for decision in actionable_decisions:
                lookup_keys = []
                decision_id = resolve_interrupt_decision_id(decision)
                if decision_id:
                    lookup_keys.append(decision_id)
                if decision.action:
                    action_key = str(decision.action)
                    if action_key not in lookup_keys:
                        lookup_keys.append(action_key)

                for key in lookup_keys:
                    provenance = provenance_map.get(key)
                    if cls._is_client_runtime_provenance_entry(provenance or {}):
                        matched_entries.append((key, provenance))
                        break

            if matched_entries:
                return matched_entries

            if not provenance_map:
                fallback = cls._record_execution_scope_provenance(record)
                if cls._is_client_runtime_provenance_entry(fallback):
                    return [("interrupt", fallback)]
            return []

        client_entries = [
            (key, provenance)
            for key, provenance in provenance_map.items()
            if cls._is_client_runtime_provenance_entry(provenance)
        ]
        if client_entries:
            return client_entries

        fallback = cls._record_execution_scope_provenance(record)
        if cls._is_client_runtime_provenance_entry(fallback):
            return [("interrupt", fallback)]
        return []

    def _expire_interrupt_for_scope_change(
        self,
        *,
        interrupt_id: str | None,
        resolution_source: str,
    ) -> None:
        if not interrupt_id or not self.hitl_interrupt_repository:
            return

        with contextlib.suppress(Exception):
            self.hitl_interrupt_repository.mark_expired(
                interrupt_id,
                resolution_source=resolution_source,
            )

    def _mark_claimed_interrupt_failed(self, interrupt_id: str | None, source: str) -> None:
        """Best-effort terminal transition for a previously claimed resume."""
        if self.hitl_interrupt_repository and interrupt_id:
            with contextlib.suppress(Exception):
                self.hitl_interrupt_repository.mark_failed(
                    interrupt_id,
                    resolution_source=source,
                )

    @staticmethod
    def _decision_action(decision: Any) -> str | None:
        if isinstance(decision, dict):
            value = decision.get("action")
        else:
            value = getattr(decision, "action", None)
        if value in (None, ""):
            return None
        return str(value)

    @classmethod
    def _validate_complete_interrupt_decisions(
        cls,
        *,
        record: Any,
        decisions: list[InterruptDecision] | None,
    ) -> None:
        pending_requests = getattr(record, "action_requests_json", None)
        if not isinstance(pending_requests, list) or not pending_requests:
            return

        action_counts: dict[str, int] = {}
        for request in pending_requests:
            if not isinstance(request, dict):
                continue
            action = request.get("action")
            if action not in (None, ""):
                action_key = str(action)
                action_counts[action_key] = action_counts.get(action_key, 0) + 1

        request_indexes_by_key: dict[str, int] = {}
        for index, request in enumerate(pending_requests):
            if not isinstance(request, dict):
                continue
            for id_key in ("tool_call_id", "task_id"):
                value = request.get(id_key)
                if value not in (None, ""):
                    request_indexes_by_key[str(value)] = index
            action = request.get("action")
            if action not in (None, "") and action_counts.get(str(action)) == 1:
                request_indexes_by_key[str(action)] = index

        seen_request_indexes: set[int] = set()
        for decision in decisions or []:
            candidate_keys: list[str] = []
            decision_id = resolve_interrupt_decision_id(decision)
            if decision_id:
                candidate_keys.append(str(decision_id))
            action = cls._decision_action(decision)
            if action:
                candidate_keys.append(action)

            matched_index = next(
                (
                    request_indexes_by_key[key]
                    for key in candidate_keys
                    if key in request_indexes_by_key
                ),
                None,
            )
            if matched_index is None:
                raise CustomHTTPException(
                    status_code=422,
                    detail="Resume decision targets an unknown pending tool call.",
                    error_code="INTERRUPT_UNKNOWN_DECISION",
                )
            if matched_index in seen_request_indexes:
                raise CustomHTTPException(
                    status_code=422,
                    detail="Only one resume decision is allowed per pending tool call.",
                    error_code="INTERRUPT_DUPLICATE_DECISION",
                )
            seen_request_indexes.add(matched_index)

            request = pending_requests[matched_index]
            allowed_raw = request.get("allowed_decisions") or request.get("allowedDecisions")
            if isinstance(allowed_raw, str):
                allowed = {allowed_raw.strip().lower()}
            elif isinstance(allowed_raw, list):
                allowed = {
                    str(value).strip().lower()
                    for value in allowed_raw
                    if isinstance(value, str) and value.strip()
                }
            else:
                allowed = {"approve", "edit", "reject"}

            decision_type = getattr(decision, "type", None)
            decision_type = getattr(decision_type, "value", decision_type)
            normalized_type = str(decision_type or "").strip().lower()
            if normalized_type not in allowed:
                raise CustomHTTPException(
                    status_code=422,
                    detail=(
                        f"Decision '{normalized_type}' is not allowed for this pending tool call."
                    ),
                    error_code="INTERRUPT_DECISION_NOT_ALLOWED",
                )

        decision_keys: set[str] = set()
        for decision in decisions or []:
            decision_id = resolve_interrupt_decision_id(decision)
            if decision_id:
                decision_keys.add(str(decision_id))
            action = cls._decision_action(decision)
            if action:
                decision_keys.add(action)

        missing: list[str] = []
        for request in pending_requests:
            if not isinstance(request, dict):
                continue

            request_keys: list[str] = []
            tool_call_id = request.get("tool_call_id")
            task_id = request.get("task_id")
            if tool_call_id not in (None, ""):
                request_keys.append(str(tool_call_id))
            elif task_id not in (None, ""):
                request_keys.append(str(task_id))
            else:
                action = request.get("action")
                if action not in (None, "") and action_counts.get(str(action)) == 1:
                    request_keys.append(str(action))

            if request_keys and not any(key in decision_keys for key in request_keys):
                missing.append(request_keys[0])

        if missing:
            display_missing = ", ".join(missing[:5])
            extra = "" if len(missing) <= 5 else f", +{len(missing) - 5} more"
            raise CustomHTTPException(
                status_code=422,
                detail=(
                    "Resume decisions must include an explicit approve, edit, or reject decision "
                    f"for every pending tool call. Missing decisions for: {display_missing}{extra}."
                ),
                error_code="INTERRUPT_INCOMPLETE_DECISIONS",
            )

    def _get_conversation_context(
        self, conversation_id: UUID, user_id: UUID | None = None
    ) -> tuple[UUID | None, str | None]:
        """Get user_id and persona from conversation."""
        conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
            conversation_id
        )
        resolved_user_id = user_id or (conversation.owner_id if conversation else None)
        persona = conversation.persona_prompt if conversation else None
        return resolved_user_id, persona

    def _create_bot_response_message(
        self,
        conversation_id: UUID,
        content: str,
        metadata: dict[str, Any],
        message_id: UUID | None = None,
    ) -> MessageRead:
        """Create and persist a bot response message.

        Retained deliberately for two kinds of caller:

        * synchronous callers such as :meth:`_persist_interrupt_bot_message`; and
        * ``except``/``finally`` handlers on the cancellation and error paths.
          Awaiting inside a handler whose task is already being cancelled raises
          ``CancelledError`` at the ``await``, which would silently skip
          persisting the error message the user is waiting for. Those paths are
          rare, so blocking briefly is the right trade for completing reliably.

        Normal terminal persistence uses
        :meth:`_acreate_bot_response_message`.
        """
        bot_response_entity = self._bot_response_entity(
            conversation_id, content, metadata, message_id
        )
        bot_message = self.repository.create(bot_response_entity)
        return self._finalize_bot_response_message(conversation_id, bot_message)

    async def _acreate_bot_response_message(
        self,
        conversation_id: UUID,
        content: str,
        metadata: dict[str, Any],
        message_id: UUID | None = None,
    ) -> MessageRead:
        """Async twin of :meth:`_create_bot_response_message`.

        Used on the terminal paths that run for every turn, so the assistant
        insert does not block the event loop mid-stream and delay other
        in-flight streams. Not for use inside exception handlers — see the sync
        method's docstring.
        """
        bot_response_entity = self._bot_response_entity(
            conversation_id, content, metadata, message_id
        )
        bot_message = await self.repository.acreate(bot_response_entity)
        return self._finalize_bot_response_message(conversation_id, bot_message)

    @staticmethod
    def _bot_response_entity(
        conversation_id: UUID,
        content: str,
        metadata: dict[str, Any],
        message_id: UUID | None,
    ) -> dict[str, Any]:
        """Build the assistant row. No database access."""
        return MessageFactory.create_bot_response(
            conversation_id=conversation_id,
            content=content,
            message_metadata=metadata,
            id=message_id,
        )

    def _finalize_bot_response_message(
        self, conversation_id: UUID, bot_message: Any
    ) -> MessageRead:
        """Invalidate cached prompt history and project the persisted row."""
        with contextlib.suppress(Exception):
            self.ai_service.invalidate_history_cache(str(conversation_id))
        return MessageRead.model_validate(bot_message)

    async def _compact_checkpoint_after_persist(
        self,
        *,
        conversation_id: UUID | None = None,
        workflow_request: WorkflowExecutionRequest | None = None,
        thread_id: str | None = None,
    ) -> None:
        resolved_thread_id = (
            thread_id
            or (workflow_request.thread_id if workflow_request is not None else None)
            or (str(conversation_id) if conversation_id is not None else None)
        )
        if not resolved_thread_id:
            return
        with contextlib.suppress(Exception):
            await self.ai_service.compact_checkpoint_after_terminal_response(resolved_thread_id)

    @staticmethod
    def _coerce_plan_lifecycle(
        raw_lifecycle: str | PlanLifecycle | None,
    ) -> PlanLifecycle | None:
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
        todos: list[dict[str, Any]] | None,
    ) -> PlanLifecycle | None:
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

        if any(status in {"in_progress", "completed", "skipped"} for status in statuses):
            return PlanLifecycle.executing

        return PlanLifecycle.draft

    def _infer_plan_lifecycle(
        self,
        *,
        response: WorkflowResponse | None,
        current_lifecycle: str | PlanLifecycle | None,
    ) -> PlanLifecycle | None:
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
        user_id: UUID | None,
        lifecycle: PlanLifecycle | None,
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
        user_id: UUID | None,
        bot_response: WorkflowResponse | None,
        current_lifecycle: str | PlanLifecycle | None,
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
        self,
        user_query: str,
        response_content: str,
        metadata: dict[str, Any],
        *,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
        request_message_id: UUID | None = None,
    ) -> None:
        """Generate follow-up suggestions and add to metadata."""
        try:
            usage_context = None
            recorder = getattr(self.ai_service, "model_usage_recorder", None)
            if recorder is not None:
                usage_context = UsageContext(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_message_id=request_message_id,
                    operation="suggestions",
                    agent_id="suggestion_generator",
                )
            suggestions = await generate_follow_up_suggestions(
                user_query=user_query,
                response_content=response_content,
                usage_context=usage_context,
                recorder=recorder,
            )
            if suggestions:
                metadata["suggested_questions"] = suggestions
        except Exception:
            pass

    def _is_first_user_message(self, conversation_id: UUID) -> bool:
        """Check if this is the first user message in the conversation."""
        try:
            # Check if conversation has a default title (needs generation)
            conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
                conversation_id
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
        user_id: UUID | None = None,
    ) -> str | None:
        """
        Generate and update conversation title asynchronously.

        Returns:
            The generated title if successful, None otherwise.
        """
        try:
            title = await self.ai_service.generate_conversation_title(
                user_message,
                user_id=user_id,
                conversation_id=conversation_id,
            )
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
        interrupt_id: str | None,
        interrupt_response: dict[str, Any],
    ) -> None:
        """Store interrupt information in Redis with timeout."""
        if not self.redis_client or not interrupt_response or not interrupt_id:
            return

        key = f"interrupt:{conversation_id}:{interrupt_id}"
        timeout_seconds = settings.hitl_approval_timeout_minutes * 60
        try:
            self.redis_client.setex(key, timeout_seconds, datetime.now(timezone.utc).isoformat())
            deadline = datetime.now(timezone.utc) + timedelta(
                minutes=settings.hitl_approval_timeout_minutes
            )
            if not interrupt_response.get("metadata"):
                interrupt_response["metadata"] = {}
            interrupt_response["metadata"]["timeout_deadline"] = deadline.isoformat()
        except Exception:
            pass

    def _clear_redis_interrupt(self, conversation_id: UUID, interrupt_id: str | None) -> None:
        """Clear interrupt information from Redis."""
        if not self.redis_client or not interrupt_id:
            return

        key = f"interrupt:{conversation_id}:{interrupt_id}"
        with contextlib.suppress(Exception):
            self.redis_client.delete(key)

    def _persist_interrupt_bot_message(
        self,
        conversation_id: UUID,
        interrupt_payload: Any,
        sanitized_persona: str | None = None,
        pending_tool_calls: Any | None = None,
        thread_id: str | None = None,
        next_nodes: Any | None = None,
        user_id: UUID | None = None,
        message_id: UUID | None = None,
        tool_artifacts: list[dict[str, Any]] | None = None,
        selected_agent: str | None = None,
        custom_agents: dict[str, Any] | None = None,
        require_durable_interrupt: bool = False,
    ) -> MessageRead:
        """
        Persist an assistant message that represents a paused workflow awaiting HITL approval.

        Also creates a durable HITLInterrupt lifecycle record so pending approvals
        are recoverable via normal message history APIs (DB-backed) and survive
        process restarts. Callers handling a claimed nested resume can require
        durable-record failures to propagate.
        """
        if isinstance(interrupt_payload, InterruptResponse):
            interrupt_dict = interrupt_payload.model_dump(mode="json")
        elif isinstance(interrupt_payload, dict):
            interrupt_dict = interrupt_payload
        else:
            interrupt_dict = {"raw": str(interrupt_payload)}

        interrupt_metadata = (
            interrupt_dict.get("metadata")
            if isinstance(interrupt_dict.get("metadata"), dict)
            else {}
        )
        raw_device_id = interrupt_metadata.get("device_id")
        interrupt_device_id: UUID | None = None
        if raw_device_id:
            with contextlib.suppress(Exception):
                interrupt_device_id = UUID(str(raw_device_id))

        metadata: dict[str, Any] = {
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

        # Preserve live_widgets from widget tools that already succeeded
        # before the interrupt paused the run.
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
            from app.core.response_constants import extract_live_widgets_from_artifacts

            live_widgets = extract_live_widgets_from_artifacts(tool_artifacts)
            if live_widgets:
                metadata["live_widgets"] = live_widgets

        self._attach_selected_agent_metadata(metadata, selected_agent, custom_agents)

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
                    execution_scope = self._derive_interrupt_execution_scope(interrupt_metadata)
                    self.hitl_interrupt_repository.create(
                        interrupt_id=interrupt_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        thread_id=_thread_id,
                        expires_at=expires_at,
                        action_requests_json=action_requests,
                        assistant_message_id=bot_message.id,
                        device_id=interrupt_device_id,
                        interrupt_metadata_json=interrupt_metadata,
                        session_id=execution_scope.get("session_id"),
                        catalog_version=execution_scope.get("catalog_version"),
                        tool_instance_id=execution_scope.get("tool_instance_id"),
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to create durable interrupt record for interrupt_id=%s: %s",
                        interrupt_id,
                        exc,
                        exc_info=True,
                    )
                    if require_durable_interrupt:
                        self.repository.delete(bot_message.id)
                        raise

        return bot_message

    async def create_message(
        self, message_create_data: MessageCreate, user_id: UUID
    ) -> MessageRead:
        # This is an async endpoint path, so its database work uses the async
        # transport; a sync call here blocks the loop for every other request.
        await self.conversation_validation_utils.avalidate_conversation_access(
            user_id, message_create_data.conversation_id
        )

        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )
        if message_create_data.role == MessageRole.user:
            refs = self._externalize_attachments_for_persist(message_create_data, user_id)
            if refs is not None and isinstance(message_entity.get("message_metadata"), dict):
                message_entity["message_metadata"]["attachments"] = refs

        created_message = await self.repository.acreate(message_entity)
        # Reserve the assistant DB id up-front so the graph can stamp the
        # outgoing AIMessage with the same id we will later persist.
        assistant_message_id = uuid4()
        # Invalidate any cached prompt history for this conversation now that
        # a new transcript row exists.
        with contextlib.suppress(Exception):
            self.ai_service.invalidate_history_cache(str(message_create_data.conversation_id))

        if message_create_data.role == MessageRole.user:
            # Load conversation once — reused for context and planning mode
            conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
                message_create_data.conversation_id
            )
            (
                resolved_user_id,
                sanitized_persona,
                workflow_request,
            ) = await self._build_user_message_workflow_request(
                message_create_data=message_create_data,
                user_id=user_id,
                conversation=conversation,
                user_message_id=created_message.id,
                assistant_message_id=assistant_message_id,
            )

            bot_response, interrupt_payload = await self._execute_user_message_workflow(
                workflow_request=workflow_request,
                conversation_id=message_create_data.conversation_id,
                user_id=resolved_user_id,
            )

            if interrupt_payload:
                user_message_read = MessageRead.model_validate(created_message)
                if isinstance(interrupt_payload, dict):
                    interrupt_payload = InterruptResponse.model_validate(interrupt_payload)
                user_message_read.interrupt = interrupt_payload

                # Persist an assistant "approval required" message so clients can
                # recover pending approvals from message history (not just SSE).
                with contextlib.suppress(Exception):
                    self._persist_interrupt_bot_message(
                        conversation_id=message_create_data.conversation_id,
                        interrupt_payload=interrupt_payload,
                        sanitized_persona=sanitized_persona,
                        thread_id=getattr(interrupt_payload, "thread_id", None),
                        pending_tool_calls=(
                            [
                                r.model_dump(mode="json")
                                for r in getattr(interrupt_payload, "action_requests", [])
                            ]
                            if isinstance(interrupt_payload, InterruptResponse)
                            else None
                        ),
                        user_id=resolved_user_id,
                        selected_agent=bot_response.agent_id if bot_response else None,
                        custom_agents=workflow_request.custom_agents,
                    )

                return user_message_read

            await self._persist_completed_workflow_response(
                conversation_id=message_create_data.conversation_id,
                user_id=resolved_user_id,
                bot_response=bot_response,
                sanitized_persona=sanitized_persona,
                workflow_request=workflow_request,
                message_id=assistant_message_id,
            )
            await self._compact_checkpoint_after_persist(
                conversation_id=message_create_data.conversation_id,
                workflow_request=workflow_request,
            )

        return MessageRead.model_validate(created_message)

    async def create_message_stream(
        self,
        message_create_data: MessageCreate,
        user_id: UUID,
        bot_message_id: UUID | None = None,
    ):
        # Reserve the assistant DB id up-front when the caller did not supply
        # one so the workflow request can carry a stable id.
        if bot_message_id is None:
            bot_message_id = uuid4()
        """
        Create a message and stream the bot response.
        Yields chunks as they arrive from the AI service.

        Integrates with GenerationRegistry so that in-flight streams can be
        cancelled via ``POST /messages/stop`` or HTTP disconnect without
        persisting cancellation/disconnect artifacts as error messages.
        """
        # Everything before the first streamed event runs on the async
        # transport: a blocking query here delays not just this response's first
        # token but every other in-flight stream sharing the event loop.
        await self.conversation_validation_utils.avalidate_conversation_access(
            user_id, message_create_data.conversation_id
        )

        # Create and persist the user message
        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )
        created_message = await self.repository.acreate(message_entity)
        user_message_id = created_message.id  # stable key for registry
        # Drop any stale prompt-history cache before the workflow reads it.
        with contextlib.suppress(Exception):
            self.ai_service.invalidate_history_cache(str(message_create_data.conversation_id))

        # Register in-flight entry
        registry = get_generation_registry()
        inflight = registry.register(
            user_message_id=user_message_id,
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
        )

        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        # Yield user message creation event
        yield make_event(
            "user_message_created",
            sequence=_next_sequence(),
            conversation_id=str(message_create_data.conversation_id),
            message_id=str(user_message_id),
            data={"message": MessageRead.model_validate(created_message).model_dump(mode="json")},
        )

        # Start async title generation only if this is a user message and the first one
        title_task = None
        if message_create_data.role == MessageRole.user:
            # Load conversation once — reused for title check, context, and planning mode
            conversation = (
                await self.conversation_validation_utils.conversation_repository.aget_by_id(
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
                        message_create_data.conversation_id,
                        message_create_data.content,
                        user_id=user_id,
                    )
                )

        def _cancel_title_task():
            """Cancel title task if running to prevent resource leaks."""
            if title_task and not title_task.done():
                title_task.cancel()

        if message_create_data.role == MessageRole.user:
            # Extract context from the already-loaded conversation
            (
                resolved_user_id,
                sanitized_persona,
                workflow_request,
            ) = await self._build_user_message_workflow_request(
                message_create_data=message_create_data,
                user_id=user_id,
                conversation=conversation,
                user_message_id=user_message_id,
                assistant_message_id=bot_message_id,
            )

            # Stream bot response generation
            bot_response = None
            bot_message_persisted = False
            stream_tool_artifacts: list[dict[str, Any]] = []
            stream_tool_args_by_id: dict[str, Any] = {}

            try:
                async for raw_event in self.ai_service.execute_request_stream(workflow_request):
                    # ---- Check cancellation before processing each event ----
                    if inflight.is_cancelled:
                        logging.info(
                            "Stream cancelled for user_message_id=%s",
                            user_message_id,
                        )
                        break

                    event = _service_event_from_ai_event(raw_event, sequence=_next_sequence())
                    event_type = event.type

                    if event_type == "agent_selected":
                        selected_agent = event.agent or event.data.get("agent")
                        inflight.selected_agent = selected_agent
                        inflight.touch()
                        yield self._agent_selected_event(
                            selected_agent,
                            workflow_request.custom_agents,
                            sequence=event.sequence,
                        )

                    elif event_type == "message_delta":
                        inflight.partial_text += event.data.get("text", "")
                        inflight.touch()
                        yield event

                    elif event_type == "reasoning_delta":
                        inflight.partial_thinking += event.data.get("text", "")
                        inflight.touch()
                        yield event

                    elif event_type == "tool_call_available":
                        inflight.touch()
                        if event.tool_call_id is not None:
                            stream_tool_args_by_id[str(event.tool_call_id)] = event.data.get("args")
                        yield event

                    elif event_type == "tool_execution_end":
                        inflight.touch()
                        # Accumulate artifacts from completed tool calls so we
                        # can derive live_widgets on interrupt messages.
                        if event.tool_name:
                            from app.ai.tool_execution import build_tool_artifact

                            output = event.data.get("output")
                            error = event.data.get("error")
                            stream_tool_artifacts.append(
                                build_tool_artifact(
                                    tool_call_id=event.tool_call_id,
                                    tool_name=event.tool_name or "unknown",
                                    tool_args=(
                                        stream_tool_args_by_id.get(str(event.tool_call_id))
                                        if event.tool_call_id is not None
                                        else None
                                    ),
                                    output_text=str(output) if output is not None else None,
                                    error=str(error) if error else None,
                                    render=event.data.get("render"),
                                )
                            )
                        yield event

                    elif event_type == "rich_items":
                        # Preserve safe progressive rich-item upserts for
                        # clients that render marker-positioned output live.
                        inflight.touch()
                        yield event

                    elif event_type == "interrupt":
                        # Yield interrupt event - workflow paused for human approval
                        interrupt_response = event.data.get("interrupt")
                        interrupt_id = (
                            interrupt_response.get("interrupt_id") if interrupt_response else None
                        )
                        self._handle_redis_interrupt_storage(
                            message_create_data.conversation_id,
                            interrupt_id,
                            interrupt_response,
                        )
                        self._set_plan_lifecycle(
                            message_create_data.conversation_id,
                            resolved_user_id,
                            PlanLifecycle.paused,
                        )

                        interrupt_thread_id = event.data.get("thread_id") or str(
                            message_create_data.conversation_id
                        )
                        yield make_event(
                            "interrupt",
                            sequence=event.sequence,
                            conversation_id=str(message_create_data.conversation_id),
                            message_id=str(bot_message_id),
                            data={
                                "thread_id": interrupt_thread_id,
                                "next": event.data.get("next"),
                                "pending_tool_calls": event.data.get("pending_tool_calls"),
                                "interrupt": interrupt_response,
                                "message": self._persist_interrupt_bot_message(
                                    conversation_id=message_create_data.conversation_id,
                                    interrupt_payload=interrupt_response,
                                    sanitized_persona=sanitized_persona,
                                    pending_tool_calls=event.data.get("pending_tool_calls"),
                                    thread_id=interrupt_thread_id,
                                    next_nodes=event.data.get("next"),
                                    user_id=resolved_user_id,
                                    message_id=bot_message_id,
                                    tool_artifacts=stream_tool_artifacts or None,
                                    selected_agent=inflight.selected_agent,
                                    custom_agents=workflow_request.custom_agents,
                                ).model_dump(mode="json"),
                            },
                        )
                        # Workflow is paused - don't create a bot message yet.
                        # Keep the entry as a paused lock token (carrying the
                        # resolved selected_agent) so a custom agent cannot be
                        # edited/deleted/detached while this run can still resume.
                        _cancel_title_task()
                        inflight.resolve()
                        registry.mark_paused(user_message_id)
                        return

                    elif event_type == "complete":
                        # Store final response
                        bot_response = event.data.get("response")
                        break

                    elif event_type == "error":
                        # Handle error
                        bot_response = event.data.get("response")
                        break

                    else:
                        # state_snapshot (legacy node_complete / continuation
                        # markers), subagent lifecycle, and other canonical
                        # events pass through without breaking.
                        inflight.touch()
                        yield event

                # ---- Handle cancellation after the loop exits ----
                if inflight.is_cancelled:
                    _cancel_title_task()
                    partial = inflight.partial_text.strip()
                    if partial:
                        partial = fix_markdown_code_blocks(partial)
                        metadata = {
                            "stopped": True,
                            "partial": True,
                            "stop_reason": "user_requested",
                            "persona_used": sanitized_persona,
                            "reply_to_user_message_id": str(user_message_id),
                        }
                        self._attach_selected_agent_metadata(
                            metadata,
                            inflight.selected_agent,
                            workflow_request.custom_agents,
                        )
                        bot_message = await self._acreate_bot_response_message(
                            conversation_id=message_create_data.conversation_id,
                            content=partial,
                            metadata=metadata,
                            message_id=bot_message_id,
                        )
                        inflight.resolve(bot_message.model_dump(mode="json"))
                    else:
                        inflight.resolve(None)
                    registry.remove(user_message_id)
                    return

                _merge_stream_tool_artifacts_into_response(bot_response, stream_tool_artifacts)

                bot_message = await self._persist_completed_workflow_response(
                    conversation_id=message_create_data.conversation_id,
                    user_id=resolved_user_id,
                    bot_response=bot_response,
                    sanitized_persona=sanitized_persona,
                    workflow_request=workflow_request,
                    message_id=bot_message_id,
                    reply_to_user_message_id=user_message_id,
                    suggestion_source_message=message_create_data.content,
                )
                bot_message_persisted = True
                await self._compact_checkpoint_after_persist(
                    conversation_id=message_create_data.conversation_id,
                    workflow_request=workflow_request,
                )

                # Resolve the inflight future with the final message
                inflight.resolve(bot_message.model_dump(mode="json"))
                registry.remove(user_message_id)

                # Emit the title update BEFORE the terminal completion so the
                # Streamlit SSE client (which stops reading after `complete`)
                # still receives it.
                title_event = None
                if title_task:
                    generated_title = await title_task
                    if generated_title:
                        title_event = make_event(
                            "title_updated",
                            sequence=_next_sequence(),
                            conversation_id=str(message_create_data.conversation_id),
                            data={
                                "title": generated_title,
                                "conversation_id": str(message_create_data.conversation_id),
                            },
                        )

                if title_event:
                    yield title_event

                # Yield final completion event with full message
                yield make_event(
                    "complete",
                    sequence=_next_sequence(),
                    conversation_id=str(message_create_data.conversation_id),
                    message_id=str(bot_message.id),
                    data={"message": bot_message.model_dump(mode="json")},
                )

            except (asyncio.CancelledError, GeneratorExit):
                # Cancellation / disconnect: persist partial text if available,
                # do NOT create an error message.
                _cancel_title_task()
                if not bot_message_persisted:
                    partial = inflight.partial_text.strip()
                    if partial:
                        partial = fix_markdown_code_blocks(partial)
                        metadata = {
                            "stopped": True,
                            "partial": True,
                            "stop_reason": "disconnect",
                            "persona_used": sanitized_persona,
                            "reply_to_user_message_id": str(user_message_id),
                        }
                        self._attach_selected_agent_metadata(
                            metadata,
                            inflight.selected_agent,
                            workflow_request.custom_agents,
                        )
                        bot_msg = self._create_bot_response_message(
                            conversation_id=message_create_data.conversation_id,
                            content=partial,
                            metadata=metadata,
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

                yield make_event(
                    "error",
                    sequence=_next_sequence(),
                    conversation_id=str(message_create_data.conversation_id),
                    message_id=str(bot_message_id),
                    data={
                        "error": str(exc),
                        "message": error_message.model_dump(mode="json"),
                    },
                )
                _cancel_title_task()

    def _validate_and_claim_interrupt_resume(
        self,
        *,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        interrupt_id: str | None,
        device_id: UUID | None,
        decisions: list[InterruptDecision] | None = None,
    ) -> Any:
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)

        if not interrupt_id:
            raise CustomHTTPException(
                # Starlette's symbolic name differs across the supported range.
                status_code=422,
                detail="interruptId is required to resume a durable approval request.",
                error_code="INTERRUPT_ID_REQUIRED",
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
            if (
                device_id is not None
                and record.device_id is not None
                and record.device_id != device_id
            ):
                raise CustomHTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail="Device ID does not match the pending interrupt state.",
                    error_code="INTERRUPT_DEVICE_MISMATCH",
                )
            now = datetime.now(timezone.utc)
            if record.status == HITLInterruptStatus.EXPIRED or (
                record.status == HITLInterruptStatus.PENDING and record.expires_at <= now
            ):
                if record.status == HITLInterruptStatus.PENDING:
                    with contextlib.suppress(Exception):
                        self.hitl_interrupt_repository.mark_expired(interrupt_id)
                raise CustomHTTPException(
                    status_code=http_status.HTTP_410_GONE,
                    detail=(
                        "This approval request has expired. Please send a new message to try again."
                    ),
                    error_code="INTERRUPT_EXPIRED",
                )
            if record.status == HITLInterruptStatus.FAILED:
                raise CustomHTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail=(
                        "This approval cannot be resumed after a failed continuation. "
                        "Please send a new message."
                    ),
                    error_code="INTERRUPT_FAILED",
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

            self._validate_complete_interrupt_decisions(
                record=record,
                decisions=decisions,
            )

            runtime_provenance = self._get_runtime_validation_provenance(
                record,
                decisions=decisions,
            )
            if runtime_provenance:
                active_session = None
                if record.device_id is not None:
                    active_session = ClientDeviceService.lookup_active_session(record.device_id)

                if active_session is None or active_session.user_id != record.user_id:
                    self._expire_interrupt_for_scope_change(
                        interrupt_id=interrupt_id,
                        resolution_source="runtime_unavailable",
                    )
                    raise CustomHTTPException(
                        status_code=http_status.HTTP_409_CONFLICT,
                        detail=(
                            "The client device session for this approval is no longer available. "
                            "Please send a new message from the active device."
                        ),
                        error_code="INTERRUPT_RUNTIME_UNAVAILABLE",
                    )

                catalog_tools = (
                    active_session.tool_catalog.get("tools", [])
                    if isinstance(active_session.tool_catalog, dict)
                    else []
                )
                catalog_by_qid = {
                    str(entry.get("qualified_id")): entry
                    for entry in catalog_tools
                    if entry.get("qualified_id")
                }
                catalog_instance_ids = {
                    str(entry.get("tool_instance_id"))
                    for entry in catalog_tools
                    if entry.get("tool_instance_id")
                }

                for _provenance_key, provenance in runtime_provenance:
                    expected_session_id = provenance.get("session_id")
                    if expected_session_id not in (None, "") and active_session.session_id != str(
                        expected_session_id
                    ):
                        self._expire_interrupt_for_scope_change(
                            interrupt_id=interrupt_id,
                            resolution_source="session_changed",
                        )
                        raise CustomHTTPException(
                            status_code=http_status.HTTP_409_CONFLICT,
                            detail=(
                                "The client device session changed after this approval "
                                "was created. Please send a new message from the active device."
                            ),
                            error_code="INTERRUPT_SESSION_MISMATCH",
                        )

                    expected_catalog_version = provenance.get("catalog_version")
                    if (
                        expected_catalog_version is not None
                        and active_session.tool_catalog_version != int(expected_catalog_version)
                    ):
                        self._expire_interrupt_for_scope_change(
                            interrupt_id=interrupt_id,
                            resolution_source="catalog_changed",
                        )
                        raise CustomHTTPException(
                            status_code=http_status.HTTP_409_CONFLICT,
                            detail=(
                                "The client device tool catalog changed after this approval "
                                "was created. Please search for the tool again and retry."
                            ),
                            error_code="INTERRUPT_CATALOG_MISMATCH",
                        )

                    expected_qualified_id = str(provenance.get("qualified_tool_id") or "").strip()
                    expected_tool_instance_id = str(
                        provenance.get("tool_instance_id") or ""
                    ).strip()
                    if expected_qualified_id:
                        catalog_entry = catalog_by_qid.get(expected_qualified_id)
                        if catalog_entry is None:
                            self._expire_interrupt_for_scope_change(
                                interrupt_id=interrupt_id,
                                resolution_source="tool_unavailable",
                            )
                            raise CustomHTTPException(
                                status_code=http_status.HTTP_409_CONFLICT,
                                detail=(
                                    "A client-local tool in this approval is no longer "
                                    "available on the active device. Please search again "
                                    "and retry."
                                ),
                                error_code="INTERRUPT_TOOL_UNAVAILABLE",
                            )

                        current_tool_instance_id = str(
                            catalog_entry.get("tool_instance_id") or ""
                        ).strip()
                        if (
                            expected_tool_instance_id
                            and current_tool_instance_id
                            and expected_tool_instance_id != current_tool_instance_id
                        ):
                            self._expire_interrupt_for_scope_change(
                                interrupt_id=interrupt_id,
                                resolution_source="tool_instance_changed",
                            )
                            raise CustomHTTPException(
                                status_code=http_status.HTTP_409_CONFLICT,
                                detail=(
                                    "A client-local tool capability changed after this "
                                    "approval was created. Please search for the tool "
                                    "again and retry."
                                ),
                                error_code="INTERRUPT_TOOL_INSTANCE_MISMATCH",
                            )
                    elif (
                        expected_tool_instance_id
                        and expected_tool_instance_id not in catalog_instance_ids
                    ):
                        self._expire_interrupt_for_scope_change(
                            interrupt_id=interrupt_id,
                            resolution_source="tool_instance_changed",
                        )
                        raise CustomHTTPException(
                            status_code=http_status.HTTP_409_CONFLICT,
                            detail=(
                                "A client-local tool capability changed after this "
                                "approval was created. Please search for the tool "
                                "again and retry."
                            ),
                            error_code="INTERRUPT_TOOL_INSTANCE_MISMATCH",
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
                    stored_time = datetime.fromisoformat(stored_timestamp.decode("utf-8"))
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
        user_id: UUID | None,
        decisions: list[InterruptDecision],
        interrupt_id: str | None,
        fetched_interrupt_record: Any = None,
    ) -> None:
        if not self.tool_approval_repository or not user_id:
            return

        stored_original_args: dict[str, Any] = {}
        stored_provenance: dict[str, dict[str, Any]] = {}
        if interrupt_id:
            try:
                if fetched_interrupt_record and fetched_interrupt_record.action_requests_json:
                    for req in fetched_interrupt_record.action_requests_json:
                        if isinstance(req, dict):
                            key = req.get("tool_call_id") or req.get("task_id")
                            if key:
                                stored_original_args[key] = req.get("args") or {}
                            action = req.get("action")
                            if action and action not in stored_original_args:
                                stored_original_args[action] = req.get("args") or {}
                stored_provenance = self._normalize_tool_provenance_map(
                    getattr(fetched_interrupt_record, "interrupt_metadata_json", None)
                )
            except Exception:
                pass

        decision_type_map = {
            InterruptDecisionType.APPROVE: DecisionType.ACCEPT,
            InterruptDecisionType.EDIT: DecisionType.EDIT,
            InterruptDecisionType.REJECT: DecisionType.REJECT,
            InterruptDecisionType.RESPOND: DecisionType.RESPOND,
        }
        for decision in decisions:
            try:
                is_edit = decision.type == InterruptDecisionType.EDIT
                decision_id = resolve_interrupt_decision_id(decision)
                orig_key = decision_id or decision.action or ""
                original_args = (
                    stored_original_args.get(orig_key)
                    or stored_original_args.get(decision.action or "")
                    or {}
                )
                provenance = (
                    stored_provenance.get(orig_key)
                    or stored_provenance.get(decision.action or "")
                    or {}
                )
                approval_device_id = None
                raw_device_id = provenance.get("device_id")
                if raw_device_id:
                    with contextlib.suppress(Exception):
                        approval_device_id = UUID(str(raw_device_id))
                approval_data = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "interrupt_id": interrupt_id or "unknown",
                    "tool_name": decision.action or "unknown",
                    "tool_call_id": decision_id or "unknown",
                    "original_args": original_args,
                    "modified_args": decision.args if is_edit else None,
                    "decision": decision_type_map.get(decision.type, DecisionType.REJECT),
                    "device_id": approval_device_id,
                    "tool_origin": provenance.get("tool_origin"),
                    "server_name": provenance.get("server_name"),
                    "qualified_tool_id": provenance.get("qualified_tool_id"),
                    "session_id": provenance.get("session_id"),
                    "catalog_version": provenance.get("catalog_version"),
                    "tool_instance_id": provenance.get("tool_instance_id"),
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
        decisions: list[InterruptDecision],
        interrupt_id: str | None = None,
        device_id: UUID | None = None,
        bot_message_id: UUID | None = None,
        inline_rich_response_v1: bool = False,
    ):
        fetched_interrupt_record = self._validate_and_claim_interrupt_resume(
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id=interrupt_id,
            device_id=device_id,
            decisions=decisions,
        )

        partial_text = ""
        bot_message_persisted = False
        resume_tool_artifacts: list[dict[str, Any]] = []
        resume_tool_args_by_id: dict[str, Any] = {}
        resume_selected_agent: str | None = None

        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        try:
            user_id, persona = self._get_conversation_context(conversation_id, user_id)
            sanitized_persona = sanitize_persona(persona)

            # Reload/validate the custom-agent map before resuming.
            self._revalidate_resume_custom_agent(user_id, conversation_id)

            self._audit_interrupt_resume_decisions(
                conversation_id=conversation_id,
                user_id=user_id,
                decisions=decisions,
                interrupt_id=interrupt_id,
                fetched_interrupt_record=fetched_interrupt_record,
            )
            resume_custom_agents = self._resolve_custom_agents_state(user_id, conversation_id)

            async for raw_event in self.ai_service.resume_interrupted_execution_stream(
                thread_id=thread_id,
                decisions=decisions,
                inline_rich_response_v1=inline_rich_response_v1,
                user_id=user_id,
                conversation_id=conversation_id,
            ):
                event = _service_event_from_ai_event(raw_event, sequence=_next_sequence())
                event_type = event.type

                if event_type == "agent_selected":
                    resume_selected_agent = event.agent or event.data.get("agent")
                    yield self._agent_selected_event(
                        resume_selected_agent,
                        resume_custom_agents,
                        sequence=event.sequence,
                    )

                elif event_type == "message_delta":
                    partial_text += event.data.get("text", "")
                    yield event

                elif event_type == "reasoning_delta":
                    yield event

                elif event_type == "tool_call_available":
                    if event.tool_call_id is not None:
                        resume_tool_args_by_id[str(event.tool_call_id)] = event.data.get("args")
                    yield event

                elif event_type == "tool_execution_end":
                    if event.tool_name:
                        from app.ai.tool_execution import build_tool_artifact

                        output = event.data.get("output")
                        error = event.data.get("error")
                        resume_tool_artifacts.append(
                            build_tool_artifact(
                                tool_call_id=event.tool_call_id,
                                tool_name=event.tool_name or "unknown",
                                tool_args=(
                                    resume_tool_args_by_id.get(str(event.tool_call_id))
                                    if event.tool_call_id is not None
                                    else None
                                ),
                                output_text=str(output) if output is not None else None,
                                error=str(error) if error else None,
                                render=event.data.get("render"),
                            )
                        )
                    yield event

                elif event_type == "rich_items":
                    # Resume streams use the same progressive rich-response
                    # contract as first-pass response generation.
                    yield event

                elif event_type == "interrupt":
                    interrupt_response = event.data.get("interrupt")
                    normalized_interrupt = self._normalize_nested_interrupt_payload(
                        interrupt_response
                    )
                    next_interrupt_id = (
                        interrupt_response.get("interrupt_id")
                        if isinstance(interrupt_response, dict)
                        else None
                    )
                    try:
                        persisted = self._persist_interrupt_bot_message(
                            conversation_id=conversation_id,
                            interrupt_payload=normalized_interrupt,
                            sanitized_persona=sanitized_persona,
                            pending_tool_calls=event.data.get("pending_tool_calls"),
                            thread_id=event.data.get("thread_id") or thread_id,
                            next_nodes=event.data.get("next"),
                            user_id=user_id,
                            message_id=bot_message_id,
                            tool_artifacts=resume_tool_artifacts or None,
                            selected_agent=resume_selected_agent,
                            custom_agents=resume_custom_agents,
                            require_durable_interrupt=True,
                        )
                    except Exception as exc:
                        self._clear_redis_interrupt(conversation_id, interrupt_id)
                        get_generation_registry().clear_paused_for_conversation(
                            user_id, conversation_id
                        )
                        self._mark_claimed_interrupt_failed(interrupt_id, "stream_exception")
                        error_message = self._create_bot_response_message(
                            conversation_id=conversation_id,
                            content=f"Error generating response: {str(exc)}",
                            metadata={"error": str(exc)},
                            message_id=None,
                        )
                        bot_message_persisted = True
                        yield make_event(
                            "error",
                            sequence=_next_sequence(),
                            conversation_id=str(conversation_id),
                            message_id=str(error_message.id),
                            data={
                                "error": str(exc),
                                "message": error_message.model_dump(mode="json"),
                                "error_code": "INTERRUPT_FAILED",
                                "status_code": 500,
                            },
                        )
                        return

                    self._clear_redis_interrupt(conversation_id, interrupt_id)
                    if self.hitl_interrupt_repository and interrupt_id:
                        with contextlib.suppress(Exception):
                            self.hitl_interrupt_repository.mark_resolved(interrupt_id)
                    self._handle_redis_interrupt_storage(
                        conversation_id,
                        next_interrupt_id,
                        interrupt_response,
                    )
                    self._set_plan_lifecycle(
                        conversation_id,
                        user_id,
                        PlanLifecycle.paused,
                    )
                    bot_message_persisted = True

                    yield make_event(
                        "interrupt",
                        sequence=event.sequence,
                        conversation_id=str(conversation_id),
                        message_id=str(bot_message_id) if bot_message_id else None,
                        data={
                            "thread_id": event.data.get("thread_id") or thread_id,
                            "next": event.data.get("next"),
                            "pending_tool_calls": event.data.get("pending_tool_calls"),
                            "interrupt": (
                                normalized_interrupt.model_dump(mode="json")
                                if isinstance(normalized_interrupt, InterruptResponse)
                                else normalized_interrupt
                            ),
                            "message": persisted.model_dump(mode="json"),
                        },
                    )
                    return

                elif event_type == "complete":
                    bot_response = event.data.get("response")
                    _merge_stream_tool_artifacts_into_response(
                        bot_response,
                        resume_tool_artifacts,
                    )

                    bot_message = await self._persist_completed_workflow_response(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        bot_response=bot_response,
                        sanitized_persona=sanitized_persona,
                        workflow_request=None,
                        message_id=bot_message_id,
                        fallback_content=ERROR_RESPONSE_AFTER_RESUME,
                    )
                    bot_message_persisted = True
                    self._clear_redis_interrupt(conversation_id, interrupt_id)
                    if self.hitl_interrupt_repository and interrupt_id:
                        with contextlib.suppress(Exception):
                            self.hitl_interrupt_repository.mark_resolved(interrupt_id)
                    await self._compact_checkpoint_after_persist(thread_id=thread_id)

                    # Resume resolved the paused run — release its lock token.
                    get_generation_registry().clear_paused_for_conversation(
                        user_id, conversation_id
                    )

                    yield make_event(
                        "complete",
                        sequence=_next_sequence(),
                        conversation_id=str(conversation_id),
                        message_id=str(bot_message.id),
                        data={"message": bot_message.model_dump(mode="json")},
                    )
                    return

                elif event_type == "error":
                    self._clear_redis_interrupt(conversation_id, interrupt_id)
                    get_generation_registry().clear_paused_for_conversation(
                        user_id, conversation_id
                    )
                    self._mark_claimed_interrupt_failed(interrupt_id, "stream_error")

                    error_msg = event.data.get("error", UNKNOWN_ERROR)
                    error_message = await self._acreate_bot_response_message(
                        conversation_id=conversation_id,
                        content=f"Error generating response: {error_msg}",
                        metadata={"error": error_msg},
                        message_id=bot_message_id,
                    )
                    bot_message_persisted = True

                    yield make_event(
                        "error",
                        sequence=_next_sequence(),
                        conversation_id=str(conversation_id),
                        message_id=str(bot_message_id) if bot_message_id else None,
                        data={
                            "error": error_msg,
                            "message": error_message.model_dump(mode="json"),
                            "error_code": "INTERRUPT_FAILED",
                            "status_code": 500,
                        },
                    )
                    return

                else:
                    # state_snapshot (legacy node_complete / continuation
                    # markers), subagent lifecycle, and other canonical
                    # events pass through without breaking.
                    yield event

            self._clear_redis_interrupt(conversation_id, interrupt_id)

            if not bot_message_persisted:
                get_generation_registry().clear_paused_for_conversation(user_id, conversation_id)
                self._mark_claimed_interrupt_failed(interrupt_id, "stream_incomplete")
                fallback_message = await self._acreate_bot_response_message(
                    conversation_id=conversation_id,
                    content=ERROR_RESPONSE_AFTER_RESUME,
                    metadata={"error": ERROR_RESPONSE_AFTER_RESUME},
                    message_id=bot_message_id,
                )
                yield make_event(
                    "error",
                    sequence=_next_sequence(),
                    conversation_id=str(conversation_id),
                    message_id=str(bot_message_id) if bot_message_id else None,
                    data={
                        "error": ERROR_RESPONSE_AFTER_RESUME,
                        "message": fallback_message.model_dump(mode="json"),
                        "error_code": "INTERRUPT_FAILED",
                        "status_code": 500,
                    },
                )

        except (asyncio.CancelledError, GeneratorExit):
            get_generation_registry().clear_paused_for_conversation(user_id, conversation_id)
            self._mark_claimed_interrupt_failed(interrupt_id, "client_disconnect")
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
            get_generation_registry().clear_paused_for_conversation(user_id, conversation_id)
            self._mark_claimed_interrupt_failed(interrupt_id, "stream_exception")
            if bot_message_persisted:
                return

            error_message = self._create_bot_response_message(
                conversation_id=conversation_id,
                content=f"Error generating response: {str(exc)}",
                metadata={"error": str(exc)},
                message_id=bot_message_id,
            )

            yield make_event(
                "error",
                sequence=_next_sequence(),
                conversation_id=str(conversation_id),
                message_id=str(bot_message_id) if bot_message_id else None,
                data={
                    "error": str(exc),
                    "message": error_message.model_dump(mode="json"),
                    "error_code": "INTERRUPT_FAILED",
                    "status_code": 500,
                },
            )

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
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)

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
        order_by: str | None = None,
        order_direction: str = "asc",
        include_feedback: bool = False,
    ) -> Paginator[MessageRead]:
        # Validate pagination parameters
        validate_pagination_params(page, limit)

        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)
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
        return Paginator.create(message_reads, paginated_messages.meta.total, page, limit)

    def get_user_messages(
        self,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
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
        return Paginator.create(message_reads, paginated_messages.meta.total, page, limit)

    def update_message(
        self,
        message_id: UUID,
        user_id: UUID,
        message_update_data: MessageUpdate,
    ) -> MessageRead:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        message_entity = self.repository.get_by_id(message_id)
        updated_message = self.repository.update(message_entity.id, message_update_data)
        if updated_message is not None:
            with contextlib.suppress(Exception):
                self.ai_service.invalidate_history_cache(str(updated_message.conversation_id))
        return MessageRead.model_validate(updated_message)

    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        self.message_validation_utils.validate_message_access(user_id, message_id)
        # Capture conversation id BEFORE deletion so we can invalidate the
        # prompt-history cache for the affected conversation even after the
        # row is soft-deleted.
        conversation_id: str | None = None
        with contextlib.suppress(Exception):
            existing = self.repository.get_by_id(message_id)
            if existing is not None and getattr(existing, "conversation_id", None):
                conversation_id = str(existing.conversation_id)
        deleted = self.repository.delete(message_id)
        if deleted and conversation_id is not None:
            with contextlib.suppress(Exception):
                self.ai_service.invalidate_history_cache(conversation_id)
        return deleted

    async def resume_workflow(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_input: str | None = None,
    ) -> MessageRead:
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)

        bot_response = await self.ai_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=user_input,
        )

        bot_response_content = extract_response_content(bot_response, NO_RESPONSE_GENERATED)
        bot_response_content = finalize_article_content(bot_response, bot_response_content)

        bot_metadata = build_bot_metadata(bot_response)
        bot_response_content, bot_metadata = await self._externalize_remote_rich_images(
            bot_response_content,
            bot_metadata,
            conversation_id,
            user_id,
        )
        if self._sync_response_plan_state(
            conversation_id=conversation_id,
            user_id=user_id,
            bot_response=bot_response,
            current_lifecycle=None,
        ):
            bot_metadata["todos_synced"] = True

        bot_message = await self._acreate_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=bot_metadata,
        )

        await self._compact_checkpoint_after_persist(
            conversation_id=conversation_id,
            thread_id=str(conversation_id),
        )

        return bot_message

    @staticmethod
    def _build_task_context_dict(task: Any | None) -> dict[str, Any] | None:
        if not task:
            return None
        return {
            "id": str(task.id),
            "description": task.description,
            "order": getattr(task, "task_order", getattr(task, "order", 0)),
            "status": (task.status.value if hasattr(task.status, "value") else str(task.status)),
        }

    def _externalize_generated_images(
        self, metadata: dict[str, Any], conversation_id: UUID, user_id: UUID
    ) -> None:
        """Move inline base64 in ``metadata['images']`` into storage, in place.
        No-op when the storage service is absent or there are no images."""
        if getattr(self, "chat_image_service", None) is None or not user_id:
            return
        images = metadata.get("images")
        if not images:
            return

        def _store(*, mime: str, data_b64: str, name: str) -> dict:
            return self.chat_image_service.store(
                conversation_id=conversation_id,
                user_id=user_id,
                mime=mime,
                data_b64=data_b64,
                name=name,
            )

        metadata["images"] = externalize_metadata_images(images, store=_store)

    async def _externalize_remote_rich_images(
        self,
        content: str,
        metadata: dict[str, Any],
        conversation_id: UUID,
        user_id: UUID | None,
    ) -> tuple[str, dict[str, Any]]:
        """Replace selected remote image URLs with protected owned references.

        Registration persists metadata only and performs no upstream request.
        Any registration/policy failure removes just that optional visual and
        its marker so assistant text persistence remains successful.
        """
        service = getattr(self, "web_image_service", None)
        rich_items = metadata.get("rich_items") if isinstance(metadata, dict) else None
        if service is None or user_id is None or not isinstance(rich_items, list):
            # Still record final selection on this path. A deployment without a
            # configured web-image service persists its selected images as-is,
            # and silently reporting nothing would look like "no images were
            # ever selected" rather than "externalization did not run".
            if isinstance(rich_items, list):
                self._record_final_image_selection(rich_items)
            return content, metadata

        updated = deepcopy(metadata)
        kept_items: list[dict[str, Any]] = []
        updated_content = content
        for item in updated.get("rich_items") or []:
            if not isinstance(item, dict):
                kept_items.append(item)
                continue

            if item.get("type") == RichItemType.image_group.value:
                if await self._externalize_group_cells(
                    item, conversation_id=conversation_id, user_id=user_id
                ):
                    kept_items.append(item)
                else:
                    updated_content = remove_inline_rich_reference(
                        updated_content, str(item.get("id") or "")
                    )
                continue

            if item.get("type") != RichItemType.image.value:
                kept_items.append(item)
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                kept_items.append(item)
                continue
            if payload.get("data"):
                kept_items.append(item)
                continue

            reference_url = await self._register_web_image_url(
                payload.get("url"),
                expected_mime=payload.get("mime_type"),
                provider=self._provider_of(item),
                conversation_id=conversation_id,
                user_id=user_id,
            )
            if reference_url is None:
                updated_content = remove_inline_rich_reference(
                    updated_content, str(item.get("id") or "")
                )
                continue
            payload["url"] = reference_url
            kept_items.append(item)

        updated["rich_items"] = kept_items
        updated["rich_reference_warnings"] = validate_rich_references(
            updated_content,
            kept_items,
        )
        self._record_final_image_selection(kept_items)
        return updated_content, updated

    async def _externalize_group_cells(
        self,
        item: dict[str, Any],
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Rewrite an image group's cell URLs in place to owned references.

        Cells that cannot be registered are dropped. Returns whether the group
        item should be kept: True when at least one cell survived, and also when
        the group carries no usable cell list at all (nothing to externalize, so
        the item is left exactly as it was). False means every cell failed and
        the caller must drop the item and remove its marker.
        """
        payload = item.get("payload")
        cells = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(cells, list):
            return True
        kept_cells: list[dict[str, Any]] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            reference_url = await self._register_web_image_url(
                cell.get("url"),
                expected_mime=cell.get("mime_type"),
                provider=self._provider_of(item),
                conversation_id=conversation_id,
                user_id=user_id,
            )
            if reference_url is None:
                continue
            cell["url"] = reference_url
            kept_cells.append(cell)
        if not kept_cells:
            return False
        payload["items"] = kept_cells
        return True

    def _record_final_image_selection(self, kept_items: list[dict[str, Any]]) -> None:
        """Count image items surviving finalization. Never raises: telemetry
        must not fail an answer that is otherwise ready to persist."""
        with contextlib.suppress(Exception):
            surviving: dict[str, int] = {}
            for kept in kept_items:
                if not isinstance(kept, dict):
                    continue
                if kept.get("type") not in {
                    RichItemType.image.value,
                    RichItemType.image_group.value,
                }:
                    continue
                provider = self._provider_of(kept)
                surviving[provider] = surviving.get(provider, 0) + 1
            for provider, count in surviving.items():
                rich_image_metrics.record_final_selection(provider=provider, count=count)

    @staticmethod
    def _provider_of(item: dict[str, Any]) -> str:
        return provenance_provider(item)

    async def _register_web_image_url(
        self,
        raw_url: Any,
        *,
        expected_mime: Any,
        provider: str,
        conversation_id: UUID,
        user_id: UUID,
    ) -> str | None:
        """Return a protected reference URL, or None when it cannot be made.

        Registration is metadata-only and performs no upstream request.
        """
        if not isinstance(raw_url, str) or not raw_url.strip():
            return None
        image_url = raw_url.strip()
        if image_url.startswith(PROTECTED_IMAGE_URL_PREFIXES):
            with contextlib.suppress(Exception):
                rich_image_metrics.record_registration(provider=provider, outcome="reused")
            return image_url
        if urlsplit(image_url).scheme.lower() != "https":
            logging.warning("Web image reference skipped code=web_image_reference_failed")
            with contextlib.suppress(Exception):
                rich_image_metrics.record_registration(provider=provider, outcome="skipped_scheme")
            return None
        # Visual verification already fetched, validated and decoded this image
        # earlier in the turn. Persisting those bytes makes an approved image
        # one fetch rather than two, and removes the window where an image
        # passes every check, is placed, and then dies at render. Nothing here
        # depends on it: no hand-off means no bytes, and the upstream fetch
        # path runs exactly as before.
        cached = None
        with contextlib.suppress(Exception):
            cached = take_verified_bytes(str(conversation_id), image_url)
        try:
            reference = await self.web_image_service.register(
                conversation_id=conversation_id,
                user_id=user_id,
                upstream_url=image_url,
                expected_mime=expected_mime,
                provider=provider,
                cached=cached,
            )
            reference_id = (
                reference.get("id")
                if isinstance(reference, dict)
                else getattr(reference, "id", None)
            )
            if reference_id is None:
                raise ValueError("missing reference id")
        except Exception:
            logging.warning("Web image reference skipped code=web_image_reference_failed")
            with contextlib.suppress(Exception):
                rich_image_metrics.record_registration(provider=provider, outcome="failed")
            return None
        with contextlib.suppress(Exception):
            rich_image_metrics.record_registration(provider=provider, outcome="registered")
        return f"/web-images/{reference_id}"

    def _externalize_attachments_for_persist(
        self, message_create_data: MessageCreate, user_id: UUID
    ) -> list[dict] | None:
        """Replace inline base64 attachments with storage references for the
        persisted row. The current-turn model call still receives the original
        inline bytes; only the DB copy is externalized. Storage failures fall
        back to the inline attachment so a send is never blocked."""
        attachments = getattr(message_create_data, "attachments", None)
        if not attachments:
            return None
        if getattr(self, "chat_image_service", None) is None:
            return list(attachments)
        refs: list[dict] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            inline_b64 = att.get("data") or att.get("base64")
            if not inline_b64:
                refs.append(att)  # already a reference or remote URL
                continue
            try:
                ref = self.chat_image_service.store(
                    conversation_id=message_create_data.conversation_id,
                    user_id=user_id,
                    mime=att.get("mime") or att.get("mimeType") or "image/png",
                    data_b64=inline_b64,
                    name=att.get("name") or "image",
                )
                refs.append(ref)
            except Exception:
                logging.warning(
                    "Chat image externalization failed; keeping inline attachment "
                    "code=chat_image_store_failed"
                )
                refs.append(att)
        return refs

    @staticmethod
    def _extract_message_execution_inputs(
        message_create_data: MessageCreate,
    ) -> tuple[list | None, dict[str, Any] | None]:
        attachments = message_create_data.attachments or None
        model_request = (
            message_create_data.model_config_field
            if (
                isinstance(message_create_data.model_config_field, dict)
                and message_create_data.model_config_field
            )
            else None
        )
        return attachments, model_request

    async def _prepare_planning_context(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message_content: str,
        planning_mode_enabled: bool,
        plan_lifecycle: PlanLifecycle | str | None = None,
    ) -> WorkflowPlanningContext:
        result = WorkflowPlanningContext(
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=False,
            plan_lifecycle=self._coerce_plan_lifecycle(plan_lifecycle),
        )

        if not self.task_plan_service or not user_id:
            return result

        try:
            existing_tasks = await self.task_plan_service.aget_conversation_tasks(
                conversation_id, user_id, include_completed=True
            )
            result.has_existing_plan = len(existing_tasks) > 0

            # Create plan if planning mode enabled but no plan exists
            if planning_mode_enabled and not result.has_existing_plan:
                created_tasks = await self.task_plan_service.create_task_plan(
                    conversation_id, message_content, user_id
                )
                result.has_existing_plan = len(created_tasks) > 0
                if result.has_existing_plan:
                    existing_tasks = await self.task_plan_service.aget_conversation_tasks(
                        conversation_id, user_id, include_completed=True
                    )

                consume_metadata = getattr(
                    self.task_plan_service,
                    "consume_last_planning_runtime_metadata",
                    None,
                )
                if callable(consume_metadata):
                    runtime_metadata = consume_metadata()
                    rubric_metadata = runtime_metadata.get("planning_rubric")
                    if isinstance(rubric_metadata, dict):
                        result.rubric_metadata = rubric_metadata

            if result.has_existing_plan:
                result.planning_mode_enabled = True

            # Get current task for execution context
            current_task = await self.task_plan_service.aget_active_or_next_task(
                conversation_id, user_id
            )
            result.current_task = self._build_task_context_dict(current_task)

            # Convert existing tasks to dict for planning agent
            if existing_tasks:
                result.tasks = [
                    {
                        "id": str(task.id),
                        "description": task.description,
                        "status": (
                            task.status.value if hasattr(task.status, "value") else str(task.status)
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

    async def _build_user_message_workflow_request(
        self,
        *,
        message_create_data: MessageCreate,
        user_id: UUID | None,
        conversation: Any,
        user_message_id: UUID | None = None,
        assistant_message_id: UUID | None = None,
    ) -> tuple[UUID | None, str | None, WorkflowExecutionRequest]:
        resolved_user_id = user_id or (conversation.owner_id if conversation else None)
        sanitized_persona = sanitize_persona(conversation.persona_prompt if conversation else None)
        planning_context = await self._prepare_planning_context(
            conversation_id=message_create_data.conversation_id,
            user_id=resolved_user_id,
            message_content=message_create_data.content,
            planning_mode_enabled=conversation.planning_mode_enabled if conversation else False,
            plan_lifecycle=self._coerce_plan_lifecycle(
                getattr(conversation, "plan_lifecycle", None)
            ),
        )
        attachments, model_request = self._extract_message_execution_inputs(message_create_data)
        custom_agents_state = await self._aresolve_custom_agents_state(
            resolved_user_id, message_create_data.conversation_id
        )
        validated_device_id = self._validate_request_device_id(
            message_create_data.device_id, resolved_user_id
        )
        hitl_policy = self._resolve_hitl_policy(resolved_user_id, validated_device_id)
        request = WorkflowExecutionRequest(
            message=message_create_data.content,
            conversation_id=str(message_create_data.conversation_id),
            user_id=str(resolved_user_id) if resolved_user_id else None,
            device_id=validated_device_id,
            persona=sanitized_persona,
            attachments=attachments,
            model_request=model_request,
            planning=planning_context,
            custom_agents=custom_agents_state,
            user_message_id=str(user_message_id) if user_message_id else None,
            assistant_message_id=(str(assistant_message_id) if assistant_message_id else None),
            inline_rich_response_v1=bool(
                getattr(message_create_data, "inline_rich_response_v1", False)
            ),
            hitl_policy=hitl_policy,
        )
        return resolved_user_id, sanitized_persona, request

    @staticmethod
    def _validate_request_device_id(device_id: Any, user_id: Any) -> str | None:
        """Treat the request ``device_id`` as untrusted per-turn input.

        The device must belong to the requesting user and have an active
        client runtime session; anything else (foreign device, disconnected
        client, replayed id) is dropped so the turn binds no client tools
        instead of another device's tools.
        """
        if device_id is None:
            return None

        from app.ai.client_runtime_tools import get_active_client_runtime_session

        session = get_active_client_runtime_session(
            user_id=str(user_id) if user_id else None,
            device_id=str(device_id),
        )
        if session is None:
            logging.warning(
                "Dropping request device_id %s for user %s: "
                "no active client runtime session for this user/device",
                device_id,
                user_id,
            )
            return None
        return str(device_id)

    def _resolve_custom_agents_state(
        self, owner_id: UUID | None, conversation_id: UUID | None
    ) -> dict[str, Any]:
        """Resolve attached custom agents into the workflow ``custom_agents`` map.

        Best-effort: returns ``{}`` when no service is wired or none are
        attached, preserving all behavior for conversations without custom
        agents. Retained for the resume and non-streaming paths;
        :meth:`_aresolve_custom_agents_state` serves the streaming path.
        """
        service = getattr(self, "custom_agent_service", None)
        if service is None or not owner_id or not conversation_id:
            return {}
        try:
            return service.build_runtime_state(owner_id, conversation_id)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to resolve custom agents for conversation: %s", exc)
            return {}

    async def _aresolve_custom_agents_state(
        self, owner_id: UUID | None, conversation_id: UUID | None
    ) -> dict[str, Any]:
        """Async twin of :meth:`_resolve_custom_agents_state`."""
        service = getattr(self, "custom_agent_service", None)
        if service is None or not owner_id or not conversation_id:
            return {}
        try:
            return await service.abuild_runtime_state(owner_id, conversation_id)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to resolve custom agents for conversation: %s", exc)
            return {}

    def _resolve_hitl_policy(self, user_id, device_id: str | None) -> dict | None:
        """Resolve device-scoped editable HITL policy for this turn.

        Returns ``None`` when no repository is wired or no user is resolved, so
        legacy turns use global policy. A turn without a validated device gets
        no editable client rules. Loading errors fail the turn instead of
        silently dropping Require rules.
        """
        repo = getattr(self, "tool_approval_setting_repository", None)
        if repo is None or not user_id:
            return None
        try:
            from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled

            grouped = (
                repo.build_policy(user_id, UUID(str(device_id)))
                if device_id
                else {
                    "client_mcp": {"servers": {}, "tools": {}},
                    "client_skill": {"servers": {}, "tools": {}},
                }
            )
            return {
                "master_enabled": is_hitl_enabled(),
                "client_rules": grouped,
                "global_tools": list(get_tools_requiring_approval()),
            }
        except Exception as exc:
            logging.exception("Failed to resolve HITL policy")
            raise RuntimeError("Unable to load the user's HITL approval policy") from exc

    @staticmethod
    def _agent_selected_event(
        agent: str | None,
        custom_agents: dict[str, Any] | None,
        *,
        sequence: int,
    ) -> V3StreamEvent:
        """Build a canonical agent_selected event, adding ``agent_name`` for custom agents.

        Consumers that only read the raw ``agent`` field are unaffected.
        """
        data: dict[str, Any] = {"agent": agent}
        if isinstance(custom_agents, dict) and agent in custom_agents:
            entry = custom_agents.get(agent)
            name = entry.get("name") if isinstance(entry, dict) else None
            if name:
                data["agent_name"] = name
        return make_event("agent_selected", sequence=sequence, agent=agent, data=data)

    @staticmethod
    def _attach_selected_agent_metadata(
        metadata: dict[str, Any],
        selected_agent: str | None,
        custom_agents: dict[str, Any] | None,
    ) -> None:
        if not selected_agent:
            return

        from app.ai.agent_metadata import attach_agent_metadata

        attach_agent_metadata(
            metadata,
            response_agent_id=selected_agent,
            selected_agent_id=selected_agent,
            custom_agents=custom_agents,
        )

    def _revalidate_resume_custom_agent(self, owner_id: UUID | None, conversation_id: UUID) -> None:
        """Fail resume if a paused run's selected custom agent is gone.

        A paused HITL run carries its selected runtime agent. If that custom
        agent has since been deleted or detached, the run can no longer resume
        with the same identity — surface a clear conflict instead of silently
        running with stale config.
        """
        if getattr(self, "custom_agent_service", None) is None or not owner_id:
            return
        from app.ai.custom_agent_runtime import is_custom_runtime_id

        registry = get_generation_registry()
        paused = [e for e in registry.find_by_conversation(conversation_id) if e.paused]
        if not paused:
            return
        attached = set(self._resolve_custom_agents_state(owner_id, conversation_id).keys())
        for entry in paused:
            selected = entry.selected_agent
            if selected and is_custom_runtime_id(selected) and selected not in attached:
                raise CustomHTTPException(
                    status_code=409,
                    detail=(
                        "The custom agent for this paused conversation is no longer "
                        "attached and cannot be resumed."
                    ),
                    error_code="CUSTOM_AGENT_RESUME_CONFLICT",
                )

    async def _execute_user_message_workflow(
        self,
        *,
        workflow_request: WorkflowExecutionRequest,
        conversation_id: UUID,
        user_id: UUID | None,
    ) -> tuple[
        WorkflowResponse | None,
        dict[str, Any] | None,
    ]:
        """
        Execute one canonical workflow request and surface interrupts separately.
        """
        bot_response = await self.ai_service.execute_request(workflow_request)

        # Handle interrupts (HITL)
        if bot_response and bot_response.metadata and "interrupt" in bot_response.metadata:
            self._set_plan_lifecycle(
                conversation_id,
                user_id,
                PlanLifecycle.paused,
            )
            return (
                bot_response,
                bot_response.metadata["interrupt"],
            )

        return bot_response, None

    async def _persist_completed_workflow_response(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID | None,
        bot_response: WorkflowResponse | None,
        sanitized_persona: str | None,
        workflow_request: WorkflowExecutionRequest | None,
        message_id: UUID | None = None,
        reply_to_user_message_id: UUID | None = None,
        suggestion_source_message: str | None = None,
        fallback_content: str = NO_RESPONSE_GENERATED,
    ) -> MessageRead:
        """Persist the assistant message for a completed workflow turn.

        ``workflow_request`` is None on the resume path, which has no original
        request object — planning-context enrichment is skipped there.
        """
        bot_response_content = fix_markdown_code_blocks(
            extract_response_content(bot_response, fallback_content)
        )
        bot_response_content = finalize_article_content(bot_response, bot_response_content)
        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)
        self._externalize_generated_images(bot_metadata, conversation_id, user_id)
        bot_response_content, bot_metadata = await self._externalize_remote_rich_images(
            bot_response_content,
            bot_metadata,
            conversation_id,
            user_id,
        )
        if reply_to_user_message_id:
            bot_metadata["reply_to_user_message_id"] = str(reply_to_user_message_id)

        if self._sync_response_plan_state(
            conversation_id=conversation_id,
            user_id=user_id,
            bot_response=bot_response,
            current_lifecycle=(
                workflow_request.planning.plan_lifecycle if workflow_request else None
            ),
        ):
            bot_metadata["todos_synced"] = True

        if bot_response and bot_response.metadata.get("planning_budget_reached"):
            bot_metadata["execution_paused"] = True
            bot_metadata["execution_pause_reason"] = PauseReason.MAX_TASKS_REACHED.value
            bot_metadata["execution_pause_message"] = (
                "Completed a planning iteration. Send a message to continue."
            )

        if (
            workflow_request is not None
            and workflow_request.planning.planning_mode_enabled
            and self.task_plan_service
            and user_id
        ):
            try:
                next_task = self.task_plan_service.get_active_or_next_task(conversation_id, user_id)
                if next_task:
                    bot_metadata["next_task"] = {
                        "id": str(next_task.id),
                        "description": next_task.description,
                        "order": next_task.task_order,
                    }
            except Exception:
                pass

        if suggestion_source_message:
            await self._generate_and_add_suggestions(
                suggestion_source_message,
                bot_response_content,
                bot_metadata,
                user_id=user_id,
                conversation_id=conversation_id,
                request_message_id=reply_to_user_message_id,
            )

        bot_message = await self._acreate_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=bot_metadata,
            message_id=message_id,
        )

        return bot_message

    def _sync_todos_to_database(
        self,
        conversation_id: UUID,
        user_id: UUID | None,
        todos: list[dict[str, Any]],
        lifecycle: PlanLifecycle | None = None,
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
