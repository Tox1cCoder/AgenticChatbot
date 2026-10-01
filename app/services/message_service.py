from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlsplit
from uuid import UUID, uuid4

try:
    import redis
except ImportError:  # pragma: no cover - exercised in environments without redis installed
    redis = None

from fastapi import status as http_status

from app.ai.suggestion_generator import generate_follow_up_suggestions
from app.ai.utils import resolve_interrupt_decision_id
from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error, workflow_error_payload
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
from app.usage import UsageContext
from app.utils.text_processing import fix_markdown_code_blocks, sanitize_persona
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params

if TYPE_CHECKING:
    from app.interfaces.task_plan_service_interface import ITaskPlanService
    from app.services.conversation_turn_coordinator import ConversationTurnCoordinator


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


def _client_error_text(exc: Exception) -> str:
    """The text a failed turn shows the client, in its error event and message.

    A ``CustomHTTPException`` detail is written for the client (the API
    returns it verbatim). Any other exception's text can carry a provider
    error with a key in its URL, a SQL statement, or a filesystem path, so it
    is logged and only the exception type leaves the process, matching the
    graph's ``_stream_failure_event``.
    """
    if isinstance(exc, CustomHTTPException) and isinstance(exc.detail, str):
        return exc.detail
    logging.error("Message generation failed", exc_info=exc)
    return f"Response generation failed ({type(exc).__name__})."


def _is_blank(value: Any) -> bool:
    return value in (None, "")


def _paused_epoch(payload: dict[str, Any]) -> int | None:
    """The graph's epoch from a pause payload, or ``None`` if it names none.

    ``None`` leaves the row's epoch untouched rather than guessing one: a wrong
    epoch on the row is a Continue the pause node will refuse.
    """
    value = payload.get("execution_epoch")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _pending_action_counts(pending_requests: list[Any]) -> dict[str, int]:
    """How many pending requests share each action name.

    An action name addresses a request only when it is unique; two calls of
    the same tool must be told apart by id.
    """
    counts: dict[str, int] = {}
    for request in pending_requests:
        if isinstance(request, dict) and not _is_blank(request.get("action")):
            key = str(request["action"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def _pending_request_indexes(
    pending_requests: list[Any], action_counts: dict[str, int]
) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, request in enumerate(pending_requests):
        if not isinstance(request, dict):
            continue
        for id_key in ("tool_call_id", "task_id"):
            value = request.get(id_key)
            if not _is_blank(value):
                indexes[str(value)] = index
        action = request.get("action")
        if not _is_blank(action) and action_counts.get(str(action)) == 1:
            indexes[str(action)] = index
    return indexes


def _pending_request_key(request: dict[str, Any], action_counts: dict[str, int]) -> str | None:
    """The one key a complete decision set must cover for this request."""
    for id_key in ("tool_call_id", "task_id"):
        value = request.get(id_key)
        if not _is_blank(value):
            return str(value)
    action = request.get("action")
    if not _is_blank(action) and action_counts.get(str(action)) == 1:
        return str(action)
    return None


def _require_allowed_decision(request: dict[str, Any], decision: Any) -> None:
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
            detail=f"Decision '{normalized_type}' is not allowed for this pending tool call.",
            error_code="INTERRUPT_DECISION_NOT_ALLOWED",
        )


#: Stream events at which a durable stop check is worth a database read. Tool
#: and model boundaries, and the route landing -- the points where the turn is
#: about to spend something. Token deltas are excluded on purpose: a read per
#: token would put a query on the hot path of every answer.
_DURABLE_STOP_CHECKPOINTS = frozenset(
    {
        "agent_selected",
        "tool_call_available",
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "reasoning_start",
        "subagent_start",
        "subagent_end",
        "state_snapshot",
    }
)


class _DurableStopWatch:
    """Reads ``generations.status`` at boundaries, not per event.

    Why this exists at all: the in-process registry can only be signalled by a
    Stop that landed on *this* worker. The Redis broadcast reaches the others,
    but a signal is best-effort — a dropped subscriber, a restarted process, a
    publish that failed after the transition committed. The row is what is
    always true, so the worker asks it.

    Why it is throttled: correctness needs the check to happen *eventually*,
    not immediately, because the row cannot un-stop. So a minimum interval
    between reads costs a little latency and removes a per-token query from
    every answer the system produces.
    """

    def __init__(self, *, control: Any, generation: Any, user_id: UUID | None) -> None:
        self._control = control
        self._generation = generation
        self._user_id = user_id
        self._last_checked = 0.0
        self._settled = False

    @property
    def _interval(self) -> float:
        return max(0.0, float(getattr(settings, "generation_stop_poll_seconds", 2.0)))

    async def stop_requested(self, event_type: str) -> bool:
        """Whether the row says this turn should stop.

        ``False`` for every reason that is not a definite yes: no lifecycle
        service, an unreadable row, a read that failed. A stop that cannot be
        confirmed must not end a turn that is producing a good answer.
        """
        if self._settled or self._control is None or self._generation is None:
            return False
        if event_type not in _DURABLE_STOP_CHECKPOINTS:
            return False

        now = time.monotonic()
        if now - self._last_checked < self._interval:
            return False
        self._last_checked = now

        from app.models.generation import GenerationStatus

        try:
            snapshot = await self._control.aget_snapshot(
                generation_id=self._generation.generation_id,
                user_id=self._user_id,
                conversation_id=self._generation.conversation_id,
            )
        except Exception as exc:  # noqa: BLE001 - a failed read never stops a turn
            logging.debug("Durable stop check failed: %s", type(exc).__name__)
            return False

        if snapshot is None:
            return False
        if snapshot.status is GenerationStatus.STOP_REQUESTED:
            self._settled = True
            return True
        return False


def _interrupt_payload_dict(interrupt_payload: Any) -> dict[str, Any]:
    if isinstance(interrupt_payload, InterruptResponse):
        return interrupt_payload.model_dump(mode="json")
    if isinstance(interrupt_payload, dict):
        return interrupt_payload
    return {"raw": str(interrupt_payload)}


def _interrupt_metadata_of(interrupt_dict: dict[str, Any]) -> dict[str, Any]:
    metadata = interrupt_dict.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _interrupt_device_id(interrupt_metadata: dict[str, Any]) -> UUID | None:
    raw_device_id = interrupt_metadata.get("device_id")
    if not raw_device_id:
        return None
    try:
        return UUID(str(raw_device_id))
    except ValueError:
        return None


def _record_stream_tool_event(
    event: V3StreamEvent,
    tool_artifacts: list[dict[str, Any]],
    tool_args_by_id: dict[str, Any],
) -> None:
    """Accumulate a tool call's arguments and its finished artifact.

    The artifacts are what a paused turn's message derives ``live_widgets``
    from, and what a finished one merges into its response.
    """
    if event.type == "tool_call_available":
        if event.tool_call_id is not None:
            tool_args_by_id[str(event.tool_call_id)] = event.data.get("args")
        return
    if event.type != "tool_execution_end" or not event.tool_name:
        return

    from app.ai.tool_execution import build_tool_artifact

    output = event.data.get("output")
    error = event.data.get("error")
    tool_artifacts.append(
        build_tool_artifact(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name or "unknown",
            tool_args=(
                tool_args_by_id.get(str(event.tool_call_id))
                if event.tool_call_id is not None
                else None
            ),
            output_text=str(output) if output is not None else None,
            error=str(error) if error else None,
            render=event.data.get("render"),
        )
    )


@dataclass(frozen=True)
class _InterruptPersistContext:
    """What an approval message records beside the interrupt payload itself."""

    sanitized_persona: str | None = None
    pending_tool_calls: Any | None = None
    thread_id: str | None = None
    next_nodes: Any | None = None
    user_id: UUID | None = None
    message_id: UUID | None = None
    tool_artifacts: list[dict[str, Any]] | None = None
    active_agent_id: str | None = None
    custom_agents: dict[str, Any] | None = None
    #: A claimed nested resume needs a failed durable record to propagate.
    require_durable_interrupt: bool = False
    partial_text: str = ""


@dataclass
class _UserTurn:
    """One streamed user turn, shared by the handlers that each own a step of it."""

    conversation_id: UUID
    user_id: UUID
    user_message_id: UUID
    bot_message_id: UUID
    generation: Any
    registry: Any
    registry_key: Any
    inflight: Any
    resolved_user_id: UUID | None = None
    sanitized_persona: str | None = None
    workflow_request: WorkflowExecutionRequest | None = None
    title_task: asyncio.Task | None = None
    bot_response: Any = None
    bot_message_persisted: bool = False
    tool_artifacts: list[dict[str, Any]] = field(default_factory=list)
    tool_args_by_id: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0

    @property
    def owner_id(self) -> UUID:
        """Who lifecycle transitions are made as."""
        return self.resolved_user_id or self.user_id

    @property
    def custom_agents(self) -> dict[str, Any] | None:
        return getattr(self.workflow_request, "custom_agents", None)

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def cancel_title_task(self) -> None:
        """Cancel the title task if it is still running, so it cannot leak."""
        if self.title_task and not self.title_task.done():
            self.title_task.cancel()


@dataclass
class _ResumeTurn:
    """One approval resume, shared by the handlers that each own a step of it."""

    thread_id: str
    conversation_id: UUID
    user_id: UUID
    interrupt_id: str | None
    bot_message_id: UUID | None
    sanitized_persona: str | None = None
    custom_agents: dict[str, Any] = field(default_factory=dict)
    active_agent_id: str | None = None
    generation: Any = None
    partial_text: str = ""
    persisted: bool = False
    tool_artifacts: list[dict[str, Any]] = field(default_factory=list)
    tool_args_by_id: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    inflight: Any = None

    @property
    def event_message_id(self) -> str | None:
        return str(self.bot_message_id) if self.bot_message_id else None

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


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
        turn_coordinator: ConversationTurnCoordinator | None = None,
        generation_control_service: Any | None = None,
        project_context_service: Any | None = None,
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
        # Serializes turns within one conversation from context snapshot
        # through response persistence. Injected rather than constructed here
        # so the lock's durability is a deployment decision, not this class's.
        self._turn_coordinator = turn_coordinator
        # Owns the durable generation lifecycle: the row that makes Stop work
        # when the command lands on a different worker than the stream, and
        # Continue resume the exact checkpoint a pause left behind.
        self.generation_control_service = generation_control_service
        # Resolves project instructions into the system prompt when wired;
        # None preserves plain persona-only behavior.
        self.project_context_service = project_context_service
        self.redis_client = self._init_redis_client()

    def _generation_control(self):
        """The durable lifecycle service, or ``None`` when it is not wired.

        Same shape as :meth:`_hold_turn`: tests build this service with
        ``__new__`` and doubles, and
        ``test_the_container_wires_the_generation_control_service`` asserts the
        container-wired one always has it, so an undurable production path
        cannot pass unnoticed.
        """
        return getattr(self, "generation_control_service", None)

    async def _astart_generation(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        logical_turn_id: UUID,
    ):
        """Allocate the lifecycle row before anything can be asked to stop it.

        Ordering is the contract: the row exists before the first streamed
        event, so a Stop arriving on the very first token already has something
        durable to transition. The logical turn is the user message id, which
        is also the checkpoint's turn segment — one identity, so a Continue can
        find the exact thread from the row alone.

        A conflict here is reported, not swallowed. The partial unique index
        allows one active lifecycle per conversation, so an ``IntegrityError``
        means a turn really is already running — retriable, and exactly what
        ``conversation_turn_conflict`` says.
        """
        control = self._generation_control()
        if control is None:
            return None

        from sqlalchemy.exc import IntegrityError

        from app.ai.workflow.state import build_checkpoint_thread_id
        from app.schemas.generation import CreateGeneration

        try:
            return await control.start_generation(
                CreateGeneration(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    logical_turn_id=str(logical_turn_id),
                    checkpoint_thread_id=build_checkpoint_thread_id(
                        str(conversation_id), str(logical_turn_id)
                    ),
                )
            )
        except IntegrityError as exc:
            raise WorkflowRoutingException(
                workflow_error(
                    "conversation_turn_conflict",
                    request_id=str(logical_turn_id),
                    details={"reason": "a generation is already active for this conversation"},
                )
            ) from exc

    @staticmethod
    def _generation_status_data(snapshot: Any) -> dict[str, Any]:
        """The lifecycle fields every transport publishes, and nothing else.

        Written out field by field rather than dumping the snapshot, so adding
        a column to ``generations`` cannot silently start publishing it. What
        is deliberately absent is the checkpoint thread — a resume handle — and
        the budget and research-accounting blobs, which are bookkeeping.
        """
        return {
            "generation_id": str(snapshot.generation_id),
            "logical_turn_id": snapshot.logical_turn_id,
            "conversation_id": str(snapshot.conversation_id),
            "status": snapshot.status.value,
            "version": snapshot.version,
            "execution_epoch": snapshot.execution_epoch,
            "continuation_id": (
                str(snapshot.continuation_id) if snapshot.continuation_id else None
            ),
            "continuation_available": bool(snapshot.continuation_available),
            "continuation_block_reason": snapshot.continuation_block_reason,
            "assistant_message_id": (
                str(snapshot.assistant_message_id) if snapshot.assistant_message_id else None
            ),
            "terminal_reason": snapshot.terminal_reason,
        }

    async def _apublish_continuation_pause(
        self,
        event: V3StreamEvent,
        *,
        generation: Any,
        conversation_id: UUID,
        user_id: UUID,
        bot_message_id: UUID,
        sanitized_persona: str | None,
        workflow_request: Any,
        inflight: Any,
        tool_artifacts: list[dict[str, Any]] | None,
        next_sequence: Any,
    ):
        """Persist the validated partial, then offer Continue on it.

        The order is the requirement, not an implementation detail. A client
        that receives ``continuation_available`` may redeem the continuation id
        immediately, and the resumed epoch is assembled around an assistant
        message that must already exist. Advertising first would produce a
        Continue whose first half was never saved.

        If persistence fails, the turn is marked ``failed`` and no answer and
        no control event are emitted. A partial nobody can read is not an
        answer, and offering to continue it would be offering to continue
        nothing.
        """
        payload = dict(event.data or {})
        content = str(payload.get("validated_content") or "").strip()
        budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else None

        metadata: dict[str, Any] = {
            "partial": True,
            "continuable": True,
            "stop_reason": "execution_budget_exhausted",
            "generation_id": payload.get("generation_id") or str(generation.generation_id)
            if generation
            else payload.get("generation_id"),
            "logical_turn_id": payload.get("logical_turn_id"),
            "execution_epoch": payload.get("execution_epoch"),
            "persona_used": sanitized_persona,
        }
        if budget:
            metadata["execution_budget"] = budget
        web_sources = payload.get("web_sources")
        if isinstance(web_sources, list) and web_sources:
            metadata["web_sources"] = [dict(item) for item in web_sources if isinstance(item, dict)]
            metadata["web_sources_version"] = 1
        rich_items = payload.get("rich_items")
        if isinstance(rich_items, list) and rich_items:
            metadata["rich_items"] = [dict(item) for item in rich_items if isinstance(item, dict)]
            metadata["rich_items_version"] = 1
        grounding_warnings = payload.get("web_grounding_warnings")
        if isinstance(grounding_warnings, list) and grounding_warnings:
            metadata["web_grounding_warnings"] = [
                dict(item) for item in grounding_warnings if isinstance(item, dict)
            ]
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
        self._attach_active_agent_metadata(
            metadata,
            payload.get("active_agent_id") or inflight.active_agent_id,
            getattr(workflow_request, "custom_agents", None),
        )

        try:
            persisted_content, metadata = await self._externalize_remote_rich_images(
                fix_markdown_code_blocks(content) if content else content,
                metadata,
                conversation_id,
                user_id,
            )
            bot_message = await self._acreate_bot_response_message(
                conversation_id=conversation_id,
                content=persisted_content,
                metadata=metadata,
                message_id=bot_message_id,
            )
            await self._mark_persisted_web_images(
                metadata,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        except Exception:
            logging.exception("Could not persist the validated partial for a paused turn")
            await self._amark_generation_failed(generation, user_id=user_id)
            yield make_event(
                "error",
                sequence=next_sequence(),
                conversation_id=str(conversation_id),
                data=workflow_error_payload(
                    workflow_error(
                        "response_persistence_failed",
                        request_id=str(bot_message_id),
                        details={"reason": "the paused partial answer could not be saved"},
                    )
                ),
            )
            return

        # An undecidable side effect blocks Continue. The partial answer is
        # still persisted and shown -- what is refused is spending another
        # epoch, because resuming could perform the mutation a second time.
        # `mark_continuable` turns a block reason into
        # `continuation_available=false` with no continuation id minted.
        block_reason = (
            "mutation_outcome_unknown" if payload.get("mutation_outcome_unknown") else None
        )

        # The row's turn id first: it is the key the Continue side restores the
        # accounting under, and a snapshot read under any other key comes back
        # empty -- which reads to the next epoch as a full fresh quota.
        logical_turn_id = getattr(generation, "logical_turn_id", None) or payload.get(
            "logical_turn_id"
        )
        try:
            offered = await self._amark_generation_continuable(
                generation,
                user_id=user_id,
                assistant_message_id=bot_message_id,
                execution_budget=budget,
                research_accounting=self._research_accounting_snapshot(
                    logical_turn_id, conversation_id
                ),
                execution_epoch=_paused_epoch(payload),
                block_reason=block_reason,
            )
        except Exception:
            # The answer is saved and streamed, so the turn is not a failure —
            # it simply cannot be continued. Saying so beats advertising a
            # continuation id no Continue could redeem.
            logging.exception("Could not offer a continuation for a paused turn")
            offered = None

        yield make_event(
            "message_end",
            sequence=next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data={"message": bot_message.model_dump(mode="json")},
        )

        from app.models.generation import GenerationStatus

        # A Stop that landed first is answered with `stopped`, not an offer.
        if offered is None or offered.status is not GenerationStatus.CONTINUABLE:
            return

        yield make_event(
            "continuation_available",
            sequence=next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data=self._generation_status_data(offered),
        )

    async def _aworker_transition(
        self,
        generation: Any,
        *,
        user_id: UUID,
        apply: Any,
        on_stop_requested: Any = None,
    ):
        """Apply one worker-side transition, re-fencing once if a Stop moved the row.

        The worker fences on the last version it saw, and a client's Stop
        advances the version without the worker seeing it. Refused for a fence
        it could not have known about, a terminal transition would leave the
        row active, and an active row refuses every later turn in the
        conversation. Re-reading once is safe: while the row is active, Stop is
        the only command a client can issue against it.

        ``on_stop_requested`` answers a row that now says ``stop_requested``
        when ``apply`` is not legal from there (completing, pausing).
        """
        from app.models.generation import GenerationStatus
        from app.services.generation_control_service import IllegalTransition

        try:
            return await apply(generation)
        except IllegalTransition:
            current = await self._generation_control().aget_snapshot(
                generation_id=generation.generation_id,
                user_id=user_id,
                conversation_id=generation.conversation_id,
            )
            if current is None or current.version == generation.version:
                raise
            if current.status is GenerationStatus.STOP_REQUESTED and on_stop_requested:
                return await on_stop_requested(current)
            return await apply(current)

    def _stop_transition(
        self,
        *,
        user_id: UUID,
        assistant_message_id: UUID | None,
        terminal_reason: str,
    ):
        """A ``mark_stopped`` fenced on whichever snapshot it is handed."""
        from app.schemas.generation import MarkStopped

        control = self._generation_control()

        async def apply(snapshot: Any):
            return await control.mark_stopped(
                MarkStopped(
                    generation_id=snapshot.generation_id,
                    conversation_id=snapshot.conversation_id,
                    user_id=user_id,
                    expected_version=snapshot.version,
                    assistant_message_id=assistant_message_id,
                    terminal_reason=terminal_reason,
                )
            )

        return apply

    async def _amark_generation_failed(
        self,
        generation: Any,
        *,
        user_id: UUID,
        terminal_reason: str = "response_persistence_failed",
    ):
        """Record a turn that could not produce a readable answer."""
        control = self._generation_control()
        if control is None or generation is None:
            return None

        async def apply(snapshot: Any):
            return await control.mark_failed(
                generation_id=snapshot.generation_id,
                user_id=user_id,
                conversation_id=snapshot.conversation_id,
                expected_version=snapshot.version,
                terminal_reason=terminal_reason,
            )

        try:
            return await self._aworker_transition(generation, user_id=user_id, apply=apply)
        except Exception as exc:  # noqa: BLE001 - bookkeeping never fails a turn
            logging.warning(
                "Could not mark generation %s failed: %s",
                generation.generation_id,
                type(exc).__name__,
            )
            return None

    async def _amark_generation_running(self, generation: Any, *, user_id: UUID):
        """Move ``starting`` to ``running`` once the turn is really executing.

        Not cosmetic. ``continuable`` is only legal from ``running`` (or a later
        active status), so a turn that paused at its budget without ever
        leaving ``starting`` could not be offered a Continue at all.

        Returns the advanced snapshot, or the original one when the transition
        could not be made — every later transition fences on the version this
        returns, so handing back a stale one would make them all fail.
        """
        control = self._generation_control()
        if control is None or generation is None:
            return generation

        try:
            return await control.mark_running(
                generation_id=generation.generation_id,
                user_id=user_id,
                conversation_id=generation.conversation_id,
                expected_version=generation.version,
            )
        except Exception as exc:  # noqa: BLE001 - a Stop already won this race
            logging.info(
                "Generation %s did not enter running: %s",
                generation.generation_id,
                type(exc).__name__,
            )
            return generation

    async def _amark_generation_completed(
        self,
        generation: Any,
        *,
        user_id: UUID,
        assistant_message_id: UUID | None,
        partial: bool = False,
        terminal_reason: str | None = None,
    ):
        """Close the lifecycle row for a turn that finished on its own.

        ``user_id`` is passed rather than read off the snapshot: the snapshot is
        what transports publish, and an owner id is not something a client
        should be able to read back out of one.

        Never raises into the stream. The answer is already persisted and
        streamed by the time this runs, so failing the turn over a bookkeeping
        write would discard a good response; the row is left for the stuck-state
        reconciliation the rollout procedure documents.

        A Stop that lost the race to the answer leaves the row at
        ``stop_requested``, where completing is not legal; the worker answers
        it with ``stopped``, keeping the finished answer on the row.
        """
        control = self._generation_control()
        if control is None or generation is None:
            return None

        from app.schemas.generation import MarkCompleted

        async def apply(snapshot: Any):
            return await control.mark_completed(
                MarkCompleted(
                    generation_id=snapshot.generation_id,
                    conversation_id=snapshot.conversation_id,
                    user_id=user_id,
                    expected_version=snapshot.version,
                    assistant_message_id=assistant_message_id,
                    partial=partial,
                    terminal_reason=terminal_reason,
                )
            )

        try:
            return await self._aworker_transition(
                generation,
                user_id=user_id,
                apply=apply,
                on_stop_requested=self._stop_transition(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                    terminal_reason=terminal_reason or "completed",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping never fails a turn
            logging.warning(
                "Could not close generation %s: %s", generation.generation_id, type(exc).__name__
            )
            return None

    @staticmethod
    def _install_research_accounting(lease: Any, *, conversation_id: UUID) -> None:
        """Restore the turn's research dedup memory for the epoch about to run.

        Both Continue paths go through this, on purpose: a Continue served by
        *this* worker rehydrates from the row exactly as one served by another
        worker does, rather than reusing whatever the in-memory entry holds. One
        path means the two cannot disagree about how much quota the epoch has.

        A turn that never searched has no accounting, and that is not an error —
        there is nothing for the next epoch to be refused against. Only a
        payload that exists and cannot be read is fatal, and it raises.
        """
        from app.ai.research_budget import install_research_budget

        payload = getattr(lease, "research_accounting", None)
        if payload is None:
            return
        install_research_budget(
            payload,
            logical_turn_id=lease.snapshot.logical_turn_id,
            conversation_id=str(conversation_id),
        )

    @staticmethod
    def _research_accounting_snapshot(
        logical_turn_id: Any,
        conversation_id: UUID,
    ) -> dict[str, Any] | None:
        """The turn's research dedup memory, for the next epoch to respect.

        Persisted on the row rather than left in process memory (R4): a
        Continue may be served by a worker that never ran this epoch, and an
        absent accounting there is indistinguishable from a fresh turn with a
        full quota.

        Never raises. Failing to snapshot costs the next epoch its dedup
        memory, which is a worse answer but still an answer; failing the *pause*
        over it would discard a validated partial the user can already see.
        The Continue side is where an unreadable payload is fatal.
        """
        from app.ai.research_budget import snapshot_research_budget

        try:
            return snapshot_research_budget(
                logical_turn_id=str(logical_turn_id) if logical_turn_id else None,
                conversation_id=str(conversation_id),
            )
        except Exception as exc:  # noqa: BLE001 - never fails a validated pause
            logging.warning("Could not snapshot research accounting: %s", type(exc).__name__)
            return None

    async def _amark_generation_continuable(
        self,
        generation: Any,
        *,
        user_id: UUID,
        assistant_message_id: UUID,
        execution_budget: dict[str, Any] | None = None,
        research_accounting: dict[str, Any] | None = None,
        execution_epoch: int | None = None,
        block_reason: str | None = None,
    ):
        """Offer Continue on a partial answer that is already persisted.

        Ordering is the whole contract of Task 5 Step 4: the assistant message
        must be committed before this runs, because the continuation id minted
        here is what a client uses to ask for more of an answer it can already
        see. Offering first and persisting second would leave a Continue that
        resumes work whose first half was never saved.

        Unlike the completion transition this one *does* propagate. A failure
        here means the offer was not recorded, so publishing
        ``continuation_available`` anyway would advertise a continuation id
        that no Continue could ever redeem.

        A Stop that arrived first is answered with ``stopped`` instead, and the
        snapshot returned says so: the caller offers nothing for it.
        """
        control = self._generation_control()
        if control is None or generation is None:
            return None

        from app.schemas.generation import MarkContinuable

        async def apply(snapshot: Any):
            return await control.mark_continuable(
                MarkContinuable(
                    generation_id=snapshot.generation_id,
                    conversation_id=snapshot.conversation_id,
                    user_id=user_id,
                    expected_version=snapshot.version,
                    assistant_message_id=assistant_message_id,
                    execution_budget=execution_budget,
                    research_accounting=research_accounting,
                    execution_epoch=execution_epoch,
                    continuation_block_reason=block_reason,
                )
            )

        return await self._aworker_transition(
            generation,
            user_id=user_id,
            apply=apply,
            on_stop_requested=self._stop_transition(
                user_id=user_id,
                assistant_message_id=assistant_message_id,
                terminal_reason="user_requested",
            ),
        )

    async def _amark_generation_awaiting_approval(
        self,
        generation: Any,
        *,
        user_id: UUID,
        assistant_message_id: UUID,
    ):
        """Release the conversation while a tool call waits on a human.

        The approval message is already persisted, and nothing runs until a
        decision arrives, possibly on another worker. Left ``running``, the row
        would refuse every later message in the conversation until the startup
        reaper failed it. Never raises: the interrupt is saved and is still
        answerable, so bookkeeping must not turn it into an error.
        """
        from app.services.generation_control_service import APPROVAL_PAUSE_REASON

        try:
            paused = await self._amark_generation_continuable(
                generation,
                user_id=user_id,
                assistant_message_id=assistant_message_id,
                block_reason=APPROVAL_PAUSE_REASON,
            )
            from app.models.generation import GenerationStatus

            if paused is not None and paused.status is GenerationStatus.STOPPED:
                await self._generation_control().invalidate_pending_approval(
                    paused, user_id=user_id
                )
            return paused
        except Exception as exc:  # noqa: BLE001 - bookkeeping never fails a turn
            logging.warning(
                "Could not record the approval pause of generation %s: %s",
                generation.generation_id,
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _approval_pause_was_stopped(snapshot):
        from app.models.generation import GenerationStatus

        return snapshot is not None and snapshot.status is GenerationStatus.STOPPED

    @staticmethod
    def _stopped_approval_event(conversation_id, sequence):
        return make_event(
            "error",
            sequence=sequence,
            conversation_id=str(conversation_id),
            data={"error": "This generation was stopped.", "error_code": "generation_stopped"},
        )

    async def _aresume_approved_generation(
        self,
        *,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
    ):
        """The approval-paused generation this resume continues, running again.

        Only a genuinely legacy thread with no lifecycle row can run untracked.
        Existing rows must reclaim successfully before the approval is consumed.
        """
        control = self._generation_control()
        if control is None:
            return None

        from app.ai.workflow.state import parse_checkpoint_thread_id

        try:
            thread_conversation, logical_turn_id = parse_checkpoint_thread_id(thread_id)
        except ValueError:
            return None
        if thread_conversation != str(conversation_id):
            return None

        from sqlalchemy.exc import IntegrityError

        paused = await control.find_by_logical_turn(
            logical_turn_id=logical_turn_id, user_id=user_id, conversation_id=conversation_id
        )
        if paused is None:
            return None  # Truly legacy threads have no lifecycle row.
        try:
            return await control.mark_resumed_after_approval(
                generation_id=paused.generation_id, user_id=user_id, conversation_id=conversation_id
            )
        except IntegrityError as exc:
            raise WorkflowRoutingException(
                workflow_error(
                    "conversation_turn_conflict",
                    request_id=logical_turn_id,
                    details={"reason": "a generation is already active for this conversation"},
                )
            ) from exc

    async def _amark_generation_stopped(
        self,
        generation: Any,
        *,
        user_id: UUID,
        assistant_message_id: UUID | None,
        terminal_reason: str = "stopped",
    ):
        """Record that this worker actually stopped.

        This is the transition that turns a client's ``stop_requested`` into
        ``stopped``. Only the owning worker can make it, because only the
        worker knows it has really let go of the turn.
        """
        control = self._generation_control()
        if control is None or generation is None:
            return None

        apply = self._stop_transition(
            user_id=user_id,
            assistant_message_id=assistant_message_id,
            terminal_reason=terminal_reason,
        )
        try:
            return await self._aworker_transition(generation, user_id=user_id, apply=apply)
        except Exception as exc:  # noqa: BLE001 - bookkeeping never fails a turn
            logging.warning(
                "Could not record the stop of generation %s: %s",
                generation.generation_id,
                type(exc).__name__,
            )
            return None

    def _hold_turn(self, conversation_id: Any, *, request_id: str):
        """Hold this conversation's turn lock for the body of the turn.

        Returns a null context when no coordinator is configured. Tests build
        this service with ``__new__`` and doubles; the container-wired service
        always has one, which
        ``test_the_container_wires_a_durable_turn_coordinator_into_message_service``
        asserts so an unlocked production path cannot pass unnoticed.
        """
        coordinator = getattr(self, "_turn_coordinator", None)
        if coordinator is None:
            return contextlib.nullcontext()
        key = str(conversation_id) if conversation_id else None
        return coordinator.hold(key, request_id=request_id)

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
            return cls._decision_runtime_provenance(record, provenance_map, decisions)

        client_entries = [
            (key, provenance)
            for key, provenance in provenance_map.items()
            if cls._is_client_runtime_provenance_entry(provenance)
        ]
        return client_entries or cls._record_runtime_provenance(record)

    @classmethod
    def _decision_runtime_provenance(
        cls,
        record: Any,
        provenance_map: dict[str, dict[str, Any]],
        decisions: list[InterruptDecision],
    ) -> list[tuple[str, dict[str, Any]]]:
        """The client-runtime provenance of the calls these decisions would run.

        Only an approve or an edit runs anything, so a set of rejections has
        nothing to validate against the device.
        """
        actionable_decisions = [
            decision
            for decision in decisions
            if decision.type in (InterruptDecisionType.APPROVE, InterruptDecisionType.EDIT)
        ]
        if not actionable_decisions:
            return []

        matched_entries = [
            entry
            for decision in actionable_decisions
            if (entry := cls._decision_provenance_entry(provenance_map, decision)) is not None
        ]
        if matched_entries:
            return matched_entries
        if not provenance_map:
            return cls._record_runtime_provenance(record)
        return []

    @classmethod
    def _decision_provenance_entry(
        cls,
        provenance_map: dict[str, dict[str, Any]],
        decision: InterruptDecision,
    ) -> tuple[str, dict[str, Any]] | None:
        """The first client-runtime entry this decision addresses, by id then action."""
        for key in cls._decision_keys(decision):
            provenance = provenance_map.get(key)
            if cls._is_client_runtime_provenance_entry(provenance or {}):
                return key, provenance
        return None

    @classmethod
    def _record_runtime_provenance(cls, record: Any) -> list[tuple[str, dict[str, Any]]]:
        """The record's own execution scope, for an interrupt with no per-call map."""
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
        """Every pending tool call gets exactly one allowed decision, or 422."""
        pending_requests = getattr(record, "action_requests_json", None)
        if not isinstance(pending_requests, list) or not pending_requests:
            return

        action_counts = _pending_action_counts(pending_requests)
        request_indexes_by_key = _pending_request_indexes(pending_requests, action_counts)

        seen_request_indexes: set[int] = set()
        for decision in decisions or []:
            candidate_keys = cls._decision_keys(decision)
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
            _require_allowed_decision(pending_requests[matched_index], decision)

        decision_keys = {
            key for decision in decisions or [] for key in cls._decision_keys(decision)
        }
        missing = [
            request_key
            for request in pending_requests
            if isinstance(request, dict)
            and (request_key := _pending_request_key(request, action_counts)) is not None
            and request_key not in decision_keys
        ]
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

    @classmethod
    def _decision_keys(cls, decision: Any) -> list[str]:
        """The ids a decision can address a pending request by, in match order."""
        keys: list[str] = []
        decision_id = resolve_interrupt_decision_id(decision)
        if decision_id:
            keys.append(str(decision_id))
        action = cls._decision_action(decision)
        if action:
            keys.append(action)
        return keys

    def _get_conversation_context(
        self, conversation_id: UUID, user_id: UUID | None = None
    ) -> tuple[UUID | None, str | None]:
        """Get user_id and the composed system instruction from a conversation.

        Returns the instruction already sanitized and composed. The caller must
        NOT run ``sanitize_persona`` over it: the composed string can exceed the
        8000-character cap legitimately, and truncating it would silently drop
        the conversation's own persona.
        """
        conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
            conversation_id
        )
        resolved_user_id = user_id or (conversation.owner_id if conversation else None)
        if self.project_context_service is None:
            return resolved_user_id, sanitize_persona(
                conversation.persona_prompt if conversation else None
            )
        return resolved_user_id, self.project_context_service.resolve_system_instruction(
            conversation
        )

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
            # Suggestions are optional; the answer is persisted without them.
            logging.warning("Follow-up suggestions failed", exc_info=True)

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
            logging.warning("Conversation title update failed", exc_info=True)
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
            self.redis_client.setex(key, timeout_seconds, datetime.now(UTC).isoformat())
            deadline = datetime.now(UTC) + timedelta(minutes=settings.hitl_approval_timeout_minutes)
            if not interrupt_response.get("metadata"):
                interrupt_response["metadata"] = {}
            interrupt_response["metadata"]["timeout_deadline"] = deadline.isoformat()
        except Exception:
            logging.warning("Could not record the interrupt timeout in Redis", exc_info=True)

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
        context: _InterruptPersistContext | None = None,
    ) -> MessageRead:
        """
        Persist an assistant message that represents a paused workflow awaiting HITL approval.

        Also creates a durable HITLInterrupt lifecycle record so pending approvals
        are recoverable via normal message history APIs (DB-backed) and survive
        process restarts. Callers handling a claimed nested resume can require
        durable-record failures to propagate.
        """
        context = context or _InterruptPersistContext()
        interrupt_dict = _interrupt_payload_dict(interrupt_payload)
        bot_message = self._create_bot_response_message(
            conversation_id=conversation_id,
            content=fix_markdown_code_blocks(context.partial_text),
            metadata=self._interrupt_message_metadata(interrupt_dict, context),
            message_id=context.message_id,
        )

        interrupt_id = interrupt_dict.get("interrupt_id")
        if not (self.hitl_interrupt_repository and context.user_id and interrupt_id):
            return bot_message
        try:
            self._create_durable_interrupt_record(
                conversation_id, interrupt_dict, context, assistant_message_id=bot_message.id
            )
        except Exception as exc:
            logging.warning(
                "Failed to create durable interrupt record for interrupt_id=%s: %s",
                interrupt_id,
                exc,
                exc_info=True,
            )
            if context.require_durable_interrupt:
                self.repository.delete(bot_message.id)
                raise
        return bot_message

    def _interrupt_message_metadata(
        self, interrupt_dict: dict[str, Any], context: _InterruptPersistContext
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "interrupt": interrupt_dict,
            "paused": True,
            "pause_reason": "tool_approval_required",
        }
        optional = {
            "persona_used": context.sanitized_persona or None,
            "thread_id": context.thread_id or None,
            "next": context.next_nodes,
            "pending_tool_calls": context.pending_tool_calls,
        }
        metadata.update({key: value for key, value in optional.items() if value is not None})

        # Preserve live_widgets from widget tools that already succeeded
        # before the interrupt paused the run.
        if context.tool_artifacts:
            metadata["tool_artifacts"] = context.tool_artifacts
            from app.core.response_constants import extract_live_widgets_from_artifacts

            live_widgets = extract_live_widgets_from_artifacts(context.tool_artifacts)
            if live_widgets:
                metadata["live_widgets"] = live_widgets

        self._attach_active_agent_metadata(metadata, context.active_agent_id, context.custom_agents)
        return metadata

    def _create_durable_interrupt_record(
        self,
        conversation_id: UUID,
        interrupt_dict: dict[str, Any],
        context: _InterruptPersistContext,
        *,
        assistant_message_id: UUID,
    ) -> None:
        """The HITL lifecycle record that makes a pending approval recoverable."""
        interrupt_metadata = _interrupt_metadata_of(interrupt_dict)
        execution_scope = self._derive_interrupt_execution_scope(interrupt_metadata)
        self.hitl_interrupt_repository.create(
            interrupt_id=interrupt_dict.get("interrupt_id"),
            conversation_id=conversation_id,
            user_id=context.user_id,
            thread_id=context.thread_id or str(conversation_id),
            expires_at=datetime.now(UTC)
            + timedelta(minutes=settings.hitl_approval_timeout_minutes),
            action_requests_json=interrupt_dict.get("action_requests") or [],
            assistant_message_id=assistant_message_id,
            device_id=_interrupt_device_id(interrupt_metadata),
            interrupt_metadata_json=interrupt_metadata,
            session_id=execution_scope.get("session_id"),
            catalog_version=execution_scope.get("catalog_version"),
            tool_instance_id=execution_scope.get("tool_instance_id"),
        )

    async def create_message(
        self, message_create_data: MessageCreate, user_id: UUID
    ) -> MessageRead:
        # Same turn lock as the streaming path: this entrypoint runs the same
        # snapshot-generate-persist sequence, so leaving it unlocked would let
        # a non-streamed turn race a streamed one in the same conversation.
        async with self._hold_turn(message_create_data.conversation_id, request_id=str(uuid4())):
            return await self._create_message_holding_turn(message_create_data, user_id)

    async def _create_message_holding_turn(
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
                        context=_InterruptPersistContext(
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
                            active_agent_id=bot_response.agent_id if bot_response else None,
                            custom_agents=workflow_request.custom_agents,
                        ),
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
        """Stream one turn while holding this conversation's turn lock.

        The lock wraps the whole turn — user-message write, context snapshot,
        generation, and response persistence — because releasing before the
        write would leave exactly the window that matters unprotected. A
        contended conversation surfaces as one typed retriable error event
        rather than an unhandled exception, so the client can retry rather
        than see a broken stream.
        """
        # Reserve the assistant DB id up-front when the caller did not supply
        # one so the workflow request can carry a stable id, and so a conflict
        # reported before any row exists still has a stable correlation id.
        if bot_message_id is None:
            bot_message_id = uuid4()

        try:
            async with (
                self._hold_turn(
                    message_create_data.conversation_id, request_id=str(bot_message_id)
                ),
                # Closed here, under the lock, rather than whenever the
                # collector gets to it: its cleanup settles the lifecycle row.
                contextlib.aclosing(
                    self._create_message_stream_holding_turn(
                        message_create_data, user_id, bot_message_id
                    )
                ) as events,
            ):
                async for event in events:
                    yield event
        except WorkflowRoutingException as exc:
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(message_create_data.conversation_id),
                data=workflow_error_payload(exc.error),
            )

    async def _create_message_stream_holding_turn(
        self,
        message_create_data: MessageCreate,
        user_id: UUID,
        bot_message_id: UUID,
    ):
        """
        Create a message and stream the bot response.
        Yields chunks as they arrive from the AI service.

        Integrates with GenerationRegistry so that in-flight streams can be
        cancelled via ``POST /messages/stop`` or HTTP disconnect without
        persisting cancellation/disconnect artifacts as error messages.

        Every way out of the turn settles its lifecycle row. The partial
        unique index admits one active generation per conversation, so an
        ending that leaves the row active refuses every later message until
        the startup reaper fails it.
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
        # Drop any stale prompt-history cache before the workflow reads it.
        with contextlib.suppress(Exception):
            self.ai_service.invalidate_history_cache(str(message_create_data.conversation_id))

        turn = await self._abegin_user_turn(
            message_create_data,
            user_id=user_id,
            user_message_id=created_message.id,
            bot_message_id=bot_message_id,
        )

        # Caught here, in the generator the caller closes, rather than in a
        # delegate: closing an async generator never reaches the one it is
        # iterating, which would be finalized only when collected. A
        # disconnect (or a Stop on this worker, which cancels the producer
        # task) arrives as ``CancelledError`` or ``GeneratorExit``.
        try:
            async for event in self._astream_turn_events(
                turn, message_create_data, created_message
            ):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            # Cancellation / disconnect: persist partial text if available,
            # do NOT create an error message.
            await self._apersist_interrupted_user_turn(turn)
            return
        except Exception as exc:
            turn.cancel_title_task()
            if not turn.bot_message_persisted:
                yield await self._afail_user_turn(turn, exc)

    async def _abegin_user_turn(
        self,
        message_create_data: MessageCreate,
        *,
        user_id: UUID,
        user_message_id: UUID,
        bot_message_id: UUID,
    ) -> _UserTurn:
        """Allocate the lifecycle row and the in-flight entry for one turn.

        The durable row comes before the first streamed event, so a Stop that
        arrives on the very first token already has something to transition.
        It enters ``running`` before ``run_start`` is published: the version
        that event carries is the fence a client's Stop is checked against,
        and publishing the ``starting`` one made every fenced Stop stale.
        ``continuable`` is also legal only from ``running``.
        """
        generation = await self._astart_generation(
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
            logical_turn_id=user_message_id,
        )
        generation = await self._amark_generation_running(generation, user_id=user_id)

        # Register in-flight entry, keyed by generation id when there is one.
        # The user message id remains the fallback so an unwired service keeps
        # working; Stop never reads it as identity either way, because the
        # durable row is what a Stop from another worker can see.
        registry = get_generation_registry()
        registry_key = generation.generation_id if generation else user_message_id
        inflight = registry.register(
            generation_id=registry_key,
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
        )
        # The event alone cannot reach a producer blocked inside a provider
        # call; the task can. It does not exist until this coroutine runs, so
        # it is attached here rather than passed to ``register``.
        inflight.task = asyncio.current_task()
        return _UserTurn(
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
            user_message_id=user_message_id,
            bot_message_id=bot_message_id,
            generation=generation,
            registry=registry,
            registry_key=registry_key,
            inflight=inflight,
        )

    async def _astream_turn_events(
        self,
        turn: _UserTurn,
        message_create_data: MessageCreate,
        created_message: Any,
    ):
        """Open the turn, then answer it, or close it if nothing is to be answered."""
        if turn.generation is not None:
            # The canonical opening event of a turn. Everything a client needs
            # to address a later Stop or Continue is here, including the version
            # that fences them (R5).
            yield make_event(
                "run_start",
                sequence=turn.next_sequence(),
                conversation_id=str(turn.conversation_id),
                message_id=str(turn.bot_message_id),
                data=self._generation_status_data(turn.generation),
            )

        # Yield user message creation event
        yield make_event(
            "user_message_created",
            sequence=turn.next_sequence(),
            conversation_id=str(turn.conversation_id),
            message_id=str(turn.user_message_id),
            data={"message": MessageRead.model_validate(created_message).model_dump(mode="json")},
        )

        if message_create_data.role != MessageRole.user:
            await self._aclose_turn_without_reply(turn)
            return

        async for event in self._astream_user_turn(turn, message_create_data):
            yield event

    async def _aclose_turn_without_reply(self, turn: _UserTurn) -> None:
        """A message nobody answers still started a lifecycle row; finish it."""
        turn.inflight.resolve(None)
        turn.registry.remove(turn.registry_key)
        await self._amark_generation_completed(
            turn.generation,
            user_id=turn.user_id,
            assistant_message_id=None,
            terminal_reason="no_reply_requested",
        )

    async def _astream_user_turn(self, turn: _UserTurn, message_create_data: MessageCreate):
        """Prepare the request, run the graph, and end the turn the way it ended."""
        await self._aprepare_user_turn(turn, message_create_data)
        stop_watch = _DurableStopWatch(
            control=self._generation_control(),
            generation=turn.generation,
            user_id=turn.owner_id,
        )

        async for raw_event in self.ai_service.execute_request_stream(turn.workflow_request):
            event = await self._anext_user_turn_event(turn, stop_watch, raw_event)
            if event is None:
                break
            if event.type in ("interrupt", "continuation_available"):
                async for paused_event in self._apause_user_turn(turn, event):
                    yield paused_event
                return
            for projected in self._project_user_turn_event(turn, event):
                yield projected

        # ---- Handle cancellation after the loop exits ----
        if turn.inflight.is_cancelled:
            await self._astop_user_turn(turn)
            return

        async for event in self._afinish_user_turn(turn, message_create_data):
            yield event

    async def _anext_user_turn_event(
        self, turn: _UserTurn, stop_watch: _DurableStopWatch, raw_event: V3StreamEvent
    ) -> V3StreamEvent | None:
        """The next event to handle, or ``None`` once the loop must end.

        It ends on a Stop (either signal) and on the terminal ``complete`` or
        ``error``, whose response is kept for persistence.
        """
        # ---- Check cancellation before processing each event ----
        if turn.inflight.is_cancelled:
            logging.info("Stream cancelled for user_message_id=%s", turn.user_message_id)
            return None

        event = _service_event_from_ai_event(raw_event, sequence=turn.next_sequence())
        if await self._stop_requested_durably(turn, stop_watch, event.type):
            return None
        if event.type in ("complete", "error"):
            turn.bot_response = event.data.get("response")
            return None
        return event

    async def _aprepare_user_turn(
        self, turn: _UserTurn, message_create_data: MessageCreate
    ) -> None:
        """Load the conversation, start the title, and build the workflow request."""
        # Load conversation once — reused for title check, context, and planning mode
        conversation = await self.conversation_validation_utils.conversation_repository.aget_by_id(
            message_create_data.conversation_id
        )
        default_titles = {"New Conversation", "Untitled", ""}
        needs_title = conversation is not None and (
            conversation.title in default_titles or conversation.title is None
        )
        if needs_title:
            turn.title_task = asyncio.create_task(
                self._generate_title_async(
                    message_create_data.conversation_id,
                    message_create_data.content,
                    user_id=turn.user_id,
                )
            )

        (
            turn.resolved_user_id,
            turn.sanitized_persona,
            turn.workflow_request,
        ) = await self._build_user_message_workflow_request(
            message_create_data=message_create_data,
            user_id=turn.user_id,
            conversation=conversation,
            user_message_id=turn.user_message_id,
            assistant_message_id=turn.bot_message_id,
        )

    @staticmethod
    async def _stop_requested_durably(
        turn: _UserTurn, stop_watch: _DurableStopWatch, event_type: str
    ) -> bool:
        """Whether the row says stop: the only signal a Stop on another worker sends."""
        # Polled at coarse boundaries rather than per token — see _DurableStopWatch.
        if not await stop_watch.stop_requested(event_type):
            return False
        logging.info("Stream stopping on durable status for generation=%s", turn.registry_key)
        # `mark_cancelled`, not `request_cancel`: this producer is stopping
        # itself and is already at a check point. Cancelling its own task would
        # raise out of the very code below that persists the partial.
        turn.inflight.mark_cancelled()
        return True

    def _project_user_turn_event(self, turn: _UserTurn, event: V3StreamEvent) -> list:
        """Record what one streamed event contributes, and what to forward."""
        turn.inflight.touch()
        if event.type == "agent_selected":
            active_agent_id = event.agent or event.data.get("agent")
            turn.inflight.active_agent_id = active_agent_id
            return [
                self._agent_selected_event(
                    active_agent_id, turn.custom_agents, sequence=event.sequence
                )
            ]
        if event.type == "message_delta":
            turn.inflight.partial_text += event.data.get("text", "")
        elif event.type == "reasoning_delta":
            turn.inflight.partial_thinking += event.data.get("text", "")
        else:
            _record_stream_tool_event(event, turn.tool_artifacts, turn.tool_args_by_id)
        # rich_items, state_snapshot, subagent lifecycle and other canonical
        # events pass through unchanged.
        return [event]

    async def _apause_user_turn(self, turn: _UserTurn, event: V3StreamEvent):
        """Persist a paused turn, release its row, and keep the entry as a lock token.

        The entry stays (paused, carrying the resolved active agent) so a
        custom agent cannot be edited, deleted or detached while this run can
        still resume.
        """
        turn.cancel_title_task()
        if event.type == "interrupt":
            yield await self._ainterrupt_user_turn(turn, event)
        else:
            # The turn paused at its execution budget with a validated partial
            # answer. Persist first, offer second: the continuation id a client
            # redeems must point at an answer that is already saved.
            async for paused_event in self._apublish_continuation_pause(
                event,
                generation=turn.generation,
                conversation_id=turn.conversation_id,
                user_id=turn.owner_id,
                bot_message_id=turn.bot_message_id,
                sanitized_persona=turn.sanitized_persona,
                workflow_request=turn.workflow_request,
                inflight=turn.inflight,
                tool_artifacts=turn.tool_artifacts or None,
                next_sequence=turn.next_sequence,
            ):
                yield paused_event
        turn.inflight.resolve()
        turn.registry.mark_paused(turn.registry_key)

    async def _ainterrupt_user_turn(self, turn: _UserTurn, event: V3StreamEvent):
        """Workflow paused for human approval: persist it, then release the row.

        The approval message is persisted and the row released *before* the
        interrupt is published. The SSE consumer stops reading at an interrupt
        and cancels the producer, so anything after the yield may never run.
        """
        interrupt_response = event.data.get("interrupt")
        interrupt_id = interrupt_response.get("interrupt_id") if interrupt_response else None
        self._handle_redis_interrupt_storage(turn.conversation_id, interrupt_id, interrupt_response)
        self._set_plan_lifecycle(turn.conversation_id, turn.resolved_user_id, PlanLifecycle.paused)

        interrupt_thread_id = event.data.get("thread_id") or str(turn.conversation_id)
        persisted = self._persist_interrupt_bot_message(
            conversation_id=turn.conversation_id,
            interrupt_payload=interrupt_response,
            context=_InterruptPersistContext(
                sanitized_persona=turn.sanitized_persona,
                partial_text=turn.inflight.partial_text,
                pending_tool_calls=event.data.get("pending_tool_calls"),
                thread_id=interrupt_thread_id,
                next_nodes=event.data.get("next"),
                user_id=turn.resolved_user_id,
                message_id=turn.bot_message_id,
                tool_artifacts=turn.tool_artifacts or None,
                active_agent_id=turn.inflight.active_agent_id,
                custom_agents=turn.custom_agents,
            ),
        )
        paused = await self._amark_generation_awaiting_approval(
            turn.generation, user_id=turn.owner_id, assistant_message_id=turn.bot_message_id
        )
        if self._approval_pause_was_stopped(paused):
            return self._stopped_approval_event(turn.conversation_id, event.sequence)
        return make_event(
            "interrupt",
            sequence=event.sequence,
            conversation_id=str(turn.conversation_id),
            message_id=str(turn.bot_message_id),
            data={
                "thread_id": interrupt_thread_id,
                "next": event.data.get("next"),
                "pending_tool_calls": event.data.get("pending_tool_calls"),
                "interrupt": interrupt_response,
                "message": persisted.model_dump(mode="json"),
            },
        )

    async def _astop_user_turn(self, turn: _UserTurn) -> None:
        """A cooperative Stop: keep the partial, then confirm the stop on the row."""
        turn.cancel_title_task()
        partial = turn.inflight.partial_text.strip()
        assistant_message_id = None
        if partial:
            bot_message = await self._acreate_bot_response_message(
                conversation_id=turn.conversation_id,
                content=fix_markdown_code_blocks(partial),
                metadata=self._stopped_partial_metadata(turn, "user_requested"),
                message_id=turn.bot_message_id,
            )
            turn.inflight.resolve(bot_message.model_dump(mode="json"))
            assistant_message_id = turn.bot_message_id
        else:
            turn.inflight.resolve(None)
        # The worker confirming it let go. Only this side can make the
        # transition, which is what turns a client's `stop_requested` into an
        # authoritative `stopped`.
        await self._amark_generation_stopped(
            turn.generation,
            user_id=turn.owner_id,
            assistant_message_id=assistant_message_id,
            terminal_reason="user_requested",
        )
        turn.registry.remove(turn.registry_key)

    def _stopped_partial_metadata(self, turn: _UserTurn, stop_reason: str) -> dict[str, Any]:
        metadata = {
            "stopped": True,
            "partial": True,
            "stop_reason": stop_reason,
            "persona_used": turn.sanitized_persona,
            "reply_to_user_message_id": str(turn.user_message_id),
        }
        self._attach_active_agent_metadata(
            metadata, turn.inflight.active_agent_id, turn.custom_agents
        )
        return metadata

    async def _afinish_user_turn(self, turn: _UserTurn, message_create_data: MessageCreate):
        """Persist the answer, close the row, then publish the title and ``complete``."""
        _merge_stream_tool_artifacts_into_response(turn.bot_response, turn.tool_artifacts)

        bot_message = await self._persist_completed_workflow_response(
            conversation_id=turn.conversation_id,
            user_id=turn.resolved_user_id,
            bot_response=turn.bot_response,
            sanitized_persona=turn.sanitized_persona,
            workflow_request=turn.workflow_request,
            message_id=turn.bot_message_id,
            reply_to_user_message_id=turn.user_message_id,
            suggestion_source_message=message_create_data.content,
        )
        turn.bot_message_persisted = True
        await self._compact_checkpoint_after_persist(
            conversation_id=turn.conversation_id,
            workflow_request=turn.workflow_request,
        )

        # Resolve the inflight future with the final message
        turn.inflight.resolve(bot_message.model_dump(mode="json"))
        await self._amark_generation_completed(
            turn.generation,
            user_id=turn.owner_id,
            assistant_message_id=turn.bot_message_id,
            terminal_reason="completed",
        )
        turn.registry.remove(turn.registry_key)

        # Emit the title update BEFORE the terminal completion so the
        # Streamlit SSE client (which stops reading after `complete`) still
        # receives it.
        if turn.title_task:
            generated_title = await turn.title_task
            if generated_title:
                yield make_event(
                    "title_updated",
                    sequence=turn.next_sequence(),
                    conversation_id=str(turn.conversation_id),
                    data={
                        "title": generated_title,
                        "conversation_id": str(turn.conversation_id),
                    },
                )

        # Yield final completion event with full message
        yield make_event(
            "complete",
            sequence=turn.next_sequence(),
            conversation_id=str(turn.conversation_id),
            message_id=str(bot_message.id),
            data={"message": bot_message.model_dump(mode="json")},
        )

    async def _apersist_interrupted_user_turn(self, turn: _UserTurn) -> None:
        """A disconnect, or a Stop that cancelled this task: keep the partial, settle.

        Writes here are synchronous or shielded on purpose: the task is being
        cancelled, and a second cancellation must not skip what the user is
        waiting for (see :meth:`_create_bot_response_message`).
        """
        turn.cancel_title_task()
        if turn.bot_message_persisted:
            return
        partial = turn.inflight.partial_text.strip()
        assistant_message_id = None
        if partial:
            bot_msg = self._create_bot_response_message(
                conversation_id=turn.conversation_id,
                content=fix_markdown_code_blocks(partial),
                metadata=self._stopped_partial_metadata(turn, "disconnect"),
                message_id=turn.bot_message_id,
            )
            turn.inflight.resolve(bot_msg.model_dump(mode="json"))
            assistant_message_id = turn.bot_message_id
        else:
            turn.inflight.resolve(None)
        turn.registry.remove(turn.registry_key)
        await asyncio.shield(
            self._amark_generation_stopped(
                turn.generation,
                user_id=turn.owner_id,
                assistant_message_id=assistant_message_id,
                # A Stop on this worker cancels the task too; the flag is
                # what tells it from a client that simply went away.
                terminal_reason=("user_requested" if turn.inflight.is_cancelled else "disconnect"),
            )
        )

    async def _afail_user_turn(self, turn: _UserTurn, exc: Exception) -> V3StreamEvent:
        """Persist the failure as the answer, fail the row, and build the error event."""
        error_text = _client_error_text(exc)
        error_message = self._create_bot_response_message(
            conversation_id=turn.conversation_id,
            content=f"Error generating response: {error_text}",
            metadata={"error": error_text},
            message_id=turn.bot_message_id,
        )

        turn.inflight.resolve(error_message.model_dump(mode="json"))
        turn.registry.remove(turn.registry_key)
        await self._amark_generation_failed(
            turn.generation, user_id=turn.owner_id, terminal_reason="stream_exception"
        )
        return make_event(
            "error",
            sequence=turn.next_sequence(),
            conversation_id=str(turn.conversation_id),
            message_id=str(turn.bot_message_id),
            data={
                "error": error_text,
                "message": error_message.model_dump(mode="json"),
            },
        )

    def _validate_and_claim_interrupt_resume(
        self,
        *,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        interrupt_id: str | None,
        device_id: UUID | None,
        decisions: list[InterruptDecision] | None = None,
        claim: bool = True,
    ) -> Any:
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)

        if not interrupt_id:
            raise CustomHTTPException(
                # Starlette's symbolic name differs across the supported range.
                status_code=422,
                detail="interruptId is required to resume a durable approval request.",
                error_code="INTERRUPT_ID_REQUIRED",
            )

        if self.hitl_interrupt_repository:
            record = self._require_resumable_interrupt_record(
                interrupt_id=interrupt_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                device_id=device_id,
            )
            self._validate_complete_interrupt_decisions(record=record, decisions=decisions)
            self._validate_interrupt_runtime_scope(record, interrupt_id, decisions)
            if claim:
                self._claim_interrupt_resume(interrupt_id, conversation_id, user_id)
            return record

        if self.redis_client:
            self._check_legacy_interrupt_expiry(conversation_id, interrupt_id)
        return None

    def _require_resumable_interrupt_record(
        self,
        *,
        interrupt_id: str,
        conversation_id: UUID,
        thread_id: str,
        device_id: UUID | None,
    ) -> Any:
        """The durable record, provided it belongs to this resume and can still take one."""
        record = self.hitl_interrupt_repository.get_by_id(interrupt_id)
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
        if device_id is not None and record.device_id is not None and record.device_id != device_id:
            raise CustomHTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Device ID does not match the pending interrupt state.",
                error_code="INTERRUPT_DEVICE_MISMATCH",
            )
        self._require_interrupt_status_resumable(record, interrupt_id)
        return record

    def _require_interrupt_status_resumable(self, record: Any, interrupt_id: str) -> None:
        now = datetime.now(UTC)
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

    def _validate_interrupt_runtime_scope(
        self,
        record: Any,
        interrupt_id: str,
        decisions: list[InterruptDecision] | None,
    ) -> None:
        """The device session and tool catalog a client-local approval was made against.

        Any drift expires the interrupt: approving a call against a session or
        tool instance that no longer exists would run something nobody saw.
        """
        runtime_provenance = self._get_runtime_validation_provenance(record, decisions=decisions)
        if not runtime_provenance:
            return

        active_session = None
        if record.device_id is not None:
            active_session = ClientDeviceService.lookup_active_session(record.device_id)
        if active_session is None or active_session.user_id != record.user_id:
            self._refuse_for_scope_change(
                interrupt_id,
                resolution_source="runtime_unavailable",
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
            self._validate_provenance_session(provenance, active_session, interrupt_id)
            self._validate_provenance_tool(
                provenance, catalog_by_qid, catalog_instance_ids, interrupt_id
            )

    def _validate_provenance_session(
        self, provenance: dict[str, Any], active_session: Any, interrupt_id: str
    ) -> None:
        expected_session_id = provenance.get("session_id")
        if expected_session_id not in (None, "") and active_session.session_id != str(
            expected_session_id
        ):
            self._refuse_for_scope_change(
                interrupt_id,
                resolution_source="session_changed",
                detail=(
                    "The client device session changed after this approval "
                    "was created. Please send a new message from the active device."
                ),
                error_code="INTERRUPT_SESSION_MISMATCH",
            )

        expected_catalog_version = provenance.get("catalog_version")
        if expected_catalog_version is not None and active_session.tool_catalog_version != int(
            expected_catalog_version
        ):
            self._refuse_for_scope_change(
                interrupt_id,
                resolution_source="catalog_changed",
                detail=(
                    "The client device tool catalog changed after this approval "
                    "was created. Please search for the tool again and retry."
                ),
                error_code="INTERRUPT_CATALOG_MISMATCH",
            )

    def _validate_provenance_tool(
        self,
        provenance: dict[str, Any],
        catalog_by_qid: dict[str, dict[str, Any]],
        catalog_instance_ids: set[str],
        interrupt_id: str,
    ) -> None:
        expected_qualified_id = str(provenance.get("qualified_tool_id") or "").strip()
        expected_tool_instance_id = str(provenance.get("tool_instance_id") or "").strip()
        if not expected_qualified_id:
            if expected_tool_instance_id and expected_tool_instance_id not in catalog_instance_ids:
                self._refuse_tool_instance_changed(interrupt_id)
            return

        catalog_entry = catalog_by_qid.get(expected_qualified_id)
        if catalog_entry is None:
            self._refuse_for_scope_change(
                interrupt_id,
                resolution_source="tool_unavailable",
                detail=(
                    "A client-local tool in this approval is no longer "
                    "available on the active device. Please search again "
                    "and retry."
                ),
                error_code="INTERRUPT_TOOL_UNAVAILABLE",
            )

        current_tool_instance_id = str(catalog_entry.get("tool_instance_id") or "").strip()
        if (
            expected_tool_instance_id
            and current_tool_instance_id
            and expected_tool_instance_id != current_tool_instance_id
        ):
            self._refuse_tool_instance_changed(interrupt_id)

    def _refuse_tool_instance_changed(self, interrupt_id: str) -> NoReturn:
        self._refuse_for_scope_change(
            interrupt_id,
            resolution_source="tool_instance_changed",
            detail=(
                "A client-local tool capability changed after this "
                "approval was created. Please search for the tool "
                "again and retry."
            ),
            error_code="INTERRUPT_TOOL_INSTANCE_MISMATCH",
        )

    def _refuse_for_scope_change(
        self,
        interrupt_id: str,
        *,
        resolution_source: str,
        detail: str,
        error_code: str,
    ) -> NoReturn:
        """Expire an approval whose execution scope moved, and refuse the resume."""
        self._expire_interrupt_for_scope_change(
            interrupt_id=interrupt_id,
            resolution_source=resolution_source,
        )
        raise CustomHTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
        )

    def _claim_interrupt_resume(
        self, interrupt_id: str, conversation_id: UUID, user_id: UUID
    ) -> None:
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

    def _check_legacy_interrupt_expiry(self, conversation_id: UUID, interrupt_id: str) -> None:
        """The Redis timeout, for a deployment with no durable interrupt records.

        A failed read resumes rather than refusing: the timeout is advisory
        here, and the claim that actually serializes a resume is the durable
        record this deployment does not have.
        """
        key = f"interrupt:{conversation_id}:{interrupt_id}"
        try:
            stored_timestamp = self.redis_client.get(key)
            if not stored_timestamp:
                return
            stored_time = datetime.fromisoformat(stored_timestamp.decode("utf-8"))
            elapsed_minutes = (datetime.now(UTC) - stored_time).total_seconds() / 60
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
            logging.warning("Interrupt expiry check failed; resuming", exc_info=True)

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
                logging.warning("Could not read the stored interrupt arguments", exc_info=True)

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
        """Resume an approval-paused turn with the user's decisions.

        The approval pause released the conversation; the resume takes the
        turn's lifecycle row back to ``running`` and, like a first stream,
        settles it on every way out.
        """
        try:
            async with (
                self._hold_turn(conversation_id, request_id=str(bot_message_id or uuid4())),
                contextlib.aclosing(
                    self._aresume_holding_turn(
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        decisions=decisions,
                        interrupt_id=interrupt_id,
                        device_id=device_id,
                        bot_message_id=bot_message_id,
                        inline_rich_response_v1=inline_rich_response_v1,
                    )
                ) as events,
            ):
                async for event in events:
                    yield event
        except WorkflowRoutingException as exc:
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data=workflow_error_payload(exc.error),
            )

    async def _aresume_holding_turn(
        self,
        *,
        thread_id,
        conversation_id,
        user_id,
        decisions,
        interrupt_id,
        device_id,
        bot_message_id,
        inline_rich_response_v1,
    ):
        from app.services.generation_control_service import GenerationControlError

        resume = _ResumeTurn(
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id=interrupt_id,
            bot_message_id=bot_message_id,
        )

        fetched_interrupt_record = self._validate_and_claim_interrupt_resume(
            thread_id=thread_id,
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id=interrupt_id,
            device_id=device_id,
            decisions=decisions,
            claim=False,
        )
        registry = get_generation_registry()
        try:
            # Reclaim before consuming the approval. Conflicts preserve its pending record.
            resume.generation = await self._aresume_approved_generation(
                thread_id=thread_id, conversation_id=conversation_id, user_id=user_id
            )
        except GenerationControlError as exc:
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data={"error": str(exc), "error_code": exc.code, **exc.detail},
            )
            return
        try:
            fetched_interrupt_record = self._validate_and_claim_interrupt_resume(
                thread_id=thread_id,
                conversation_id=conversation_id,
                user_id=user_id,
                interrupt_id=interrupt_id,
                device_id=device_id,
                decisions=decisions,
            )
            # Validate the original paused agent before replacing its registry token.
            await self._aprepare_resume(resume, decisions, fetched_interrupt_record)
            self._register_resumed_generation(resume, registry)
            async with contextlib.aclosing(
                self._astream_resume(
                    resume, decisions, inline_rich_response_v1=inline_rich_response_v1
                )
            ) as events:
                async for event in events:
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            await asyncio.shield(self._apersist_disconnected_resume(resume))
            return
        except CustomHTTPException:
            self._release_resume_claim(resume, "approval_resume_refused")
            await self._amark_generation_failed(
                resume.generation, user_id=user_id, terminal_reason="approval_claim_refused"
            )
            raise
        except Exception as exc:
            if resume.persisted:
                self._release_resume_claim(resume, "stream_exception")
                return
            error_text = _client_error_text(exc)
            yield await self._afail_resume(
                resume,
                error_text=error_text,
                content=f"Error generating response: {error_text}",
                source="stream_exception",
                message_id=resume.bot_message_id,
                blocking=True,
            )
        finally:
            self._release_resumed_generation(resume, registry)

    @staticmethod
    def _register_resumed_generation(resume, registry):
        if resume.generation is not None:
            previous = registry.get(resume.generation.generation_id)
            resume.active_agent_id = (
                getattr(previous, "active_agent_id", None) or resume.active_agent_id
            )
            resume.inflight = registry.register(
                generation_id=resume.generation.generation_id,
                conversation_id=resume.conversation_id,
                user_id=resume.user_id,
                active_agent_id=resume.active_agent_id,
            )
            resume.inflight.task = asyncio.current_task()

    @classmethod
    def _release_resumed_generation(cls, resume, registry):
        if resume.inflight is not None:
            cls._release_generation_entry(
                resume.generation.generation_id, resume.inflight, registry
            )

    @staticmethod
    def _release_generation_entry(generation_id, inflight, registry):
        inflight.resolve()
        if not inflight.paused:
            registry.remove(generation_id)

    @staticmethod
    def _resume_registry_key(resume):
        if resume.generation is not None:
            return resume.generation.generation_id
        from app.ai.workflow.state import parse_checkpoint_thread_id

        try:
            conversation, logical_turn = parse_checkpoint_thread_id(resume.thread_id)
        except ValueError:
            return None
        return logical_turn if conversation == str(resume.conversation_id) else None

    @classmethod
    def _paused_resume_registry_entry(cls, resume):
        key = cls._resume_registry_key(resume)
        if key is None:
            return None, None
        entry = get_generation_registry().get(key)
        if entry is None or not entry.paused:
            return key, None
        if entry.user_id != resume.user_id or entry.conversation_id != resume.conversation_id:
            return key, None
        return key, entry

    @classmethod
    def _clear_paused_resume_entry(cls, resume):
        key, entry = cls._paused_resume_registry_entry(resume)
        if entry is not None:
            get_generation_registry().remove(key)

    async def _aprepare_resume(
        self,
        resume: _ResumeTurn,
        decisions: list[InterruptDecision],
        fetched_interrupt_record: Any,
    ) -> None:
        """Reclaim the row, then load what the resumed run is answered with.

        The row is reclaimed first so that a refusal below (a detached custom
        agent) fails it together with the interrupt, rather than leaving it
        offering an approval nothing can resume any more.
        """
        resume.user_id, resume.sanitized_persona = self._get_conversation_context(
            resume.conversation_id, resume.user_id
        )

        # Revalidate only the agent selected by this turn, including another worker's record.
        _, paused_entry = self._paused_resume_registry_entry(resume)
        metadata = getattr(fetched_interrupt_record, "interrupt_metadata_json", None) or {}
        resume.active_agent_id = getattr(paused_entry, "active_agent_id", None) or metadata.get(
            "active_agent_id"
        )
        self._revalidate_resume_custom_agent(
            resume.user_id, resume.conversation_id, resume.active_agent_id
        )

        self._audit_interrupt_resume_decisions(
            conversation_id=resume.conversation_id,
            user_id=resume.user_id,
            decisions=decisions,
            interrupt_id=resume.interrupt_id,
            fetched_interrupt_record=fetched_interrupt_record,
        )
        resume.custom_agents = self._resolve_custom_agents_state(
            resume.user_id, resume.conversation_id
        )

    async def _astream_resume(
        self,
        resume: _ResumeTurn,
        decisions: list[InterruptDecision],
        *,
        inline_rich_response_v1: bool,
    ):
        """Forward the resumed run, and end it the way the run ended."""
        stream = self.ai_service.resume_interrupted_execution_stream(
            thread_id=resume.thread_id,
            decisions=decisions,
            inline_rich_response_v1=inline_rich_response_v1,
            user_id=resume.user_id,
            conversation_id=resume.conversation_id,
        )
        stop_watch = _DurableStopWatch(
            control=self._generation_control(), generation=resume.generation, user_id=resume.user_id
        )
        async with contextlib.aclosing(stream):
            async for raw_event in stream:
                event = _service_event_from_ai_event(raw_event, sequence=resume.next_sequence())
                if await self._aresume_should_stop(resume, stop_watch, event.type):
                    await self._apersist_disconnected_resume(resume)
                    return
                if event.type in {"interrupt", "complete", "error", "continuation_available"}:
                    for terminal in await self._aresume_terminal_events(resume, event):
                        yield terminal
                    return
                for projected in self._project_resume_event(resume, event):
                    yield projected
        yield await self._afail_resume(
            resume,
            error_text=ERROR_RESPONSE_AFTER_RESUME,
            content=ERROR_RESPONSE_AFTER_RESUME,
            source="stream_incomplete",
            message_id=resume.bot_message_id,
        )

    async def _aresume_terminal_events(self, resume, event):
        if event.type == "interrupt":
            return [await self._areinterrupt_resume(resume, event)]
        if event.type == "complete":
            return [await self._acomplete_resume(resume, event)]
        if event.type == "continuation_available":
            return await self._apause_resume(resume, event)
        error = event.data.get("error", UNKNOWN_ERROR)
        return [
            await self._afail_resume(
                resume,
                error_text=error,
                content=f"Error generating response: {error}",
                source="stream_error",
                message_id=resume.bot_message_id,
            )
        ]

    @staticmethod
    async def _aresume_should_stop(resume, stop_watch, event_type):
        if resume.inflight is None:
            return False
        if resume.inflight.is_cancelled:
            return True
        if await stop_watch.stop_requested(event_type):
            resume.inflight.mark_cancelled()
            return True
        return False

    def _project_resume_event(self, resume: _ResumeTurn, event: V3StreamEvent) -> list:
        """Record what one resumed event contributes, and what to forward."""
        if event.type == "agent_selected":
            resume.active_agent_id = event.agent or event.data.get("agent")
            if resume.inflight is not None:
                resume.inflight.active_agent_id = resume.active_agent_id
            return [
                self._agent_selected_event(
                    resume.active_agent_id, resume.custom_agents, sequence=event.sequence
                )
            ]
        if event.type == "message_delta":
            resume.partial_text += event.data.get("text", "")
            if resume.inflight is not None:
                resume.inflight.partial_text = resume.partial_text
        else:
            _record_stream_tool_event(event, resume.tool_artifacts, resume.tool_args_by_id)
        # reasoning, rich_items (same progressive contract as a first pass),
        # state_snapshot, subagent lifecycle and other canonical events pass
        # through unchanged.
        return [event]

    async def _areinterrupt_resume(self, resume: _ResumeTurn, event: V3StreamEvent):
        """The resumed run asked for another approval: persist it, pause the row again."""
        interrupt_response = event.data.get("interrupt")
        normalized_interrupt = self._normalize_nested_interrupt_payload(interrupt_response)
        next_interrupt_id = (
            interrupt_response.get("interrupt_id") if isinstance(interrupt_response, dict) else None
        )
        interrupt_thread_id = event.data.get("thread_id") or resume.thread_id
        try:
            persisted = self._persist_interrupt_bot_message(
                conversation_id=resume.conversation_id,
                interrupt_payload=normalized_interrupt,
                context=_InterruptPersistContext(
                    sanitized_persona=resume.sanitized_persona,
                    partial_text=resume.partial_text,
                    pending_tool_calls=event.data.get("pending_tool_calls"),
                    thread_id=interrupt_thread_id,
                    next_nodes=event.data.get("next"),
                    user_id=resume.user_id,
                    message_id=resume.bot_message_id,
                    tool_artifacts=resume.tool_artifacts or None,
                    active_agent_id=resume.active_agent_id,
                    custom_agents=resume.custom_agents,
                    require_durable_interrupt=True,
                ),
            )
        except Exception as exc:
            error_text = _client_error_text(exc)
            return await self._afail_resume(
                resume,
                error_text=error_text,
                content=f"Error generating response: {error_text}",
                source="stream_exception",
                message_id=None,
                blocking=True,
                announce_created_message=True,
            )

        self._clear_redis_interrupt(resume.conversation_id, resume.interrupt_id)
        self._mark_resume_interrupt_resolved(resume)
        self._handle_redis_interrupt_storage(
            resume.conversation_id, next_interrupt_id, interrupt_response
        )
        self._set_plan_lifecycle(resume.conversation_id, resume.user_id, PlanLifecycle.paused)
        resume.persisted = True
        paused = await self._amark_generation_awaiting_approval(
            resume.generation, user_id=resume.user_id, assistant_message_id=persisted.id
        )
        if self._approval_pause_was_stopped(paused):
            return self._stopped_approval_event(resume.conversation_id, event.sequence)
        self._retain_approval_registry_entry(resume, persisted)

        return make_event(
            "interrupt",
            sequence=event.sequence,
            conversation_id=str(resume.conversation_id),
            message_id=resume.event_message_id,
            data={
                "thread_id": interrupt_thread_id,
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

    @staticmethod
    def _retain_approval_registry_entry(resume, message):
        if resume.inflight is not None:
            resume.inflight.resolve(message.model_dump(mode="json"))
            resume.inflight.active_agent_id = resume.active_agent_id
            resume.inflight.paused = True
            resume.inflight.task = None

    async def _acomplete_resume(self, resume: _ResumeTurn, event: V3StreamEvent):
        """Persist the resumed answer, resolve the approval, close the row."""
        bot_response = event.data.get("response")
        _merge_stream_tool_artifacts_into_response(bot_response, resume.tool_artifacts)

        bot_message = await self._persist_completed_workflow_response(
            conversation_id=resume.conversation_id,
            user_id=resume.user_id,
            bot_response=bot_response,
            sanitized_persona=resume.sanitized_persona,
            workflow_request=None,
            message_id=resume.bot_message_id,
            fallback_content=ERROR_RESPONSE_AFTER_RESUME,
        )
        resume.persisted = True
        if resume.inflight is not None:
            resume.inflight.resolve(bot_message.model_dump(mode="json"))
        self._clear_redis_interrupt(resume.conversation_id, resume.interrupt_id)
        self._mark_resume_interrupt_resolved(resume)
        await self._compact_checkpoint_after_persist(thread_id=resume.thread_id)

        # Resume resolved the paused run — release its lock token.
        self._clear_paused_resume_entry(resume)
        await self._amark_generation_completed(
            resume.generation,
            user_id=resume.user_id,
            assistant_message_id=bot_message.id,
            terminal_reason="completed",
        )

        return make_event(
            "complete",
            sequence=resume.next_sequence(),
            conversation_id=str(resume.conversation_id),
            message_id=str(bot_message.id),
            data={"message": bot_message.model_dump(mode="json")},
        )

    async def _apause_resume(self, resume: _ResumeTurn, event: V3StreamEvent) -> list:
        """The approved work ran out of budget: persist the partial, offer Continue.

        The same rule as a first pause, through the same publisher. Collected
        before anything is yielded so the approval is settled however soon the
        client stops reading: the decision was consumed either way.
        """
        paused_events = [
            paused_event
            async for paused_event in self._apublish_continuation_pause(
                event,
                generation=resume.generation,
                conversation_id=resume.conversation_id,
                user_id=resume.user_id,
                bot_message_id=resume.bot_message_id or uuid4(),
                sanitized_persona=resume.sanitized_persona,
                workflow_request=None,
                inflight=SimpleNamespace(active_agent_id=resume.active_agent_id),
                tool_artifacts=resume.tool_artifacts or None,
                next_sequence=resume.next_sequence,
            )
        ]
        resume.persisted = True
        self._clear_redis_interrupt(resume.conversation_id, resume.interrupt_id)
        self._clear_paused_resume_entry(resume)
        if any(paused_event.type == "error" for paused_event in paused_events):
            self._mark_claimed_interrupt_failed(resume.interrupt_id, "response_persistence_failed")
        else:
            self._mark_resume_interrupt_resolved(resume)
        return paused_events

    def _mark_resume_interrupt_resolved(self, resume: _ResumeTurn) -> None:
        if self.hitl_interrupt_repository and resume.interrupt_id:
            with contextlib.suppress(Exception):
                self.hitl_interrupt_repository.mark_resolved(resume.interrupt_id)

    def _release_resume_claim(self, resume: _ResumeTurn, source: str) -> None:
        """Give up the claimed approval: its Redis timer, its lock token, its record."""
        self._clear_redis_interrupt(resume.conversation_id, resume.interrupt_id)
        self._clear_paused_resume_entry(resume)
        self._mark_claimed_interrupt_failed(resume.interrupt_id, source)

    async def _afail_resume(
        self,
        resume: _ResumeTurn,
        *,
        error_text: str,
        content: str,
        source: str,
        message_id: UUID | None,
        blocking: bool = False,
        announce_created_message: bool = False,
    ) -> V3StreamEvent:
        """Fail the resume: release the claim, persist the error, fail the row.

        ``blocking`` persists synchronously, for the exception handlers where an
        await could be cancelled out from under the write (see
        :meth:`_create_bot_response_message`). ``announce_created_message``
        names the persisted error message in the event instead of the reserved
        id, for the one caller that persists without one.
        """
        self._release_resume_claim(resume, source)
        message_kwargs = {
            "conversation_id": resume.conversation_id,
            "content": content,
            "metadata": {"error": error_text},
            "message_id": message_id,
        }
        try:
            if blocking:
                error_message = self._create_bot_response_message(**message_kwargs)
            else:
                error_message = await self._acreate_bot_response_message(**message_kwargs)
        finally:
            resume.persisted = True
            await asyncio.shield(
                self._amark_generation_failed(
                    resume.generation, user_id=resume.user_id, terminal_reason=source
                )
            )
        return make_event(
            "error",
            sequence=resume.next_sequence(),
            conversation_id=str(resume.conversation_id),
            message_id=(
                str(error_message.id) if announce_created_message else resume.event_message_id
            ),
            data={
                "error": error_text,
                "message": error_message.model_dump(mode="json"),
                "error_code": "INTERRUPT_FAILED",
                "status_code": 500,
            },
        )

    @staticmethod
    def _resume_stop_reason(resume):
        return (
            "user_requested" if resume.inflight and resume.inflight.is_cancelled else "disconnect"
        )

    async def _apersist_disconnected_resume(self, resume: _ResumeTurn) -> None:
        """The client went away mid-resume: keep the partial, stop the row."""
        self._clear_paused_resume_entry(resume)
        self._mark_claimed_interrupt_failed(resume.interrupt_id, "client_disconnect")
        if resume.persisted:
            return

        assistant_message_id = None
        try:
            partial = resume.partial_text.strip()
            if partial:
                bot_message = await self._acreate_bot_response_message(
                    conversation_id=resume.conversation_id,
                    content=fix_markdown_code_blocks(partial),
                    metadata={
                        "stopped": True,
                        "partial": True,
                        "stop_reason": self._resume_stop_reason(resume),
                        "persona_used": resume.sanitized_persona,
                    },
                    message_id=resume.bot_message_id,
                )
                assistant_message_id = bot_message.id
                if resume.inflight is not None:
                    resume.inflight.resolve(bot_message.model_dump(mode="json"))
        finally:
            resume.persisted = True
            await asyncio.shield(
                self._amark_generation_stopped(
                    resume.generation,
                    user_id=resume.user_id,
                    assistant_message_id=assistant_message_id,
                    terminal_reason=self._resume_stop_reason(resume),
                )
            )

    async def continue_message_generation_stream(
        self,
        *,
        generation_id: UUID,
        continuation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        idempotency_key: str,
        expected_version: int,
        bot_message_id: UUID | None = None,
        inline_rich_response_v1: bool = False,
    ):
        """Resume a paused turn from its exact checkpoint.

        What this deliberately does not do is as much of the contract as what
        it does. It does not call ``create_message_stream``, the router,
        ``sendMessage``, or regenerate, and it appends no user message —
        synthetic or hidden. Continue is more of the *same* answer to the
        *same* question; anything that adds a turn would re-route it and could
        land it on a different specialist than the one holding the evidence.

        The conversation lock is reacquired for the same reason a new turn
        holds it: this writes an assistant message for the conversation.
        """
        control = self._generation_control()
        if control is None:
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data={"error": "Generation controls are not enabled on this deployment."},
            )
            return

        await self.conversation_validation_utils.avalidate_conversation_access(
            user_id, conversation_id
        )
        if bot_message_id is None:
            bot_message_id = uuid4()

        try:
            async with (
                self._hold_turn(conversation_id, request_id=str(bot_message_id)),
                # Closed here, under the lock, rather than whenever the
                # collector gets to it: its cleanup settles the lifecycle row.
                contextlib.aclosing(
                    self._acontinue_holding_turn(
                        generation_id=generation_id,
                        continuation_id=continuation_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        idempotency_key=idempotency_key,
                        expected_version=expected_version,
                        bot_message_id=bot_message_id,
                        inline_rich_response_v1=inline_rich_response_v1,
                    )
                ) as events,
            ):
                async for event in events:
                    yield event
        except WorkflowRoutingException as exc:
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data=workflow_error_payload(exc.error),
            )

    async def _acontinue_holding_turn(
        self,
        *,
        generation_id: UUID,
        continuation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        idempotency_key: str,
        expected_version: int,
        bot_message_id: UUID,
        inline_rich_response_v1: bool,
    ):
        """Lease the epoch, then stream it. Refusals stay typed."""
        from app.schemas.generation import ContinueGenerationCommand
        from app.services.generation_control_service import (
            ContinuationUnavailable,
            GenerationControlError,
        )

        control = self._generation_control()
        try:
            leased = await control.lease_continuation(
                ContinueGenerationCommand(
                    generation_id=generation_id,
                    continuation_id=continuation_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key,
                    expected_version=expected_version,
                    inline_rich_response_v1=inline_rich_response_v1,
                )
            )
        except GenerationControlError as exc:
            # A refusal is the answer, not a broken stream: the client asked
            # for something the lifecycle will not allow and the reason is safe
            # to say.
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data={"error": str(exc), "error_code": exc.code, **exc.detail},
            )
            return
        if leased.replayed:
            # The first attempt with this key already resumed the graph. Its
            # lease names an epoch the checkpoint has left, and the pause node
            # answers a stale epoch by finalizing the pause that is live now.
            yield make_event(
                "error",
                sequence=0,
                conversation_id=str(conversation_id),
                data={
                    "error": "This Continue was already redeemed.",
                    "error_code": ContinuationUnavailable.code,
                    "replayed": True,
                },
            )
            return
        lease = leased.lease

        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        registry = get_generation_registry()
        inflight = registry.register(
            generation_id=generation_id,
            conversation_id=conversation_id,
            user_id=user_id,
            active_agent_id=lease.active_agent_id
            or getattr(registry.get(generation_id), "active_agent_id", None),
        )
        inflight.task = asyncio.current_task()

        try:
            # R4: restore the turn's research accounting before anything can spend
            # it. An unreadable payload fails the Continue rather than proceeding,
            # because an empty budget looks exactly like a fresh turn's full quota
            # and the epoch would re-run every search the last one already paid for.
            try:
                self._install_research_accounting(lease, conversation_id=conversation_id)
            except Exception as exc:
                logging.warning(
                    "Refusing a continuation whose research accounting is unreadable: %s", exc
                )
                await self._amark_generation_failed(
                    lease.snapshot,
                    user_id=user_id,
                    terminal_reason="research_accounting_unreadable",
                )
                yield make_event(
                    "error",
                    sequence=_next_sequence(),
                    conversation_id=str(conversation_id),
                    data={
                        "error": (
                            "This answer cannot be continued: its research accounting could "
                            "not be restored."
                        ),
                        "error_code": "research_accounting_unreadable",
                    },
                )
                return

            yield make_event(
                "run_start",
                sequence=_next_sequence(),
                conversation_id=str(conversation_id),
                message_id=str(bot_message_id),
                data=self._generation_status_data(lease.snapshot),
            )

            async with contextlib.aclosing(
                self._astream_continuation(
                    lease,
                    continuation_id=continuation_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    bot_message_id=bot_message_id,
                    inline_rich_response_v1=inline_rich_response_v1,
                    inflight=inflight,
                    next_sequence=_next_sequence,
                )
            ) as events:
                async for event in events:
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            await asyncio.shield(
                self._astop_continuation(
                    lease,
                    inflight=inflight,
                    user_id=user_id,
                    bot_message_id=bot_message_id,
                    reason="user_requested" if inflight.is_cancelled else "disconnect",
                )
            )
            raise
        except Exception as exc:
            registry.remove(generation_id)
            await self._amark_generation_failed(
                lease.snapshot, user_id=user_id, terminal_reason="stream_exception"
            )
            yield make_event(
                "error",
                sequence=_next_sequence(),
                conversation_id=str(conversation_id),
                message_id=str(bot_message_id),
                data={"error": _client_error_text(exc)},
            )
            return

        finally:
            self._release_generation_entry(generation_id, inflight, registry)

    async def _astop_continuation(self, lease, *, inflight, user_id, bot_message_id, reason):
        assistant_message_id = None
        try:
            if inflight.partial_text.strip() and not inflight.done.done():
                message = await self._acreate_bot_response_message(
                    conversation_id=lease.snapshot.conversation_id,
                    content=fix_markdown_code_blocks(inflight.partial_text.strip()),
                    metadata={
                        **self._continuation_metadata(lease),
                        "stopped": True,
                        "stop_reason": reason,
                    },
                    message_id=bot_message_id,
                )
                assistant_message_id = message.id
                inflight.resolve(message.model_dump(mode="json"))
        finally:
            await self._amark_generation_stopped(
                lease.snapshot,
                user_id=user_id,
                assistant_message_id=assistant_message_id,
                terminal_reason=reason,
            )
            inflight.resolve()

    async def _astream_continuation(
        self,
        lease: Any,
        *,
        continuation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        bot_message_id: UUID,
        inline_rich_response_v1: bool,
        inflight: Any,
        next_sequence: Any,
    ):
        """Project the resumed graph's events, persisting whatever it ends with.

        The epoch handed back to the graph is ``paused_epoch``, not the leased
        one: the row has advanced but the checkpoint has not, and the pause node
        fences against its own state.
        """
        stream = self.ai_service.resume_generation_control_stream(
            thread_id=lease.checkpoint_thread_id,
            action="continue",
            continuation_id=str(continuation_id),
            expected_epoch=lease.paused_epoch,
            inline_rich_response_v1=inline_rich_response_v1,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        tool_artifacts = []
        tool_args_by_id = {}
        stop_watch = _DurableStopWatch(
            control=self._generation_control(), generation=lease.snapshot, user_id=user_id
        )
        async with contextlib.aclosing(stream):
            async for raw_event in stream:
                event = _service_event_from_ai_event(raw_event, sequence=next_sequence())
                if inflight.is_cancelled or await stop_watch.stop_requested(event.type):
                    inflight.mark_cancelled()
                    break
                events = await self._ahandle_continuation_event(
                    event,
                    lease=lease,
                    inflight=inflight,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    bot_message_id=bot_message_id,
                    next_sequence=next_sequence,
                    tool_artifacts=tool_artifacts,
                    tool_args_by_id=tool_args_by_id,
                )
                for projected in events:
                    yield projected
                if event.type in {"interrupt", "error", "continuation_available", "complete"}:
                    return

        if inflight.is_cancelled:
            await self._astop_continuation(
                lease,
                inflight=inflight,
                user_id=user_id,
                bot_message_id=bot_message_id,
                reason="user_requested",
            )
            return
        await self._asettle_unanswered_epoch(lease, inflight=inflight, user_id=user_id)

    async def _ahandle_continuation_event(
        self,
        event,
        *,
        lease,
        inflight,
        user_id,
        conversation_id,
        bot_message_id,
        next_sequence,
        tool_artifacts,
        tool_args_by_id,
    ):
        """Settle terminal events before publication; retain streamed partial text."""
        if event.type == "interrupt":
            resume = _ResumeTurn(
                thread_id=lease.checkpoint_thread_id,
                conversation_id=conversation_id,
                user_id=user_id,
                interrupt_id=None,
                bot_message_id=bot_message_id,
                generation=lease.snapshot,
                active_agent_id=inflight.active_agent_id,
                inflight=inflight,
                partial_text=inflight.partial_text,
                tool_artifacts=tool_artifacts,
            )
            paused = await self._areinterrupt_resume(resume, event)
            inflight.resolve(paused.data.get("message"))
            return [paused]
        if event.type == "error":
            await self._amark_generation_failed(
                lease.snapshot, user_id=user_id, terminal_reason="stream_error"
            )
            inflight.resolve()
            return [event]
        if event.type == "continuation_available":
            async with contextlib.aclosing(
                self._apublish_continuation_pause(
                    event,
                    generation=lease.snapshot,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    bot_message_id=bot_message_id,
                    sanitized_persona=None,
                    workflow_request=None,
                    inflight=inflight,
                    tool_artifacts=tool_artifacts,
                    next_sequence=next_sequence,
                )
            ) as paused:
                return [item async for item in paused]
        if event.type == "complete":
            response = event.data.get("response")
            if response is None:
                await self._asettle_unanswered_epoch(lease, inflight=inflight, user_id=user_id)
                return []
            _merge_stream_tool_artifacts_into_response(response, tool_artifacts)
            return [
                await self._acomplete_continuation(
                    response,
                    lease=lease,
                    inflight=inflight,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    bot_message_id=bot_message_id,
                    next_sequence=next_sequence,
                )
            ]
        if event.type == "message_delta":
            inflight.partial_text += event.data.get("text", "")
        _record_stream_tool_event(event, tool_artifacts, tool_args_by_id)
        inflight.touch()
        return [event]

    async def _acomplete_continuation(
        self,
        bot_response,
        *,
        lease,
        inflight,
        user_id,
        conversation_id,
        bot_message_id,
        next_sequence,
    ):
        """Persist the continued answer and resolve Stop callers before publishing it."""
        bot_response_content = fix_markdown_code_blocks(
            str(getattr(getattr(bot_response, "message", None), "content", "") or "")
            or inflight.partial_text
        )
        bot_response_content = finalize_article_content(bot_response, bot_response_content)
        metadata = build_bot_metadata(bot_response)
        metadata.update(self._continuation_metadata(lease))
        self._externalize_generated_images(metadata, conversation_id, user_id)
        bot_response_content, metadata = await self._externalize_remote_rich_images(
            bot_response_content,
            metadata,
            conversation_id,
            user_id,
        )
        bot_message = await self._acreate_bot_response_message(
            conversation_id=conversation_id,
            content=bot_response_content,
            metadata=metadata,
            message_id=bot_message_id,
        )
        await self._mark_persisted_web_images(
            metadata,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        inflight.resolve(bot_message.model_dump(mode="json"))
        await self._amark_generation_completed(
            lease.snapshot,
            user_id=user_id,
            assistant_message_id=bot_message_id,
            partial=True,
            terminal_reason="continued",
        )

        return make_event(
            "complete",
            sequence=next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data={"message": bot_message.model_dump(mode="json")},
        )

    async def _asettle_unanswered_epoch(self, lease: Any, *, inflight: Any, user_id: UUID):
        """Close the row for a continued epoch that ended without an answer.

        Only a cancelled epoch is a stop. One that simply ended failed, and
        calling it ``user_requested`` would blame the user for it.
        """
        if inflight.is_cancelled:
            return await self._amark_generation_stopped(
                lease.snapshot,
                user_id=user_id,
                assistant_message_id=None,
                terminal_reason="user_requested",
            )
        return await self._amark_generation_failed(
            lease.snapshot, user_id=user_id, terminal_reason="stream_incomplete"
        )

    @staticmethod
    def _continuation_metadata(lease: Any) -> dict[str, Any]:
        """Metadata for the assistant message a continued epoch produced.

        ``partial`` stays true across the whole chain. The answer was assembled
        over more than one epoch, and a reader that treats the last one as the
        whole answer would misattribute where the evidence came from.
        """
        return {
            "partial": True,
            "continued": True,
            "generation_id": str(lease.snapshot.generation_id),
            "logical_turn_id": lease.snapshot.logical_turn_id,
            "execution_epoch": lease.execution_epoch,
        }

    async def stop_message_generation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_message_id: UUID,
        wait_seconds: float = 5.0,
    ) -> dict:
        """Request cancellation of an in-flight streaming generation.

        Returns a dict with:

        * ``status`` -- ``"cancelled"`` when the producer confirmed and its
          partial is persisted, ``"stop_requested"`` when it has not answered
          yet, ``"not_inflight"`` when this process holds no such generation.
        * ``message`` -- the persisted partial/final message, when there is one.

        ``stop_requested`` is not a failure and not a lie. The producer may be
        mid-provider-call, possibly in another process, and reporting
        ``cancelled`` before it confirms tells the user their tool call was
        abandoned when it may still be running.

        The registry entry survives a timeout deliberately. The request gave up;
        the producer has not, and removing the entry here is what left a retried
        Stop with nothing to cancel while the turn was still going.

        This is the turn-scoped entry point: the caller holds a user message id,
        which is the logical turn. It resolves that to the durable generation
        and delegates, so there is one Stop implementation rather than a
        process-local one beside a durable one.
        """
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)

        control = self._generation_control()
        if control is None:
            return await self._astop_locally(
                conversation_id, user_id, user_message_id, wait_seconds
            )

        snapshot = await control.find_by_logical_turn(
            logical_turn_id=str(user_message_id),
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if snapshot is None:
            return {"status": "not_inflight", "message": None}

        result = await self.stop_generation(
            generation_id=snapshot.generation_id,
            conversation_id=conversation_id,
            user_id=user_id,
            # Derived, not client-supplied: this endpoint predates the fenced
            # command and has no key to send. A retry therefore genuinely
            # re-issues rather than replaying, which the durable transition
            # already makes idempotent.
            idempotency_key=f"legacy-stop-{snapshot.generation_id}",
            expected_version=snapshot.version,
        )
        return self._legacy_stop_result(result, user_id=user_id)

    async def aget_generation(
        self,
        *,
        generation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
    ):
        """The current lifecycle state, or ``None`` if it is not this owner's.

        The read a client polls after a ``stop_requested``, and the one a
        reconnecting client uses to find out whether the turn it lost is still
        running, finished, or waiting to be continued. ``None`` rather than a
        refusal, because "not yours" and "not there" must be indistinguishable.
        """
        control = self._generation_control()
        if control is None:
            return None
        self.conversation_validation_utils.validate_conversation_access(user_id, conversation_id)
        return await control.aget_snapshot(
            generation_id=generation_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def stop_generation(
        self,
        *,
        generation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        idempotency_key: str,
        expected_version: int,
    ):
        """Stop one generation durably, and report only what is true.

        The durable transition comes first and the local shortcut second, in
        that order on purpose: a worker that misses the in-process signal still
        finds ``stop_requested`` on its next check, whereas a signal sent
        without the transition reaches only this process.

        A wait timeout returns ``stop_requested``. It is a successful pending
        state, not an exception and not ``stopped`` — the worker may be
        mid-provider-call in another process, and claiming it stopped would tell
        a user their tool call was abandoned when it may still be running.
        """
        control = self._generation_control()
        if control is None:
            raise RuntimeError("generation controls are not enabled on this deployment")

        from app.models.generation import TERMINAL_STATUSES
        from app.schemas.generation import StopGenerationCommand
        from app.services.generation_control_service import IllegalTransition

        # The authorization and the durable transition. Its result is
        # deliberately not returned: a stop this worker owns may still be
        # settling, and `await_stop_settled` below is what reports the state
        # the turn actually reached.
        try:
            requested = await control.request_stop(
                StopGenerationCommand(
                    generation_id=generation_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key,
                    expected_version=expected_version,
                )
            )
        except IllegalTransition:
            # A Stop that lost a race with the turn's own ending. The command is
            # genuinely illegal — nothing is left to stop — but the user asked
            # for the turn to be over and it is, so reporting where it landed is
            # the idempotent answer rather than an error. Narrow on purpose: an
            # illegal transition on a row that is still active is a real defect
            # and stays an exception.
            settled = await control.aget_snapshot(
                generation_id=generation_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if settled is None or settled.status not in TERMINAL_STATUSES:
                raise
            self._release_stopped_approval_entry(generation_id, settled)
            return settled

        if self._release_stopped_approval_entry(generation_id, requested):
            return requested

        # Accelerate the owning worker if it happens to be this one. The entry
        # is left in place: removing it here is what left a retried Stop with
        # nothing to cancel while the turn was still running.
        get_generation_registry().request_cancel(generation_id)

        return await control.await_stop_settled(
            generation_id=generation_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _release_stopped_approval_entry(generation_id, snapshot):
        from app.models.generation import TERMINAL_STATUSES

        if snapshot.status not in TERMINAL_STATUSES:
            return False
        registry = get_generation_registry()
        entry = registry.get(generation_id)
        if entry is not None and entry.paused:
            entry.resolve()
            registry.remove(generation_id)
        return True

    async def _astop_locally(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_message_id: UUID,
        wait_seconds: float,
    ) -> dict:
        """Process-local Stop, for a deployment with no lifecycle service wired.

        Kept only for that case. It cannot stop a turn owned by another worker,
        which is the whole reason the durable lifecycle exists.
        """
        registry = get_generation_registry()
        entry = registry.get(user_message_id)

        # A missing entry and one owned by somebody else answer identically:
        # "it exists but is not yours" is information about another user's
        # conversation.
        if entry is None:
            return {"status": "not_inflight", "message": None}
        if entry.conversation_id != conversation_id or entry.user_id != user_id:
            return {"status": "not_inflight", "message": None}

        entry.request_cancel()

        try:
            result = await asyncio.wait_for(asyncio.shield(entry.done), timeout=wait_seconds)
        except (TimeoutError, asyncio.CancelledError):
            return {"status": "stop_requested", "message": None}

        registry.remove(user_message_id)
        return {"status": "cancelled", "message": result}

    def _legacy_stop_result(self, snapshot: Any, *, user_id: UUID) -> dict:
        """Project a lifecycle snapshot into the legacy Stop response shape.

        ``cancelled`` is kept as the name for a confirmed stop because existing
        clients switch on it. The durable statuses travel alongside under
        ``generation``, which is what a client should move to reading.
        """
        from app.models.generation import GenerationStatus

        confirmed = {
            GenerationStatus.STOPPED,
            GenerationStatus.COMPLETED_PARTIAL,
            GenerationStatus.COMPLETED,
        }
        status = "cancelled" if snapshot.status in confirmed else "stop_requested"

        message = None
        if snapshot.assistant_message_id is not None:
            # Best effort. The status is the answer; the message is a
            # convenience, and a read failure must not turn a successful stop
            # into an error.
            with contextlib.suppress(Exception):
                message = self.get_by_id(snapshot.assistant_message_id, user_id).model_dump(
                    mode="json"
                )

        return {
            "status": status,
            "message": message,
            "generation": self._generation_status_data(snapshot),
        }

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
        # A marker whose id resolves to nothing renders as "rich item <id> is
        # unavailable". That is right for an item that existed and failed to
        # register — the reader is told a visual is missing — but a model
        # invented id names nothing that ever existed, so the caption is pure
        # noise about the model's own mistake. Strip those markers and let the
        # prose stand; the warning list is recomputed from the cleaned content.
        for warning in validate_rich_references(updated_content, kept_items):
            updated_content = remove_inline_rich_reference(updated_content, warning["id"])
        updated_content = updated_content.strip()
        updated["rich_reference_warnings"] = validate_rich_references(
            updated_content,
            kept_items,
        )
        self._record_final_image_selection(kept_items)
        return updated_content, updated

    async def _mark_persisted_web_images(
        self,
        metadata: dict[str, Any],
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> None:
        service = getattr(self, "web_image_service", None)
        if service is None:
            return
        reference_ids: list[UUID] = []
        for item in metadata.get("rich_items") or ():
            payload = item.get("payload") if isinstance(item, dict) else None
            locators = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(locators, list):
                locators = [payload]
            for locator in locators:
                url = locator.get("url") if isinstance(locator, dict) else None
                if not isinstance(url, str) or not url.startswith("/web-images/"):
                    continue
                try:
                    reference_ids.append(UUID(url.removeprefix("/web-images/").split("?", 1)[0]))
                except ValueError:
                    continue
        if reference_ids:
            await service.mark_selected(
                list(dict.fromkeys(reference_ids)),
                user_id=user_id,
                conversation_id=conversation_id,
            )

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
        try:
            reference = await self.web_image_service.register(
                conversation_id=conversation_id,
                user_id=user_id,
                upstream_url=image_url,
                expected_mime=expected_mime,
                provider=provider,
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
        if self.project_context_service is None:
            sanitized_persona = sanitize_persona(
                conversation.persona_prompt if conversation else None
            )
        else:
            # Async twin: this runs on the event loop before the first token.
            sanitized_persona = await self.project_context_service.aresolve_system_instruction(
                conversation
            )
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
    def _attach_active_agent_metadata(
        metadata: dict[str, Any],
        active_agent_id: str | None,
        custom_agents: dict[str, Any] | None,
    ) -> None:
        if not active_agent_id:
            return

        from app.ai.agent_metadata import attach_agent_metadata

        attach_agent_metadata(
            metadata,
            response_agent_id=active_agent_id,
            active_agent_id=active_agent_id,
            custom_agents=custom_agents,
        )

    def _revalidate_resume_custom_agent(
        self,
        owner_id: UUID | None,
        conversation_id: UUID,
        active_agent_id: str | None = None,
    ) -> None:
        """Refuse this turn when its selected custom agent has been deleted or detached."""
        if getattr(self, "custom_agent_service", None) is None or not owner_id:
            return
        from app.ai.custom_agent_runtime import is_custom_runtime_id

        if not active_agent_id or not is_custom_runtime_id(active_agent_id):
            return
        attached = self._resolve_custom_agents_state(owner_id, conversation_id)
        if active_agent_id not in attached:
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
                logging.warning("Could not attach the next planned task", exc_info=True)

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
        if user_id is not None:
            await self._mark_persisted_web_images(
                bot_metadata, conversation_id=conversation_id, user_id=user_id
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
