from __future__ import annotations

import asyncio
import contextlib
import logging
import time
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
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
        self._attach_active_agent_metadata(
            metadata,
            payload.get("active_agent_id") or inflight.active_agent_id,
            getattr(workflow_request, "custom_agents", None),
        )

        try:
            bot_message = await self._acreate_bot_response_message(
                conversation_id=conversation_id,
                content=fix_markdown_code_blocks(content) if content else content,
                metadata=metadata,
                message_id=bot_message_id,
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

        try:
            offered = await self._amark_generation_continuable(
                generation,
                user_id=user_id,
                assistant_message_id=bot_message_id,
                execution_budget=budget,
                research_accounting=self._research_accounting_snapshot(
                    payload.get("logical_turn_id"), conversation_id
                ),
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

        if offered is None:
            return

        yield make_event(
            "continuation_available",
            sequence=next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data=self._generation_status_data(offered),
        )

    async def _amark_generation_failed(self, generation: Any, *, user_id: UUID):
        """Record a turn that could not produce a readable answer."""
        control = self._generation_control()
        if control is None or generation is None:
            return None

        try:
            return await control.mark_failed(
                generation_id=generation.generation_id,
                user_id=user_id,
                conversation_id=generation.conversation_id,
                expected_version=generation.version,
                terminal_reason="response_persistence_failed",
            )
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
        """
        control = self._generation_control()
        if control is None or generation is None:
            return None

        from app.schemas.generation import MarkCompleted

        try:
            return await control.mark_completed(
                MarkCompleted(
                    generation_id=generation.generation_id,
                    conversation_id=generation.conversation_id,
                    user_id=user_id,
                    expected_version=generation.version,
                    assistant_message_id=assistant_message_id,
                    partial=partial,
                    terminal_reason=terminal_reason,
                )
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
        """
        control = self._generation_control()
        if control is None or generation is None:
            return None

        from app.schemas.generation import MarkContinuable

        return await control.mark_continuable(
            MarkContinuable(
                generation_id=generation.generation_id,
                conversation_id=generation.conversation_id,
                user_id=user_id,
                expected_version=generation.version,
                assistant_message_id=assistant_message_id,
                execution_budget=execution_budget,
                research_accounting=research_accounting,
                continuation_block_reason=block_reason,
            )
        )

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

        from app.schemas.generation import MarkStopped

        try:
            return await control.mark_stopped(
                MarkStopped(
                    generation_id=generation.generation_id,
                    conversation_id=generation.conversation_id,
                    user_id=user_id,
                    expected_version=generation.version,
                    assistant_message_id=assistant_message_id,
                    terminal_reason=terminal_reason,
                )
            )
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
        active_agent_id: str | None = None,
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

        self._attach_active_agent_metadata(metadata, active_agent_id, custom_agents)

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
            async with self._hold_turn(
                message_create_data.conversation_id, request_id=str(bot_message_id)
            ):
                async for event in self._create_message_stream_holding_turn(
                    message_create_data, user_id, bot_message_id
                ):
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

        # The durable row comes before the first streamed event, so a Stop that
        # arrives on the very first token already has something to transition.
        generation = await self._astart_generation(
            conversation_id=message_create_data.conversation_id,
            user_id=user_id,
            logical_turn_id=user_message_id,
        )

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

        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        if generation is not None:
            # The canonical opening event of a turn. Everything a client needs
            # to address a later Stop or Continue is here, including the version
            # that fences them (R5).
            yield make_event(
                "run_start",
                sequence=_next_sequence(),
                conversation_id=str(message_create_data.conversation_id),
                message_id=str(bot_message_id),
                data=self._generation_status_data(generation),
            )

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

            # `starting` to `running` before the graph is entered. A turn that
            # paused at its budget while still `starting` could not be offered
            # a Continue, because `continuable` is not legal from there.
            generation = await self._amark_generation_running(
                generation, user_id=resolved_user_id or user_id
            )

            # Stream bot response generation
            bot_response = None
            bot_message_persisted = False
            stream_tool_artifacts: list[dict[str, Any]] = []
            stream_tool_args_by_id: dict[str, Any] = {}

            stop_watch = _DurableStopWatch(
                control=self._generation_control(),
                generation=generation,
                user_id=resolved_user_id or user_id,
            )

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

                    # The row is the authority on whether this turn was asked
                    # to stop, and it is the only thing a Stop that landed on
                    # another worker could have changed. Polled at coarse
                    # boundaries rather than per token — see _DurableStopWatch.
                    if await stop_watch.stop_requested(event_type):
                        logging.info(
                            "Stream stopping on durable status for generation=%s",
                            registry_key,
                        )
                        # `mark_cancelled`, not `request_cancel`: this producer
                        # is stopping itself and is already at a check point.
                        # Cancelling its own task would raise out of the very
                        # code below that persists the partial.
                        inflight.mark_cancelled()
                        break

                    if event_type == "agent_selected":
                        active_agent_id = event.agent or event.data.get("agent")
                        inflight.active_agent_id = active_agent_id
                        inflight.touch()
                        yield self._agent_selected_event(
                            active_agent_id,
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
                                    active_agent_id=inflight.active_agent_id,
                                    custom_agents=workflow_request.custom_agents,
                                ).model_dump(mode="json"),
                            },
                        )
                        # Workflow is paused - don't create a bot message yet.
                        # Keep the entry as a paused lock token (carrying the
                        # resolved active_agent_id) so a custom agent cannot be
                        # edited/deleted/detached while this run can still resume.
                        _cancel_title_task()
                        inflight.resolve()
                        registry.mark_paused(registry_key)
                        return

                    elif event_type == "continuation_available":
                        # The turn paused at its execution budget with a
                        # validated partial answer. Persist first, offer second:
                        # the continuation id a client redeems must point at an
                        # answer that is already saved, or Continue would resume
                        # work whose first half was never written down.
                        _cancel_title_task()
                        async for paused_event in self._apublish_continuation_pause(
                            event,
                            generation=generation,
                            conversation_id=message_create_data.conversation_id,
                            user_id=resolved_user_id or user_id,
                            bot_message_id=bot_message_id,
                            sanitized_persona=sanitized_persona,
                            workflow_request=workflow_request,
                            inflight=inflight,
                            tool_artifacts=stream_tool_artifacts or None,
                            next_sequence=_next_sequence,
                        ):
                            yield paused_event
                        inflight.resolve()
                        registry.mark_paused(registry_key)
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
                        self._attach_active_agent_metadata(
                            metadata,
                            inflight.active_agent_id,
                            workflow_request.custom_agents,
                        )
                        bot_message = await self._acreate_bot_response_message(
                            conversation_id=message_create_data.conversation_id,
                            content=partial,
                            metadata=metadata,
                            message_id=bot_message_id,
                        )
                        inflight.resolve(bot_message.model_dump(mode="json"))
                        # The worker confirming it let go. Only this side can
                        # make the transition, which is what turns a client's
                        # `stop_requested` into an authoritative `stopped`.
                        await self._amark_generation_stopped(
                            generation,
                            user_id=resolved_user_id or user_id,
                            assistant_message_id=bot_message_id,
                            terminal_reason="user_requested",
                        )
                    else:
                        inflight.resolve(None)
                        await self._amark_generation_stopped(
                            generation,
                            user_id=resolved_user_id or user_id,
                            assistant_message_id=None,
                            terminal_reason="user_requested",
                        )
                    registry.remove(registry_key)
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
                await self._amark_generation_completed(
                    generation,
                    user_id=resolved_user_id or user_id,
                    assistant_message_id=bot_message_id,
                    terminal_reason="completed",
                )
                registry.remove(registry_key)

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
                        self._attach_active_agent_metadata(
                            metadata,
                            inflight.active_agent_id,
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
                    registry.remove(registry_key)
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
                registry.remove(registry_key)

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
        resume_active_agent_id: str | None = None

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
                    resume_active_agent_id = event.agent or event.data.get("agent")
                    yield self._agent_selected_event(
                        resume_active_agent_id,
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
                            active_agent_id=resume_active_agent_id,
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
            async with self._hold_turn(conversation_id, request_id=str(bot_message_id)):
                async for event in self._acontinue_holding_turn(
                    generation_id=generation_id,
                    continuation_id=continuation_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key,
                    expected_version=expected_version,
                    bot_message_id=bot_message_id,
                    inline_rich_response_v1=inline_rich_response_v1,
                ):
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
        from app.services.generation_control_service import GenerationControlError

        control = self._generation_control()
        try:
            lease = await control.prepare_continue(
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

        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

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

        registry = get_generation_registry()
        inflight = registry.register(
            generation_id=generation_id,
            conversation_id=conversation_id,
            user_id=user_id,
            active_agent_id=lease.active_agent_id,
        )
        inflight.task = asyncio.current_task()

        yield make_event(
            "run_start",
            sequence=_next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data=self._generation_status_data(lease.snapshot),
        )

        async for event in self._astream_continuation(
            lease,
            continuation_id=continuation_id,
            conversation_id=conversation_id,
            user_id=user_id,
            bot_message_id=bot_message_id,
            inline_rich_response_v1=inline_rich_response_v1,
            inflight=inflight,
            next_sequence=_next_sequence,
        ):
            yield event

        registry.remove(generation_id)

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
        bot_response = None
        partial_text = ""

        stream = self.ai_service.resume_generation_control_stream(
            thread_id=lease.checkpoint_thread_id,
            action="continue",
            continuation_id=str(continuation_id),
            expected_epoch=lease.paused_epoch,
            inline_rich_response_v1=inline_rich_response_v1,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        async for raw_event in stream:
            if inflight.is_cancelled:
                break
            event = _service_event_from_ai_event(raw_event, sequence=next_sequence())
            if event.type == "message_delta":
                partial_text += event.data.get("text", "")
                inflight.partial_text = partial_text
                inflight.touch()
                yield event
            elif event.type == "complete":
                bot_response = event.data.get("response")
                break
            elif event.type == "error":
                yield event
                return
            elif event.type == "continuation_available":
                # The turn ran out of budget again. Same rule as the first
                # pause, and the epoch cap in `validate_output` is what stops
                # this from recurring forever.
                async for paused_event in self._apublish_continuation_pause(
                    event,
                    generation=lease.snapshot,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    bot_message_id=bot_message_id,
                    sanitized_persona=None,
                    workflow_request=None,
                    inflight=inflight,
                    tool_artifacts=None,
                    next_sequence=next_sequence,
                ):
                    yield paused_event
                return
            else:
                inflight.touch()
                yield event

        if bot_response is None:
            await self._amark_generation_stopped(
                lease.snapshot,
                user_id=user_id,
                assistant_message_id=None,
                terminal_reason="user_requested",
            )
            return

        bot_message = await self._acreate_bot_response_message(
            conversation_id=conversation_id,
            content=fix_markdown_code_blocks(
                str(getattr(getattr(bot_response, "message", None), "content", "") or "")
                or partial_text
            ),
            metadata=self._continuation_metadata(lease, bot_response),
            message_id=bot_message_id,
        )
        inflight.resolve(bot_message.model_dump(mode="json"))
        await self._amark_generation_completed(
            lease.snapshot,
            user_id=user_id,
            assistant_message_id=bot_message_id,
            partial=True,
            terminal_reason="continued",
        )

        yield make_event(
            "complete",
            sequence=next_sequence(),
            conversation_id=str(conversation_id),
            message_id=str(bot_message_id),
            data={"message": bot_message.model_dump(mode="json")},
        )

    @staticmethod
    def _continuation_metadata(lease: Any, bot_response: Any) -> dict[str, Any]:
        """Metadata for the assistant message a continued epoch produced.

        ``partial`` stays true across the whole chain. The answer was assembled
        over more than one epoch, and a reader that treats the last one as the
        whole answer would misattribute where the evidence came from.
        """
        metadata: dict[str, Any] = {
            "partial": True,
            "continued": True,
            "generation_id": str(lease.snapshot.generation_id),
            "logical_turn_id": lease.snapshot.logical_turn_id,
            "execution_epoch": lease.execution_epoch,
        }
        response_metadata = getattr(bot_response, "metadata", None)
        if isinstance(response_metadata, dict):
            for key in ("images", "execution_budget"):
                if response_metadata.get(key):
                    metadata[key] = response_metadata[key]
        artifacts = getattr(bot_response, "tool_artifacts", None)
        if artifacts:
            metadata["tool_artifacts"] = list(artifacts)
        return metadata

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

    async def aresolve_generation_for_turn(
        self,
        *,
        user_message_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
    ):
        """The generation for one logical turn, for a client that has only that.

        Fenced through the logical turn rather than "whatever is active in this
        conversation": that shortcut would resolve a stale turn id to the turn
        running now.
        """
        control = self._generation_control()
        if control is None:
            return None
        return await control.find_by_logical_turn(
            logical_turn_id=str(user_message_id),
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
            await control.request_stop(
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
            return settled

        # Accelerate the owning worker if it happens to be this one. The entry
        # is left in place: removing it here is what left a retried Stop with
        # nothing to cancel while the turn was still running.
        get_generation_registry().request_cancel(generation_id)

        return await control.await_stop_settled(
            generation_id=generation_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

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
        except (asyncio.TimeoutError, asyncio.CancelledError):
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
            selected = entry.active_agent_id
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
