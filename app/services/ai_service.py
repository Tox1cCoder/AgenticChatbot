import asyncio
from time import perf_counter
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver

from ..ai.prompts import TITLE_GENERATION_PROMPT
from ..ai.schemas import (
    AgentResponse as AIAgentResponse,
)
from ..ai.schemas import (
    InterruptDecision as AIInterruptDecision,
)
from ..ai.schemas import (
    InterruptResponse as AIInterruptResponse,
)
from ..ai.schemas import (
    WorkflowExecutionRequest as AIWorkflowExecutionRequest,
)
from ..ai.utils import make_json_safe
from ..ai.workflow.contracts import WORKFLOW_ERROR_CODES, WorkflowError
from ..ai.workflow.errors import workflow_error, workflow_error_payload
from ..core.config import settings
from ..core.response_constants import (
    ERROR_NO_RESPONSE,
    ERROR_NO_RESPONSE_RESUME,
    UNKNOWN_ERROR,
)
from ..interfaces.workflow_runtime_interface import IWorkflowRuntime
from ..repositories.conversation import ConversationRepository
from ..schemas.workflow import (
    InterruptDecision,
    InterruptResponse,
    WorkflowExecutionRequest,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from ..usage import (
    UsageContext,
    begin_usage_operation,
    bind_usage_context,
)
from ..utils.text_processing import sanitize_persona
from .event_streaming.events import make_event
from .event_streaming.tool_state import infer_tool_state


def _parse_uuid(value: Any) -> UUID | None:
    """Best-effort UUID parse that never raises on malformed identifiers."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class AIService:
    def __init__(
        self,
        workflow_runtime: IWorkflowRuntime,
        conversation_repository: ConversationRepository,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.checkpointer = checkpointer
        self.workflow = workflow_runtime
        self.conversation_repository = conversation_repository

    async def initialize(self) -> None:
        await self.workflow.initialize()

    @property
    def model_usage_recorder(self):
        """The workflow's usage recorder (or ``None``), so callers like
        ``MessageService`` can attribute title/suggestion calls without a
        container lookup."""
        return getattr(self.workflow, "model_usage_recorder", None)

    def invalidate_history_cache(self, conversation_id: str) -> None:
        self.workflow.invalidate_history_cache(conversation_id)

    async def compact_checkpoint_after_terminal_response(self, thread_id: str | None) -> None:
        if not self.checkpointer or not thread_id:
            return
        await self.workflow.compact_checkpoint_after_terminal_response(str(thread_id))

    def _build_error_response(
        self,
        message: str = ERROR_NO_RESPONSE,
        *,
        code: str = "finalization_failed",
        request_id: str = "unknown",
    ) -> WorkflowResponse:
        """Build the service-level failure response.

        ``message`` stays the display copy the API renders; ``error`` carries
        the machine-readable code, retriability, and request id so a caller can
        decide what to do without reading English.
        """
        return WorkflowResponse(
            agent_type="chat",
            agent_id="chat_agent",
            message=WorkflowResponseMessage(content=message),
            metadata={"error": True},
            error=workflow_error_payload(workflow_error(code, request_id=request_id)),
        )

    @staticmethod
    def _build_tool_end_rich_items(
        *,
        render: Any,
        result: Any,
        tool_call_id: Any,
        tool_name: Any,
    ) -> list[dict[str, Any]]:
        """Derive transient `rich_items` upsert records from a tool_end event.

        Returns only safe non-image records. Image candidates are excluded
        because finalization owns their public visibility. Live widgets use
        their dedicated compact mount record rather than a generic tool view.
        """
        if not isinstance(render, dict) or not tool_call_id:
            return []
        from app.ai.tool_execution import (
            build_live_widget_candidate_from_tool_result,
            build_tool_render_candidate,
        )
        from app.core.rich_response import select_transient_upsert_items

        normalized_tool_name = str(tool_name or "")
        candidate = build_live_widget_candidate_from_tool_result(
            result,
            tool_name=normalized_tool_name,
        )
        if candidate is None:
            candidate = build_tool_render_candidate(
                render,
                tool_call_id=str(tool_call_id),
                tool_name=normalized_tool_name,
            )
        if candidate is None:
            return []
        return list(select_transient_upsert_items([candidate]))

    def _prepare_request(self, request: WorkflowExecutionRequest) -> WorkflowExecutionRequest:
        if request.persona is not None or not request.conversation_id:
            return request
        try:
            conversation = self.conversation_repository.get_by_id(UUID(request.conversation_id))
            raw_persona = conversation.persona_prompt if conversation else None
        except Exception:
            raw_persona = None
        return request.model_copy(update={"persona": sanitize_persona(raw_persona)})

    @staticmethod
    def _to_ai_request(request: WorkflowExecutionRequest) -> AIWorkflowExecutionRequest:
        return AIWorkflowExecutionRequest.model_validate(request.model_dump(mode="python"))

    @staticmethod
    def _to_ai_decisions(decisions: list[InterruptDecision]) -> list[AIInterruptDecision]:
        return [
            AIInterruptDecision.model_validate(decision.model_dump(mode="python"))
            for decision in decisions
        ]

    @staticmethod
    def _normalize_interrupt_payload(payload: Any) -> dict[str, Any] | Any:
        if payload is None:
            return None
        if isinstance(payload, InterruptResponse):
            return payload.model_dump(mode="json")
        if isinstance(payload, AIInterruptResponse):
            return InterruptResponse.model_validate(payload.model_dump(mode="python")).model_dump(
                mode="json"
            )
        if isinstance(payload, dict):
            try:
                return InterruptResponse.model_validate(payload).model_dump(mode="json")
            except Exception:
                return make_json_safe(payload)
        if hasattr(payload, "model_dump"):
            try:
                dumped = payload.model_dump(mode="python")
                return InterruptResponse.model_validate(dumped).model_dump(mode="json")
            except Exception:
                pass
        return {"raw": str(payload)}

    def _to_service_response(self, response: AIAgentResponse | None) -> WorkflowResponse | None:
        if response is None:
            return None

        message = getattr(response, "message", None)
        message_metadata = dict(getattr(message, "metadata", None) or {})
        metadata = dict(getattr(response, "metadata", None) or {})
        interrupt_payload = metadata.get("interrupt")
        if interrupt_payload is not None:
            metadata["interrupt"] = self._normalize_interrupt_payload(interrupt_payload)

        return WorkflowResponse(
            agent_type=str(
                getattr(getattr(response, "agent_type", None), "value", response.agent_type)
            ),
            agent_id=str(getattr(response, "agent_id", "")),
            message=WorkflowResponseMessage(
                role=str(
                    getattr(
                        getattr(message, "role", None),
                        "value",
                        getattr(message, "role", "assistant"),
                    )
                ),
                content=str(getattr(message, "content", "") or ""),
                metadata=message_metadata,
            ),
            metadata=metadata,
            tool_artifacts=getattr(response, "tool_artifacts", None),
            error=self._service_error_payload(response, metadata),
            suggested_questions=getattr(response, "suggested_questions", None),
        )

    @staticmethod
    def _service_error_payload(
        response: AIAgentResponse, metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Project the terminal failure as the typed public payload.

        The finalizer records the whole ``WorkflowError`` in workflow metadata;
        the agent-level ``error`` field carries only its code. Preferring the
        recorded error keeps ``retriable`` and ``request_id`` intact instead of
        making the caller infer them from a string.
        """
        recorded = (metadata.get("workflow") or {}).get("error")
        if isinstance(recorded, dict) and recorded.get("code"):
            return workflow_error_payload(WorkflowError.model_validate(recorded))

        code = getattr(response, "error", None)
        if not code:
            return None
        if code in WORKFLOW_ERROR_CODES:
            return workflow_error_payload(workflow_error(code, request_id="unknown"))
        # An agent-level diagnostic that is not a terminal workflow code still
        # reaches the caller as structured data rather than raw prose.
        return workflow_error_payload(
            workflow_error(
                "tool_execution_failed", request_id="unknown", details={"reason": "agent_error"}
            )
        )

    @staticmethod
    def _workflow_usage_context(request: WorkflowExecutionRequest) -> UsageContext:
        """Attribution for a workflow turn.

        ``user_message_id`` is the row persisted before execution; the reserved
        ``assistant_message_id`` is never used as a request-message FK.
        """
        return UsageContext(
            user_id=_parse_uuid(request.user_id),
            conversation_id=_parse_uuid(request.conversation_id),
            request_message_id=_parse_uuid(request.user_message_id),
            correlation_id=request.thread_id,
            operation="workflow",
        )

    async def execute_request(self, request: WorkflowExecutionRequest) -> WorkflowResponse:
        prepared_request = self._prepare_request(request)
        with bind_usage_context(self._workflow_usage_context(prepared_request)):
            response = await self.workflow.execute_request(self._to_ai_request(prepared_request))
        if response:
            normalized_response = self._to_service_response(response)
            if normalized_response:
                return normalized_response
        return self._build_error_response()

    async def execute_request_stream(self, request: WorkflowExecutionRequest):
        prepared_request = self._prepare_request(request)
        workflow_stream = self.workflow.execute_request_stream(
            self._to_ai_request(prepared_request)
        )
        context_bound_stream = self._iterate_in_usage_context(
            workflow_stream,
            self._workflow_usage_context(prepared_request),
        )
        async for mapped_event in self._map_workflow_stream(
            context_bound_stream,
            emit_rich_items=bool(
                getattr(settings, "inline_rich_response_enabled", False)
                and getattr(prepared_request, "inline_rich_response_v1", False)
            ),
        ):
            yield mapped_event

    @staticmethod
    async def _iterate_in_usage_context(workflow_stream, usage_context: UsageContext):
        """Advance and close a workflow source inside its attribution context.

        The binding deliberately ends before yielding to an API adapter. Async
        generators may be closed by a different task/context when a client
        disconnects, so a ContextVar token must never span that outward yield.
        """
        iterator = workflow_stream.__aiter__()
        try:
            while True:
                try:
                    with bind_usage_context(usage_context):
                        event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                with bind_usage_context(usage_context):
                    await aclose()

    async def resume_workflow(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_input: str | None = None,
    ) -> WorkflowResponse:
        thread_id = str(conversation_id) if conversation_id and self.checkpointer else None

        if not thread_id:
            return self._build_error_response(
                "Cannot resume: Checkpointing not enabled or conversation ID missing"
            )

        # Ownership is rebuilt from the authenticated arguments, never from
        # checkpoint state.
        resume_context = UsageContext(
            user_id=_parse_uuid(user_id),
            conversation_id=_parse_uuid(conversation_id),
            correlation_id=thread_id,
            operation="workflow",
        )
        with bind_usage_context(resume_context):
            response = await self.workflow.resume(
                thread_id=thread_id,
                user_input=user_input,
            )

        if response:
            normalized_response = self._to_service_response(response)
            if normalized_response:
                return normalized_response

        return self._build_error_response(ERROR_NO_RESPONSE_RESUME)

    async def _map_workflow_stream(self, workflow_stream, *, emit_rich_items: bool = False):
        """Curate the workflow's canonical ``V3StreamEvent`` stream for the API.

        The workflow already emits canonical events; this method re-stamps
        sequence numbers and layers on the value-adds the service owns: tool
        ``duration_ms`` timing, tool ``error`` inference, transient
        ``rich_items`` emission, service-layer normalization of ``interrupt``
        and ``error`` payloads, and buffering the terminal ``complete``
        response through :meth:`_to_service_response`.
        """
        final_response = None
        tool_started_at: dict[str, float] = {}
        sequence = 0

        def _next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        async for event in workflow_stream:
            etype = event.type

            if etype == "tool_call_available":
                if event.tool_call_id is not None:
                    tool_started_at[str(event.tool_call_id)] = perf_counter()
                yield event.model_copy(update={"sequence": _next_sequence()})

            elif etype == "tool_execution_end":
                tool_data = dict(event.data)
                result = tool_data.get("output")
                tool_call_id = event.tool_call_id
                if tool_call_id is not None:
                    started_at = tool_started_at.pop(str(tool_call_id), None)
                    if started_at is not None:
                        tool_data["duration_ms"] = int((perf_counter() - started_at) * 1000)
                if infer_tool_state(phase="end", result=result) == "error":
                    tool_data["error"] = str(result)
                yield event.model_copy(update={"sequence": _next_sequence(), "data": tool_data})
                # Emit a `rich_items` upsert for safe non-image candidates as
                # soon as the tool result exists. Image records are never
                # streamed transiently; canvas source is excluded.
                if emit_rich_items:
                    rich_items = self._build_tool_end_rich_items(
                        render=tool_data.get("render"),
                        result=result,
                        tool_call_id=tool_call_id,
                        tool_name=event.tool_name,
                    )
                    if rich_items:
                        yield make_event(
                            "rich_items",
                            sequence=_next_sequence(),
                            data={"operation": "upsert", "items": rich_items},
                        )

            elif etype == "complete":
                final_response = self._to_service_response(event.data.get("response"))

            elif etype == "error":
                error_msg = event.data.get("error", UNKNOWN_ERROR)
                display_error = str(error_msg).strip() or UNKNOWN_ERROR
                if not display_error.lower().startswith("error:"):
                    display_error = f"Error: {display_error}"
                yield make_event(
                    "error",
                    sequence=_next_sequence(),
                    data={
                        "error": str(error_msg),
                        "response": self._build_error_response(display_error),
                    },
                )

            elif etype == "interrupt":
                interrupt_payload = event.data.get("interrupt")
                normalized_interrupt = self._normalize_interrupt_payload(interrupt_payload)
                interrupt_message = None
                if isinstance(normalized_interrupt, dict):
                    interrupt_metadata = normalized_interrupt.get("metadata")
                    if isinstance(interrupt_metadata, dict):
                        message_value = interrupt_metadata.get("message") or interrupt_metadata.get(
                            "reason"
                        )
                        if isinstance(message_value, str) and message_value.strip():
                            interrupt_message = message_value.strip()
                yield make_event(
                    "interrupt",
                    sequence=_next_sequence(),
                    data={
                        "next": event.data.get("next", []),
                        "thread_id": event.data.get("thread_id"),
                        "pending_tool_calls": event.data.get("pending_tool_calls"),
                        "interrupt": normalized_interrupt,
                        "message": interrupt_message,
                    },
                )

            else:
                # message_delta, reasoning_delta, agent_selected, state_snapshot,
                # image_preview, rich_items, subagent_* and forward-compatible
                # events pass through with a fresh sequence number.
                yield event.model_copy(update={"sequence": _next_sequence()})

        if final_response:
            yield make_event(
                "complete", sequence=_next_sequence(), data={"response": final_response}
            )
        else:
            yield make_event(
                "complete",
                sequence=_next_sequence(),
                data={"response": self._build_error_response()},
            )

    async def resume_interrupted_execution_stream(
        self,
        thread_id: str,
        decisions: list[InterruptDecision],
        *,
        inline_rich_response_v1: bool = False,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ):
        if not self.checkpointer:
            yield make_event(
                "error",
                sequence=1,
                data={"error": "Cannot resume: Checkpointing not enabled"},
            )
            return

        # Ownership comes from the authenticated caller arguments; the thread id
        # is the conversation id when a distinct one was not supplied.
        resume_context = UsageContext(
            user_id=_parse_uuid(user_id),
            conversation_id=_parse_uuid(conversation_id) or _parse_uuid(thread_id),
            correlation_id=thread_id,
            operation="workflow",
        )
        workflow_stream = self.workflow.resume_with_decisions_stream(
            thread_id=thread_id,
            decisions=self._to_ai_decisions(decisions),
        )
        context_bound_stream = self._iterate_in_usage_context(workflow_stream, resume_context)
        async for mapped_event in self._map_workflow_stream(
            context_bound_stream,
            emit_rich_items=bool(
                getattr(settings, "inline_rich_response_enabled", False) and inline_rich_response_v1
            ),
        ):
            yield mapped_event

    def get_bot_response_sync(
        self,
        user_message: str,
        conversation_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> WorkflowResponse:
        request = WorkflowExecutionRequest(
            message=user_message,
            conversation_id=str(conversation_id) if conversation_id else None,
            user_id=str(user_id) if user_id else None,
            thread_id=str(conversation_id) if conversation_id and self.checkpointer else None,
        )
        return asyncio.run(self.execute_request(request))

    async def generate_conversation_title(
        self,
        user_message: str,
        *,
        user_id: UUID | str | None = None,
        conversation_id: UUID | str | None = None,
    ) -> str:
        """
        Generate a concise, descriptive title for a conversation based on the first user message.

        Args:
            user_message: The first message from the user
            user_id: Authenticated user the title call is attributed to
            conversation_id: Conversation the title belongs to, if known

        Returns:
            A short, descriptive title (max 50 characters)
        """
        try:
            from ..ai.agent_config import AGENT_CONFIG, create_langchain_model

            llm = create_langchain_model(
                agent_type="title_generator",
                include_thinking=False,
            )

            prompt = TITLE_GENERATION_PROMPT.format(user_message=user_message)

            recorder = getattr(self.workflow, "model_usage_recorder", None)
            if recorder is not None:
                title_context = UsageContext(
                    user_id=_parse_uuid(user_id),
                    conversation_id=_parse_uuid(conversation_id),
                    operation="title_generation",
                    agent_id="title_generator",
                )
                # ``record_one_async_attempt`` re-raises on failure; the existing
                # except below still falls back to a truncated title.
                with bind_usage_context(title_context), begin_usage_operation() as operation:
                    response = await recorder.record_one_async_attempt(
                        call=lambda: llm.ainvoke(prompt),
                        provider="gemini",
                        model=AGENT_CONFIG["title_generator"]["model"],
                        operation=operation,
                    )
            else:
                response = await llm.ainvoke(prompt)
            raw_title = response.content

            if isinstance(raw_title, list):
                # Handle list content (e.g. from Gemini)
                title_text = ""
                for part in raw_title:
                    if isinstance(part, dict) and part.get("type") == "text":
                        title_text += part.get("text", "")
                    elif isinstance(part, str):
                        title_text += part
                raw_title = title_text

            title = str(raw_title).strip()

            # Clean up the title
            title = title.strip("\"'")  # Remove quotes
            title = title.rstrip(".")  # Remove trailing period

            # Ensure it's not too long
            if len(title) > 50:
                title = title[:47] + "..."

            # Fallback to truncated message if generation fails
            if not title or len(title) < 3:
                title = user_message[:50]
                if len(user_message) > 50:
                    title = title[:47] + "..."

            return title

        except Exception:
            # Fallback: use truncated user message
            title = user_message[:50]
            if len(user_message) > 50:
                title = title[:47] + "..."
            return title
