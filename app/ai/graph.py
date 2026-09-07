import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import RemoveMessage
from langgraph.types import Command
from langsmith import tracing_context
from qdrant_client import QdrantClient

from ..core.config import settings
from ..core.response_constants import NO_RESPONSE_GENERATED
from ..interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..interfaces.workflow_runtime_interface import IWorkflowRuntime
from ..models.enums import PlanLifecycle
from ..observability.rag import rag_metrics
from ..observability.routing import get_routing_metrics_recorder
from ..services.event_streaming.events import IMAGE_PREVIEW_STATUS_PARTIAL, make_event
from ..services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
    flush_answer_text,
)
from ..services.event_streaming.langchain_v3 import iter_v3_events_from_graph
from ..services.rag_grounding import GroundedAnswerGate
from ..services.tool_execution_receipt_service import ToolExecutionReceiptService
from .agent_metadata import (
    attach_agent_metadata,
    normalize_handoff_metadata,
    normalize_subagent_metadata,
)
from .agents.canvas_agent import CanvasAgent, build_canvas_specialist_definition
from .agents.chat_agent import ChatAgent, build_chat_specialist_definition
from .agents.custom_agent import build_custom_specialist_definition
from .agents.image_generator_agent import (
    ImageGeneratorAgent,
    build_image_generator_specialist_definition,
)
from .agents.planning_agent import PlanningAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent, build_search_specialist_definition
from .canvas_state import CANVAS_EDIT_DENIED_TOOL_NAMES, CanvasArtifactSnapshot
from .history import ConversationHistoryProvider
from .hitl_config import (
    build_interrupt_response,
    pending_interrupt_payload,
)
from .image_context import (
    build_multimodal_content,
    describe_attachment_rejections,
    has_image_parts,
    use_chat_image_loader,
)
from .image_generation import (
    ImagePreviewPublisher,
    MediaDeliveryService,
    use_image_preview_emitter,
    use_media_delivery_service,
)
from .model_factory import ModelFactory
from .research_budget import reset_research_budget
from .schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    ContinuationSignal,
    GraphState,
    GraphStateView,
    InterruptDecision,
    MessageRole,
    TodoStatus,
    WorkflowExecutionRequest,
)
from .skills_tool import get_available_skill_summaries
from .time_context import build_runtime_time_context_block
from .utils import (
    address_decisions_to_interrupts,
    apply_hitl_decisions,
    build_interrupt_resume_payload,
    coerce_response_text,
    make_json_safe,
    normalize_tool_call,
)
from .workflow.contracts import OutcomeProvenance, ResponseOutcome, TurnIdentity
from .workflow.custom_agents import CustomAgentsMixin
from .workflow.graph_builder import build_workflow_graph
from .workflow.inventory import CUSTOM_AGENT_NODE
from .workflow.planning_execution import (
    PLANNING_AGENT_ID,
    PlanningLimits,
    PlanningNodeFactory,
    PlanningWorkerRuntime,
    TodoActionOutcome,
    build_dispatch_control_tool,
)
from .workflow.rag_execution import (
    ProductionRagRuntime,
    RagExecutionGraph,
    RagExecutionRequest,
)
from .workflow.routing import RoutingContextBuilder, RoutingService
from .workflow.runtime_context import WorkflowRuntimeContext, build_runtime_inventory
from .workflow.specialists import SpecialistFactory, SpecialistRequest
from .workflow.state import build_checkpoint_thread_id
from .workflow.tool_loop import ToolLoopMixin
from .workflow.transitions import TransitionResolver

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository
    from ..usage.recorder import ModelUsageRecorder

_apply_decisions = apply_hitl_decisions

# A pause is whatever the checkpoint says is pending. There is deliberately no
# node-name allowlist here any more: it never listed ``planning_worker``, so a
# Planning worker waiting on a human read as a crashed turn, and every node
# added later would have had to remember to join the set.


def _graph_stream_writer() -> Any:
    """The live custom-event writer, when there is a run to write into."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (RuntimeError, ImportError):  # pragma: no cover - outside a run
        return None


def _last_human_text(messages: list[Any]) -> str:
    """The question this RAG turn is answering."""
    for message in reversed(messages or ()):
        if getattr(message, "type", None) != "human":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def apply_accumulated_thinking(response: Any, accumulated_thinking: str) -> None:
    """Carry streamed thought text onto a terminal response as a fallback.

    ``accumulated_thinking`` is assembled from canonical ``reasoning_delta``
    events. For OpenAI those deltas *are* the reasoning summary the agent
    already stored under ``reasoning_summary``, so copying them into
    ``thinking_summary`` would make the trace panel show the same text twice.
    Persisted agent metadata stays authoritative in both fields.
    """
    if not accumulated_thinking:
        return

    metadata = response.metadata
    if metadata.get("thinking_summary"):
        return

    reasoning_summary = metadata.get("reasoning_summary")
    if (
        isinstance(reasoning_summary, str)
        and reasoning_summary.strip() == accumulated_thinking.strip()
    ):
        return

    metadata["thinking_summary"] = accumulated_thinking


def _build_inline_rich_inventory_for_state(context: dict[str, Any] | None) -> str:
    """Compute the bounded rich-item inventory block for the current turn.

    Returns an empty string if the rollout flag is off, the request did not
    advertise the capability, or no candidates exist.
    """
    if not isinstance(context, dict):
        return ""
    context["_presented_rich_image_ids"] = []
    if not getattr(settings, "inline_rich_response_enabled", False):
        return ""
    if not context.get("inline_rich_response_v1"):
        return ""
    candidates = context.get("rich_item_candidates") or []
    if not candidates:
        return ""
    from .prompts import build_rich_response_guidance

    presented_image_ids: list[str] = []
    guidance = build_rich_response_guidance(
        candidates=list(candidates),
        enabled=True,
        capability=True,
        presented_image_ids=presented_image_ids,
    )
    context["_presented_rich_image_ids"] = presented_image_ids
    return guidance


class MultiAgentWorkflow(
    ToolLoopMixin,
    CustomAgentsMixin,
    IWorkflowRuntime,
):
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: Any,
        checkpointer: BaseCheckpointSaver | None = None,
        document_repository: Optional["DocumentRepository"] = None,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        history_provider: ConversationHistoryProvider | None = None,
        model_usage_recorder: "ModelUsageRecorder | None" = None,
        chat_image_service: Any | None = None,
        routing_service: Any | None = None,
        tool_execution_receipt_repository: Any | None = None,
    ):
        self.qdrant_client = qdrant_client
        # Resolves stored image references back to base64 for the model when
        # replaying conversation history (bytes live outside message metadata).
        self.chat_image_service = chat_image_service
        # Canonical prompt-history source. Workflows without it intentionally
        # run without persisted history rather than using a second source.
        self.history_provider = history_provider
        # Routing-v2: one bounded context builder and one routing service own
        # new-turn classification. Nothing else may select an agent.
        self.routing_context_builder = RoutingContextBuilder(
            history_provider=history_provider,
            document_repository=document_repository,
            settings=settings,
            skill_summary_provider=get_available_skill_summaries,
        )
        self.routing_service = routing_service or RoutingService(
            runtime_model_resolver=runtime_model_resolver,
            model_factory=ModelFactory,
            settings=settings,
            context_builder=self.routing_context_builder,
            metrics=get_routing_metrics_recorder(),
            usage_recorder=model_usage_recorder,
        )
        self.chat_agent = ChatAgent(
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_service=embedding_service,
            collection_name=settings.qdrant_collection_name,
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.search_agent = SearchAgent(
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.image_generator_agent = ImageGeneratorAgent(
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.planning_agent = PlanningAgent(
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.canvas_agent = CanvasAgent(
            runtime_model_resolver=runtime_model_resolver,
            recorder=model_usage_recorder,
        )
        self.agents = {
            "chat_agent": self.chat_agent,
            "rag_agent": self.rag_agent,
            "search_agent": self.search_agent,
            "image_generator_agent": self.image_generator_agent,
            "planning_agent": self.planning_agent,
            "canvas_agent": self.canvas_agent,
        }

        self.checkpointer = checkpointer
        self.document_repository = document_repository
        # Stored so per-turn CustomAgent instances resolve models the same way
        # base agents do. Custom agents are built on demand from workflow state.
        self._runtime_model_resolver = runtime_model_resolver
        self._model_usage_recorder = model_usage_recorder

        # Durable mutation receipts. Absent (no repository wired), mutations run
        # exactly as before: the crash gap stays open and is not concealed.
        self._receipt_service = (
            ToolExecutionReceiptService(repository=tool_execution_receipt_repository)
            if tool_execution_receipt_repository is not None
            else None
        )
        self._specialist_factory = self._build_specialist_factory(self._receipt_service)

        # One compiled RAG topology for the whole workflow. Both entry points --
        # the top-level RAG specialist and a Planning RAG worker -- run this
        # object, because two graphs were how one path could skip validation.
        self.rag_execution_graph = RagExecutionGraph(
            runtime=ProductionRagRuntime(
                rag_agent=self.rag_agent,
                agent_lookup=self._agent_for_rag_request,
                settings=settings,
            ),
            grounded_answer_gate=self._grounded_answer_gate(),
            settings=settings,
        )
        # The worker runtime holds the *same* graph object rather than a factory,
        # so a worker cannot end up on a differently configured RAG path.
        planning_limits = PlanningLimits.from_settings(settings)
        self.planning_worker_runtime = PlanningWorkerRuntime(
            specialist_factory=self._specialist_factory,
            rag_execution_graph=self.rag_execution_graph,
            limits=planning_limits,
        )
        # Planning's six parent-graph nodes. The factory holds only
        # collaborators; the topology it describes is registered by
        # build_workflow_graph, which is the sole owner of graph shape.
        self.planning_node_factory = PlanningNodeFactory(
            call_model=self._planning_model_call,
            worker_runtime=self.planning_worker_runtime,
            limits=planning_limits,
            inventory_for=self._planning_inventory_for,
            resolve_allowed_tools=self._planning_allowed_tools,
            apply_todo_actions=self._planning_apply_todo_actions,
            review_rubric=self._planning_review_rubric,
        )

        self.graph = self._build_graph()
        self._cleanup_agents = [
            self.chat_agent,
            self.search_agent,
            self.rag_agent,
            self.image_generator_agent,
            self.planning_agent,
            self.canvas_agent,
        ]
        self._initialized = False

    def _agent_for_rag_request(self, request: Any) -> Any:
        """The agent whose tool map a RAG tool round executes against.

        Non-search tool calls resolve through the RAG agent's own map so a
        deferred tool it loaded mid-turn stays callable in the same turn.
        """
        return self.rag_agent

    @property
    def model_usage_recorder(self) -> "ModelUsageRecorder | None":
        """The workflow's usage recorder, so callers holding the workflow (e.g.
        ``AIService`` for title/suggestion generation) can record attempts
        without a container lookup. ``None`` when usage tracking is disabled."""
        return self._model_usage_recorder

    def _get_current_turn_messages(self, messages: list) -> list:
        if not messages:
            return messages

        # Find the last HumanMessage index
        last_human_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        if last_human_idx is None:
            return messages

        # Return only messages from the current turn
        return messages[last_human_idx:]

    # ============================================================
    # Shared Helper Methods
    # ============================================================

    async def _get_history_context(
        self,
        conversation_id: str | None,
        user_id: str | None,
        *,
        agent_key: str = "chat",
        current_message_id: str | None = None,
    ):
        """Build a full ``ConversationHistoryContext`` from the provider.

        Returns ``None`` when the provider is not wired or identifiers are
        missing.
        """
        if not conversation_id or not user_id or self.history_provider is None:
            return None
        try:
            return await self.history_provider.build_context(
                conversation_id=conversation_id,
                user_id=user_id,
                current_message_id=current_message_id,
                agent_key=agent_key,
            )
        except Exception as exc:
            logger.warning(
                "History provider failed for %s/%s: %s",
                conversation_id,
                user_id,
                exc,
            )
            return None

    async def _get_conversation_history(
        self,
        conversation_id: str | None,
        user_id: str | None,
        agent_key: str | None = None,
        *,
        state: GraphState | None = None,
    ) -> list:
        """
        Get sequence-scoped conversation history from the canonical provider.

        When the workflow is wired with a ``ConversationHistoryProvider``
        (production path), the provider is the single source of truth: it
        returns DB-backed messages already excluding the current user turn
        by ``user_message_id`` and the durable memory sequence cursor. Owned
        valid memory is already the first lower-priority history message.

        When the provider is absent, no persisted history is injected.
        """
        if not conversation_id or not user_id:
            return []

        # Provider path — preferred.
        if getattr(self, "history_provider", None) is not None:
            current_message_id = None
            if state is not None:
                current_message_id = state.get("user_message_id")
            context = await self._get_history_context(
                conversation_id,
                user_id,
                agent_key=agent_key or "chat",
                current_message_id=current_message_id,
            )
            if context is not None:
                return list(context.messages)

        return []

    async def _get_active_canvas_snapshot(
        self,
        conversation_id: str | None,
        user_id: str | None,
    ) -> CanvasArtifactSnapshot | None:
        """Return durable canvas state without exposing failures to routing."""
        provider = getattr(self, "history_provider", None)
        if not conversation_id or not user_id or provider is None:
            return None
        loader = getattr(provider, "get_latest_canvas_artifact", None)
        if not callable(loader):
            return None
        try:
            return await loader(conversation_id=conversation_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "Canvas state lookup failed for conversation=%s: %s",
                conversation_id,
                exc,
            )
            return None

    def invalidate_history_cache(self, conversation_id: str) -> None:
        """Invalidate cached history for a conversation (call when new messages added)."""
        if getattr(self, "history_provider", None) is not None:
            self.history_provider.invalidate(conversation_id)

    def _find_last_human_message_index(self, messages: list) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                return idx
        return None

    def _merge_tool_artifacts(
        self, state: GraphState, response: AgentResponse, append_images: bool = False
    ) -> None:
        state_view = GraphStateView(state)
        tool_artifacts = state_view.tool_artifacts()
        tool_images = state_view.tool_images()

        if tool_artifacts:
            response.tool_artifacts = tool_artifacts

        if tool_images:
            if not response.metadata:
                response.metadata = {}
            if append_images:
                existing_images = response.metadata.get("images", [])
                existing_images.extend(tool_images)
                response.metadata["images"] = existing_images
            else:
                response.metadata["images"] = tool_images

        # Forward turn-scoped rich-item candidates into the response metadata
        # so `build_bot_metadata()` can finalize the public `rich_items`
        # registry at the workflow boundary. Stripped from persisted output.
        context = state_view.context()
        rich_enabled = bool(
            isinstance(context, dict)
            and getattr(settings, "inline_rich_response_enabled", False)
            and context.get("inline_rich_response_v1")
        )
        rich_candidates = context.get("rich_item_candidates") if rich_enabled else None
        if rich_enabled:
            if not response.metadata:
                response.metadata = {}
            response.metadata["_inline_rich_response_v1"] = True
            if rich_candidates:
                response.metadata["_rich_item_candidates"] = list(rich_candidates)
            presented_image_ids = context.get("_presented_rich_image_ids")
            if isinstance(presented_image_ids, list):
                response.metadata["_presented_rich_image_ids"] = list(presented_image_ids)

    def _attach_final_agent_metadata(
        self,
        state: GraphState,
        response: AgentResponse,
    ) -> None:
        if response.metadata is None:
            response.metadata = {}

        custom_agents = GraphStateView(state).custom_agents()
        attach_agent_metadata(
            response.metadata,
            response_agent_id=response.agent_id,
            active_agent_id=state.get("active_agent_id"),
            custom_agents=custom_agents,
        )

        handoff = normalize_handoff_metadata(GraphStateView(state).context())
        if handoff:
            response.metadata["handoff"] = handoff

        subagent_results = normalize_subagent_metadata(
            response.metadata.get("subagent_results"),
            custom_agents,
        )
        if subagent_results:
            response.metadata["subagent_results"] = subagent_results

    def _finalize_agent_response(self, state: GraphState, response: AgentResponse) -> GraphState:
        self._attach_final_agent_metadata(state, response)
        state["response"] = response

        ai_kwargs: dict[str, Any] = {"content": response.message.content}
        assistant_message_id = state.get("assistant_message_id")
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
            # Stamp intermediate tool-calling AIMessages with a deterministic,
            # derived id so checkpoint compaction can remove them on terminal
            # response. Without an id they survive across turns and pollute
            # state["messages"] / LangSmith traces.
            if assistant_message_id:
                intermediate_idx = sum(
                    1
                    for m in state.get("messages", [])
                    if isinstance(m, AIMessage)
                    and getattr(m, "id", "")
                    and str(m.id).startswith(f"{assistant_message_id}-tool-")
                )
                ai_kwargs["id"] = f"{assistant_message_id}-tool-{intermediate_idx}"
        elif assistant_message_id:
            ai_kwargs["id"] = assistant_message_id
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    def _build_initial_state_from_request(self, request: WorkflowExecutionRequest) -> GraphState:
        tasks = list(request.planning.tasks)
        # Stamp the current-turn HumanMessage with the persisted DB id so
        # downstream nodes can identify it uniquely (and so checkpoint
        # compaction can RemoveMessage it by id later).
        human_message_kwargs: dict[str, Any] = {"content": request.message}
        if request.user_message_id:
            human_message_kwargs["id"] = request.user_message_id
        initial_state: GraphState = {
            "messages": [HumanMessage(**human_message_kwargs)],
            "context": {
                "inline_rich_response_v1": bool(getattr(request, "inline_rich_response_v1", False)),
                "hitl_policy": getattr(request, "hitl_policy", None),
            },
        }

        if request.conversation_id is not None:
            initial_state["conversation_id"] = request.conversation_id
        if request.user_id is not None:
            initial_state["user_id"] = request.user_id
        # Always overwrite the checkpointed device binding, None included: a
        # turn without a connected client must never inherit a previous
        # client's device_id from the checkpoint and dispatch tools there.
        initial_state["device_id"] = request.device_id
        if request.model_request is not None:
            initial_state["model_request"] = request.model_request
        if request.user_message_id is not None:
            initial_state["user_message_id"] = request.user_message_id
        if request.assistant_message_id is not None:
            initial_state["assistant_message_id"] = request.assistant_message_id
        initial_state["response"] = None
        initial_state["execution_phase"] = "routing"
        turn_id = request.turn_id or request.user_message_id
        if request.conversation_id and turn_id:
            initial_state["turn_identity"] = TurnIdentity(
                request_id=request.request_id or turn_id,
                turn_id=turn_id,
                checkpoint_thread_id=build_checkpoint_thread_id(request.conversation_id, turn_id),
            )
        initial_state["persona"] = request.persona
        initial_state["planning_mode_enabled"] = request.planning.planning_mode_enabled
        initial_state["has_existing_plan"] = request.planning.has_existing_plan
        initial_state["iteration_count"] = None
        initial_state["custom_agents"] = dict(request.custom_agents or {})

        if request.planning.current_task:
            initial_state["current_task"] = request.planning.current_task
            initial_state["task_plan_id"] = request.planning.current_task.get("id")
        if tasks:
            initial_state["all_tasks"] = tasks

        if request.attachments:
            initial_state["attachments"] = request.attachments
        if tasks:
            todos: list[dict[str, Any]] = []
            for i, task in enumerate(tasks):
                if not isinstance(task, dict):
                    continue
                todos.append(
                    {
                        "id": task.get("id", str(i)),
                        "description": task.get("description", ""),
                        "status": task.get("status", TodoStatus.PENDING.value),
                        "order": task.get("order", task.get("task_order", i)),
                    }
                )
            initial_state["todos"] = todos
            initial_state["current_task_index"] = self._find_first_pending_task(todos)

        # Initialize planning call count for budget tracking
        initial_state["planning_call_count"] = 0

        # A new user turn gets a clean research budget; the previous turn's
        # deduplication must not suppress a legitimate follow-up question.
        reset_research_budget(str(request.conversation_id) if request.conversation_id else None)

        # Derive planning_phase from persisted lifecycle:
        # If lifecycle is "executing", set execution phase so the planning agent
        # doesn't ask for confirmation again.
        if request.planning.plan_lifecycle == PlanLifecycle.executing:
            initial_state["planning_phase"] = "executing"
        else:
            # Default to "planning" phase - only switch to "executing" when user requests
            initial_state["planning_phase"] = "planning"

        # Persist lifecycle
        initial_state["plan_lifecycle"] = request.planning.plan_lifecycle

        # Seed turn-scoped Planning rubric metadata from service-level plan
        # creation/modification so the final response can surface it.
        if request.planning.rubric_metadata:
            context = dict(initial_state.get("context") or {})
            context["planning_rubric"] = request.planning.rubric_metadata
            initial_state["context"] = context

        return initial_state

    @staticmethod
    def _get_state_attachments(state: GraphState) -> list[Any]:
        return GraphStateView(state).attachments()

    def _build_turn_messages_with_attachments(
        self,
        current_turn_messages: list[Any],
        attachments: list[Any],
    ) -> tuple[list[Any], bool]:
        if not current_turn_messages:
            current_turn_messages = [HumanMessage(content="")]

        messages_copy = list(current_turn_messages)
        last_human_idx = self._find_last_human_message_index(messages_copy)
        if last_human_idx is None:
            last_human_idx = len(messages_copy)
            messages_copy.append(HumanMessage(content=""))

        original_message = messages_copy[last_human_idx]
        original_content = original_message.content
        user_text = coerce_response_text(original_content)
        multimodal_content = build_multimodal_content(user_text, attachments)

        rejections = describe_attachment_rejections(attachments)
        if rejections:
            logger.info(
                "dropped %d unusable image attachment(s) code=attachment_dropped: %s",
                len(rejections),
                "; ".join(rejections),
            )

        if not has_image_parts(multimodal_content):
            return messages_copy, False

        kwargs: dict[str, Any] = {"content": multimodal_content}
        message_id = getattr(original_message, "id", None)
        if message_id:
            kwargs["id"] = message_id
        messages_copy[last_human_idx] = HumanMessage(**kwargs)
        return messages_copy, True

    @staticmethod
    def _mark_response_has_images(response: AgentResponse, has_images: bool) -> None:
        if not has_images:
            return
        response.metadata = dict(response.metadata or {})
        response.metadata["has_images"] = True

    def _apply_current_turn_attachments(
        self,
        state: GraphState,
        current_turn_messages: list[Any],
    ) -> tuple[list[Any], bool]:
        attachments = self._get_state_attachments(state)
        if not attachments:
            return current_turn_messages, False
        return self._build_turn_messages_with_attachments(current_turn_messages, attachments)

    @staticmethod
    def _get_planning_flags(state: GraphState) -> tuple[bool, bool]:
        return GraphStateView(state).planning_flags()

    def _find_first_pending_task(self, tasks: list[dict[str, Any]]) -> int | None:
        """Find the first pending or in-progress task index in the task list."""
        for i, task in enumerate(tasks):
            status = task.get("status", "pending")
            if status in ("pending", "in_progress"):
                return i
        return None

    async def initialize(self) -> None:
        if self._initialized:
            return

        for agent in [
            self.chat_agent,
            self.search_agent,
            self.image_generator_agent,
            self.rag_agent,
            self.planning_agent,
        ]:
            if hasattr(agent, "_init_tools"):
                await agent._init_tools()
            elif hasattr(agent, "_init_mcp"):
                await agent._init_mcp()

        self._initialized = True

    def _build_graph(self) -> StateGraph:
        return build_workflow_graph(
            self,
            checkpointer=self.checkpointer,
            context_schema=WorkflowRuntimeContext,
        )

    # ------------------------------------------------------------------
    # hand_off delegation helper
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Delegated-agent message scoping
    # ------------------------------------------------------------------
    def _messages_for_active_agent(
        self,
        state: GraphState,
        agent_name: str,
        messages: list,
    ) -> list:
        """Return the current-turn message slice scoped for ``agent_name``.

        When the active turn started as a handoff (``state["context"]["handoff"]
        ["active"]`` is True and the target matches ``agent_name``), strip the
        source agent's handoff narration ``AIMessage`` and its matching
        ``ToolMessage(name="hand_off")`` so the delegated agent receives the
        user's original request without the routing chatter as conversational
        context.

        Falls back to ``_get_current_turn_messages`` semantics when no handoff
        is active or the target does not match.
        """
        current_turn = self._get_current_turn_messages(messages)

        handoff = (state.get("context") or {}).get("handoff") if isinstance(state, dict) else None
        if (
            not isinstance(handoff, dict)
            or not handoff.get("active")
            or handoff.get("target_agent") != agent_name
        ):
            return current_turn

        tool_call_id = handoff.get("tool_call_id")
        filtered: list = []
        for message in current_turn:
            if isinstance(message, ToolMessage):
                if getattr(message, "name", None) == "hand_off":
                    continue
                if tool_call_id and getattr(message, "tool_call_id", None) == tool_call_id:
                    continue
            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                if any(
                    (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))
                    == "hand_off"
                    for tc in tool_calls
                ):
                    continue
            filtered.append(message)
        return filtered

    @staticmethod
    def _get_interrupt_payload_from_state(
        state_values: dict[str, Any],
        fallback_tool_calls: list[Any],
    ) -> dict[str, Any]:
        """Recover the pending interrupt payload from checkpoint state."""
        state_view = GraphStateView(state_values)
        action_requests = state_view.pending_action_requests()
        if not action_requests:
            action_requests = [normalize_tool_call(tool_call) for tool_call in fallback_tool_calls]

        payload: dict[str, Any] = {"action_requests": action_requests}

        interrupt_metadata = state_view.interrupt_metadata()
        if interrupt_metadata:
            payload["metadata"] = interrupt_metadata

        return payload

    @staticmethod
    def _interrupt_payload_from_pending_interrupts(snapshot: Any) -> dict[str, Any] | None:
        """Recover the live interrupt payload from the checkpoint's pending interrupts.

        A node suspends *before* its writes land in the checkpoint, so the
        ``pending_action_requests`` that ``_prepare_interrupt_payload`` stashes in
        ``state["context"]`` right before ``interrupt()`` is never committed at pause
        time. Worse, once the first approval cycle resolves (the node returns normally)
        a now-stale value gets committed and would be reused for every later interrupt
        on the thread. Reading it back from ``snapshot.values`` therefore yields tool
        calls whose ids no longer match the resumed node's ``last_message.tool_calls``.

        The interrupt's own value, however, IS checkpointed and current. Prefer it so
        the action_requests (and provenance metadata) presented to the human always
        line up with the tool calls the graph will apply decisions to on resume.
        """
        interrupts = list(getattr(snapshot, "interrupts", None) or ())
        if not interrupts:
            for task in getattr(snapshot, "tasks", None) or ():
                interrupts.extend(getattr(task, "interrupts", None) or ())

        action_requests: list[Any] = []
        metadata: dict[str, Any] = {}
        for item in interrupts:
            value = getattr(item, "value", None)
            if not isinstance(value, dict):
                continue
            requests = value.get("action_requests")
            if isinstance(requests, list):
                action_requests.extend(requests)
            value_metadata = value.get("metadata")
            if isinstance(value_metadata, dict):
                metadata.update(value_metadata)

        if not action_requests:
            return None

        payload: dict[str, Any] = {"action_requests": action_requests}
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _build_interrupt_agent_response(
        self,
        state_snapshot: Any,
        thread_id: str | None,
        fallback_conversation_id: str | None = None,
    ) -> AgentResponse | None:
        """The response for a turn waiting on a human, or nothing.

        The pending interrupts decide this, not the parent message list. A
        Planning worker's gated tool call lives in the worker's own private
        messages and never reaches the parent's, so requiring a parent
        ``AIMessage`` with tool calls here meant a paused worker produced no
        approval request at all.
        """
        if not state_snapshot or not thread_id:
            return None

        payload = pending_interrupt_payload(state_snapshot)
        if payload is None:
            return None

        values = getattr(state_snapshot, "values", {})
        state_view = GraphStateView(values)
        conversation_id = state_view.conversation_id() or fallback_conversation_id or ""
        interrupt_response = build_interrupt_response(
            payload.to_interrupt_payload(), thread_id, conversation_id
        )

        active_agent_id = state_view.active_agent_id() or "search_agent"

        agent = self.agents.get(active_agent_id)
        agent_type = (
            agent.agent_type if agent and hasattr(agent, "agent_type") else AgentType.SEARCH
        )

        response = AgentResponse(
            agent_type=agent_type,
            agent_id=active_agent_id or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            metadata={"interrupt": interrupt_response},
        )
        self._attach_final_agent_metadata(values, response)
        return response

    def build_transition_resolver(self) -> TransitionResolver:
        """The sole parent-level transition resolver for this workflow.

        It validates against the same live inventory the router used, so a
        target that was routable at routing time is judged by the same rules
        when a specialist tries to hand off to it.
        """
        return TransitionResolver(
            inventory=build_runtime_inventory(
                base_agent_ids=list(self.agents.keys()),
                custom_agents={},
                max_custom_agents=settings.router_context_max_custom_agents,
            ),
            max_delegation_depth=settings.max_handoff_delegation_depth,
        )

    async def _prepare_turn_runtime_context(self, state: GraphState) -> WorkflowRuntimeContext:
        """Snapshot the routable inventory and descriptive context for one turn.

        Canvas state and attached custom agents are *context*: they widen what
        the router can choose, and never preselect an agent.
        """
        self._reset_agent_trail(state)

        conversation_id = state.get("conversation_id")
        active_canvas = await self._get_active_canvas_snapshot(
            conversation_id, state.get("user_id")
        )
        active_canvas_descriptor = active_canvas.descriptor() if active_canvas else None
        if active_canvas_descriptor:
            context = dict(state.get("context") or {})
            context["active_canvas"] = active_canvas_descriptor
            state["context"] = context

        inventory = build_runtime_inventory(
            base_agent_ids=list(self.agents.keys()),
            custom_agents=GraphStateView(state).custom_agents(),
            max_custom_agents=settings.router_context_max_custom_agents,
        )
        return WorkflowRuntimeContext(
            routing_service=self.routing_service,
            inventory=inventory,
            routing_context_builder=self.routing_context_builder,
            active_canvas=active_canvas_descriptor,
            runtime_time=build_runtime_time_context_block().strip() or None,
        )

    def _conversation_has_documents(self, conversation_id: str | None) -> bool:
        if not conversation_id or not self.document_repository:
            return False
        try:
            return self.document_repository.count_by_conversation(UUID(conversation_id)) > 0
        except (ValueError, Exception):
            return False

    async def _aconversation_has_documents(self, conversation_id: str | None) -> bool:
        """Async twin of :meth:`_conversation_has_documents`.

        Routing runs before the first token, so this COUNT must not block the
        event loop and stall other in-flight streams.
        """
        if not conversation_id or not self.document_repository:
            return False
        try:
            count = await self.document_repository.acount_by_conversation(UUID(conversation_id))
            return count > 0
        except (ValueError, Exception):
            return False

    async def _compact_checkpoint_after_terminal_response(
        self,
        *,
        config: dict[str, Any] | None,
        thread_id: str | None,
    ) -> None:
        """Drop checkpoint messages once the request reaches a terminal state.

        DB is the canonical conversation transcript. Once the graph finishes
        (snapshot.next empty), the checkpoint's message history is stale
        duplicate memory — and intermediate tool-calling AIMessages /
        ToolMessages without ids would otherwise persist forever, leaking
        previous-turn tool calls into the next stream and into LangSmith
        traces. We now stamp ALL graph-produced messages with derived ids,
        so a single RemoveMessage sweep clears them.

        No-op when there is no checkpointer, no thread, or the graph is in an
        interrupted state (snapshot.next non-empty).
        """
        if not self.checkpointer or not config or not thread_id:
            return
        try:
            snapshot = await self.graph.aget_state(config)
        except Exception as exc:
            logger.warning("Checkpoint compaction aget_state failed: %s", exc)
            return
        if snapshot is None or getattr(snapshot, "next", None):
            return

        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", []) if isinstance(values, dict) else []
        removals = [RemoveMessage(id=msg.id) for msg in messages if getattr(msg, "id", None)]
        if not removals:
            return

        try:
            with tracing_context(enabled=False):
                await self.graph.aupdate_state(config, {"messages": removals})
        except Exception as exc:
            logger.warning(
                "Checkpoint compaction aupdate_state failed for thread=%s: %s",
                thread_id,
                exc,
            )

    async def compact_checkpoint_after_terminal_response(self, thread_id: str | None) -> None:
        """Public service-layer hook for post-persistence checkpoint cleanup."""
        if not thread_id:
            return
        config = self._build_graph_config(thread_id)
        await self._compact_checkpoint_after_terminal_response(
            config=config,
            thread_id=thread_id,
        )

    def _build_graph_config(self, thread_id: str | None = None) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        react_iterations = max(1, int(getattr(settings, "react_agent_max_iterations", 1) or 1))
        planning_iterations = max(
            1, int(getattr(settings, "planning_max_iterations", react_iterations) or 1)
        )
        max_iterations = max(react_iterations, planning_iterations)
        min_recursion_limit = (2 * max_iterations) + 5
        configured_recursion_limit = getattr(settings, "react_agent_recursion_limit", None)
        recursion_limit = (
            int(configured_recursion_limit)
            if configured_recursion_limit and configured_recursion_limit > 0
            else min_recursion_limit
        )
        if recursion_limit < min_recursion_limit:
            recursion_limit = min_recursion_limit
        config["recursion_limit"] = recursion_limit

        if self.checkpointer and thread_id:
            config.setdefault("configurable", {})["thread_id"] = thread_id

        return config or None

    def _resolve_thread_id(
        self,
        thread_id: str | None,
        conversation_id: str | None,
        turn_id: str | None = None,
    ) -> str | None:
        """Return the per-turn routing-v2 checkpoint thread id.

        Resume passes the exact stored ``thread_id``; it is never reconstructed
        from the conversation id. A new turn always gets its own thread, so
        append reducers stay turn-local and no v1 checkpoint is ever loaded.
        """
        if thread_id:
            return thread_id
        if conversation_id and turn_id:
            return build_checkpoint_thread_id(conversation_id, turn_id)
        return None

    def _resume_runtime_context(self, values: dict[str, Any]) -> "WorkflowRuntimeContext":
        """Rebuild the runtime context for a resumed turn, from its own state.

        A resume re-enters whatever node was paused and may still reach the
        transition resolver, which reads the *live* inventory from
        ``runtime.context`` precisely so a custom agent detached mid-turn
        cannot be handed off to. Without a context it silently falls back to the
        inventory captured when the graph was compiled.

        Read-only, unlike ``_prepare_turn_runtime_context``: the checkpointed
        values belong to a turn already in flight and must not be mutated on
        the way back in.
        """
        inventory = build_runtime_inventory(
            base_agent_ids=list(self.agents.keys()),
            custom_agents=GraphStateView(values).custom_agents(),
            max_custom_agents=settings.router_context_max_custom_agents,
        )
        return WorkflowRuntimeContext(
            routing_service=self.routing_service,
            inventory=inventory,
            routing_context_builder=self.routing_context_builder,
            active_canvas=(values.get("context") or {}).get("active_canvas"),
            runtime_time=build_runtime_time_context_block().strip() or None,
        )

    @staticmethod
    def _set_continuation_signal(
        state: GraphState,
        *,
        reason: str,
        scope: str,
        count: int,
        limit: int,
    ) -> None:
        """Record why a loop stopped short of a final answer.

        Read back by ``_get_planning_pause_details`` to report the pause to the
        caller. It does not resume anything: a turn that runs out of budget
        ends, and a task that needs more work is decomposed by Planning.
        """
        context = GraphStateView(state).context_copy()
        context["continuation_signal"] = {
            "reason": reason,
            "scope": scope,
            "count": count,
            "limit": limit,
        }
        state["context"] = context

    @staticmethod
    def _last_message_is_tool_output(state: GraphState | dict[str, Any]) -> bool:
        messages = GraphStateView(state).messages()
        return bool(messages and isinstance(messages[-1], ToolMessage))

    @staticmethod
    def _mark_force_final_response(
        state: GraphState,
        *,
        reason: str,
        scope: str,
        count: int,
        limit: int,
    ) -> None:
        context = GraphStateView(state).context_copy()
        context["force_final_response"] = True
        context["tool_budget"] = {
            "reason": reason,
            "scope": scope,
            "count": count,
            "limit": limit,
        }
        state["context"] = context

    @staticmethod
    def _final_response_kwargs(state: GraphState) -> dict[str, Any]:
        context = GraphStateView(state).context()
        kwargs: dict[str, Any] = {}

        # Inline rich-response inventory: surfaces compact item descriptors to
        # the answer-producing agent only when the rollout flag is enabled and
        # the request advertised the capability.
        inventory = _build_inline_rich_inventory_for_state(context)
        if inventory:
            kwargs["rich_response_inventory"] = inventory

        if not context.get("force_final_response"):
            return kwargs

        budget = context.get("tool_budget")
        count = budget.get("count") if isinstance(budget, dict) else None
        limit = budget.get("limit") if isinstance(budget, dict) else None
        if isinstance(count, int) and isinstance(limit, int):
            notice = (
                f"Tool-use budget reached after {count}/{limit} tool iteration(s). "
                "Use the tool results already present in this conversation and produce "
                "the best final answer now. Do not call any more tools."
            )
        else:
            notice = (
                "Tool-use budget reached. Use the tool results already present in this "
                "conversation and produce the best final answer now. Do not call any more tools."
            )
        kwargs.update({"disable_tools": True, "tool_budget_notice": notice})
        return kwargs

    @staticmethod
    def _finalize_forced_final_response(
        state: GraphState,
        response: AgentResponse,
    ) -> AgentResponse:
        context = GraphStateView(state).context_copy()
        if not context.get("force_final_response"):
            return response

        budget = context.get("tool_budget")
        if response.metadata is None:
            response.metadata = {}
        if isinstance(budget, dict):
            response.metadata["tool_budget_exhausted"] = dict(budget)
        else:
            response.metadata["tool_budget_exhausted"] = True

        if response.message.tool_calls:
            logger.warning(
                "Model returned tool calls during forced final response; dropping %d call(s)",
                len(response.message.tool_calls),
            )
            response.message.tool_calls = None
            if not coerce_response_text(response.message.content):
                response.message.content = (
                    "I reached the tool-use limit before I could make additional tool calls. "
                    "Based on the tool results already gathered, I cannot complete the "
                    "remaining lookup reliably in this turn."
                )

        context.pop("force_final_response", None)
        context.pop("tool_budget", None)
        state["context"] = context
        return response

    @staticmethod
    def _get_continuation_signal(
        state_values: dict[str, Any] | None,
    ) -> ContinuationSignal:
        return GraphStateView(state_values).continuation_signal()

    @classmethod
    def _get_planning_pause_details(
        cls,
        state_values: dict[str, Any] | None,
    ) -> tuple[str | None, bool]:
        state_view = GraphStateView(state_values)
        pause_reason = state_view.context().get("pause_reason")
        signal = cls._get_continuation_signal(state_values)
        signal_scope = signal.get("scope")
        signal_reason = signal.get("reason")

        if not pause_reason and signal_scope == "planning" and isinstance(signal_reason, str):
            pause_reason = signal_reason

        planning_budget_reached = (
            signal_scope == "planning" and signal_reason == "max_iterations_reached"
        )
        return pause_reason, planning_budget_reached

    @classmethod
    def _attach_planning_state_metadata(
        cls,
        response: AgentResponse,
        state_values: dict[str, Any] | None,
    ) -> AgentResponse:
        if not isinstance(state_values, dict):
            return response

        if response.metadata is None:
            response.metadata = {}

        state_view = GraphStateView(state_values)
        todos = state_values.get("todos", [])
        if todos:
            response.metadata["todos"] = todos

        response.metadata["planning_call_count"] = state_view.planning_call_count()

        context = state_view.context()
        if context.get("all_tasks_completed"):
            response.metadata["all_tasks_completed"] = True
        for key in ("subagent_dispatches", "subagent_results"):
            value = context.get(key)
            if isinstance(value, list) and value:
                response.metadata[key] = make_json_safe(value)

        # Enrich Planning worker results with display identity (agent_name,
        # agent_kind, custom_agent_id) against the final custom-agent map while
        # keeping the existing ``subagent_results`` UI contract.
        subagent_results = normalize_subagent_metadata(
            response.metadata.get("subagent_results"),
            state_view.custom_agents(),
        )
        if subagent_results:
            response.metadata["subagent_results"] = subagent_results

        worker_artifacts = context.get("subagent_worker_artifacts")
        if isinstance(worker_artifacts, dict) and worker_artifacts:
            response.metadata["subagent_worker_artifacts"] = make_json_safe(worker_artifacts)

        planning_rubric = context.get("planning_rubric")
        if isinstance(planning_rubric, dict) and planning_rubric:
            response.metadata["planning_rubric"] = make_json_safe(planning_rubric)

        pause_reason, planning_budget_reached = cls._get_planning_pause_details(state_values)
        if planning_budget_reached:
            response.metadata["planning_budget_reached"] = True
        if pause_reason:
            response.metadata["pause_reason"] = pause_reason

        return response

    def _active_client_session_id(
        self,
        *,
        user_id: str | None,
        device_id: str | None,
    ) -> str | None:
        if not user_id or not device_id:
            return None
        try:
            from .client_runtime_tools import get_active_client_runtime_session

            session = get_active_client_runtime_session(user_id=user_id, device_id=device_id)
        except Exception:
            return None
        return session.session_id if session is not None else None

    def _persist_deferred_tool_snapshot_to_state(
        self,
        state: GraphState,
        *,
        agent: Any,
    ) -> None:
        state_view = GraphStateView(state)
        conversation_id = state_view.conversation_id()
        if not conversation_id:
            return

        agent_key = (
            getattr(agent, "tool_state_key", None)
            or getattr(agent, "agent_config_key", None)
            or getattr(agent, "agent_id", None)
            or "unknown"
        )
        user_id = state_view.user_id()
        device_id = state_view.device_id()
        session_id = self._active_client_session_id(user_id=user_id, device_id=device_id)

        try:
            from .deferred_tool_state import get_deferred_tool_state

            snapshot = get_deferred_tool_state().snapshot(
                conversation_id=conversation_id,
                agent_key=agent_key,
                device_id=device_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.debug("Failed to persist deferred tool snapshot: %s", exc)
            return

        if not snapshot.get("server_tools") and not snapshot.get("client_tools"):
            return

        context = state_view.context_copy()
        context["deferred_tool_snapshot"] = make_json_safe(snapshot)
        state["context"] = context

    def _hydrate_deferred_tool_snapshot_from_state(
        self,
        state: GraphState,
        *,
        agent: Any,
    ) -> bool:
        state_view = GraphStateView(state)
        context = state_view.context()
        snapshot = context.get("deferred_tool_snapshot")
        if not isinstance(snapshot, dict):
            return False

        conversation_id = state_view.conversation_id()
        if not conversation_id:
            return False

        agent_key = (
            getattr(agent, "tool_state_key", None)
            or getattr(agent, "agent_config_key", None)
            or getattr(agent, "agent_id", None)
            or "unknown"
        )
        user_id = state_view.user_id()
        device_id = state_view.device_id()
        session_id = self._active_client_session_id(user_id=user_id, device_id=device_id)

        try:
            from .deferred_tool_state import get_deferred_tool_state

            restored = get_deferred_tool_state().restore(
                conversation_id=conversation_id,
                agent_key=agent_key,
                snapshot=snapshot,
                device_id=device_id,
                session_id=session_id,
                user_id=user_id,
            )
        except Exception as exc:
            logger.debug("Failed to hydrate deferred tool snapshot: %s", exc)
            return False

        return bool(restored.get("server_tools") or restored.get("client_tools"))

    @staticmethod
    def _lookup_tool_render_payload(
        state_values: dict[str, Any] | None,
        tool_call_id: Any,
    ) -> dict[str, Any] | None:
        if not state_values or not tool_call_id:
            return None
        context = state_values.get("context")
        if not isinstance(context, dict):
            return None
        render_results = context.get("tool_render_results")
        if not isinstance(render_results, dict):
            return None
        render = render_results.get(str(tool_call_id))
        return render if isinstance(render, dict) else None

    # ------------------------------------------------------------------
    # Routing-v2 specialist subgraphs
    # ------------------------------------------------------------------

    # ============================================================
    # Planning node collaborators
    # ============================================================

    async def _planning_model_call(self, state: GraphState) -> AgentResponse:
        """One Planning model turn, and nothing else.

        Deliberately free of state mutation and of the grounding gate: the node
        owns both, so a helper that also wrote state would give the turn two
        places to decide what happened. ``dispatch_subagents`` is bound here as
        a schema only -- the server reads the proposal and owns the tasks.
        """
        state_view = GraphStateView(state)
        messages = state_view.messages()
        conversation_id = state_view.conversation_id()
        user_id = state_view.user_id()

        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="planning", state=state
        )
        context = state_view.context_copy()
        should_describe_plan = bool(context.pop("generate_plan_response", False))
        if should_describe_plan:
            state["context"] = context

        current_turn_messages = self._messages_for_active_agent(
            state, state.get("active_agent_id") or PLANNING_AGENT_ID, messages
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state, current_turn_messages
        )

        multi_agent_kwargs = self._multi_agent_kwargs(state, PLANNING_AGENT_ID)
        internal_tools = [
            build_dispatch_control_tool(),
            *(multi_agent_kwargs.get("internal_tools") or []),
        ]
        custom_workers = [
            {
                "runtime_agent_id": entry.get("runtime_agent_id") or runtime_id,
                "name": entry.get("name"),
                "description": entry.get("description"),
            }
            for runtime_id, entry in state_view.custom_agents().items()
        ]

        response = await self.planning_agent.invoke_model_with_history(
            messages=current_turn_messages,
            conversation_history=conversation_history,
            persona=state.get("persona"),
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            todos=state.get("todos") or [],
            current_task_index=state.get("current_task_index"),
            planning_phase=state.get("planning_phase", "planning"),
            should_describe_plan=should_describe_plan,
            internal_tools=internal_tools or None,
            custom_workers=custom_workers or None,
            planning_rubric_feedback=context.get("planning_rubric_feedback"),
            handoff_target_descriptions=multi_agent_kwargs.get("handoff_target_descriptions"),
            multi_agent_activity=multi_agent_kwargs.get("multi_agent_activity"),
            **self._final_response_kwargs(state),
        )
        self._mark_response_has_images(response, has_images)
        return self._finalize_forced_final_response(state, response)

    def _planning_inventory_for(self, state: GraphState):
        """The live inventory a dispatch proposal is validated against.

        The same snapshot the router and the transition resolver use, so an
        agent that is routable for this request is judged by one rule wherever
        it is named.
        """
        return build_runtime_inventory(
            base_agent_ids=list(self.agents.keys()),
            custom_agents=GraphStateView(state).custom_agents(),
            max_custom_agents=settings.router_context_max_custom_agents,
        )

    @staticmethod
    def _planning_allowed_tools(agent_id: str, state: GraphState) -> tuple[str, ...]:
        """Per-task tool narrowing for one worker.

        Empty means "this agent's own scope", not "no tools": the restriction a
        dispatch actually applies today is the *agent identity* it chose, and a
        chat worker already cannot reach canvas tools. Returning a name list
        here would require async tool discovery per task, so per-task narrowing
        is not derived yet -- ``WorkerToolScopeMiddleware`` enforces whatever
        this returns, so narrowing becomes available the moment it does.
        """
        return ()

    async def _planning_apply_todo_actions(
        self, state: GraphState, calls: list[dict[str, Any]]
    ) -> TodoActionOutcome:
        """Apply this turn's ``write_todos`` calls through the shared applier.

        The plan-mutation rules and the todo cap live in ``todo_actions``;
        reimplementing them in the graph layer is how they would drift.
        """
        from .todo_actions import apply_write_todos_action

        todos = list(state.get("todos") or [])
        current_task_index = state.get("current_task_index")
        max_todos = getattr(settings, "max_todos_per_plan", 50)

        messages: list[Any] = []
        actions: list[str] = []
        had_error = False

        for call in calls:
            tool_args = call.get("args") or {}
            try:
                todos, current_task_index, result, action = apply_write_todos_action(
                    todos=todos,
                    current_task_index=current_task_index,
                    tool_args=tool_args,
                    max_todos=max_todos,
                )
                actions.append(action)
                if action == "set_todos" and result.startswith("Error: Plan exceeds maximum"):
                    logger.warning(
                        "Rejected plan with %d todos (max: %d)",
                        len(tool_args.get("todos", []) or []),
                        max_todos,
                    )
            except Exception as exc:  # noqa: BLE001 - reported back to the model
                raw_action = tool_args.get("action")
                action = getattr(raw_action, "value", raw_action)
                result = f"Error executing {action}: {exc}"
                had_error = True

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=str(call.get("id") or ""),
                    name=str(call.get("name") or "write_todos"),
                    status="error" if had_error else "success",
                )
            )

        return TodoActionOutcome(
            todos=todos,
            current_task_index=current_task_index,
            tool_messages=tuple(messages),
            actions=tuple(actions),
            had_error=had_error,
        )

    def _grounded_answer_gate(self) -> GroundedAnswerGate:
        """The one grounding gate this workflow uses.

        Planning grounds its own synthesis, so the gate has to be reachable
        without a RAG agent in play. Its configuration never depended on one --
        only the cached instance did.
        """
        gate = getattr(getattr(self, "rag_agent", None), "grounded_answer_gate", None)
        if isinstance(gate, GroundedAnswerGate):
            return gate
        return GroundedAnswerGate(
            getattr(settings, "min_citation_coverage", 0.5),
            metrics=rag_metrics,
        )

    async def _review_planning_todos_with_rubric(
        self,
        *,
        state: GraphState,
        todos: list[dict[str, Any]],
        source: str = "planning_actions",
    ) -> Any:
        """Grade a candidate plan, degrading to a typed fallback on any failure.

        A grader outage must not block the plan: the attempt records why it
        could not grade rather than silently reporting a pass.
        """
        from app.ai.planning_rubric import FALLBACK_PLANNING_RUBRIC, PlanningRubricAttempt

        if not getattr(settings, "planning_rubric_enabled", True):
            return PlanningRubricAttempt(
                status="disabled",
                iterations=0,
                source=source,
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
            )

        context = GraphStateView(state).context_copy()
        previous = context.get("planning_rubric")
        previous_iterations = 0
        if (
            isinstance(previous, dict)
            and previous.get("source") == source
            and previous.get("status") == "needs_revision"
        ):
            try:
                previous_iterations = int(previous.get("iterations") or 0)
            except (TypeError, ValueError):
                previous_iterations = 0
        try:
            # PlanningAgent owns the grader so provider/model behavior stays
            # centralized; this is the only thing it is still kept for besides
            # the Planning model and prompt.
            return await self.planning_agent.review_todos_with_planning_rubric(
                user_message=self._latest_user_text(state),
                candidate_todos=todos,
                existing_todos=state.get("all_tasks") or [],
                plan_modified=bool(state.get("has_existing_plan")),
                source=source,
                start_iteration=previous_iterations,
                user_id=state.get("user_id"),
                model_request=state.get("model_request"),
            )
        except Exception as exc:  # noqa: BLE001 - a grader outage is not a plan failure
            logger.warning("Planning rubric review failed: %s", exc)
            return PlanningRubricAttempt(
                status="grader_error",
                iterations=previous_iterations,
                source=source,
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
                feedback=f"Planning rubric review failed: {exc}",
                error=f"Planning rubric review failed: {exc}",
            )

    async def _planning_review_rubric(self, state: GraphState, todos: list[dict[str, Any]]):
        """Grade a plan mutation against the planning rubric."""
        return await self._review_planning_todos_with_rubric(
            state=state, todos=todos, source="planning_actions"
        )

    def _build_specialist_factory(self, receipt_service: Any = None) -> SpecialistFactory:
        """Register the standard specialists as per-invocation subgraphs.

        Definitions come from the agent modules so each agent keeps its own
        prompt and tool knowledge; the loop itself belongs to ``create_agent``.
        """
        definitions = {
            "chat_agent": build_chat_specialist_definition(self.chat_agent),
            "search_agent": build_search_specialist_definition(self.search_agent),
            "canvas_agent": build_canvas_specialist_definition(self.canvas_agent),
            "image_generator_agent": build_image_generator_specialist_definition(
                self.image_generator_agent
            ),
        }
        return SpecialistFactory(
            definitions=definitions,
            runtime_model_resolver=self._runtime_model_resolver,
            model_factory=ModelFactory,
            usage_recorder=self._model_usage_recorder,
            settings=settings,
            receipt_service=receipt_service,
        )

    async def _specialist_request_for(self, node_name: str, state: GraphState) -> SpecialistRequest:
        """Assemble one invocation's authenticated scope and inputs."""
        state_view = GraphStateView(state)
        messages = state_view.messages()
        conversation_id = state_view.conversation_id()
        user_id = state_view.user_id()
        active_agent_id = state_view.active_agent_id() or node_name

        # The image generator deliberately borrows the chat history budget.
        history_key = "search" if node_name == "search_agent" else "chat"
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key=history_key, state=state
        )

        current_turn_messages = self._messages_for_active_agent(state, active_agent_id, messages)
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state, current_turn_messages
        )

        # Split the legacy invocation kwargs by owner: what shapes the prompt
        # goes to the prompt factory, what shapes the tool set goes to the tool
        # factory. Collapsing them would silently re-bind tools on a forced
        # final response.
        invocation_kwargs: dict[str, Any] = {
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, active_agent_id),
        }
        disable_tools = bool(invocation_kwargs.pop("disable_tools", False))
        internal_tools = invocation_kwargs.pop("internal_tools", None)
        excluded_tool_names = invocation_kwargs.pop("excluded_tool_names", None)
        include_hand_off = invocation_kwargs.pop("include_hand_off", None)

        # A top-level mutation is receipt-backed too. Its dispatch is
        # "top-level" and its task is the active specialist, so it can never
        # collide on an execution key with a delegated call.
        turn_identity = state.get("turn_identity")
        extras: dict[str, Any] = {
            "has_images": has_images,
            "disable_tools": disable_tools,
            "internal_tools": internal_tools,
            "excluded_tool_names": excluded_tool_names,
            "include_hand_off": include_hand_off,
            "system_prompt_kwargs": invocation_kwargs,
            "thread_id": getattr(turn_identity, "checkpoint_thread_id", None),
            "turn_id": getattr(turn_identity, "turn_id", None),
            "task_id": active_agent_id,
        }
        if node_name == "canvas_agent":
            previous_artifact = await self._get_active_canvas_snapshot(conversation_id, user_id)
            extras["previous_artifact"] = previous_artifact
            extras["system_prompt_kwargs"]["previous_artifact"] = previous_artifact
            if previous_artifact is not None:
                # Editing an existing canvas returns the whole updated artifact,
                # so an in-place widget mutation would be overwritten by the
                # rewrite. Not binding the tools is what keeps the model from
                # spending a turn on one.
                extras["excluded_tool_names"] = CANVAS_EDIT_DENIED_TOOL_NAMES

        return SpecialistRequest(
            agent_id=active_agent_id,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=state_view.device_id(),
            persona=state.get("persona"),
            model_request=state.get("model_request"),
            messages=current_turn_messages,
            history=self._convert_history_for_specialist(conversation_history),
            state=dict(state) if isinstance(state, dict) else {},
            hitl_policy=state_view.context().get("hitl_policy"),
            attachments=state_view.attachments(),
            extras=extras,
        )

    def _convert_history_for_specialist(self, conversation_history: list[Any]) -> list[Any]:
        """Convert canonical prompt history into LangChain messages.

        The conversion (multimodal parts, memory framing) is agent-independent,
        so one agent's implementation is reused rather than duplicated.
        """
        if not conversation_history:
            return []
        return self.chat_agent._convert_history_to_langchain_messages(conversation_history)

    async def invoke_specialist_subgraph(self, node_name: str, state: GraphState):
        """Run one standard specialist and return its ``ResponseOutcome``.

        Custom agents resolve their definition from the live attachment on
        every turn, so an edited or detached configuration takes effect
        immediately.
        """
        request = await self._specialist_request_for(node_name, state)

        if node_name == "rag_agent":
            # RAG runs the shared compiled graph rather than a create_agent
            # subgraph: the model drives retrieval and every answer passes
            # through the same grounding gate a Planning RAG worker uses.
            return self._enrich_specialist_outcome(
                state, request, await self._invoke_rag_specialist(request, state)
            )

        if node_name == CUSTOM_AGENT_NODE:
            custom_agent = self._build_custom_agent(state, request.agent_id)
            if custom_agent is None:
                logger.warning(
                    "custom_agent subgraph reached for unattached id '%s'", request.agent_id
                )
                raise ValueError(f"custom agent {request.agent_id} is not attached")
            self._specialist_factory.register(build_custom_specialist_definition(custom_agent))

        agent = self._resolve_runtime_agent(state, request.agent_id)
        # Deferred tools loaded on an earlier turn live in the checkpoint, not
        # in process memory, so they have to be restored before the subgraph
        # resolves its tool set and persisted after it may have loaded more.
        self._hydrate_deferred_tool_snapshot_from_state(state, agent=agent)
        with (
            use_image_preview_emitter(self._build_image_preview_emitter(state)),
            use_media_delivery_service(self._build_media_delivery_service(state)),
        ):
            outcome = await self._specialist_factory.invoke(request)
        self._persist_deferred_tool_snapshot_to_state(state, agent=agent)

        return self._enrich_specialist_outcome(state, request, outcome)

    async def _invoke_rag_specialist(
        self, request: SpecialistRequest, state: GraphState
    ) -> ResponseOutcome:
        """Run the shared RAG graph for a public turn and map its result.

        The same object a Planning RAG worker runs, differing only in what the
        result becomes: a ``ResponseOutcome`` here, a ``WorkerResult`` there.
        Two graphs were how one path could quietly skip validation.
        """
        state_view = GraphStateView(state)
        history = await self._get_conversation_history(
            request.conversation_id, request.user_id, agent_key="rag", state=state
        )

        result = await self.rag_execution_graph.ainvoke(
            RagExecutionRequest(
                objective=_last_human_text(state_view.messages()),
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                device_id=request.device_id,
                persona=request.persona,
                model_request=request.model_request,
                history=list(history or []),
                hitl_policy=dict(request.hitl_policy or {}),
                attachments=list(request.attachments),
                mode="public",
            )
        )

        evidence = tuple(result.evidence)
        artifacts = tuple(result.artifacts)
        images = tuple(result.images)
        policies: tuple[str, ...] = ("public_content",)
        if evidence:
            policies = (*policies, "rag_grounding")

        response = AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=result.content),
            metadata={
                "grounded_answer": result.grounding.to_metadata(),
                # Same key the create_agent specialists use, so the graph reads
                # one place regardless of which execution path answered.
                **(
                    {"execution_budget": dict(result.execution_budget)}
                    if result.execution_budget
                    else {}
                ),
                **({"images": list(images)} if images else {}),
            },
            tool_artifacts=list(artifacts) or None,
        )
        return ResponseOutcome(
            agent_id="rag_agent",
            response=response,
            provenance=OutcomeProvenance(
                output_policy_ids=policies,
                evidence=evidence,
                artifacts=artifacts,
                images=images,
            ),
        )

    def _enrich_specialist_outcome(
        self, state: GraphState, request: SpecialistRequest, outcome: Any
    ) -> Any:
        """Attach the domain metadata the public response still needs.

        Artifact and image provenance stays server-owned: it is copied from
        what the tool pipeline recorded, never from model text.
        """
        self._record_specialist_tool_results(state, outcome.provenance)
        response = outcome.response
        self._mark_response_has_images(response, bool(request.extras.get("has_images")))
        response = self._finalize_forced_final_response(state, response)
        append_images = request.agent_id == "image_generator_agent"
        self._merge_tool_artifacts(state, response, append_images=append_images)
        self._attach_final_agent_metadata(state, response)
        return outcome.model_copy(update={"response": response})

    def _record_specialist_tool_results(self, state: GraphState, provenance: Any) -> None:
        """Publish a subgraph's tool records into turn context.

        The trace panel, rich-item registry, and evidence lookup all read turn
        context rather than the subgraph's private state, so a specialist that
        keeps its records to itself is invisible to every one of them.
        """
        artifacts = list(provenance.artifacts)
        images = list(provenance.images)
        if not artifacts and not images:
            return

        context = GraphStateView(state).context_copy()
        if artifacts:
            context["tool_artifacts"] = [*context.get("tool_artifacts", []), *artifacts]
            renders = dict(context.get("tool_render_results", {}))
            renders.update(
                {
                    str(artifact["tool_call_id"]): make_json_safe(artifact["render"])
                    for artifact in artifacts
                    if artifact.get("tool_call_id") and isinstance(artifact.get("render"), dict)
                }
            )
            if renders:
                context["tool_render_results"] = renders
        if images:
            context["tool_images"] = [*context.get("tool_images", []), *images]

        # Candidate records are turn-internal handoff data. Lift and remove
        # them before tool artifacts can reach persisted response metadata.
        self._lift_rich_candidates(context, artifacts)
        state["context"] = context
        self._update_tool_error_streak(state, artifacts)

    # ------------------------------------------------------------------
    # Custom-agent multiplexing
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Multi-agent awareness (roster + per-turn invocation trail)
    # ------------------------------------------------------------------
    def _build_chat_image_loader(self, user_id):
        """Build a per-run resolver that maps a stored ``image_id`` to a base64
        data URL, so historical image references are re-sent to the model. The
        per-run count is capped by ``chat_image_history_rehydrate_limit`` (0 =
        unlimited) to bound storage reads and injected bytes per request."""
        service = getattr(self, "chat_image_service", None)
        if service is None or not user_id:
            return None
        limit = settings.chat_image_history_rehydrate_limit
        state = {"resolved": 0, "dropped": 0}

        def _loader(image_id: str) -> str | None:
            if limit and state["resolved"] >= limit:
                state["dropped"] += 1
                if state["dropped"] == 1:
                    logger.info(
                        "chat image rehydrate cap reached (limit=%d); dropping older "
                        "history images from model context code=chat_image_rehydrate_capped",
                        limit,
                    )
                return None
            try:
                data_url = service.load_data_url(UUID(str(image_id)), user_id)
            except Exception:
                return None
            if data_url:
                state["resolved"] += 1
            return data_url

        return _loader

    def _build_image_preview_emitter(self, state: GraphState):
        """Bind an image-preview emitter to this run's custom event channel.

        Returns None when previews are disabled or there is no run to write
        into. Previews used to travel on a side queue reached through a weak
        token in checkpoint state; a resumed run resolved that token to a dead
        sink and had to rebind one under it. The graph's own channel needs
        neither, and it works identically on a resume.

        The side queue also bounded memory by dropping bulky in-progress frames
        once it saturated. That protection is still needed and is applied here
        instead: the custom channel does **not** backpressure its writer -- a
        node emitting 2000 frames finishes while a sleeping consumer holds one,
        so without a cap a slow client buffers every partial. Only partials are
        droppable; a final delivery and every lifecycle frame always go through.
        """
        if not settings.enable_image_streaming:
            return None
        writer = _graph_stream_writer()
        if writer is None:
            return None

        cap = int(getattr(settings, "image_preview_max_partials_per_image", 512) or 512)
        written_partials: dict[Any, int] = {}

        def _emit(payload: dict[str, Any]) -> None:
            if payload.get("status") == IMAGE_PREVIEW_STATUS_PARTIAL:
                item_id = payload.get("item_id")
                seen = written_partials.get(item_id, 0)
                if seen >= cap:
                    return
                written_partials[item_id] = seen + 1
            writer({"type": "image_preview", **payload})

        return _emit

    def _build_media_delivery_service(self, state: GraphState) -> MediaDeliveryService:
        """Bind this run's media delivery service at the graph boundary.

        Carries the run's user/conversation context and the storage backend so
        the agent can persist a final image the moment it is produced without
        reaching into the DI container. The preview publisher captures the sink
        installed by ``use_image_preview_emitter`` (None on resume/non-streaming,
        making previews a no-op while durable persistence still runs)."""

        def _coerce_uuid(value: Any) -> Any:
            if value is None or isinstance(value, UUID):
                return value
            try:
                return UUID(str(value))
            except (ValueError, TypeError, AttributeError):
                return value

        conversation_id = state.get("conversation_id") if isinstance(state, dict) else None
        user_id = state.get("user_id") if isinstance(state, dict) else None
        return MediaDeliveryService(
            storage=getattr(self, "chat_image_service", None),
            conversation_id=_coerce_uuid(conversation_id),
            user_id=_coerce_uuid(user_id),
            preview_publisher=ImagePreviewPublisher(
                enabled=settings.enable_image_streaming,
                max_b64_chars=settings.image_stream_preview_max_b64_chars,
            ),
            request_id=str(conversation_id) if conversation_id else None,
        )

    @staticmethod
    def _latest_user_text(state: GraphState) -> str:
        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                return str(message.content or "")
        return ""

    def _get_agent_type(self, active_agent_id: str | None) -> AgentType:
        agent_type_map = {
            "chat_agent": AgentType.CHAT,
            "rag_agent": AgentType.RAG,
            "search_agent": AgentType.SEARCH,
            "image_generator_agent": AgentType.IMAGE_GENERATOR,
            "planning_agent": AgentType.PLANNING,
            "canvas_agent": AgentType.CANVAS,
        }
        return agent_type_map.get(active_agent_id, AgentType.CHAT)

    @staticmethod
    def _merge_unique_items(
        existing_items: list[Any] | None, new_items: list[Any] | None
    ) -> list[Any]:
        merged = list(existing_items) if isinstance(existing_items, list) else []
        if not isinstance(new_items, list):
            return merged

        for item in new_items:
            if item not in merged:
                merged.append(item)

        return merged

    def _attach_context_outputs(
        self, state: dict[str, Any], response: AgentResponse
    ) -> AgentResponse:
        state_view = GraphStateView(state)
        tool_artifacts = self._merge_unique_items(
            response.tool_artifacts, state_view.tool_artifacts()
        )
        response.tool_artifacts = tool_artifacts or None

        if response.metadata is None:
            response.metadata = {}

        images = self._merge_unique_items(response.metadata.get("images"), state_view.tool_images())
        if images:
            response.metadata["images"] = images

        self._attach_final_agent_metadata(state, response)
        return response

    def _finalized_response(
        self,
        state: dict[str, Any] | None,
    ) -> AgentResponse | None:
        """Return the finalizer's validated response, or nothing.

        Named for what it does. Its predecessor salvaged assistant-looking
        text out of stream chunks and checkpoint messages -- a path around
        ``validate_output`` publishing content no output policy had approved.
        The salvage went in ``051092b``; the misleading name outlived it.

        This deliberately does not salvage. It used to scan accumulated stream
        chunks and checkpoint messages for assistant-looking text and publish
        that — a path around ``validate_output`` carrying content no output
        policy had approved, attributed to an agent the runtime inferred.

        A turn that produced no finalized response is a failed turn: the caller
        emits its typed error rather than a recovered draft.
        """
        if not isinstance(state, dict):
            return None

        response = state.get("response")
        if not response:
            return None

        # A response still carrying tool calls is mid-flight, not an answer.
        response_message = getattr(response, "message", None)
        if getattr(response_message, "tool_calls", None):
            return None
        if not coerce_response_text(getattr(response_message, "content", None)):
            return None

        return self._attach_context_outputs(state, response)

    # ------------------------------------------------------------------
    # Continuation helpers
    # ------------------------------------------------------------------

    async def execute_request(self, request: WorkflowExecutionRequest) -> AgentResponse | None:
        initial_state = self._build_initial_state_from_request(request)
        conversation_id = request.conversation_id
        thread_id = self._resolve_thread_id(
            request.thread_id, conversation_id, request.turn_id or request.user_message_id
        )
        config = self._build_graph_config(thread_id)
        # Routing runs inside the graph's `route` node; the runtime context
        # carries the collaborators it needs without entering checkpoint state.
        runtime_context = await self._prepare_turn_runtime_context(initial_state)

        # `context=` is the only channel LangGraph reads. A context placed in
        # `config` is silently dropped, which left `route` seeing
        # runtime.context=None and failing every turn with `runtime_missing`.
        result = await self.graph.ainvoke(
            initial_state, config=config, context=runtime_context
        )

        if self.checkpointer and thread_id:
            state_snapshot = await self.graph.aget_state(config)
            if state_snapshot.next and len(state_snapshot.next) > 0:
                interrupt_agent_response = self._build_interrupt_agent_response(
                    state_snapshot, thread_id, conversation_id
                )
                if interrupt_agent_response:
                    return interrupt_agent_response

        final_state = result if isinstance(result, dict) else None
        agent_response = self._finalized_response(final_state)
        if not agent_response and self.checkpointer and thread_id:
            final_snapshot = await self.graph.aget_state(config)
            final_state = (
                final_snapshot.values
                if final_snapshot and hasattr(final_snapshot, "values")
                else None
            )
            agent_response = self._finalized_response(final_state)

        final_agent_id = (
            final_state.get("active_agent_id") if isinstance(final_state, dict) else None
        )
        if agent_response and final_agent_id == "planning_agent":
            agent_response = self._attach_planning_state_metadata(agent_response, final_state)

        return agent_response

    async def resume(
        self,
        thread_id: str,
        user_input: str | None = None,
    ) -> AgentResponse | None:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot resume.")

        config = self._build_graph_config(thread_id)

        # Get current state to extract tool calls for auto-approval
        state_snapshot = await self.graph.aget_state(config)
        messages = state_snapshot.values.get("messages", [])

        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            # Auto-approve all tool calls
            resume_data = [
                {
                    "task_id": tc.get("id"),
                    "tool_call_id": tc.get("id"),
                    "type": "approve",
                    "args": None,
                }
                for tc in messages[-1].tool_calls
            ]
        else:
            resume_data = user_input

        result = await self.graph.ainvoke(
            Command(resume=resume_data),
            config=config,
            context=self._resume_runtime_context(state_snapshot.values),
        )

        # Check for further interrupts
        final_snapshot = await self.graph.aget_state(config)
        interrupt_response = self._build_interrupt_agent_response(
            final_snapshot,
            thread_id,
            final_snapshot.values.get("conversation_id"),
        )
        if interrupt_response:
            return interrupt_response

        response = self._finalized_response(result)
        if not response:
            final_state = (await self.graph.aget_state(config)).values
            response = self._finalized_response(final_state)

        return response

    async def resume_with_decisions_stream(
        self,
        thread_id: str,
        decisions: list[InterruptDecision],
    ):
        if not self.checkpointer:
            yield make_event(
                "error",
                sequence=0,
                data={"error": "Checkpointing is not enabled, cannot resume."},
            )
            return

        config = self._build_graph_config(thread_id)
        state_snapshot = await self.graph.aget_state(config)

        pending = pending_interrupt_payload(state_snapshot)
        if pending is None:
            raise ValueError("Workflow is not waiting on a human decision")

        resume_data = address_decisions_to_interrupts(
            build_interrupt_resume_payload(decisions), pending
        )

        active_agent_id = state_snapshot.values.get("active_agent_id", "search_agent")
        conversation_id = state_snapshot.values.get("conversation_id")

        # Resume parity needs nothing installed any more. An image generated
        # after a HITL resume emits its preview on the graph's own custom
        # channel, which the resumed run has just as much as the original did.
        # This used to rebind a live sink under a token the checkpoint still
        # carried, because the weakref behind it died with the first stream.
        resume_state_update: dict[str, Any] | None = None

        yield make_event(
            "agent_selected",
            sequence=0,
            agent=active_agent_id,
            data={"agent": active_agent_id},
        )

        suppressed_nodes: set = {"image_generator_agent"}
        suppress_tokens = active_agent_id in suppressed_nodes
        projector = GraphPublicStreamProjector(
            tool_end_events_from_node_state=self._tool_end_events_from_node_state,
            suppress_internal_stream_chunks=settings.suppress_internal_stream_chunks,
        )
        ctx = StreamProjectionContext(
            last_emitted_agent=active_agent_id,
            suppress_tokens=suppress_tokens,
        )

        # Rehydrate historical image references for the model on resume too,
        # otherwise a replayed turn drops prior images from context.
        chat_image_loader = self._build_chat_image_loader(state_snapshot.values.get("user_id"))

        try:
            merged = iter_v3_events_from_graph(
                self.graph,
                Command(resume=resume_data, update=resume_state_update),
                config=config,
                context=self._resume_runtime_context(state_snapshot.values),
            )
            with use_chat_image_loader(chat_image_loader):
                async for event in merged:
                    for public_event in projector.map_event(event, ctx):
                        yield public_event
        except Exception as exc:
            yield make_event("error", sequence=0, data={"error": str(exc)})
            return

        async for public_event in self._finish_stream(
            ctx, config=config, thread_id=thread_id, conversation_id=conversation_id
        ):
            yield public_event

    async def execute_request_stream(self, request: WorkflowExecutionRequest):
        initial_state = self._build_initial_state_from_request(request)
        conversation_id = request.conversation_id
        thread_id = self._resolve_thread_id(
            request.thread_id, conversation_id, request.turn_id or request.user_message_id
        )
        config = self._build_graph_config(thread_id)
        user_id = request.user_id

        # Prefetch conversation history in parallel with the router LLM call.
        # By the time the agent node needs history, the cache will be warm.
        history_prefetch: asyncio.Task | None = None
        if conversation_id and user_id:
            history_prefetch = asyncio.create_task(
                self._get_conversation_history(conversation_id, user_id)
            )

        # Routing happens inside the graph's `route` node. The stream adapter
        # observes the resulting state update and projects `agent_selected`
        # from it; it never pre-runs a node.
        try:
            runtime_context = await self._prepare_turn_runtime_context(initial_state)
        except Exception as exc:
            if history_prefetch and not history_prefetch.done():
                history_prefetch.cancel()
            yield make_event("error", sequence=0, data={"error": str(exc)})
            return

        # Per-stream accumulator state for the canonical event mapper. Token
        # suppression for agents whose raw stream is internal (image
        # generation) is applied once the route lands.
        projector = GraphPublicStreamProjector(
            tool_end_events_from_node_state=self._tool_end_events_from_node_state,
            suppress_internal_stream_chunks=settings.suppress_internal_stream_chunks,
        )
        ctx = StreamProjectionContext(
            last_emitted_agent=None,
            suppress_tokens=False,
        )

        chat_image_loader = self._build_chat_image_loader(user_id)

        try:
            merged = iter_v3_events_from_graph(
                self.graph, initial_state, config=config, context=runtime_context
            )
            with use_chat_image_loader(chat_image_loader):
                async for event in merged:
                    for public_event in projector.map_event(event, ctx):
                        yield public_event
        except Exception as exc:
            yield make_event("error", sequence=0, data={"error": str(exc)})
            return

        async for public_event in self._finish_stream(
            ctx, config=config, thread_id=thread_id, conversation_id=conversation_id
        ):
            yield public_event

    async def _finish_stream(
        self,
        ctx: StreamProjectionContext,
        *,
        config: dict[str, Any] | None,
        thread_id: str | None,
        conversation_id: str | None,
    ):
        """Close out a streamed turn: pending interrupt, or the final response.

        Both entry points end the same way, and the citation filter has to be
        flushed before either — it may still be holding the tail of a marker it
        had not yet seen the end of, and dropping it would truncate the answer.
        """
        for public_event in flush_answer_text(ctx):
            yield public_event

        snapshot = None
        if self.checkpointer and thread_id:
            try:
                snapshot = await self.graph.aget_state(config)
            except Exception as exc:  # noqa: BLE001 - reported, never published
                yield make_event("error", sequence=0, data={"error": str(exc)})
                return

        if snapshot is not None and snapshot.next:
            interrupt_event = self._pending_interrupt_event(
                snapshot, thread_id=thread_id, conversation_id=conversation_id
            )
            if interrupt_event is not None:
                yield interrupt_event
                return

        # The checkpoint is authoritative when it holds the finalized response;
        # the streamed updates are what a run without a checkpointer leaves
        # behind. Neither is a salvage path — both carry only what ``finalize``
        # already validated.
        for final_state in (getattr(snapshot, "values", None), ctx.last_state_values):
            response = self._finalized_response(final_state)
            if response is not None:
                break
        else:
            final_state = None
            response = None

        if response is None:
            yield make_event("error", sequence=0, data={"error": NO_RESPONSE_GENERATED})
            return

        if not ctx.internal_content_only:
            apply_accumulated_thinking(response, ctx.accumulated_thinking)
        if isinstance(final_state, dict) and final_state.get("active_agent_id") == "planning_agent":
            response = self._attach_planning_state_metadata(response, final_state)

        yield make_event("complete", sequence=0, data={"response": response})

    def _pending_interrupt_event(
        self,
        snapshot: Any,
        *,
        thread_id: str | None,
        conversation_id: str | None,
    ):
        """The interrupt event for a paused turn, or nothing if it is not paused.

        A pause is whatever the checkpoint is still waiting on. Anything else
        pending is a run that failed to reach the finalizer, which is an error
        rather than a question for the user.
        """
        pending = pending_interrupt_payload(snapshot)
        if pending is None:
            return None

        payload = pending.to_interrupt_payload()
        return make_event(
            "interrupt",
            sequence=0,
            data={
                "next": snapshot.next,
                "thread_id": thread_id,
                "pending_tool_calls": payload["action_requests"],
                "interrupt": build_interrupt_response(payload, thread_id, conversation_id or ""),
            },
        )

    async def get_state(self, thread_id: str) -> dict:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot get state.")

        config = self._build_graph_config(thread_id)
        snapshot = await self.graph.aget_state(config)

        pending = pending_interrupt_payload(snapshot)

        return {
            # "Interrupted" means waiting on a human, not merely having a next
            # node -- a turn mid-execution also has one.
            "interrupted": pending is not None,
            "next": snapshot.next,
            "values": snapshot.values,
            "pending_tool_calls": list(pending.action_requests) if pending else None,
            "pending_interrupt_ids": list(pending.interrupt_ids) if pending else [],
            "active_agent_id": snapshot.values.get("active_agent_id"),
        }

    async def cleanup(self):
        for agent in self._cleanup_agents:
            if hasattr(agent, "cleanup"):
                with contextlib.suppress(Exception):
                    await agent.cleanup()


def create_workflow(
    qdrant_client: QdrantClient,
    embedding_service: Any,
    checkpointer: BaseCheckpointSaver | None = None,
    document_repository: Optional["DocumentRepository"] = None,
    runtime_model_resolver: IRuntimeModelResolver | None = None,
    history_provider: ConversationHistoryProvider | None = None,
    model_usage_recorder: "ModelUsageRecorder | None" = None,
    chat_image_service: Any | None = None,
    tool_execution_receipt_repository: Any | None = None,
) -> MultiAgentWorkflow:
    """
    Create multi-agent workflow with required shared dependencies.
    """
    return MultiAgentWorkflow(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        checkpointer=checkpointer,
        document_repository=document_repository,
        runtime_model_resolver=runtime_model_resolver,
        history_provider=history_provider,
        model_usage_recorder=model_usage_recorder,
        chat_image_service=chat_image_service,
        tool_execution_receipt_repository=tool_execution_receipt_repository,
    )
