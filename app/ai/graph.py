import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cachetools import TTLCache
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
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
from ..services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)
from ..services.event_streaming.langchain_v3 import iter_v3_events_from_graph
from ..services.event_streaming.subagents import (
    SubagentEventSink,
    register_subagent_event_sink,
    stream_with_subagent_events,
)
from .agent_metadata import (
    attach_agent_metadata,
    normalize_handoff_metadata,
    normalize_subagent_metadata,
)
from .agents.canvas_agent import CanvasAgent
from .agents.chat_agent import ChatAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .agents.planning_agent import PlanningAgent
from .agents.rag_agent import RAGAgent
from .agents.router import Router
from .agents.search_agent import SearchAgent
from .custom_agent_runtime import is_custom_runtime_id
from .history import ConversationHistoryProvider
from .hitl_config import (
    build_interrupt_response,
)
from .image_context import build_multimodal_content, has_image_parts
from .memory import get_memory_manager
from .rag_tool_actions import canonicalize_rag_tool_call, execute_search_documents_action
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
from .token_instrumentation import (
    HistoryBudgetConfig,
    trim_history_to_budget,
)
from .tool_context import tool_execution_context
from .tool_execution import (
    apply_tool_output_offload,
    build_tool_artifact,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from .utils import (
    apply_hitl_decisions,
    build_interrupt_resume_payload,
    coerce_response_text,
    make_json_safe,
    normalize_tool_call,
)
from .workflow.custom_agents import CustomAgentsMixin
from .workflow.graph_builder import build_workflow_graph
from .workflow.planning_loop import PlanningLoopMixin
from .workflow.rag_loop import RagLoopMixin
from .workflow.tool_loop import ToolLoopMixin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository
    from .planning_subagents import SubagentModelOverride

_apply_decisions = apply_hitl_decisions

# Generic agents pause in the dedicated ``approval`` node. Planning and RAG
# perform their approval gates inside their tool-loop nodes, so LangGraph
# reports those nodes as the pending continuation target.
_APPROVAL_INTERRUPT_NODES = frozenset({"approval", "planning_tools", "rag_tools"})


def _has_approval_interrupt(next_nodes: Any) -> bool:
    return bool(set(next_nodes or ()) & _APPROVAL_INTERRUPT_NODES)


def _build_inline_rich_inventory_for_state(context: dict[str, Any] | None) -> str:
    """Compute the bounded rich-item inventory block for the current turn.

    Returns an empty string if the rollout flag is off, the request did not
    advertise the capability, or no candidates exist.
    """
    if not isinstance(context, dict):
        return ""
    if not getattr(settings, "inline_rich_response_enabled", False):
        return ""
    if not context.get("inline_rich_response_v1"):
        return ""
    candidates = context.get("rich_item_candidates") or []
    if not candidates:
        return ""
    from .prompts import build_rich_response_guidance

    return build_rich_response_guidance(
        candidates=list(candidates),
        enabled=True,
        capability=True,
    )


class MultiAgentWorkflow(
    ToolLoopMixin,
    CustomAgentsMixin,
    RagLoopMixin,
    PlanningLoopMixin,
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
    ):
        self.qdrant_client = qdrant_client
        # Canonical prompt-history source. When ``None`` the workflow falls
        # back to the legacy ``MemoryManager``-driven path so tests that
        # construct ``MultiAgentWorkflow.__new__`` directly keep working.
        self.history_provider = history_provider
        self.router = Router()
        self.chat_agent = ChatAgent(runtime_model_resolver=runtime_model_resolver)
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_service=embedding_service,
            collection_name=settings.qdrant_collection_name,
            runtime_model_resolver=runtime_model_resolver,
        )
        self.search_agent = SearchAgent(runtime_model_resolver=runtime_model_resolver)
        self.image_generator_agent = ImageGeneratorAgent(
            runtime_model_resolver=runtime_model_resolver
        )
        self.planning_agent = PlanningAgent(runtime_model_resolver=runtime_model_resolver)
        self.canvas_agent = CanvasAgent(runtime_model_resolver=runtime_model_resolver)
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

        # Conversation history cache with bounded size + automatic TTL eviction.
        # Replaces the plain dict to prevent unbounded memory growth.
        self._history_cache_ttl_seconds: int = 60
        self._history_cache: TTLCache = TTLCache(maxsize=256, ttl=self._history_cache_ttl_seconds)
        # Per-conversation lock to prevent duplicate DB lookups under concurrency.
        self._history_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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

        Returns ``None`` when the provider is not wired (e.g. tests that
        construct ``MultiAgentWorkflow.__new__`` directly) or when ids are
        missing — callers must handle this and fall back to the legacy path.
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
        Get conversation history with caching and budget trimming.

        When the workflow is wired with a ``ConversationHistoryProvider``
        (production path), the provider is the single source of truth: it
        returns DB-backed messages already excluding the current user turn
        by ``user_message_id`` and the durable summary cursor. The summary
        text is mirrored into ``state['history_summary']`` so existing
        agent nodes that read that field keep working.

        When the provider is absent (legacy/test path) we fall back to the
        ``MemoryManager`` + tail-position exclusion.

        History is trimmed according to agent-specific settings:
        - {agent_key}_history_max_messages
        - {agent_key}_history_max_tokens
        """
        if not conversation_id or not user_id:
            return []

        # Provider path — preferred.
        if self.history_provider is not None:
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
                if state is not None and context.summary:
                    state["history_summary"] = context.summary
                    if context.summary_message_id:
                        state["summary_cursor_message_id"] = context.summary_message_id
                return list(context.messages)

        # Legacy fallback (no provider wired).
        cache_key = conversation_id
        async with self._history_locks[cache_key]:
            if cache_key in self._history_cache:
                cached_history = self._history_cache[cache_key]
                if agent_key:
                    budget_config = HistoryBudgetConfig.for_agent(agent_key, settings)
                    return trim_history_to_budget(
                        cached_history,
                        max_messages=budget_config.max_messages,
                        max_tokens=budget_config.max_tokens,
                    )
                return cached_history

            try:
                memory_manager = get_memory_manager()
                conv_memory = await memory_manager.get_memory(
                    UUID(conversation_id), UUID(user_id), force_refresh=True
                )
                history = conv_memory.get_recent_messages(limit=None, exclude_last=1)

                self._history_cache[cache_key] = history

                if agent_key:
                    budget_config = HistoryBudgetConfig.for_agent(agent_key, settings)
                    return trim_history_to_budget(
                        history,
                        max_messages=budget_config.max_messages,
                        max_tokens=budget_config.max_tokens,
                    )
                return history
            except Exception:
                return []

    def invalidate_history_cache(self, conversation_id: str) -> None:
        """Invalidate cached history for a conversation (call when new messages added)."""
        self._history_cache.pop(conversation_id, None)
        if self.history_provider is not None:
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
            selected_agent_id=state.get("selected_agent"),
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

        # Remember the agent that produced this turn's response so the next
        # turn can stick to it (custom-agent stickiness — see _route_node). The
        # last writer in a turn wins, so after a handoff this lands on the agent
        # that actually answered, not the source.
        selected = state.get("selected_agent")
        if selected:
            state["last_agent"] = selected

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
        initial_state["selected_agent"] = None
        initial_state["response"] = None
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
        return build_workflow_graph(self, checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # hand_off delegation helper
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Delegated-agent message scoping
    # ------------------------------------------------------------------
    def _messages_for_selected_agent(
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

    async def _should_call_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        normalized_calls = [normalize_tool_call(tc) for tc in last_message.tool_calls]
        selected_agent_name = state.get("selected_agent")
        agent = self._resolve_runtime_agent(state, selected_agent_name)
        handoff_tool = self._handoff_tool_for_agent(state, selected_agent_name)
        scoped_internal_tools = [handoff_tool] if handoff_tool else None
        if await self._needs_approval(
            state,
            normalized_calls,
            agent=agent,
            internal_tools=scoped_internal_tools,
        ):
            return "approval"

        return "tools"

    def _route_tool_output(self, state: GraphState) -> str:
        state_view = GraphStateView(state)
        iteration_count = state_view.iteration_count()
        max_iterations = max(1, int(settings.react_agent_max_iterations))
        selected_agent = state_view.selected_agent() or "end"

        if (
            selected_agent != "end"
            and selected_agent not in self.agents
            and not self._is_attached_custom_agent(state, selected_agent)
        ):
            return "end"

        can_route_for_final_response = (
            selected_agent != "end" and self._last_message_is_tool_output(state)
        )

        streak = (state_view.context() or {}).get("tool_error_streak")
        if isinstance(streak, dict) and streak.get("count", 0) >= streak.get("limit", 3):
            if can_route_for_final_response:
                self._mark_force_final_response(
                    state,
                    reason="consecutive_tool_errors",
                    scope="runtime",
                    count=int(streak.get("count") or 0),
                    limit=int(streak.get("limit") or 0),
                )
                return self._route_target_for(state, selected_agent)
            self._set_continuation_signal(
                state,
                should_continue=False,
                reason="consecutive_tool_errors",
                scope="runtime",
                count=int(streak.get("count") or 0),
                limit=int(streak.get("limit") or 0),
            )
            return "end"

        # Soft-limit: if auto-continue is enabled, trigger continuation at
        # a fraction of the budget so the outer loop can start a new round
        # before the hard LangGraph recursion limit is hit.
        if settings.auto_continue_enabled:
            soft_limit = max(1, int(max_iterations * settings.auto_continue_soft_limit_ratio))
            if iteration_count >= soft_limit:
                if can_route_for_final_response:
                    self._mark_force_final_response(
                        state,
                        reason="soft_budget",
                        scope="runtime",
                        count=iteration_count,
                        limit=soft_limit,
                    )
                    return self._route_target_for(state, selected_agent)

                self._set_continuation_signal(
                    state,
                    should_continue=True,
                    reason="soft_budget",
                    scope="runtime",
                    count=iteration_count,
                    limit=soft_limit,
                )
                return "end"

        if iteration_count >= max_iterations:
            if can_route_for_final_response:
                self._mark_force_final_response(
                    state,
                    reason="max_iterations_reached",
                    scope="runtime",
                    count=iteration_count,
                    limit=max_iterations,
                )
                return self._route_target_for(state, selected_agent)

            if settings.auto_continue_enabled and state_view.messages():
                self._set_continuation_signal(
                    state,
                    should_continue=True,
                    reason="max_iterations_reached",
                    scope="runtime",
                    count=iteration_count,
                    limit=max_iterations,
                )
            else:
                self._set_continuation_signal(
                    state,
                    should_continue=False,
                    reason="max_iterations_reached",
                    scope="runtime",
                    count=iteration_count,
                    limit=max_iterations,
                )
            return "end"

        return self._route_target_for(state, selected_agent)

    def _build_interrupt_agent_response(
        self,
        state_snapshot: Any,
        thread_id: str | None,
        fallback_conversation_id: str | None = None,
    ) -> AgentResponse | None:
        if not state_snapshot or not thread_id:
            return None

        values = getattr(state_snapshot, "values", {})
        state_view = GraphStateView(values)
        messages = state_view.messages()
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None

        conversation_id = state_view.conversation_id() or fallback_conversation_id or ""
        interrupt_payload = self._get_interrupt_payload_from_state(
            values,
            last_message.tool_calls,
        )
        interrupt_response = build_interrupt_response(interrupt_payload, thread_id, conversation_id)

        selected_agent = state_view.selected_agent() or "search_agent"

        agent = self.agents.get(selected_agent)
        agent_type = (
            agent.agent_type if agent and hasattr(agent, "agent_type") else AgentType.SEARCH
        )

        response = AgentResponse(
            agent_type=agent_type,
            agent_id=selected_agent or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            metadata={"interrupt": interrupt_response},
        )
        self._attach_final_agent_metadata(values, response)
        return response

    async def _route_node(self, state: GraphState) -> GraphState:
        # Reset delegation counter and the per-turn invocation trail at the
        # start of each new user turn.
        state["delegation_count"] = 0
        self._reset_agent_trail(state)

        if state.get("selected_agent"):
            self._record_agent_invocation(state, state.get("selected_agent"), via="preselected")
            return state

        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        planning_mode_enabled, has_existing_plan = self._get_planning_flags(state)

        # Custom-agent stickiness: keep a natural follow-up on the custom agent
        # that handled the previous turn instead of letting the router silently
        # re-route a terse follow-up to a base agent (which would lose the custom
        # agent's deferred tools/skills and trigger a handoff storm). Skipped
        # when planning supervises; released inside the helper when the user
        # explicitly names a different attached custom agent.
        if not (planning_mode_enabled and has_existing_plan):
            sticky_agent = self._sticky_custom_agent(state, content)
            if sticky_agent:
                state["selected_agent"] = sticky_agent
                self._record_agent_invocation(state, sticky_agent, via="sticky")
                return state

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={
                "persona": state.get("persona"),
                "user_id": state.get("user_id"),
                "device_id": state.get("device_id"),
            },
        )

        available_agents = list(self.agents.keys())

        # Surface attached custom agents to the router (runtime ids as routable
        # targets + descriptors for the prompt / deterministic matching).
        custom_descriptors = self._custom_agent_descriptors(state)
        available_agents.extend(d["runtime_agent_id"] for d in custom_descriptors)

        selected_agent = await self.router.route_message(
            agent_msg,
            available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            custom_agent_descriptors=custom_descriptors or None,
        )

        if selected_agent == "rag_agent" and not has_documents:
            selected_agent = "chat_agent"

        state["selected_agent"] = selected_agent
        self._record_agent_invocation(state, selected_agent, via="router")
        return state

    def _conversation_has_documents(self, conversation_id: str | None) -> bool:
        if not conversation_id or not self.document_repository:
            return False
        try:
            return self.document_repository.count_by_conversation(UUID(conversation_id)) > 0
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
    ) -> str | None:
        """Return the checkpoint thread id, falling back to conversation id when omitted."""
        if thread_id:
            return thread_id
        if conversation_id:
            return conversation_id
        return None

    @staticmethod
    def _set_continuation_signal(
        state: GraphState,
        *,
        should_continue: bool,
        reason: str,
        scope: str,
        count: int,
        limit: int,
    ) -> None:
        context = GraphStateView(state).context_copy()
        context["continuation_signal"] = {
            "should_continue": should_continue,
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
    def _get_requested_continuation_reason(
        cls,
        state_values: dict[str, Any] | None,
    ) -> str | None:
        signal = cls._get_continuation_signal(state_values)
        if not signal.get("should_continue"):
            return None
        reason = signal.get("reason")
        return reason if isinstance(reason, str) else None

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

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat", state=state
        )

        device_id = state.get("device_id")
        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "chat_agent",
            messages,
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        response = await self.chat_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=device_id,
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, "chat_agent"),
        )

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    # ------------------------------------------------------------------
    # Custom-agent multiplexing
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Multi-agent awareness (roster + per-turn invocation trail)
    # ------------------------------------------------------------------
    async def _custom_agent_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        selected_agent = state.get("selected_agent")
        agent = self._build_custom_agent(state, selected_agent)
        if agent is None:
            logger.warning(
                "custom_agent node reached for unattached/unknown id '%s'", selected_agent
            )
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        device_id = state.get("device_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat", state=state
        )
        current_turn_messages = self._messages_for_selected_agent(state, selected_agent, messages)
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        response = await agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=device_id,
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, selected_agent),
        )

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _search_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="search", state=state
        )

        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "search_agent",
            messages,
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        response = await self.search_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, "search_agent"),
        )

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        # NOTE: Image generator deliberately borrows the "chat" history budget.
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat", state=state
        )

        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "image_generator_agent",
            messages,
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        response = await self.image_generator_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, "image_generator_agent"),
        )

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response, append_images=True)
        return self._finalize_agent_response(state, response)

    async def _canvas_node(self, state: GraphState) -> GraphState:
        """Canvas agent node — generates self-contained HTML/SVG/React artifacts."""
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat", state=state
        )

        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "canvas_agent",
            messages,
        )
        current_turn_messages, has_images = self._apply_current_turn_attachments(
            state,
            current_turn_messages,
        )

        response = await self.canvas_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            **self._final_response_kwargs(state),
            **self._multi_agent_kwargs(state, "canvas_agent"),
        )

        self._mark_response_has_images(response, has_images)

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _run_agent_in_isolated_context(
        self,
        *,
        agent_name: str,
        task_prompt: str,
        parent_state: GraphState,
        related_todo_ids: list[str] | None = None,
        model_override: "SubagentModelOverride | None" = None,
        task_id: str | None = None,
    ) -> AgentResponse:
        """Run a single graph-agent against an isolated child state.

        Worker intermediate messages stay local to the child state — they
        are NOT appended to the parent ``messages`` list. The worker
        inherits the parent's scoped identifiers (``conversation_id``,
        ``user_id``, ``device_id``), persona, and model overrides so
        MCP/client tools and per-user model routing keep working.

        ``model_override`` (Phase 10) overlays a task-local model assignment
        onto the worker's ``model_request`` without mutating parent or
        sibling worker routing.
        """
        from .planning_subagents import build_worker_model_request

        if agent_name == "planning_agent":
            raise ValueError(
                "planning_agent is not a valid subagent target — recursive planning is forbidden."
            )

        agent = self.agents.get(agent_name)
        if agent is None and is_custom_runtime_id(agent_name):
            # Custom worker: resolve from the parent conversation's attached
            # custom agents (live config). Rejected if not attached.
            agent = self._build_custom_agent(parent_state, agent_name)
        if agent is None:
            raise ValueError(f"Unknown subagent target: {agent_name}")

        conversation_id = parent_state.get("conversation_id")
        user_id = parent_state.get("user_id")
        device_id = parent_state.get("device_id")
        persona = parent_state.get("persona")
        agent_key = getattr(agent, "agent_config_key", None) or agent_name
        # Deferred tool state is keyed by tool_state_key (custom agents use their
        # runtime id). Keep agent_key for model routing only.
        tool_state_key = getattr(agent, "tool_state_key", None) or agent_key
        model_request = build_worker_model_request(
            parent_model_request=parent_state.get("model_request"),
            agent_key=agent_key,
            override=model_override,
        )
        worker_history_summary: str | None = None
        # ``purpose`` + ``subagent_task_id`` let the v3 stream translator
        # attribute this worker's model deltas to its subagent row instead of
        # leaking them into the main answer/thinking stream.
        run_config = RunnableConfig(
            tags=["internal", "planning_subagent", f"subagent:{agent_name}"],
            metadata={
                "internal": True,
                "purpose": "planning_subagent",
                "subagent": True,
                "subagent_agent": agent_name,
                "subagent_task_id": task_id or agent_name,
            },
        )

        worker_message = HumanMessage(content=task_prompt)
        worker_turn = self._apply_current_turn_attachments(parent_state, [worker_message])
        worker_messages, worker_has_images = worker_turn

        # RAG worker: drive the same search_documents loop used by the graph,
        # but keep all intermediate context local to this worker.
        if agent_name == "rag_agent":
            max_agentic_images = getattr(settings, "agentic_rag_max_images", 6)
            rag_context = dict(parent_state.get("context") or {})
            tool_context: list[str] = []
            accumulated_artifacts: list[dict[str, Any]] = []
            rag_tool_map: dict[str, Any] | None = None
            rag_worker_iterations = 0
            rag_worker_error_signature: dict[str, str] | None = None
            rag_worker_error_count = 0
            rag_worker_error_limit = max(
                1,
                int(getattr(settings, "tool_execution_consecutive_errors_limit", 3) or 3),
            )
            rag_worker_iteration_limit = max(
                1, int(getattr(settings, "agentic_max_iterations", 50) or 50)
            )

            while True:
                rag_worker_iterations += 1
                if rag_worker_iterations > rag_worker_iteration_limit:
                    return AgentResponse(
                        agent_type=getattr(agent, "agent_type", AgentType.RAG),
                        agent_id=agent_name,
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT,
                            content="Worker stopped after reaching its tool iteration limit.",
                        ),
                        metadata={"pause_reason": "worker_max_iterations"},
                        tool_artifacts=list(accumulated_artifacts),
                    )
                agent_msg = AgentMessage(
                    role=MessageRole.USER,
                    content=task_prompt,
                    metadata={
                        "persona": persona,
                        "history": [],
                        "original_query": task_prompt,
                        "tool_context": list(tool_context),
                        "agentic_images": list(rag_context.get("agentic_images") or []),
                        "model_request": model_request,
                        "user_id": user_id,
                        "device_id": device_id,
                        "history_summary": worker_history_summary,
                        "run_config": run_config,
                    },
                    attachments=self._get_state_attachments(parent_state),
                )
                response = await agent.process_message(agent_msg, conversation_id)

                if response.error:
                    if accumulated_artifacts:
                        response.tool_artifacts = accumulated_artifacts
                    return response

                tool_calls = response.message.tool_calls or []
                if not tool_calls:
                    if accumulated_artifacts:
                        existing_artifacts = list(response.tool_artifacts or [])
                        for artifact in accumulated_artifacts:
                            if artifact not in existing_artifacts:
                                existing_artifacts.append(artifact)
                        response.tool_artifacts = existing_artifacts
                    return response

                normalized_calls = [
                    canonicalize_rag_tool_call(normalize_tool_call(tc)) for tc in tool_calls
                ]
                if await self._needs_approval(parent_state, normalized_calls, agent=agent):
                    if response.metadata is None:
                        response.metadata = {}
                    response.metadata["requires_approval"] = True
                    response.metadata["pause_reason"] = "awaiting_approval"
                    if accumulated_artifacts:
                        response.tool_artifacts = accumulated_artifacts
                    return response

                rag_iteration_start = len(accumulated_artifacts)
                for tool_call_data in normalized_calls:
                    tool_name = tool_call_data.get("name")
                    tool_id = tool_call_data.get("id")
                    tool_args = tool_call_data.get("args", {})

                    if tool_name == "search_documents":
                        result, _, evidence = await execute_search_documents_action(
                            rag_agent=self.rag_agent,
                            conversation_id=conversation_id,
                            tool_args=tool_args,
                            context=rag_context,
                            max_agentic_images=max_agentic_images,
                            user_id=user_id,
                        )
                        parsed_error: dict[str, Any] | None = None
                        if isinstance(result, str):
                            with contextlib.suppress(Exception):
                                candidate = json.loads(result)
                                if (
                                    isinstance(candidate, dict)
                                    and candidate.get("status") == "error"
                                ):
                                    parsed_error = candidate
                        error = result if parsed_error or result.startswith("Error") else None
                        public_text, blob_info = apply_tool_output_offload(
                            output_text=result,
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            conversation_id=conversation_id,
                            user_id=user_id,
                        )
                        artifact = build_tool_artifact(
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            output_text=public_text,
                            error=error,
                        )
                        if parsed_error:
                            artifact["error_type"] = parsed_error.get("error_type")
                            artifact["retryable"] = bool(parsed_error.get("retryable"))
                        if blob_info:
                            artifact.update(blob_info)
                        if evidence:
                            artifact["rag_evidence"] = make_json_safe(evidence)
                        accumulated_artifacts.append(artifact)
                        tool_context.append(public_text or "")
                        continue

                    if rag_tool_map is None:
                        rag_tool_map = await ensure_agent_tool_map(
                            agent,
                            conversation_id=conversation_id,
                            user_id=user_id,
                            device_id=device_id,
                        )
                    with tool_execution_context(
                        conversation_id, user_id, tool_state_key, device_id
                    ):
                        outputs, artifacts, _images = await execute_tool_calls(
                            tool_calls=[tool_call_data],
                            tool_map=rag_tool_map,
                            capture_images=False,
                            device_id=device_id,
                            agent=agent,
                            conversation_id=conversation_id,
                            user_id=user_id,
                        )
                    accumulated_artifacts.extend(artifacts)
                    for output in outputs:
                        tool_context.append(output.get("content", ""))

                rag_error_artifacts = [
                    artifact
                    for artifact in accumulated_artifacts[rag_iteration_start:]
                    if isinstance(artifact, dict) and artifact.get("status") == "error"
                ]
                if rag_error_artifacts:
                    signature = self._tool_error_signature(rag_error_artifacts[0])
                    if signature == rag_worker_error_signature:
                        rag_worker_error_count += 1
                    else:
                        rag_worker_error_signature = signature
                        rag_worker_error_count = 1
                    if rag_worker_error_count >= rag_worker_error_limit:
                        return AgentResponse(
                            agent_type=getattr(agent, "agent_type", AgentType.RAG),
                            agent_id=agent_name,
                            message=AgentMessage(
                                role=MessageRole.ASSISTANT,
                                content=(
                                    "Worker stopped after repeated tool errors. Use the "
                                    "available tool results to explain the blocker."
                                ),
                            ),
                            metadata={
                                "pause_reason": "consecutive_tool_errors",
                                "tool_error_streak": {
                                    "count": rag_worker_error_count,
                                    "limit": rag_worker_error_limit,
                                    "signature": signature,
                                },
                            },
                            tool_artifacts=list(accumulated_artifacts),
                        )
                else:
                    rag_worker_error_signature = None
                    rag_worker_error_count = 0

        # Generic agent worker: tool-loop until final response, approval, or error.

        tool_map: dict[str, Any] | None = None
        accumulated_worker_artifacts: list[dict[str, Any]] = []
        worker_iterations = 0
        worker_error_signature: dict[str, str] | None = None
        worker_error_count = 0
        worker_error_limit = max(
            1,
            int(getattr(settings, "tool_execution_consecutive_errors_limit", 3) or 3),
        )
        worker_iteration_limit = max(
            1, int(getattr(settings, "react_agent_max_iterations", 50) or 50)
        )

        while True:
            worker_iterations += 1
            if worker_iterations > worker_iteration_limit:
                return AgentResponse(
                    agent_type=getattr(agent, "agent_type", AgentType.CHAT),
                    agent_id=agent_name,
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content="Worker stopped after reaching its tool iteration limit.",
                    ),
                    metadata={"pause_reason": "worker_max_iterations"},
                    tool_artifacts=list(accumulated_worker_artifacts),
                )
            response = await agent.invoke_model_with_history(
                messages=list(worker_messages),
                conversation_history=[],
                persona=persona,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
                model_request=model_request,
                history_summary=worker_history_summary,
                run_config=run_config,
                # Workers run isolated; graph-level hand_off cannot apply here, so
                # bind it off to keep the worker from wasting tokens on no-op
                # delegation calls.
                include_hand_off=False,
            )

            if response.error:
                return response

            tool_calls = response.message.tool_calls or []
            if not tool_calls:
                if accumulated_worker_artifacts:
                    existing = list(response.tool_artifacts or [])
                    existing.extend(accumulated_worker_artifacts)
                    response.tool_artifacts = existing
                self._mark_response_has_images(response, worker_has_images)
                return response

            normalized_worker_calls = [normalize_tool_call(tc) for tc in tool_calls]
            # Build the worker tool_map once (reused across loop iterations so a tool
            # loaded via tool_search stays available) and feed it to the approval gate
            # so provenance resolves without a second ensure_agent_tool_map call.
            if tool_map is None:
                tool_map = await ensure_agent_tool_map(
                    agent,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                )
            if await self._needs_approval(
                parent_state, normalized_worker_calls, tool_map=tool_map
            ):
                if response.metadata is None:
                    response.metadata = {}
                response.metadata["requires_approval"] = True
                response.metadata["pause_reason"] = "awaiting_approval"
                if accumulated_worker_artifacts:
                    existing = list(response.tool_artifacts or [])
                    existing.extend(accumulated_worker_artifacts)
                    response.tool_artifacts = existing
                return response

            ai_kwargs: dict[str, Any] = {"content": response.message.content or ""}
            if tool_calls:
                ai_kwargs["tool_calls"] = tool_calls
            worker_messages.append(AIMessage(**ai_kwargs))

            with tool_execution_context(conversation_id, user_id, tool_state_key, device_id):
                outputs, artifacts, _images = await execute_tool_calls(
                    tool_calls=tool_calls,
                    tool_map=tool_map,
                    capture_images=False,
                    device_id=device_id,
                    agent=agent,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

            for output in outputs:
                worker_messages.append(
                    ToolMessage(
                        content=output.get("content", ""),
                        tool_call_id=output.get("tool_call_id"),
                        name=output.get("name") or "tool",
                    )
                )

            accumulated_worker_artifacts.extend(artifacts)
            response.tool_artifacts = list(accumulated_worker_artifacts)

            error_artifacts = [
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict) and artifact.get("status") == "error"
            ]
            if error_artifacts:
                signature = self._tool_error_signature(error_artifacts[0])
                if signature == worker_error_signature:
                    worker_error_count += 1
                else:
                    worker_error_signature = signature
                    worker_error_count = 1
                if worker_error_count >= worker_error_limit:
                    return AgentResponse(
                        agent_type=getattr(agent, "agent_type", AgentType.CHAT),
                        agent_id=agent_name,
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT,
                            content=(
                                "Worker stopped after repeated tool errors. Use the available "
                                "tool results to explain the blocker."
                            ),
                        ),
                        metadata={
                            "pause_reason": "consecutive_tool_errors",
                            "tool_error_streak": {
                                "count": worker_error_count,
                                "limit": worker_error_limit,
                                "signature": signature,
                            },
                        },
                        tool_artifacts=list(accumulated_worker_artifacts),
                    )
            else:
                worker_error_signature = None
                worker_error_count = 0

    @staticmethod
    def _latest_user_text(state: GraphState) -> str:
        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                return str(message.content or "")
        return ""

    def _should_continue(self, state: GraphState) -> str:
        selected_agent = state.get("selected_agent")
        if selected_agent in self.agents:
            return selected_agent
        # Any attached custom runtime id routes to the single static node.
        if is_custom_runtime_id(selected_agent) and self._is_attached_custom_agent(
            state, selected_agent
        ):
            return "custom_agent"
        return "end"

    def _get_agent_type(self, selected_agent: str | None) -> AgentType:
        agent_type_map = {
            "chat_agent": AgentType.CHAT,
            "rag_agent": AgentType.RAG,
            "search_agent": AgentType.SEARCH,
            "image_generator_agent": AgentType.IMAGE_GENERATOR,
            "planning_agent": AgentType.PLANNING,
            "canvas_agent": AgentType.CANVAS,
        }
        return agent_type_map.get(selected_agent, AgentType.CHAT)

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

    def _recover_terminal_response(
        self,
        state: dict[str, Any] | None,
        *,
        fallback_content: str | None = None,
        selected_agent: str | None = None,
    ) -> AgentResponse | None:
        if not isinstance(state, dict):
            return None

        response = state.get("response")
        fallback_text = coerce_response_text(fallback_content)

        if response:
            response_message = getattr(response, "message", None)
            if getattr(response_message, "tool_calls", None):
                response = None
            else:
                response_content = coerce_response_text(getattr(response_message, "content", None))
                if fallback_text and response.message and not response_content:
                    response.message.content = fallback_text
                return self._attach_context_outputs(state, response)

        content = fallback_text
        if not content:
            messages = state.get("messages", [])
            for message in reversed(messages):
                if not isinstance(message, AIMessage):
                    continue
                if getattr(message, "tool_calls", None):
                    continue
                content = coerce_response_text(getattr(message, "content", None))
                if content:
                    break

        if not content:
            return None

        final_selected_agent = state.get("selected_agent") or selected_agent
        recovered_response = AgentResponse(
            agent_type=self._get_agent_type(final_selected_agent),
            agent_id=final_selected_agent or "unknown",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=content,
            ),
            metadata={},
        )
        return self._attach_context_outputs(state, recovered_response)

    # ------------------------------------------------------------------
    # Continuation helpers
    # ------------------------------------------------------------------

    def _build_continuation_state(
        self,
        previous_state: dict[str, Any],
        round_num: int,
        reason: str,
    ) -> GraphState:
        """Build a new initial state that carries forward context from a previous round.

        Keeps routing stable (selected_agent preserved), resets per-round
        counters so each round has a fresh budget, and clears "stop" flags
        that would immediately short-circuit the next round.
        """
        state: GraphState = dict(previous_state or {})
        state_view = GraphStateView(state)

        # Reset per-round counters so the next round has fresh budget.
        state["iteration_count"] = 0
        state["planning_call_count"] = 0

        # Clear response artifacts from previous round (only the final
        # round's response is used).
        state.pop("response", None)

        # Clear "stop" flags so they don't immediately short-circuit the
        # next round.
        ctx = state_view.context_copy()
        ctx.pop("pause_reason", None)
        ctx.pop("continuation_signal", None)

        # NOTE (P2 regression-check): `conversation_summarized` is deliberately
        # *not* popped here so the summarize node skips re-summarization in
        # subsequent auto-continue rounds within the same user turn.
        # If this key were cleared, each continuation round would re-trigger
        # summarization and double-remove messages already covered.

        # Add lightweight trace context (helpful for logs/prompts).
        ctx["continuation_round"] = round_num
        ctx["continuation_reason"] = reason
        state["context"] = ctx

        return state

    async def _capture_state_for_continuation(
        self,
        config: dict[str, Any] | None,
        thread_id: str | None,
        fallback_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture current graph state for continuation.

        When using checkpointer, reads the latest checkpoint snapshot.
        Otherwise, falls back to the in-memory state accumulated from
        the streaming loop's "updates" events.
        """
        if self.checkpointer and thread_id:
            try:
                snapshot = await self.graph.aget_state(config)
                if snapshot and hasattr(snapshot, "values") and snapshot.values:
                    return dict(snapshot.values)
            except Exception as exc:
                logger.warning("Failed to capture checkpoint state for continuation: %s", exc)
        return dict(fallback_state) if fallback_state else {}

    async def execute_request(self, request: WorkflowExecutionRequest) -> AgentResponse | None:
        initial_state = self._build_initial_state_from_request(request)
        conversation_id = request.conversation_id
        thread_id = self._resolve_thread_id(request.thread_id, conversation_id)
        config = self._build_graph_config(thread_id)

        # ── Auto-Continue outer loop ──────────────────────────────────
        max_rounds = settings.auto_continue_max_rounds if settings.auto_continue_enabled else 1
        total_iterations = 0
        start_time = time.monotonic()
        current_state = initial_state
        result: dict[str, Any] | None = None

        for round_num in range(1, max_rounds + 1):
            # Safety: wall-clock timeout
            if (
                round_num > 1
                and time.monotonic() - start_time > settings.auto_continue_timeout_seconds
            ):
                logger.warning(
                    "Auto-continue timeout reached after %d rounds (execute)",
                    round_num - 1,
                )
                break

            should_continue = False
            continue_reason: str | None = None

            try:
                result = await self.graph.ainvoke(current_state, config=config)
            except GraphRecursionError:
                logger.warning(
                    "GraphRecursionError caught in round %d (execute) — will attempt continuation",
                    round_num,
                )
                should_continue = True
                continue_reason = "recursion_limit"
                result = None

            # Detect soft-budget continuation signal from the result
            if not should_continue and result:
                continue_reason = self._get_requested_continuation_reason(result)
                if continue_reason:
                    should_continue = True

            if not should_continue:
                break  # Normal completion

            # Safety: total iteration cap
            captured = await self._capture_state_for_continuation(
                config=config,
                thread_id=thread_id,
                fallback_state=result,
            )
            round_iterations = int(
                (captured.get("iteration_count") or 0) + (captured.get("planning_call_count") or 0)
            )
            total_iterations += round_iterations
            if total_iterations >= settings.auto_continue_max_total_iterations:
                logger.warning(
                    "Auto-continue total iteration cap reached: %d (execute)",
                    total_iterations,
                )
                break

            if round_num >= max_rounds:
                logger.info("Auto-continue max rounds (%d) reached (execute)", max_rounds)
                break

            # Prepare state for next round
            current_state = self._build_continuation_state(
                previous_state=captured,
                round_num=round_num + 1,
                reason=continue_reason or "soft_budget",
            )
            logger.info(
                "Auto-continue (execute): starting round %d (reason=%s, total_iters=%d)",
                round_num + 1,
                continue_reason,
                total_iterations,
            )

        # ── Post-loop: finalize response ──────────────────────────────
        if self.checkpointer and thread_id:
            state_snapshot = await self.graph.aget_state(config)
            if state_snapshot.next and len(state_snapshot.next) > 0:
                interrupt_agent_response = self._build_interrupt_agent_response(
                    state_snapshot, thread_id, conversation_id
                )
                if interrupt_agent_response:
                    return interrupt_agent_response

        if result is None:
            # GraphRecursionError on last round with no usable result
            result = await self._capture_state_for_continuation(
                config=config, thread_id=thread_id, fallback_state=None
            )

        final_state = result if isinstance(result, dict) else None
        agent_response = self._recover_terminal_response(final_state)
        if not agent_response and self.checkpointer and thread_id:
            final_snapshot = await self.graph.aget_state(config)
            final_state = (
                final_snapshot.values
                if final_snapshot and hasattr(final_snapshot, "values")
                else None
            )
            agent_response = self._recover_terminal_response(final_state)

        final_selected_agent = (
            final_state.get("selected_agent") if isinstance(final_state, dict) else None
        )
        if agent_response and final_selected_agent == "planning_agent":
            agent_response = self._attach_planning_state_metadata(agent_response, final_state)

        # Add continuation metadata when multiple rounds ran
        if agent_response and total_iterations > 0:
            if agent_response.metadata is None:
                agent_response.metadata = {}
            agent_response.metadata["continuation_rounds"] = round_num
            agent_response.metadata["total_iterations"] = total_iterations

        if (
            agent_response
            and isinstance(agent_response.metadata, dict)
            and "interrupt" in agent_response.metadata
        ):
            return agent_response

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

        result = await self.graph.ainvoke(Command(resume=resume_data), config=config)

        # Check for further interrupts
        final_snapshot = await self.graph.aget_state(config)
        if _has_approval_interrupt(final_snapshot.next):
            interrupt_response = self._build_interrupt_agent_response(
                final_snapshot,
                thread_id,
                final_snapshot.values.get("conversation_id"),
            )
            if interrupt_response:
                return interrupt_response

        response = self._recover_terminal_response(result)
        if not response:
            final_state = (await self.graph.aget_state(config)).values
            response = self._recover_terminal_response(final_state)

        return response

    async def resume_with_decisions_stream(
        self,
        thread_id: str,
        decisions: list[InterruptDecision],
    ):
        if not self.checkpointer:
            yield {
                "type": "error",
                "error": "Checkpointing is not enabled, cannot resume.",
            }
            return

        config = self._build_graph_config(thread_id)
        state_snapshot = await self.graph.aget_state(config)

        if not state_snapshot.next or len(state_snapshot.next) == 0:
            raise ValueError("Workflow is not in interrupted state")
        if not _has_approval_interrupt(state_snapshot.next):
            raise ValueError(f"Unexpected interrupt state: next nodes are {state_snapshot.next}")

        resume_data = build_interrupt_resume_payload(decisions)

        selected_agent = state_snapshot.values.get("selected_agent", "search_agent")
        conversation_id = state_snapshot.values.get("conversation_id")

        yield {"type": "agent_selected", "agent": selected_agent}

        suppressed_nodes: set = {"image_generator_agent"}
        suppress_tokens = selected_agent in suppressed_nodes
        projector = GraphPublicStreamProjector(
            tool_end_events_from_node_state=self._tool_end_events_from_node_state,
            suppress_internal_stream_chunks=settings.suppress_internal_stream_chunks,
        )
        ctx = StreamProjectionContext(
            last_emitted_agent=selected_agent,
            suppress_tokens=suppress_tokens,
        )

        max_rounds = settings.auto_continue_max_rounds if settings.auto_continue_enabled else 1
        round_num = 1
        continue_reason: str | None = "initial"
        total_iterations = 0
        start_time = time.monotonic()
        current_state: Any = Command(resume=resume_data)

        while round_num <= max_rounds:
            if (
                round_num > 1
                and time.monotonic() - start_time > settings.auto_continue_timeout_seconds
            ):
                logger.warning("Auto-continue timeout reached after %d rounds", round_num - 1)
                break

            if round_num > 1 and settings.auto_continue_emit_events:
                yield {
                    "type": "continuation_start",
                    "round": round_num,
                    "max_rounds": max_rounds,
                    "reason": continue_reason,
                }

            should_continue = False
            continue_reason = None

            try:
                async for event in iter_v3_events_from_graph(
                    self.graph, current_state, config=config
                ):
                    for public_event in projector.map_event(event, ctx):
                        yield public_event

            except GraphRecursionError:
                logger.warning(
                    "GraphRecursionError caught in round %d — will attempt continuation",
                    round_num,
                )
                should_continue = True
                continue_reason = "recursion_limit"

            except Exception as e:
                yield {"type": "error", "error": str(e)}
                return

            if not should_continue and ctx.last_state_values:
                continue_reason = self._get_requested_continuation_reason(ctx.last_state_values)
                if continue_reason:
                    should_continue = True

            if not should_continue:
                break

            captured = await self._capture_state_for_continuation(
                config=config,
                thread_id=thread_id,
                fallback_state=ctx.last_state_values,
            )
            round_iterations = int(
                (captured.get("iteration_count") or 0) + (captured.get("planning_call_count") or 0)
            )
            total_iterations += round_iterations
            if total_iterations >= settings.auto_continue_max_total_iterations:
                logger.warning(
                    "Auto-continue total iteration cap reached: %d",
                    total_iterations,
                )
                break

            if round_num >= max_rounds:
                logger.info(
                    "Auto-continue max rounds (%d) reached — returning partial result",
                    max_rounds,
                )
                break

            current_state = self._build_continuation_state(
                previous_state=captured,
                round_num=round_num + 1,
                reason=continue_reason or "soft_budget",
            )
            logger.info(
                "Auto-continue: starting round %d (reason=%s, total_iters=%d)",
                round_num + 1,
                continue_reason,
                total_iterations,
            )
            round_num += 1
            ctx.current_tool_calls = {}

        accumulated_content = ctx.accumulated_content
        accumulated_thinking = ctx.accumulated_thinking
        _internal_content_only = ctx.internal_content_only
        last_state_values = ctx.last_state_values

        try:
            snapshot = await self.graph.aget_state(config)

            if snapshot.next and len(snapshot.next) > 0:
                messages = snapshot.values.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if (
                        isinstance(last_msg, AIMessage)
                        and hasattr(last_msg, "tool_calls")
                        and last_msg.tool_calls
                    ):
                        interrupt_payload = self._interrupt_payload_from_pending_interrupts(
                            snapshot
                        ) or self._get_interrupt_payload_from_state(
                            snapshot.values,
                            last_msg.tool_calls,
                        )
                        pending_tool_calls = interrupt_payload["action_requests"]
                        interrupt_response = build_interrupt_response(
                            interrupt_payload,
                            thread_id,
                            conversation_id or "",
                        )
                        yield {
                            "type": "interrupt",
                            "next": snapshot.next,
                            "thread_id": thread_id,
                            "pending_tool_calls": pending_tool_calls,
                            "interrupt": interrupt_response,
                        }
                        return

            final_state = snapshot.values if snapshot and hasattr(snapshot, "values") else {}
            fallback_content = (
                accumulated_content if not suppress_tokens and not _internal_content_only else None
            )
            response = self._recover_terminal_response(
                final_state,
                fallback_content=fallback_content,
                selected_agent=selected_agent,
            )
            if not response:
                response = self._recover_terminal_response(
                    last_state_values,
                    fallback_content=fallback_content,
                    selected_agent=selected_agent,
                )
            if response:
                if (
                    accumulated_thinking
                    and not _internal_content_only
                    and not response.metadata.get("thinking_summary")
                ):
                    response.metadata["thinking_summary"] = accumulated_thinking

                response_state = final_state if final_state else last_state_values
                final_selected_agent = (
                    response_state.get("selected_agent")
                    if isinstance(response_state, dict)
                    else selected_agent
                ) or selected_agent
                if final_selected_agent == "planning_agent":
                    response = self._attach_planning_state_metadata(response, response_state)

                if round_num > 1:
                    response.metadata["continuation_rounds"] = round_num
                    response.metadata["total_iterations"] = total_iterations

                yield {"type": "complete", "response": response}
            else:
                yield {"type": "error", "error": NO_RESPONSE_GENERATED}
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    async def execute_request_stream(self, request: WorkflowExecutionRequest):
        initial_state = self._build_initial_state_from_request(request)
        # Custom subagents (dispatch_subagents) emit lifecycle events into this
        # sink; it is drained between graph supersteps below. Only a weakref
        # token enters graph state (state must stay msgpack-serializable for
        # checkpointing); _build_planning_internal_tools resolves it back.
        subagent_event_sink = SubagentEventSink()
        if isinstance(initial_state, dict):
            context = initial_state.setdefault("context", {})
            if isinstance(context, dict):
                context["subagent_event_sink_token"] = register_subagent_event_sink(
                    subagent_event_sink
                )
        conversation_id = request.conversation_id
        thread_id = self._resolve_thread_id(request.thread_id, conversation_id)
        config = self._build_graph_config(thread_id)
        user_id = request.user_id

        # Prefetch conversation history in parallel with the router LLM call.
        # By the time the agent node needs history, the cache will be warm.
        history_prefetch: asyncio.Task | None = None
        if conversation_id and user_id:
            history_prefetch = asyncio.create_task(
                self._get_conversation_history(conversation_id, user_id)
            )

        try:
            routed_state = await self._route_node(initial_state)
            selected_agent = routed_state.get("selected_agent")
            initial_state["selected_agent"] = selected_agent
        except Exception as e:
            if history_prefetch and not history_prefetch.done():
                history_prefetch.cancel()
            yield {"type": "error", "error": str(e)}
            return

        yield {"type": "agent_selected", "agent": selected_agent}

        # For image_generator_agent the streamed LLM tokens are the internal
        # enhanced prompt — not meant for the user.  Suppress token events and
        # let the final "complete" response (which holds the user-facing text)
        # be the only content the client sees.
        # Extend this set with any future agent whose raw tokens are internal.
        suppressed_nodes: set = {"image_generator_agent"}
        suppress_tokens = selected_agent in suppressed_nodes

        # Per-stream accumulator state shared across continuation rounds and the
        # canonical event mapper.
        projector = GraphPublicStreamProjector(
            tool_end_events_from_node_state=self._tool_end_events_from_node_state,
            suppress_internal_stream_chunks=settings.suppress_internal_stream_chunks,
        )
        ctx = StreamProjectionContext(
            last_emitted_agent=selected_agent,
            suppress_tokens=suppress_tokens,
        )

        # ── Auto-Continue outer loop ──────────────────────────────────
        max_rounds = settings.auto_continue_max_rounds if settings.auto_continue_enabled else 1
        round_num = 1
        continue_reason: str | None = "initial"
        total_iterations = 0
        start_time = time.monotonic()
        current_state = initial_state

        while round_num <= max_rounds:
            # Safety: wall-clock timeout across all rounds
            if (
                round_num > 1
                and time.monotonic() - start_time > settings.auto_continue_timeout_seconds
            ):
                logger.warning("Auto-continue timeout reached after %d rounds", round_num - 1)
                break

            # Emit continuation_start event for rounds > 1
            if round_num > 1 and settings.auto_continue_emit_events:
                yield {
                    "type": "continuation_start",
                    "round": round_num,
                    "max_rounds": max_rounds,
                    "reason": continue_reason,
                }

            should_continue = False
            continue_reason = None

            try:
                merged = stream_with_subagent_events(
                    iter_v3_events_from_graph(self.graph, current_state, config=config),
                    subagent_event_sink,
                )
                async for event in merged:
                    for public_event in projector.map_event(event, ctx):
                        yield public_event

            except GraphRecursionError:
                logger.warning(
                    "GraphRecursionError caught in round %d — will attempt continuation",
                    round_num,
                )
                should_continue = True
                continue_reason = "recursion_limit"

            except Exception as e:
                yield {"type": "error", "error": str(e)}
                return

            # ── Check if graph ended because we *want* to continue ─────
            if not should_continue and ctx.last_state_values:
                continue_reason = self._get_requested_continuation_reason(ctx.last_state_values)
                if continue_reason:
                    should_continue = True

            if not should_continue:
                break  # Normal completion — exit loop and finalize

            # ── Safety: total iteration cap ────────────────────────────
            captured = await self._capture_state_for_continuation(
                config=config,
                thread_id=thread_id,
                fallback_state=ctx.last_state_values,
            )
            round_iterations = int(
                (captured.get("iteration_count") or 0) + (captured.get("planning_call_count") or 0)
            )
            total_iterations += round_iterations
            if total_iterations >= settings.auto_continue_max_total_iterations:
                logger.warning(
                    "Auto-continue total iteration cap reached: %d",
                    total_iterations,
                )
                break

            if round_num >= max_rounds:
                logger.info(
                    "Auto-continue max rounds (%d) reached — returning partial result",
                    max_rounds,
                )
                break

            # ── Prepare state for next round ───────────────────────────
            current_state = self._build_continuation_state(
                previous_state=captured,
                round_num=round_num + 1,
                reason=continue_reason or "soft_budget",
            )
            logger.info(
                "Auto-continue: starting round %d (reason=%s, total_iters=%d)",
                round_num + 1,
                continue_reason,
                total_iterations,
            )
            round_num += 1
            # Reset per-round tool call tracking (accumulators persist)
            ctx.current_tool_calls = {}

        accumulated_content = ctx.accumulated_content
        accumulated_thinking = ctx.accumulated_thinking
        _internal_content_only = ctx.internal_content_only
        last_state_values = ctx.last_state_values

        # ── Post-stream: finalize response ─────────────────────────────
        if self.checkpointer and thread_id:
            try:
                snapshot = await self.graph.aget_state(config)

                if snapshot.next and len(snapshot.next) > 0:
                    messages = snapshot.values.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if (
                            isinstance(last_msg, AIMessage)
                            and hasattr(last_msg, "tool_calls")
                            and last_msg.tool_calls
                        ):
                            interrupt_payload = self._interrupt_payload_from_pending_interrupts(
                                snapshot
                            ) or self._get_interrupt_payload_from_state(
                                snapshot.values,
                                last_msg.tool_calls,
                            )
                            pending_tool_calls = interrupt_payload["action_requests"]
                            interrupt_response = build_interrupt_response(
                                interrupt_payload,
                                thread_id,
                                conversation_id or "",
                            )
                            yield {
                                "type": "interrupt",
                                "next": snapshot.next,
                                "thread_id": thread_id,
                                "pending_tool_calls": pending_tool_calls,
                                "interrupt": interrupt_response,
                            }
                            return

                final_state = snapshot.values if snapshot and hasattr(snapshot, "values") else {}
                fallback_content = (
                    accumulated_content
                    if not suppress_tokens and not _internal_content_only
                    else None
                )
                response = self._recover_terminal_response(
                    final_state,
                    fallback_content=fallback_content,
                    selected_agent=selected_agent,
                )
                if not response:
                    response = self._recover_terminal_response(
                        last_state_values,
                        fallback_content=fallback_content,
                        selected_agent=selected_agent,
                    )
                if response:
                    if (
                        accumulated_thinking
                        and not _internal_content_only
                        and not response.metadata.get("thinking_summary")
                    ):
                        response.metadata["thinking_summary"] = accumulated_thinking

                    response_state = final_state if final_state else last_state_values
                    final_selected_agent = (
                        response_state.get("selected_agent")
                        if isinstance(response_state, dict)
                        else selected_agent
                    ) or selected_agent
                    if final_selected_agent == "planning_agent":
                        response = self._attach_planning_state_metadata(response, response_state)

                    # Add continuation metadata when multiple rounds ran
                    if round_num > 1:
                        response.metadata["continuation_rounds"] = round_num
                        response.metadata["total_iterations"] = total_iterations

                    yield {"type": "complete", "response": response}
                else:
                    yield {"type": "error", "error": NO_RESPONSE_GENERATED}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
        else:
            fallback_content = (
                accumulated_content if not suppress_tokens and not _internal_content_only else None
            )
            response = self._recover_terminal_response(
                last_state_values,
                fallback_content=fallback_content,
                selected_agent=selected_agent,
            )
            if response:
                if (
                    accumulated_thinking
                    and not _internal_content_only
                    and not response.metadata.get("thinking_summary")
                ):
                    response.metadata["thinking_summary"] = accumulated_thinking

                response_state = last_state_values
                final_selected_agent = (
                    response_state.get("selected_agent")
                    if isinstance(response_state, dict)
                    else selected_agent
                ) or selected_agent
                if final_selected_agent == "planning_agent":
                    response = self._attach_planning_state_metadata(response, response_state)

                if round_num > 1:
                    response.metadata["continuation_rounds"] = round_num
                    response.metadata["total_iterations"] = total_iterations

                yield {"type": "complete", "response": response}
            else:
                yield {"type": "error", "error": NO_RESPONSE_GENERATED}

    async def get_state(self, thread_id: str) -> dict:
        if not self.checkpointer:
            raise ValueError("Checkpointing is not enabled, cannot get state.")

        config = self._build_graph_config(thread_id)
        snapshot = await self.graph.aget_state(config)

        pending_tool_calls = None
        messages = snapshot.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            if (
                isinstance(last_message, AIMessage)
                and hasattr(last_message, "tool_calls")
                and last_message.tool_calls
            ):
                pending_tool_calls = last_message.tool_calls

        return {
            "interrupted": bool(snapshot.next),
            "next": snapshot.next,
            "values": snapshot.values,
            "pending_tool_calls": pending_tool_calls,
            "selected_agent": snapshot.values.get("selected_agent"),
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
    )
