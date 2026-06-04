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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import RemoveMessage
from langgraph.types import Command, interrupt
from langsmith import tracing_context
from qdrant_client import QdrantClient

from ..core.config import settings
from ..core.response_constants import NO_RESPONSE_GENERATED
from ..interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..interfaces.workflow_runtime_interface import IWorkflowRuntime
from ..models.enums import PlanLifecycle
from .agent_metadata import (
    agent_identity,
    attach_agent_metadata,
    base_agent_capability,
    normalize_handoff_metadata,
    normalize_subagent_metadata,
)
from .agents.canvas_agent import CanvasAgent
from .agents.chat_agent import ChatAgent
from .agents.custom_agent import CustomAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .agents.planning_agent import PlanningAgent
from .agents.rag_agent import RAGAgent
from .agents.router import Router
from .agents.search_agent import SearchAgent
from .custom_agent_runtime import build_custom_agent_runtime_spec, is_custom_runtime_id
from .hand_off_tool import MAX_DELEGATION_DEPTH, create_hand_off_tool
from .history import ConversationHistoryProvider
from .hitl_config import build_interrupt_response, requires_human_approval
from .memory import get_memory_manager
from .rag_tool_actions import execute_search_documents_action
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
from .todo_actions import apply_write_todos_action
from .token_instrumentation import (
    HistoryBudgetConfig,
    trim_history_to_budget,
    truncate_tool_result,
)
from .tool_context import tool_execution_context
from .tool_execution import (
    apply_tool_output_offload,
    build_rejected_tool_artifacts,
    build_tool_artifact,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from .utils import (
    apply_hitl_decisions,
    build_interrupt_resume_payload,
    coerce_response_text,
    find_pending_tool_call_message,
    make_json_safe,
    normalize_tool_call,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository
    from .planning_subagents import SubagentModelOverride

_apply_decisions = apply_hitl_decisions


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


class MultiAgentWorkflow(IWorkflowRuntime):
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

    @staticmethod
    def _is_internal_stream_chunk(metadata: Any) -> bool:
        """Return True if this stream chunk originates from an internal (non-user-facing) LLM run.

        Checks, in priority order:
        1. The `tags` list for the 'internal' tag (set via RunnableConfig in generate_summary).
        2. The nested `metadata` dict's `internal` key (also set via RunnableConfig).
        3. The `langgraph_node` key as a hard-coded fallback for the summarize node.
        """
        if not isinstance(metadata, dict):
            return False

        tags = metadata.get("tags") or []
        if "internal" in tags:
            return True

        nested_metadata = metadata.get("metadata") or {}
        if isinstance(nested_metadata, dict) and nested_metadata.get("internal") is True:
            return True

        # Hard-coded fallback: always suppress the summarize node regardless of tags.
        return metadata.get("langgraph_node") == "summarize"

    @staticmethod
    def _consume_stream_text_chunk(
        accumulated_content: str, text_chunk: Any
    ) -> tuple[str, str | None]:
        """
        Return (new_accumulated_content, delta_to_emit) for a streaming text chunk.

        Handles both:
        - Cumulative chunks (Gemini): Each chunk contains all text accumulated so far
        - Incremental chunks (OpenAI): Each chunk contains only new text
        """
        if not text_chunk:
            return accumulated_content, None

        chunk_text = coerce_response_text(text_chunk)
        if not chunk_text:
            return accumulated_content, None

        if accumulated_content:
            # Exact repeat of what we've already accumulated - skip it
            if chunk_text == accumulated_content:
                return accumulated_content, None

            # Cumulative chunk: new chunk starts with what we already have (Gemini)
            if chunk_text.startswith(accumulated_content):
                delta = chunk_text[len(accumulated_content) :]
                return chunk_text, delta or None

            # Duplicate tail chunk - skip it
            if accumulated_content.endswith(chunk_text):
                return accumulated_content, None

        # Incremental chunk (OpenAI) or first chunk - append and emit
        return accumulated_content + chunk_text, chunk_text

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
            },
        }

        if request.conversation_id is not None:
            initial_state["conversation_id"] = request.conversation_id
        if request.user_id is not None:
            initial_state["user_id"] = request.user_id
        if request.device_id is not None:
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

        return initial_state

    @staticmethod
    def _get_state_attachments(state: GraphState) -> list[Any]:
        return GraphStateView(state).attachments()

    @staticmethod
    def _normalize_attachment_image_url(attachment: Any) -> str | None:
        if not isinstance(attachment, dict):
            return None

        mime = (
            attachment.get("mime")
            or attachment.get("mimeType")
            or attachment.get("mediaType")
            or attachment.get("contentType")
            or "image/jpeg"
        )
        mime = str(mime).strip() if mime else "image/jpeg"

        candidate_values = [
            attachment.get("data"),
            attachment.get("url"),
            attachment.get("path"),
            attachment.get("image"),
            attachment.get("source"),
        ]

        for candidate in candidate_values:
            if isinstance(candidate, dict):
                candidate = (
                    candidate.get("url")
                    or candidate.get("data")
                    or candidate.get("base64")
                    or candidate.get("path")
                )
            if not isinstance(candidate, str):
                continue

            raw_value = candidate.strip()
            if not raw_value:
                continue

            if raw_value.startswith("data:"):
                return raw_value

            if raw_value.startswith(("http://", "https://", "blob:")):
                return raw_value

            # Guard against accidentally treating local file-system paths as base64.
            if ":\\" in raw_value or raw_value.startswith(("/", "./", "../")):
                continue

            return f"data:{mime};base64,{raw_value}"

        return None

    def _build_chat_turn_messages_with_attachments(
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

        original_content = messages_copy[last_human_idx].content
        user_text = coerce_response_text(original_content)

        multimodal_parts: list[dict[str, Any]] = []
        if user_text:
            multimodal_parts.append({"type": "text", "text": user_text})

        for attachment in attachments:
            image_url = self._normalize_attachment_image_url(attachment)
            if not image_url:
                continue
            multimodal_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )

        has_images = any(part.get("type") == "image_url" for part in multimodal_parts)
        if has_images:
            messages_copy[last_human_idx] = HumanMessage(content=multimodal_parts)

        return messages_copy, has_images

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
        workflow = StateGraph(GraphState)

        # Memory refactor 2026-04-29: durable summary refresh now happens
        # after assistant persistence, not on the request hot path.
        # ``_summarization_node`` is kept as a no-op fallback so any in-flight
        # checkpoints that previously routed through "summarize" still resolve.
        workflow.add_node("summarize", self._summarization_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)
        workflow.add_node("planning_agent", self._planning_node)
        workflow.add_node("canvas_agent", self._canvas_node)
        # Single static node that multiplexes every runtime custom-agent id
        # (custom_agent:<uuid>). The graph is never rebuilt per conversation.
        workflow.add_node("custom_agent", self._custom_agent_node)
        workflow.add_node("planning_tools", self._planning_tools_node)
        workflow.add_node("rag_tools", self._rag_tools_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", self._tool_node)

        # START -> route directly. Long-term summarization no longer runs on
        # the streaming hot path — it is refreshed after the assistant turn
        # is persisted (see MessageService).
        workflow.add_edge(START, "route")

        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "chat_agent": "chat_agent",
                "rag_agent": "rag_agent",
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "planning_agent": "planning_agent",
                "canvas_agent": "canvas_agent",
                "custom_agent": "custom_agent",
                "end": END,
            },
        )

        # Consolidate conditional edges for agents that use standard tool calling
        tool_calling_agents = [
            "chat_agent",
            "search_agent",
            "image_generator_agent",
            "canvas_agent",
            "custom_agent",
        ]
        for agent_name in tool_calling_agents:
            workflow.add_conditional_edges(
                agent_name,
                self._should_call_tools,
                {
                    "approval": "approval",
                    "tools": "tools",
                    "end": END,
                },
            )

        # RAG agent: direct to END if not agentic, or rag_tools loop if agentic
        workflow.add_conditional_edges(
            "rag_agent",
            self._should_call_rag_tools,
            {
                "rag_tools": "rag_tools",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "rag_tools",
            self._should_continue_rag,
            {
                "rag_agent": "rag_agent",
                "end": END,
            },
        )

        # Planning agent ReAct loop: planning_agent → planning_tools → planning_agent OR end
        workflow.add_conditional_edges(
            "planning_agent",
            self._should_call_planning_tools,
            {
                "planning_tools": "planning_tools",
                "end": END,
            },
        )

        # planning_tools fans out either back into the planning loop, ends the
        # turn, or — when the Planning Agent invoked hand_off — routes to the
        # delegated top-level agent. Wiring every agent here is necessary so
        # LangGraph accepts ``selected_agent`` as a valid return from
        # ``_should_continue_planning``.
        planning_tools_routing = {agent_name: agent_name for agent_name in self.agents}
        planning_tools_routing["custom_agent"] = "custom_agent"
        planning_tools_routing["end"] = END
        workflow.add_conditional_edges(
            "planning_tools",
            self._should_continue_planning,
            planning_tools_routing,
        )

        workflow.add_edge("approval", "tools")

        # Dynamic tool routing map based on agent registry
        # Include ALL agents (including rag_agent) so hand_off delegation works.
        tool_routing_map = {agent_name: agent_name for agent_name in self.agents}
        tool_routing_map["custom_agent"] = "custom_agent"
        tool_routing_map["end"] = END

        workflow.add_conditional_edges(
            "tools",
            self._route_tool_output,
            tool_routing_map,
        )

        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    async def _tool_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        pending_tool_message = find_pending_tool_call_message(messages)
        if not pending_tool_message:
            return state

        pending_message_idx, pending_message = pending_tool_message

        # Skip tool calls that already have a ToolMessage (e.g. HITL rejections).
        # This keeps the AIMessage tool_calls intact (needed for a valid LLM message
        # sequence) while avoiding re-execution of calls that were already resolved.
        already_resolved_ids = {
            msg.tool_call_id
            for msg in messages[pending_message_idx + 1 :]
            if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None)
        }
        tool_calls_pending = [
            tc
            for tc in pending_message.tool_calls
            if normalize_tool_call(tc).get("id") not in already_resolved_ids
        ]
        if not tool_calls_pending:
            # All tool calls for this AI message are already resolved (all rejected).
            # Return early so _route_tool_output can send the agent back to re-respond.
            return state

        selected_agent_name = state.get("selected_agent")
        agent = self._resolve_runtime_agent(state, selected_agent_name)
        if not agent:
            logger.warning(
                "Skipping tool execution: selected_agent '%s' not in agent registry",
                selected_agent_name,
            )
            # Strip tool_calls from the pending AIMessage to prevent downstream
            # routing confusion when the tools node cannot execute anything.
            sanitized = AIMessage(content=pending_message.content or "")
            state["messages"] = (
                messages[:pending_message_idx] + [sanitized] + messages[pending_message_idx + 1 :]
            )
            return state

        tool_map = await ensure_agent_tool_map(
            agent,
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            device_id=state.get("device_id"),
        )
        if not tool_map:
            return state

        tool_outputs, tool_artifacts, all_images = await self._execute_agent_tool_calls(
            state=state,
            agent=agent,
            tool_calls=tool_calls_pending,
            tool_map=tool_map,
            capture_images=True,
        )
        self._apply_tool_outputs_to_state(
            state,
            tool_outputs=tool_outputs,
            tool_artifacts=tool_artifacts,
            all_images=all_images,
            truncate_outputs=True,
        )

        # ── Inter-agent delegation (hand_off tool) ──────────────────────
        state = self._apply_hand_off_if_present(state, tool_outputs)

        return state

    # ------------------------------------------------------------------
    # hand_off delegation helper
    # ------------------------------------------------------------------
    def _apply_hand_off_if_present(self, state: GraphState, tool_outputs: list) -> GraphState:
        """Detect a hand_off tool result and re-route to the target agent.

        If ``delegation_count`` exceeds ``MAX_DELEGATION_DEPTH`` the delegation
        is rejected and an explanatory ToolMessage is appended instead.
        """
        hand_off_output = None
        for output in tool_outputs:
            if output.get("name") == "hand_off":
                hand_off_output = output
                break
        if hand_off_output is None:
            return state

        try:
            payload = json.loads(hand_off_output["content"])
            target_agent = payload.get("hand_off")
            reason = payload.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Malformed hand_off tool output; ignoring delegation")
            return state

        # Validate target agent exists: a base agent or an attached custom agent.
        if target_agent not in self.agents and not self._is_attached_custom_agent(
            state, target_agent
        ):
            logger.warning("hand_off requested unknown/unattached agent '%s'", target_agent)
            tool_call_id = hand_off_output.get("tool_call_id")
            if tool_call_id:
                state.setdefault("messages", []).append(
                    ToolMessage(
                        content=(
                            f"Hand-off refused: '{target_agent}' is not a valid target. "
                            "It is not a base agent and not a custom agent attached to this "
                            "conversation. Answer the request yourself or hand off to a listed "
                            "target."
                        ),
                        tool_call_id=tool_call_id,
                        name="hand_off",
                    )
                )
            return state

        # Circuit-breaker: cap delegation depth
        delegation_count = state.get("delegation_count") or 0
        if delegation_count >= MAX_DELEGATION_DEPTH:
            logger.warning(
                "Delegation depth %d reached limit of %d; refusing hand_off to '%s'",
                delegation_count,
                MAX_DELEGATION_DEPTH,
                target_agent,
            )
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=(
                        f"Delegation refused: maximum depth of {MAX_DELEGATION_DEPTH} reached. "
                        "Please answer the user's request directly."
                    ),
                    tool_call_id=hand_off_output["tool_call_id"],
                    name="hand_off",
                )
            )
            return state

        previous_agent = state.get("selected_agent")
        logger.info(
            "Delegating from '%s' → '%s' (reason: %s)",
            previous_agent,
            target_agent,
            reason,
        )
        state["selected_agent"] = target_agent
        state["delegation_count"] = delegation_count + 1

        # Control-plane handoff metadata — used by ``_messages_for_selected_agent``
        # to strip handoff control AIMessage/ToolMessage pairs out of the
        # delegated agent's prompt, and by the streamer to emit an
        # ``agent_selected`` event when the active agent changes.
        context = state.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        context["handoff"] = {
            "active": True,
            "source_agent": previous_agent,
            "target_agent": target_agent,
            "reason": reason,
            "tool_call_id": hand_off_output.get("tool_call_id"),
        }
        state["context"] = context
        self._record_agent_invocation(state, target_agent, via="handoff", reason=reason)
        return state

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

    async def _prepare_interrupt_payload(
        self,
        state: GraphState,
        *,
        tool_calls: list[Any],
        agent: Any | None,
        tool_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach device/runtime provenance to pending tool approvals."""
        state_view = GraphStateView(state)
        normalized_calls = [normalize_tool_call(tool_call) for tool_call in tool_calls]
        if tool_map is None and agent is not None:
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=state_view.conversation_id(),
                user_id=state_view.user_id(),
                device_id=state_view.device_id(),
            )

        device_id = state_view.device_id()
        provenance: dict[str, dict[str, Any]] = {}
        enriched_calls: list[dict[str, Any]] = []

        for tool_call in normalized_calls:
            enriched_call = dict(tool_call)
            tool_call_id = enriched_call.get("id") or enriched_call.get("tool_call_id")
            if tool_call_id and "tool_call_id" not in enriched_call:
                enriched_call["tool_call_id"] = tool_call_id

            tool_name = enriched_call.get("name")
            tool = tool_map.get(tool_name) if tool_map and tool_name else None
            tool_metadata = getattr(tool, "metadata", None) if tool is not None else None

            provenance_entry: dict[str, Any] = {}
            if device_id:
                provenance_entry["device_id"] = device_id
            if isinstance(tool_metadata, dict):
                for field_name in (
                    "tool_origin",
                    "server_name",
                    "qualified_tool_id",
                    "tool_instance_id",
                    "session_id",
                    "catalog_version",
                ):
                    if field_name not in tool_metadata:
                        continue
                    if tool_metadata[field_name] is None:
                        continue
                    if tool_metadata[field_name] == "":
                        continue
                    provenance_entry[field_name] = tool_metadata[field_name]

            if provenance_entry:
                provenance_key = str(tool_call_id or tool_name or len(provenance))
                provenance[provenance_key] = provenance_entry

            enriched_calls.append(enriched_call)

        interrupt_metadata: dict[str, Any] = {}
        if device_id:
            interrupt_metadata["device_id"] = device_id
        if provenance:
            interrupt_metadata["tool_provenance"] = provenance

        context = state_view.context_copy()
        context["pending_action_requests"] = enriched_calls
        if interrupt_metadata:
            context["interrupt_metadata"] = interrupt_metadata
        state["context"] = context

        payload: dict[str, Any] = {"action_requests": enriched_calls}
        if interrupt_metadata:
            payload["metadata"] = interrupt_metadata
        return payload

    async def _approval_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name) if selected_agent_name else None
        interrupt_payload = await self._prepare_interrupt_payload(
            state,
            tool_calls=last_message.tool_calls,
            agent=agent,
        )
        interrupt_payload["action_requests"]

        # Label the stop reason before yielding to the human so callers can
        # distinguish approval-gate pauses from budget/error pauses.
        context = dict(state.get("context") or {})
        context["pause_reason"] = "awaiting_approval"
        state["context"] = context

        human_decisions = interrupt(interrupt_payload)
        context = GraphStateView(state).context_copy()
        context.pop("pause_reason", None)
        state["context"] = context

        if not human_decisions:
            _, rejected_feedback = _apply_decisions(last_message.tool_calls, [])
        else:
            _, rejected_feedback = _apply_decisions(last_message.tool_calls, human_decisions)

        rejection_messages = [
            ToolMessage(
                content=rejected_feedback[tc.get("id")],
                tool_call_id=tc.get("id"),
                name=tc.get("name"),
            )
            for tc in last_message.tool_calls
            if tc.get("id") in rejected_feedback
        ]

        if rejected_feedback:
            context = state.get("context", {})
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(
                build_rejected_tool_artifacts(
                    tool_calls=last_message.tool_calls,
                    rejected_feedback=rejected_feedback,
                )
            )
            context["tool_artifacts"] = existing_artifacts
            state["context"] = context

        state["messages"] = messages[:-1] + [last_message] + rejection_messages

        return state

    def _should_call_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        tool_names = [normalize_tool_call(tc).get("name") for tc in last_message.tool_calls]
        if requires_human_approval(tool_names):
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

    async def _summarization_node(self, state: GraphState) -> GraphState:
        """No-op compatibility node.

        Long-term summary refresh now happens after assistant persistence,
        not on the streaming hot path. This node only exists so that any
        legacy checkpoint that previously routed through ``summarize`` does
        not error out — it just passes the state through unchanged.
        """
        return state

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
        custom_agents = GraphStateView(state).custom_agents()
        custom_descriptors = [
            {
                "runtime_agent_id": entry.get("runtime_agent_id") or runtime_id,
                "name": entry.get("name"),
                "description": entry.get("description"),
                "agent_order": entry.get("agent_order", 0),
            }
            for runtime_id, entry in custom_agents.items()
        ]
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

        pause_reason, planning_budget_reached = cls._get_planning_pause_details(state_values)
        if planning_budget_reached:
            response.metadata["planning_budget_reached"] = True
        if pause_reason:
            response.metadata["pause_reason"] = pause_reason

        return response

    async def _execute_agent_tool_calls(
        self,
        *,
        state: GraphState,
        agent: Any,
        tool_calls: list[Any],
        tool_map: dict[str, Any] | None = None,
        capture_images: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        if not tool_calls:
            return [], [], []

        state_view = GraphStateView(state)
        conversation_id = state_view.conversation_id()
        user_id = state_view.user_id()
        device_id = state_view.device_id()
        agent_key = (
            getattr(agent, "tool_state_key", None)
            or getattr(agent, "agent_config_key", None)
            or getattr(agent, "agent_id", None)
            or "unknown"
        )

        self._hydrate_deferred_tool_snapshot_from_state(state, agent=agent)

        if tool_map is None:
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
        if not tool_map:
            return [], [], []

        with tool_execution_context(
            conversation_id,
            user_id,
            agent_key,
            device_id,
        ):
            outputs, artifacts, images = await execute_tool_calls(
                tool_calls=tool_calls,
                tool_map=tool_map,
                capture_images=capture_images,
                device_id=device_id,
                agent=agent,
                conversation_id=conversation_id,
                user_id=user_id,
            )
        self._persist_deferred_tool_snapshot_to_state(state, agent=agent)
        return outputs, artifacts, images

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

    def _tool_end_events_from_node_state(
        self,
        *,
        node_state: dict[str, Any],
        last_state_values: dict[str, Any] | None,
        emitted_tool_result_ids: set[str],
    ):
        messages = node_state.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        messages_to_emit = messages
        if messages and isinstance(messages[-1], ToolMessage):
            first_trailing_index = len(messages) - 1
            while first_trailing_index > 0 and isinstance(
                messages[first_trailing_index - 1], ToolMessage
            ):
                first_trailing_index -= 1
            messages_to_emit = messages[first_trailing_index:]

        for message in messages_to_emit:
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = getattr(message, "tool_call_id", None)
            dedupe_key = str(tool_call_id or f"{getattr(message, 'name', 'unknown')}:{id(message)}")
            if dedupe_key in emitted_tool_result_ids:
                continue
            emitted_tool_result_ids.add(dedupe_key)

            event_payload = {
                "type": "tool_end",
                "name": getattr(message, "name", "unknown"),
                "tool_call_id": tool_call_id,
                "result": make_json_safe(message.content),
            }
            render_payload = self._lookup_tool_render_payload(
                last_state_values,
                tool_call_id,
            ) or self._lookup_tool_render_payload(node_state, tool_call_id)
            if render_payload:
                event_payload["render"] = render_payload
            yield event_payload

    def _apply_tool_outputs_to_state(
        self,
        state: GraphState,
        *,
        tool_outputs: list[dict[str, Any]],
        tool_artifacts: list[dict[str, Any]] | None = None,
        all_images: list[dict[str, str]] | None = None,
        truncate_outputs: bool = False,
        mirror_to_response: bool = False,
    ) -> None:
        max_chars = getattr(settings, "tool_result_max_chars", 0) or 0
        truncation_suffix = getattr(
            settings,
            "tool_result_truncation_suffix",
            "\n\n[Output truncated - full result available in tool artifacts]",
        )

        assistant_message_id = state.get("assistant_message_id")
        for output in tool_outputs:
            content = output["content"]
            if truncate_outputs and max_chars > 0:
                content, was_truncated = truncate_tool_result(
                    content,
                    max_chars=max_chars,
                    truncation_suffix=truncation_suffix,
                )
                if was_truncated:
                    logger.debug(
                        "Truncated tool output for %s from %d to %d chars",
                        output["name"],
                        len(output["content"]),
                        len(content),
                    )

            tool_kwargs: dict[str, Any] = {
                "content": content,
                "tool_call_id": output["tool_call_id"],
                "name": output["name"],
            }
            # Stamp ToolMessages with a derived id so terminal-turn compaction
            # can remove them. tool_call_id alone is unique within a turn but
            # langchain BaseMessage id is what RemoveMessage targets.
            if assistant_message_id and output.get("tool_call_id"):
                tool_kwargs["id"] = f"{assistant_message_id}-toolmsg-{output['tool_call_id']}"
            state.setdefault("messages", []).append(ToolMessage(**tool_kwargs))

        state["iteration_count"] = (state.get("iteration_count") or 0) + 1

        context = GraphStateView(state).context_copy()
        if tool_artifacts:
            existing_artifacts = list(context.get("tool_artifacts", []))
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        render_results = dict(context.get("tool_render_results", {}))
        for output in tool_outputs:
            tool_call_id = output.get("tool_call_id")
            render = output.get("render")
            if tool_call_id and isinstance(render, dict):
                render_results[str(tool_call_id)] = make_json_safe(render)
        if render_results:
            context["tool_render_results"] = render_results
        if all_images:
            existing_images = list(context.get("tool_images", []))
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        # Lift artifact-attached rich-item candidates into turn-scoped context.
        # Each artifact may carry `_rich_item_candidates`; merge unique by id.
        if tool_artifacts:
            existing_candidates: list[dict[str, Any]] = list(
                context.get("rich_item_candidates", [])
            )
            seen_ids = {c.get("id") for c in existing_candidates if isinstance(c, dict)}
            for artifact in tool_artifacts:
                if not isinstance(artifact, dict):
                    continue
                candidates = artifact.get("_rich_item_candidates")
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    cid = candidate.get("id")
                    if not cid or cid in seen_ids:
                        continue
                    existing_candidates.append(candidate)
                    seen_ids.add(cid)
            if existing_candidates:
                context["rich_item_candidates"] = existing_candidates
        state["context"] = context

        if mirror_to_response:
            response = state.get("response")
            if response and tool_outputs:
                if response.tool_artifacts is None:
                    response.tool_artifacts = []
                for output in tool_outputs:
                    response.tool_artifacts.append(
                        {
                            "tool": output["name"],
                            "result": output["content"],
                        }
                    )
                state["response"] = response

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat", state=state
        )

        attachments = self._get_state_attachments(state)
        device_id = state.get("device_id")
        current_turn_messages = self._messages_for_selected_agent(
            state,
            state.get("selected_agent") or "chat_agent",
            messages,
        )
        has_images = False
        if attachments:
            current_turn_messages, has_images = self._build_chat_turn_messages_with_attachments(
                current_turn_messages,
                attachments,
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

        if has_images:
            if response.metadata is None:
                response.metadata = {}
            response.metadata["has_images"] = True

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    # ------------------------------------------------------------------
    # Custom-agent multiplexing
    # ------------------------------------------------------------------
    def _resolve_runtime_agent(self, state: GraphState, selected_agent: str | None) -> Any:
        """Resolve a base agent or build a custom agent for the selected id."""
        if selected_agent in self.agents:
            return self.agents[selected_agent]
        if is_custom_runtime_id(selected_agent):
            return self._build_custom_agent(state, selected_agent)
        return None

    def _custom_handoff_targets(self, state: GraphState, runtime_agent_id: str | None) -> list[str]:
        """Valid hand_off targets for a custom agent: base agents + other custom."""
        targets = list(self.agents)
        targets.extend(
            cid for cid in GraphStateView(state).custom_agents() if cid != runtime_agent_id
        )
        return targets

    def _custom_handoff_target_descriptions(
        self, state: GraphState, runtime_agent_id: str | None
    ) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        # Base agents: capability blurbs so a limited-toolset agent can recognise
        # which specialist to delegate to when work falls outside its own tools.
        for agent_id in self.agents:
            if agent_id == runtime_agent_id:
                continue
            blurb = base_agent_capability(agent_id)
            if blurb:
                descriptions[agent_id] = blurb
        # Other attached custom agents.
        for cid, entry in GraphStateView(state).custom_agents().items():
            if cid == runtime_agent_id or not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "Custom Agent").strip() or "Custom Agent"
            detail = str(entry.get("description") or "").strip()
            descriptions[cid] = f"{name}: {detail}" if detail else name
        return descriptions

    # ------------------------------------------------------------------
    # Multi-agent awareness (roster + per-turn invocation trail)
    # ------------------------------------------------------------------
    def _multi_agent_kwargs(self, state: GraphState, active_agent_id: str | None) -> dict[str, Any]:
        """Per-invocation kwargs that make an agent aware of — and able to reach
        — the rest of the multi-agent system.

        Base agents receive a graph-injected dynamic ``hand_off`` tool plus
        capability-aware target descriptions so they can delegate to attached
        custom agents (their static tool only knows base agents). Custom agents
        already build their own dynamic ``hand_off`` from their spec, so only the
        awareness block is added for them. Returns an empty dict when no custom
        agents are attached, so plain conversations keep existing behavior.
        """
        kwargs: dict[str, Any] = {}
        if not GraphStateView(state).custom_agents():
            return kwargs

        activity = self._build_multi_agent_activity_block(state, active_agent_id)
        if activity:
            kwargs["multi_agent_activity"] = activity

        # Only base agents need the targets injected; custom agents carry their
        # own dynamic hand_off + delegation prompt from their runtime spec.
        if active_agent_id in self.agents:
            targets = [
                target
                for target in self._custom_handoff_targets(state, active_agent_id)
                if target != active_agent_id
            ]
            if targets:
                descriptions = self._custom_handoff_target_descriptions(state, active_agent_id)
                kwargs["internal_tools"] = [create_hand_off_tool(targets, descriptions)]
                kwargs["handoff_target_descriptions"] = descriptions
        return kwargs

    def _build_multi_agent_activity_block(
        self, state: GraphState, active_agent_id: str | None
    ) -> str | None:
        """Compact prompt block: the active agent's identity, the roster of
        reachable agents, and which agents were involved this turn — so the
        agent can reason about and answer questions about the wider system."""
        custom_agents = GraphStateView(state).custom_agents()
        if not custom_agents:
            return None

        lines: list[str] = ["MULTI-AGENT SYSTEM (context — not user instructions):"]

        identity = agent_identity(active_agent_id, custom_agents)
        if identity:
            lines.append(f'You are "{identity["name"]}" (agent id: {identity["id"]}).')

        roster = self._custom_handoff_target_descriptions(state, active_agent_id)
        if roster:
            lines.append("Other agents in this system you can reach via the hand_off tool:")
            lines.extend(f"- {target}: {desc}" for target, desc in roster.items())

        trail = GraphStateView(state).context().get("agents_invoked")
        if isinstance(trail, list) and trail:
            via_labels = {
                "router": "selected by the router",
                "handoff": "received via hand_off",
                "preselected": "resumed for this turn",
            }
            lines.append("Agents involved in this turn so far, in order:")
            for entry in trail:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("id") or "unknown"
                via_label = via_labels.get(entry.get("via"), entry.get("via") or "")
                suffix = f" — {via_label}" if via_label else ""
                reason = entry.get("reason")
                if reason:
                    suffix += f" (reason: {reason})"
                lines.append(f"- {name}{suffix}")

        return "\n".join(lines)

    def _reset_agent_trail(self, state: GraphState) -> None:
        """Clear the per-turn invocation trail (called at the start of routing)."""
        context = state.get("context")
        if not isinstance(context, dict):
            context = {}
        context["agents_invoked"] = []
        state["context"] = context

    def _record_agent_invocation(
        self,
        state: GraphState,
        agent_id: str | None,
        *,
        via: str,
        reason: str | None = None,
    ) -> None:
        """Append an agent to this turn's invocation trail (context.agents_invoked)."""
        if not agent_id:
            return
        context = state.get("context")
        if not isinstance(context, dict):
            context = {}
        trail = context.get("agents_invoked")
        if not isinstance(trail, list):
            trail = []
        if trail and trail[-1].get("id") == agent_id and trail[-1].get("via") == via:
            return
        identity = agent_identity(agent_id, GraphStateView(state).custom_agents())
        entry: dict[str, Any] = {
            "id": agent_id,
            "name": identity["name"] if identity else agent_id,
            "kind": identity["kind"] if identity else "base",
            "via": via,
        }
        if reason:
            entry["reason"] = reason
        trail.append(entry)
        context["agents_invoked"] = trail
        state["context"] = context

    def _build_custom_agent(
        self, state: GraphState, runtime_agent_id: str | None
    ) -> CustomAgent | None:
        """Build a live CustomAgent from the workflow ``custom_agents`` state.

        Built fresh each invocation so edited configuration applies to future
        turns (live config). Returns ``None`` if the agent is not attached.
        """
        entry = GraphStateView(state).custom_agents().get(runtime_agent_id)
        if not entry:
            return None
        spec = build_custom_agent_runtime_spec(
            entry,
            allowed_handoff_targets=self._custom_handoff_targets(state, runtime_agent_id),
            handoff_target_descriptions=self._custom_handoff_target_descriptions(
                state, runtime_agent_id
            ),
        )
        return CustomAgent(spec, runtime_model_resolver=self._runtime_model_resolver)

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

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _rag_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="rag", state=state
        )
        device_id = state.get("device_id")

        last_human_idx = self._find_last_human_message_index(messages)
        original_query = messages[last_human_idx].content if last_human_idx is not None else content

        tool_context = []
        if last_human_idx is not None:
            for msg in messages[last_human_idx + 1 :]:
                if isinstance(msg, ToolMessage):
                    # ``hand_off`` ToolMessages are control-plane signals, not
                    # RAG evidence — skip them so handoff JSON does not leak
                    # into the delegated agent's tool context.
                    if getattr(msg, "name", None) == "hand_off":
                        continue
                    tool_context.append(msg.content)

        rag_context = state.get("context", {}) or {}
        metadata = {
            "persona": state.get("persona"),
            "history": conversation_history,
            "original_query": original_query,
            "tool_context": tool_context,
            "agentic_images": rag_context.get(
                "agentic_images", []
            ),  # Pass images for multimodal LLM
            "model_request": state.get("model_request"),
            "user_id": user_id,
            "device_id": device_id,
            "history_summary": state.get("history_summary"),
        }
        # When ``_should_continue_rag`` decides the budget is exhausted, it
        # sets these flags on the context so this RAG turn becomes a no-tools
        # synthesis pass. Forward them into the AgentMessage metadata —
        # ``RAGAgent._process_message_agentic`` reads them.
        if rag_context.get("rag_force_final_response"):
            metadata["rag_force_final_response"] = True
            notice = rag_context.get("rag_tool_budget_notice")
            if notice:
                metadata["rag_tool_budget_notice"] = notice

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=original_query,
            metadata=metadata,
            attachments=self._get_state_attachments(state),
        )

        response = await self.rag_agent.process_message(agent_msg, conversation_id)
        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        state["response"] = response

        ai_kwargs = {"content": response.message.content or ""}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    async def _rag_tools_node(self, state: GraphState) -> GraphState:
        """
        Execute RAG document exploration tools (agentic mode).

        - SCAN_ALL: Preview all documents in conversation
        - READ_DOCUMENT: Full content of specific document
        - SEARCH_CHUNKS: Vector search
        - GREP_DOCUMENT: Regex search
        - LIST_DOCUMENTS: List available documents
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        tool_outputs = []
        context = state.get("context", {})
        tool_artifacts: list[dict[str, Any]] = []
        all_images: list[dict[str, str]] = []
        max_agentic_images = getattr(settings, "agentic_rag_max_images", 6)

        # Track agentic iteration count
        agentic_iteration = context.get("agentic_rag_iteration", 0) + 1
        context["agentic_rag_iteration"] = agentic_iteration

        normalized_tool_calls = [
            normalize_tool_call(tool_call) for tool_call in last_message.tool_calls
        ]

        non_search_tool_calls = [
            tc for tc in normalized_tool_calls if tc.get("name") != "search_documents"
        ]
        non_search_outputs_by_id: dict[str, dict[str, Any]] = {}
        rejected_feedback: dict[str, str] = {}
        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name) if selected_agent_name else None

        if non_search_tool_calls:
            tool_calls_to_execute = list(non_search_tool_calls)

            tool_names = [tc.get("name") for tc in non_search_tool_calls]
            if requires_human_approval(tool_names):
                interrupt_payload = await self._prepare_interrupt_payload(
                    state,
                    tool_calls=non_search_tool_calls,
                    agent=agent,
                )
                human_decisions = interrupt(interrupt_payload)

                if not human_decisions:
                    tool_calls_to_execute, rejected_feedback = _apply_decisions(
                        non_search_tool_calls, []
                    )
                else:
                    tool_calls_to_execute, rejected_feedback = _apply_decisions(
                        non_search_tool_calls, human_decisions
                    )

                for tc_id, feedback in rejected_feedback.items():
                    non_search_outputs_by_id[tc_id] = {"content": feedback}

            if rejected_feedback:
                tool_artifacts.extend(
                    build_rejected_tool_artifacts(
                        tool_calls=non_search_tool_calls,
                        rejected_feedback=rejected_feedback,
                    )
                )

            if tool_calls_to_execute:
                tool_map = (
                    await ensure_agent_tool_map(
                        agent,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        device_id=state.get("device_id"),
                    )
                    if agent
                    else {}
                )

                # Extract context for tool execution. Use the deferred-state key
                # (tool_state_key) so custom agents resolve their own loaded tools.
                user_id = state.get("user_id")
                if agent:
                    agent_key = (
                        getattr(agent, "tool_state_key", None)
                        or getattr(agent, "agent_config_key", None)
                        or "rag"
                    )
                else:
                    agent_key = "rag"
                device_id = state.get("device_id")

                # Execute tools with context set for deferred tool loading support
                with tool_execution_context(
                    conversation_id,
                    user_id,
                    agent_key,
                    device_id,
                ):
                    outputs, artifacts, images = await execute_tool_calls(
                        tool_calls=tool_calls_to_execute,
                        tool_map=tool_map,
                        capture_images=True,
                        device_id=device_id,
                        agent=agent,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                for output in outputs:
                    if output.get("tool_call_id"):
                        non_search_outputs_by_id[output["tool_call_id"]] = output
                tool_artifacts.extend(artifacts)
                all_images.extend(images)

        for tool_call_data in normalized_tool_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            if tool_name != "search_documents":
                stored = non_search_outputs_by_id.get(tool_id)
                entry: dict[str, Any] = {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                }
                if stored is not None:
                    entry["content"] = stored.get("content", "")
                    render = stored.get("render")
                    if isinstance(render, dict):
                        entry["render"] = render
                else:
                    entry["content"] = f"Error: Tool {tool_name} not found"
                tool_outputs.append(entry)
                continue

            result, _, evidence = await execute_search_documents_action(
                rag_agent=self.rag_agent,
                conversation_id=conversation_id,
                tool_args=tool_args,
                context=context,
                max_agentic_images=max_agentic_images,
                user_id=state.get("user_id"),
            )

            error = result if result.startswith("Error") else None
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
            if blob_info:
                artifact.update(blob_info)
            if evidence:
                artifact["rag_evidence"] = make_json_safe(evidence)
            tool_artifacts.append(artifact)
            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": public_text,
                }
            )

        # Add tool messages to state
        for output in tool_outputs:
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=output["content"],
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        state["context"] = context

        # Increment iteration count
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        if tool_artifacts:
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        render_results = dict(context.get("tool_render_results", {}))
        for output in tool_outputs:
            tool_call_id = output.get("tool_call_id")
            render = output.get("render")
            if tool_call_id and isinstance(render, dict):
                render_results[str(tool_call_id)] = make_json_safe(render)
        if render_results:
            context["tool_render_results"] = render_results
        if all_images:
            existing_images = context.get("tool_images", [])
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        return state

    def _should_call_rag_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            context = state.get("context", {}) or {}
            if context.get("rag_force_final_response"):
                logger.warning(
                    "RAG final no-tools pass still emitted tool calls; ending "
                    "instead of executing more RAG tools."
                )
                return "end"
            return "rag_tools"

        return "end"

    def _should_continue_rag(self, state: GraphState) -> str:
        """
        Determine if RAG agentic loop should continue or end.

        When the iteration budget is reached, route the model back to
        ``rag_agent`` for one final no-tools synthesis pass so the user
        always sees an answer rather than a truncated tool log. If we have
        already forced that final pass once and the model still asked for
        tools, end the graph to avoid an infinite loop.
        """
        context = state.get("context", {})
        agentic_iteration = context.get("agentic_rag_iteration", 0)

        max_iterations = settings.agentic_max_iterations
        if agentic_iteration >= max_iterations:
            if context.get("rag_force_final_response"):
                logger.warning(
                    "RAG agentic loop already had its final no-tools pass "
                    "(iteration=%d, max=%d); ending to avoid an infinite loop.",
                    agentic_iteration,
                    max_iterations,
                )
                return "end"

            logger.warning(
                "RAG agentic loop reached max iterations (%d); forcing one "
                "final no-tools synthesis pass.",
                max_iterations,
            )
            context["rag_force_final_response"] = True
            context["rag_tool_budget_notice"] = (
                "The RAG tool budget is exhausted. Produce the final answer "
                "from the retrieved document evidence already available. "
                "Do not call tools."
            )
            state["context"] = context
            return "rag_agent"

        return "rag_agent"

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
        run_config = RunnableConfig(
            tags=["internal", "planning_subagent", f"subagent:{agent_name}"],
            metadata={
                "internal": True,
                "purpose": "planning_subagent",
                "subagent": True,
                "subagent_agent": agent_name,
            },
        )

        worker_message = HumanMessage(content=task_prompt)

        # RAG worker: drive the same search_documents loop used by the graph,
        # but keep all intermediate context local to this worker.
        if agent_name == "rag_agent":
            max_agentic_images = getattr(settings, "agentic_rag_max_images", 6)
            rag_context = dict(parent_state.get("context") or {})
            tool_context: list[str] = []
            accumulated_artifacts: list[dict[str, Any]] = []
            rag_tool_map: dict[str, Any] | None = None

            while True:
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

                normalized_calls = [normalize_tool_call(tc) for tc in tool_calls]
                tool_call_names = [tc.get("name") or "" for tc in normalized_calls]
                if requires_human_approval(tool_call_names):
                    if response.metadata is None:
                        response.metadata = {}
                    response.metadata["requires_approval"] = True
                    response.metadata["pause_reason"] = "awaiting_approval"
                    if accumulated_artifacts:
                        response.tool_artifacts = accumulated_artifacts
                    return response

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
                        error = result if result.startswith("Error") else None
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

        # Generic agent worker: tool-loop until final response, approval, or error.

        worker_messages: list[Any] = [worker_message]
        tool_map: dict[str, Any] | None = None
        accumulated_worker_artifacts: list[dict[str, Any]] = []

        while True:
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
                return response

            tool_call_names = [normalize_tool_call(tc).get("name") or "" for tc in tool_calls]
            if requires_human_approval(tool_call_names):
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

            if tool_map is None:
                tool_map = await ensure_agent_tool_map(
                    agent,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                )
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

    def _build_planning_internal_tools(
        self,
        state: GraphState,
        *,
        executable: bool = False,
    ) -> list[Any]:
        """Return Planning-supervisor-only internal tools for this turn.

        ``dispatch_subagents`` is bound whenever the feature flag is on and
        Planning mode is active. ``planning_phase`` and plan presence are NOT
        binding gates — they are prompt-level guidance. Hiding the tool prevents
        the user from testing subagents on a fresh planning conversation.
        """
        if not getattr(settings, "planning_subagents_enabled", False):
            return []
        if not state.get("planning_mode_enabled"):
            return []

        from .planning_subagents import (
            PlanningSubagentDispatcher,
            create_dispatch_subagents_tool,
        )

        dispatcher = (
            PlanningSubagentDispatcher(workflow=self, settings=settings) if executable else None
        )
        dispatch_tool = create_dispatch_subagents_tool(
            dispatcher=dispatcher,
            parent_state_provider=(lambda: state) if executable else None,
        )
        return [dispatch_tool]

    async def _planning_node(self, state: GraphState) -> GraphState:
        """
        Planning agent node with tool-calling ReAct pattern.

        Uses invoke_model_with_history to get responses that may include tool calls
        for the write_todos tool. Supports both initial plan creation and
        ongoing task management.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="planning", state=state
        )

        context = state.get("context", {})

        # Get planning phase and check if we need to generate plan response
        planning_phase = state.get("planning_phase", "planning")
        should_generate_plan_response = context.get("generate_plan_response", False)

        # Clear the flag after reading
        if should_generate_plan_response:
            context["generate_plan_response"] = False
            state["context"] = context

        # Get current todos from state (may have been updated by planning_tools)
        todos = state.get("todos", [])
        current_task_index = state.get("current_task_index")

        # Increment planning call count for budget tracking
        planning_call_count = (state.get("planning_call_count") or 0) + 1
        state["planning_call_count"] = planning_call_count

        # Debug logging for observability
        logger.debug(
            f"[Planning Node] phase={planning_phase}, call_count={planning_call_count}, "
            f"generate_plan_response={should_generate_plan_response}, "
            f"todos_count={len(todos)}, current_task_index={current_task_index}"
        )

        persona = state.get("persona")

        # Get only current turn messages for the model
        current_turn_messages = self._get_current_turn_messages(messages)

        # Build the Planning-mode subagent dispatch tool (only when allowed).
        internal_tools = self._build_planning_internal_tools(state)

        # Call the planning agent with history
        custom_workers = [
            {
                "runtime_agent_id": entry.get("runtime_agent_id") or runtime_id,
                "name": entry.get("name"),
                "description": entry.get("description"),
            }
            for runtime_id, entry in GraphStateView(state).custom_agents().items()
        ]

        response = await self.planning_agent.invoke_model_with_history(
            messages=current_turn_messages,
            conversation_history=conversation_history,
            persona=persona,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
            todos=todos,
            current_task_index=current_task_index,
            planning_phase=planning_phase,
            should_describe_plan=should_generate_plan_response,
            internal_tools=internal_tools or None,
            custom_workers=custom_workers or None,
            **self._final_response_kwargs(state),
        )

        # Check if agent switched to executing phase via response metadata
        if response.metadata.get("planning_phase"):
            state["planning_phase"] = response.metadata["planning_phase"]

        if response.metadata.get("todos"):
            state["todos"] = response.metadata["todos"]

        # Mark that final summary was generated if this was a summary call
        if context.get("generate_final_summary") and not response.message.tool_calls:
            context["final_summary_generated"] = True
            state["context"] = context

        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _planning_tools_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        todos = list(state.get("todos", []))  # Make a copy
        current_task_index = state.get("current_task_index")
        tool_outputs = []
        write_todos_actions: list[str] = []

        # Track errors for circuit breaker
        context = GraphStateView(state).context_copy()
        had_error = False
        max_todos = getattr(settings, "max_todos_per_plan", 50)
        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        # Separate write_todos calls from external/MCP tool calls.
        normalized_calls = [normalize_tool_call(tc) for tc in last_message.tool_calls]
        external_tool_calls = [tc for tc in normalized_calls if tc.get("name") != "write_todos"]
        write_todos_calls = [tc for tc in normalized_calls if tc.get("name") == "write_todos"]

        tool_map: dict[str, Any] = {}
        if external_tool_calls:
            tool_map = await ensure_agent_tool_map(
                self.planning_agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=state.get("device_id"),
            )

        # Make Planning-supervisor-only tools (e.g. dispatch_subagents)
        # callable from this turn. They are not part of the agent's MCP/server
        # tool registry, so ensure_agent_tool_map doesn't surface them.
        if any(tc.get("name") == "dispatch_subagents" for tc in external_tool_calls):
            for tool in self._build_planning_internal_tools(state, executable=True):
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name not in tool_map:
                    tool_map[tool_name] = tool

        tool_names_all = [tc.get("name") for tc in normalized_calls]
        logger.debug(f"[Planning Tools Node] Executing tools: {tool_names_all}")

        # --- HITL approval gate for external (non-write_todos) tool calls ---
        rejected_tool_ids: dict[str, str] = {}  # tool_call_id -> rejection reason
        rejected_feedback: dict[str, str] = {}
        approved_external_calls = list(external_tool_calls)

        if external_tool_calls:
            ext_tool_names = [tc.get("name") for tc in external_tool_calls]
            if requires_human_approval(ext_tool_names):
                # Label the stop reason before yielding to the human.
                _ctx = dict(state.get("context") or {})
                _ctx["pause_reason"] = "awaiting_approval"
                state["context"] = _ctx

                interrupt_payload = await self._prepare_interrupt_payload(
                    state,
                    tool_calls=external_tool_calls,
                    agent=self.planning_agent,
                    tool_map=tool_map,
                )
                human_decisions = interrupt(interrupt_payload)
                _ctx = GraphStateView(state).context_copy()
                _ctx.pop("pause_reason", None)
                state["context"] = _ctx

                if not human_decisions:
                    approved_external_calls, rejected_feedback = _apply_decisions(
                        external_tool_calls, []
                    )
                else:
                    approved_external_calls, rejected_feedback = _apply_decisions(
                        external_tool_calls, human_decisions
                    )

                for tc_id, feedback in rejected_feedback.items():
                    rejected_tool_ids[tc_id] = feedback

        # Emit rejection ToolMessages for rejected external calls
        for tc in external_tool_calls:
            tc_id = tc.get("id")
            if tc_id in rejected_tool_ids:
                tool_outputs.append(
                    {
                        "tool_call_id": tc_id,
                        "name": tc.get("name"),
                        "content": rejected_tool_ids[tc_id],
                    }
                )

        # --- Artifact + image tracking (BP-1) ---
        tool_artifacts: list[dict[str, Any]] = []
        all_images: list[dict[str, str]] = []

        if rejected_feedback:
            tool_artifacts.extend(
                build_rejected_tool_artifacts(
                    tool_calls=external_tool_calls,
                    rejected_feedback=rejected_feedback,
                )
            )

        if approved_external_calls:
            (
                external_outputs,
                external_artifacts,
                external_images,
            ) = await self._execute_agent_tool_calls(
                state=state,
                agent=self.planning_agent,
                tool_calls=approved_external_calls,
                tool_map=tool_map,
                capture_images=True,
            )
            tool_outputs.extend(external_outputs)
            tool_artifacts.extend(external_artifacts)
            all_images.extend(external_images)
            had_error = had_error or any(
                artifact.get("status") == "error" for artifact in external_artifacts
            )

            # If the Planning Agent invoked hand_off, switch the selected agent so
            # the conditional edge from planning_tools can route to the target.
            self._apply_hand_off_if_present(state, external_outputs)

        # Execute write_todos calls (always permitted — internal state mutations)
        for tool_call_data in write_todos_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            try:
                todos, current_task_index, result, action = apply_write_todos_action(
                    todos=todos,
                    current_task_index=current_task_index,
                    tool_args=tool_args,
                    max_todos=max_todos,
                )
                write_todos_actions.append(action)

                if action == "set_todos" and result.startswith("Error: Plan exceeds maximum"):
                    requested = len(tool_args.get("todos", []) or [])
                    logger.warning(
                        "Rejected plan with %d todos (max: %d)",
                        requested,
                        max_todos,
                    )

            except Exception as e:
                raw_action = tool_args.get("action")
                action = raw_action.value if hasattr(raw_action, "value") else raw_action
                result = f"Error executing {action}: {str(e)}"
                had_error = True  # Mark error for circuit breaker

            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
                }
            )

        # Update state with new todos
        state["todos"] = todos
        state["current_task_index"] = current_task_index

        # Check if plan was modified (SET_TODOS, ADD_TODO, UPDATE_TODO, REMOVE_TODO)
        # and mark context so we can return to agent for response generation
        context = state.get("context", {})
        plan_modifying_actions = {"set_todos", "add_todo", "update_todo", "remove_todo"}
        execution_actions = {"start_todo", "complete_todo"}

        for action in write_todos_actions:
            if action in plan_modifying_actions:
                context["plan_just_modified"] = True
            if action in execution_actions:
                state["planning_phase"] = "executing"
                logger.debug(f"Switched to executing phase due to {action} action")
        state["context"] = context
        self._apply_tool_outputs_to_state(
            state,
            tool_outputs=tool_outputs,
            tool_artifacts=tool_artifacts,
            all_images=all_images,
            mirror_to_response=True,
        )

        # Update consecutive_errors counter for circuit breaker. Increments
        # below the warn-threshold are routine retries and stay at INFO so the
        # WARNING level remains a useful "near the breaker" signal regardless
        # of how aggressively ``planning_consecutive_errors_limit`` is tuned.
        context = GraphStateView(state).context_copy()
        if had_error:
            context["consecutive_errors"] = context.get("consecutive_errors", 0) + 1
            current = context["consecutive_errors"]
            error_limit = settings.planning_consecutive_errors_limit
            # Require at least 2 errors before warning (a single transient
            # failure should never warn) AND warn only at the step before the
            # breaker fires. The breaker itself logs its own WARNING when it
            # actually trips, so we don't duplicate that here.
            warn_threshold = max(2, error_limit - 1)
            if current >= warn_threshold and current < error_limit:
                logger.warning("Planning consecutive errors: %d/%d", current, error_limit)
            else:
                logger.info("Planning consecutive errors: %d/%d", current, error_limit)
        else:
            # Reset on successful iteration
            context["consecutive_errors"] = 0
        state["context"] = context

        return state

    def _should_call_planning_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        return "planning_tools"

    def _should_continue_planning(self, state: GraphState) -> str:
        # hand_off applied during this planning_tools turn re-routes the
        # conversation to a different top-level agent. Honor the new
        # selected_agent so the planning loop yields to the target node.
        delegated_agent = state.get("selected_agent")
        if isinstance(delegated_agent, str) and delegated_agent != "planning_agent":
            is_base_agent = delegated_agent in self.agents
            is_attached_custom = self._is_attached_custom_agent(state, delegated_agent)
            if is_base_agent or is_attached_custom:
                delegated_node = self._route_target_for(state, delegated_agent)
                logger.info(
                    "Planning hand_off detected: routing planning_tools to %s via %s",
                    delegated_agent,
                    delegated_node,
                )
                return delegated_node

        planning_call_count = state.get("planning_call_count", 0)
        max_iterations = int(getattr(settings, "planning_max_iterations", 0) or 0)
        planning_budget_enabled = max_iterations > 0
        context = GraphStateView(state).context_copy()
        planning_phase = state.get("planning_phase", "planning")
        last_message_is_tool_output = self._last_message_is_tool_output(state)

        # If plan state was just mutated, always give the Planning Agent one
        # response pass so the user does not get an empty tool-calling message.
        if context.get("plan_just_modified"):
            context["plan_just_modified"] = False
            context["generate_plan_response"] = True
            state["context"] = context
            if planning_budget_enabled and planning_call_count >= max_iterations:
                context["pause_reason"] = "max_iterations_reached"
                state["context"] = context
                self._mark_force_final_response(
                    state,
                    reason="max_iterations_reached",
                    scope="planning",
                    count=planning_call_count,
                    limit=max_iterations,
                )
            logger.debug(
                "[Should Continue Planning] Decision: planning_agent (plan_just_modified=True)"
            )
            return "planning_agent"

        # Hard budget: if the latest thing is a tool result, route back once
        # with tools disabled for a final synthesis instead of ending on the
        # empty intermediate tool-calling response.
        if planning_budget_enabled and planning_call_count >= max_iterations:
            context["pause_reason"] = "max_iterations_reached"
            state["context"] = context
            if last_message_is_tool_output:
                self._mark_force_final_response(
                    state,
                    reason="max_iterations_reached",
                    scope="planning",
                    count=planning_call_count,
                    limit=max_iterations,
                )
                logger.warning(
                    "Planning budget reached after tool output: %d >= %d; "
                    "routing to final synthesis",
                    planning_call_count,
                    max_iterations,
                )
                return "planning_agent"

            self._set_continuation_signal(
                state,
                should_continue=settings.auto_continue_enabled,
                reason="max_iterations_reached",
                scope="planning",
                count=planning_call_count,
                limit=max_iterations,
            )
            logger.warning(f"Planning budget exceeded: {planning_call_count} >= {max_iterations}")
            return "end"

        # Soft-limit: if auto-continue is enabled, trigger continuation at
        # a fraction of the planning budget.
        if planning_budget_enabled and settings.auto_continue_enabled:
            soft_limit = max(1, int(max_iterations * settings.auto_continue_soft_limit_ratio))
            if planning_call_count >= soft_limit:
                if last_message_is_tool_output:
                    logger.debug(
                        "[Should Continue Planning] Decision: planning_agent "
                        "(soft budget reached after tool output; reconcile before pausing)"
                    )
                    return "planning_agent"

                self._set_continuation_signal(
                    state,
                    should_continue=True,
                    reason="soft_budget",
                    scope="planning",
                    count=planning_call_count,
                    limit=soft_limit,
                )
                logger.info(
                    "Planning soft-limit reached: %d >= %d, requesting auto-continue",
                    planning_call_count,
                    soft_limit,
                )
                return "end"

        # Circuit breaker: check consecutive errors
        consecutive_errors = context.get("consecutive_errors", 0)
        max_consecutive_errors = settings.planning_consecutive_errors_limit

        if consecutive_errors >= max_consecutive_errors:
            context["pause_reason"] = "consecutive_errors_limit"
            state["context"] = context
            self._set_continuation_signal(
                state,
                should_continue=False,
                reason="consecutive_errors_limit",
                scope="planning",
                count=consecutive_errors,
                limit=max_consecutive_errors,
            )
            logger.warning(
                f"Planning circuit breaker triggered: {consecutive_errors} consecutive errors"
            )
            return "end"

        # In planning phase, always give the agent a chance to respond after tools
        if planning_phase == "planning":
            messages = state.get("messages", [])
            last_msg_type = type(messages[-1]).__name__ if messages else "None"

            # If last message is a ToolMessage, give agent a chance to process results
            if messages and isinstance(messages[-1], ToolMessage):
                logger.debug(
                    "[Should Continue Planning] Decision: planning_agent "
                    "(last_message=ToolMessage, agent needs to respond)"
                )
                return "planning_agent"

            # Agent already responded with text - end planning loop
            logger.debug(
                f"[Should Continue Planning] Decision: end (planning_phase=planning, "
                f"last_message_type={last_msg_type})"
            )
            return "end"

        # === EXECUTING PHASE LOGIC ===
        todos = state.get("todos", [])
        if todos:
            pending_count = sum(
                1
                for t in todos
                if t.get("status")
                in (
                    TodoStatus.PENDING.value,
                    "pending",
                    TodoStatus.IN_PROGRESS.value,
                    "in_progress",
                )
            )
            if pending_count == 0:
                context["all_tasks_completed"] = True
                state["context"] = context

                # Check if we've already generated the final summary
                if context.get("final_summary_generated"):
                    return "end"

                # Need one more iteration to generate completion summary
                context["generate_final_summary"] = True
                state["context"] = context
                return "planning_agent"

        return "planning_agent"

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

    def _is_attached_custom_agent(self, state: GraphState, runtime_agent_id: str | None) -> bool:
        """True when ``runtime_agent_id`` is an attached custom agent in state."""
        if not is_custom_runtime_id(runtime_agent_id):
            return False
        return runtime_agent_id in GraphStateView(state).custom_agents()

    def _route_target_for(self, state: GraphState, selected_agent: str) -> str:
        """Map a selected agent to its graph node name (custom ids → custom_agent)."""
        if is_custom_runtime_id(selected_agent) and self._is_attached_custom_agent(
            state, selected_agent
        ):
            return "custom_agent"
        return selected_agent

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
        if (
            final_snapshot.next
            and len(final_snapshot.next) > 0
            and "approval" in final_snapshot.next
        ):
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
        if "approval" not in state_snapshot.next:
            raise ValueError(f"Unexpected interrupt state: next nodes are {state_snapshot.next}")

        resume_data = build_interrupt_resume_payload(decisions)

        selected_agent = state_snapshot.values.get("selected_agent", "search_agent")
        conversation_id = state_snapshot.values.get("conversation_id")

        yield {"type": "agent_selected", "agent": selected_agent}
        last_emitted_agent = selected_agent

        accumulated_content = ""
        accumulated_thinking = ""
        current_tool_calls = {}
        emitted_tool_call_ids = set()
        emitted_tool_result_ids = set()

        suppressed_nodes: set = {"image_generator_agent"}
        suppress_tokens = selected_agent in suppressed_nodes
        _internal_content_only: bool = True

        max_rounds = settings.auto_continue_max_rounds if settings.auto_continue_enabled else 1
        round_num = 1
        continue_reason: str | None = "initial"
        total_iterations = 0
        start_time = time.monotonic()
        current_state: Any = Command(resume=resume_data)
        last_state_values: dict[str, Any] | None = None

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
                async for chunk in self.graph.astream(
                    current_state, config=config, stream_mode=["messages", "updates"]
                ):
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        mode, data = chunk

                        if mode == "messages":
                            message_chunk, metadata = data

                            if isinstance(message_chunk, ToolMessage):
                                continue

                            if (
                                settings.suppress_internal_stream_chunks
                                and self._is_internal_stream_chunk(metadata)
                            ):
                                continue

                            _internal_content_only = False

                            if (
                                hasattr(message_chunk, "content_blocks")
                                and message_chunk.content_blocks
                            ):
                                for block in message_chunk.content_blocks:
                                    block_type = block.get("type")

                                    if block_type == "text":
                                        text_content = block.get("text", "")
                                        accumulated_content, delta = (
                                            self._consume_stream_text_chunk(
                                                accumulated_content, text_content
                                            )
                                        )
                                        if delta and not suppress_tokens:
                                            yield {"type": "token", "content": delta}

                                    elif block_type == "thinking":
                                        thinking_content = block.get("thinking", "") or block.get(
                                            "text", ""
                                        )
                                        if thinking_content:
                                            accumulated_thinking += thinking_content
                                            yield {
                                                "type": "thinking",
                                                "content": thinking_content,
                                            }

                                    elif block_type == "reasoning":
                                        reasoning_content = block.get("reasoning", "") or block.get(
                                            "text", ""
                                        )
                                        if reasoning_content:
                                            accumulated_thinking += reasoning_content
                                            yield {
                                                "type": "thinking",
                                                "content": reasoning_content,
                                            }

                                    elif block_type == "tool_call_chunk":
                                        tool_index = block.get("index", 0)
                                        tool_id = block.get("id")
                                        tool_name = block.get("name")
                                        tool_args = block.get("args", "")

                                        if tool_index not in current_tool_calls:
                                            current_tool_calls[tool_index] = {
                                                "id": tool_id,
                                                "name": tool_name,
                                                "args": "",
                                            }

                                        if tool_args:
                                            current_tool_calls[tool_index]["args"] += tool_args

                                        if tool_name and not current_tool_calls[tool_index]["name"]:
                                            current_tool_calls[tool_index]["name"] = tool_name
                                        if tool_id and not current_tool_calls[tool_index]["id"]:
                                            current_tool_calls[tool_index]["id"] = tool_id

                            elif hasattr(message_chunk, "content") and isinstance(
                                message_chunk.content, list
                            ):
                                for part in message_chunk.content:
                                    if isinstance(part, dict):
                                        part_type = part.get("type", "")

                                        if part_type == "thinking":
                                            thinking_content = part.get("thinking", "") or part.get(
                                                "text", ""
                                            )
                                            if thinking_content:
                                                accumulated_thinking += thinking_content
                                                yield {
                                                    "type": "thinking",
                                                    "content": thinking_content,
                                                }
                                        elif part_type == "reasoning":
                                            reasoning_content = part.get(
                                                "reasoning", ""
                                            ) or part.get("text", "")
                                            if reasoning_content:
                                                accumulated_thinking += reasoning_content
                                                yield {
                                                    "type": "thinking",
                                                    "content": reasoning_content,
                                                }
                                        elif part_type == "text":
                                            text_content = part.get("text", "")
                                            accumulated_content, delta = (
                                                self._consume_stream_text_chunk(
                                                    accumulated_content, text_content
                                                )
                                            )
                                            if delta and not suppress_tokens:
                                                yield {
                                                    "type": "token",
                                                    "content": delta,
                                                }
                                    elif isinstance(part, str) and part:
                                        accumulated_content, delta = (
                                            self._consume_stream_text_chunk(
                                                accumulated_content, part
                                            )
                                        )
                                        if delta and not suppress_tokens:
                                            yield {"type": "token", "content": delta}

                            elif (
                                hasattr(message_chunk, "content")
                                and message_chunk.content
                                and isinstance(message_chunk.content, str)
                            ):
                                content = coerce_response_text(message_chunk.content)
                                accumulated_content, delta = self._consume_stream_text_chunk(
                                    accumulated_content, content
                                )
                                if delta and not suppress_tokens:
                                    yield {"type": "token", "content": delta}

                            if (
                                hasattr(message_chunk, "chunk_position")
                                and message_chunk.chunk_position == "last"
                            ):
                                for tool_call in current_tool_calls.values():
                                    if tool_call["name"]:
                                        tool_call_id = tool_call["id"]
                                        if tool_call_id and tool_call_id in emitted_tool_call_ids:
                                            continue
                                        if tool_call_id:
                                            emitted_tool_call_ids.add(tool_call_id)

                                        try:
                                            args = (
                                                json.loads(tool_call["args"])
                                                if tool_call["args"]
                                                else {}
                                            )
                                        except Exception:
                                            args = tool_call["args"]

                                        yield {
                                            "type": "tool_start",
                                            "name": tool_call["name"],
                                            "tool_call_id": tool_call_id,
                                            "args": make_json_safe(args),
                                        }
                                current_tool_calls = {}

                        elif mode == "updates":
                            for node_name, node_state in data.items():
                                if isinstance(node_state, dict):
                                    if last_state_values is None:
                                        last_state_values = {}
                                    last_state_values.update(node_state)

                                    new_agent = node_state.get("selected_agent")
                                    if (
                                        isinstance(new_agent, str)
                                        and new_agent != last_emitted_agent
                                    ):
                                        last_emitted_agent = new_agent
                                        yield {
                                            "type": "agent_selected",
                                            "agent": new_agent,
                                            "reason": "handoff",
                                        }

                                if node_name in ("planning_agent", "planning_tools"):
                                    node_info = {"node": node_name}

                                    if node_name == "planning_agent" and "messages" in node_state:
                                        messages = node_state.get("messages", [])
                                        if messages:
                                            last_msg = (
                                                messages[-1]
                                                if isinstance(messages, list)
                                                else messages
                                            )
                                            if (
                                                isinstance(last_msg, AIMessage)
                                                and hasattr(last_msg, "tool_calls")
                                                and last_msg.tool_calls
                                            ):
                                                node_info["tool_calls"] = [
                                                    {
                                                        "name": normalized_tc.get("name"),
                                                        "id": normalized_tc.get("id"),
                                                        "args": make_json_safe(
                                                            normalized_tc.get("args", {})
                                                        ),
                                                    }
                                                    for normalized_tc in (
                                                        normalize_tool_call(tc)
                                                        for tc in last_msg.tool_calls
                                                    )
                                                ]

                                    if node_name == "planning_tools":
                                        todos = node_state.get("todos", [])
                                        if todos:
                                            node_info["todos_count"] = len(todos)
                                            node_info["current_task_index"] = node_state.get(
                                                "current_task_index"
                                            )

                                    yield {"type": "node_complete", **node_info}

                                if "messages" in node_state:
                                    messages = node_state["messages"]
                                    if messages:
                                        last_msg = (
                                            messages[-1] if isinstance(messages, list) else messages
                                        )

                                        if isinstance(last_msg, AIMessage):
                                            if (
                                                hasattr(last_msg, "tool_calls")
                                                and last_msg.tool_calls
                                            ):
                                                for tool_call in last_msg.tool_calls:
                                                    normalized_tool_call = normalize_tool_call(
                                                        tool_call
                                                    )
                                                    tool_call_id = normalized_tool_call.get("id")
                                                    if (
                                                        tool_call_id
                                                        and tool_call_id
                                                        not in emitted_tool_call_ids
                                                    ):
                                                        emitted_tool_call_ids.add(tool_call_id)
                                                        yield {
                                                            "type": "tool_start",
                                                            "name": normalized_tool_call.get(
                                                                "name", "unknown"
                                                            ),
                                                            "tool_call_id": tool_call_id,
                                                            "args": make_json_safe(
                                                                normalized_tool_call.get("args", {})
                                                            ),
                                                        }

                                        elif isinstance(last_msg, ToolMessage):
                                            for (
                                                event_payload
                                            ) in self._tool_end_events_from_node_state(
                                                node_state=node_state,
                                                last_state_values=last_state_values,
                                                emitted_tool_result_ids=emitted_tool_result_ids,
                                            ):
                                                yield event_payload
                    else:
                        logger.debug(f"Unexpected stream chunk format: {type(chunk)}")

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

            if not should_continue and last_state_values:
                continue_reason = self._get_requested_continuation_reason(last_state_values)
                if continue_reason:
                    should_continue = True

            if not should_continue:
                break

            captured = await self._capture_state_for_continuation(
                config=config,
                thread_id=thread_id,
                fallback_state=last_state_values,
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
            current_tool_calls = {}

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
                        interrupt_payload = self._get_interrupt_payload_from_state(
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
        last_emitted_agent = selected_agent

        # Agentic RAG is the only path. Document-aware chat streams through the
        # LangGraph pipeline so the search_documents tool loop can execute.
        accumulated_content = ""
        accumulated_thinking = ""  # Track thinking content for non-RAG agents
        current_tool_calls = {}  # Track tool call chunks by index
        emitted_tool_call_ids = set()  # Track which tool calls have had tool_start emitted
        emitted_tool_result_ids = set()

        # For image_generator_agent the streamed LLM tokens are the internal
        # enhanced prompt — not meant for the user.  Suppress token events and
        # let the final "complete" response (which holds the user-facing text)
        # be the only content the client sees.
        # Extend this set with any future agent whose raw tokens are internal.
        suppressed_nodes: set = {"image_generator_agent"}
        suppress_tokens = selected_agent in suppressed_nodes

        # Track whether *every* content chunk received so far has been internal.
        # When True the accumulated_content / accumulated_thinking fallbacks remain
        # empty (no internal content should leak into them) and we guard the
        # post-stream fallback branch accordingly.
        _internal_content_only: bool = True

        # ── Auto-Continue outer loop ──────────────────────────────────
        max_rounds = settings.auto_continue_max_rounds if settings.auto_continue_enabled else 1
        round_num = 1
        continue_reason: str | None = "initial"
        total_iterations = 0
        start_time = time.monotonic()
        current_state = initial_state
        last_state_values: dict[str, Any] | None = None

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
                # - "messages": Stream LLM tokens with metadata (includes tool_call_chunks)
                # - "updates": Stream state updates after each node (includes completed messages)
                async for chunk in self.graph.astream(
                    current_state, config=config, stream_mode=["messages", "updates"]
                ):
                    # Handle tuple format from multiple stream modes
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        mode, data = chunk

                        if mode == "messages":
                            # LLM token streaming - data is (message_chunk, metadata)
                            message_chunk, metadata = data

                            # Skip ToolMessage - tool results are handled in updates mode
                            if isinstance(message_chunk, ToolMessage):
                                continue

                            # Drop output from internal LLM runs (e.g. summarization node)
                            # so that internal summaries, reasoning, and tool-call chunks
                            # from those nodes never reach the client or pollute accumulators.
                            if (
                                settings.suppress_internal_stream_chunks
                                and self._is_internal_stream_chunk(metadata)
                            ):
                                continue

                            # Mark that at least one non-internal chunk has arrived.
                            _internal_content_only = False

                            # Handle text content using content_blocks (latest pattern)
                            if (
                                hasattr(message_chunk, "content_blocks")
                                and message_chunk.content_blocks
                            ):
                                for block in message_chunk.content_blocks:
                                    block_type = block.get("type")

                                    if block_type == "text":
                                        text_content = block.get("text", "")
                                        accumulated_content, delta = (
                                            self._consume_stream_text_chunk(
                                                accumulated_content, text_content
                                            )
                                        )
                                        if delta and not suppress_tokens:
                                            yield {"type": "token", "content": delta}

                                    # Handle thinking block type
                                    elif block_type == "thinking":
                                        thinking_content = block.get("thinking", "") or block.get(
                                            "text", ""
                                        )
                                        if thinking_content:
                                            accumulated_thinking += thinking_content
                                            yield {
                                                "type": "thinking",
                                                "content": thinking_content,
                                            }

                                    # Handle reasoning block type (LangChain Google GenAI)
                                    elif block_type == "reasoning":
                                        reasoning_content = block.get("reasoning", "") or block.get(
                                            "text", ""
                                        )
                                        if reasoning_content:
                                            accumulated_thinking += reasoning_content
                                            yield {
                                                "type": "thinking",
                                                "content": reasoning_content,
                                            }

                                    elif block_type == "tool_call_chunk":
                                        # Stream tool call chunks as they arrive
                                        tool_index = block.get("index", 0)
                                        tool_id = block.get("id")
                                        tool_name = block.get("name")
                                        tool_args = block.get("args", "")

                                        # Initialize or update tool call tracking
                                        if tool_index not in current_tool_calls:
                                            current_tool_calls[tool_index] = {
                                                "id": tool_id,
                                                "name": tool_name,
                                                "args": "",
                                            }

                                        # Accumulate args
                                        if tool_args:
                                            current_tool_calls[tool_index]["args"] += tool_args

                                        # Update name/id if present
                                        if tool_name and not current_tool_calls[tool_index]["name"]:
                                            current_tool_calls[tool_index]["name"] = tool_name
                                        if tool_id and not current_tool_calls[tool_index]["id"]:
                                            current_tool_calls[tool_index]["id"] = tool_id

                                pass  # Content blocks handled

                            # Handle content as list (when include_thoughts=True)
                            # LangChain returns content as list with thinking/reasoning and text parts
                            elif hasattr(message_chunk, "content") and isinstance(
                                message_chunk.content, list
                            ):
                                for part in message_chunk.content:
                                    if isinstance(part, dict):
                                        part_type = part.get("type", "")

                                        if part_type == "thinking":
                                            thinking_content = part.get("thinking", "") or part.get(
                                                "text", ""
                                            )
                                            if thinking_content:
                                                accumulated_thinking += thinking_content
                                                yield {
                                                    "type": "thinking",
                                                    "content": thinking_content,
                                                }
                                        elif part_type == "reasoning":
                                            reasoning_content = part.get(
                                                "reasoning", ""
                                            ) or part.get("text", "")
                                            if reasoning_content:
                                                accumulated_thinking += reasoning_content
                                                yield {
                                                    "type": "thinking",
                                                    "content": reasoning_content,
                                                }
                                        elif part_type == "text":
                                            text_content = part.get("text", "")
                                            accumulated_content, delta = (
                                                self._consume_stream_text_chunk(
                                                    accumulated_content, text_content
                                                )
                                            )
                                            if delta and not suppress_tokens:
                                                yield {
                                                    "type": "token",
                                                    "content": delta,
                                                }
                                    elif isinstance(part, str) and part:
                                        accumulated_content, delta = (
                                            self._consume_stream_text_chunk(
                                                accumulated_content, part
                                            )
                                        )
                                        if delta and not suppress_tokens:
                                            yield {"type": "token", "content": delta}

                            # Fallback: Handle legacy string content attribute
                            elif (
                                hasattr(message_chunk, "content")
                                and message_chunk.content
                                and isinstance(message_chunk.content, str)
                            ):
                                content = coerce_response_text(message_chunk.content)
                                accumulated_content, delta = self._consume_stream_text_chunk(
                                    accumulated_content, content
                                )
                                if delta and not suppress_tokens:
                                    yield {"type": "token", "content": delta}

                            # Check for chunk completion and emit complete tool calls
                            if (
                                hasattr(message_chunk, "chunk_position")
                                and message_chunk.chunk_position == "last"
                            ):
                                # Emit accumulated tool calls
                                for tool_call in current_tool_calls.values():
                                    if tool_call["name"]:  # Only emit if we have a name
                                        tool_call_id = tool_call["id"]
                                        # Skip if already emitted
                                        if tool_call_id and tool_call_id in emitted_tool_call_ids:
                                            continue
                                        if tool_call_id:
                                            emitted_tool_call_ids.add(tool_call_id)

                                        try:
                                            # Parse args if it's a JSON string
                                            args = (
                                                json.loads(tool_call["args"])
                                                if tool_call["args"]
                                                else {}
                                            )
                                        except Exception:
                                            args = tool_call["args"]

                                        yield {
                                            "type": "tool_start",
                                            "name": tool_call["name"],
                                            "tool_call_id": tool_call_id,
                                            "args": make_json_safe(args),
                                        }
                                # Clear for next message
                                current_tool_calls = {}

                        elif mode == "updates":
                            # State updates - check for completed messages with thinking/tools
                            for node_name, node_state in data.items():
                                # Track latest state values for continuation fallback
                                if isinstance(node_state, dict):
                                    if last_state_values is None:
                                        last_state_values = {}
                                    last_state_values.update(node_state)

                                    # Emit a second ``agent_selected`` event when
                                    # this node update changed the active agent
                                    # (e.g. ``hand_off`` rerouted from planning_agent
                                    # to search_agent). This lets the UI swap its
                                    # status indicator to the delegated agent before
                                    # the next set of tokens arrives.
                                    new_agent = node_state.get("selected_agent")
                                    if (
                                        isinstance(new_agent, str)
                                        and new_agent != last_emitted_agent
                                    ):
                                        last_emitted_agent = new_agent
                                        yield {
                                            "type": "agent_selected",
                                            "agent": new_agent,
                                            "reason": "handoff",
                                        }

                                # Emit node completion event for planning nodes
                                if node_name in ("planning_agent", "planning_tools"):
                                    # Extract relevant info from the node state
                                    node_info = {"node": node_name}

                                    # For planning_agent, include tool call info
                                    if node_name == "planning_agent" and "messages" in node_state:
                                        messages = node_state.get("messages", [])
                                        if messages:
                                            last_msg = (
                                                messages[-1]
                                                if isinstance(messages, list)
                                                else messages
                                            )
                                            if (
                                                isinstance(last_msg, AIMessage)
                                                and hasattr(last_msg, "tool_calls")
                                                and last_msg.tool_calls
                                            ):
                                                node_info["tool_calls"] = [
                                                    {
                                                        "name": normalized_tc.get("name"),
                                                        "id": normalized_tc.get("id"),
                                                        "args": make_json_safe(
                                                            normalized_tc.get("args", {})
                                                        ),
                                                    }
                                                    for normalized_tc in (
                                                        normalize_tool_call(tc)
                                                        for tc in last_msg.tool_calls
                                                    )
                                                ]

                                    # For planning_tools, include execution results
                                    if node_name == "planning_tools":
                                        todos = node_state.get("todos", [])
                                        if todos:
                                            node_info["todos_count"] = len(todos)
                                            node_info["current_task_index"] = node_state.get(
                                                "current_task_index"
                                            )

                                    yield {"type": "node_complete", **node_info}

                                if "messages" in node_state:
                                    messages = node_state["messages"]
                                    if messages:
                                        last_msg = (
                                            messages[-1] if isinstance(messages, list) else messages
                                        )

                                        # Handle AIMessage - extract tool_calls only
                                        if isinstance(last_msg, AIMessage):
                                            # Handle tool calls
                                            if (
                                                hasattr(last_msg, "tool_calls")
                                                and last_msg.tool_calls
                                            ):
                                                for tool_call in last_msg.tool_calls:
                                                    normalized_tool_call = normalize_tool_call(
                                                        tool_call
                                                    )
                                                    tool_call_id = normalized_tool_call.get("id")
                                                    # Only emit if not already emitted from messages mode
                                                    if (
                                                        tool_call_id
                                                        and tool_call_id
                                                        not in emitted_tool_call_ids
                                                    ):
                                                        emitted_tool_call_ids.add(tool_call_id)
                                                        yield {
                                                            "type": "tool_start",
                                                            "name": normalized_tool_call.get(
                                                                "name", "unknown"
                                                            ),
                                                            "tool_call_id": tool_call_id,
                                                            "args": make_json_safe(
                                                                normalized_tool_call.get("args", {})
                                                            ),
                                                        }

                                        # Handle ToolMessage (result)
                                        elif isinstance(last_msg, ToolMessage):
                                            for (
                                                event_payload
                                            ) in self._tool_end_events_from_node_state(
                                                node_state=node_state,
                                                last_state_values=last_state_values,
                                                emitted_tool_result_ids=emitted_tool_result_ids,
                                            ):
                                                yield event_payload
                    else:
                        # Single mode or legacy format - try to handle gracefully
                        logger.debug(f"Unexpected stream chunk format: {type(chunk)}")

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
            if not should_continue and last_state_values:
                continue_reason = self._get_requested_continuation_reason(last_state_values)
                if continue_reason:
                    should_continue = True

            if not should_continue:
                break  # Normal completion — exit loop and finalize

            # ── Safety: total iteration cap ────────────────────────────
            captured = await self._capture_state_for_continuation(
                config=config,
                thread_id=thread_id,
                fallback_state=last_state_values,
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
            current_tool_calls = {}

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
                            interrupt_payload = self._get_interrupt_payload_from_state(
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
