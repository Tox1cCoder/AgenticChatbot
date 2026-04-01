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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..core.response_constants import NO_RESPONSE_GENERATED
from ..interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..interfaces.workflow_runtime_interface import IWorkflowRuntime
from ..models.enums import PlanLifecycle
from .agents.canvas_agent import CanvasAgent
from .agents.chat_agent import ChatAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .agents.planning_agent import PlanningAgent
from .agents.rag_agent import RAGAgent
from .agents.router import Router
from .agents.search_agent import SearchAgent
from .hand_off_tool import MAX_DELEGATION_DEPTH
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
from .summarization_middleware import summarize_for_state
from .todo_actions import apply_write_todos_action
from .token_instrumentation import (
    HistoryBudgetConfig,
    trim_history_to_budget,
    truncate_tool_result,
)
from .tool_context import tool_execution_context
from .tool_execution import (
    build_rejected_tool_artifacts,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from .utils import (
    apply_hitl_decisions,
    coerce_response_text,
    find_pending_tool_call_message,
    make_json_safe,
    normalize_tool_call,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository

_apply_decisions = apply_hitl_decisions


class MultiAgentWorkflow(IWorkflowRuntime):
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        checkpointer: BaseCheckpointSaver | None = None,
        document_repository: Optional["DocumentRepository"] = None,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
    ):
        self.qdrant_client = qdrant_client
        self.router = Router()
        self.chat_agent = ChatAgent(runtime_model_resolver=runtime_model_resolver)
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
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

    async def _get_conversation_history(
        self,
        conversation_id: str | None,
        user_id: str | None,
        agent_key: str | None = None,
    ) -> list:
        """
        Get conversation history with caching and budget trimming.

        Uses a bounded TTLCache (auto-evicts after ``_history_cache_ttl_seconds``)
        and a per-conversation ``asyncio.Lock`` to prevent duplicate DB lookups
        when concurrent requests hit the same conversation.

        History is trimmed according to agent-specific settings:
        - {agent_key}_history_max_messages
        - {agent_key}_history_max_tokens

        Args:
            conversation_id: The conversation UUID
            user_id: The user UUID
            agent_key: Agent type key (chat, rag, search, planning) for budget lookup
        """
        if not conversation_id or not user_id:
            return []

        cache_key = conversation_id

        async with self._history_locks[cache_key]:
            # TTLCache handles expiry automatically — a simple ``in`` check suffices.
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

                # Cache the full (untrimmed) history; TTLCache auto-evicts.
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

    def _find_last_human_message_index(self, messages: list) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                return idx
        return None

    def _has_tool_context(self, messages: list, last_human_idx: int | None) -> bool:
        if last_human_idx is None:
            return False

        return any(
            isinstance(m, (AIMessage, ToolMessage))
            and (isinstance(m, ToolMessage) or (hasattr(m, "tool_calls") and m.tool_calls))
            for m in messages[last_human_idx + 1 :]
        )

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

    def _finalize_agent_response(self, state: GraphState, response: AgentResponse) -> GraphState:
        state["response"] = response

        ai_kwargs = {"content": response.message.content}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    def _build_initial_state_from_request(self, request: WorkflowExecutionRequest) -> GraphState:
        tasks = list(request.planning.tasks)
        initial_state: GraphState = {
            "messages": [HumanMessage(content=request.message)],
            "context": {},
        }

        if request.conversation_id is not None:
            initial_state["conversation_id"] = request.conversation_id
        if request.user_id is not None:
            initial_state["user_id"] = request.user_id
        if request.device_id is not None:
            initial_state["device_id"] = request.device_id
        if request.model_request is not None:
            initial_state["model_request"] = request.model_request
        initial_state["selected_agent"] = None
        initial_state["response"] = None
        initial_state["persona"] = request.persona
        initial_state["planning_mode_enabled"] = request.planning.planning_mode_enabled
        initial_state["has_existing_plan"] = request.planning.has_existing_plan
        initial_state["iteration_count"] = None

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

        # Persist lifecycle so agents can read execution state without prompt inference
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

        # Add summarization node first - runs ONCE at start of each request
        workflow.add_node("summarize", self._summarization_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)
        workflow.add_node("planning_agent", self._planning_node)
        workflow.add_node("canvas_agent", self._canvas_node)
        workflow.add_node("planning_tools", self._planning_tools_node)
        workflow.add_node("rag_tools", self._rag_tools_node)
        workflow.add_node("approval", self._approval_node)
        workflow.add_node("tools", self._tool_node)

        # Route: START -> summarize -> route -> agents
        # Summarization runs ONCE before routing, not on every agent iteration
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "route")

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
                "end": END,
            },
        )

        # Consolidate conditional edges for agents that use standard tool calling
        tool_calling_agents = [
            "chat_agent",
            "search_agent",
            "image_generator_agent",
            "canvas_agent",
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

        workflow.add_conditional_edges(
            "planning_tools",
            self._should_continue_planning,
            {
                "planning_agent": "planning_agent",
                "end": END,
            },
        )

        workflow.add_edge("approval", "tools")

        # Dynamic tool routing map based on agent registry
        # Include ALL agents (including rag_agent) so hand_off delegation works.
        tool_routing_map = {agent_name: agent_name for agent_name in self.agents}
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
        agent = self.agents.get(selected_agent_name)
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

        # Validate target agent exists
        if target_agent not in self.agents:
            logger.warning("hand_off requested unknown agent '%s'; ignoring", target_agent)
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

        logger.info(
            "Delegating from '%s' → '%s' (reason: %s)",
            state.get("selected_agent"),
            target_agent,
            reason,
        )
        state["selected_agent"] = target_agent
        state["delegation_count"] = delegation_count + 1
        return state

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
                for field_name in ("tool_origin", "server_name", "qualified_tool_id"):
                    if tool_metadata.get(field_name):
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

        # Always keep the original AIMessage with ALL tool_calls intact.
        # Rejection ToolMessages must reference tool_call_ids present in the preceding
        # AIMessage — stripping tool_calls would orphan them and cause the LLM to
        # ignore the rejections (leading to hallucinated answers).
        # The tools node filters out tool_calls that already have ToolMessages.
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
        max_iterations = settings.react_agent_max_iterations

        # Soft-limit: if auto-continue is enabled, trigger continuation at
        # a fraction of the budget so the outer loop can start a new round
        # before the hard LangGraph recursion limit is hit.
        if settings.auto_continue_enabled:
            soft_limit = int(max_iterations * settings.auto_continue_soft_limit_ratio)
            if iteration_count >= soft_limit:
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

        selected_agent = state_view.selected_agent() or "end"

        if selected_agent != "end" and selected_agent not in self.agents:
            logger.warning(
                f"Selected agent '{selected_agent}' not found in registry, ending conversation"
            )
            return "end"

        return selected_agent

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

        return AgentResponse(
            agent_type=agent_type,
            agent_id=selected_agent or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            metadata={"interrupt": interrupt_response},
        )

    async def _summarization_node(self, state: GraphState) -> GraphState:
        """
        Summarization node that runs ONCE at the start of each user request.
        """
        try:
            conversation_id: str | None = state.get("conversation_id")
            # This will check thresholds and apply summarization if needed.
            # summarize_for_state is fail-closed: on any error it returns state unchanged.
            state = await summarize_for_state(state, conversation_id=conversation_id)
        except Exception as e:
            # Defensive catch — summarize_for_state is already fail-closed but keep
            # the node from crashing the graph on any unexpected exception.
            logger.warning("Summarization node error (continuing anyway): %s", e)

        return state

    async def _route_node(self, state: GraphState) -> GraphState:
        # Reset delegation counter at the start of each new user turn
        state["delegation_count"] = 0

        if state.get("selected_agent"):
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

        selected_agent = await self.router.route_message(
            agent_msg,
            available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
        )

        if selected_agent == "rag_agent" and not has_documents:
            selected_agent = "chat_agent"

        state["selected_agent"] = selected_agent
        return state

    def _conversation_has_documents(self, conversation_id: str | None) -> bool:
        if not conversation_id or not self.document_repository:
            return False
        try:
            return self.document_repository.count_by_conversation(UUID(conversation_id)) > 0
        except (ValueError, Exception):
            return False

    def _build_graph_config(self, thread_id: str | None = None) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        recursion_limit = getattr(settings, "react_agent_recursion_limit", None)
        if recursion_limit and recursion_limit > 0:
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

        if tool_map is None:
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
        if not tool_map:
            return [], [], []

        agent_key = (
            getattr(agent, "agent_config_key", None)
            or getattr(agent, "agent_id", None)
            or "unknown"
        )

        with tool_execution_context(
            conversation_id,
            user_id,
            agent_key,
            device_id,
        ):
            return await execute_tool_calls(
                tool_calls=tool_calls,
                tool_map=tool_map,
                capture_images=capture_images,
                device_id=device_id,
                agent=agent,
                conversation_id=conversation_id,
                user_id=user_id,
            )

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

            state.setdefault("messages", []).append(
                ToolMessage(
                    content=content,
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        state["iteration_count"] = (state.get("iteration_count") or 0) + 1

        context = GraphStateView(state).context_copy()
        if tool_artifacts:
            existing_artifacts = list(context.get("tool_artifacts", []))
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        if all_images:
            existing_images = list(context.get("tool_images", []))
            existing_images.extend(all_images)
            context["tool_images"] = existing_images
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
            conversation_id, user_id, agent_key="chat"
        )

        attachments = self._get_state_attachments(state)
        device_id = state.get("device_id")
        current_turn_messages = self._get_current_turn_messages(messages)
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
        )

        if has_images:
            if response.metadata is None:
                response.metadata = {}
            response.metadata["has_images"] = True

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
            conversation_id, user_id, agent_key="rag"
        )
        device_id = state.get("device_id")

        last_human_idx = self._find_last_human_message_index(messages)
        original_query = messages[last_human_idx].content if last_human_idx is not None else content

        tool_context = []
        if last_human_idx is not None:
            for msg in messages[last_human_idx + 1 :]:
                if isinstance(msg, ToolMessage):
                    tool_context.append(msg.content)

        metadata = {
            "persona": state.get("persona"),
            "history": conversation_history,
            "original_query": original_query,
            "tool_context": tool_context,
            "agentic_images": state.get("context", {}).get(
                "agentic_images", []
            ),  # Pass images for multimodal LLM
            "model_request": state.get("model_request"),
            "user_id": user_id,
            "device_id": device_id,
            "history_summary": state.get("history_summary"),
        }

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=original_query,
            metadata=metadata,
            attachments=self._get_state_attachments(state),
        )

        response = await self.rag_agent.process_message(agent_msg, conversation_id)
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
        non_search_outputs_by_id: dict[str, str] = {}
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
                    non_search_outputs_by_id[tc_id] = feedback

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

                # Extract context for tool execution
                user_id = state.get("user_id")
                agent_key = getattr(agent, "agent_config_key", None) if agent else "rag"
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
                        non_search_outputs_by_id[output["tool_call_id"]] = output["content"]
                tool_artifacts.extend(artifacts)
                all_images.extend(images)

        for tool_call_data in normalized_tool_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            if tool_name != "search_documents":
                tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": non_search_outputs_by_id.get(
                            tool_id, f"Error: Tool {tool_name} not found"
                        ),
                    }
                )
                continue

            result, _ = await execute_search_documents_action(
                rag_agent=self.rag_agent,
                conversation_id=conversation_id,
                tool_args=tool_args,
                context=context,
                max_agentic_images=max_agentic_images,
            )

            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result,
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
        if all_images:
            existing_images = context.get("tool_images", [])
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        return state

    def _should_call_rag_tools(self, state: GraphState) -> str:
        if not settings.agentic_rag_enabled:
            return "end"

        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "rag_tools"

        return "end"

    def _should_continue_rag(self, state: GraphState) -> str:
        """
        Determine if RAG agentic loop should continue or end.
        """
        context = state.get("context", {})
        agentic_iteration = context.get("agentic_rag_iteration", 0)

        # Check iteration limit
        max_iterations = settings.agentic_max_iterations
        if agentic_iteration >= max_iterations:
            logger.warning(
                f"RAG agentic loop reached max iterations ({max_iterations}), forcing end"
            )
            return "end"

        # Continue to RAG agent for more processing
        return "rag_agent"

    async def _search_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="search"
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.search_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
        )

        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        # NOTE: Image generator deliberately borrows the "chat" history budget.
        # If independent tuning is needed, add image_generator_history_max_* settings.
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="chat"
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.image_generator_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
        )

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
            conversation_id, user_id, agent_key="chat"
        )

        current_turn_messages = self._get_current_turn_messages(messages)

        response = await self.canvas_agent.invoke_model_with_history(
            current_turn_messages,
            conversation_history,
            state.get("persona"),
            conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
            model_request=state.get("model_request"),
            history_summary=state.get("history_summary"),
        )

        self._merge_tool_artifacts(state, response)
        return self._finalize_agent_response(state, response)

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
            conversation_id, user_id, agent_key="planning"
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

        # Call the planning agent with history
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

        tool_map = await ensure_agent_tool_map(
            self.planning_agent,
            conversation_id=conversation_id,
            user_id=user_id,
            device_id=state.get("device_id"),
        )

        # Separate write_todos calls from external/MCP tool calls
        normalized_calls = [normalize_tool_call(tc) for tc in last_message.tool_calls]
        external_tool_calls = [tc for tc in normalized_calls if tc.get("name") != "write_todos"]
        write_todos_calls = [tc for tc in normalized_calls if tc.get("name") == "write_todos"]

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
            external_outputs, external_artifacts, external_images = (
                await self._execute_agent_tool_calls(
                    state=state,
                    agent=self.planning_agent,
                    tool_calls=approved_external_calls,
                    tool_map=tool_map,
                    capture_images=True,
                )
            )
            tool_outputs.extend(external_outputs)
            tool_artifacts.extend(external_artifacts)
            all_images.extend(external_images)
            had_error = had_error or any(
                artifact.get("status") == "error" for artifact in external_artifacts
            )

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

        # Update consecutive_errors counter for circuit breaker
        context = GraphStateView(state).context_copy()
        if had_error:
            context["consecutive_errors"] = context.get("consecutive_errors", 0) + 1
            logger.warning(f"Planning consecutive errors: {context['consecutive_errors']}")
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
        planning_call_count = state.get("planning_call_count", 0)
        max_iterations = settings.planning_max_iterations

        # Soft-limit: if auto-continue is enabled, trigger continuation at
        # a fraction of the planning budget.
        if settings.auto_continue_enabled:
            soft_limit = int(max_iterations * settings.auto_continue_soft_limit_ratio)
            if planning_call_count >= soft_limit:
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

        # Check iteration budget (hard limit)
        if planning_call_count >= max_iterations:
            context = GraphStateView(state).context_copy()
            context["pause_reason"] = "max_iterations_reached"
            state["context"] = context
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

        # Circuit breaker: check consecutive errors
        context = GraphStateView(state).context_copy()
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

        context = GraphStateView(state).context_copy()
        planning_phase = state.get("planning_phase", "planning")

        # If plan was just created/modified, return to agent for confirmation response
        if context.get("plan_just_modified"):
            context["plan_just_modified"] = False
            context["generate_plan_response"] = True
            state["context"] = context
            logger.debug(
                "[Should Continue Planning] Decision: planning_agent (plan_just_modified=True)"
            )
            return "planning_agent"

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

        images = self._merge_unique_items(
            response.metadata.get("images"), state_view.tool_images()
        )
        if images:
            response.metadata["images"] = images

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
            if (
                fallback_text
                and response.message
                and not coerce_response_text(getattr(response.message, "content", None))
            ):
                response.message.content = fallback_text
            return self._attach_context_outputs(state, response)

        content = fallback_text
        if not content:
            messages = state.get("messages", [])
            for message in reversed(messages):
                if not isinstance(message, AIMessage):
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

    async def resume_execution(self, thread_id: str, resume_value: Any) -> AgentResponse | None:
        if not self.checkpointer:
            raise RuntimeError("Checkpointing must be enabled for resume_execution")

        config = self._build_graph_config(thread_id)
        result = await self.graph.ainvoke(Command(resume=resume_value), config=config)
        response = self._recover_terminal_response(result)
        if response:
            return response

        final_snapshot = await self.graph.aget_state(config)
        final_state = (
            final_snapshot.values if final_snapshot and hasattr(final_snapshot, "values") else None
        )
        return self._recover_terminal_response(final_state)

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

        resume_data = [
            {
                "task_id": d.task_id,
                "tool_call_id": d.task_id,
                "type": d.type.value if hasattr(d.type, "value") else d.type,
                "args": d.args,
            }
            for d in decisions
        ]

        selected_agent = state_snapshot.values.get("selected_agent", "search_agent")
        conversation_id = state_snapshot.values.get("conversation_id")

        yield {"type": "agent_selected", "agent": selected_agent}

        accumulated_content = ""
        accumulated_thinking = ""
        current_tool_calls = {}
        emitted_tool_call_ids = set()

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
                                            yield {
                                                "type": "tool_end",
                                                "name": getattr(last_msg, "name", "unknown"),
                                                "tool_call_id": getattr(
                                                    last_msg, "tool_call_id", None
                                                ),
                                                "result": make_json_safe(last_msg.content),
                                            }
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

    # ------------------------------------------------------------------
    # Fast-path helpers (traditional RAG streaming)
    # ------------------------------------------------------------------

    async def _run_fast_path_summarization(
        self, config: dict[str, Any], thread_id: str | None
    ) -> str | None:
        """Load checkpoint state and run summarization for fast-path RAG.

        The LangGraph pipeline (summarize node) is bypassed on the
        traditional-RAG streaming path.  This helper replicates the
        rolling summarization logic so fast-path and graph-path behave
        identically.

        Fail-closed: on any error or timeout the existing summary is returned
        unchanged and no checkpoint updates are performed.

        Returns the (possibly updated) history_summary, or ``None``.
        """
        if not self.checkpointer or not thread_id:
            return None

        try:
            cp_snapshot = await self.graph.aget_state(config)
            cp_values = cp_snapshot.values if cp_snapshot else {}
            history_summary = cp_values.get("history_summary")

            cp_messages = cp_values.get("messages", [])
            if not cp_messages:
                return history_summary

            from .summarization_middleware import (
                _get_config as _get_summ_config,
            )
            from .summarization_middleware import (
                apply_summarization_to_state,
                generate_summary,
                get_messages_to_summarize,
                should_summarize,
            )

            # Ignore persisted conversation_summarized flag — each
            # fast-path request is a fresh opportunity for rolling
            # summarization.
            if not should_summarize(cp_messages, already_summarized=False):
                return history_summary

            s_cfg = _get_summ_config()
            to_summarize = get_messages_to_summarize(cp_messages, s_cfg)
            if not to_summarize:
                return history_summary

            timeout_seconds: int = settings.summarization_timeout_seconds
            try:
                new_summary = await asyncio.wait_for(
                    generate_summary(to_summarize, s_cfg, existing_summary=history_summary),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Fast-path summarization timed out after %ds — keeping existing summary",
                    timeout_seconds,
                )
                return history_summary
            except Exception as gen_err:
                logger.warning(
                    "Fast-path generate_summary failed — keeping existing summary: %s",
                    gen_err,
                )
                return history_summary

            # Only persist checkpoint updates on genuine success.
            _tmp_state: dict[str, Any] = {
                "messages": cp_messages,
                "context": cp_values.get("context", {}),
            }
            apply_summarization_to_state(
                _tmp_state,
                new_summary,
                to_summarize,
                s_cfg,
                conversation_id=str(cp_values.get("conversation_id") or ""),
            )
            await self.graph.aupdate_state(
                config,
                {
                    "history_summary": _tmp_state["history_summary"],
                    "history_summary_updated_at": _tmp_state["history_summary_updated_at"],
                    "summary_cursor_message_id": _tmp_state.get("summary_cursor_message_id"),
                    "messages": _tmp_state["messages"],
                    "context": _tmp_state["context"],
                },
            )
            return new_summary
        except Exception as e:
            logger.warning(
                "Fast-path summarization/checkpoint read failed (continuing without summary): %s",
                e,
            )
            return None

    async def _persist_fast_path_turn(
        self,
        config: dict[str, Any],
        thread_id: str | None,
        user_message: str,
        response: AgentResponse | None,
    ) -> None:
        """Persist user + assistant messages to checkpoint after fast-path RAG.

        The fast path bypasses graph execution, so messages are never
        written to checkpoint state.  This helper appends them so that
        subsequent summarization runs see the full conversation.
        """
        if not self.checkpointer or not thread_id:
            return

        try:
            reply_content = response.message.content if response and response.message else ""
            await self.graph.aupdate_state(
                config,
                {
                    "messages": [
                        HumanMessage(content=user_message),
                        AIMessage(content=reply_content),
                    ],
                },
            )
        except Exception as cp_err:
            logger.warning(
                "Fast-path: failed to persist turn to checkpoint: %s",
                cp_err,
            )

    async def execute_request_stream(self, request: WorkflowExecutionRequest):
        initial_state = self._build_initial_state_from_request(request)
        conversation_id = request.conversation_id
        thread_id = self._resolve_thread_id(request.thread_id, conversation_id)
        config = self._build_graph_config(thread_id)
        user_id = request.user_id
        message = request.message
        persona = request.persona
        attachments = request.attachments

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

        # For traditional RAG we use the agent's custom streaming implementation.
        # For agentic RAG we must stream the LangGraph execution so tool calls can run.
        rag_provider = None
        model_request = initial_state.get("model_request")
        if isinstance(model_request, dict):
            rag_cfg = model_request.get("rag")
            if not isinstance(rag_cfg, dict):
                rag_cfg = model_request.get("all")
            if isinstance(rag_cfg, dict):
                rag_provider = str(rag_cfg.get("provider") or "").strip().lower() or None

        if (
            selected_agent == "rag_agent"
            and not settings.agentic_rag_enabled
            and rag_provider != "openai"
        ):
            # Traditional RAG streaming — bypasses the graph pipeline.
            history_summary = await self._run_fast_path_summarization(config, thread_id)

            conversation_history = await self._get_conversation_history(
                conversation_id, user_id, agent_key="rag"
            )

            agent_msg = AgentMessage(
                role=MessageRole.USER,
                content=message,
                metadata={
                    "history": conversation_history,
                    "persona": persona,
                    "model_request": initial_state.get("model_request"),
                    "user_id": user_id,
                    "history_summary": history_summary,
                },
                attachments=attachments,
            )

            try:
                fast_path_response: AgentResponse | None = None
                async for event in self.rag_agent.stream_message(agent_msg, conversation_id):
                    event_type = event.get("type")
                    if event_type in ["thinking", "token", "tool_start", "tool_end"]:
                        yield event
                    elif event_type == "complete":
                        fast_path_response = event.get("response")
                        if fast_path_response:
                            yield {
                                "type": "complete",
                                "response": fast_path_response,
                            }
                        # Persist turn to checkpoint so subsequent
                        # summarization runs see the full conversation.
                        await self._persist_fast_path_turn(
                            config, thread_id, message, fast_path_response
                        )
                        return
                    elif event_type == "error":
                        yield event
                        return
                yield {"type": "error", "error": "RAG agent stream ended unexpectedly"}
            except Exception as e:
                yield {"type": "error", "error": str(e)}
            return

        accumulated_content = ""
        accumulated_thinking = ""  # Track thinking content for non-RAG agents
        current_tool_calls = {}  # Track tool call chunks by index
        emitted_tool_call_ids = set()  # Track which tool calls have had tool_start emitted

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
                                            yield {
                                                "type": "tool_end",
                                                "name": getattr(last_msg, "name", "unknown"),
                                                "tool_call_id": getattr(
                                                    last_msg, "tool_call_id", None
                                                ),
                                                "result": make_json_safe(last_msg.content),
                                            }
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
    embedding_model: SentenceTransformer,
    checkpointer: BaseCheckpointSaver | None = None,
    document_repository: Optional["DocumentRepository"] = None,
    runtime_model_resolver: IRuntimeModelResolver | None = None,
) -> MultiAgentWorkflow:
    """
    Create multi-agent workflow with required shared dependencies.
    """
    return MultiAgentWorkflow(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        checkpointer=checkpointer,
        document_repository=document_repository,
        runtime_model_resolver=runtime_model_resolver,
    )
